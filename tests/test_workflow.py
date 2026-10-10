import copy
import io
import json
import math
import shutil
import subprocess
from pathlib import Path
from urllib.error import HTTPError

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from training.dataset import arrays, load
from training.fetch_replays import fetch

ROOT = Path(__file__).resolve().parents[1]


def run(engine, *args, document=None):
    return subprocess.run(
        [str(engine), *map(str, args)],
        input=document,
        text=True,
        capture_output=True,
        check=True,
    )


def annotation(document, candidate, label=1):
    event = copy.deepcopy(candidate["evidence"])
    return {
        "replay_sha256": document["replay_sha256"],
        "kind": candidate["kind"],
        "frame": candidate["frame"],
        "initiator": event["initiator"],
        "victim": event["victim"],
        "player_id": candidate["player_id"],
        "raw_time": candidate["raw_time"],
        "evidence": event,
        "label": label,
        "engine_revision": document["revision"],
        "source_event_ids": ["external-consumer-event"],
    }


def dataset(document, candidates):
    candidates = [
        candidate
        for candidate in candidates
        if candidate["player_id"] == candidate["evidence"]["initiator"]
    ]
    reference = {
        "sha256": document["replay_sha256"],
        "download": {"source": "ballchasing", "id": "fixture-id"},
    }
    if document.get("rocket_league_id"):
        reference["rocket_league_id"] = document["rocket_league_id"]
    return {
        "schema_version": 1,
        "replays": [reference],
        "annotations": [
            annotation(document, candidate, index % 2)
            for index, candidate in enumerate(candidates)
        ],
    }


@pytest.fixture(scope="session")
def extracted(engine):
    facts = run(engine, "events", ROOT / "tests/fixtures/sample.replay").stdout
    document = json.loads(facts)
    batch = json.loads(run(engine, "candidates", document=facts).stdout)[0]
    return document, batch


@pytest.fixture
def prepared_inputs(tmp_path, extracted):
    document, batch = extracted
    labels = dataset(document, batch["candidates"])
    labels["annotations"][1]["reject_reason"] = "missing_context"
    annotations = tmp_path / "annotations.json"
    annotations.write_text(json.dumps(labels))
    recordings = tmp_path / "replays"
    recordings.mkdir()
    shutil.copyfile(
        ROOT / "tests/fixtures/sample.replay",
        recordings / f"{document['replay_sha256']}.replay",
    )
    return annotations, recordings, tmp_path / "examples.parquet", labels


def test_synthetic_contact_preserves_identity_and_features(engine):
    batches = json.loads(
        run(
            engine, "candidates", document=(ROOT / "examples/bump.json").read_text()
        ).stdout
    )
    candidate = batches[0]["candidates"][0]
    assert candidate["source_event_ids"] == ["bump:42"]
    assert candidate["player_id"] == {"Steam": 1}
    assert candidate["frame"] == 42 and candidate["raw_time"] == 10.0
    assert candidate["features"] == pytest.approx(
        [0.8, 900, 600, 90, 0.5, 17, 17, math.hypot(20, 5120)]
    )
    assert batches[0]["replay_sha256"] == "0" * 64


def test_real_replay_extracts_intrinsic_identity_and_contacts(extracted):
    document, batch = extracted
    assert (
        document["replay_sha256"]
        == "127d794a72ecbb02c3bd21cf4d0260cb25f6ba3f8d8b84a87839ee35a8985392"
    )
    assert document["rocket_league_id"] == "3A8451AE42FE51A53CD35E9016B123BD"
    assert len(batch["candidates"]) == 10
    contact = batch["candidates"][0]
    assert contact["frame"] == 1675
    assert contact["raw_time"] == pytest.approx(62.40333938598633)
    assert contact["features"] == pytest.approx(
        [
            0.9849797487,
            1361.345947,
            894.323914,
            1.752369,
            2956.311279,
            278.360168,
            286.119995,
            2892.466334,
        ]
    )


@pytest.fixture(scope="session")
def combined_export(engine):
    return json.loads(
        run(engine, "extract", ROOT / "tests/fixtures/sample.replay").stdout
    )


def test_combined_export_preserves_events_frames_and_boost(
    engine, combined_export, extracted
):
    document, batch = extracted
    assert set(combined_export["outputs"]) == {"events", "replay_data"}
    for field in (
        "name",
        "schema_version",
        "status",
        "revision",
        "replay_sha256",
        "rocket_league_id",
    ):
        assert combined_export[field] == document[field]
    timeline = combined_export["outputs"]["events"]
    assert timeline["events"]["events"] == document["events"]
    boost = [
        track
        for track in timeline["accumulation_tracks"]
        if track["quantity"] == "boost_amount"
    ]
    contact = batch["candidates"][0]["evidence"]
    tracked_players = [track["player_id"] for track in boost if track["points"]]
    assert contact["initiator"] in tracked_players
    assert contact["victim"] in tracked_players
    frames = combined_export["outputs"]["replay_data"]["frame_data"]
    assert frames["players"] and len(frames["metadata_frames"]) > 1
    assert frames["metadata_frames"][contact["frame"]]["time"] == pytest.approx(
        contact["time"]
    )
    for participant in (contact["initiator"], contact["victim"]):
        player = next(
            data for player_id, data in frames["players"] if player_id == participant
        )
        body = player["frames"][contact["frame"]]["Data"]["rigid_body"]
        assert body["linear_velocity"] is not None
    assert (
        json.loads(
            run(engine, "candidates", document=json.dumps(combined_export)).stdout
        )[0]
        == batch
    )


@pytest.mark.parametrize("selection", ["events", "replay_data", "replay_data,events"])
def test_output_selection_preserves_selected_payloads(
    engine, combined_export, selection
):
    document = json.loads(
        run(
            engine,
            "extract",
            ROOT / "tests/fixtures/sample.replay",
            "--outputs",
            selection,
        ).stdout
    )
    assert set(document["outputs"]) == set(selection.split(","))
    for name, payload in document["outputs"].items():
        assert payload == combined_export["outputs"][name]
    if selection == "replay_data":
        assert (
            json.loads(
                run(engine, "replay-data", ROOT / "tests/fixtures/sample.replay").stdout
            )
            == document["outputs"]["replay_data"]
        )


def test_output_catalog_declares_all_types_available(engine):
    catalog = json.loads(run(engine, "outputs").stdout)["outputs"]
    available = {item["name"] for item in catalog if item["implemented"]}
    assert available == {"events", "replay_data", "stats", "stats_timeline", "features"}


@pytest.mark.parametrize(
    ("selection", "message"),
    [
        ("unknown", "unknown output"),
        ("", "unknown output"),
        ("events,", "unknown output"),
        ("events,events", "duplicate output"),
    ],
)
def test_invalid_output_selection_fails_before_reading_replay(
    engine, selection, message
):
    result = subprocess.run(
        [str(engine), "extract", "missing.replay", "--outputs", selection],
        text=True,
        capture_output=True,
    )
    assert result.returncode == 1 and not result.stdout
    assert message in result.stderr


def test_combined_input_requires_events_output(engine, combined_export):
    document = copy.deepcopy(combined_export)
    del document["outputs"]["events"]
    result = subprocess.run(
        [str(engine), "candidates"],
        input=json.dumps(document),
        text=True,
        capture_output=True,
    )
    assert result.returncode == 1 and not result.stdout
    assert "must include a valid events output" in result.stderr


@pytest.mark.parametrize(
    "mutation", ["failed_supplier", "duplicate_incident", "inconsistent_teams"]
)
def test_invalid_candidate_inputs_fail(engine, mutation):
    document = json.loads((ROOT / "examples/bump.json").read_text())
    if mutation == "failed_supplier":
        document["status"] = "error"
    elif mutation == "duplicate_incident":
        duplicate = copy.deepcopy(document["events"][0])
        duplicate["meta"]["id"] = "different-event-id"
        document["events"].append(duplicate)
    else:
        document["events"][0]["payload"]["payload"]["victim_is_team_0"] = False
    with pytest.raises(subprocess.CalledProcessError):
        run(engine, "candidates", document=json.dumps(document))


def test_preparation_preserves_labels_and_uses_shared_features(
    engine, prepared_inputs, extracted
):
    annotations, recordings, output, labels = prepared_inputs
    assert (
        json.loads(run(engine, "validate-annotations", annotations).stdout)[
            "annotations"
        ]
        == 5
    )
    result = run(engine, "prepare", annotations, recordings, output, "bumping_teammate")
    assert json.loads(result.stdout)["examples"] == 5
    table, contract = load(output)
    x, y, groups = arrays(table, contract)
    assert x.shape == (5, 8) and x.dtype == np.float32
    assert x == pytest.approx(
        np.array(
            [
                c["features"]
                for c in extracted[1]["candidates"]
                if c["player_id"] == c["evidence"]["initiator"]
            ],
            dtype=np.float32,
        )
    )
    assert y.tolist() == [0, 1, 0, 1, 0]
    assert groups.tolist() == [extracted[0]["replay_sha256"]] * 5
    assert table["reject_reason"].to_pylist() == [
        None,
        "missing_context",
        None,
        None,
        None,
    ]
    assert (
        json.loads(table["initiator"][0].as_py())
        == labels["annotations"][0]["initiator"]
    )
    assert json.loads(table["source_event_ids"][0].as_py()) != [
        "external-consumer-event"
    ]
    assert contract["replays"] == labels["replays"]


@pytest.mark.parametrize(
    "mutation",
    [
        "label",
        "duplicate",
        "victim",
        "undeclared_replay",
        "unrelated_player",
        "missing_player",
    ],
)
def test_annotation_contract_rejects_invalid_imports(engine, prepared_inputs, mutation):
    path, _, _, labels = prepared_inputs
    if mutation == "label":
        labels["annotations"][0]["label"] = 2
    elif mutation == "duplicate":
        labels["annotations"].append(copy.deepcopy(labels["annotations"][0]))
    elif mutation == "victim":
        labels["annotations"][0]["victim"] = {"Steam": 100}
    elif mutation == "unrelated_player":
        labels["annotations"][0]["player_id"] = {"Steam": "unrelated"}
    elif mutation == "missing_player":
        del labels["annotations"][0]["player_id"]
    else:
        labels["annotations"][0]["replay_sha256"] = "b" * 64
    path.write_text(json.dumps(labels))
    with pytest.raises(subprocess.CalledProcessError):
        run(engine, "validate-annotations", path)


def test_native_preparation_rejects_victim_before_reading_replay(engine, tmp_path):
    document = json.loads((ROOT / "examples/bump.json").read_text())
    candidates = json.loads(
        run(engine, "candidates", document=json.dumps(document)).stdout
    )[0]
    review = annotation(document, candidates["candidates"][1])
    path = tmp_path / "annotations.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "replays": [{"sha256": document["replay_sha256"]}],
                "annotations": [review],
            }
        )
    )
    run(engine, "validate-annotations", path)
    output = tmp_path / "existing.parquet"
    output.write_bytes(b"existing table")
    with pytest.raises(subprocess.CalledProcessError, match="non-zero exit") as failure:
        run(
            engine,
            "prepare",
            path,
            tmp_path / "missing-replays",
            output,
            "bumping_teammate",
        )
    assert "native preparation requires initiator labels" in failure.value.stderr
    assert output.read_bytes() == b"existing table"


@pytest.mark.parametrize(
    "mutation", ["missing_incident", "wrong_recording", "wrong_intrinsic_id"]
)
def test_failed_preparation_preserves_existing_table(engine, prepared_inputs, mutation):
    path, recordings, output, labels = prepared_inputs
    output.write_bytes(b"existing table")
    if mutation == "missing_incident":
        labels["annotations"][0]["frame"] = 1
        labels["annotations"][0]["evidence"]["frame"] = 1
    elif mutation == "wrong_recording":
        (recordings / f"{labels['replays'][0]['sha256']}.replay").write_bytes(
            b"other recording"
        )
    else:
        labels["replays"][0]["rocket_league_id"] = "different-recording"
    path.write_text(json.dumps(labels))
    with pytest.raises(subprocess.CalledProcessError):
        run(engine, "prepare", path, recordings, output, "bumping_teammate")
    assert output.read_bytes() == b"existing table"
    assert sorted(p.name for p in output.parent.iterdir()) == [
        "annotations.json",
        "examples.parquet",
        "replays",
    ]


@pytest.fixture(scope="session")
def prepared_table(engine, extracted, tmp_path_factory):
    root = tmp_path_factory.mktemp("prepared-table")
    document, batch = extracted
    path = root / "annotations.json"
    path.write_text(json.dumps(dataset(document, batch["candidates"])))
    recordings = root / "replays"
    recordings.mkdir()
    shutil.copyfile(
        ROOT / "tests/fixtures/sample.replay",
        recordings / f"{document['replay_sha256']}.replay",
    )
    output = root / "examples.parquet"
    run(engine, "prepare", path, recordings, output, "bumping_teammate")
    return pq.read_table(output)


@pytest.mark.parametrize("mutation", ["duplicate", "bad_label", "nonfinite_feature"])
def test_training_reader_rejects_corrupt_tables(prepared_table, tmp_path, mutation):
    table = prepared_table
    if mutation == "duplicate":
        table = pa.concat_tables([table, table.slice(0, 1)])
    else:
        name = "label" if mutation == "bad_label" else "closing_speed"
        values = table[name].to_pylist()
        values[0] = 2 if mutation == "bad_label" else float("nan")
        table = table.set_column(
            table.schema.get_field_index(name),
            name,
            pa.array(values, type=table[name].type),
        )
    output = tmp_path / "corrupt.parquet"
    pq.write_table(table, output)
    with pytest.raises(ValueError):
        load(output)


def test_replay_download_checks_bytes_and_preserves_reference(
    prepared_inputs, monkeypatch
):
    path, recordings, _, labels = prepared_inputs
    recording = recordings / f"{labels['replays'][0]['sha256']}.replay"
    payload = recording.read_bytes()
    recording.unlink()
    calls = []

    class Opener:
        def open(self, request, timeout):
            calls.append(
                (request.full_url, request.get_header("Authorization"), timeout)
            )
            return io.BytesIO(payload)

    monkeypatch.setattr("training.fetch_replays.build_opener", lambda *_: Opener())
    assert fetch(path, recordings, "test-token") == 1
    assert recording.read_bytes() == payload
    assert fetch(path, recordings, None) == 0
    assert calls == [
        ("https://ballchasing.com/api/replays/fixture-id/file", "test-token", 60)
    ]


@pytest.mark.parametrize("failure", ["bad_bytes", "http_error"])
def test_failed_download_leaves_no_recording(prepared_inputs, monkeypatch, failure):
    path, recordings, _, _ = prepared_inputs
    for recording in recordings.iterdir():
        recording.unlink()

    class Opener:
        def open(self, request, timeout):
            if failure == "http_error":
                raise HTTPError(request.full_url, 429, "Rate limited", {}, None)
            return io.BytesIO(b"wrong replay")

    monkeypatch.setattr("training.fetch_replays.build_opener", lambda *_: Opener())
    with pytest.raises((ValueError, HTTPError)):
        fetch(path, recordings, "test-token")
    assert list(recordings.iterdir()) == []


def test_exported_catalog_works_with_rust_inference(engine, prepared_table, tmp_path):
    from training.export import model_entry, write_manifest

    table_path = tmp_path / "training.parquet"
    pq.write_table(prepared_table, table_path)
    shutil.copyfile(
        ROOT / "crates/inference/tests/fixtures/square.onnx", tmp_path / "model.onnx"
    )
    (tmp_path / "evaluation.json").write_text("{}")
    entries = [
        model_entry(
            tmp_path,
            table_path,
            version=version,
            model_file="model.onnx",
            evaluation_file="evaluation.json",
            keep_threshold=threshold,
            source_revision="export-fixture",
            status="experimental",
        )
        for version, threshold in [("1.0.0", 0.5), ("1.1.0", 0.9)]
    ]
    write_manifest(tmp_path, "1.1.0", entries)
    catalog = json.loads((tmp_path / "manifest.json").read_bytes())
    assert len(catalog["models"]) == 2
    assert "dataset_sha256" not in entries[0]["provenance"]
    assert json.loads(run(engine, "validate-manifest", tmp_path).stdout)["models"] == 2
    predictions = json.loads(
        run(
            engine,
            "score",
            tmp_path,
            "bumping_teammate",
            "1.0.0",
            document=(ROOT / "examples/bump.json").read_text(),
        ).stdout
    )
    assert predictions["predictions"][0]["score"] == pytest.approx(0.64)
    previous = (tmp_path / "manifest.json").read_bytes()
    (tmp_path / "model.onnx").write_bytes(b"changed artifact")
    with pytest.raises(ValueError, match="hash mismatch"):
        write_manifest(tmp_path, "1.2.0", entries)
    assert (tmp_path / "manifest.json").read_bytes() == previous


def test_export_discovers_external_weights(engine, prepared_table, tmp_path):
    from training.export import model_entry, write_manifest

    table_path = tmp_path / "training.parquet"
    pq.write_table(prepared_table, table_path)
    fixtures = ROOT / "crates/inference/tests/fixtures"
    shutil.copyfile(fixtures / "external.onnx", tmp_path / "model.onnx")
    shutil.copyfile(fixtures / "external.data", tmp_path / "external.data")
    (tmp_path / "evaluation.json").write_text("{}")
    entry = model_entry(
        tmp_path,
        table_path,
        version="1.0.0",
        model_file="model.onnx",
        evaluation_file="evaluation.json",
        keep_threshold=0.5,
        source_revision="export-fixture",
        status="experimental",
    )
    assert {file["path"] for file in entry["files"]} == {
        "model.onnx",
        "evaluation.json",
        "external.data",
    }
    write_manifest(tmp_path, "1.0.0", [entry])
    predictions = json.loads(
        run(
            engine,
            "score",
            tmp_path,
            "bumping_teammate",
            "1.0.0",
            document=(ROOT / "examples/bump.json").read_text(),
        ).stdout
    )
    assert predictions["predictions"][0]["score"] == pytest.approx(0.64)


def test_candidate_retains_complete_supplier_payload(engine):
    document = json.loads((ROOT / "examples/bump.json").read_text())
    payload = document["events"][0]["payload"]["payload"]
    payload["additional_context"] = {"ball_position": [0, 200, 300]}
    batch = json.loads(run(engine, "candidates", document=json.dumps(document)).stdout)[
        0
    ]
    assert batch["candidates"][0]["evidence"] == payload
