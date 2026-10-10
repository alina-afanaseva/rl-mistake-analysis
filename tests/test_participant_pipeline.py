"""Verify participant identities across review, training export, and inference."""

import json
import subprocess
from pathlib import Path

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper

from training.bump_dataset import prepare
from training.bump_features import compute
from training.dataset import arrays, load
from training.export import model_entry, write_manifest

ROOT = Path(__file__).resolve().parents[1]
SAMPLING = ROOT / "examples/bump_sampling.json"


def analysis():
    players = [{"Steam": "1"}, {"Steam": "2"}]

    def body(x):
        return {
            "location": {"x": x, "y": 0, "z": 17},
            "rotation": {"x": 0, "y": 0, "z": 0, "w": 1},
            "linear_velocity": {"x": 0, "y": 0, "z": 0},
        }

    bump = {
        "frame": 2,
        "time": 0.25,
        "initiator": players[0],
        "victim": players[1],
        "is_team_bump": True,
        "initiator_is_team_0": True,
        "victim_is_team_0": True,
        "contact_distance": 5,
        "confidence": 0.9,
        "closing_speed": 400,
        "victim_impulse": 200,
        "strength": 0.2,
        "initiator_position": [0, 0, 17],
        "victim_position": [100, 0, 17],
    }
    return {
        "name": "subtr-actor",
        "schema_version": 1,
        "status": "ok",
        "revision": "participant-fixture",
        "replay_sha256": "a" * 64,
        "outputs": {
            "events": {
                "events": {
                    "events": [
                        {
                            "meta": {"id": "bump:2", "stream": "bump"},
                            "payload": {"kind": "bump", "payload": bump},
                        },
                        {
                            "meta": {"id": "possession:0", "stream": "possession"},
                            "payload": {
                                "kind": "possession",
                                "payload": {
                                    "time": 0.0,
                                    "end_time": 0.875,
                                    "possession_state": "neutral",
                                },
                            },
                        },
                    ]
                },
                "accumulation_tracks": [
                    {
                        "player_id": player,
                        "quantity": "boost_amount",
                        "points": [{"frame": 0, "value": 100}],
                    }
                    for player in players
                ],
            },
            "replay_data": {
                "frame_data": {
                    "metadata_frames": [{"time": frame / 8} for frame in range(8)],
                    "ball_data": {
                        "frames": [
                            {"Data": {"rigid_body": body(1000)}} for _ in range(8)
                        ]
                    },
                    "players": [
                        [
                            player,
                            {
                                "frames": [
                                    {
                                        "Data": {
                                            "rigid_body": body(index * 100),
                                            "dodge_active": False,
                                        }
                                    }
                                    for _ in range(8)
                                ]
                            },
                        ]
                        for index, player in enumerate(players)
                    ],
                },
                "boost_pad_events": [],
                "boost_pads": [],
            },
        },
    }


def run(engine, *args, document):
    return json.loads(
        subprocess.run(
            [str(engine), *map(str, args)],
            input=json.dumps(document),
            text=True,
            capture_output=True,
            check=True,
        ).stdout
    )


def model(path, names):
    inputs = [
        helper.make_tensor_value_info("features", TensorProto.FLOAT, [None, len(names)])
    ]
    outputs = [
        helper.make_tensor_value_info("probability", TensorProto.FLOAT, [None, 1])
    ]
    initializers = [
        helper.make_tensor(
            "role_index", TensorProto.INT64, [1], [names.index("p_is_initiator")]
        ),
        helper.make_tensor(
            "missing_index",
            TensorProto.INT64,
            [1],
            [names.index("p_velocity_delta_up")],
        ),
        helper.make_tensor("one", TensorProto.FLOAT, [1], [1]),
    ]
    nodes = [
        helper.make_node("Gather", ["features", "role_index"], ["role"], axis=1),
        helper.make_node("Sub", ["one", "role"], ["victim"]),
        helper.make_node(
            "Gather", ["features", "missing_index"], ["measurement"], axis=1
        ),
        helper.make_node("IsNaN", ["measurement"], ["missing"]),
        helper.make_node("Cast", ["missing"], ["missing_float"], to=TensorProto.FLOAT),
        helper.make_node("Mul", ["victim", "missing_float"], ["probability"]),
    ]
    graph = helper.make_graph(nodes, "participant-test", inputs, outputs, initializers)
    onnx.save(
        helper.make_model(
            graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=9
        ),
        path,
    )


def test_same_rows_are_reviewed_prepared_and_scored(engine, tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT.parent / "event-review"))
    from store import Store

    document = analysis()
    events = {
        key: document[key]
        for key in ("name", "schema_version", "status", "revision", "replay_sha256")
    }
    events["events"] = document["outputs"]["events"]["events"]["events"]
    batch = run(engine, "candidates", document=events)[0]
    assert len(batch["candidates"]) == 2
    assert len(batch["feature_names"]) == 8
    assert "sampling" not in batch and "feature_definitions" not in batch
    assert batch["candidates"][0]["features"] == batch["candidates"][1]["features"]
    source = tmp_path / "analysis.json"
    source.write_text(json.dumps(document))
    features = tmp_path / "features.parquet"
    compute([source], engine, SAMPLING, features)
    store = Store(tmp_path / "review.sqlite3")
    for index, candidate in enumerate(batch["candidates"]):
        candidate["player_name"] = ("Alice", "Bob")[index]
    store.import_document(
        {
            "format": "candidate_batch",
            "document": batch,
            "replay_id": "match",
            "time_offset": 0,
        }
    )
    for event in store.export()["events"]:
        store.mutate(
            {
                "action": "review_event",
                "event_id": event["event_id"],
                "label": int(event["player_id"] == {"Steam": "2"}),
            }
        )
    reviews = tmp_path / "reviews.json"
    reviews.write_text(json.dumps(store.export(batch["input_schema"])))
    training = tmp_path / "training.parquet"
    assert prepare(features, reviews, training)["reviewed_rows"] == 2
    table, contract = load(training)
    X, y, groups = arrays(table, contract)
    by_player = {
        json.loads(player)["Steam"]: index
        for index, player in enumerate(table["player_id"].to_pylist())
    }
    measurements = run(engine, "bump-features", SAMPLING, document=document)
    assert len(measurements["feature_definitions"]) == 105
    for candidate in batch["candidates"]:
        index = by_player[candidate["player_id"]["Steam"]]
        row = next(
            row
            for row in measurements["rows"]
            if row["player_id"] == candidate["player_id"]
        )
        np.testing.assert_equal(
            X[index],
            np.array(
                [row["features"][name] for name in contract["feature_names"]],
                dtype=np.float32,
            ),
        )
        assert y[index] == int(candidate["player_id"] == {"Steam": "2"})
    assert groups.tolist() == ["a" * 64] * 2
    model(tmp_path / "model.onnx", contract["feature_names"])
    (tmp_path / "evaluation.json").write_text("{}")
    entry = model_entry(
        tmp_path,
        training,
        version="test",
        model_file="model.onnx",
        evaluation_file="evaluation.json",
        keep_threshold=0.5,
        source_revision="participant-test",
        status="experimental",
    )
    assert entry["sampling"] == json.loads(SAMPLING.read_text())
    write_manifest(tmp_path, "test", [entry])
    predictions = run(
        engine, "score", tmp_path, "bumping_teammate", "test", document=document
    )
    assert len(predictions["predictions"]) == 2
    assert predictions["input_schema"] == batch["input_schema"]
    assert predictions["feature_names"] == batch["feature_names"]
    assert predictions["model_input_schema"] == contract["input_schema"]
    assert len(predictions["model_feature_names"]) == 105
    for prediction in predictions["predictions"]:
        candidate = prediction["candidate"]
        original = next(
            row
            for row in batch["candidates"]
            if row["player_id"] == candidate["player_id"]
        )
        assert candidate["features"] == original["features"]
        assert len(candidate["features"]) == 8
        expected = int(candidate["player_id"] == {"Steam": "2"})
        assert prediction["score"] == pytest.approx(expected)
        assert prediction["keep"] == bool(expected)
    del entry["sampling"]
    write_manifest(tmp_path, "test", [entry])
    with pytest.raises(subprocess.CalledProcessError):
        run(engine, "score", tmp_path, "bumping_teammate", "test", document=document)
