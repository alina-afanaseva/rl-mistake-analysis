import json
import shutil
import subprocess
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from training.bump_features import compute

ROOT = Path(__file__).resolve().parents[1]
SAMPLING = ROOT / "examples/bump_sampling.json"


def test_feature_table_preserves_types_subjects_sampling_and_nulls(
    engine, combined_analysis, tmp_path
):
    output = tmp_path / "features.parquet"
    alias = tmp_path / "alias.json"
    shutil.copyfile(combined_analysis, alias)
    summary = compute([combined_analysis, alias], engine, SAMPLING, output)
    table = pq.read_table(output)
    metadata = json.loads(table.schema.metadata[b"rl_mistake_analysis_features"])
    assert summary == {"analyses": 2, "replays": 1, "rows": 10, "features": 105}
    assert metadata["replays"][0]["replay_ids"] == ["sample", "alias"]
    assert metadata["replays"][0]["analysis_files"] == [
        str(combined_analysis),
        str(alias),
    ]
    assert metadata["input_schema"] == "bumping_teammate.context.v1"
    assert metadata["sampling"] == json.loads(SAMPLING.read_text())
    assert table.schema.field("p_is_initiator").type == pa.bool_()
    assert table.schema.field("q_boost_before_pct").type == pa.float32()
    assert table.schema.field("touches_before_next_goal").type == pa.uint32()
    assert not table.schema.field("p_is_initiator").nullable
    assert table.schema.field("q_velocity_delta_up").nullable
    contacts = {}
    for row in table.to_pylist():
        contacts.setdefault(row["source_event_id"], []).append(row)
        samples = json.loads(row["samples"])
        for participant in ("p", "q"):
            sample = samples[participant]
            if sample["status"] == "complete":
                assert sample["after_time"] > row["raw_time"]
                assert sample["after_frame"] > sample["peak_frame"]
            else:
                assert sample["after_frame"] is None
                assert row[f"{participant}_velocity_delta_up"] is None
    for pair in contacts.values():
        assert len(pair) == 2
        assert {row["p_is_initiator"] for row in pair} == {False, True}
        assert pair[0]["player_id"] == pair[1]["teammate_id"]
        assert pair[1]["player_id"] == pair[0]["teammate_id"]
        assert pair[0]["p_boost_before_pct"] == pair[1]["q_boost_before_pct"]
    assert table["p_previous_demolition_age_s"].null_count > 0
    assert "label" not in table.column_names


def test_failed_input_and_duplicate_replay_do_not_replace_a_saved_table(
    engine, combined_analysis, tmp_path
):
    output = tmp_path / "features.parquet"
    output.write_bytes(b"existing table")
    invalid = tmp_path / "invalid.json"
    invalid.write_text("{}")
    with pytest.raises(ValueError, match="invalid.json"):
        compute([combined_analysis, invalid], engine, SAMPLING, output)
    assert output.read_bytes() == b"existing table"
    with pytest.raises(ValueError, match="Duplicate replay analysis"):
        compute([combined_analysis, combined_analysis], engine, SAMPLING, output)
    assert output.read_bytes() == b"existing table"


def test_conflicting_analyses_of_one_replay_do_not_replace_a_saved_table(
    engine, combined_analysis, tmp_path
):
    document = json.loads(combined_analysis.read_text())
    document["revision"] = "conflicting supplier revision"
    conflicting = tmp_path / "conflicting.json"
    conflicting.write_text(json.dumps(document))
    output = tmp_path / "features.parquet"
    output.write_bytes(b"existing table")
    with pytest.raises(ValueError, match="Conflicting analyses"):
        compute([combined_analysis, conflicting], engine, SAMPLING, output)
    assert output.read_bytes() == b"existing table"


def test_cli_accepts_an_explicit_victim_subject(engine, combined_analysis):
    with combined_analysis.open("rb") as source:
        batch = json.loads(
            subprocess.run(
                [str(engine), "bump-features", str(SAMPLING)],
                stdin=source,
                capture_output=True,
                check=True,
            ).stdout
        )
    victim = batch["rows"][0]["victim"]
    with combined_analysis.open("rb") as source:
        selected = json.loads(
            subprocess.run(
                [
                    str(engine),
                    "bump-features",
                    str(SAMPLING),
                    "--player",
                    json.dumps(victim),
                ],
                stdin=source,
                capture_output=True,
                check=True,
            ).stdout
        )
    assert selected["rows"]
    assert all(row["player_id"] == victim for row in selected["rows"])
    assert any(not row["features"]["p_is_initiator"] for row in selected["rows"])
