import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from training.bump_dataset import METADATA_KEY, arrays, load, prepare


@pytest.fixture
def feature_files(tmp_path):
    participants = [{"Steam": "123"}, {"Epic": "456"}]
    rows = [
        {
            "replay_sha256": "a" * 64,
            "frame": 42,
            "initiator": json.dumps(participants[0]),
            "victim": json.dumps(participants[1]),
            "player_id": json.dumps(player),
            "raw_time": 10.25,
            "source_event_id": "bump:42:0",
            "engine_revision": "extraction-revision",
            "p_is_initiator": index == 0,
            "p_velocity_delta_toward": None if index == 0 else -50.0,
            "touches_before_next_goal": None,
        }
        for index, player in enumerate(participants)
    ]
    contract = {
        "schema_version": 1,
        "kind": "bumping_teammate",
        "input_schema": "bumping_teammate.context.v1",
        "feature_definitions": [
            {"name": "touches_before_next_goal", "data_type": "u32", "nullable": True},
            {"name": "p_is_initiator", "data_type": "boolean", "nullable": False},
            {"name": "p_velocity_delta_toward", "data_type": "f32", "nullable": True},
        ],
    }
    schema = pa.schema(
        [
            pa.field(name, pa.string())
            for name in (
                "replay_sha256",
                "initiator",
                "victim",
                "player_id",
                "source_event_id",
                "engine_revision",
            )
        ]
        + [
            pa.field("frame", pa.uint64()),
            pa.field("raw_time", pa.float64()),
            pa.field("p_is_initiator", pa.bool_()),
            pa.field("p_velocity_delta_toward", pa.float32()),
            pa.field("touches_before_next_goal", pa.uint32()),
        ],
        metadata={METADATA_KEY: json.dumps(contract).encode()},
    )
    features = tmp_path / "features.parquet"
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), features)
    reviews = [
        {
            **{name: row[name] for name in ("replay_sha256", "frame", "raw_time")},
            **{
                name: json.loads(row[name])
                for name in ("initiator", "victim", "player_id")
            },
            "kind": "bumping_teammate",
            "label": index,
            "engine_revision": "review-revision",
            "source_event_ids": [row["source_event_id"]],
        }
        for index, row in enumerate(rows)
    ]
    annotations = tmp_path / "annotations.json"
    annotations.write_text(json.dumps({"schema_version": 1, "annotations": reviews}))
    return features, annotations, tmp_path / "training.parquet", reviews


def test_preserves_each_subject_label_nulls_and_contract_order(feature_files):
    features, annotations, output, _ = feature_files
    original = features.read_bytes()
    assert prepare(features, annotations, output) == {
        "reviewed_rows": 2,
        "unreviewed_rows": 0,
    }
    table, contract = load(output, labeled=True)
    X, y, groups = arrays(table, contract)
    assert X.dtype == np.float32
    np.testing.assert_equal(X, [[np.nan, 1, np.nan], [np.nan, 0, -50]])
    np.testing.assert_array_equal(y, [0, 1])
    assert groups.tolist() == ["a" * 64, "a" * 64]
    assert table["annotation_engine_revision"].to_pylist() == ["review-revision"] * 2
    assert features.read_bytes() == original


def test_omits_unreviewed_player_without_assigning_a_negative(feature_files):
    features, annotations, output, reviews = feature_files
    annotations.write_text(
        json.dumps({"schema_version": 1, "annotations": reviews[1:]})
    )
    assert prepare(features, annotations, output) == {
        "reviewed_rows": 1,
        "unreviewed_rows": 1,
    }
    table, _ = load(output, labeled=True)
    assert table["label"].to_pylist() == [1]
    assert json.loads(table["player_id"][0].as_py()) == reviews[1]["victim"]


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ("duplicate", "Duplicate player-specific annotation"),
        ("missing_player", "explicit judged player_id"),
        ("unmatched", "absent from features"),
        ("time", "timestamps differ"),
        ("label", "integer 0 or 1"),
    ],
)
def test_invalid_reviews_do_not_replace_a_previous_output(
    feature_files, change, message
):
    features, annotations, output, reviews = feature_files
    if change == "duplicate":
        reviews.append(reviews[0])
    elif change == "missing_player":
        del reviews[0]["player_id"]
    elif change == "unmatched":
        reviews[0]["frame"] += 1
    elif change == "time":
        reviews[0]["raw_time"] += 0.01
    elif change == "label":
        reviews[0]["label"] = True
    annotations.write_text(json.dumps({"schema_version": 1, "annotations": reviews}))
    output.write_bytes(b"previous output")
    with pytest.raises(ValueError, match=message):
        prepare(features, annotations, output)
    assert output.read_bytes() == b"previous output"


def test_rejects_observed_nonfinite_features_and_duplicate_subjects(feature_files):
    features, _, _, _ = feature_files
    table = pq.read_table(features)
    pq.write_table(pa.concat_tables([table, table]), features)
    with pytest.raises(ValueError, match="Duplicate player-specific collision"):
        load(features)
    table = table.set_column(
        table.schema.get_field_index("p_velocity_delta_toward"),
        "p_velocity_delta_toward",
        pa.array([None, np.inf], type=pa.float32()),
    )
    pq.write_table(table, features)
    with pytest.raises(ValueError, match="Observed feature values must be finite"):
        load(features)
