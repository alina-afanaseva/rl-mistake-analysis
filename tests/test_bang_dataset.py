import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from training.bang_dataset import prepare


def inputs(tmp_path):
    features = tmp_path / "features.parquet"
    annotations = tmp_path / "annotations.json"
    row = {
        "replay_sha256": "a" * 64,
        "frame": 42,
        "touch_id": 7,
        "player_id": json.dumps({"Epic": "player"}),
        "raw_time": 1.4,
        "source_event_id": "touch:42:0",
        "engine_revision": "feature-supplier",
        "candidate_incoming_ball_speed": 123.0,
        "second_touch_delay_s": None,
    }
    review = {
        **row,
        "kind": "bang_with_time",
        "player_id": {"Epic": "player"},
        "source_event_ids": [row["source_event_id"]],
        "engine_revision": "review-supplier",
        "label": 1,
    }
    annotations.write_text(json.dumps({"schema_version": 1, "annotations": [review]}))
    return features, annotations, row


def feature_table(row, names=("candidate_incoming_ball_speed", "second_touch_delay_s")):
    base = {
        key: value
        for key, value in row.items()
        if key not in ("candidate_incoming_ball_speed", "second_touch_delay_s")
    }
    base.update({name: row[name] for name in names})
    table = pa.Table.from_pylist([base])
    for name in names:
        table = table.set_column(
            table.schema.get_field_index(name), name, table[name].cast(pa.float32())
        )
    contract = {
        "schema_version": 1,
        "kind": "bang_with_time",
        "input_schema": "bang_with_time.context.v1.test.v1",
        "sampling": {"method": "native_touch_stages"},
        "feature_definitions": [
            {
                "name": name,
                "data_type": "f32",
                "nullable": True,
                "source": "replay_data.frame_data",
            }
            for name in names
        ],
    }
    return table.replace_schema_metadata(
        {b"rl_mistake_analysis_features": json.dumps(contract).encode()}
    )


def test_one_annotation_joins_different_feature_sets_without_changing_order_or_nulls(
    tmp_path,
):
    features, annotations, row = inputs(tmp_path)
    for names in (
        ("candidate_incoming_ball_speed", "second_touch_delay_s"),
        ("second_touch_delay_s",),
    ):
        table = feature_table(row, names)
        pq.write_table(table, features)
        output = tmp_path / "labeled.parquet"
        assert prepare(features, annotations, output) == {
            "reviewed_rows": 1,
            "unreviewed_rows": 0,
        }
        labeled = pq.read_table(output)
        assert labeled.column_names == [
            *table.column_names,
            "label",
            "annotation_engine_revision",
            "reject_reason",
        ]
        assert labeled.schema.metadata == table.schema.metadata
        assert labeled["second_touch_delay_s"].to_pylist() == [None]
        assert labeled["annotation_engine_revision"].to_pylist() == ["review-supplier"]


@pytest.mark.parametrize(
    "field,value",
    [
        ("frame", 43),
        ("touch_id", 8),
        ("raw_time", 1.5),
        ("source_event_id", "touch:43:0"),
        ("player_id", '{"Epic":"other"}'),
    ],
)
def test_nearby_or_different_touch_never_receives_a_label(tmp_path, field, value):
    features, annotations, row = inputs(tmp_path)
    row[field] = value
    pq.write_table(feature_table(row), features)
    with pytest.raises(ValueError):
        prepare(features, annotations, tmp_path / "labeled.parquet")


def test_incident_identity_matches_when_annotation_omits_source_ids(tmp_path):
    features, annotations, row = inputs(tmp_path)
    document = json.loads(annotations.read_text())
    del document["annotations"][0]["source_event_ids"]
    annotations.write_text(json.dumps(document))
    pq.write_table(feature_table(row), features)
    output = tmp_path / "labeled.parquet"
    assert prepare(features, annotations, output)["reviewed_rows"] == 1
    assert pq.read_table(output)["label"].to_pylist() == [1]
