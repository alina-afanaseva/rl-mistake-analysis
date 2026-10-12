"""Join reviewed incidents to player-specific teammate-bump context features."""

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

METADATA_KEY = b"rl_mistake_analysis_features"
IDENTITY_FIELDS = ("replay_sha256", "frame", "initiator", "victim", "player_id")
DATA_TYPES = {"boolean": pa.bool_(), "f32": pa.float32(), "u32": pa.uint32()}
CONTEXT_SCHEMA = "bumping_teammate.context.v1"


def supports_schema(schema):
    """Named v1 subsets share the base context measurements."""
    return isinstance(schema, str) and (
        schema == CONTEXT_SCHEMA
        or re.fullmatch(rf"{re.escape(CONTEXT_SCHEMA)}\.[A-Za-z0-9_]+\.v1", schema)
        is not None
    )


def identity(record):
    participants = [record[name] for name in IDENTITY_FIELDS[2:]]
    if any(not isinstance(player, dict) or not player for player in participants):
        raise ValueError("Player identities must be nonempty JSON objects")
    initiator, victim, player = participants
    if initiator == victim or player not in (initiator, victim):
        raise ValueError("The judged player must be one of two distinct participants")
    checksum, frame = record["replay_sha256"], record["frame"]
    if not isinstance(checksum, str) or not re.fullmatch(r"[0-9a-f]{64}", checksum):
        raise ValueError("Invalid replay checksum")
    if type(frame) is not int or frame < 0:
        raise ValueError("Frames must be nonnegative integers")
    return (
        checksum,
        frame,
        *(json.dumps(player, sort_keys=True) for player in participants),
    )


def load(path, *, labeled=False):
    """Validate the context contract, preserving declared feature order and nulls."""
    table = pq.read_table(path)
    table.validate(full=True)
    if not table.schema.metadata or METADATA_KEY not in table.schema.metadata:
        raise ValueError("Missing teammate-bump context metadata")
    contract = json.loads(table.schema.metadata[METADATA_KEY])
    if (
        contract["schema_version"] != 1
        or contract["kind"] != "bumping_teammate"
        or not supports_schema(contract["input_schema"])
        or not table.num_rows
    ):
        raise ValueError("Unsupported teammate-bump context contract or empty table")
    definitions = contract["feature_definitions"]
    names = [feature["name"] for feature in definitions]
    required = [*IDENTITY_FIELDS, "raw_time", "engine_revision", "source_event_id"]
    if labeled:
        required.append("label")
    if (
        not names
        or any(not isinstance(name, str) or not name.strip() for name in names)
        or len(set(names)) != len(names)
        or set(names)
        & {*required, "label", "annotation_engine_revision", "reject_reason"}
        or len(set(table.column_names)) != table.num_columns
        or any(name not in table.column_names for name in [*required, *names])
    ):
        raise ValueError("Invalid context feature names or missing required columns")
    if any(table[name].null_count for name in required):
        raise ValueError(
            "Required identities, timestamps, revisions and labels cannot be null"
        )
    if not pa.types.is_integer(table["frame"].type):
        raise ValueError("Frames must be integers")
    if not pa.types.is_floating(table["raw_time"].type):
        raise ValueError("Replay timestamps must be floating-point numbers")
    if not np.isfinite(table["raw_time"].to_numpy()).all():
        raise ValueError("Replay timestamps must be finite")
    for feature in definitions:
        name = feature["name"]
        if (
            feature["data_type"] not in DATA_TYPES
            or table[name].type != DATA_TYPES[feature["data_type"]]
        ):
            raise ValueError(f"Feature type differs from its contract: {name}")
        if type(feature["nullable"]) is not bool:
            raise ValueError(f"Invalid feature nullability: {name}")
        if table[name].null_count and not feature["nullable"]:
            raise ValueError(f"Non-nullable feature contains nulls: {name}")
        if not np.isfinite(table[name].drop_null().to_numpy()).all():
            raise ValueError(f"Observed feature values must be finite: {name}")
    keys = []
    for row in table.select(IDENTITY_FIELDS).to_pylist():
        for name in IDENTITY_FIELDS[2:]:
            row[name] = json.loads(row[name])
        keys.append(identity(row))
    if len(set(keys)) != len(keys):
        raise ValueError("Duplicate player-specific collision")
    if labeled and (
        not pa.types.is_integer(table["label"].type)
        or any(value not in (0, 1) for value in table["label"].to_pylist())
    ):
        raise ValueError("Labels must be integer 0 or 1")
    return table, contract


def prepare(features, annotations, output):
    """Select reviewed subjects by exact replay, collision, player and timestamp."""
    output = Path(output)
    if output.suffix != ".parquet":
        raise ValueError("The labeled table must use a .parquet extension")
    if output.resolve() in (Path(features).resolve(), Path(annotations).resolve()):
        raise ValueError("The labeled table must not replace its inputs")
    table, _ = load(features)
    document = json.loads(Path(annotations).read_text())
    if document["schema_version"] != 1 or not document["annotations"]:
        raise ValueError(
            "Provide a version-1 annotation document with reviewed players"
        )
    rows = table.select([*IDENTITY_FIELDS, "raw_time", "source_event_id"]).to_pylist()
    lookup = {}
    for index, row in enumerate(rows):
        for name in IDENTITY_FIELDS[2:]:
            row[name] = json.loads(row[name])
        lookup[identity(row)] = index, row
    indices, labels, revisions, reasons = [], [], [], []
    seen = set()
    for review in document["annotations"]:
        if review["kind"] != "bumping_teammate":
            raise ValueError("Annotations must review teammate bumps")
        if "player_id" not in review:
            raise ValueError("Each review requires an explicit judged player_id")
        key = identity(review)
        if key in seen:
            raise ValueError("Duplicate player-specific annotation")
        seen.add(key)
        if key not in lookup:
            raise ValueError(
                f"Reviewed player-specific collision is absent from features: {key}"
            )
        index, row = lookup[key]
        if review["raw_time"] != row["raw_time"]:
            raise ValueError("Review and feature replay timestamps differ")
        if "source_event_ids" in review and review["source_event_ids"] != [
            row["source_event_id"]
        ]:
            raise ValueError("Review and feature source events differ")
        if type(review["label"]) is not int or review["label"] not in (0, 1):
            raise ValueError("Review labels must be integer 0 or 1")
        if (
            not isinstance(review["engine_revision"], str)
            or not review["engine_revision"].strip()
        ):
            raise ValueError(
                "Review engine_revision must identify the reviewed extraction"
            )
        indices.append(index)
        labels.append(review["label"])
        revisions.append(review["engine_revision"])
        reasons.append(review.get("reject_reason"))
    prepared = table.take(indices)
    for name, values, data_type in (
        ("label", labels, pa.uint8()),
        ("annotation_engine_revision", revisions, pa.string()),
        ("reject_reason", reasons, pa.string()),
    ):
        if name in prepared.column_names:
            raise ValueError(
                f"Extracted table already contains a review column: {name}"
            )
        prepared = prepared.append_column(name, pa.array(values, type=data_type))
    output.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(prepared, output, compression="snappy")
    return {
        "reviewed_rows": len(indices),
        "unreviewed_rows": table.num_rows - len(indices),
    }


def arrays(table, contract):
    """Convert boolean and numeric measurements to float32; nulls become NaN."""
    features = np.column_stack(
        [
            table[feature["name"]].cast(pa.float32()).to_numpy()
            for feature in contract["feature_definitions"]
        ]
    )
    return features, table["label"].to_numpy(), table["replay_sha256"].to_numpy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("features", type=Path)
    parser.add_argument("annotations", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    print(json.dumps(prepare(args.features, args.annotations, args.output)))


if __name__ == "__main__":
    main()
