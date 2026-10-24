"""Read bang-with-time context tables and join exact native touch reviews."""

import json
import re

from training import context_dataset

METADATA_KEY = context_dataset.METADATA_KEY
CONTEXT_SCHEMA = "bang_with_time.context.v1"
arrays = context_dataset.arrays


IDENTITY_FIELDS = ("replay_sha256", "frame", "touch_id", "player_id")


def identity(row):
    if not re.fullmatch(r"[0-9a-f]{64}", row["replay_sha256"]):
        raise ValueError("Invalid replay fingerprint")
    if any(
        type(row[name]) is not int or row[name] < 0 for name in ("frame", "touch_id")
    ):
        raise ValueError("Frame and touch ID must be nonnegative integers")
    player = row["player_id"]
    if not isinstance(player, dict) or not player:
        raise ValueError("An explicit player identity is required")
    return (
        *(row[name] for name in IDENTITY_FIELDS[:3]),
        json.dumps(player, sort_keys=True),
    )


def supports_schema(schema):
    return context_dataset.supports_schema(schema, CONTEXT_SCHEMA)


def load(path, *, labeled=False):
    return context_dataset.load(
        path,
        kind="bang_with_time",
        supports_schema=supports_schema,
        identity_fields=IDENTITY_FIELDS,
        json_fields=("player_id",),
        identity=identity,
        duplicate_message="Duplicate touch identity",
        labeled=labeled,
    )


def prepare(features, annotations, output):
    return context_dataset.prepare(
        features,
        annotations,
        output,
        kind="bang_with_time",
        load=load,
        identity_fields=IDENTITY_FIELDS,
        json_fields=("player_id",),
        identity=identity,
    )


if __name__ == "__main__":
    context_dataset.main(prepare, __doc__)
