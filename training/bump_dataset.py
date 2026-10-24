"""Join reviewed incidents to player-specific teammate-bump context features."""

import json
import re

from training import context_dataset

METADATA_KEY = context_dataset.METADATA_KEY
IDENTITY_FIELDS = ("replay_sha256", "frame", "initiator", "victim", "player_id")
CONTEXT_SCHEMA = "bumping_teammate.context.v1"


def supports_schema(schema):
    return context_dataset.supports_schema(schema, CONTEXT_SCHEMA)


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
    return context_dataset.load(
        path,
        kind="bumping_teammate",
        supports_schema=supports_schema,
        identity_fields=IDENTITY_FIELDS,
        json_fields=IDENTITY_FIELDS[2:],
        identity=identity,
        duplicate_message="Duplicate player-specific collision",
        labeled=labeled,
    )


def prepare(features, annotations, output):
    return context_dataset.prepare(
        features,
        annotations,
        output,
        kind="bumping_teammate",
        load=load,
        identity_fields=IDENTITY_FIELDS,
        json_fields=IDENTITY_FIELDS[2:],
        identity=identity,
    )


arrays = context_dataset.arrays


if __name__ == "__main__":
    context_dataset.main(prepare, __doc__)
