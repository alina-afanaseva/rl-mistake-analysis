"""Compute native bang-with-time measurements from combined replay analyses."""

import json
from pathlib import Path

from training.context_features import compute as compute_context
from training.context_features import main


def compute(paths, engine, sampling, output):
    return compute_context(
        paths,
        engine,
        output,
        kind="bang_with_time",
        command=["bang-features", str(sampling)],
        expected_sampling=json.loads(Path(sampling).read_text()),
        participant_fields=(),
        integer_fields=("touch_id",),
    )


if __name__ == "__main__":
    main(compute, __doc__)
