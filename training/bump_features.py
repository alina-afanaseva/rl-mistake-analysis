"""Compute subject-specific bump measurements from combined replay analyses."""

import json
from pathlib import Path

from training.context_features import compute as compute_context
from training.context_features import main


def compute(paths, engine, sampling, output):
    return compute_context(
        paths,
        engine,
        output,
        kind="bumping_teammate",
        command=["bump-features", str(sampling)],
        expected_sampling=json.loads(Path(sampling).read_text()),
        participant_fields=("teammate_id", "initiator", "victim"),
    )


if __name__ == "__main__":
    main(compute, __doc__)
