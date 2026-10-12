"""Train the selected teammate-bump classifier and export a verified model release."""

import argparse
import hashlib
import json
from importlib.metadata import version
from pathlib import Path

import numpy as np
import onnx
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from xgboost import XGBClassifier

from training.dataset import arrays, load, select_features
from training.export import model_entry, write_manifest
from training.onnx_export import verify, xgboost

CONFIG = Path(__file__).parent / "configs/bumping_teammate_0.1.0.json"


def partitions(labels, groups, test_replays):
    """Reserve the explicitly supplied replays without moving rows between groups."""
    if (
        not isinstance(test_replays, list)
        or not test_replays
        or any(not isinstance(replay, str) for replay in test_replays)
        or len(set(test_replays)) != len(test_replays)
        or set(test_replays) - set(groups)
    ):
        raise ValueError("Test replays must be unique, known replay checksums")
    mask = np.isin(groups, test_replays)
    train, test = np.flatnonzero(~mask), np.flatnonzero(mask)
    for rows in (train, test):
        if not np.array_equal(np.unique(labels[rows]), [0, 1]):
            raise ValueError(
                "Training and test partitions must each contain both classes"
            )
    return train, test


def metrics(labels, probabilities, threshold):
    predictions = probabilities >= threshold
    values = {
        "precision": precision_score(labels, predictions, zero_division=np.nan),
        "recall": recall_score(labels, predictions),
        "f1": f1_score(labels, predictions),
        "roc_auc": roc_auc_score(labels, probabilities),
        "average_precision": average_precision_score(labels, probabilities),
        "log_loss": log_loss(labels, probabilities, labels=[0, 1]),
        "brier_score": brier_score_loss(labels, probabilities),
    }
    return {
        **{
            name: float(value) if np.isfinite(value) else None
            for name, value in values.items()
        },
        "confusion_matrix": confusion_matrix(
            labels, predictions, labels=[0, 1]
        ).tolist(),
    }


def train(training_table, test_replays, output, *, source_revision, config=CONFIG):
    """Fit one fixed configuration, evaluate complete held-out replays, and verify ONNX."""
    settings = json.loads(Path(config).read_text())
    if settings["kind"] != "bumping_teammate" or settings["model"] != "xgboost":
        raise ValueError("This recipe trains the selected teammate-bump XGBoost model")
    release_version = settings["version"]
    if (
        not isinstance(release_version, str)
        or not release_version.strip()
        or release_version in (".", "..")
        or "/" in release_version
        or "\\" in release_version
        or settings["status"] not in ("experimental", "approved")
    ):
        raise ValueError("Provide a valid model release version and status")
    threshold = settings["keep_threshold"]
    if (
        type(threshold) not in (int, float)
        or not np.isfinite(threshold)
        or not 0 <= threshold <= 1
        or not source_revision.strip()
    ):
        raise ValueError("Provide a valid threshold and training source revision")
    output = Path(output)
    if output.exists():
        raise FileExistsError("Use a new output directory for each training run")
    table, source_contract = load(training_table)
    if source_contract["kind"] != settings["kind"]:
        raise ValueError("Training data and recipe mistake kinds differ")
    contract = select_features(
        source_contract, settings["feature_names"], settings["input_schema"]
    )
    if contract["sampling"] != settings["sampling"]:
        raise ValueError("Training data and recipe sampling methods differ")
    features, labels, groups = arrays(table, contract)
    train_rows, test_rows = partitions(
        labels, groups, json.loads(Path(test_replays).read_text())
    )
    classifier = XGBClassifier(**settings["hyperparameters"])
    classifier.fit(features[train_rows], labels[train_rows])
    probabilities = classifier.predict_proba(features)[:, 1]
    graph = xgboost(classifier, features)
    validation = verify(graph, features, probabilities)
    with Path(training_table).open("rb") as stream:
        dataset_hash = hashlib.file_digest(stream, "sha256").hexdigest()
    report = {
        "schema_version": 1,
        "kind": settings["kind"],
        "version": settings["version"],
        "model": settings["model"],
        "hyperparameters": settings["hyperparameters"],
        "class_weighting": "none",
        "decision_threshold": threshold,
        "decision_rule": "positive probability >= decision_threshold",
        "feature_contract": {
            key: contract[key]
            for key in (
                "schema_version",
                "kind",
                "input_schema",
                "sampling",
                "feature_names",
                "feature_definitions",
            )
        },
        "dataset": {
            "training_parquet_sha256": dataset_hash,
            "rows": len(labels),
            "replays": len(set(groups)),
            "train_rows": len(train_rows),
            "test_rows": len(test_rows),
            "train_replays": len(set(groups[train_rows])),
            "test_replays": len(set(groups[test_rows])),
        },
        "partition_metrics": {
            "train": metrics(labels[train_rows], probabilities[train_rows], threshold),
            "test": metrics(labels[test_rows], probabilities[test_rows], threshold),
        },
        "onnx_validation": validation,
        "source_revision": source_revision,
        "packages": {
            name: version(name)
            for name in (
                "numpy",
                "xgboost",
                "scikit-learn",
                "onnx",
                "onnxmltools",
                "onnxruntime",
            )
        },
    }
    relative = Path("models") / settings["kind"] / settings["version"]
    model_directory = output / relative
    model_directory.mkdir(parents=True, exist_ok=False)
    onnx.save(graph, model_directory / "model.onnx")
    (model_directory / "evaluation.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    classifier.save_model(output / "xgboost.json")
    (output / "splits.json").write_text(
        json.dumps({"train": train_rows.tolist(), "test": test_rows.tolist()}, indent=2)
        + "\n"
    )
    np.savez_compressed(
        output / "predictions.npz", labels=labels, probabilities=probabilities
    )
    entry = model_entry(
        output,
        training_table,
        version=settings["version"],
        model_file=(relative / "model.onnx").as_posix(),
        evaluation_file=(relative / "evaluation.json").as_posix(),
        keep_threshold=threshold,
        source_revision=source_revision,
        status=settings["status"],
        feature_names=settings["feature_names"],
        input_schema=settings["input_schema"],
    )
    write_manifest(output, settings["version"], [entry])
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("training_table", type=Path)
    parser.add_argument("test_replays", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--config", type=Path, default=CONFIG)
    args = parser.parse_args()
    report = train(
        args.training_table,
        args.test_replays,
        args.output,
        source_revision=args.source_revision,
        config=args.config,
    )
    print(json.dumps(report["partition_metrics"], allow_nan=False))


if __name__ == "__main__":
    main()
