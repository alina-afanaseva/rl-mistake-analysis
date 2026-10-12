"""Tune binary classifiers and compare their held-out predictions."""

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    fbeta_score,
    log_loss,
    make_scorer,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, check_cv

SCORERS = {
    "accuracy": "accuracy",
    "balanced_accuracy": "balanced_accuracy",
    "precision": make_scorer(precision_score, zero_division=np.nan),
    "recall": "recall",
    "f1": "f1",
    "f0_5": make_scorer(fbeta_score, beta=0.5),
    "f2": make_scorer(fbeta_score, beta=2),
    "mcc": "matthews_corrcoef",
    "roc_auc": "roc_auc",
    "average_precision": "average_precision",
    "neg_log_loss": "neg_log_loss",
    "neg_brier_score": "neg_brier_score",
}


@dataclass
class ModelSpec:
    name: str
    estimator: Any
    param_grid: dict | list[dict] = field(default_factory=dict)
    fit_params: dict = field(default_factory=dict)
    onnx_exporter: Callable | None = None


@dataclass
class Evaluation:
    name: str
    search: GridSearchCV
    metrics: dict[str, float]
    predictions: np.ndarray
    probabilities: np.ndarray
    confusion_matrix: np.ndarray
    labels: np.ndarray
    curves: dict[str, np.ndarray]
    report: str
    onnx_exporter: Callable | None = None


def select_best(results):
    """Choose the largest validation score; exact ties retain the caller's order."""
    results = list(results)
    if not results or any(
        not np.isfinite(result.search.best_score_) for result in results
    ):
        raise ValueError("Model selection requires finite validation scores")
    selection = results[0].search.scoring["selection"]
    if any(result.search.scoring["selection"] != selection for result in results):
        raise ValueError("Models must use the same selection metric")
    return max(results, key=lambda result: result.search.best_score_)


def evaluate_models(
    models,
    X_train,
    y_train,
    X_test,
    y_test,
    *,
    cv,
    groups=None,
    scoring="average_precision",
    n_jobs=1,
):
    """Search training folds, refit, and evaluate each model on the same test set.

    Labels must be 0 and 1, with both classes present in each partition.
    Feature inputs pass directly to the estimators. Fit parameters belong to
    each model; learned preprocessing belongs inside its estimator pipeline.
    """
    models = list(models)
    names = [model.name for model in models]
    if not models or any(not name.strip() for name in names):
        raise ValueError("Provide at least one named model")
    if len(set(names)) != len(names):
        raise ValueError("Model names must be unique")
    for labels in (y_train, y_test):
        if not np.array_equal(np.unique(labels), [0, 1]):
            raise ValueError("Training and test labels must each contain both 0 and 1")

    splitter = check_cv(cv, y=y_train, classifier=True)
    folds = list(splitter.split(X_train, y_train, groups))
    labels = np.asarray(y_train)
    for train, validation in folds:
        for indices in (train, validation):
            if not np.array_equal(np.unique(labels[indices]), [0, 1]):
                raise ValueError(
                    "Every training and validation fold needs both classes"
                )

    results = []
    for model in models:
        search = GridSearchCV(
            model.estimator,
            model.param_grid,
            scoring={**SCORERS, "selection": scoring},
            cv=folds,
            n_jobs=n_jobs,
            refit="selection",
            error_score="raise",
        )
        search.fit(X_train, y_train, **model.fit_params)
        predictions = np.asarray(search.predict(X_test)).reshape(-1)
        positive_class = np.flatnonzero(search.classes_ == 1).item()
        probabilities = search.predict_proba(X_test)[:, positive_class]
        metrics = {
            "accuracy": accuracy_score(y_test, predictions),
            "balanced_accuracy": balanced_accuracy_score(y_test, predictions),
            "precision": precision_score(y_test, predictions, zero_division=np.nan),
            "recall": recall_score(y_test, predictions),
            "f1": f1_score(y_test, predictions),
            "f0_5": fbeta_score(y_test, predictions, beta=0.5),
            "f2": fbeta_score(y_test, predictions, beta=2),
            "mcc": matthews_corrcoef(y_test, predictions),
            "roc_auc": roc_auc_score(y_test, probabilities),
            "average_precision": average_precision_score(y_test, probabilities),
            "log_loss": log_loss(y_test, probabilities, labels=[0, 1]),
            "brier_score": brier_score_loss(y_test, probabilities),
        }
        matrix = confusion_matrix(y_test, predictions, labels=[0, 1])
        tn, fp, fn, tp = map(int, matrix.ravel())
        metrics.update(
            tn=tn,
            fp=fp,
            fn=fn,
            tp=tp,
            specificity=tn / (tn + fp),
            false_positive_rate=fp / (tn + fp),
            false_negative_rate=fn / (fn + tp),
            negative_predictive_value=tn / (tn + fn) if tn + fn else np.nan,
            test_count=len(predictions),
            positive_prevalence=(tp + fn) / len(predictions),
            flagged_fraction=(tp + fp) / len(predictions),
        )
        precision, recall, pr_thresholds = precision_recall_curve(y_test, probabilities)
        fpr, tpr, roc_thresholds = roc_curve(
            y_test, probabilities, drop_intermediate=False
        )
        results.append(
            Evaluation(
                name=model.name,
                search=search,
                metrics=metrics,
                predictions=predictions,
                probabilities=probabilities,
                confusion_matrix=matrix,
                labels=np.asarray(y_test).reshape(-1),
                curves={
                    "precision": precision,
                    "recall": recall,
                    "pr_thresholds": pr_thresholds,
                    "fpr": fpr,
                    "tpr": tpr,
                    "roc_thresholds": roc_thresholds,
                },
                report=classification_report(y_test, predictions, zero_division=np.nan),
                onnx_exporter=model.onnx_exporter,
            )
        )
    return results
