"""Log binary model comparisons and selected artifacts to MLflow."""

import importlib.metadata
import json
import shutil
from pathlib import Path

import joblib
import mlflow
import mlflow.onnx
import mlflow.sklearn
import numpy as np
import onnx
import pandas as pd
from mlflow import MlflowClient

from training.evaluation import evaluate_models as evaluate
from training.evaluation import select_best
from training.onnx_export import verify


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def evaluate_models(
    models,
    X_train,
    y_train,
    X_test,
    y_test,
    *,
    output,
    cv,
    groups=None,
    scoring="average_precision",
    n_jobs=1,
):
    """Preserve exact fit inputs, folds, searches, predictions and fitted models."""
    parent = mlflow.active_run()
    if parent is None:
        raise ValueError("Start an MLflow run before evaluating models")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    kwargs = dict(cv=cv, groups=groups, scoring=scoring, n_jobs=n_jobs)
    args = (list(models), X_train, y_train, X_test, y_test)
    joblib.dump({"args": args, "kwargs": kwargs}, output / "fit_inputs.joblib")
    results = evaluate(*args, **kwargs)
    best = select_best(results)
    selected_run = None
    selected_onnx = None
    for index, result in enumerate(results):
        directory = output / f"model-{index}"
        directory.mkdir()
        joblib.dump(result.search, directory / "search.joblib")
        np.savez(
            directory / "predictions.npz",
            predictions=result.predictions,
            probabilities=result.probabilities,
            confusion_matrix=result.confusion_matrix,
            labels=result.labels,
        )
        np.savez(directory / "curves.npz", **result.curves)
        (directory / "classification_report.txt").write_text(result.report)
        pd.DataFrame(result.search.cv_results_).to_csv(
            directory / "cv_results.csv", index=False
        )
        with mlflow.start_run(
            run_name=result.name,
            nested=True,
            experiment_id=parent.info.experiment_id,
            tags={
                **{
                    key: value
                    for key, value in parent.data.tags.items()
                    if not key.startswith("mlflow.")
                },
                "selected": str(result is best).lower(),
            },
        ) as trial:
            client = MlflowClient()
            client.log_inputs(
                trial.info.run_id,
                client.get_run(parent.info.run_id).inputs.dataset_inputs,
            )
            mlflow.log_params(result.search.best_estimator_.get_params(deep=True))
            mlflow.log_params(
                {
                    f"best.{key}": value
                    for key, value in result.search.best_params_.items()
                }
            )
            mlflow.log_metrics(result.metrics)
            mlflow.log_metric("cv_best_score", result.search.best_score_)
            mlflow.log_metrics(
                {
                    key: float(values[result.search.best_index_])
                    for key, values in result.search.cv_results_.items()
                    if key.startswith(
                        (
                            "mean_test_",
                            "std_test_",
                            "mean_fit_time",
                            "mean_score_time",
                        )
                    )
                }
            )
            fitted = result.search.best_estimator_
            if result.onnx_exporter is not None:
                graph = result.onnx_exporter(fitted, X_train)
                positive_class = np.flatnonzero(result.search.classes_ == 1).item()
                features = np.concatenate([np.asarray(X_train), np.asarray(X_test)])
                probabilities = np.concatenate(
                    [
                        fitted.predict_proba(X_train)[:, positive_class],
                        result.probabilities,
                    ]
                )
                validation = verify(graph, features, probabilities)
                write_json(directory / "onnx_validation.json", validation)
                onnx.save_model(graph, directory / "model.onnx")
                mlflow.log_metric(
                    "onnx_max_absolute_error", validation["max_absolute_error"]
                )
                model = mlflow.onnx.log_model(
                    graph,
                    name="model",
                    save_as_external_data=False,
                    onnx_execution_providers=["CPUExecutionProvider"],
                    onnx_session_options={
                        "intra_op_num_threads": 1,
                        "inter_op_num_threads": 1,
                    },
                    pip_requirements=[
                        f"{name}=={importlib.metadata.version(name)}"
                        for name in ("mlflow", "onnx", "onnxruntime", "numpy")
                    ],
                )
            else:
                model = mlflow.sklearn.log_model(
                    fitted,
                    name="model",
                    pip_requirements=[
                        f"{name}=={importlib.metadata.version(name)}"
                        for name in (
                            "mlflow",
                            "scikit-learn",
                            "numpy",
                            "scipy",
                            "skops",
                        )
                    ],
                )
            mlflow.log_artifacts(directory, "evaluation")
            if result is best:
                selected_run = trial.info.run_id
                model_uri = model.model_uri
                if result.onnx_exporter is not None:
                    selected_onnx = directory
    selected = output / "selected"
    selected.mkdir()
    joblib.dump(best.search.best_estimator_, selected / "model.joblib")
    joblib.dump(best.search.best_params_, selected / "hyperparameters.joblib")
    if selected_onnx is not None:
        for name in ("model.onnx", "onnx_validation.json"):
            shutil.copyfile(selected_onnx / name, selected / name)
    selection = {
        "name": best.name,
        "run_id": selected_run,
        "model_uri": model_uri,
        "model_file": "model.onnx" if selected_onnx is not None else "model.joblib",
        "selection_metric": str(best.search.scoring["selection"]),
        "validation_score": best.search.best_score_,
        "tie_rule": "first model in caller order",
        "parameters": MlflowClient().get_run(selected_run).data.params,
    }
    write_json(selected / "selection.json", selection)
    mlflow.set_tags({"selected_model": best.name, "selected_model_run": selected_run})
    mlflow.log_artifacts(selected, "selected")
    return results
