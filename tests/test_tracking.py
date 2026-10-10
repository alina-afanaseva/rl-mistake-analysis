import json

import joblib
import mlflow
import mlflow.onnx
import mlflow.sklearn
import numpy as np
import pytest
from catboost import CatBoostClassifier
from mlflow import MlflowClient
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

from training.evaluation import ModelSpec
from training.onnx_export import catboost, verify
from training.tracking import evaluate_models


@pytest.fixture
def tracking(tmp_path, monkeypatch):
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    previous = mlflow.get_tracking_uri()
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    mlflow.set_tracking_uri(uri)
    experiment = mlflow.create_experiment(
        "model-comparison", artifact_location=(tmp_path / "artifacts").as_uri()
    )
    try:
        yield experiment
    finally:
        mlflow.set_tracking_uri(previous)


@pytest.fixture
def examples():
    labels = np.tile([0, 1], 20)
    groups = np.repeat(np.arange(20), 2)
    features = np.column_stack([labels * 4 - 2, groups]).astype(np.float32)
    return features[:32], labels[:32], features[32:], labels[32:], groups[:32]


def test_tracking_requires_an_active_run(tmp_path, examples):
    X_train, y_train, X_test, y_test, groups = examples
    output = tmp_path / "evaluation"
    with pytest.raises(ValueError, match="Start an MLflow run"):
        evaluate_models(
            [ModelSpec("logistic", LogisticRegression())],
            X_train,
            y_train,
            X_test,
            y_test,
            output=output,
            cv=GroupKFold(2),
            groups=groups,
        )
    assert not output.exists()


def test_tracking_preserves_metrics_searches_and_selected_parameters(
    tmp_path, tracking, examples
):
    X_train, y_train, X_test, y_test, groups = examples
    output = tmp_path / "evaluation"
    models = [
        ModelSpec("constant", DummyClassifier(strategy="prior")),
        ModelSpec("logistic", LogisticRegression(), {"C": [0.1, 1]}),
    ]
    with mlflow.start_run(
        experiment_id=tracking, tags={"dataset_version": "v1"}
    ) as run:
        mlflow.log_input(
            mlflow.data.from_numpy(X_train, targets=y_train), context="training"
        )
        results = evaluate_models(
            models,
            X_train,
            y_train,
            X_test,
            1 - y_test,
            output=output,
            cv=GroupKFold(2),
            groups=groups,
            scoring="roc_auc",
        )
        run_id = run.info.run_id
    client = MlflowClient()
    children = client.search_runs([tracking], f"tags.mlflow.parentRunId = '{run_id}'")
    dataset = client.get_run(run_id).inputs.dataset_inputs[0].dataset
    assert len(children) == 2
    assert all(child.info.status == "FINISHED" for child in children)
    for child in children:
        result = next(r for r in results if r.name == child.data.tags["mlflow.runName"])
        assert set(result.metrics) <= child.data.metrics.keys()
        assert child.data.metrics["cv_best_score"] == result.search.best_score_
        assert "mean_test_neg_log_loss" in child.data.metrics
        assert "std_test_average_precision" in child.data.metrics
        assert child.data.tags["dataset_version"] == "v1"
        assert child.inputs.dataset_inputs[0].dataset.digest == dataset.digest
    selection = json.loads((output / "selected/selection.json").read_text())
    selected = client.get_run(selection["run_id"])
    assert selection["name"] == "logistic"
    assert selection["selection_metric"] == "roc_auc"
    assert selected.data.tags["selected"] == "true"
    assert selected.data.metrics["roc_auc"] == 0
    assert selected.data.params["best.C"]
    search = joblib.load(output / "model-1/search.joblib")
    assert len(search.cv_results_["params"]) == 2
    assert len(search.cv) == 2
    assert (
        joblib.load(output / "selected/hyperparameters.joblib") == search.best_params_
    )
    np.testing.assert_array_equal(
        np.load(output / "model-1/predictions.npz")["labels"], 1 - y_test
    )
    assert np.load(output / "model-1/curves.npz")["pr_thresholds"].size > 0
    assert joblib.load(output / "fit_inputs.joblib")["kwargs"]["scoring"] == "roc_auc"
    model = mlflow.sklearn.load_model(selection["model_uri"])
    np.testing.assert_allclose(
        model.predict_proba(X_test)[:, 1], results[1].probabilities
    )
    assert client.list_artifacts(run_id, "selected")


def incorrect_export(model, features):
    graph = catboost(model, features)
    graph.graph.initializer[0].int64_data[0] = 0
    return graph


@pytest.mark.parametrize("invalid", [False, True])
def test_tracking_verifies_onnx_before_saving_a_selected_model(
    tmp_path, tracking, examples, invalid
):
    X_train, y_train, X_test, y_test, groups = examples
    output = tmp_path / "evaluation"
    model = ModelSpec(
        "catboost",
        CatBoostClassifier(
            iterations=5,
            depth=2,
            verbose=False,
            thread_count=1,
            allow_writing_files=False,
        ),
        {"learning_rate": [0.1, 0.2]},
        onnx_exporter=incorrect_export if invalid else catboost,
    )

    def evaluate():
        with mlflow.start_run(experiment_id=tracking):
            return evaluate_models(
                [model],
                X_train,
                y_train,
                X_test,
                y_test,
                output=output,
                cv=GroupKFold(2),
                groups=groups,
            )

    if invalid:
        with pytest.raises(AssertionError, match="Not equal to tolerance"):
            evaluate()
        assert not (output / "selected").exists()
        assert all(
            run.info.status == "FAILED"
            for run in MlflowClient().search_runs([tracking])
        )
    else:
        result = evaluate()[0]
        selected = output / "selected"
        selection = json.loads((selected / "selection.json").read_text())
        assert selection["model_file"] == "model.onnx"
        graph = mlflow.onnx.load_model(selection["model_uri"])
        verify(graph, X_test, result.probabilities)
        validation = json.loads((selected / "onnx_validation.json").read_text())
        assert validation["rows"] == 40
        assert validation["execution_provider"] == "CPUExecutionProvider"
        assert (selected / "model.onnx").is_file()
