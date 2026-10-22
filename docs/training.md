# Training and model releases

## Prepare data for training

Work from a checkout of this repository. The Python modules in `training/` download recordings, read prepared feature tables, and generate release metadata. Use them from your model-training scripts.

A consumer exports reviewed incidents using the [annotation contract](formats.md#human-annotations). Each review carries a label and refers to a recording with a download ID and checksum. The ID locates the recording; the checksum verifies its bytes.

With the [development environment](../README.md#develop-and-contribute) activated, build the preparation tool:

```sh
cargo build --locked -p rl-mistake-analysis-cli
```

Validate reviews and retrieve recordings:

```sh
target/debug/rl-mistake-analysis validate-annotations data/annotations.json
uv run --locked python -m training.fetch_replays data/annotations.json data/replays
```

Ballchasing downloads require `BALLCHASING_TOKEN`. Use the extraction and annotation-join commands in the [mistake guide](../README.md#implemented-mistakes) to prepare a labeled feature table. Preparation matches reviews to incidents and recalculates features using the candidate library. Every selected review must match before saving the table.

## Dataset inputs

Read the resulting Parquet table in Python:

```python
from training.dataset import arrays, load

table, contract = load("artifacts/training.parquet")
X, y, replay_groups = arrays(table, contract)
```

`X` contains features, `y` contains labels, and `replay_groups` identifies recordings for grouped training and evaluation splits. Training reads the prepared table. Downloaded recordings in `data/` and generated tables in `artifacts/` are ignored by Git.

Model inputs are float32 values in contract order. Boolean measurements become `0` and `1`; unavailable measurements become `NaN`. `training.dataset.select_features(contract, names, input_schema)` selects ordered inputs from the source feature definitions and preserves their frame selection settings.

## Compare your own models

`training.evaluation.evaluate_models` runs the same grid search, refit, prediction, and metrics for each `ModelSpec`. Supply training and held-out test data and a cross-validation strategy. Features pass directly to each estimator. Binary labels use `1` for a mistake and `0` for a non-mistake.

```python
from sklearn.model_selection import StratifiedGroupKFold

from training.evaluation import ModelSpec, evaluate_models

models = [ModelSpec("my_model", estimator, parameter_grid)]
results = evaluate_models(
    models,
    X_train,
    y_train,
    X_test,
    y_test,
    cv=StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42),
    groups=replay_groups_train,
)

for result in results:
    print(result.name, result.search.best_params_, result.metrics)
    fitted_model = result.search.best_estimator_
```

Add models with `ModelSpec(name, estimator, param_grid, fit_params)`. Supply the estimator and parameter grid from your own model definition. Estimators must support the scikit-learn classifier interface, including `predict_proba`; learned preprocessing can be included in an estimator's `Pipeline`. For tracked ONNX output, declare an `onnx_exporter` accepting the fitted estimator and training features and returning an ONNX `ModelProto`. The exported graph must include all fitted preprocessing.

Every model uses the same validation folds. Grid search selects parameters by average precision by default; `scoring` and `n_jobs` are configurable. `select_best(results)` chooses the largest validation score across models; exact ties preserve the caller's model order. Held-out test scores measure final performance. Scores for losses use scikit-learn's negative convention, so `scoring="neg_log_loss"` selects the smallest log loss.

Each parameter combination records validation accuracy, balanced accuracy, precision, recall, F1, F0.5, F2, MCC, ROC AUC, average precision, log loss, and Brier score. Results also retain held-out labels, probabilities, predictions, confusion counts, specificity, error rates, class prevalence, classification reports, and complete precision-recall and ROC curves. Saved labels and probabilities support further metrics and threshold analyses. Precision is `NaN` when undefined. Failed fits raise errors.

Keep all examples from a replay together when creating the train/test partition, as well as within cross-validation. Estimator fit parameters use training data. Every training, validation, and test partition must contain both classes for the reported metrics.

Run `nix-shell` to activate the training environment. Nix supplies scikit-learn and the native ONNX Runtime library. The shell installs CatBoost, XGBoost, the ONNX converters, and Python ONNX Runtime in `.venv`. The `training` dependency group also supports environments managed entirely by uv: `uv sync --locked --group training`.

## Save evaluation results

`training.tracking.evaluate_models` accepts the same models, data, and evaluation options, plus an `output` directory. Call it inside an active MLflow run:

```python
import mlflow

from training.tracking import evaluate_models

with mlflow.start_run():
    results = evaluate_models(
        models,
        X_train,
        y_train,
        X_test,
        y_test,
        output="artifacts/evaluation",
        cv=StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42),
        groups=replay_groups_train,
    )
```

Each model gets a child MLflow run with its metrics and hyperparameters. The output directory retains fit inputs, validation folds, complete parameter searches, labels, predictions, and metric curves. The selected model is saved under `selected/` with its exact hyperparameters and a selection record linking its MLflow run. Choose a different output directory for each evaluation. Configure MLflow's tracking server and experiment through its normal API or environment variables. The shell installs the official MLflow package in `.venv`; uv environments can install it with `uv sync --locked --group training --group tracking`.

Declare `onnx_exporter=training.onnx_export.catboost` in a numeric CatBoost model definition to export it during tracking. The exporter produces a dynamic float32 `[N, F]` input and a single `[N, 1]` mistake-probability output. CPU ONNX Runtime checks the graph against native probabilities on all training and test rows, including missing values. Tracking logs the ONNX model in MLflow and stores `model.onnx` and `onnx_validation.json` with the selected hyperparameters. Failed conversion or verification raises an error. MLflow's scikit-learn serializer retains fitted Python models and searches. CatBoost's [ONNX exporter](https://catboost.ai/docs/en/concepts/apply-onnx-ml) supports numerical features only.

Average precision is the default selection criterion because it measures mistake ranking across thresholds. Log loss and Brier score are also recorded; choose `neg_log_loss` when probability accuracy is the selection objective. See scikit-learn's [metric guidance](https://scikit-learn.org/stable/modules/model_evaluation.html) and [average precision definition](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html).

## Release models

Code and models have independent versions. One `manifest.json` catalog can list several releases of each mistake kind. Each entry describes its feature contract, threshold, evaluation report, and artifact checksums.

`training.export.model_entry` generates metadata from an exported ONNX model, a prepared table, and an evaluation report. `training.export.write_manifest` writes the catalog. Publish the catalog and artifacts to a Hugging Face model repository; consumers download the files for selected releases.
