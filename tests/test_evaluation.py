import numpy as np
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.tree import DecisionTreeClassifier

from training.evaluation import ModelSpec, evaluate_models, select_best


@pytest.fixture
def examples():
    labels = np.tile([0, 1], 20)
    groups = np.repeat(np.arange(20), 2)
    features = np.column_stack([labels * 4 - 2, groups])
    test_features = np.array([[-3, 21], [3, 21], [-1, 22], [1, 22]])
    return features, labels, test_features, np.array([0, 1, 0, 1]), groups


def test_models_share_grouped_folds_and_report_held_out_predictions(examples):
    X_train, y_train, X_test, y_test, groups = examples
    models = [
        ModelSpec(
            "tree", DecisionTreeClassifier(random_state=42), {"max_depth": [1, 2]}
        ),
        ModelSpec("logistic", LogisticRegression(), {"C": [0.1, 1]}),
    ]
    results = evaluate_models(
        models,
        X_train,
        y_train,
        X_test,
        y_test,
        cv=GroupKFold(4),
        groups=groups,
    )
    for result in results:
        np.testing.assert_array_equal(result.predictions, y_test)
        np.testing.assert_array_equal(result.confusion_matrix, [[2, 0], [0, 2]])
        assert result.metrics["average_precision"] == 1
        assert result.metrics["roc_auc"] == 1
        assert result.search.best_params_
        assert len(result.search.cv_results_["params"]) == 2
        for train, validation in result.search.cv:
            assert set(groups[train]).isdisjoint(groups[validation])
    for left, right in zip(results[0].search.cv, results[1].search.cv, strict=True):
        for left_indices, right_indices in zip(left, right, strict=True):
            np.testing.assert_array_equal(left_indices, right_indices)
    for model in models:
        assert not hasattr(model.estimator, "classes_")


def test_grid_fit_errors_are_not_suppressed(examples):
    X_train, y_train, X_test, y_test, _ = examples
    model = ModelSpec("invalid", DecisionTreeClassifier(), {"max_depth": [0]})
    with pytest.raises(ValueError):
        evaluate_models([model], X_train, y_train, X_test, y_test, cv=2)


def test_undefined_precision_is_explicit(examples):
    X_train, y_train, X_test, y_test, _ = examples
    model = ModelSpec("constant", DummyClassifier(strategy="constant", constant=0))
    result = evaluate_models([model], X_train, y_train, X_test, y_test, cv=2)[0]
    assert np.isnan(result.metrics["precision"])
    assert result.metrics["recall"] == 0


def test_single_class_validation_fails_before_training(examples):
    X_train, y_train, X_test, y_test, _ = examples
    model = ModelSpec("tree", DecisionTreeClassifier())
    folds = [(np.arange(2, 40), np.array([0]))]
    with pytest.raises(ValueError, match="Every training and validation fold"):
        evaluate_models([model], X_train, y_train, X_test, y_test, cv=folds)


def test_selection_uses_validation_scores_and_not_held_out_performance(examples):
    X_train, y_train, X_test, y_test, _ = examples
    results = evaluate_models(
        [
            ModelSpec("tree", DecisionTreeClassifier(random_state=42)),
            ModelSpec("constant", DummyClassifier(strategy="prior")),
        ],
        X_train,
        y_train,
        X_test,
        1 - y_test,
        cv=2,
        scoring="roc_auc",
    )
    assert results[0].metrics["roc_auc"] < results[1].metrics["roc_auc"]
    assert select_best(results).name == "tree"
    assert "mean_test_neg_brier_score" in results[0].search.cv_results_
    np.testing.assert_array_equal(results[0].labels, 1 - y_test)
    assert results[0].curves["pr_thresholds"].size > 0
