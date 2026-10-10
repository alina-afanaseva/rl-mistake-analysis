import joblib
import numpy as np
import pytest
from catboost import CatBoostClassifier

from training.onnx_export import catboost, verify


@pytest.mark.parametrize("nan_mode", ["Min", "Max"])
def test_numeric_export_matches_native_probabilities_with_missing_values(
    nan_mode, tmp_path
):
    values = np.tile([-2, -1, 1, 2, np.nan], 8)
    features = np.repeat(values[:, None], 105, axis=1).astype(np.float32)
    labels = ((values > 0) | np.isnan(values)).astype(int)
    model = CatBoostClassifier(
        iterations=20,
        depth=2,
        nan_mode=nan_mode,
        thread_count=1,
        verbose=False,
        allow_writing_files=False,
    ).fit(features, labels)
    graph = catboost(model, features)
    probabilities = model.predict_proba(features)[:, 1]
    for rows in (1, len(features)):
        validation = verify(graph, features[:rows], probabilities[:rows])
        assert validation["features"] == 105
        assert validation["rows"] == rows
        assert validation["max_absolute_error"] < 1e-6
    with pytest.raises(AssertionError):
        verify(graph, features, 1 - probabilities)
    joblib.dump(model, tmp_path / "model.joblib")
    restored = joblib.load(tmp_path / "model.joblib")
    np.testing.assert_allclose(restored.predict_proba(features)[:, 1], probabilities)
    verify(catboost(restored, features), features, probabilities)


def test_rejects_native_categorical_features():
    features = [["a"], ["b"], ["a"], ["b"]]
    model = CatBoostClassifier(
        iterations=2, thread_count=1, verbose=False, allow_writing_files=False
    ).fit(features, [0, 1, 0, 1], cat_features=[0])
    with pytest.raises(ValueError, match="numeric features only"):
        catboost(model, features)


def test_rejects_incompatible_feature_width_and_labels():
    features = np.arange(8, dtype=np.float32).reshape(-1, 1)
    model = CatBoostClassifier(
        iterations=2, thread_count=1, verbose=False, allow_writing_files=False
    ).fit(features, [0, 1] * 4)
    with pytest.raises(ValueError, match="feature width"):
        catboost(model, np.zeros((8, 2)))
    model.fit(features, [1, 2] * 4)
    with pytest.raises(ValueError, match="binary labels"):
        catboost(model, features)
