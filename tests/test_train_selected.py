"""Verify fixed training, replay isolation, and subset scoring through the CLI."""

import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from test_participant_pipeline import SAMPLING, analysis, run
from xgboost import XGBClassifier

from training.bump_features import compute
from training.dataset import arrays, load, select_features
from training.train_selected import CONFIG, partitions, train


def test_selected_recipe_exports_model_scored_by_the_cli(engine, tmp_path):
    document = analysis()
    source = tmp_path / "analysis.json"
    source.write_text(json.dumps(document))
    feature_path = tmp_path / "features.parquet"
    compute([source], engine, SAMPLING, feature_path)
    features = pq.read_table(feature_path)
    tables = []
    for replay in range(24):
        table = features.set_column(
            features.schema.get_field_index("replay_sha256"),
            "replay_sha256",
            pa.array([f"{replay:064x}"] * 2),
        )
        tables.append(table.append_column("label", pa.array([0, 1], type=pa.uint8())))
    training_path = tmp_path / "training.parquet"
    pq.write_table(pa.concat_tables(tables), training_path)
    test_replays = tmp_path / "test_replays.json"
    test_replays.write_text(json.dumps([f"{replay:064x}" for replay in range(20, 24)]))
    output = tmp_path / "release"
    report = train(training_path, test_replays, output, source_revision="training-test")
    settings = json.loads(CONFIG.read_text())
    table, contract = load(training_path)
    selected = select_features(
        contract, settings["feature_names"], settings["input_schema"]
    )
    values, labels, groups = arrays(table, selected)
    native = XGBClassifier()
    native.load_model(output / "xgboost.json")
    saved = np.load(output / "predictions.npz")
    np.testing.assert_array_equal(saved["labels"], labels)
    np.testing.assert_allclose(
        saved["probabilities"], native.predict_proba(values)[:, 1]
    )
    splits = json.loads((output / "splits.json").read_text())
    assert not set(groups[splits["train"]]) & set(groups[splits["test"]])
    assert len(splits["train"]) == 40 and len(splits["test"]) == 8
    assert report["feature_contract"]["feature_names"] == settings["feature_names"]
    assert report["onnx_validation"]["rows"] == len(labels)
    assert report["onnx_validation"]["features"] == len(settings["feature_names"])
    predictions = run(
        engine,
        "score",
        output,
        settings["kind"],
        settings["version"],
        document=document,
    )
    assert len(predictions["predictions"]) == 2
    by_player = {
        json.loads(player)["Steam"]: score
        for player, score in zip(
            table["player_id"].to_pylist()[:2], saved["probabilities"][:2], strict=True
        )
    }
    for prediction in predictions["predictions"]:
        expected = by_player[prediction["candidate"]["player_id"]["Steam"]]
        assert prediction["score"] == pytest.approx(expected, abs=1e-6)
        assert prediction["keep"] == bool(expected >= settings["keep_threshold"])
    with pytest.raises(FileExistsError):
        train(training_path, test_replays, output, source_revision="training-test")


@pytest.mark.parametrize(
    "replays", [[], ["missing"], ["train", "train"], ["train", "test"]]
)
def test_invalid_holdout_never_creates_a_new_split(replays):
    with pytest.raises(ValueError):
        partitions(
            np.array([0, 1, 0, 1]),
            np.array(["train", "train", "test", "test"]),
            replays,
        )


@pytest.mark.parametrize(
    "names", [[], ["unknown"], ["p_is_initiator", "p_is_initiator"]]
)
def test_invalid_subset_is_an_error(names):
    contract = {
        "input_schema": "bumping_teammate.context.v1",
        "feature_definitions": [{"name": "p_is_initiator"}],
    }
    with pytest.raises(ValueError):
        select_features(contract, names, contract["input_schema"])
