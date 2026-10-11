"""Export binary classifiers and verify their probability tensor with ONNX Runtime."""

import tempfile
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper


def catboost(model, example):
    """Export a numeric CatBoost classifier with one positive-class probability output."""
    if not np.array_equal(model.classes_, [0, 1]):
        raise ValueError("CatBoost ONNX export requires binary labels 0 and 1")
    if model.get_cat_feature_indices() or model.get_text_feature_indices():
        raise ValueError("CatBoost ONNX export supports numeric features only")
    features = np.asarray(example, dtype=np.float32)
    if features.ndim != 2 or features.shape[1] != len(model.feature_names_):
        raise ValueError("Export example must match the fitted feature width")
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / "model.onnx"
        model.save_model(str(path), format="onnx")
        graph = onnx.load(path)
    maps = [node for node in graph.graph.node if node.op_type == "ZipMap"]
    if len(maps) != 1:
        raise ValueError("Expected one CatBoost probability map")
    probabilities = maps[0].input[0]
    graph.graph.node.remove(maps[0])
    graph.graph.initializer.append(
        helper.make_tensor("positive_class_index", TensorProto.INT64, [1], [1])
    )
    graph.graph.node.append(
        helper.make_node(
            "Gather", [probabilities, "positive_class_index"], ["score"], axis=1
        )
    )
    if not any(opset.domain == "" for opset in graph.opset_import):
        graph.opset_import.append(helper.make_opsetid("", 13))
    graph.ir_version = 8
    del graph.graph.output[:]
    graph.graph.output.append(
        helper.make_tensor_value_info("score", TensorProto.FLOAT, [None, 1])
    )
    onnx.checker.check_model(graph)
    return graph


def xgboost(model, example):
    """Export numeric XGBoost inputs to one positive-class probability tensor."""
    from onnxmltools import convert_xgboost
    from onnxmltools.convert.common.data_types import FloatTensorType

    features = np.asarray(example, dtype=np.float32)
    if not np.array_equal(model.classes_, [0, 1]):
        raise ValueError("XGBoost ONNX export requires binary labels 0 and 1")
    if features.ndim != 2 or features.shape[1] != model.n_features_in_:
        raise ValueError("Export example must match the fitted feature width")
    feature_types = model.get_booster().feature_types
    if feature_types is not None and "c" in feature_types:
        raise ValueError("XGBoost ONNX export supports numeric features only")
    graph = convert_xgboost(
        model,
        name="teammate_bump_xgboost",
        initial_types=[("features", FloatTensorType([None, features.shape[1]]))],
        target_opset=13,
    )
    graph.graph.initializer.append(
        helper.make_tensor("positive_class_index", TensorProto.INT64, [1], [1])
    )
    graph.graph.node.append(
        helper.make_node(
            "Gather",
            [graph.graph.output[1].name, "positive_class_index"],
            ["positive_probability"],
            axis=1,
        )
    )
    for name, value in (("probability_min", 0.0), ("probability_max", 1.0)):
        graph.graph.initializer.append(
            helper.make_tensor(name, TensorProto.FLOAT, [], [value])
        )
    graph.graph.node.append(
        helper.make_node(
            "Clip",
            ["positive_probability", "probability_min", "probability_max"],
            ["score"],
        )
    )
    if not any(opset.domain == "" for opset in graph.opset_import):
        graph.opset_import.append(helper.make_opsetid("", 13))
    graph.ir_version = 8
    del graph.graph.output[:]
    graph.graph.output.append(
        helper.make_tensor_value_info("score", TensorProto.FLOAT, [None, 1])
    )
    onnx.checker.check_model(graph)
    return graph


def verify(graph, features, probabilities):
    """Require a dynamic float32 [N, F] -> [N, 1] graph matching native probabilities."""
    features = np.asarray(features, dtype=np.float32)
    expected = np.asarray(probabilities).reshape(-1)
    if features.ndim != 2 or not len(features) or np.isinf(features).any():
        raise ValueError(
            "ONNX verification requires numeric rows without infinite values"
        )
    if len(expected) != len(features) or not np.isfinite(expected).all():
        raise ValueError("Provide one finite native probability per verification row")
    onnx.checker.check_model(graph)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(
        graph.SerializeToString(), options, providers=["CPUExecutionProvider"]
    )
    session.disable_fallback()
    if len(session.get_inputs()) != 1 or len(session.get_outputs()) != 1:
        raise ValueError("ONNX model must have one input and one output")
    input_tensor, output_tensor = session.get_inputs()[0], session.get_outputs()[0]
    for tensor, width in ((input_tensor, features.shape[1]), (output_tensor, 1)):
        if (
            tensor.type != "tensor(float)"
            or len(tensor.shape) != 2
            or isinstance(tensor.shape[0], int)
            or tensor.shape[1] != width
        ):
            raise ValueError(
                f"ONNX {tensor.name} must be a dynamic float32 [N, {width}] tensor"
            )
    actual = session.run([output_tensor.name], {input_tensor.name: features})[0]
    if actual.shape != (len(features), 1) or not np.isfinite(actual).all():
        raise ValueError("ONNX output must contain one finite probability per row")
    if ((actual < 0) | (actual > 1)).any():
        raise ValueError("ONNX probabilities must lie between 0 and 1")
    atol, rtol = 1e-6, 1e-5
    np.testing.assert_allclose(actual[:, 0], expected, atol=atol, rtol=rtol)
    return {
        "rows": len(features),
        "features": features.shape[1],
        "max_absolute_error": float(np.max(np.abs(actual[:, 0] - expected))),
        "absolute_tolerance": atol,
        "relative_tolerance": rtol,
        "execution_provider": "CPUExecutionProvider",
    }
