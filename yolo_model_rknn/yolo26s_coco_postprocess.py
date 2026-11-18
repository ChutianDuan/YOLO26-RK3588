"""NumPy/OpenCV YOLO26 split-output preprocessing and NMS-free postprocessing.

Shared by the training environment and the isolated RKNN environment.
"""
from __future__ import annotations

from pathlib import Path
import tempfile

import cv2
import numpy as np


def image_input(path: Path, size: tuple[int, int]) -> tuple[np.ndarray, dict]:
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"Unreadable image: {path}")
    h, w = image.shape[:2]
    height, width = size
    ratio = min(width / w, height / h)
    rw, rh = max(1, round(w * ratio)), max(1, round(h * ratio))
    resized = cv2.resize(image, (rw, rh), interpolation=cv2.INTER_LINEAR)
    left, top = (width - rw) // 2, (height - rh) // 2
    padded = cv2.copyMakeBorder(resized, top, height-rh-top, left, width-rw-left,
                               cv2.BORDER_CONSTANT, value=(114, 114, 114))
    return cv2.cvtColor(padded, cv2.COLOR_BGR2RGB), {
        "width": w, "height": h, "scale_x": rw / w, "scale_y": rh / h,
        "left": left, "top": top,
    }


def float_input(rgb: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(rgb.transpose(2, 0, 1)[None], dtype=np.float32) / np.float32(255)


def split_outputs(outputs: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    if len(outputs) != 2:
        raise ValueError("Expected two outputs: boxes [1,4,N], scores [1,C,N]")
    boxes, scores = (np.asarray(x, dtype=np.float32) for x in outputs)
    if boxes.ndim == 3 and boxes.shape[1] != 4:
        boxes, scores = scores, boxes
    if (boxes.ndim != 3 or scores.ndim != 3 or boxes.shape[:2] != (1, 4)
            or scores.shape[0] != 1 or boxes.shape[2] != scores.shape[2]
            or not boxes.shape[2] or not np.isfinite(boxes).all() or not np.isfinite(scores).all()):
        raise ValueError("Invalid or non-finite split YOLO26 outputs")
    return boxes, scores


def topk_detections(outputs: list[np.ndarray], max_det: int = 300) -> np.ndarray:
    boxes, scores = split_outputs(outputs)
    boxes, scores = boxes[0].T, scores[0].T
    k = min(max_det, len(boxes))
    # Match Detect.get_topk_index: anchor TopK, then anchor/class pair TopK.
    anchors = np.argsort(-scores.max(axis=1), kind="stable")[:k]
    flat = scores[anchors].reshape(-1)
    pairs = np.argsort(-flat, kind="stable")[:k]
    selected = anchors[pairs // scores.shape[1]]
    return np.column_stack((boxes[selected], flat[pairs], pairs % scores.shape[1])).astype(np.float32)


def detections(outputs: list[np.ndarray], transform: dict, threshold: float = .001) -> list:
    rows = topk_detections(outputs)
    result = []
    for x1, y1, x2, y2, score, cls in rows:
        if score < threshold:
            continue
        box = ((x1-transform["left"])/transform["scale_x"],
               (y1-transform["top"])/transform["scale_y"],
               (x2-transform["left"])/transform["scale_x"],
               (y2-transform["top"])/transform["scale_y"])
        box = tuple(float(np.clip(v, 0, transform["width"] if i % 2 == 0 else transform["height"]))
                    for i, v in enumerate(box))
        if box[2] > box[0] and box[3] > box[1]:
            result.append((int(cls), float(score), box))
    return result


def extract_split_onnx(source: Path, target: Path, classes: int) -> None:
    """Find the decoded boxes/sigmoid concat by shape, not export node names."""
    import onnx

    if target.exists():
        raise FileExistsError(f"Refusing to overwrite: {target}")
    graph = onnx.shape_inference.infer_shapes(onnx.load(str(source)))
    shapes = {v.name: [d.dim_value for d in v.type.tensor_type.shape.dim]
              for v in [*graph.graph.value_info, *graph.graph.input, *graph.graph.output]}
    shapes.update({value.name: list(value.dims) for value in graph.graph.initializer})
    producers = {name: n for n in graph.graph.node for name in n.output}
    candidates = []
    for node in graph.graph.node:
        if node.op_type != "Concat" or len(node.input) != 2:
            continue
        a, b = (shapes.get(name, []) for name in node.input)
        if (len(a) == len(b) == 3 and a[0] in (0, 1) and a[1] == 4 and b[:2] == [1, classes]
                and a[2] == b[2] and a[2] > 0
                and producers[node.input[1]].op_type == "Sigmoid"
                and any(x.name == "axis" and x.i == 1 for x in node.attribute)):
            candidates.append(node)
    if len(candidates) != 1:
        raise ValueError(f"Expected one YOLO26 decoded one2one concat, found {len(candidates)}")
    originals = list(candidates[0].input)
    outputs = ["boxes", "scores"]
    for name, alias, channels in zip(originals, outputs, (4, classes)):
        if alias in producers or alias in shapes:
            raise ValueError(f"Output alias already exists: {alias}")
        graph.graph.node.append(onnx.helper.make_node("Identity", [name], [alias]))
        graph.graph.value_info.append(onnx.helper.make_tensor_value_info(
            alias, onnx.TensorProto.FLOAT, [1, channels, shapes[name][2]]))
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="yolo_split_") as directory:
        inferred = Path(directory) / "inferred.onnx"
        extracted = Path(directory) / "split.onnx"
        onnx.save(graph, str(inferred))
        onnx.utils.extract_model(str(inferred), str(extracted), [graph.graph.input[0].name], outputs)
        split = onnx.load(str(extracted))
        onnx.checker.check_model(split)
        onnx.save(split, str(target))
