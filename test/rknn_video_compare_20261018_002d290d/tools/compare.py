"""Create an FP32 baseline and score detection JSONL against the bundled BDD GT."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
from statistics import mean
import time

from evaluate_bdd100k_tracking import ObjectRecord, box_iou, canonical_class, detection_metrics

ROOT = Path(__file__).resolve().parent.parent
IGNORE = "__ignore_region__"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def load_context() -> tuple[dict, dict, dict]:
    manifest = json.loads((ROOT / "manifest.json").read_text())
    gt_path = ROOT / manifest["ground_truth"]["path"]
    if sha256(gt_path) != manifest["ground_truth"]["sha256"]:
        raise ValueError("Ground-truth hash mismatch")
    labels, crowds = {}, {}
    stride = manifest["ground_truth"]["label_index_to_source_frame_stride"]
    for frame in json.loads(gt_path.read_text()):
        index = frame["index"] * stride
        if index in labels or not 0 <= index < manifest["video"]["decoded_frame_count"]:
            raise ValueError("Invalid ground-truth frame alignment")
        labels[index], crowds[index] = [], []
        for item in frame["labels"]:
            distractor = item["category"] in {"other person", "other vehicle", "trailer"}
            name = IGNORE if distractor else canonical_class(item["category"])
            if name is None:
                continue
            box = tuple(float(item["box2d"][key]) for key in ("x1", "y1", "x2", "y2"))
            if not all(math.isfinite(v) for v in box) or box[2] <= box[0] or box[3] <= box[1]:
                raise ValueError("Invalid ground-truth box")
            target = crowds if distractor or item["attributes"].get("Crowd", False) else labels
            target[index].append(ObjectRecord(index, str(item["id"]), name, box))
    if len(labels) != manifest["ground_truth"]["labeled_frames"]:
        raise ValueError("Ground-truth frame count mismatch")
    mapping = json.loads((ROOT / "frame_map.json").read_text())
    if {row["frame_index"] for row in mapping} != set(labels):
        raise ValueError("Frame map differs from ground truth")
    return manifest, labels, crowds


def load_predictions(path: Path, manifest: dict, indices: set[int]) -> tuple[dict, dict]:
    predicted, seen, summary = {}, set(), {}
    ignored_classes = 0
    for line in path.read_text().splitlines():
        row = json.loads(line)
        if row.get("type") == "summary":
            if summary:
                raise ValueError("Multiple prediction summaries")
            summary = row
            continue
        if row.get("type") != "frame":
            raise ValueError("Expected frame/summary JSONL records")
        index = row["frame_index"]
        if not isinstance(index, int) or index in seen or not 0 <= index < manifest["video"]["decoded_frame_count"]:
            raise ValueError("Duplicate or invalid source frame index")
        seen.add(index)
        if index not in indices:
            continue
        if row.get("is_detection_frame") is not True:
            raise ValueError("Expected detection on every labeled frame; disable LK/stride for this comparison")
        if (row["width"], row["height"]) != (manifest["video"]["width"], manifest["video"]["height"]):
            raise ValueError("Predictions must use original video coordinates")
        predicted[index] = []
        for j, item in enumerate(row["detections"]):
            cls, score = item["class_id"], float(item["score"])
            if not isinstance(cls, int) or not 0 <= cls < len(manifest["classes"]):
                raise ValueError("Invalid predicted class ID")
            if item.get("class_name", manifest["classes"][cls]) != manifest["classes"][cls]:
                raise ValueError("Class name/ID mismatch")
            box = tuple(float(v) for v in item["bbox"])
            if (len(box) != 4 or not all(math.isfinite(v) for v in (*box, score))
                    or not 0 <= score <= 1 or box[2] <= box[0] or box[3] <= box[1]
                    or box[0] < 0 or box[1] < 0 or box[2] > row["width"] or box[3] > row["height"]):
                raise ValueError("Non-finite or invalid prediction")
            if cls not in manifest["annotated_class_ids"]:
                ignored_classes += 1
                continue
            predicted[index].append(ObjectRecord(index, str(j), manifest["classes"][cls], box, score))
    if set(predicted) != indices:
        raise ValueError(f"Missing labeled source frames: {sorted(indices - set(predicted))[:10]}")
    return predicted, {"summary": summary, "excluded_unannotated_class_predictions": ignored_classes,
                       "sha256": sha256(path), "records": len(seen)}


def suppress_ignored(predictions: dict, labels: dict, crowds: dict, iou: float) -> tuple[dict, int]:
    """Same normal-GT-first ignore policy as the repository PTQ evaluator."""
    result, ignored = {}, 0
    for index, items in predictions.items():
        used_labels, matched = set(), set()
        for j in sorted(range(len(items)), key=lambda j: items[j].score, reverse=True):
            prediction = items[j]
            best = None
            for k, label in enumerate(labels[index]):
                if k in used_labels or label.class_name != prediction.class_name:
                    continue
                overlap = box_iou(label.box, prediction.box)
                if overlap >= iou and (best is None or overlap > best[0]):
                    best = overlap, k
            if best is not None:
                used_labels.add(best[1])
                matched.add(j)
        result[index] = []
        for j, item in enumerate(items):
            overlaps = any(c.class_name in {IGNORE, item.class_name} and box_iou(c.box, item.box) >= iou
                           for c in crowds[index])
            if j not in matched and overlaps:
                ignored += 1
            else:
                result[index].append(item)
    return result, ignored


def score_predictions(predicted: dict, labels: dict, crowds: dict) -> dict:
    points = {}
    for step in range(10):
        iou = round(.5 + .05 * step, 2)
        candidates = {index: [p for p in items if p.score >= .001] for index, items in predicted.items()}
        candidates, ignored = suppress_ignored(candidates, labels, crowds, iou)
        pooled, per_class = detection_metrics(labels, candidates, iou)
        points[f"{iou:.2f}"] = {"pooled": pooled, "per_class": per_class, "ignored": ignored}
    operating = {index: [p for p in items if p.score >= .25] for index, items in predicted.items()}
    operating, ignored = suppress_ignored(operating, labels, crowds, .5)
    pooled, per_class = detection_metrics(labels, operating, .5)
    classes = copy.deepcopy(points["0.50"]["per_class"])
    for name, row in classes.items():
        aps = [point["per_class"].get(name, {}).get("ap50") for point in points.values()]
        row["ap50_95"] = mean(v for v in aps if v is not None) if any(v is not None for v in aps) else None
        empty = {key: 0 for key in ("ground_truth", "predictions", "true_positives", "false_positives", "false_negatives",
                                    "precision", "recall", "f1", "mean_matched_iou", "minimum_prediction_score")}
        row["operating_point"] = {key: value for key, value in per_class.get(name, empty).items() if key != "ap50"}
    return {"map50": points["0.50"]["pooled"]["map50"],
            "map50_95": mean(point["pooled"]["map50"] for point in points.values()),
            "operating_point": {key: value for key, value in pooled.items() if key != "map50"},
            "ignored_predictions_at_025": ignored, "per_class": classes,
            "classes_with_ground_truth": [name for name, row in classes.items() if row["ground_truth"]],
            "iou_results": points}


def evaluate(args: argparse.Namespace) -> None:
    manifest, labels, crowds = load_context()
    report = {"video": manifest["video"], "evaluation": manifest["evaluation"], "models": {}}
    for argument in args.prediction:
        name, separator, raw_path = argument.partition("=")
        if not separator or not name or name in report["models"]:
            raise ValueError("Use unique NAME=PATH prediction arguments")
        predicted, metadata = load_predictions(Path(raw_path), manifest, set(labels))
        report["models"][name] = {**score_predictions(predicted, labels, crowds), "input": metadata}
        print(f"Scored {name}: {len(predicted)} labeled source frames", flush=True)
    report["deltas_vs_fp32"] = {}
    if "FP32" in report["models"]:
        baseline = report["models"]["FP32"]
        for name, model in report["models"].items():
            if name == "FP32":
                continue
            person, before = model["per_class"]["person"], baseline["per_class"]["person"]
            report["deltas_vs_fp32"][name] = {
                "map50_pp": 100 * (model["map50"] - baseline["map50"]),
                "map50_95_pp": 100 * (model["map50_95"] - baseline["map50_95"]),
                "person_ap50_pp": 100 * (person["ap50"] - before["ap50"]),
                "person_ap50_95_pp": 100 * (person["ap50_95"] - before["ap50_95"]),
                **{f"person_{key}_pp": 100 * (person["operating_point"][key] - before["operating_point"][key])
                   for key in ("precision", "recall", "f1")}}
    markdown = args.output.with_suffix(".md")
    if args.output.exists() or markdown.exists():
        raise FileExistsError("Refusing to overwrite an existing report")
    lines = ["# Video detection comparison", "", "AP101; confidence floor 0.001; P/R/F1 at 0.25, IoU 0.50.",
             "202 labeled frames from one BDD100K training video; not an independent validation result.", "",
             "| Model | mAP50 | mAP50-95 | Person AP50 | Person AP50-95 | Person P | Person R | Person F1 |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for name, model in report["models"].items():
        person = model["per_class"]["person"]
        op = person["operating_point"]
        values = [model["map50"], model["map50_95"], person["ap50"], person["ap50_95"], op["precision"], op["recall"], op["f1"]]
        lines.append("| " + name + " | " + " | ".join(f"{v*100:.2f}%" for v in values) + " |")
    lines += ["", "mAP averages classes with GT; traffic lights/signs have no annotations and are excluded.",
              "The field scored is detections, not LK/track output. Frame alignment is source_frame=label_index*6.",
              "Crowd/distractor overlaps are ignored after matching ordinary GT. Results are not official COCO metrics.",
              "FP32 infers only labeled frames; its total elapsed time/FPS cannot be compared with a full-video board run."]
    write_json(args.output, report)
    with markdown.open("x") as stream:
        stream.write("\n".join(lines) + "\n")
    print(markdown, flush=True)


def baseline(args: argparse.Namespace) -> None:
    import cv2
    import onnxruntime as ort
    from yolo26_deploy import detections, float_input, split_outputs

    manifest, labels, _ = load_context()
    if sha256(args.model) != manifest["model"]["fp32_onnx_sha256"]:
        raise ValueError("Expected the same FP32 ONNX used to produce the RKNN weights")
    video = ROOT / manifest["video"]["path"]
    if sha256(video) != manifest["video"]["sha256"]:
        raise ValueError("Video hash mismatch")
    if args.output.exists():
        raise FileExistsError(args.output)
    options = ort.SessionOptions()
    options.intra_op_num_threads = args.threads
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(args.model), options, providers=["CPUExecutionProvider"])
    if session.get_inputs()[0].shape != [1, 3, 608, 800]:
        raise ValueError("Unexpected FP32 input shape")
    cv2.setNumThreads(2)
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise ValueError("Cannot decode video")
    times = {key: [] for key in ("decode", "preprocess", "infer", "postprocess")}
    count, inferred, warmed = 0, 0, False
    start = time.perf_counter()
    try:
        with args.output.open("x") as stream:
            while True:
                tick = time.perf_counter()
                ok, image = capture.read()
                if not ok:
                    break
                times["decode"].append((time.perf_counter() - tick) * 1000)
                index = count
                count += 1
                if index not in labels:
                    continue
                tick = time.perf_counter()
                height, width = image.shape[:2]
                ratio = min(800 / width, 608 / height)
                rw, rh = round(width * ratio), round(height * ratio)
                left, top = (800-rw)//2, (608-rh)//2
                image = cv2.resize(image, (rw, rh), interpolation=cv2.INTER_LINEAR)
                rgb = cv2.cvtColor(cv2.copyMakeBorder(image, top, 608-rh-top, left, 800-rw-left,
                    cv2.BORDER_CONSTANT, value=(114, 114, 114)), cv2.COLOR_BGR2RGB)
                tensor = float_input(rgb)
                transform = {"width": width, "height": height, "scale_x": rw/width,
                             "scale_y": rh/height, "left": left, "top": top}
                times["preprocess"].append((time.perf_counter()-tick)*1000)
                if not warmed:
                    for _ in range(5):
                        session.run(None, {session.get_inputs()[0].name: tensor})
                    warmed = True
                tick = time.perf_counter()
                outputs = list(split_outputs(session.run(None, {session.get_inputs()[0].name: tensor})))
                times["infer"].append((time.perf_counter()-tick)*1000)
                if [list(v.shape) for v in outputs] != [[1, 4, 9975], [1, 10, 9975]]:
                    raise ValueError("Unexpected FP32 output shapes")
                tick = time.perf_counter()
                boxes = detections(outputs, transform, threshold=.001)
                times["postprocess"].append((time.perf_counter()-tick)*1000)
                row = {"type": "frame", "frame_index": index,
                       "timestamp_ms": index / manifest["video"]["fps"] * 1000,
                       "width": width, "height": height, "is_detection_frame": True,
                       "tracks_source": "detected", "detections": [
                           {"class_id": cls, "class_name": manifest["classes"][cls], "score": score, "bbox": box}
                           for cls, score, box in boxes]}
                stream.write(json.dumps(row, allow_nan=False) + "\n")
                inferred += 1
                if inferred % 25 == 0:
                    stream.flush()
                    print(f"FP32: {inferred}/{len(labels)} labeled frames", flush=True)
            if count != manifest["video"]["decoded_frame_count"] or inferred != len(labels):
                raise ValueError("Incomplete video decode or inference")
            timing = {}
            for name, values in times.items():
                ordered = sorted(values)
                position = (len(ordered)-1) * .95
                lo, hi = math.floor(position), math.ceil(position)
                p95 = ordered[lo] + (ordered[hi]-ordered[lo]) * (position-lo)
                timing[name] = {"count": len(values), "mean_ms": mean(values), "p95_ms": p95, "total_ms": sum(values)}
            summary = {"type": "summary", "frame_count": count, "inference_count": inferred,
                       "inference_subset": "annotated_frames_only", "warmup_calls": 5,
                       "model_sha256": sha256(args.model), "video_sha256": sha256(video),
                       "runtime": "onnxruntime", "runtime_version": ort.__version__, "opencv_version": cv2.__version__,
                       "providers": session.get_providers(), "threads": args.threads,
                       "candidate_score_floor": .001, "timing": timing,
                       "elapsed_ms_including_decode_and_warmup": (time.perf_counter()-start)*1000}
            stream.write(json.dumps(summary, allow_nan=False) + "\n")
    finally:
        capture.release()
    print(json.dumps(summary), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    fp32 = commands.add_parser("baseline")
    fp32.add_argument("--model", type=Path, required=True)
    fp32.add_argument("--output", type=Path, default=ROOT / "baseline/fp32_predictions.jsonl")
    fp32.add_argument("--threads", type=int, default=8)
    scoring = commands.add_parser("evaluate")
    scoring.add_argument("--prediction", action="append", required=True, metavar="NAME=PATH")
    scoring.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "baseline":
        baseline(args)
    else:
        evaluate(args)


if __name__ == "__main__":
    main()
