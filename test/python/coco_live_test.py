#!/usr/bin/python3
"""COCO640 camera_live configuration, MPP replay and explicit model precedence."""
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile


binary, fixture_tool, source = map(Path, sys.argv[1:])
env = dict(os.environ)
for key in ("LD_LIBRARY_PATH", "LD_PRELOAD"):
    env.pop(key, None)
config = source / "config/camera_coco640.yaml"
model_dir = source.parent / "yolo_model_rknn"

with tempfile.TemporaryDirectory(prefix="coco_live_") as directory:
    work = Path(directory)
    fixture = work / "fixture"
    video = source.parent / "test/rknn_video_compare_20261018_002d290d/video/002d290d-90f2bab2.mov"
    subprocess.run([str(fixture_tool), "video", str(video), str(fixture), "60"],
                   env=env, check=True, capture_output=True, timeout=30)
    common = [str(binary), "--config", str(config), "--preview", "off",
              "--replay", str(fixture / "manifest.jsonl")]
    output = work / "coco.jsonl"
    result = subprocess.run([*common, "--output", str(output)], env=env,
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0 and result.stdout == "", result.stderr
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    meta, summary = rows[0], rows[-1]
    assert Path(meta["model"]) == model_dir / "yolo26s_coco_fp16.rknn", meta
    assert (meta["model_width"], meta["model_height"], meta["num_classes"]) == (640, 640, 80)
    assert meta["padding"] == {"top": 80, "bottom": 80, "left": 0, "right": 0, "rgb": [114,114,114]}
    assert math.isclose(meta["scale_x"], .8, abs_tol=1e-6)
    assert math.isclose(meta["scale_y"], .8, abs_tol=1e-6)
    assert meta["bbox_coordinates"] == "stabilized_800x600"
    assert meta["schema_version"] == 2 and meta["stabilization"]["enabled"]
    assert summary["status"] == "completed" and summary["inference_count"] > 0, summary
    frames = [row for row in rows if row["type"] == "frame"]
    assert frames and all((row["width"], row["height"]) == (800,600) for row in frames)
    detections = [d for row in rows if row["type"] == "detection" for d in row["detections"]]
    assert detections, "COCO replay did not detect any objects"
    for detection in detections:
        assert 0 <= detection["class_id"] < 80, detection
        assert 0 <= detection["score"] <= 1 and math.isfinite(detection["score"]), detection
        x1,y1,x2,y2 = detection["bbox"]
        assert 0 <= x1 < x2 <= 800 and 0 <= y1 < y2 <= 600, detection

    # A shape-incompatible explicit override must be used and rejected, rather
    # than being silently ignored in favor of the compatible config model.
    rejected = work / "override.jsonl"
    result = subprocess.run([*common, "--model", str(model_dir / "yolo26s_800x608_int8.rknn"),
                             "--output", str(rejected)], env=env,
                            capture_output=True, text=True, timeout=15)
    assert result.returncode != 0 and "RKNN input must be" in result.stderr, result.stderr
    assert not rejected.exists(), "invalid model created live output"

print("PASS COCO640 config-selected FP16, MPP replay, source coordinates and explicit model override")
