# YOLO26s COCO RK3588 models

Project integration status (2026-11-18): the C++ application validates the split-output FP16 model with `yolo_run/config/camera_coco640.yaml`; see [board deployment](../yolo_run/deploy/COCO640.md) and the [on-board camera and replay measurements](../README.md). The conversion/simulator results below remain separate evidence; they do not establish board validation of all precision variants.

Target: RK3588, RKNN-Toolkit2 2.5.0, matching Runtime 2.5.0.

Models:
- yolo26s_coco_fp16.rknn: non-quantized RKNN build (floating precision selected by Toolkit2).
- yolo26s_coco_int8.rknn: native Toolkit2 INT8 calibration using 128 COCO128 images.
- yolo26s_coco_fp16_int8.rknn: native hybrid build; the six terminal detection heads request FP16.

All models come from the original FP32 COCO model, not from ORT QDQ graphs.
No training or source-weight changes are involved.

## Runtime interface

1. Resize with aspect ratio to 640x640 and center-pad with 114.
2. Supply uint8 RGB NHWC [1,640,640,3], fmt=NHWC, pass_through=0.
   Normalization (mean=0, std=255) is inside RKNN; do not divide by 255 again.
3. Read two outputs in float (C API want_float=1), or dequantize according to
   runtime tensor attributes:
   - boxes [1,4,8400]: xyxy coordinates in the padded 640x640 image.
   - scores [1,80,8400]: sigmoid class probabilities in COCO class order.
4. Apply YOLO26 one2one two-stage TopK=300 on CPU (no NMS), then undo letterboxing.
   The included yolo26s_coco_postprocess.py provides this implementation.
   yolo26s_coco_classes.json contains the 80 class names.

These outputs differ from the original end-to-end ONNX [1,300,6].
A YOLOv8 raw-head/DFL decoder must not be used for these two decoded outputs.
Check tensor names/dimensions via rknn_query rather than assuming output order.

Each build is checked with the host simulator on three original COCO128 images.
This does not establish full COCO accuracy, board inference compatibility, latency,
or throughput. See yolo26s_coco_rknn_report.json for hashes and simulation errors.
