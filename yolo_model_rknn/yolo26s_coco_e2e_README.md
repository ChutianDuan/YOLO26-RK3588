# YOLO26s COCO integrated RKNN interface

Project integration status (2026-11-18): the current C++ `YoloEngine` requires two split outputs (`boxes` and `scores`) and does not accept the single `[1,300,6]` output below. These models are conversion artifacts, not a drop-in replacement for the deployed split-output model.

Target: RK3588. Built with RKNN-Toolkit2 2.5.0 for matching RKNN Runtime 2.5.0.
The e2e filenames distinguish these integrated models from the previous split-output models.

Models: yolo26s_coco_e2e_fp16.rknn, yolo26s_coco_e2e_int8.rknn,
and yolo26s_coco_e2e_fp16_int8.rknn.

The original one2one detection head, box decoding and two TopK stages are retained.
The application does not need to implement TopK or NMS separately.
Compiler placement shows Conv on NPU and TopK/GatherElements on CPU inside the RKNN runtime.
Integration into one file describes the runtime interface, not all-NPU execution.
No latency advantage is claimed without a board comparison.

## Input

- uint8 RGB NHWC [1,640,640,3], fmt=NHWC, pass_through=0.
- Resize with preserved aspect ratio and center-pad with 114 before inference.
- Mean=0 and std=255 are inside RKNN; do not divide pixels by 255 again.

## Output

- A single tensor [1,300,6]. Request C API want_float=1.
- Columns: x1, y1, x2, y2, score, class_id.
- Coordinates are in the padded 640x640 image.
- Apply the application's confidence threshold and reverse letterboxing.
- COCO class order is listed in yolo26s_coco_e2e_classes.json.

Each model is smoke-tested with three original COCO128 images using the live host simulator.
Quantized models use all 128 COCO128 images for native Toolkit2 calibration.
This is not a full COCO accuracy benchmark or board performance/compatibility verification.

INT8 denotes the quantized convolutional body. Decoding, scores, TopK and final
output assembly retain FP16 to prevent coordinate/score shared-scale loss.
The fp16_int8 variant additionally keeps the six terminal detection Conv heads FP16.
