# Video detection comparison

AP101; confidence floor 0.001; P/R/F1 at 0.25, IoU 0.50.
202 labeled frames from one BDD100K training video; not an independent validation result.

| Model | mAP50 | mAP50-95 | Person AP50 | Person AP50-95 | Person P | Person R | Person F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| FP32 | 48.58% | 24.48% | 52.90% | 17.56% | 58.06% | 55.86% | 56.93% |

mAP averages classes with GT; traffic lights/signs have no annotations and are excluded.
The field scored is detections, not LK/track output. Frame alignment is source_frame=label_index*6.
Crowd/distractor overlaps are ignored after matching ordinary GT. Results are not official COCO metrics.
FP32 infers only labeled frames; its total elapsed time/FPS cannot be compared with a full-video board run.
