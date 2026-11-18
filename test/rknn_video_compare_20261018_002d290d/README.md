# 800×608 FP32 与 RKNN 视频检测对比材料

本材料已完成本地 FP32 检测基线，并准备了 RKNN FP16、INT8 的独立测试配置和统一评测入口。此次未执行板端推理，也没有修改现有 C++ 源码或部署配置。

公开仓库包含评测工具、配置、清单和基线指标。运行下列涉及原始素材或逐帧预测的步骤前，请自行准备对应文件，并按 `SHA256SUMS` 校验。

## 文件与数据

- `video/002d290d-90f2bab2.mov`：原始 H.264 视频，1280×720，1210 帧，约 40.19 秒，19,714,756 字节。没有重编码、改帧率或预先绘制检测框。
- `gt/002d290d-90f2bab2.json`：202 个标注帧，2587 个人体普通真值框；另有 213 个人体 crowd 框作为忽略区域。保留原始目标 ID、遮挡与截断属性。
- `frame_map.json`：标注索引与解码帧索引的对应关系。
- `baseline/fp32_predictions.jsonl`：本地 FP32 的 202 个标注帧检测结果及运行摘要；`fp32_metrics.json`、`fp32_metrics.md` 给出精度结果。
- `configs/rknn_fp16_detection.yaml`、`configs/rknn_int8_detection.yaml`：仅供本次对比的配置。
- `classes.json`、`interface.json`、`manifest.json`：类别、输入输出、模型/视频/真值哈希及评测口径。
- `tools/compare.py`：FP32 基线与统一精度评测；其中 `evaluate` 子命令只依赖 Python 3.10+ 标准库。
- `SHA256SUMS`：上传文件的逐文件校验清单。

视频容器的**编码尺寸是 720×1280，并带有 90° 旋转标记**；本地 OpenCV 按旋转标记解码后为标注所使用的 1280×720。远端 FFprobe 已完整解码确认同样为 1210 帧。板端需要保留自动方向处理；如果改用 MPP 读取编码帧，须先按旋转标记得到相同方向，再做 letterbox。评测器要求输出使用 1280×720 原图坐标，会拒绝未正确处理方向的结果。视频原文件保持不变，未为消除旋转标记而重新编码。

这是定向选择的 **BDD100K training 视频**，用于对比同一权重在不同后端的变化，不能视为独立测试集精度。此视频普通真值覆盖 person、rider、car、truck、bus、bike 六类；motor、train 没有普通真值。MOT 标注不包含 traffic light、traffic sign，评测会排除这两类预测，而不是将它们算成误报。mAP 只对存在普通真值的六类取平均，与十类完整验证集 mAP 不同。

## 必须保持一致的评测条件

1. 使用生成已上传 RKNN 权重的同一份 FP32 ONNX，输入 `[1,3,608,800]`。FP32 SHA-256 为 `4e8404d9135d22a369dd0c3009dbeee3236fa4e06e0468df562791cd329459f1`。
2. 原图按比例缩放至 800×450，上下各填充 79 像素，填充值 114；BGR 转 RGB。FP32 使用 NCHW float32 `/255`；RKNN 常规输入使用 NHWC uint8 RGB，由模型配置完成归一化。
3. 输出为 `boxes [1,4,9975]` 和 `scores [1,10,9975]`，坐标是模型输入上的 xyxy，分数已做 Sigmoid。复现两阶段 TopK，最多 300 框，不添加 NMS。RKNN 输出需请求浮点数或正确反量化。
4. 保存候选分数下限 **0.001**，再在评测时计算 **conf=0.25、IoU=0.50** 的 Precision、Recall、F1、TP、FP、FN。不要以 0.25 过滤后的结果计算完整 AP。
5. AP 使用 101 点插值；AP50-95 为 IoU 0.50:0.05:0.95 的平均。先匹配普通真值，再忽略未匹配且与同类 crowd / 任意类 distractor 区域重叠的预测。此为仓库评测口径，不是官方 COCO 全套评测。
6. **标注索引 n 对应源视频解码帧 n×6，均从 0 开始。** 最后一个有效标注对应源帧 1206；超出视频尾部的标注索引 202 已排除。不要依靠四舍五入的时间戳对齐。
7. 本次评测读取 `detections`，不读取 `tracks`。板端配置设为逐帧检测、关闭异步和 LK 补帧，并先使用 OpenCV 预处理，减少调度和预处理差异。

本地仅推理 202 个标注帧，但完整解码 1210 帧；板端可逐帧推理整段视频，评测器仍只取相同的 202 个标注帧。两者整段耗时和 FPS 不能直接相比；推理调用次数、耗时口径与硬件也必须单独记录。

## 板端复现命令

材料上传目录：`/home/yolo/test/rknn_video_compare_20261018_002d290d`。

以下命令供后续运行。要求现有 `/home/yolo/yolo_run` 已完成构建；本次没有替你构建或执行它。如可执行文件位置不同，只调整 `binary` 变量。

```bash
cd /home/yolo/test/rknn_video_compare_20261018_002d290d
sha256sum -c SHA256SUMS

binary=/home/yolo/yolo_run/build-rk3588/yolo_run
result_dir=$(mktemp -d "$PWD/results.XXXXXXXX")

"$binary" \
  --input "$PWD/video/002d290d-90f2bab2.mov" \
  --config "$PWD/configs/rknn_fp16_detection.yaml" \
  --output "$result_dir/rknn_fp16.jsonl"

"$binary" \
  --input "$PWD/video/002d290d-90f2bab2.mov" \
  --config "$PWD/configs/rknn_int8_detection.yaml" \
  --output "$result_dir/rknn_int8.jsonl"
```

两个配置分别读取 `/home/yolo/yolo_model_rknn/` 中已上传的同名权重。权重的预期 SHA-256 在 `manifest.json` 中。结果文件使用新建目录，避免覆盖历史结果。

在板端已有 Python 3.10+ 环境中运行下列评测；也可以将两个 JSONL 拷回本机后，通过 `conda run -n yolo python` 调用同一命令。评测不需要 RKNN、OpenCV、ONNX Runtime 或额外安装依赖。

```bash
python tools/compare.py evaluate \
  --prediction "FP32=$PWD/baseline/fp32_predictions.jsonl" \
  --prediction "RKNN_FP16=$result_dir/rknn_fp16.jsonl" \
  --prediction "RKNN_INT8=$result_dir/rknn_int8.jsonl" \
  --output "$result_dir/comparison.json"
```

生成 `comparison.json` 和 `comparison.md`，包含每类与人体的 AP50、AP50-95、固定阈值 Precision/Recall/F1、误报和漏检，以及相对 FP32 的百分点变化。漏掉标注帧、帧号重复、坐标空间错误、非有限数值、类别映射错误或混入补帧结果时会拒绝评测。

## 本地复现与已执行验证

使用本仓库 `yolo` 环境。下面的模型路径指向本次训练导出的原始文件，材料包没有再次复制模型权重。

```bash
conda run -n yolo python tools/compare.py baseline \
  --model /home/ubuntu/YOLO/model/runs/detect/bdd100k_yolo26s_800x608_person_20261017_155242/export/fp32.onnx \
  --output baseline/fp32_repeat.jsonl

conda run -n yolo python tools/compare.py evaluate \
  --prediction FP32=baseline/fp32_repeat.jsonl \
  --output baseline/fp32_repeat_metrics.json

conda run -n yolo python -m unittest discover -s tools -p test_compare.py -v
```

基线使用 ONNX Runtime CPU、8 个推理线程、2 个 OpenCV 线程、5 次预热；调用次数及预处理/推理/后处理耗时在 JSONL 最后一行。重复输出路径必须不存在。

已执行：原视频和真值哈希核对、完整视频解码、202 个标注帧 FP32 推理、输入输出尺寸和有限性检查、统一精度统计，以及 AP/重复预测、crowd/distractor 忽略、缺帧/重复帧、补帧与 NaN 拒绝的四项回归测试。

## 本次本地基线实测

| 指标 | FP32 |
|---|---:|
| 六类 mAP50 | 48.58% |
| 六类 mAP50-95 | 24.48% |
| 人体 AP50 | 52.90% |
| 人体 AP50-95 | 17.56% |
| 人体 Precision @0.25 | 58.06% |
| 人体 Recall @0.25 | 55.86% |
| 人体 F1 @0.25 | 56.93% |

人体真值 2587，TP=1445、FP=1044、FN=1142。
逐类固定阈值统计读取 JSON 的 `per_class.<class>.operating_point`；其外层 P/R/计数使用 0.001 候选下限，供 AP 诊断，不应当作 0.25 工作点结果。

本材料评测器与原 `model.ptq_analysis.quality` 在本次全部 202 个标注帧上的整体 AP、逐类 AP 和固定阈值计数逐项一致，见 `baseline/evaluator_parity.json`。
