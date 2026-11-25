# RK3588 工业视觉与 YOLO26 部署

在 RK3588 上用 C++17 运行 YOLO26。工程提供 USB 摄像头检测、实时跟踪与防抖、本地图片和视频推理、最多四路视频并行处理，以及可选的 H.264 推流。仓库包含板端源码、配置和 RKNN 权重。模型转换在另一台 Linux 主机上用 RKNN-Toolkit2 2.5.0 完成，目标芯片为 `rk3588`。

**单路实摄结果**　在 800×600、30 FPS 的 USB MJPEG 摄像头上，COCO640 INT8 使用两个独立 NPU Context，采集到解码采用单槽 FIFO，进程限定在四个 Cortex-A76 核上。三个 60 秒窗口内共采集 5,402 帧，每帧都完成了检测、跟踪校正和画面输出。采集到检测完成、采集到应用内画框完成的 P99 分别为 **70.41 ms 和 26.20 ms**，取三轮中位数。画面线程可以使用先前的跟踪快照，两个分支分别计时。

## 环境与构建

实测板卡为视美泰 Shimeta AIoT RK3588A V1.3，8 GB 内存，ARM64 Linux `6.6.89-kdev`，RKNPU 驱动 0.9.8。板端使用系统 GCC 14.2、CMake 3.31、Ninja 和 OpenCV 4.10。CMake 从 `/opt/rockchip/cmake` 导入 RKNN Runtime 2.5.0、librga 1.10.6 与 MPP，使用系统 FFmpeg 开发库和 nlohmann_json。环境检查还会核对已有的 `rk3588-infer` Python 环境与 `env/` 中的依赖锁。

在仓库根目录执行。脚本用系统工具链原生构建，并发为 4。产物位于 `yolo_run/build-rk3588/`。

```bash
bash scripts/dev.sh doctor
bash scripts/dev.sh build
./yolo_run/build-rk3588/camera_live --help
```

## 四个运行入口

| 程序 | 输入 | 默认模型与行为 |
|---|---|---|
| `yolo_run` | 本地图片或视频 | 交通 10 类，800×608 FP16。视频异步检测目标为 4 FPS，中间帧使用 LK 和 ByteTrack。 |
| `camera_run` | USB YUYV，800×600，请求 15 FPS | 交通 INT8，V4L2 DMA-BUF 采集，单 Context 逐帧检测，可保存快照。 |
| `camera_live` | USB MJPEG，800×600，请求 30 FPS；也支持 JPEG 序列回放 | 交通 INT8，MPP 解码，三个独立单核 Context，ByteTrack/Kalman 跟踪。防抖和本地预览默认开启，推流默认关闭。 |
| `yolo_multi` | 1～4 路本地视频 | 交通 INT8，三个单核 Worker 轮询各路就绪帧。各路独立跟踪、预测和输出。 |

以下命令从仓库根目录运行，使用系统动态库。每次运行用 `mktemp -d` 新建结果目录。摄像头默认按 USB 设备的持久名称打开；用 `v4l2-ctl --list-devices` 查看本机节点，必要时给摄像头命令加 `--device /dev/videoN`。

**本地图片或视频**　将输入路径替换为自己的文件。视频输入对应的标注输出扩展名使用 `.avi`。

```bash
input=/path/to/image.jpg
out=$(mktemp -d)
env -u LD_LIBRARY_PATH -u LD_PRELOAD ./yolo_run/build-rk3588/yolo_run \
  --input "$input" --output "$out/results.jsonl" --annotated "$out/marked.jpg"
```

**YUYV 逐帧检测**　采集 150 帧，保存检测记录和一张快照。

```bash
out=$(mktemp -d)
env -u LD_LIBRARY_PATH -u LD_PRELOAD ./yolo_run/build-rk3588/camera_run \
  --io dmabuf --max-frames 150 --output "$out/results.jsonl" --snapshot "$out/frame.png"
```

**MJPEG 实时检测与跟踪**　以下命令适用于无桌面的板卡。结束时 JSONL 的 `summary` 给出 `frame_count`、`inference_count`、`accepted_results` 和各阶段丢帧计数。

```bash
out=$(mktemp -d)
env -u LD_LIBRARY_PATH -u LD_PRELOAD ./yolo_run/build-rk3588/camera_live \
  --preview off --max-frames 300 --output "$out/live.jsonl"
```

**四路本地视频**　可传入四个不同文件。以下示例把同一个文件独立解码四次，关闭预览与录像，运行 10 秒。整体统计写入 `channels/summary.json`，每路另有检测与画面记录。

```bash
clip=/path/to/video.mp4
out=$(mktemp -d)
env -u LD_LIBRARY_PATH -u LD_PRELOAD ./yolo_run/build-rk3588/yolo_multi \
  --input "$clip" --input "$clip" --input "$clip" --input "$clip" \
  --duration-seconds 10 --preview off --record off --output-dir "$out/channels"
```

### 模型、坐标与缓冲区

交通模型使用 800×608 输入、10 类标签。COCO 双输出模型使用 640×640 输入、80 类标签。切换到 COCO 时传 `--config yolo_run/config/camera_coco640.yaml`，默认加载 FP16。再传 `--model yolo_model_rknn/yolo26s_coco_int8.rknn` 可换用 INT8。配置内的路径相对配置文件解析，命令行模型路径相对当前目录解析。

以 COCO 为例，RGA 将 800×600 的画布缩放为 640×480，上下各补 80 行，填充值为 114。RKNN 接收 RGB UINT8 NHWC，归一化已在模型中配置。C++ 根据 tensor 属性识别 `boxes[1,4,8400]` 和 `scores[1,80,8400]`，执行两阶段 TopK，最多保留 300 项，再将坐标映回画布。此引擎使用双输出权重，单输出 e2e 权重的接口不同。

```text
YUYV  → V4L2 DMA-BUF → RGA → RGB DMA-BUF → RKNN → 检测框
MJPEG → V4L2 DMA-BUF → MPP 解码为 NV12 → 固定本帧防抖 ROI
                                          ├→ RGA 转 RGB → RKNN → 跟踪校正
                                          └→ RGA 转 BGR → 预测、画框、预览或编码
```

MJPEG 采集使用 4 个 V4L2 队列槽、8 块压缩输入 DMA-BUF。采集线程换入空闲缓冲后归还队列槽，MPP 继续读取已交给它的压缩包。解码后的 NV12 由检测与画面分支共享。RGA 同步写入各 Worker 自己的 RGB 分配，RKNN 导入该 RGB 的 FD，执行完成并释放帧租约后才能覆盖这块输入。CPU 通过 Cache Sync 访问映射内存。文件视频另有一次 BGR 上传，防抖小图分析、LK 和画框也会使用 CPU。

防抖默认从原图选取 720×540 ROI，缩放回 800×600。检测框、预览和推流共用这张画布。JSONL 逐帧保存坐标域和 ROI；传 `--stabilize off` 可恢复完整视野。

### 可选的局域网推流

MPP 将画框后的图像编码为 H.264，程序封装 MPEG-TS 发往本机 UDP 12340。独立 FFmpeg 发布器复制码流给 MediaMTX。使用推流时可在板上执行 `sh yolo_run/deploy/install_mediamtx.sh` 将固定版本 v1.21.2 下载到项目目录。以下三组命令分别在仓库根目录的三个终端运行。

```bash
# 终端一：启动 MediaMTX，填写观看端可达的板端 IP
read -r -p '板端局域网 IP：' board_ip
MTX_WEBRTCADDITIONALHOSTS="$board_ip" ./tools/mediamtx/v1.21.2/mediamtx yolo_run/deploy/mediamtx.yml
```

```bash
# 终端二：从本机 UDP 接收并发布
/usr/bin/python3 yolo_run/deploy/publish_rtsp.py --input-port 12340 --url rtsp://127.0.0.1:8554/camera1
```

```bash
# 终端三：启动摄像头与编码器
out=$(mktemp -d)
env -u LD_LIBRARY_PATH -u LD_PRELOAD ./yolo_run/build-rk3588/camera_live \
  --preview off --stream on --output "$out/stream.jsonl"
```

浏览器地址为 `http://<板端IP>:8889/camera1`，RTSP 播放器地址为 `rtsp://<板端IP>:8554/camera1`。三个前台进程可分别用 Ctrl+C 结束。

## 板端历史实测

### 真实摄像头与 30 FPS 配置

以下三组均使用 COCO640 INT8、NPU Core 0 和 1 各一个 Context、8 个解码 Surface，开启防抖，关闭预览与推流。每组运行三轮，每轮预热 10 秒、计量 60 秒。帧数为窗口合计，缺号来自 V4L2 sequence；延迟从采集计到各自分支完成，表中为各轮 P99 的中位数。

| 压缩包队列与 CPU 调度 | 有效采集 | 真实检测 | 跟踪校正 | 画面输出 | 设备缺号 | 检测 P99 | 画面 P99 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 单槽替换，默认调度 | 5,403 | 5,358 | 5,358 | 5,358 | 0 | 111.34 ms | 66.22 ms |
| 单槽 FIFO，默认调度 | 5,353 | 5,353 | 5,353 | 5,353 | 46 | 154.23 ms | 107.02 ms |
| 单槽 FIFO，整个进程绑定 A76 | 5,402 | 5,402 | 5,402 | 5,402 | 0 | 70.41 ms | 26.20 ms |

第一组有 45 个已采集的压缩包在解码前被替换。第二组处理了所有成功采集的帧，但设备序号出现了缺口。第三组三轮各处理 1,801、1,801、1,800 帧。下面用正式程序运行第三组配置，每轮单独保存业务 JSONL 和原始 Trace。

```bash
out=$(mktemp -d)
for rep in 1 2 3; do
  env -u LD_LIBRARY_PATH -u LD_PRELOAD taskset -c 4-7 ./yolo_run/build-rk3588/camera_live \
    --config yolo_run/config/camera_coco640.yaml --model yolo_model_rknn/yolo26s_coco_int8.rknn \
    --preview off --stream off --stabilize on --core-masks 1,2 \
    --queue-policy latest --infer-depth 1 --surface-count 8 --packet-fifo on \
    --warmup-seconds 10 --benchmark-seconds 60 \
    --trace-dir "$out/trace-$rep" --output "$out/run-$rep.jsonl" || break
done
```

### 同一输入下的精度类型、核数与队列

COCO640 回放矩阵使用同一段 MJPEG 序列，24 个对照项分别按 30 FPS 和不限速两种方式回放，每项重复三次，共 144 轮。下表选取 60 秒计量窗口。帧数为三轮合计，FPS 与 P99 为三轮中位数。

| 30 FPS 定速回放 | 真实检测 / 有效输入 | 画面输出 | 检测 FPS | 检测 P99 | 画面 P99 |
|---|---:|---:|---:|---:|---:|
| INT8，单 Context / Core 0 | 3,911 / 5,400 | 5,400 | 21.62 | 105.34 ms | 33.47 ms |
| INT8，单 Context / 三核 mask 7 | 3,663 / 5,400 | 5,400 | 20.25 | 108.17 ms | 33.85 ms |
| INT8，两个 Context / 各用一核 | 5,400 / 5,400 | 5,400 | 30.00 | 71.38 ms | 27.66 ms |
| INT8，三个 Context / 各用一核 | 5,400 / 5,400 | 5,400 | 30.00 | 72.07 ms | 28.04 ms |
| FP16，三个 Context / 各用一核 | 2,737 / 5,400 | 5,400 | 15.55 | 267.46 ms | 35.98 ms |

| 不限速回放，COCO640 INT8 | 检测 FPS | 检测 P99 | 画面 P99 |
|---|---:|---:|---:|
| 单 Context / Core 0 | 21.88 | 246.11 ms | 151.67 ms |
| 单 Context / 三核 mask 7 | 19.58 | 281.75 ms | 170.75 ms |
| 两个 Context / 各用一核 | 37.18 | 151.20 ms | 102.56 ms |
| 三个 Context / 各用一核 | 51.43 | 126.78 ms | 71.82 ms |

在三个单核 Context、30 FPS 的 INT8 回放中，仅改变**解码之后的检测等待队列**，四组均完成逐帧检测。深度 1 和 4 的 FIFO 检测 P99 分别为 77.85 和 72.09 ms；深度 4 丢最旧帧为 74.91 ms，单槽保留最新帧为 76.32 ms。此处调节的是送入推理的帧。实摄表调节的是送往 MPP 解码器的压缩包。

**十分钟连续回放**使用三个单核 Context。三个 600 秒定速窗口共 54,000 个有效输入，全部完成检测和画面输出。定速及不限速共六轮长测中，采样到的最低 NPU 频率为 1,000 MHz，SoC 峰值温度为 53.62 °C。

### 四路文件与摄像头推流

四路文件实测使用交通 INT8 800×608、软件解码、三个独立单核 Context，关闭预览和录像。下表每行对应一次板端运行。输出 FPS 包含跟踪预测和可能重复的源画面。

| 四路负载 | 各路真实检测 FPS | 合计检测 FPS | 每路输出 FPS |
|---|---|---:|---:|
| 早期基线，600 秒 | 13.17 / 13.18 / 13.31 / 13.02 | 52.67 | 约 30.11 |
| 原生 Kalman 预测版，10 秒 | 13.37 / 13.57 / 13.57 / 12.17 | 约 52.68 | 约 30.33 |
| 原生预测版，120 秒，每路请求上限 8 FPS | 7.26 / 7.25 / 7.28 / 7.27 | 约 29.06 | 约 30.13 |

另一组早期单路实摄使用交通 INT8、三 Context、MPP H.264 和 RTSP 发布，关闭预览，在防抖接入前计量 1,800 秒。输出、推理、编码各 53,988 帧，约 29.956 FPS。采集到编码完成的 P99 为 39 ms，设备序号有一次缺口。网络传输、播放器缓冲和屏幕呈现的延迟需在观看端另外测量。

### 交通模型检测精度

下面一组使用 BDD100K 的一条训练序列，1,210 个源帧中有 202 个标注时刻，按有普通真值的六类计算 101 点 AP。候选置信度下限为 0.001，忽略 crowd/distractor。模型为交通 10 类、800×608，数据限定于该片段。

| 模型与预处理 | 六类 mAP50 | 六类 mAP50–95 |
|---|---:|---:|
| 主机 FP32，OpenCV | 48.58% | 24.48% |
| 板端 FP16，RGA | 48.83% | 24.30% |
| 板端 INT8，RGA | 28.27% | 9.99% |

在这套输入与评测条件下，板端 FP16 与 INT8 的 mAP50 相差 20.56 个百分点。COCO640 回放测量的是吞吐、覆盖和延迟；其 80 类精度以及独立工业数据集上的表现仍待评测。
