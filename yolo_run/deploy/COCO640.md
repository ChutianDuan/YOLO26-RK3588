# YOLO26 COCO 640×640 FP16 与电脑浏览器监控

使用 [camera_coco640.yaml](../config/camera_coco640.yaml)，加载已上传的
`yolo26s_coco_fp16.rknn` 和 `yolo26s_coco_classes.json`。模型契约以
[转换说明](../../yolo_model_rknn/yolo26s_coco_rknn_README.md) 为准：Toolkit2 / Runtime 2.5.0、RK3588、batch 1。

## 输入、输出与内存

- 摄像头继续采集 800×600；MJPEG 请求 30 FPS，YUYV 请求 15 FPS。
- `camera_run` 或 `camera_live --stabilize off`：RGA 读取完整 800×600，缩放为 640×480，并在 RGB 640×640 中上下各补 80 行 114。`camera_live` 默认防抖开启：读取本帧 720×540 ROI，再映射到相同的 800×600 虚拟画布/640×480 模型有效区域，会裁去部分边缘视野。
- 输入是 RGB UINT8 NHWC，`pass_through=0`。模型已配置 mean=0、std=255，应用不再除以 255。
- 输出请求 `want_float=1`，根据查询到的属性识别 `boxes[1,4,8400]` 和 `scores[1,80,8400]`，不依赖输出顺序。boxes 已解码为模型坐标 xyxy，scores 已做 sigmoid。
- C++ 执行两阶段 TopK=300，不加 NMS；检测框恢复为 800×600 输入画布坐标：`x=x_model/0.8`、`y=(y_model-80)/0.8`，再限制到 800×600 边界。
- 解码 NV12、推理 RGB、编码 NV12 仍是独立缓冲。每个推理 Worker 独占一个 Context、预处理器及 RGB DMA-BUF，RGA 同步完成后才交给 NPU；最后一个帧租约释放后才覆盖 RGB。CPU 首次初始化补边，后续复用；不宣称端到端零拷贝。
- 浏览器显示 800×600 画布、COCO 类别和跟踪 ID；默认是防抖画布，关闭防抖才是完整原图。640×640 补边画布只用于推理。JSONL v2 记录坐标域和逐帧 ROI，逆映射见 [防抖说明](STABILIZATION.md)。

`--model` 优先于显式 `--config` 中的模型路径。摄像头入口不传 `--config` 时仍沿用旧的 800×608 INT8 默认模型；旧配置和标签继续保留。切换新模型时须同时指定这份 640 配置，不能只换模型路径而保留旧尺寸、旧标签。

## 构建与单张图验证

```bash
cd /home/yolo
bash scripts/dev.sh build
run_dir=$(mktemp -d /home/yolo/test/results/coco640.XXXXXXXX)
env -u LD_LIBRARY_PATH -u LD_PRELOAD ./yolo_run/build-rk3588/yolo_run \
  --config /home/yolo/yolo_run/config/camera_coco640.yaml \
  --input /opt/rockchip/src/rknn_model_zoo/examples/yolo26/model/bus.jpg \
  --output "$run_dir/bus.jsonl" --annotated "$run_dir/bus.jpg"
```

无图形桌面时验证摄像头，程序处理 300 个输出帧后退出，输出写入全新文件：

```bash
cd /home/yolo/yolo_run
env -u LD_LIBRARY_PATH -u LD_PRELOAD ./build-rk3588/camera_live \
  --config /home/yolo/yolo_run/config/camera_coco640.yaml \
  --preview off --decoder mpp --max-frames 300
```

默认设备为 `/dev/v4l/by-id/usb-Generic_USB_Camera_200901010001-video-index0`。
需要换摄像头时用 `--device` 指定实际采集节点，并核对其支持的格式。

## 在本地电脑浏览器观看

在板端三个终端分别手动运行。运行 `hostname -I`，选择观看端可以访问的板端局域网地址。复用已有 MediaMTX，无需安装系统服务。

终端 A：

```bash
cd /home/yolo/yolo_run
read -r -p '输入观看端可访问的板端 IP：' board_ip
MTX_WEBRTCADDITIONALHOSTS="$board_ip" \
  ../tools/mediamtx/v1.21.2/mediamtx deploy/mediamtx.yml
```

终端 B：

```bash
cd /home/yolo/yolo_run
/usr/bin/python3 deploy/publish_rtsp.py \
  --input-port 12340 --url rtsp://127.0.0.1:8554/camera1
```

终端 C：

```bash
cd /home/yolo/yolo_run
env -u LD_LIBRARY_PATH -u LD_PRELOAD ./build-rk3588/camera_live \
  --config /home/yolo/yolo_run/config/camera_coco640.yaml \
  --preview off --decoder mpp --stream on \
  --stream-port 12340 --stream-bitrate 2000000
```

本地电脑打开 `http://<板端IP>:8889/camera1`。
网页通过 WebRTC 接收视频，使用 TCP 8889 和 UDP 8189；RTSP 播放器也可以打开
`rtsp://<板端IP>:8554/camera1`。三个自建进程分别按 Ctrl+C 退出。
协议、缓冲与故障处理详见 [推流说明](STREAMING.md)。

## 回归与验证边界

```bash
cd /home/yolo/test
cmake --preset rk3588-release
cmake --build --preset rk3588-release --parallel 4
ctest --preset rk3588-release -R '^coco_(core|hardware|live|stream)$' --output-on-failure
```

`coco_core` 无需设备；`coco_hardware` 验证 RGA 缩放、RGB/补边、stride、租约，以及真实 FP16 同一 RGB 的 CPU/FD 输入等价性；`coco_live` 使用 MJPEG 回放验证 MPP、三个 Worker、源坐标与配置/命令行模型优先级；`coco_stream` 用真实摄像头和 FP16 模型验证 H.264、RTSP 解码、WebRTC 协商及自建进程故障恢复。设备测试串行执行，推流测试仅启动并清理自身进程。

日常 `bash scripts/dev.sh test` 按 `cpu` 标签运行九项纯软件回归，已包含 `coco_core`；COCO 硬件验证使用以上命令。
输出 FPS 包含跟踪预测帧，不能当作实际检测 FPS；编码完成耗时不包含网络和浏览器缓冲。
完整 COCO 精度、长测和电脑端实际观看延迟需另行测量。本次板端证据见
[COCO640 验证报告](../../test/reports/coco640_validation.md)。
