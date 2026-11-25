# 单路 H.264、RTSP 与 WebRTC

核对日期：2026-11-18（Asia/Shanghai）。`camera_live` 默认启用平移防抖，推流与检测/跟踪采用同一防抖画布；使用 `--stabilize off` 输出完整原图。坐标和 ROI 见 [防抖说明](STABILIZATION.md)。下文的三十分钟数字是防抖接入前版本的历史测量。

`camera_live` 在当前 MJPEG/MPP 实时入口上增加可选编码。第一版固定为 800×600、名义 30 FPS、无音频，保留现有模型、检测与跟踪调度。关闭推流时不创建编码 Worker 或编码 DMA-BUF 池。

```text
V4L2 MJPEG → MPP 解码 NV12 DMA-BUF
  ├─ RGA → RKNN → 跟踪快照
  └─ RGA → BGR → CPU 绘制框/ID → RGA → NV12 DMA-BUF → MPP H.264
       → MPEG-TS / loopback UDP → 独立 FFmpeg / stream copy
       → RTSP TCP → 独立 MediaMTX → RTSP 或 WebRTC 客户端
```

## 构建与启动

复用 `/opt/rockchip`、`/opt/rknn` SDK 及系统 OpenCV、FFmpeg 开发库；不使用 Conda C++ 库，不需要重新编译 FFmpeg。系统 FFmpeg 没有 `h264_rkmpp` 编码器；H.264 编码由本工程直接调用 MPP 完成。

```bash
cd /home/yolo/yolo_run
cmake --preset rk3588-release
cmake --build --preset rk3588-release --parallel 4
```

MediaMTX 使用项目内官方 v1.21.2 ARM64 程序，独立于 CMake。当前已下载并核验；新设备可以手动执行：

```bash
/bin/sh deploy/install_mediamtx.sh
```

脚本同时核验发布方 `checksums.sha256` 与固定 SHA-256：
`180a235374a548d9133cdd5e20f5dbd8b01317be1c86e0d7e66df585dc42ed6f`。
官方下载：[v1.21.2](https://github.com/bluenviron/mediamtx/releases/tag/v1.21.2)。不注册服务，不配置开机启动。

以下三个进程分别在独立终端手动启动。JSONL 必须使用新文件名。

```bash
cd /home/yolo/yolo_run
read -r -p '输入观看端可访问的板端 IP：' board_ip
MTX_WEBRTCADDITIONALHOSTS="$board_ip" \
  ../tools/mediamtx/v1.21.2/mediamtx deploy/mediamtx.yml
```

```bash
cd /home/yolo/yolo_run
/usr/bin/python3 deploy/publish_rtsp.py --input-port 12340 \
  --url rtsp://127.0.0.1:8554/camera1
```

```bash
cd /home/yolo/yolo_run
./build-rk3588/camera_live --preview off --decoder mpp \
  --stream on --stream-port 12340 --stream-bitrate 2000000 \
  --output camera1-new.jsonl
```

`--stream` 默认 `off`；端口为 1～65535，码率为 64000～14000000 bps。解码只支持 MPP，`--decoder opencv` 会被拒绝。码率默认 2 Mbps，最大码率同时限制到 Baseline Level 3.1 的 14 Mbps。

可用 `--replay manifest.jsonl` 替换摄像头进行独立重放调试。生产入口不接受测试故障参数。

## 观看与协议

| 用途 | 地址或端口 |
|---|---|
| Mac VLC/ffplay | `rtsp://<板端IP>:8554/camera1` |
| 浏览器内置 WebRTC 页面 | `http://<板端IP>:8889/camera1` |
| WebRTC WHEP 协商 | `/camera1/whep`，HTTP TCP 8889 |
| WebRTC 媒体 | UDP 8189 |
| C++ 到 FFmpeg | 仅本机 `127.0.0.1:12340` UDP MPEG-TS |

板端地址变化时修改启动环境变量和观看地址。MediaMTX 同时从网卡收集 ICE 地址，不需要在 C++ 中写入 IP。发布权限只允许本机向 `/camera1` 发布；观看无需账号。RTMP/HLS/SRT/MoQ、录制和管理接口关闭。此配置面向局域网，跨网、TURN 和认证部署不在第一版验收范围内。

浏览器通过 WebRTC 观看；它不会直接打开 RTSP。H.264 为 Constrained Baseline、Level 3.1、无 B 帧，每个 IDR 带 SPS/PPS。GOP 上限 30，距上次关键帧满一秒也请求 IDR。发布器启动在 GOP 中途时，等待下一个关键帧期间可能看到短暂的 PPS 解析错误。

板端检查：

```bash
/usr/bin/ffprobe -v error -rtsp_transport tcp \
  -show_entries stream=codec_name,profile,level,width,height,has_b_frames,color_range,color_space \
  -of json rtsp://127.0.0.1:8554/camera1
/usr/bin/ffmpeg -nostdin -v error -rtsp_transport tcp \
  -i rtsp://127.0.0.1:8554/camera1 -frames:v 60 -f null -
```

## 所有权、同步与背压

Renderer 的 BGR `Mat` 是借用视图，不能跨帧留给编码线程。输出线程绘制完成、`Renderer::finish()` 结束 CPU RW Cache Sync 后，在下一次 render 前同步 RGA 转换到编码池。

编码池固定三槽，NV12 可见尺寸 800×600、stride 800×608，UV 起点为 `800*608`，padding 初始化为 limited 黑色。目标采用 BT.601 limited，MPP prep/frame/VUI 和 MPEG-TS 元信息保持一致。源 MJPEG NV12 原有的 full-range 转换仍由 Renderer 处理。

每槽具有独立帧租约；MPP 的 FD 注册缓存仅延长 allocation 存活期，不占用下一帧的租约。唯一编码 Worker 从容量一的 latest 队列取帧。新提交替换旧等待项；无空闲槽时丢弃本次推流提交，不等待池腾空。MPP 输入在全部分片收齐到 EOI 前不可复用。输出 DMA-BUF 和压缩码流容量各固定为 2 MiB，超出时禁用通道。

每帧 PTS 从源单调时间相对首帧计算并换算为 90 kHz；无 B 帧时 DTS=PTS。丢帧不会压缩时间间隔。MPEG-TS 通过非阻塞、未连接的 loopback UDP socket 发送，每包最多 1316 字节；没有接收方时不会等待。已报告的发送失败会丢弃当次数据并请求下一 IDR；UDP 接收端丢包无法由发送成功判定。

实采 MJPEG 的输入 DMA-BUF 直接交给 MPP。CPU 仍会清零包尾、分析防抖小图、画框和读取编码码流，发布还涉及进程间传输。

## 故障与统计

发布脚本始终最多管理一个 FFmpeg 子进程；使用 `-c:v copy`，不重编码。异常退出后按 1、2、4、8 秒退避重启，健康运行三十秒后重置退避；输出时间十秒不前进时终止该子进程并重试。其 SIGINT/SIGTERM 只清理自己的子进程，不操作 YOLO 或 MediaMTX。

MediaMTX 停止、发布器缺席或观看端变慢时，推理及 JSONL 继续。MediaMTX 需手动重新启动，FFmpeg 会自动重新发布；播放器可能需要重新打开流。

编码线程异常会设置 `stream.status=error` 并禁用推流，推理继续，整个 `camera_live` 的退出状态仍表示原检测流水线状态。使用方应同时检查 `summary.stream.status`。MPP 超时不等于 DMA 完成：先 reset/destroy Context，成功才释放导入和帧租约；无法确认时保留该固定资源集合至进程退出，禁止继续复用。同步 RGA 返回无法确认完成的错误时，沿用工程现有的进程终止策略，避免覆盖设备仍在访问的 BGR/NV12。

`frame.stream_submitted` 仅表示进入编码等待槽；之后仍可能被新等待帧替换。周期 `resource.stream` 和 `summary.stream` 包括实际编码数、关键帧数、池/队列丢帧、关闭舍弃、发送字节/错误，以及转换、编码、采集到编码完成的耗时分位数。

`output_fps` 是本地输出速率，`stream.encoded_fps` 是编码速率。`capture_to_encoded` 不包含发布、网络和客户端缓冲，不能当作观看端延迟。

## 验证与长测

保持 MediaMTX 和发布器运行，使用“构建与启动”中的摄像头命令实测。按 Ctrl+C 正常结束后，检查 JSONL 最后一条 `summary` 的 `stream.status`、实际编码数、丢帧及发送错误；观看端可同时执行上面的 `ffprobe` 和 60 帧解码命令。

在防抖接入前版本，真实 USB 摄像头以交通 INT8、split、关闭预览运行 1,800 秒，输出、推理和编码各 53,988 帧，编码 P99 为 3 ms，采集到编码完成 P99 为 39 ms。编码池丢帧、等待槽替换和发送错误各为 0，设备序号有一次缺口。

Mac VLC/ffplay 和 Safari/Chrome 的真实播放需要在客户端另外验收；WHEP SDP 成功不代表媒体已经在浏览器播放。玻璃到玻璃延迟可同时拍摄现场计时器与 Mac 画面测量，不能用不同主机未经同步的时钟直接相减。
