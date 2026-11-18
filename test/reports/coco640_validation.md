# YOLO26 COCO 640×640 FP16 板端适配验收

日期：2026-11-18（Asia/Shanghai）。板卡 ARM64 RK3588，运行内核 6.6.89-kdev，RKNN Runtime 2.5.0 / 驱动 0.9.8，RGA 1.10.6，系统 OpenCV 4.10.0；继续使用原有 SDK 与 C++17 工具链，并发构建 4。

## 模型与实现

- 实际加载 `yolo_model_rknn/yolo26s_coco_fp16.rknn`，SHA256 为 `386cb69125156bc20bd5ee9dff75f7de472dfbed280b9a1d0bce94374f7cca45`，与转换报告一致。另两份 INT8/混合模型也只读核对了哈希，本次未执行它们的 COCO 板端推理。
- Runtime 查询确认两个输出为 FP16：`boxes[1,4,8400]`、`scores[1,80,8400]`；应用请求浮点输出。输入 RGB UINT8 NHWC，`pass_through=0`，不重复 `/255`。
- 查询到输入 stride 为 640×640，按模型原生容量要求分配 2,457,600 字节，未按紧密 UINT8 的 1,228,800 字节缩小 DMA-BUF。
- 800×600 经 RGA 缩放为 640×480，上下各补 80 行 RGB 114；使用模型提供的 COCO80 顺序，执行 CPU 两阶段 TopK=300、不加 NMS，坐标恢复到 800×600。
- 摄像头入口支持 `--model` > 显式 `--config` 中模型 > 未指定配置时的旧 INT8 默认模型。模型尺寸、类别和候选数不再固定为旧值，监控渲染与编码仍为 800×600。

## 编译与回归

正式工程与测试工程均编译通过。**4 项新增测试和 21 项既有测试，共 25 项全部通过**，分两批串行执行：

```bash
cd /home/yolo
bash scripts/dev.sh build
cd test
cmake --preset rk3588-release
cmake --build --preset rk3588-release --parallel 4
ctest --preset rk3588-release -R '^coco_(core|hardware|live)$' --output-on-failure
ctest --preset rk3588-release -E '^coco_(core|hardware|live)$' --output-on-failure
```

- `coco_core`：COCO80 标签、阈值数量、交换输出顺序、8400 候选、旧 9975 布局、缩放补边、坐标反变换/边界与无 NMS TopK。
- `coco_hardware`：真实 RGA 的 NV12 800×600、有额外行/列 stride、RGB 640×640、颜色与补边、帧租约及缓冲复用；同一 RGB 的真实 RKNN CPU/FD 输入和重复推理结果一致；公交车夹具检测到 person、bus。
- `coco_live`：MPP MJPEG 回放、三个独立 Worker、源坐标、配置选中新 FP16，显式不兼容模型覆盖被正确拒绝。
- `coco_stream`：真实摄像头、新 FP16 配置、H.264/RTSP 解码、WebRTC WHEP 协商、慢客户端/编码、编码错误隔离，以及测试自建发布进程的恢复。
- 既有回归包括旧模型的图片/视频/多路、DMA/CPU 等价、摄像头、MPP/RGA、推流和故障清理。

## 实机证据

原始记录位于 [本次结果目录](../results/coco640.1QhtaMMH/)，关键文件为
[验证摘要](../results/coco640.1QhtaMMH/verification.json)、
[22 项回归日志](../results/coco640.1QhtaMMH/regression.log)、
[单图结果](../results/coco640.1QhtaMMH/bus.jsonl)、
[单图带框画面](../results/coco640.1QhtaMMH/bus.jpg)、
[YUYV 摄像头结果](../results/coco640.1QhtaMMH/camera_yuyv.jsonl)。

`camera_run` 另用新配置完成 10 帧真实 YUYV / DMA-BUF 推理，未发生应用层 YUYV 上传；输出元数据是 640×640、80 类、上下补 80，截图尺寸 640×640，转换输出字节数 1,228,800。此为短功能测试，不用其冷启动帧率判断长期性能。

新模型推流的完整记录见 [COCO 推流报告](../results/coco_stream/integration_1794963041023109741/report.json)。约 24.49 秒主摄像头运行中输出并编码 729 帧，输出约 29.77 FPS，实际执行 397 次检测（按该运行时段折算约 16.21 次/秒）；输出含跟踪预测帧。RTSP 解码确认 H.264 Constrained Baseline、800×600、无 B 帧。WebRTC 页面 HTTP 和 WHEP 协商通过。测试发送 SIGINT 后正常结束，测试创建的进程均已清理。

## 验证边界

已验证编译、板端执行、接口/像素/坐标和同输入路径一致性。未执行完整 COCO 精度评估；同输入 CPU/FD 等价不能代替 ONNX 与 RKNN 的精度比较。新模型尚未完成长时间性能/温度稳定性测量，未测电脑浏览器实际媒体播放及观看端延迟。短测输出帧率不能解释为同等检测频率，采集到编码完成耗时不能解释为观看延迟。

没有修改 SDK、内核、驱动、网络配置或系统服务。启动新模型及电脑浏览器观看的完整步骤见 [COCO640 部署说明](../../yolo_run/deploy/COCO640.md)。
