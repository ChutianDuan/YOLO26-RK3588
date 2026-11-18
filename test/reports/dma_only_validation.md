# DMA-BUF 单路径与 MJPEG 采集拷贝删除验收

验证日期：2026-11-18（Asia/Shanghai）。原始证据：[本次运行目录](/home/yolo/test/results/dma_only_20261118_092834)，包括 [26 项回归日志](/home/yolo/test/results/dma_only_20261118_092834/ctest-full.log)、[交替短测和长测报告](/home/yolo/test/results/dma_only_20261118_092834/benchmark-final/report.json)、[恢复后的推流进程](/home/yolo/test/results/dma_only_20261118_092834/resumed/process.json)。

## 实现结果

- 生产预处理统一为 RGA，TensorInput 只携带必填 DMA 描述；删除 CPU RGB、OpenCV MJPEG 解码和 RKNN `rknn_inputs_set` 运行分支。CPU 颜色/推理参考移入测试代码。
- YUYV 删除 MMAP/copy/auto 回退和强制上传；MJPEG 使用 4 个 V4L2 队列槽、8 块预分配 DMA-BUF，DQBUF 后换入空闲缓冲，原缓冲交给 MPP，不复制压缩数据。回放仍保留 4 槽。
- 保留包尾 256 字节清零、Cache Sync、容量/格式边界、帧租约和硬件完成确认。FD/稳定布局校验集中到新导入，缓存命中时不再逐帧 fcntl/fstat。
- 保持 `--preprocess rga`、`--decoder mpp`、`--io dmabuf` 和 RGA 配置写法，删除的 CPU/copy/auto 值明确失败。
- 相对本轮开始时快照，生产 C++ 源码/头文件净减少 126 行。已有未提交的 COCO 后处理、配置和部署说明内容保留。

## 构建与结果正确性

- `bash scripts/dev.sh build` 和 `bash scripts/dev.sh test` 成功，原六项日常回归通过。
- 完整 CTest **26/26 通过**，总耗时 241.25 秒。包括真实 MJPEG DMA 轮换、池压力、损坏帧丢弃、REQBUFS/QBUF 初始化失败清理、保留包存活、YUYV/MPP/RGA 像素参考、FP16/INT8 与 COCO80 的 CPU参考/FD 推理等价、租约复用、EOF/SIGINT、真实摄像头 RTSP 与 COCO 推流、超时清理和四路视频。
- 测试中的 SDK 全局 FD 在首次导入时延迟创建：故障清理检查先预热 SDK；稳定性比较使用第 30 帧起的资源采样，同时保留启动样本。原始首次试验日志仍保留在证据目录。
- `ldd` 确认复用现有 RKNN 2.5.0、librga、MPP，OpenCV/libstdc++ 来自系统；生产 camera_live 不再引用 `rknn_inputs_set` 或 OpenCV `imdecode`。

## 同配置交替短测

真实 USB 摄像头，800×600 MJPEG/30 FPS；COCO80 FP16 640×640，split 三 Worker，无预览、无推流。保存旧二进制后，新旧各运行 300 帧、三轮交替取中位数。

| 指标 | 改动前 | 改动后 |
|---|---:|---:|
| 输出 FPS | 29.66 | 29.73 |
| 实际检测 FPS | 16.17 | 16.20 |
| 输出延迟 P95（ms） | 35 | 35 |
| 进程 CPU（100%=一个逻辑核） | 187.53% | 187.90% |
| 应用层压缩数据复制 / 300 输出帧 | 约 8.5 MB | 0 B |

本次短测未体现明显 CPU 降幅或检测吞吐提升；确认的收益是删除应用层压缩包搬运与 CPU 双路径。温度、场景和调度仍会影响短测，不能据此宣称性能大幅改善。

## 新版 3000 帧连续运行

- 完成 3000 输出帧、1627 次实际检测；输出 29.98 FPS，实际检测 16.26 FPS。LatestSlot 维持原有的最新帧策略，输出 FPS 不代表每帧都执行检测。
- `compressed_copy_bytes=0`，压缩输入池和解码输出池耗尽均为 0；损坏包、时间戳丢弃均为 0。驱动序号间隙记录为 2，不宣称全链路无丢帧。
- 第 30 帧起的 100 个样本 FD 均为 51；RSS 266.50 → 266.50 MiB，最高温度 49.92 ℃。
- 内核日志可读取，基准与连续运行期间未发现新增 UVC/MPP/RGA/RKNPU 错误。

## 推流恢复与边界

经本次弹窗授权，旧进程 231721 通过 SIGINT 正常停止；验证结束后，新程序按原 COCO640 配置、UDP 12340、2 Mbps、无预览参数恢复，PID **240358**，使用新日志目录，不覆盖原结果。已确认持续输出、编码状态 running、发送错误为 0。

本次未更改系统、内核、驱动、SDK、启动配置或系统服务设置。文件图片/视频解码与 BGR 上传、画框、CPU LK 和压缩编码结果读取仍保留必要 CPU 访问；不宣称端到端零拷贝。

复现构建、回归与 3000 帧命令见 [测试指南](../README.md#像素摄像头与防抖)。
