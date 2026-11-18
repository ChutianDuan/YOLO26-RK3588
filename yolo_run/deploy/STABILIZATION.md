# camera_live 平移防抖

当前已接入的 USB 摄像头使用 800×600 MJPEG、30 FPS。防抖默认开启，CPU 估计背景平移，RGA 用同一帧 ROI 完成检测预处理与显示补偿，RKNN 继续使用现有模型和输入布局。

```bash
bash /home/yolo/scripts/dev.sh build
/home/yolo/yolo_run/build-rk3588/camera_live \
  --config /home/yolo/yolo_run/config/camera_coco640.yaml \
  --preview off --stabilize on --stabilize-alpha 0.15 \
  --max-frames 3000 --output /tmp/stabilization-NEW.jsonl
```

输出文件必须不存在。`--stabilize off` 使用完整原图；`--stabilize-alpha 1` 使用固定中心裁剪，适合隔离缩放与防抖的影响。α 必须在 `(0,1]`；默认 0.15，以 30 FPS 为参考，实际按帧时间间隔换算。

## 数据与坐标

每帧 MPP NV12 先由 RGA 缩小为 320×240 NV12，CPU 直接读取其 Y 平面。防抖器复用两个小型 DMA-BUF，上一帧灰度读取结束后才归还槽；RGA 的 `IM_SYNC` 和 CPU `START/END READ` 分别保证设备完成与缓存一致性。

光流估计完成后，把 ROI 与诊断值按值绑定到该帧。原 NV12 和 Surface 租约继续共享给检测及输出，两路分别在既有 `improcess()` 中融合 ROI 裁剪、缩放和颜色转换。只有小图分析增加一次 RGA 调用，不增加全尺寸中间 NV12，也不做全尺寸 CPU 变换。

防抖保留 720×540 视野，中心 ROI 为 `(40,30,720,540)`，最终输出仍是 800×600。原图位置与防抖画布位置的关系为：

```
view_x = (raw_x - roi.x) * 800 / 720
view_y = (raw_y - roi.y) * 600 / 540
raw_x  = roi.x + view_x * 720 / 800
raw_y  = roi.y + view_y * 540 / 600
```

RKNN letterbox 及后处理对应这个 800×600 虚拟画布；预览、推流和跟踪也使用该坐标。分析背景时，使用不晚于前帧的快照预测目标位置，再按前帧实际 ROI 逆映射，外扩框后屏蔽目标。

## 算法和失锁

- 4×3 网格，每格最多 16 个角点，合计不超过 192；点数组、掩码和两套金字塔按固定尺寸复用。
- LK 使用 21×21 窗口、最高第 2 层金字塔、最多 12 次迭代（epsilon=0.03）；前后向 LK 误差上限 1.5 个分析像素；平移中位数内点半径 1.5 像素，至少 12 个内点、65% 内点率，并覆盖四个网格及至少两行、两列。
- 位移乘以 2.5 还原到原图。补偿使用 `C=(1-alpha_dt)*(C+delta)`，等价于累计轨迹减去 EMA，避免存储不断增长的累计值。
- 补偿限制在 x ±40、y ±30；ROI 取最近偶数位置并限制在原图内。背景右移时 ROI 同向右移。
- 失锁时不累计不可靠位移，现有偏移按 EMA 平滑回中。首帧与超过 250 ms 的时间间隔重建参考；长间隔用一个正常帧步长回中，避免直接跳回中心。

这里只补偿平移。旋转、透视、滚动快门畸变与运动模糊不能通过平移裁剪消除；纹理不足或背景被遮挡时会降低有效估计率。

## JSONL v2

元信息 `schema_version=2`，增加 `stabilization` 参数。开启时 `bbox_coordinates=stabilized_800x600`，关闭时为 `source_800x600`。

每条 `frame` 和 `detection` 都记录自身帧的 `stabilization`：ROI、运动/补偿值、点数与内点率、参考重置/失锁、掩码快照时间以及小图/光流/总耗时。异步检测记录不借用最新输出帧的 ROI。`summary.stabilization` 汇总处理数、有效数、失锁数、限幅数及耗时分布。

## 验证

```bash
bash /home/yolo/scripts/dev.sh test
ctest --test-dir /home/yolo/test/build-rk3588 --output-on-failure -j1
# 下面两个目录必须是新目录；需要独占摄像头/MPP/RGA/NPU。
/home/yolo/test/build-rk3588/bin/stabilization_test --artifacts /tmp/stabilization-demo-NEW
/usr/bin/python3 /home/yolo/test/tools/benchmark_dma_capture.py \
  --compare-stabilization --output-dir /tmp/stabilization-benchmark-NEW
```

`stabilization_core` 验证平移、背景掩码、EMA、主动平移、失锁、时间间隔及边界；`stabilization_media` 使用真实 RGA 核对 ROI、非紧密 stride、颜色、letterbox 和灰度双槽。对照工具经过 MPP/RGA 处理可重复的抖动 JPEG，保存并排视频、截图、ROI 和 RMS 指标，并比较相同裁剪输入在单 Context 与三个单核 Context 的推理结果。

基准工具交替运行关闭/开启各三轮 300 帧，随后开启防抖连续 3000 输出帧。分别记录输出 FPS、实际检测 FPS、CPU、延迟、新增防抖耗时、FD、RSS 和温度；输出 FPS 不代表每帧检测。测试完成后退出自己的采集进程，不启动长期推流。

本次完整回归、对照视频、3000 帧性能和现场背景不足的验证边界见 [验收报告](../../test/reports/stabilization_validation.md)。
