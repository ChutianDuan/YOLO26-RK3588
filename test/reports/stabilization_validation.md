# camera_live 平移防抖验收

时间：2026-11-18 10:42:03（Asia/Shanghai）。证据目录：[本次运行](/home/yolo/test/results/stabilization_20261118_101652)。

## 实现与接口

- `camera_live` 默认启用 CPU 背景角点/LK + EMA + RGA ROI 补偿，提供 `--stabilize on|off`、`--stabilize-alpha 0.15`。当前采集仍为 800×600/30 FPS，固定裁剪 720×540、缩放到 800×600。
- 分流前计算并固定每帧 ROI；检测预处理和渲染在原有 RGA 调用中融合裁剪/缩放/颜色转换。额外图像分配仅两个 320×240 NV12 DMA-BUF（225 KiB），CPU 读取其 Y 平面。
- 最大 192 个角点，4×3 网格，前后向 LK、空间覆盖和中位数内点校验；目标框按前帧实际 ROI 逆映射后屏蔽。LK 21×21、最高第 2 层、最多 12 次迭代、epsilon 0.03。
- `C=(1-alpha_dt)*(C+delta)`，时间间隔自适应；补偿限幅 x±40、y±30，ROI 就近偶数对齐。估计失败时零位移输入、平滑回中，长间隔重建参考。
- JSONL `schema_version=2`：检测/跟踪/输出均使用防抖后的 800×600 画布，每帧和每条异步检测保存自身 ROI、质量与耗时，`bbox_coordinates` 区分 `stabilized_800x600` / `source_800x600`。
- 原 RKNN 模型/输入布局、DMA 输入接口、SDK 和系统环境保持；正式程序没有全尺寸 CPU warp 或 RKNN CPU 输入分支。

源代码：[防抖模块](/home/yolo/yolo_run/video/frame_stabilizer.cpp)、[流水线](/home/yolo/yolo_run/camera/live_pipeline.cpp)、[ROI 预处理](/home/yolo/yolo_run/image/image_processing.cpp)。运行说明：[防抖指南](../../yolo_run/deploy/STABILIZATION.md)。

## 编译与正确性

- `bash scripts/dev.sh build`、`bash scripts/dev.sh test` 成功；C++17、系统 GCC/OpenCV/libstdc++、现有 RKNN/RGA/MPP，构建并发 4。
- 完整 CTest **28/28 通过**，耗时 260.98 秒：[完整日志](/home/yolo/test/results/stabilization_20261118_101652/ctest-full.log)。
- 新测试覆盖平移符号/尺度、EMA、持续平移跟随、前景屏蔽、限幅和偶数对齐、低纹理失锁/恢复、时间间隔、真实 RGA 非紧密 stride 裁剪、RGB/letterbox、检测/渲染 ROI 一致，以及单 Context / 三个单核 Context 的裁剪输入推理结果一致性。
- 实时 CLI 回归覆盖默认开启、显式关闭、固定中心裁剪、无效参数、掩码时间因果性、同一帧异步检测/输出 ROI、一致的停止和异常收尾。
- 回归发现新增统计项使实时函数栈帧过大，尺寸不匹配模型抛出异常时崩溃。固定容量直方图改为启动时堆分配，避免大型栈帧；相同非法模型已正常以状态 1 报错退出，COCO 回归及完整回归通过。最初失败与修复后的日志均保留。

## 可重复抑抖效果

经过真实 MPP 解码、RGA 下采样/裁剪/颜色转换的 180 帧可重复抖动回放。两侧都使用 90% 裁剪，左侧固定中心，右侧动态 ROI。背景位置 RMS 根据已知输入平移与实际 ROI 计算；独立 RGA 像素测试核对 ROI 映射，避免仅验证数值滤波。

| 方向 | 固定裁剪 RMS（输出像素） | 防抖 RMS | 降低 |
|---|---:|---:|---:|
| 水平 | 10.54 | 1.46 | 86.1% |
| 垂直 | 6.95 | 1.28 | 81.5% |

超过“高频抖动 RMS 至少降低 50%”目标。对照视频为 **1600×600、30 FPS、180 帧**，左右输入为同一帧；源图片原有黑色侧边已从测试素材中去除。

- [并排视频](/home/yolo/test/results/stabilization_20261118_101652/comparison-clean/comparison.avi)
- [对照截图](/home/yolo/test/results/stabilization_20261118_101652/comparison-clean/comparison.jpg)
- [ROI / 位移逐帧记录](/home/yolo/test/results/stabilization_20261118_101652/comparison-clean/roi.jsonl)
- [抑抖指标](/home/yolo/test/results/stabilization_20261118_101652/comparison-clean/report.json)

## 真实摄像头性能与稳定性

同一 COCO80 FP16/640×640 配置、split 三 Worker、无预览/推流，防抖关闭与开启交替各运行三次 300 输出帧，取中位数：[原始报告](/home/yolo/test/results/stabilization_20261118_101652/benchmark/report.json)。

| 指标 | 关闭 | 开启 |
|---|---:|---:|
| 输出 FPS | 29.64 | 29.59 |
| 实际检测 FPS | 13.99 | 14.01 |
| 输出延迟 P95（ms） | 36 | 49 |
| 进程 CPU（100%=一个逻辑核） | 208.29% | 246.19% |

CPU 增量约 0.38 个逻辑核。三轮开启防抖的附加耗时 P95 为 13～15 ms。输出 FPS 保持超过 29，检测吞吐基本持平；关闭与开启的视野不同，不把检测数量差异解释为模型精度变化。

开启防抖连续运行 **3000 输出帧**：

- 输出 **29.98 FPS**，实际检测 **14.31 FPS**，共 1432 次检测。
- 防抖平均 10.49 ms、P95 14 ms；其中灰度 RGA 平均 0.54 ms。
- 第 30 帧起的 100 个资源样本 FD 均为 **53**；RSS 271.47 → 271.47 MiB；最高温度 46.23 ℃。
- 应用层压缩数据复制 0 B，输入包池和解码输出池耗尽均为 0，基准期间没有新增 UVC/MPP/RGA/RKNPU 错误日志。

### 当前实机画面的验证边界

本次实际摄像头画面被近距离人物大面积占据。一个典型检测框约为 `(252,1)–(797,593)`；外扩并屏蔽后，剩余背景点及空间覆盖不足。

3000 帧中，**17 帧有效估计、2982 帧失锁回中、1 次参考重置**，限幅 0 次。因此该实机长测证明的是遮挡条件下的降级路径、吞吐与稳定性，**不能用来宣称当前现场画面已持续有效防抖**。抑抖量化效果来自有纹理背景的可重复硬件回放；现场需要保留足够静态背景。旋转、透视、滚动快门和运动模糊不属于本版平移补偿能力。

## 交付状态

全部测试子进程已退出，摄像头、RGA、MPP、NPU 设备无本次测试进程占用；长期推流保持关闭。原有空闲 MediaMTX 进程保持原状态。未更改内核、驱动、SDK、启动配置、系统服务设置或设备权限。
