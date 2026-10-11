# v21.2 AMP 数值稳定性评估

## 诊断结果

- D1：正式 autocast FP16 + fresh GradScaler(65536)。loss 0.15326565504074097；up/gate 权重梯度输出溢出，四个 LWC 参数梯度非有限。
- 原始 D2：关闭 autocast，学生计算 FP32，同一 65536 scale。loss 0.15303973853588104；LWC 梯度有限，但 QuantLinear 保留的原始未优化 HALF weight leaf 在末端 ToCopyBackward0 收到 Inf。因此这份结果只证明 LWC 路径有限。
- D2_FULL_FP32：在 QuantLinear 创建前使原始权重 leaf 也为 FP32，同一冻结数值输入/目标/参数、同一 65536 scale。全部 270 个观察到的 backward 事件梯度有限，leaf dtype 仅 FP32，LWC 梯度有限。
- D3：保持 D1 的 AMP forward，只以单位 scale backward。loss 与 D1 完全相同，全部观察到的 backward 梯度和 LWC 梯度有限。

所有对照仅执行一次 disposable backward，不调用实际 AdamW step。原始 D2 结果保留，不用后续对照掩盖其限制。全 FP32 输入和目标来自冻结 HALF 缓存值的提升，不能恢复此前 HALF 舍入损失。

## 1. 官方 commit 的 W4A16 默认是否使用 AMP？

使用。冻结官方 `feffe8ea87d80f7bb57b6e25e7cff9dc950fcc14` 的 main.py 默认 `deactive_amp=False`，W4 不触发 8<=bits<16 的自动关闭条件。block driver 将 qlayer.float()，训练 forward 使用 CUDA autocast，NativeScaler 每个 block 新建 GradScaler。默认路径是 AMP；这并不构成“所有配置强制要求 AMP”的声明。

来源是仓库保存的冻结源码及哈希，未依赖当前 upstream HEAD。

## 2. FP32 是否属于官方支持配置？

官方暴露 `--deactive_amp`，明确有 FP32 student traincast/cache 分支。不能据此认证当前 W4 Qwen port 的完整 FP32 校准已可运行。

需要区分：teacher forward 仍有无条件 autocast，GradScaler 包装仍保留；官方 driver 在最终传播前无条件 qlayer.half()（官方 quantize/omniquant.py:277），随后使用 traincast()（:284）。关闭 AMP 时 traincast=nullcontext、缓存为 FP32，这一 dtype 交接需要另行端到端验证。port 的对应位置为 :468/:475。该项目前只是源码风险，不是执行确认的 bug，也不是实际 D1 故障原因。

本轮 D2_FULL_FP32 只验证一个层的 forward/backward，不认证官方 flag 的完整流程，也不等于完全关闭 teacher AMP 的正式算法。

## 3. 改 FP32 后还能否称官方 OmniQuant baseline？

本轮不能给出该认证。如果下一轮选择官方暴露的数值选项，应先独立确认 W4、Qwen port、teacher/cache/final propagation 的 dtype 一致性和完整流程，再冻结一个明确标注精度设置的新版本。使用官方选项、未修改 LWC/optimizer/objective 的情况下，可以准确称“官方算法/移植、显式数值设置”；不能把它当作当前 v21.1 默认 AMP baseline 的无变化续跑。

从 block20 改精度也不能自动视为全程 FP32 baseline：此前学习状态和 prefix 均来自 AMP。是否允许这种续跑及如何公平比较，应由下一轮人工/Sol 决定。本轮没有作此决定。

## 4. calibration cost / memory 如何变化？

实测单样本、带 instrumentation 的 D1 peak allocated 为 632392192 bytes；全 FP32 为 726469120 bytes。两者只有一个 layer 的诊断状态，不能外推整模型峰值。D1/D2_FULL_FP32 的 instrumented timing 分别为 1.985632/1.696428 秒，包括 hooks/statistics 且没有足够重复，不能据此宣称 FP32 更快。

可确定的缓存结构成本：128×2048×896，两个 hidden buffers，HALF 总计 939524096 bytes（0.875 GiB），FP32 总计 1879048192 bytes（1.75 GiB），增加 0.875 GiB。其他 FP32 激活和工作区会增加，但本轮未测完整校准峰值或耗时；没有更新 24h gate，也没有声明新配置通过该 gate。

## 决策

已证明精度/loss scaling 敏感性；正式协议保持冻结与 STOP。没有调整 LR、gradient clipping、scaler 配置、epsilon、量化器参数化，也没有完成 layer21。
