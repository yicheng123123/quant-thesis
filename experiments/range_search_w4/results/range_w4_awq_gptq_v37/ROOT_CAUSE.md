# v37 根因与影响

## 范围与来源

本审计只读取并复核 v36 已保存的 S0 候选、P1 held 输出和 S1/P2 search capture。没有读取 S1/P3，没有运行 S1 selector、S1 GPTQ 候选、PPL 或整模型量化。v36 原始报告、缓存、checkpoint 和协议均保留不动。新增诊断及修复结果均写在 v37。

## 问题 1：G00/down/alpha=1 缺少直接输出评分 parity 字段

`candidate_and_parity_down` 的恢复分支在从进度文件读回 native alpha=1 权重后重新比较了 native 参数和 GPTQ group entry，但跳过了只位于后续候选循环 alpha=1 分支中的 `direct_row_output_scores` 计算。因此 v36 的 G00/down parity 记录少一项证据；候选权重、量化参数、score bank 和实际选择过程没有因此改变。

v37 使用已保存的相同 P0 输入、pristine down 权重、alpha=1 候选与直接评分公式重算全部 896 个输出行。保存的 alpha=1 native parity 与候选 W、scale、zero、g_idx 及 group entries 完全一致；该候选直接评分与 v36 逐行 alpha=1 score 完全一致，最大绝对差为 0，0/896 行不一致。因此缺失的是 parity 记录字段，不是新的候选或质量数值。

账本记录曾执行一次 fresh native alpha=1 GPTQ parity 尝试（7.130 秒），但它没有成为最终 parity 证据；最终修复依赖可核验的 v36 native parity 张量和该张量上的直接分数重算。这个尝试仍完整计入预算。

## 问题 2：P1/P2 postnorm 在 S0 G11 坐标下捕获

v36 每个 split 的候选输入准备循环按 G00、G01、G10、G11 顺序写入 L12 `post_attention_layernorm.weight`。G11 使用 `base_norm / D`，循环结束后没有把 norm 恢复为 pristine。紧接着调用的 `prepare_prefix_inputs` 没有显式设定 norm，也没有把 norm/checkpoint 状态绑定到 capture sidecar；缓存命中只核验文章 token hash、文件大小和文件自身 hash。P1 因而以 S0 的 `base_norm / D` 捕获，held evaluation 完成后又恢复到进入时的同一缩放状态，后续 S1/P2 也以此状态捕获。

v37 的逐元素复放定位了这一点：

- S0/P0 的旧 postnorm 与 pristine norm 对相同 prenorm 的复放完全一致。
- S0/P1 的旧 postnorm 与 `norm(base_norm / D_S0, prenorm)` **14,680,064 / 14,680,064** 个元素完全一致；与 pristine norm 复放不一致，最大绝对差 8.40625，平均绝对差 0.1966383。
- S1/P2 的旧 postnorm 同样与 `norm(base_norm / D_S0, prenorm)` 完全一致；与 pristine norm 复放不一致，最大绝对差 8.296875，平均绝对差 0.1964157。
- P2 的 hidden、prenorm、文章/token hash 与独立干净进程完全相同；prefix checkpoint、base/prefix norm 参数 hash、FP16 dtype 和 epsilon 一致。hook 的 pre/post alias 在 block 前后均未变化。因此这不是 token、模型 revision 或 tensor alias 的差异。
- 在 clean prefix 上按 pristine norm 复捕获 P2 后，postnorm 对 norm(prenorm) 逐元素完全一致。

**影响：** S0/P0 的搜索数据、D/alpha 选择、锁定候选、GPTQ 输入与量化张量不受这个 capture 次序影响。S0/P1 的旧 reference 输入错误，导致 v36 保存的四臂 held MSE 无效。四臂保存的实际 full-block 输出本身在相应坐标下正确；v37 使用锁定张量重跑完整 L12 block，四臂输出都与 v36 保存输出逐元素一致。用 pristine P1 reference 重算后，S0 held 结果可用于本轮门槛判断。

v36 的 S1/P2 D=I 检查因此在不正确的模块状态下触发，且发生在 selector/GPTQ 候选之前。根因已定位；v37 的干净 pristine capture 验证了修复语义。S1 正式质量实验没有运行。

## 问题 3：FP16 fold-error reference 漏掉 gate/up 的 W*D

v36 的 `selected_fold_correctness` 和 held fold diagnostic 将 postnorm weight 改成 `base_norm / D`，但 gate/up 仍使用原始 W。合法折叠应同时使用 `W_gate' = W_gate * D`、`W_up' = W_up * D`；down 权重不变。v36 保存的 fold-error MSE 和依赖它的 5x signal-floor 判断不可用。该诊断没有参与候选选择或 GPTQ 候选生成。

v37 用正确完整折叠重算：P0 search fold MSE 为 `1.34169e-8`，P1 held fold MSE 为 `1.31290e-8`，最大绝对误差均为 0.00390625。用修正后的 P1 fold error 计算，G11 MSE / fold error 为约 31,680.19，原 5x signal-floor gate 通过。其他质量门槛仍失败。

## 修复后的 S0 判断

候选参数与 S0 selection lock 的 weight/scale/zero/g_idx hash 全部一致；所有 10 点 row argmin 均从保存分数重放通过。v37 最初两项 down alpha replay 报错是验证器将 FP32 alpha 与 FP64 十进制 alpha 直接要求逐位相等造成的；按保存字段 dtype 转换后，全部选择一致。G00/down/alpha=1 直接分数 parity 通过。v37 重建 pristine P1 norm 输入，并复跑四臂完整 block，输出均与历史已保存输出完全一致。

修正 reference 后，P1 held 完整 MLP MSE：

| Arm | MSE |
|---|---:|
| G00 | 0.0004564555428 |
| G01 | 0.0004018796443 |
| G10 | 0.0004705618642 |
| G11 | 0.0004159299523 |

G11 相对 G01 的 held loss **增加 3.496%**，没有达到冻结的至少 1% 改善门槛。per-projection 和 row-tail 条件也失败；fold signal-floor 通过。结论是 **B：接入问题已定位并修复，S0 held 可用，但质量门槛失败**。因此不执行 S1，也不查看 P3。

该 S0 是对已查看 P1 的修复后复核，不是新盲测；本结果不能作为新数据泛化或 PPL 结论。
