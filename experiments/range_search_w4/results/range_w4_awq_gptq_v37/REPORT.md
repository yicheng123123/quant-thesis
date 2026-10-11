# v37 — v36 正确性修复与 S0 有效性

## 决定

**B：接入/参考问题已定位并修复；修复后的 S0 held 结果有效，但冻结质量门槛失败。停止，不运行 S1。**

v36 原始目录和停止记录保持只读。S0 使用已锁定的 v36 候选与选择，重建 pristine P1 reference 后重新评价；这是对已查看 held 文章的修复后复核，不是新盲测。v36 的原始 held 数值不再引用为有效结果。

## 三项问题

| 项目 | 根因 | 影响 | v37 证据 |
|---|---|---|---|
| G00/down/alpha=1 直接评分记录缺失 | resume parity 分支跳过 alpha=1 分数计算 | 只缺 parity 字段；不影响候选或选择 | 同一 saved native W 的全行直接分数与保存的 896 行 alpha=1 score 完全一致；最大差 0 |
| S0/P1 与 S1/P2 D=I capture 不一致 | G11 后 norm 留为 S0 `base_norm/D`，后续 capture 未复位 | P1 held reference 错；P2 D=I 前置检查在错误状态下失败 | P1/P2 旧 postnorm 都与 S0 `base_norm/D` 重放逐元素一致；P0 与 pristine 重放一致；clean P2 pristine capture 自身重放完全一致 |
| fold-error reference 漏 `gate/up W*D` | 折叠只改 norm，未折 gate/up 输入列 | v36 fold error 与 5x signal-floor 判断无效 | 完整折叠后 P0/P1 fold MSE 分别 `1.34169e-8`、`1.31290e-8` |

详细受影响任务和张量见 [BUG_IMPACT_TABLE.csv](./BUG_IMPACT_TABLE.csv)，修复建议 diff 见 [FIX_DIFF.patch](./FIX_DIFF.patch)。

## S0 修复后结果

S0 P0 search cache 与 pristine norm 重放逐元素相同。v36 selection lock 中的 D、四臂 weight/scale/zero/g_idx 均在 v37 逐项核验通过。保存候选 score 重放了所有逐行 alpha argmin；初版 v37 checker 的两项 down 差异来自 FP32 已存 alpha 与 FP64 十进制 alpha 的直接逐位比较，按记录 dtype 转换后完全一致。完整记录见 `correctness/S0_SELECTION_REPLAY.json`。

v37 从冻结 P1 prenorm 和 pristine norm 构造新的 held 输入，并用已锁定张量重跑完整 L12 block。G00/G01/G10/G11 四臂输出均与 v36 保存的 full-block 输出逐元素相同。新的 P1 reference hash 为 `bf5f0b09fb47e9a64fc0aaacc18ccbd6d9c5aaab943056416b088db67380e0f4`。所有修复后指标从该 reference 重新计算。

| Arm | 修复后 P1 held full-MLP MSE | 对照 |
|---|---:|---:|
| G00 原坐标 + native | 0.0004564555428 | — |
| G01 原坐标 + scalar10 | 0.0004018796443 | — |
| G10 AWQ D + native | 0.0004705618642 | — |
| G11 同一 D + scalar10 | 0.0004159299523 | — |

G10 相对 G00 loss 增加 **3.09%**；G11 相对 G10 loss 降低 **11.61%**；组合 G11 相对 G01 loss 增加 **3.496%**，因此没有达到至少 1% 的主推进门槛。修正 fold signal-floor 通过，但 per-projection 不回退超过 5% 和 projection row-tail 两项均失败。逐行尾部、逐文章和投影数据在 `raw/S0_HELD_REPAIRED_*.csv`。

G11 相对 G01 的三个 projection pooled MSE 变化分别是：gate **+18.46%**、up **−0.827%**、down **−0.041%**。逐行上，gate 有 2,195/4,864 行退化超过 10%、717 行超过 50%；up 有 753/4,864 行超过 10%、71 行超过 50%；down 有 2/896 行超过 10%、0 行超过 50%。尾部门槛失败不是少量平均误差波动。

这些是局部 reconstruction SSE/MSE，不是 PPL。没有运行 PPL。

## 停止边界

S0 repair 后主门槛未过，不补跑 S1。v37 没有运行新的候选搜索、S1 GPTQ、P3 held 评价、PPL 或整模型量化。S1/P2 的错误运行状态已通过 norm/D 逐元素重放定位；正确的 pristine P2 capture 已验证，但这不等于 S1 质量结果。

## 成本

v36 已计入 `180.335929 s`；v37 再计 `110.325063 s`，累计 `290.660992 s`。低于 v37 的 300 秒子预算和原 1800 秒总上限。该账本按同步后的累计进程时间保守计量，包含模型加载、预热、失败尝试、GPU工作、输出收尾和差额对账，不是单独的 GPU kernel 时间。v37 具体阶段、6 条失败/诊断尝试及峰值显存见 [BUDGET_SUMMARY.json](./raw/BUDGET_SUMMARY.json) 与 [BUDGET_LEDGER.csv](./raw/BUDGET_LEDGER.csv)。

## 复现

使用 v36 冻结环境 `.venv-gptq`，命令见 [REPRODUCE.md](./REPRODUCE.md)。v36/v35 文件与公共量化实现未修改。
