# v31 — 固定 GPTQ 候选的生成数据—选择数据分离诊断

> 状态：完成的回顾性探索诊断。所有 64 篇文章及历史结果已被查看；这不是盲测，也不是正式泛化验证。

## 数据关系与冻结定义

- S0 候选的 GPTQ Hessian 由 WT2 train 完整文章 0–15 构建；原选择也在 A=0–15，独立选择在 B=16–31，评价在 C=32–47、D=48–63。
- S1 候选 Hessian 由文章 32–47 构建；原选择在 C，独立选择在 D=48–63，评价在 A、B。
- 每组 16 篇完整文章、每篇 2048 tokens；所有文章组不拆 token。v18 固定 M2 前缀的 128 篇校准文章与这 64 篇的文章哈希交集为 0。
- 复用同一固定候选 bank、v24 直接输出误差、固定前缀输入和 scalar10 网格。GPTQ 的 W4A16 symmetric dynamic G128、block128、damp 0.1、desc_act=false 语义未变。没有重跑 GPTQ 或前向。
- 原选择从 v24/v30 已存 A/C search 分数复现；独立选择使用已存 B/D 分数；四个新评价组只补 direct score，不重做候选。并列按精确分数相同后取较大 alpha。oracle 仅事后诊断。
- 完整行×候选×组长表共 1,920 项：960 项来自本轮 direct score，960 项复用已审计的 v24/v30 A/C search 与 B/D held 分数；所有新增分数均校验候选、Hessian、协议及输入缓存哈希。
- 原始 cross-score JSON 的 `score_mse_by_row` 键 0–3 表示顶层 `rows` 列表中的位置；`CROSS_GROUP_ROW_ALPHA_SCORES.csv` 已按该列表映射为实际输出行号，避免把位置编号当成模型行号。

## 生成—选择数据分离结果

改善定义：`(原选择 SSE - 独立选择 SSE) / 原选择 SSE`；正值表示独立选择更低。SSE 是预先固定的 4 个输出行上的 pooled SSE，逐行按 `MSE×32768` 汇总，不代表完整 Linear 的所有行。W/T/L 使用 v24 冻结的逐行重复 alpha=1 epsilon。
同一 bank/row 在两个评价组重复出现，以下 pooled 描述统计按其共同评价范围汇总，不把两个评价组当成独立实验样本。

### Bank × evaluation group

| Bank | Eval | n row-group | Independent vs original pooled SSE | W/T/L vs original | W/T/L vs α=1 | Median row gain | Worst row gain | >10% worse vs original / α=1 | Top-3 share of positive savings | Original / independent regret to oracle |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| S0 | C | 24 | +4.531% | 5/18/1 | 23/0/1 | +0.000% | -2.761% (L0_up_proj_R672_S0) | 0 / 0 | +78.448% | 4.66% / 0.14% |
| S0 | D | 24 | +1.323% | 3/18/3 | 23/0/1 | +0.000% | -10.420% (L23_down_proj_R0_S0) | 1 / 0 | +100.000% | 3.11% / 1.81% |
| S1 | A | 24 | -0.025% | 0/22/2 | 22/1/1 | +0.000% | -8.785% (L0_down_proj_R224_S1) | 0 / 0 | +0.000% | 1.56% / 1.58% |
| S1 | B | 24 | -0.004% | 1/22/1 | 22/1/1 | +0.000% | -3.658% (L0_down_proj_R672_S1) | 0 / 0 | +100.000% | 1.03% / 1.04% |

### Bank pooled over both evaluation groups (descriptive paired repeats)

| Bank | Independent vs original pooled SSE | W/T/L | Median row-group gain | Worst row-group gain | >10% worse vs original / α=1 | Top-1 / top-3 positive-saving concentration | Original / independent oracle regret |
|---|---:|---:|---:|---:|---:|---:|---:|
| S0 | +2.978% | 8/36/4 | +0.000% | -10.420% (L23_down_proj_R0_S0) | 1 / 0 | +30.785% / +63.324% | 3.91% / 0.96% |
| S1 | -0.014% | 1/44/3 | +0.000% | -8.785% (L0_down_proj_R224_S1) | 0 / 0 | +100.000% / +100.000% | 1.29% / 1.30% |

### Target × bank（两评价组配对汇总）

| Bank | Target | Independent vs original pooled SSE | W/T/L | Median gain | Worst gain (row) | Severe >10% vs original / α=1 | Top-3 positive saving share | Oracle regret original / independent |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| S0 | L0 down_proj | +0.000% | 0/8/0 | +0.000% | +0.000% (L0_down_proj_R0_S0) | 0 / 0 | +0.000% | 0.07% / 0.07% |
| S0 | L0 up_proj | +0.180% | 2/4/2 | +0.000% | -2.761% (L0_up_proj_R672_S0) | 0 / 0 | +100.000% | 1.15% / 0.97% |
| S0 | L12 down_proj | +0.000% | 0/8/0 | +0.000% | +0.000% (L12_down_proj_R0_S0) | 0 / 0 | +0.000% | 0.00% / 0.00% |
| S0 | L12 up_proj | +0.000% | 0/8/0 | +0.000% | +0.000% (L12_up_proj_R0_S0) | 0 / 0 | +0.000% | 0.00% / 0.00% |
| S0 | L23 down_proj | +0.554% | 2/4/2 | +0.000% | -10.420% (L23_down_proj_R0_S0) | 1 / 0 | +100.000% | 2.53% / 1.99% |
| S0 | L23 up_proj | +6.503% | 4/4/0 | +3.965% | +0.000% (L23_up_proj_R0_S0) | 0 / 0 | +88.934% | 6.50% / 0.00% |
| S1 | L0 down_proj | -1.607% | 1/4/3 | +0.000% | -8.785% (L0_down_proj_R224_S1) | 0 / 0 | +100.000% | 0.26% / 1.84% |
| S1 | L0 up_proj | +0.000% | 0/8/0 | +0.000% | +0.000% (L0_up_proj_R0_S1) | 0 / 0 | +0.000% | 0.00% / 0.00% |
| S1 | L12 down_proj | +0.000% | 0/8/0 | +0.000% | +0.000% (L12_down_proj_R0_S1) | 0 / 0 | +0.000% | 0.10% / 0.10% |
| S1 | L12 up_proj | +0.000% | 0/8/0 | +0.000% | +0.000% (L12_up_proj_R0_S1) | 0 / 0 | +0.000% | 0.00% / 0.00% |
| S1 | L23 down_proj | +0.000% | 0/8/0 | +0.000% | +0.000% (L23_down_proj_R0_S1) | 0 / 0 | +0.000% | 0.99% / 0.99% |
| S1 | L23 up_proj | +0.000% | 0/8/0 | +0.000% | +0.000% (L23_up_proj_R0_S1) | 0 / 0 | +0.000% | 2.03% / 2.03% |

两 bank × 两组的描述性总体 pooled 改善为 **+1.517%**；W/T/L=9/80/7，最差行变化 -10.420%（L23_down_proj_R0_S0）。该总体值只是有限固定行和固定候选 bank 的汇总，不能作为独立样本统计推断。

### 对 α=1 原生候选

每个 target × bank × eval 的相对 α=1 指标见 `TARGET_SUMMARY.csv`；两评价组配对后的 target 汇总见 `TARGET_POOLED_SUMMARY.csv`；全部逐行结果、最差与负项见 `ROW_EVALUATION_COMPARISONS.csv`。完整 W/T/L 不通过 PPL 或量化误差代理推断。

### Alpha 分布与配对行变化

每个 bank 有 24 个预先固定行。原 selector 与 independent selector 的逐 alpha 计数见 `ALPHA_DISTRIBUTION.csv`；每行所选 alpha、选择 margin、两个评价组的误差与 oracle alpha 见 `SELECTIONS_BY_ROW.csv`。两个评价组按同一行配对的均值见 `PAIRED_ROW_SUMMARY.csv`。

| Bank | Changed rows | Change rate | Original selector alpha counts | Independent selector alpha counts | Paired row units positive / negative | Median paired-row mean gain |
|---|---:|---:|---|---|---:|---:|
| S0 | 6/24 | 25.0% | 0.8:10, 0.9:13, 1.0:1 | 0.7:1, 0.8:7, 0.9:16 | 4 / 2 | +0.000% |
| S1 | 2/24 | 8.3% | 0.8:8, 0.9:15, 1.0:1 | 0.8:10, 0.9:13, 1.0:1 | 0 / 2 | +0.000% |

## 逐 target 结果

每 target × bank × eval 的全部统计见 `TARGET_SUMMARY.csv`，不从 structured 模式推断 scalar 结果。低于 0 的提升表示退化；最差行和严重退化没有过滤。

## 成本与范围

- 新 direct-score 调用：240；row-alpha 分数：960；GPTQ candidate executions=0，new model forwards=0。
- CUDA init、warmup、候选读取、评分和检查点写入计入的总 wall：43.996s / 600s。
- 实测评分 kernel 汇总 38.351s；候选读取汇总 0.264s；其余未归因/初始化/校验/IO 4.973s。峰值 allocated/reserved=128,135,680/157,286,400 bytes。
- 此成本只适用于本次 240 个完整 Linear candidate × 16-document-group 的行级 direct scoring。它不是 GPTQ 校准成本、整模型成本或在线推理加速。

## 结论边界

本轮只检验：在已有固定 GPTQ 候选集合上，把选择分数从候选 Hessian/原 search 文章转移到独立文章，是否改变并改善有限固定行的后续文章输出 SSE。结果不能证明 multiple-comparison overfitting，也不能推出完整模型泛化、PPL 或定理。由于历史文章已被查看，后续如要作正式泛化主张，应事先冻结全新文章集合、选择和评价一次性流程及判据；本轮结果仅能作为预注册设计依据。

**质量评测/PPL：NOT_RUN。整模型量化：NOT_RUN。候选 GPTQ 重算：NOT_RUN（按冻结协议为 0 次）。**

复现与文件索引见 `REPRODUCE.md`。
