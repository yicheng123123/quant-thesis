# v32 — 固定通道缩放 × scalar10 range search：2×2 局部诊断

> 全部 64 篇文章及历史结果此前已被查看。本轮为回顾性探索筛查；仅覆盖六个 target 的四个固定输出行、两个 split。

## 冻结方法与数据

G00 为原坐标 α=1；G01 为原坐标 scalar10 search 选择；G10 为固定 D 的缩放坐标 α=1；G11 使用同一个 D 做 scalar10 search 选择。正的改善表示误差下降。
S0 由 A=0–15 完整文章估计 D/H 并选择，B=16–31 评价；S1 由 C=32–47 估计/选择，D=48–63 评价。每篇 2048 tokens，held 从未参与 D/H 或 alpha 选择。四臂使用固定 v18 M2 前缀缓存输入及同一原始输出参考。
D 的每通道统计为 search X 的 `mean(abs(X[:,j]))`，`sqrt(max(a,1e-8))` 后除以端点几何均值。D 保存为 FP32，W′ 与 X′转换为 FP16；缩放分支由 X′用原生 `GPTQ.process_batch` 重新构建 FP32 H。动态 G128、原生 compensation 和 damp 0.1 未改变。这是固定 activation-based scaling 诊断，并非官方 AWQ 或可学习缩放。

## 正确性

D=I 在 896/4864 两种输入宽度上精确复现 v24 H、α=1 的 Q/scale/zero/g_idx 和逐行原评分。缩放坐标 α=1 的 12/12 个 target/split 与原生 GPTQ 张量精确一致。首个 target 的 `1.0→0.8→1.0` 复跑一致，W′与 H 未受候选污染。未量化变换的 FP64/FP32/FP16 数值差异分别保存，完整 search/held 的 FP16 坐标漂移也已记录。
原始参考输出的显式 `X′Q′ᵀ−XWᵀ` 与冻结的分解评分在两个预设 target 的全部 held 文章上比较，最大相对分数差 2.29e-07；分解评分精确复现保存的 held 分数。

## 四臂 held 结果

下列 SSE 仅为固定行 × 16 篇 held 文章，每行 MSE×32768 汇总；不是完整 Linear 输出或 PPL。

| Split | G00 SSE | G01 SSE | G10 SSE | G11 SSE |
|---|---:|---:|---:|---:|
| S0 | 2515.996476 | 2023.691860 | 2125.082827 | 1905.776191 |
| S1 | 2476.484797 | 1820.882374 | 1986.615101 | 1796.999550 |

| Split | Comparison | Pooled SSE gain | W/T/L | Median row gain | Worst row gain | >10% regressions | Top-3 share of positive savings |
|---|---|---:|---:|---:|---:|---:|---:|
| S0 | G10_vs_G00_scaling_at_alpha1 | +15.537% | 12/0/12 | -2.348% | -67.039% (L0_down_proj_R0_S0) | 9 | +80.803% |
| S0 | G11_vs_G10_search_after_scaling | +10.320% | 22/0/2 | +12.616% | -16.971% (L23_up_proj_R672_S0) | 1 | +61.176% |
| S0 | G11_vs_G01_combination_vs_original_search | +5.827% | 12/0/12 | -2.162% | -60.656% (L0_down_proj_R0_S0) | 10 | +81.433% |
| S1 | G10_vs_G00_scaling_at_alpha1 | +19.781% | 12/0/12 | +0.923% | -76.525% (L0_down_proj_R0_S1) | 9 | +73.698% |
| S1 | G11_vs_G10_search_after_scaling | +9.545% | 22/1/1 | +10.611% | -2.828% (L23_up_proj_R0_S1) | 0 | +57.491% |
| S1 | G11_vs_G01_combination_vs_original_search | +1.312% | 11/0/13 | -2.404% | -84.132% (L0_down_proj_R0_S1) | 9 | +85.279% |

### Target × split 的 G11 相对 G01

| Split | Target | Pooled SSE gain | W/T/L | Worst row gain |
|---|---|---:|---:|---:|
| S0 | L0 down_proj | -35.279% | 0/0/4 | -60.656% (L0_down_proj_R0_S0) |
| S0 | L0 up_proj | -0.568% | 3/0/1 | -35.483% (L0_up_proj_R0_S0) |
| S0 | L12 down_proj | -18.656% | 1/0/3 | -37.990% (L12_down_proj_R448_S0) |
| S0 | L12 up_proj | -7.727% | 3/0/1 | -54.634% (L12_up_proj_R672_S0) |
| S0 | L23 down_proj | +23.412% | 3/0/1 | -32.549% (L23_down_proj_R672_S0) |
| S0 | L23 up_proj | -8.953% | 2/0/2 | -48.905% (L23_up_proj_R672_S0) |
| S1 | L0 down_proj | -38.370% | 0/0/4 | -84.132% (L0_down_proj_R0_S1) |
| S1 | L0 up_proj | +4.331% | 3/0/1 | -28.256% (L0_up_proj_R0_S1) |
| S1 | L12 down_proj | -18.039% | 1/0/3 | -41.431% (L12_down_proj_R448_S1) |
| S1 | L12 up_proj | -6.582% | 3/0/1 | -53.487% (L12_up_proj_R672_S1) |
| S1 | L23 down_proj | +18.778% | 3/0/1 | -38.326% (L23_down_proj_R672_S1) |
| S1 | L23 up_proj | -15.132% | 1/0/3 | -36.558% (L23_up_proj_R672_S1) |

### Interaction

`I=(L10−L11)−(L00−L01)` 与每行 held MSE 处于同一单位；正值说明缩放后搜索的绝对损失下降更大，不代表 G11 必然最好。

| Split | Pooled I SSE | I / G00 SSE | positive / zero / negative rows |
|---|---:|---:|---:|
| S0 | -272.997980 | -10.850% | 10/0/14 |
| S1 | -465.986871 | -18.816% | 12/0/12 |

## 判读与停点

G11 相对 G01 的 pooled SSE 虽下降 +3.688%，但仅 23/0/25 行胜/平/负，中位逐行改善 -2.404%；19 行退化超过 10%，最差 -84.132%。两个 split 的 pooled 改善不能代表跨 target 稳定改善。
L23 down_proj 两个 split 单独贡献 364.132 SSE 净节省，超过全体的 141.798；去掉该 target，剩余 target 的 G11 相对 G01 为 -10.437%。L0/L12 down_proj 在两个 split 均退化，L23 up_proj 也均退化。
G11 与 G01 的 search 选择 alpha 在 19/48 个行×split 单位不同。缩放后的搜索相对同 D 的 G10 有局部收益，但不能据此推断组合优于原搜索；总体 I=-738.985 SSE，为负。
FP16 坐标变换的 held 未量化漂移逐行 MSE 最大 1.39e-07，已经计入 G10/G11 对原参考的损失；本实验没有将缩放效果与这项舍入误差分离。

结论：此固定 D 方案在预定局部单位上没有稳定优于原 scalar10 搜索，应停止当前方案的扩展。不据此否定其他预先定义的 scaling 方法；若将来重启，需先冻结合法的整模型 D 折叠和新文章验证协议。


全部 48 行、三种对照的数值与负例分别见 `FOUR_ARM_ROW_RESULTS.csv`、`CONTRAST_SUMMARY.csv`、`NEGATIVE_CASES.csv`。完整 target/split、pooled、alpha 分布和集中度保留在原始 CSV/JSON。

## 成本

三次 GPU 进程累计 wall 190.446s / 1200s；其中候选 bank 为 120 次完整 Linear GPTQ，另有原生 α=1 parity 12 次、D=I 2 次、状态复跑 1 次，**实际完整 Linear GPTQ 共 135 次**。搜索逐行分数 480 个；新 held 逐行分数 96 个。
候选 GPTQ 累计 133.869s；D 统计 7.417s、缩放 H 3.358s、因子处理 1.172s、search 评分 10.153s。峰值 allocated/reserved 为 2,153,906,176/2,369,781,760 bytes。其他阶段和未归因时间见 `COST_SUMMARY.json`。

## 结论边界

本轮仅检验一个预固定 D 的局部机制。G10/G11 使用完全相同的 D；D 没有按 held 结果或每臂分别调节。`Q_eff=Q′D^-1` 只可辅助原坐标误差分析，不能据此把它当成仍可直接按原 G128 格式 packing 的权重。整模型部署须另行解决合法 D 折叠及下游输入传播，本轮未修改模型。

全部文章与历史结果已被查看，本轮不构成盲测或正式泛化证据。未运行整模型量化或 PPL，不能把局部 SSE 变化称为模型质量或在线推理收益。

复现命令见 `REPRODUCE.md`；逐文件 SHA256 见 `OUTPUT_MANIFEST.csv`。
