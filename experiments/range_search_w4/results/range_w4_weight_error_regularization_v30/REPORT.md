# v30：固定候选上的权重误差正则化诊断

状态：COMPLETE_ARTIFACT_ONLY_DIAGNOSTIC。全程 CPU 19.9s / 900s 预算，wall 18.2s。
仅读取冻结 v23/v24/v25 产物；没有新前向、CUDA、GPTQ、held、PPL 或整模型量化。

## 对象与口径

使用 Qwen2.5-0.5B-Instruct revision `7ae557604adf67be50417f59c2c2f167def9a775` 的 v24 冻结实验：原 scalar10 alpha=0.1…1.0；L0/L12/L23 的 up/down；每个线性层固定行 [0,224,448,672]；S0/S1。共 48 个 row-split 单元、120 个四行 candidate tensor states、480 个 row-alpha 观察。v11 早期结果未合并；v28 浮点选择差异未作为分布退化证据。
直接 search/held 分数来自 v24 原文件，candidate `weight_fp16` 是保存的 FP16 反量化权重。checkpoint safetensors 内目标 W 存为 BF16；v24 的冻结模型 dtype 是 FP16，因此 v30 将来源 W cast 成 FP16 后计算 e=Q−W，并核对转换后 tensor hash。每个候选的四行 Q 张量、scale、zero、g_idx dtype/shape/hash均逐项核验；held sidecar 的 `candidate_weight_sha256` 与同一 Q 张量 hash 一致，search 原始 CSV 与候选 sidecar 的 row 分数一致。
误差口径：output MSE 是每个 row、每个 input token 的局部投影输出残差平方均值；output SSE=MSE×该 split 的 16×2048 tokens。权重 MSE=||Q−W||²/input_dim；正则化只使用平方范数 ||Q−W||²。详见 `CANDIDATE_ROW_TABLE.csv`。

## 统计量和归一化

原始 search 输入及未阻尼 Hessian 均可读。GPTQ `process_batch` 在每篇文章 batch 上令 `H = (2/n_documents) Σ_doc X_docᵀX_doc`，这里 nsamples=16、每篇2048 tokens；缓存 H 在阻尼/求逆前保存。H 已按文章数取平均，但每篇仍累加2048个 token，因此 v30 用 `C_search=H/(2×2048)=Σ_token x xᵀ/N_tokens`，`kappa=trace(C_search)/input_dim`，并由原始 FP16 search 输入独立重算 trace。核对关系是 `trace(H)/(2×2048×input_dim)`；不把 raw H/2 直接当逐 token 协方差，也不使用 damped H 或 inverse。详见 `KAPPA_NORMALIZATION.csv`。
rho=tau*kappa，tau 预固定为 {0, 0.01, 0.1, 1}。rho=0 按 v24 原 `selected_by_original_10` 结果直接复现；正 rho 的并列按更大 alpha 处理。完整有限候选下包络的全部区间、交点和选择都在 held 数据打开前由 search MSE 与权重平方范数生成，详见 `SELECTION_PATH*.csv` 与 `FULL_PATH_*_SEARCH_ONLY.csv`。

## search 选择与 alpha=1 的机制对照

四类数量（48 row-split observations）：1) search 改善、权重误差增大、held 退化 0；2) search 改善、权重误差增大、held 改善 0；3) search 改善、权重误差未增大、held 退化 0；4) 其他 48。第4类中，search 严格改善 46 个；其权重平方误差增大的有 0 个。search 改善且权重误差未增大的单元，held 相对 alpha=1 有 46 胜、0 平、0 负。held win/tie/loss 沿用 v24 每个单元的 frozen epsilon。所有单元的实际 alpha 和误差变化见 `MECHANISM_RESULTS.csv`，target/split 数量见 `MECHANISM_SUMMARY.csv`。
10 个候选的 Pareto 审计共 480 个点，其中被另一个候选同时在 search output MSE 与 ||Q−W||² 支配 415 个，未支配 65 个。逐候选支配者见 `PARETO_DOMINANCE.csv`；单位与 split 计数见 `PARETO_UNIT_SUMMARY.csv`。

## 预固定 tau 和完整路径的 held 事后诊断

下表使用 pooled held SSE，列出所有范围；行级负结果全部保存在 `NEGATIVE_RESULTS.csv`。S0/S1 和六个 target 的完整比较在 `REGULARIZATION_SUMMARY.csv`，逐行选择和 held 分数在 `TAU_SELECTIONS.csv`。完整路径区间及交点另见 `FULL_PATH_INTERVAL_SUMMARY.csv` 和 `FULL_PATH_BOUNDARY_SUMMARY.csv`。胜/平/负阈值沿用 v24 epsilon；严重退化计数沿用原协议的相对 alpha=1 held MSE <−10% 口径。正收益集中度按 top1/top3 row 对 gross positive held SSE saving 的占比，净收益占比同时列出。

正的 held change 表示 pooled held SSE 降低。

| selector | vs search ALL/S0/S1 (%) | search SSE change (+ = sacrifice) | norm² reduction (+ = decrease) | W/T/L | median / worst row (%) | top1 / top3 gross positive share | severe vs alpha1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| tau=0 | 0.0000 / 0.0000 / 0.0000 | 0 | 0 | 0/48/0 | 0.0000 / 0.0000 | 0.0 / 0.0% | 0 |
| tau=0.01 | 0.0000 / 0.0000 / 0.0000 | 0 | 0 | 0/48/0 | 0.0000 / 0.0000 | 0.0 / 0.0% | 0 |
| tau=0.1 | -0.7364 / -0.0081 / -1.5459 | 4.2618 | 0.00175546 | 0/45/3 | 0.0000 / -14.9586 | 0.0 / 0.0% | 0 |
| tau=1 | -3.3767 / -1.0954 / -5.9121 | 90.1335 | 0.00728638 | 3/34/11 | 0.0000 / -30.7269 | 52.5 / 100.0% | 0 |

对 alpha=1 的 pooled held change、权重范数下降与行级统计：

| selector | held change vs alpha1 ALL/S0/S1 (%) | norm² reduction vs alpha1 (%) | W/T/L | median / worst row (%) | top1 / top3 gross positive share |
|---|---:|---:|---:|---:|---:|
| tau=0 | 22.9927 / 19.5670 / 26.4731 | 12.0161 | 46/2/0 | 15.7852 / 0.0000 | 10.7 / 31.3% |
| tau=0.01 | 22.9927 / 19.5670 / 26.4731 | 12.0161 | 46/2/0 | 15.7852 / 0.0000 | 10.7 / 31.3% |
| tau=0.1 | 22.4256 / 19.5605 / 25.3365 | 12.2047 | 46/2/0 | 15.7852 / 0.0000 | 11.0 / 32.1% |
| tau=1 | 20.3924 / 18.6859 / 22.1261 | 12.7988 | 46/2/0 | 14.1553 / 0.0000 | 12.1 / 33.4% |

预固定 tau 的六个 target pooled 与 S0/S1 held 结果（相对原 search；% 和 W/T/L）：

| scope | tau=0.01 | tau=0.1 | tau=1 |
|---|---:|---:|---:|
| L0/up_proj | 0.0000% (0/8/0) | 0.0000% (0/8/0) | -1.5897% (1/4/3) |
| L0/up_proj/S0 | 0.0000% (0/4/0) | 0.0000% (0/4/0) | -2.2648% (1/2/1) |
| L0/up_proj/S1 | 0.0000% (0/4/0) | 0.0000% (0/4/0) | -0.8861% (0/2/2) |
| L0/down_proj | 0.0000% (0/8/0) | 0.0000% (0/8/0) | 0.0000% (0/8/0) |
| L0/down_proj/S0 | 0.0000% (0/4/0) | 0.0000% (0/4/0) | 0.0000% (0/4/0) |
| L0/down_proj/S1 | 0.0000% (0/4/0) | 0.0000% (0/4/0) | 0.0000% (0/4/0) |
| L12/up_proj | 0.0000% (0/8/0) | -0.0500% (0/7/1) | -0.0500% (0/7/1) |
| L12/up_proj/S0 | 0.0000% (0/4/0) | -0.0987% (0/3/1) | -0.0987% (0/3/1) |
| L12/up_proj/S1 | 0.0000% (0/4/0) | 0.0000% (0/4/0) | 0.0000% (0/4/0) |
| L12/down_proj | 0.0000% (0/8/0) | -0.0467% (0/7/1) | -0.3422% (0/6/2) |
| L12/down_proj/S0 | 0.0000% (0/4/0) | -0.0937% (0/3/1) | -0.0937% (0/3/1) |
| L12/down_proj/S1 | 0.0000% (0/4/0) | 0.0000% (0/4/0) | -0.5892% (0/3/1) |
| L23/up_proj | 0.0000% (0/8/0) | 0.0000% (0/8/0) | -3.0646% (1/6/1) |
| L23/up_proj/S0 | 0.0000% (0/4/0) | 0.0000% (0/4/0) | 1.3519% (1/3/0) |
| L23/up_proj/S1 | 0.0000% (0/4/0) | 0.0000% (0/4/0) | -7.9903% (0/3/1) |
| L23/down_proj | 0.0000% (0/8/0) | -1.6420% (0/7/1) | -4.5695% (1/3/4) |
| L23/down_proj/S0 | 0.0000% (0/4/0) | 0.0000% (0/4/0) | -3.3933% (1/1/2) |
| L23/down_proj/S1 | 0.0000% (0/4/0) | -3.5054% (0/3/1) | -5.9043% (0/2/2) |

完整正 rho 下包络包含 18 个 rho 区间，另有 17 个全局交点。相对原 search 选择 pooled held 为正的区间 2 个、为负的区间 15 个。正区间：P0004 rho∈(0.0421195757155783176947477461496041375519007478272965561234226, 0.0915838218575383142996551334024640407559703378325761166679190) pooled=0.0124%，top1正收益占比=100.0%；P0005 rho∈(0.0915838218575383142996551334024640407559703378325761166679190, 0.0957931097913271696500571797569147204420267353151272362443481) pooled=0.0060%，top1正收益占比=100.0%。这些事后区间不用于挑选最终 rho，只描述固定 10 候选集的有限路径。
跨 split 的相同行是配对观测：共24个 layer/projection/row 身份各自出现在 S0、S1；这里不将48个单元解释为独立样本，也不作显著性或泛化推断。

## 结论边界

所有预固定正tau pooled held相对原search选择均无正收益，权重误差下降不能支持所检验机制，或伴随明显search/held损失。
不进入新文章验证；先停止该正则化方向。
这个诊断只是在固定 GPTQ 候选上重选 alpha；没有改候选生成 Hessian 或 GPTQ 补偿。权重误差下降不等同于鲁棒性提升；没有 held/PPL 保证，也没有证明既往退化由 Hessian 不确定性造成，更没有提出新的权重偏移正则。v28 的浮点选择差异未计入机制证据。

## 文件与复现

`SOURCE_MANIFEST.csv` 列出实际读取来源及其 SHA-256；`AUDIT_CHECKS.json` 汇总候选张量、分数、Hessian normalization 和 provenance 检查；`RUN_SUMMARY.json` 记录范围与执行边界。完整负项见 `NEGATIVE_RESULTS.csv`，跨 split 配对行见 `PAIRED_SPLIT_OUTCOMES.csv`，全部候选和 target/split 结果分别见候选表与 summary CSV。

在 PowerShell 中从仓库根目录复现：

```powershell
& 'E:/quant-thesis/.venv-gptq/Scripts/python.exe' -B -X utf8 'E:/quant-thesis/experiments/range_search_w4/results/range_w4_weight_error_regularization_v30/analyze_v30.py'
```
