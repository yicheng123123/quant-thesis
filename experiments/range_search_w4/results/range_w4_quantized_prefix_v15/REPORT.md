# v15 Quantized-Prefix Transfer Audit

**预注册 verdict：TRANSFER_SIGNAL_PRESERVED**。该 gate 仅用于研究路线判断，不表示统计显著性。

## 实验实现与正确性

QPREFIX 由 GPTQModel 2.2.0 原生 true-sequential alpha=1 W4A16 symmetric G128 对 C01–C16 量化完整模型生成。随后对同一 48 篇文档做量化模型 forward，缓存目标 block boundary，并在重放目标 block 时只把当前目标 projection 临时恢复为 pristine FP16 Linear；目标之前的 native quantized groups 保持生效。Torch 后端使用 eager `dequantize_weight` 路径；Windows 环境没有 Triton，因此未启用仅属性能优化的 `torch.compile/inductor`。

- 最终 correctness：**PASS**；8 项 native alpha=1 parity 全部通过，group-entry weight、scale、zero-point、最终 FP16 权重和 search/heldout SSE 均一致。
- 48 rows × FP/QPREFIX 两个输入 regime，共 96 条 row-regime 结果；候选数 **960**。
- v8–v14 历史结果树 SHA256 逐文件复核：True。

## Transfer 结果

- alpha exact agreement：33/48 (68.8%)；within one grid step：47/48 (97.9%)。
- selected alpha counts，FP：`{'0.1': 0, '0.2': 0, '0.3': 0, '0.4': 0, '0.5': 0, '0.6': 0, '0.7': 0, '0.8': 15, '0.9': 30, '1.0': 3}`；QPREFIX：`{'0.1': 0, '0.2': 0, '0.3': 0, '0.4': 0, '0.5': 0, '0.6': 0, '0.7': 0, '0.8': 14, '0.9': 30, '1.0': 4}`。
- gain FP 与 QPREFIX 的 Spearman：0.4330。
- pooled heldout gain：FP_INPUT 13.87%，QPREFIX_INPUT 15.81%；median row gain：12.13% / 12.92%。
- FP-positive rows：37；其中 QPREFIX 仍 positive 35，positive→negative 1。
- gain < −5/−10/−20/−40%：FP 5/1/1/0 rows；QPREFIX 0/0/0/0 rows。
- 最差 row gain：FP L14_o_proj_R670 (-38.45%)；QPREFIX L18_o_proj_R793 (-1.96%)。
- pooled SSE 节省集中度：最大单 row 正向节省 / net 节省 FP 33.6%、QPREFIX 26.7%；前 5 rows FP 66.0%、QPREFIX 54.2%。
- heldout D 局部 MSE 的 QPREFIX/FP 比值中位数：0.9961。此比值只描述两种输入分布下 local baseline reconstruction，不代表总体模型质量。

### Alpha selection 与误差方向诊断

Positive→negative 的 1 个案例中，alpha 改变 0 个、alpha 相同 1 个。唯一案例 L6_o_proj_R790 的 alpha_FP=alpha_QPREFIX=0.8；FP 产生的候选在 QPREFIX heldout 上 gain 为 7.48%，而 QPREFIX 产生的候选在 FP heldout 上 gain 为 -8.48%。两个 regime 使用不同 Hessian/GPTQ 补偿分别生成候选，因此这些 cross-evaluation 不足以区分输入误差加权与候选误差向量变化；本轮无法进一步归因。
Input/Gram drift by layer (descriptive): {"2": {"relative_activation_drift_mean": 0.1070916885510087, "cosine_similarity_mean": 0.9927373826503754, "relative_gram_difference_mean": 0.04509536596015096}, "6": {"relative_activation_drift_mean": 0.27447523549199104, "cosine_similarity_mean": 0.9607857465744019, "relative_gram_difference_mean": 0.22647985443472862}, "14": {"relative_activation_drift_mean": 0.27930376306176186, "cosine_similarity_mean": 0.9601049721240997, "relative_gram_difference_mean": 0.19625361636281013}, "18": {"relative_activation_drift_mean": 0.26267626136541367, "cosine_similarity_mean": 0.9653761386871338, "relative_gram_difference_mean": 0.23245966620743275}}。L2 到 L6 漂移增加，后续层有波动而非单调增长；层间差异不构成因果证据。

## 成本与范围

同步计时组件合计 281.34s；native prefix GPTQ 133.08s，FP capture 2.64s，QPREFIX forward + target replay 7.32s，Hessian builds 2.05s，candidate core/search-loss 133.84s，heldout evaluation 0.66s。总和不计脚本启动和计时区间外的文件序列化。
候选成本预估 401.39s，实际同步计时 133.84s。30 个已完成槽位从校验通过的缓存恢复，另 930 个本次计算；unique workload 仍为 960。

本轮只测局部 projection 输出重建，没有运行 PPL、下游任务或完整模型 quality gate。Transfer verdict 不能外推为最终模型收益。
