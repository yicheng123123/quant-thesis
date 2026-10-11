# v27 评分目标与二次型推导

## 正式目标审计

v18 的 `run_formal.quantize_search_module` 对候选 FP16 权重 `Q` 调用 `run_precheck.score_rows(x_mm, pristine, Q)`。`x_mm` 是当前 M2 已提交量化前缀产生、在目标 Linear 前捕获的同一输入；v26 已将 L0 up/down 的输入和重新构建的 H 哈希与 v18 checkpoint 核对。两个目标 Linear 的原始 checkpoint 均无 bias 键。即使有相同 bias，输入相同且在局部线性输出中会抵消；本轮核验的是实际无 bias 的两个目标。

原评分先做 `e = Q.float() - W.float()`，再按 4 篇文档一块将 FP16 `X` 转 FP32，计算 `X @ e.T`。每个输出先转换 FP64、平方，并按行累加；最后除以 `128 × 2048 = 262144`。没有 mask、丢弃 token、文章重加权或单独的参考输出缓存。每篇都是 2048 token，故全部 token 等权，文章总权也相同。该目标是局部 Linear preactivation 重建误差，不包含后续非线性、整块输出或最终 PPL。由于浮点实现先减权重再矩阵乘法，它也不承诺与分别计算两个输出后相减逐位相同；本轮保持原实现作为裁判。

令完整输入展平为 `X ∈ R^(N×d)`，行误差 `e_i = Q_i - W_i`，`N=262144`。数学上

```
score_i = ||X e_i||_2² / N
        = e_i^T (X^T X / N) e_i
        = e_i^T C e_i.
```

`C` 是完整、未加 damping 的 Gram 矩阵。这个表达保留所有非对角项及 G128 组间交叉项。实现从同一 FP16 输入按正式 4 文档块转 FP32，累积 FP32 `X.T @ X`，除以 262144；候选误差由相同的 FP16 W/Q 转 FP32，算 FP32 `e @ C`，再以 FP64 乘积及行求和。原评分为 FP32 GEMM 后 FP64 平方累加；两条浮点路径有不同运算顺序，因此比较逐行选择时使用**精确 alpha 相等**，分数只报告误差，未设置可掩盖选择差异的容差。

安装版 GPTQ 的 `process_batch` 把每篇 2048-token 输入视为 `batch_size=1`，以 `sqrt(2/nsamples)` 缩放，再递推更新 H。理想算术下，128 篇后的**原始** H 等于 `2/128 × Σ_doc X_doc^T X_doc = 4096 C`。但它逐篇在 FP32 缩放/累加；本轮 C 按 4 篇块独立构建，浮点数值不能直接认定 `H/4096 == C`。GPTQ 候选还在 H 的副本上处理 dead columns 和 damping。v27 **没有复用任何 H、damped H、inverse 或 GPTQ 内部累计 loss** 来评分。C 构建成本已单独计入对照。

`score_candidate(..., mode="direct")` 默认调用冻结的 v18 原评分；只有显式 `mode="quadratic", c=C` 才使用新路径。v18 正式入口未修改。额外 FP64-C 或 FP64-X 诊断没有运行；报告中的精度仅对应上述候选评分实现。
