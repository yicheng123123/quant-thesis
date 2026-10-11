# 目标等价性与精确 worst-case

## 1. 实际评分坐标与归一化

来源：v24 `score_selected_rows`，v30 `score_and_weight_candidates` / `create_search_only_paths`，native GPTQ `process_batch`。v30 没有独立 PROTOCOL.json，其冻结定义在实际脚本、RUN_SUMMARY 和所引用 v24 协议内。

令 X∈R^(N×d)，当前原始目标行 w∈R^d，固定 GPTQ 候选 qα，eα=qα−w。这里“原始”是同一个目标 Linear 的 pristine FP16 权重；不是已经补偿过的列状态，也不是 BF16 checkpoint 直接相减。

v24 缓存 X 为 FP16，送入评分后转 FP32；q/w 转 FP32 后作差；两文章一块进行 FP32 GEMM，平方与累加转 FP64。v30 的 ||e||² 亦为 FP32 差分后 FP64 平方求和，不能误写为所有平方和都用 FP32。

参考目标是相同 X 下的原始 Linear 输出；实际实现直接计算 X e，不引入不同参考前缀的输出。相同 bias 若存在则抵消；没有 bias 量化或新增 bias 残差项。全部 2048 tokens 等权，无额外 mask、逐文章重加权或 token 子集，16 篇 search 的 N=32768。固定文章长度使逐文章平均与逐 token 平均相同。

定义未阻尼二阶矩（**不是中心化协方差**）：

\[
C=\frac1N X^TX,\quad b_\alpha=\frac{\|Xe_\alpha\|^2}{N}=e_\alpha^TCe_\alpha,
\quad m_\alpha=\|e_\alpha\|^2,\quad L_{weight}=m_\alpha/d.
\]

所有非对角项及跨 G128 组项都保留。上述等式为实数数学关系，不是两种浮点执行路径逐位一致承诺；v28 的数值选择差异仍成立。本轮使用原直接分数，不重新构建 C 或重评分。

native raw H 已按文章数 n 平均：

\[
H=\frac2n\sum_{a=1}^n X_a^TX_a=2T C,\quad T=2048.
\]

因此 C=H/4096，不能再除 n，也不能直接用 H/2。dead-column 替换、damping、排序后的 H、inverse、Cholesky factor 均不是此 C。复用 v30 已存 kappa 审计最大 H/输入相对差为 2.1430284569685965e−7，来源于有限精度；本轮没有把 H 代替原直接分数。

## 2. 参数映射：与 v30 是同一目标

新 Robust Hessian selector：Jρ(α)=bα+ρmα。v30 正是该目标，ρ=τκ，κ=tr(C)/d。

- output MSE＋weight squared norm：ρ=τκ，无新目标。
- output SSE＋λ||e||²：λ=Nρ。
- output MSE＋λ·weight MSE：λ=dρ。
- raw H quadratic＋γ||e||²：γ=2Tρ。
- m_out 行完整矩阵的 mean output MSE 为 tr(ECEᵀ)/m_out；相配惩罚为 ρ||E||F²/m_out。若惩罚写成全矩阵 weight MSE，其系数仍为 dρ。

固定共同 τ 并不是固定共同 ρ：每个 layer/projection/split 的 κ 不同。共同 τ 只沿 ρ_M=τκ_M 的一条射线运动；任意 search-only 矩阵级 ρ_M 可在多维空间运动。不能逐行自选 ρ 并把它称为矩阵级规则。

## 3. 谱范数不确定集合的精确解

假设 C=Cᵀ⪰0，ρ≥0。集合：

\[
\mathcal U_\rho(C)=\{C+\Delta:\Delta=\Delta^T,\ \|\Delta\|_{op}\le\rho,\ C+\Delta\succeq0\}.
\]

对任意 e，|eᵀΔe|≤||Δ||op||e||²≤ρ||e||²，因此

\[
\sup_{\widetilde C\in\mathcal U_\rho(C)}e^T\widetilde C e
=e^TCe+\rho\|e\|^2.
\]

e≠0 时取 Δ*=ρeeᵀ/||e||²：对称、PSD、谱范数ρ；C+Δ* PSD，达到上界。e=0 时损失恒0，取Δ=0即可；ρ=0 时退化为原目标。即使额外要求 Δ PSD，上述最优值不变。C PSD 确保集合非空；浮点保存的 H 未做本轮 PSD 特征值验证，不能把未验证矩阵的 PSD 当作数值检查结论。

完整 E 的共享扰动同样精确：sup tr(E(C+Δ)Eᵀ)=tr(ECEᵀ)+ρ||E||F²，取Δ=ρI可同时达到所有行的上界。因此这个各向同性 robust 目标本身不产生新的行间耦合。

如果另要求 tr(Δ)=0、固定对角、特定文章混合凸包或其他结构，上述证人可能不可行，不能沿用此精确式。本轮不修改集合来制造新方向。

确定性条件 ||C_population−C_search||op≤ρ 一旦成立，就给出每个 e 的总体平均风险上界；它不保证被选择候选的真实风险优于ρ=0，不保证单文章、跨领域或 PPL 改善，也不说明怎样从 search 样本估计一个覆盖该事件的ρ。

## 4. joint minimizer 不可改变的证明

若候选 a 同时达到 min b 和 min m，则对任意候选 j 与ρ≥0：

\[
J_\rho(j)-J_\rho(a)=(b_j-b_a)+\rho(m_j-m_a)\ge0.
\]

所以 a 始终属于 argmin。若共同最小者唯一，则ρ>0时选择唯一；ρ=0仍执行原输出损失并列规则。一般有并列时须区分“始终是最优之一”和“固定 tie-break 始终选同一个”，不能由集合结论擅自推后者。本轮原ρ=0选择及精确正ρ路径核验，31个单位确实全程不换选；17个存在切换。

此外，线性惩罚只能选到下凸包支持候选；Pareto 未被支配并不普遍意味着某ρ下可选。本次480个候选只有65个有效区间。不能期望ρ估计创造候选bank中不存在的误差方向。
