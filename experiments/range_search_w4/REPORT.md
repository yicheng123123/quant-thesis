# W4 范围搜索：基线与代表层诊断（2026-09-29）

## 审计与协议

仓库没有项目级 `AGENTS.md`；仅依赖目录中的第三方包带有同名文件。主协议参照 `practice/full_w4a16/paths.py`、`11_quantize_group_size_ablation_todo.py`、保存的 G128 metadata 和 `benchmark/configs/gptq_g128.yaml`。历史可核验的整模型记录为 Qwen/Qwen2.5-0.5B-Instruct、FP16 加载、W4A16、对称 G128、64×512 WikiText-2 train 校准、damp 0.1、`desc_act=False`、`true_sequential=True`、`batch_size=1`，固定校准池 SHA256 为 `2bd6cacca4c7894b00d8779d76fc9b1d5776345bd69bc0d981054b2dbf579a76`。历史 formal WikiText-2 test 评测（2048 窗、stride 512、299077 targets）保存 FP16 PPL 12.463619、G128 GPTQ PPL 13.846270；它们不是本轮方法的 PPL。

仓库 `quant.py` 的所谓对称 MinMax 是 `[-7,7]` RTN。实际 GPTQModel 2.2.0 普通 affine 量化器用 `[0,15]`、round-to-even、clamp，并在 `sym=True` 时使用 zero-point 8；scale 是估计边界宽度除以 15。全零行回退到 `[-1,1]`；全正行还有 GPTQModel 特有的 `xmin=0` 行为。默认 `mse=0`，因此历史 GPTQ 没有额外内部 MSE clipping，但在每个动态 G128 组开始时调用 `find_params`，对**已被前序误差补偿修改的权重**重新估计范围。直接给原权重找到的外部范围会被覆盖。`benchmark/adapters/gptq.py` 只加载已量化 checkpoint，没有范围注入接口。本轮另建固定范围入口，不改历史适配器或 checkpoint。

本轮冻结协议在 `protocol.json`：同一模型本地 revision `7ae557604adf67be50417f59c2c2f167def9a775`，RTX 4060 Laptop GPU，模型 FP16，量化/Hessian/局部误差 FP32；层 0、12、23 的 `self_attn.q_proj`，各取输出行 0、448。按 WikiText-2 **完整文章**先选 8 个至少 128 token 的文章；前 4 篇 search，后 4 篇 held-out，每篇只取前 128 token。文章和 token 哈希见 `results/range_w4_local_v7/manifest.json`。seed 42；最终 WikiText-2 test 未用于候选选择。输入 `X` 来自 FP 模型的投影前激活，不是前序量化层传播的输入。代表行只覆盖这个局部条件，不能代表整模型误差传播。

固定 GPTQ 使用 `desc_act=False`、damp 0.1、block 128，复用同一 `H`/逆 Cholesky 因子；每个候选从同一原始行重新开始，保留 GPTQ 列间误差补偿。给定外部范围后不再调用 `find_params`。在这个配置下 `H` 跨输出行共享，而各行误差补偿独立，因此逐行评价数学可分。对有死输入列的情况程序直接报错，避免与 GPTQModel 的清零分支语义不一致；这批数据没有触发。

对称主实验合法地只搜单比例 `alpha=beta∈{0.1,…,1.0}`，含 `(1,1)`。非对称格式单独做二维控制：`alpha,beta∈{0.1,…,1.0}`，与非对称 10 候选单比例、0.01 步长的 100 候选单比例比较。每行对称 10/10、二维 100/100、密集单比例 100/100 个候选（去重前/后）；实际 scale、zero-point 和选择值保存于行 JSON/`methods.csv`。不得把对称与非对称格式的绝对差当成搜索收益。

## 路径与结果

A 是 MinMax 参数后运行**固定参数** GPTQ；B 分别按 RTN 权重 MSE 或 RTN 输出 MSE 选择范围，再运行同一固定 GPTQ；C 对每个候选运行同一固定 GPTQ，按 search 的 GPTQ 输出 MSE 选择。`L_output = mean((X(W_hat-W)^T)^2)`，`L_weight=mean((W_hat-W)^2)`。原始逐行数值与失败/未改善样例均在 `results/range_w4_local_v7/layer*_row*.json`；`summary.json` 由原始 JSON 独立汇总。

- **对称主协议：**相对 A 的 held-out 输出 MSE，中位改善 B 权重代理 30.69%（6/6 行改善），B RTN 输出代理 27.03%（4/6 改善、2 行持平），C GPTQ 目标 43.26%（6/6 改善）。这说明在选定行和文章上缩范围有局部收益；RTN 所选范围通常可转移到固定 GPTQ，但代理目标与 GPTQ 目标并不一致。C 相对 B 输出代理的 held-out 结果为 5 行改善、1 行持平，不能据此承诺其他行或整模型也改善。
- **二维自由度：**仅在非对称格式内部，二维 100 候选 C 相对 10 候选单比例的 held-out 中位改善 1.45%（3 行改善、3 行持平）；相对同样 100 次候选评价的密集单比例，中位改善 2.55%（4 改善、2 退化）。B RTN 输出代理的二维方案相对 10 候选单比例为 2 改善、4 退化，中位 **−7.74%**。因此目前没有稳定证据支持“独立左右端点”本身胜出；预算和目标选择的影响不能混写成二维收益。
- **组间耦合：**G128 RTN 独立组搜索在 42 个组的候选排序中，局部组损失与整行损失在 search 的 36 组、held-out 的 34 组选择不同的最佳比例。定义 `e_g=X_g(q_g-w_g)`，直接保存 `Σ||e_g||²`、`||Σe_g||²` 及差额。1 次预设坐标 sweep 只替换当前组 RTN 权重，接受 22 次 search 改进；held-out 有 5/6 行改善，但第 23 层行 448 **变差**。该缓存残差经直接重算检查。另对 MinMax 和独立组选解各完整重跑一次固定 GPTQ：独立组选解的 held-out 整行 MSE 为 6/6 改善。这还不能证明 RTN 坐标调整在 GPTQ 下也有效；若改组后重跑 GPTQ，必须重新计算后续受影响组。

“网格包含 MinMax”只保证相同 search 数据、相同评价流程、相同目标下最小 search 损失不高于 MinMax；本报告没有把它延伸为 held-out 或 PPL 保证。代表行是事前固定，未按收益挑选。

## 成本、正确性与边界

v7 代表实验的程序内总耗时 18.40 秒（含文章读取、FP 前向捕获、Hessian/逆因子和候选评价）；其中捕获 2.05 秒。GPU 同步后计时。捕获峰值 allocated 1,044,917,760 bytes，reserved 1,103,101,952 bytes；模型卸载后的搜索峰值 allocated 19,207,680 bytes。每行对称 10 个、非对称二维 100 个、非对称密集单比例 100 个固定 GPTQ 候选，6 行共 1260 次；G128 诊断每行另有 70 次初选、280 次排序指标、70 次坐标指标和 2 次完整 GPTQ 评价。候选按 16 个一批运行，行级 JSON 可中断恢复。共享批量计算的 `evaluation_seconds` 是该格式**全部候选的总计**，不是单个方法的独占耗时。

以本机 896 输入维 `q_proj` 每行对称 10 候选的实测速率，粗乘模型约 304128 个 Transformer Linear 输出行，可得逐行实现约十几小时的量级估计。批量跨行计算可能缩短时间，4864 输入维的投影及模型接入、packing、保存又可能增加时间；这**不是严格下界或完整运行基准**。因此本轮没有启动全模型候选级 GPTQ，也没有新 PPL 或下游任务数字。

最小检查 6/6 通过：对称/非对称网格与 GPTQModel 量化器一致；MinMax 与 GPTQModel 逐通道结果一致；自定义分组 scale/zero 确实被 GPTQModel 使用；批量候选与逐候选结果一致且原始权重不变；直接输出 MSE 等于等价写法；缓存分组残差与直接重算一致。请用 `README.md` 的命令复现。

初版 `results/range_w4_local_v1` 错将文章内 `= … ; … =` 表头视作新文章，可能造成文章跨 search/held-out；**其 held-out 结果无效**。v2–v6 是修正与计量补全的保留中间产物；仅 v7 用作本报告结论。历史 benchmark、基线和 checkpoint 没有覆盖。

## 推进判断

**A. 论文方向：**固定 GPTQ 的范围选择值得继续做一轮更宽但预注册的检验，优先验证“搜索目标与最终 GPTQ 目标失配”和“组间交叉项是否在 GPTQ 重跑语义下重要”。二维端点自由度目前证据弱，不能定为核心贡献。下一道证据门槛是更多预固定行/文章、整层与量化前缀输入、同协议整模型 PPL 和可接受的校准成本。完成成本优化和全模型接入前不扩展矩阵或下游任务。

**B. 工程项目：**已能展示 GPTQModel 参数语义审计、固定范围接入、候选状态隔离、GPU 分块评价、文档级留出、原始 JSON/CSV、可恢复行缓存、交叉项与残差校验，以及诚实的成本/负结果报告。它是**离线校准研究工具**；尚不支持在线推理加速或新整模型质量提升声明。
