# 冻结链条与 v18 继承关系

- **v16** 在模型级结果之前冻结 Qwen2.5-0.5B-Instruct revision `7ae557604adf67be50417f59c2c2f167def9a775`、WikiText-2 train 的 128 篇完整文档各 2048 tokens、WikiText-2 test、C4 validation 固定预算、W4A16 G128 和共享评测器。`VERIFY_AUDIT.json` 的数据重建与跨集合文档 hash 检查为 PASS。v19 对三份冻结 manifest 与 v18 副本重新逐字节核验。
- **v17.1** 定位 native parity 初始偏差为 damping 前 FP32 对角均值的 contiguous/reduction 顺序，恢复调用原生 GPTQ core 的接口。完整 k_proj 与预定局部层级对 weight/scale/zero/output MSE 达到 exact parity；独立切片行实现没有被认证。
- **v17.2** 十个 full-shape α 候选从同一 pristine W 重跑，反转候选顺序、选中行 commit、无候选状态污染和 q/k/v 提交后的 self-prefix 传播检查 PASS。其 48.79 小时成本估计曾导致停止。
- **v17.3** 缓存只依赖 H 的逆矩阵准备，原生 `GPTQ.quantize()`、动态 group range 与 compensation 保留；十个候选对 v17.2 保存的 k_proj reference、row stitch 与 downstream H hash 通过。工作单位澄清为 1,680 次 full-Linear GPTQ execution 与 3,041,280 个 row-alpha score slot；每层四组共享输入、95 次必要依赖转换。
- **v18** 当前源码 bundle 哈希独立重算为 `4cda17690e9d0246899320c349abf2a123d70695a6352d7a33e1ee0561651975`，与正式 manifest、48 个 layer checkpoint 一致。正式脚本仍采用 v17.1 原生 core / v17.3 alpha-independent H 预处理、动态 group `find_params`、整 Linear 候选与依赖边界刷新。35 个 alpha=1 native reference 覆盖层 0/6/12/18/23 的每种投影，weight/scale/zero/g_idx 均 exact。完整机器复算见 `NATIVE_SEMANTICS_AUDIT.json`、`CHECKPOINT_AUDIT.json`。

**成本历史核验：** v17.3 `UPDATED_FULLMODEL_COST.md` 明确指出旧 48.79 小时估计使用了正确的 **1,680 次候选执行**；错误在将 native GPTQ 量化耗时误作 activation capture、额外整模型 refresh，以及未共享 Hessian 分组。v18 的 `PRECHECK_REPORT.md` 和正式报告正确区分 1,680 次 full-Linear GPTQ execution 与 3,041,280 个 row-alpha score slot，并没有复述“304 万次 GPTQ 调用”的错误。后续引用旧成本时采用 v17.3 的纠正。

证据边界：v17.3 没有发布独立的上游期望 hash 清单；v18 记录了实际消费文件 hash。v19 核验当前源码等于 v18 正式 bundle，但无法凭记录重建不存在的前置封存状态。
