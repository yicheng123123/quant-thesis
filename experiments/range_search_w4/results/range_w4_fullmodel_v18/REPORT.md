v18 全模型实验与模型级评测已完成，达到冻结判定 **`FULLMODEL_M2_QUALITY_IMPROVED`**，现按要求停止，等待独立审阅。

**Correctness：PASS。** matched-shape 预检、1,680 次完整 Linear 候选、304,128 行覆盖、35 次 α=1 原生 GPTQ parity 检查、量化有限性与元数据、M1/M2 保存重载均通过。六项模型/数据集评测的 token 数与窗口加权 NLL 核验也全部通过。

| 方法 | WT2 PPL / NLL | C4 固定预算 PPL / NLL |
|---|---:|---:|
| M0 FP16 | 12.46341 / 2.522797 | 19.24856 / 2.957436 |
| M1 Native GPTQ | 13.88437 / 2.630764 | 22.23392 / 3.101619 |
| M2 Pure Search | 13.76056 / 2.621806 | 22.03288 / 3.092536 |

M2 相对 M1：WT2 PPL 下降 **0.12381（0.8917%）**，NLL/token 下降 0.008957；C4 固定预算 PPL 下降 **0.20104（0.9042%）**，NLL/token 下降 0.009083。两个量化模型在两套评测上仍都逊于 FP16。C4 是固定预算验证结果，不是标准全量 C4 PPL。

M2 共进行 **1,680 次整 Linear GPTQ 候选执行**，产生 **3,041,280 个 row-alpha 评分位**；二者没有混作同一种工作量。M2 实际校准耗时 **2,115.94 秒**，M1 为 **648.88 秒**，约为其 3.26 倍；合计校准耗时 2,770.94 秒。M2 预运行保守估算为 **1.338 小时**，通过未变更的 48 小时 gate。校准峰值 CUDA allocated/reserved 为 2.19/2.50 GiB。

M2 全模型 α 分布以 0.9 为主：α=0.9 有 226,034 行（74.32%），α=0.8 有 54,155 行（17.81%），α=1.0 有 20,310 行（6.68%）；均值 0.8863，中位数 0.9。局部行输出 MSE 平均改善 12.83%，而模型级 WT2 PPL 改善 0.89%；本次局部与模型级指标方向一致，但单个模型结果不足以证明局部收益能预测 PPL 收益。

交付物：

- [总报告](E:/quant-thesis/experiments/range_search_w4/results/range_w4_fullmodel_v18/REPORT.md) 与 [运行报告](E:/quant-thesis/experiments/range_search_w4/results/range_w4_fullmodel_v18/RUN_REPORT.md)
- [总览 JSON](E:/quant-thesis/experiments/range_search_w4/results/range_w4_fullmodel_v18/SUMMARY.json)、[模型对比 CSV](E:/quant-thesis/experiments/range_search_w4/results/range_w4_fullmodel_v18/FULLMODEL_COMPARISON.csv)
- [α 分布直方图](E:/quant-thesis/experiments/range_search_w4/results/range_w4_fullmodel_v18/ALPHA_DISTRIBUTION.csv)、[按层/投影 α 统计](E:/quant-thesis/experiments/range_search_w4/results/range_w4_fullmodel_v18/ALPHA_SUMMARY_BY_LAYER_PROJECTION.csv)
- [WT2 结果](E:/quant-thesis/experiments/range_search_w4/results/range_w4_fullmodel_v18/evaluation/WT2_RESULTS.csv)、[C4 结果](E:/quant-thesis/experiments/range_search_w4/results/range_w4_fullmodel_v18/evaluation/C4_RESULTS.csv)、[评测正确性](E:/quant-thesis/experiments/range_search_w4/results/range_w4_fullmodel_v18/evaluation/EVAL_CORRECTNESS.json)
- [M1 运行报告](E:/quant-thesis/experiments/range_search_w4/results/range_w4_fullmodel_v18/M1/RUN_REPORT.md)、[M2 运行报告](E:/quant-thesis/experiments/range_search_w4/results/range_w4_fullmodel_v18/M2/RUN_REPORT.md)

评测入口遇到的导入路径和窗口数断言问题已修复；失败尝试未纳入结果，完整六项矩阵随后重跑并通过。日志还记录了 Transformers 对 SDPA 滑窗注意力实现的提醒；报告保留了这一评测语义 caveat。v16、v17.1、v17.2 冻结文件哈希审计通过；v17.3 未发布上游期望哈希清单，因此 v18 记录其实际消费哈希，不作额外的哈希比对结论。