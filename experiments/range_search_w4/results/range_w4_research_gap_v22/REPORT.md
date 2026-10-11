# v22 — Scientific gap audit

**Verdict：`NEXT_RESEARCH_DIRECTION_FROZEN`**

**PRIMARY：固定native dynamic GPTQ下的staged block/downstream-aware pure range search。**  
**BACKUP：同格式/同local目标下的2-region structured group α search。**

本轮完成研究链审计、只读M3参数诊断、三个判别实验设计和方向冻结；没有新算法实现、GPTQ、模型forward、训练或PPL。完成后STOP。

## 核心判断

剩余gap最合理的工作解释是 **M2 local目标的下游敏感度缺失，加上共享row α容量受限**；优先验证前者，后者为备选。当前无法识别两者贡献，更不能把M3优点全部称作“LWC更强”。格式（M2 symmetric / M3 asymmetric）、FP/quant teacher trajectory、联合梯度优化与GPTQ有无补偿也同时不同，完整因果归因尚缺消融。

| Frozen stream | M0 PPL | M1 PPL | M2 PPL | M3 PPL | M2/M3 recovery of M1→M0 PPL degradation |
|---|---:|---:|---:|---:|---:|
| WT2 test | 12.46340950 | 13.88436957 | 13.76055605 | 13.54525697 | 8.713371% / 23.865033% |
| C4 fixed-budget | 19.24855582 | 22.23392445 | 22.03288316 | 21.44542796 | 6.734220% / 26.412031% |

M3额外降低NLL .015769795/ .027024553，分别为现有M2降低M1 NLL的1.7605/2.9752倍。细网格若解释全部gap需达到这个模型级效果尺度；当前α histogram没有提供中间点或传递到NLL的证据，不能据其分布断言可达或不可达。故H3是第三优先的低成本诊断，而非预设答案。

## 新增实测：M3 range artifact

- CPU解析最终checkpoint：24blocks、168Linear、304,128rows、2,795,520row/groups、5,591,040 endpoint标量；finite/shape/配对和读取输入hash前后检查PASS。
- 全局lower/upper系数均值≈.94912、median≈.97576；54.63%行组内端点平均系数跨度>.05；深层跨度明显更大，L0/L12/L23 median=.0073/.0289/.2224。
- v系数均值≈.868、gate≈.962，projection差异明显；endpoint不对称均值差=.01376，仅7.15%组差>.05。统计按group/row权重明确区分，down有38组的机会效应保留。
- **这些不是M2 α等价值**：不同原始range、格式、补偿、目标；未计算真实scale/zero/weight反量化。只支持组/层模式存在，不证明它们带来PPL改善。

详细数据为 `M3_RANGE_STATISTICS.json`、`M2_M3_MODULE_RANGE_SUMMARY.csv`、预设行示例；输入SHA与quality全精度复算分别见 `INPUT_PROVENANCE.json`、`QUALITY_GAP_RECALC.json`。

## 保留的历史证据

v11异质/严重负尾且pooled收益集中；v12独立Gate FAIL，不能返回gain/margin调阈值；v13灾难并非主要由search文章不一致预警，方向×activation geometry恒等式是回溯诊断；v14coverage改善native却未系统消除clipping风险；v15quantized-prefix信号保留但只局部；v18/v19 M2 full-model有可信的小幅收益，12.826748%是search row-relative平均而非held；v21.5 M3完整有效且在这两条流优于M2。没有把这些负结果删除或改写为成功。

## 决策实验与工程约束

E1：blocks0/12/23的q/up，48预选row-split单位，同10候选只改选择目标；120 full-Linear GPTQ executions，15,360单文档blockforward预算，检验独立held block排名与gain。不能把独立block最优row一起stitch后当正式方法。

E2：up/down的2-region、两sweep；保持local目标，完整GPTQ重跑；同时比较原10与46-score scalar对照，分清score预算与full/per-row execution、真实wall差异。

E3：同参数/local目标的10→16固定grid，并报10点评价预算的refined覆盖；held oracle仅诊断，不参与选择。

三项共享确定性新完整文档split规则；本轮没有抽样或假造manifest。未来各项先preflight correctness和同步成本，1.5×保守估计≤6h/项、总≤24h，不通过STOP。成功的小型block诊断也不自动授权PPL；后续需完整Linear/staged与模型级验证。历史48h full-model gate保持不变。

## 为什么不优先旧方向

Gate独立泛化失败；多calibration未修复clipping风险；文章投票对高度一致的灾难case没有针对性证据；pure acceleration不再回答主要质量gap。当前M2实际校准2115.94s≈35.27min，1680 full-Linear候选产生3,041,280 row-alpha scores，不是304万GPTQ执行。若blockforward成为瓶颈再profile/cache，不提前替换原算法语义。

M3 exact calibration timer缺失；3052s日志跨度不与M2 perf_counter等价，所以不报告M3/M2 wall ratio或质量成本优越性。当前研究选择不依赖此不可恢复比值。

## 论文与工程价值

论文：有baseline、负结果/机制链及可检验问题；还缺目标同格式消融、capacity/grid/teacher/预算控制、held block稳定性、full-model迁移、跨模型重复。当前只支撑研究方向，不支撑新方法有效或新颖性/发表承诺。

工程：已实现可追溯量化框架、native parity、row-stitch、self-prefix缓存、恢复/导出、统一evaluator与成本审计，可支撑量化研究工程项目。新block方法尚未实现；dense-dequant结果不证明packed在线推理速度。离线搜索优化必须叫校准优化。

## 交付与重现

12个要求的专题Markdown齐全，另有CPU分析脚本、原始派生JSON/CSV、质量差距复算和输入provenance。所有新文件位于本v22目录，历史源文件hash前后相同。报告矩阵数字来自已完成冻结实验；新诊断只有参数读取，不将计划标记为运行。

从repo root重现本轮只读数值分析：

```powershell
& 'E:\quant-thesis\.venv-gptq\Scripts\python.exe' -B 'E:\quant-thesis\experiments\range_search_w4\results\range_w4_research_gap_v22\analyze_artifacts_v22.py'
```

`CORRECTNESS.json` scope为artifact analysis；不覆盖训练/量化correctness。原v18/v19/v21.5的correctness结论引用其历史证据。**STOP：不开始实现上述算法，不运行新模型评测。**
