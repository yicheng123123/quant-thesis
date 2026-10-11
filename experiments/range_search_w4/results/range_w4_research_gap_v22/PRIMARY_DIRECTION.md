# PRIMARY_DIRECTION（冻结）

**Staged block/downstream-aware pure range search under fixed native dynamic GPTQ。**

先不增加组粒度、位宽、模型或新格式。保持symmetric W4A16/G128、原生dynamic MinMax估计时机与误差补偿、同一共享α网格，把范围选择目标从Linear-local提升为实际block输出重建；先通过同候选单行目标消融，再设计有限staged配置搜索。

## 正式 research question

> Under fixed symmetric W4A16 G128 native dynamic GPTQ, can selecting the same shared-alpha range candidates using downstream block reconstruction, rather than Linear-local row reconstruction, improve independent held-out block fidelity at a bounded calibration cost, and subsequently recover more model likelihood without gradient-based calibration?

这里“subsequently”是尚待检验的第二阶段，当前没有结果。不承诺闭合M3 gap；M3含asymmetric、组双端点、FP teacher alignment等混杂，不能作为单因素因果oracle。

## 选择理由

目标差异影响对同一error direction的权重，能通过E1保持候选和自由度不变做较干净的诊断。v18/v19已有工程与模型级baseline，v13/v15提供机制边界；M3整体更好提供更强比较对象。H2的组模式更直接可观测，但扩容量同时增加过拟合与预算风险；先测H1的信息价值更高。

不是结论“H1已证实”，而是对信息价值/风险的冻结选择。E1失败则保留负结果，下一授权任务切换BACKUP；不调teacher或层行追正结果。

## 实施前约束与推进

遵循 `MINIMAL_DISCRIMINATING_EXPERIMENTS.md` E1；先同block输入teacher，后才考虑FP trajectory teacher单独消融。Block目标不行可分，不能将每行独立block argmin直接全部stitch；正式接受须在共同context重算全block。qkv/MLP依赖边界正确refresh；所有候选fresh GPTQ state。不得复用被改动的compensation trajectory。

E1只判别目标ranking/heldout，成功后需要完整Linear/联合staged小规模验证，才可能授权全模型。PPL只按原冻结WT2/C4 evaluator检验，不参与选候选或超参。最终跨模型和预算/格式消融是论文证据，尚缺。

## Thesis value

可形成问题闭环：已完成M1/M2/M3 baseline → 已有error-direction机制审计 → 受控objective设计 → 参数/网格/预算/teacher/格式消融 → full-model与跨模型评测。**当前只完成baseline与机制动机/方向设计，没有新方法实现或收益。** 若held-out失败，负结果及local-to-global边界仍有毕设分析价值；是否足够形成方法论文取决于之后证据，不承诺原创或发表。

## Resume value

已有可讲述内容：原生GPTQ semantics审计、真实row-stitch与parity、可恢复full-model pipeline、1,680 full-Linear候选覆盖304,128行、calibration/Eval hash与窗口加权NLL、量化prefix依赖缓存、M3 port与AMP/resume/export correctness、成本分解。新block方法目前只能写“设计并冻结诊断协议”，实现和性能只能以后凭产物补充。Dense-dequant评测不证明int4 kernel推理加速；离线校准优化不能称在线推理加速。
