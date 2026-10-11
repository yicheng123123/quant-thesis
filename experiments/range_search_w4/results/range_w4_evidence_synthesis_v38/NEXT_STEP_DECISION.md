# v38 下一步决策

## 决定

**收束当前纯范围搜索扩展；不建议再启动新的质量搜索或整模型实验。**

现有结果足以支撑毕设：M2 scalar10 在一个冻结小模型协议上通过量化正确性、完整模型评测与成本核算，给出相对 M1 的小幅 PPL 收益。扩展方向的证据不支持继续占用 GPU：dense46 跨 split gate 失败；quadratic scorer 有选型差异；v29 没有独立 direct-free hybrid；正则、固定 D 和 shared-D 均未过稳定性要求；v37 修复后 S0 退化 3.496%。

## 停止依据

- v25 pooled 局部收益为正，但 S0/S1 方向相反，已按冻结规则停止后续 pilot/full-model。
- v30-v33 是固定候选和已查看文章上的回顾性分析，不能作为盲测泛化证据。
- v37 有效 S0 是已查看 P1 的修复后复核；primary、projection 和 row-tail gates 失败；S1/P3/PPL 未运行，不据此否定 AWQ 总体。
- v28/v29 计时针对评分器/双评分阶段，未完成新评分器独立全 M2 calibration wall。
- M3 在 L21 非有限 LWC gradient 处停止，无完整模型/PPL；不是量化质量失败。

## 当前工作

把论文限定在冻结协议与现有证据：M0/M1/M2 质量和成本、dense46 跨 split 负结果、评分浮点选择风险、AWQ 分支修复后 S0 的适用边界。明确 C4 fixed-budget、回顾性复核、NOT_RUN 项和局部 SSE/PPL 的区别。除非毕业评审明确要求外部有效性证据，否则停止方法扩展。

本决策不提出待执行的新实验，不覆盖历史 checkpoint、公共量化实现或已有结果。

