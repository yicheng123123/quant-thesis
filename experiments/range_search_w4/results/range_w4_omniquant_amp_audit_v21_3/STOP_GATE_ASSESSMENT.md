# STOP gate assessment

分类：`STOP_GATE_TOO_STRICT`。

最终：`STOP_GATE_TOO_STRICT_RESUME_ALLOWED`，仅意味着控制流审计支持下一轮修正runner并准备resume，不是本轮执行授权，也不保证完整训练通过。

官方single-step：finite loss0.15326565504074097 → found_inf1 → optimizer.step未执行 → LWC及optimizer state不变 → scale65536→32768 → 函数正常返回。属于`RECOVERABLE_DYNAMIC_LOSS_SCALE_OVERFLOW`。

时序更正：v21.1并未阻止scaler处理，本来已经完成skip/backoff；随后post-scaler callback把残留的非有限grad当fatal，阻止官方下一sample。因此结论“too strict”成立，但理由是**拒绝已合法处理的overflow**，不是“在scaler前中断”。

同一disposable L21 optimizer/scaler连续samples0–3：一次skip，之后三次finite update，全部参数和AdamW state保持有限。每样本只跑一次，无重试，无手动scale，无LR/precision/quantizer变更。

v21.2的根因定位仍有效；“formal M3遇到梯度gate而STOP”是历史事实；将该event解释为官方算法本身fatal不成立。可以保留非有限grad/norm日志，不把known scaler skip的诊断Inf/NaN当参数损坏。

本轮没有patch runner、改变冻结协议、保存新formal checkpoint、执行正式resume。block20仍为最后合法checkpoint。完整epoch/后续层稳定性尚未验证；无需T4解释已验证的scaler行为。
