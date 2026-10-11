# 简历描述（证据限定版）

- 在 Qwen2.5-0.5B-Instruct 上完成动态 GPTQ W4A16 对称 G128 scalar10 范围搜索整模型评测，覆盖 168 个 Linear/304,128 行，执行 1,680 次完整 Linear 候选 GPTQ；相对原生 GPTQ，WikiText-2 test PPL 下降 0.89%，C4 固定预算验证 PPL 下降 0.90%。M2 离线校准 2,115.9 秒，原生 GPTQ 648.9 秒；没有推理加速结论。

- 完成全模型评分数值审计：二次型 scorer 的构建加评分阶段由直接评分 718.5 秒降至 73.7 秒，但 1/304,128 行因浮点近并列改变 alpha；v29 混合规则在双评分中复现选择，但需对 82/168 个完整 Linear 做 direct recheck，尚无独立 direct-free 混合校准运行。因此只报告 scorer-stage 成本，不声称完整校准或在线推理加速。

