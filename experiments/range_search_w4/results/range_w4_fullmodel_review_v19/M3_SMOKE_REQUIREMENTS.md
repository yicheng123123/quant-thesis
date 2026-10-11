# 下一轮 M3 Qwen2.5 port smoke 的进入条件

仅在隔离的 port 副本和独立环境执行；保留官方 commit `feffe8ea87d80f7bb57b6e25e7cff9dc950fcc14` 的 LWC、quantizer、loss、AdamW/AMP、block reconstruction 和 20 epoch 语义。烟测需逐项保存输入、输出、dtype、容差、计时和 peak VRAM：

1. **A quant-off forward parity：** 原生 Qwen2.5 与 port 关量化路径在固定 token/attention kwargs 上 exact 或预声明容差 allclose，覆盖 RoPE、GQA、残差、norm、cache_position。
2. **B 七投影发现：** 每层 q/k/v/o/up/gate/down 正确替换，shape/bias 一致；embedding、norm、lm_head 未误量化。
3. **C LWC 参数存在：** 按原生 per-row-group bound factors 检查参数量/init，LET 关闭。
4. **D finite gradient：** 小样本 block backward 的 LWC 梯度有限；原始模型权重/norm/bias 保持冻结。
5. **E optimizer update：** AdamW 真实更新 LWC，记录前后参数 hash/数值；不得只验证 zero-grad。
6. **F tiny block reconstruction：** 官方 teacher/student 来源和 loss 保留；损失有限，并在微型步骤中下降或给出可解释的有限轨迹，不用最终评测调参。
7. **G dense export/reload：** QuantLinear 转原生 Qwen2.5 dense FP16 后保存重载；同输入 logits parity、权重/配置 hash 可核验。
8. **H shared evaluator tiny smoke：** 使用冻结 token ID 的极小预定前缀检查 evaluator 接线，不能将 smoke PPL 当最终质量或选择超参。
9. **I 成本：** 同步 GPU 的实际 step wall、完整校准外推、peak allocated/reserved 和 IO；若成本或语义 gate 未过，停止正式 M3。

若 architecture plumbing 不能在不修改算法核心的情况下维持 parity，应记为 `M3_NO_GO` 并等待重新定义，而非静默替换成新方法。
