# v21.2 — Layer21 非有限梯度根因审计

日期：2026-10-01。本轮 verdict：`M3_NUMERICAL_ROOT_CAUSE_IDENTIFIED`。

根因类别：`OFFICIAL_NUMERICAL_INSTABILITY_REPRODUCED`，即审计状态下官方 LWC/AMP 语义的 loss-scaled FP16 权重梯度输出溢出。Correctness：**PASS，仅指根因审计；正式 M3 仍为 STOP**。

## 冻结对象与证据来源

模型 Qwen2.5-0.5B-Instruct，revision `7ae557604adf67be50417f59c2c2f167def9a775`；OmniQuant commit `feffe8ea87d80f7bb57b6e25e7cff9dc950fcc14`；W4A16 asymmetric affine G128、整数[0,15]、LWC on、LET off。RTX4060 Laptop，torch2.6.0+cu124，transformers4.49.0。正式参数使用 FP32，AMP 线性运算使用 FP16。

校准128×2048，seed20261005，manifest SHA256 `664419273f63df3c0b0746b18bb66b593a5384bbe6ba8afcd6f4905c9479aae3`。复用 block20 checkpoint、原始 tokens/order、原生恢复路径、RNG 与 FP teacher。没有重新采样。

已审阅 v21.1 REPORT/FINAL_STATUS/correctness/ARTIFACT_VALIDATION、STOP_DIAGNOSTIC、checkpoint manifests、training/failure logs、formal runner、v20 smoke/v20.1 preflight、冻结官方源码与 Qwen adapter。记录见 PROTOCOL、SOURCE_NUMERICAL_AUDIT 与 PROTECTED_HASHES。

恢复过程重新传播已完成的 block0..20，到了 L21 optimizer 初始化边界即捕获状态并停止；不训练这些 blocks。完整校准缓存用于忠实重建，只保留诊断所需0–3样本和层状态。父目录受保护源码/checkpoints/logs/reports 的最终哈希与审计前一致。

## 1. 失败是否稳定复现？

三次独立重放 loss 均严格等于历史 `0.15326565504074097`。同四个 LWC 梯度张量、非有限数量/位置/张量值逐字节一致。

有限的 q_proj upper/lower 梯度 hash 有差异，因此不声称全部梯度 bitwise deterministic。最初额外使用“全部梯度 hash 相同”的过严审计检查得到 FAIL，保留在 FAILURE_REPRODUCIBILITY_STRICT_GRAD_HASH_CHECK.json；按用户要求的失败身份（loss、非有限张量与位置）汇总为 PASS，没有改实验参数或重跑寻求通过。历史仅保存聚合梯度旗标，无法与历史 named gradient 逐项比较。

## 2. 哪些 LWC 参数坏？

`native_layer.mlp.gate_proj.weight_quantizer` 和 `native_layer.mlp.up_proj.weight_quantizer` 的 `upbound_factor`、`lowbound_factor`，共四个 FP32 参数张量，shape 均[34048,1]。

- gate upper/lower：每个34046有限、1 NaN、1 -Inf。
- up upper/lower：每个34046有限、2 NaN。
- 其余10个 LWC 参数梯度均有限。

位置均为 G128 flatten indices3969/24402，对应输出行567/3486、输入group0。完整逐参数、逐阶段统计见 LWC_GRADIENT_FAILURE_MAP.csv（3次×3阶段×14参数=126行）。数量相同的两个 gate 参数和两个 up 参数没有可据此判定的“严重度”排序；up_proj 是最早溢出的算子。

## 3. 第一次 non-finite 在哪里？

完整 D1 trace 观察298个 autograd nodes/298 backward events。首个有限→非有限边界为事件10（从0起计）`up_proj / MmBackward0`。

incoming gradient `[2048,4864]` 为 HALF、全有限、absmax478.75；保存的输入 HALF absmax256.5。返回 activation gradient `[2048,896]` 全有限；返回 transposed weight gradient `[896,4864]` 出现两个 +Inf。首个标记到的非有限 key tensor 是 FP32 `up_proj.dequantized_weight` 的梯度；它已经从上游 HALF 结果继承 Inf。

相同操作数验证：W[567,62] FP64dot122799.37525627553，W[3486,62] FP64dot69136.64730513132；FP16最大有限值65504。超限恰好两处，位置与原生 Inf 完全一致，HALF 重算也给出相同 Inf mask。事件17的 gate_proj 权重梯度出现独立的两个 -Inf。

这是对权重梯度结果表示溢出的确认，没有测量 kernel 内部 accumulator dtype。主证据是 FIRST_NONFINITE_BACKWARD.json 与 GEMM_OVERFLOW_VERIFICATION.json，操作数保留在 FIRST_GEMM_OPERANDS_D1_OPERAND.pt。

## 4. forward 是否有限？

所有观察到的关键 forward tensors 和 loss 有限，包括输入、teacher、7个 projections、attention output、residual、SwiGLU、学生输出、量化器 intermediates。代表 absmax：学生输入1696；post-attention norm FP32输出256.552734（GEMM输入HALF256.5）；gate46.65625；SiLU38.375；up48.1875；SwiGLU product1410；down1350；block输出346；teacher输出67.75。

SDPA内部 attention scores 没有单独物化，以保留原生 backend。因此结论是“全部观察到的阶段有限”，并不声称观测了每个 kernel 内部中间值。见 FORWARD_NUMERICS.json。

## 5. 是否与 AMP/loss scaling 有关？

是。正式初始 GradScaler=65536，scaled loss=10044.41796875。Inf 在 scaled backward 的线性权重梯度产生，早于 unscale 和 optimizer；unscale 不能把 Inf/NaN 变回有限值。原生 scaler 识别失败、跳过 AdamW、降至32768。

诊断 D3 保持同一 AMP forward/loss，仅单位 scale backward，所有观察到的 backward 梯度与 LWC 梯度有限。该操作只用于机制检查，没有作为正式数值设置应用。见 AMP_STATE、AMP_DIAGNOSTIC_COMPARISON。

## 6. FP32 diagnostic 是否有限？

全 FP32 对照是。D2_FULL_FP32 的所有 leaf 均 FP32，270个观察到的 backward events 全有限，loss=0.15303973853588104，仍使用65536 scale。

原始 D2 只关闭 autocast/promote计算，LWC 梯度虽有限，但官方 QuantLinear buffer 保留的原始未优化 HALF weight leaf 在末端 ToCopyBackward0 收到 Inf。此结果也保留，不能称全图有限。两种 FP32 对照均不是全流程 `--deactive_amp` 正式校准，输入/teacher 是冻结 HALF 数值提升。详见 AMP_NUMERICAL_STABILITY_ASSESSMENT.md。

## 7. 是否与 optimizer state 有关？

首次溢出不来自 optimizer。官方和 port 都每 block 新建 AdamW/scaler，block20 checkpoint 没有也不应继承下一层 optimizer moments。L21初始/最终 state均为空，exp_avg/exp_avg_sq/step尚未分配；step prehook计数0，参数不变。RNG恢复检查 Python/NumPy/CPU/CUDA 全通过。见 OPTIMIZER_STATE_AUDIT、RNG_RESTORATION_CHECK。

## 8. 是否 layer21-specific？

在有限对照内支持 L21 敏感性。真实 learned quant-prefix下 fresh L20 loss=0.07404229044914246（严格复现历史L20第一样本），梯度有限；真实 L21 非有限。

统一 native FP-prefix 控制、同sample0：L20 loss0.01764606684446335有限；L21 loss0.07143910974264145非有限；L22 loss0.021916182711720467有限；L23 loss0.06680014729499817有限。各层 LWC 都按官方初始化4。这不支持把问题泛化成所有深层均不稳定。

没有已训练 L21 checkpoint，所以实际正式 L22/23 quant-prefix不存在；此处明确是 FP-prefix诊断，不能认证后续正式层稳定。见 LAYER_SPECIFICITY.json。

## 9. 是否 sample0-specific？

不是仅sample0。L21各次从同fresh状态独立开始，samples0–3全部非有限；loss依次0.153265655、0.171205491、0.167916998、0.169040576。没有 commit。

exact IDs/hash/来源文章在 CALIBRATION_SAMPLE_AUDIT.json，均长度2048、ID有效、所有层复用相同样本。sample0 token SHA256为 `c78bbcff1a14dd8e68750ef774c10b342287e9393e7c047c6598c9c1a62b8fd7`。四个序列均以token284开头，不能排除共同前缀/格式贡献。

up_proj两个超限点主要由首token、input channel62的256.5激活贡献；该通道预训练RMSNorm weight8.6875有限。native FP-prefix也出现大但有限的输入。因此未发现数据损坏证据，不把“大激活”直接当作错误数据，也没有删除/替换样本。

## 10. T4 是否复现？

未执行。没有在Kaggle迁移/重放，不生成DEVICE_COMPARISON.json，不推断4060特有或跨设备一致。

## 11. 根因类别？

C：`OFFICIAL_NUMERICAL_INSTABILITY_REPRODUCED`，限定为当前 Qwen port、冻结状态和4060。共享算法文件与官方相同；upstream原生模型家族仅源码比较，没有运行。没有确认 implementation bug/port bug；设备特有性未建立。

量化器scale有限，无1e-5 clamp hits，无sqrt/log非法运算，L21初始LWC全4，与L0/12/20官方构造相同。后续量化器导数传播Inf并产生NaN，不能归因为首次量化器数值奇点。详见 ROOT_CAUSE_ASSESSMENT、QUANTIZER_NUMERICS_AUDIT、LWC_PARAMETER_STATE_HISTORY。

## 12. 是否存在保持语义的最小修复？

本轮未确认实现/port错误，所以不提供 MINIMAL_FIX_PLAN，也没有修改源码。改变AMP或loss scaling属于数值协议审议，不能私自套用为 correctness fix。官方 `--deactive_amp` 的端到端W4 dtype交接仍未验证，不能直接推荐续跑。

## 13. 正式 M3 是否值得继续？

仍有研究价值：已完成21/24 blocks，且首次失败的机制已明确。但当前冻结版本不能宣布可继续或完成。下一轮先决定允许的官方数值设置、恢复/重跑的公平性、完整dtype边界与cost/memory gate；本轮没有运行这些任务。当前正式进度仍为53760/61440 committed updates、87.5%，最后合法checkpoint仍是block20。

## 成本、交付与限制

共17次单样本 backward 诊断（含3复现、trace/操作数验证、layer/sample控制、全FP32补充）；实际AdamW step=0、正式commit=0。真实prefix重建26.293427秒，累计带hook/statistics的单样本重放11.732042秒；这只是部分计时，未计模型加载、FP-prefix capture、CPU审计、IO，不能写成整个审计总耗时或校准速度。诊断最大 allocated726469120 bytes、reserved832569344 bytes。

原始JSON、CSV、tensor fixtures、源码哈希、parent保护哈希和可复现命令均在本目录。correctness_checks PASS证明审计链一致；ARTIFACT_INTEGRITY_CHECK补充JSON/CSV/fixture/code哈希检查。父目录内容保护见PROTECTED_ARTIFACT_CHECK。

工程交付是固定checkpoint的最小失败重放、named梯度统计、原生autograd边界追踪、同操作数FP64验证和无更新的精度对照。本轮没有新PPL、最终M3模型、加速结论或发表承诺。

**STOP：正式baseline未改变，未完成layer21，未继续训练layer22/23，未export/PPL/C4。**
