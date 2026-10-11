# v21.2 根因分类

## Verdict

`M3_NUMERICAL_ROOT_CAUSE_IDENTIFIED`

分类：`OFFICIAL_NUMERICAL_INSTABILITY_REPRODUCED`，对应 C：在审计的官方 LWC/AMP 语义下，loss-scaled FP16 权重梯度输出溢出。

范围限定为 Qwen2.5-0.5B-Instruct architecture port、冻结的 v21.1 状态、RTX 4060 Laptop。此分类不表示已经在官方原生支持模型或 T4 上复现。

## 已验证的因果链

1. block20 完成后，新建 L21 的 AdamW 和 GradScaler；LWC 参数全部初始化为有限的 4。三次独立重放得到相同 loss `0.15326565504074097` 和相同四个非有限 LWC 梯度张量、相同位置与值。
2. 观察到的 forward 均有限。loss scale 为 65536，scaled loss 为 10044.41796875。
3. backward 第 10 个事件（从 0 起计）的 `native_layer.mlp.up_proj.linear_operation / MmBackward0`，输入梯度和保存的操作数均有限，但返回的 FP16 权重梯度 `[896,4864]` 首次出现两个 +Inf。激活梯度仍有限。
4. 对相同 HALF 操作数重算 `X.T @ grad_output`。FP64 点积：W[567,62] = 122799.37525627553，W[3486,62] = 69136.64730513132，均超过 65504。FP32 超限位置、原生 Inf 位置和重算 HALF Inf 位置完全一致。
5. gate_proj 在后续独立的 MmBackward0 边界产生两个 -Inf。非有限权重梯度传入反量化权重与 affine 量化器导数，最终污染 gate/up 的 upper/lower LWC 梯度。最初的 NaN 并非 forward 或参数初始化产生。
6. 相同 AMP forward 改为单位 backward scale 的诊断，loss 完全相同，所有观察到的 backward 梯度有限。全 FP32 图、相同 65536 scale 的诊断，loss 为 0.15303973853588104，所有观察到的 backward 梯度也有限。

以上定位到算子边界和 FP16 结果表示溢出；没有探测 GEMM 内部 accumulator dtype，不作内部累加精度的断言。

## 激活贡献与量化器

up_proj 两个超限点主要由 token position 0、token ID 284、归一化后 input channel62 = 256.5 驱动。对应的 incoming scaled HALF gradients 为 478.75 和 269.5。channel62 的预训练 post-attention RMSNorm weight 为 8.6875，有限。

全局 forward scale 范围约为 0.000911–0.094877，没有触及 1e-5 下限；normalized weight 有限。LWC sigmoid(4) 约为 0.9820，没有异常训练历史。scale/zero-point 的非有限导数发生在上游 Inf 传入之后。hooks 未独立分离 autograd 引擎合并多条非有限分支时的内部运算，因此不将首个 NaN 强行归因到某个未观察的加法。

## 分类排除与保留

- A implementation bug：未确认。恢复、RNG、顺序、fresh optimizer/scaler 检查通过。
- B architecture-port bug：未确认。共享 quantizer、QuantLinear、quantize/utils、NativeScaler 文件与冻结官方源码逐字节一致；Qwen 使用原生 forward，MLP 公式与官方 Llama 相同。没有发现被遗漏的 MLP GEMM 数值保护。
- C AMP/numerical overflow：本轮已定位并数值验证。
- D data-specific pathology：sample0 不是唯一失败样本；0–3 均失败，但均共享首 token284。未排除语料共同前缀/格式的贡献，不能声称所有样本均失败，也不能声称数据损坏。
- E optimizer-state issue：不是本轮首次溢出的来源。AdamW 状态为空、step 未执行、参数未变。
- F quantizer derivative：传播并产生后续 NaN，不是首次 Inf 的来源。
- device specificity：T4 未执行，无法分类为设备特有问题。

## 停止状态

没有确认可仅作为 correctness correction 的实现/移植修复，故不生成 `MINIMAL_FIX_PLAN.md`。FP32 或调整 loss scaling 属于需要重新审议的数值协议选择，本轮没有修改正式 baseline。正式 M3 仍停在 block20；没有继续 L21、export、PPL 或 C4。

主要证据：`FIRST_NONFINITE_BACKWARD.json`、`GEMM_OVERFLOW_VERIFICATION.json`、`AMP_DIAGNOSTIC_COMPARISON.json`、`SOURCE_NUMERICAL_AUDIT.json`。
