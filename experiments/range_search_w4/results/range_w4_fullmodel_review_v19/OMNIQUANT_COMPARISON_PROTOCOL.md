# M3 公平比较冻结协议（本轮不执行）

**Gate：`V18_RESULT_AUDIT_PASS_M3_GO` 仅允许下一轮做隔离的 Qwen2.5 architecture port 与 smoke。** 正式 M3 须先满足 `M3_SMOKE_REQUIREMENTS.md`，且实测 step 成本/峰值显存可接受。当前没有任何 M3 质量结果。

## 对照矩阵

- M0：原始 FP16 Qwen/Qwen2.5-0.5B-Instruct，revision `7ae557604adf67be50417f59c2c2f167def9a775`。
- M1：v18 独立执行的 native dynamic GPTQ，W4A16、symmetric G128、block128、damp0.1、`desc_act=false`、`static_groups=false`、原生二阶补偿。
- M2：同一 GPTQ 算法及格式，动态 G128 range 上十个共享行 α 候选；每个 Linear 做十次完整候选，按行局部输出 MSE 选择后 stitch。使用冻结 v18 checkpoint，不重调 grid。
- M3：官方 OmniQuant commit `feffe8ea87d80f7bb57b6e25e7cff9dc950fcc14` 的 **weight-only W4A16 G128** 配置，LWC on、LET off、20 epochs、官方 asymmetric affine quantizer、block reconstruction、原有 loss/optimizer/epoch 语义；Qwen2.5 仅作架构 plumbing。先核验 official source SHA/Git blob，再移植到隔离目录。不得更改 LWC 参数化、loss、量化器、优化器或 block reconstruction 以追求运行/质量；若必须改核心，停止并重新定义 baseline，不能继续称其为 M3 OmniQuant。

## 共同数据、评测、成本

四法使用同一 revision/tokenizer 和 v16 已冻结的 128 篇 WikiText-2 train 文档各 2048 token；M3 应尽可能使用同样 262,144 calibration token 预算，并逐文档校验 token ID/order/hash。最终 WT2 test 299,078 input IDs/299,077 scored targets 与 C4 validation 固定预算 131,072 IDs/131,071 targets，均沿用 v18 相同 `benchmark.eval.ppl.evaluate_ppl`、max_length2048、stride512、无 chat template/额外 special tokens、FP16 SDPA/TF32 off。M3 在原生 Qwen2.5 dense-dequant FP16 checkpoint 中导出、保存、重载后评测；这只是质量比较，不能声称物理 int4 存储或在线推理加速。

同时报告 calibration wall、CUDA peak allocated/reserved、前处理/训练/导出分项、最终 checksum、reload parity；M1/M2 成本保留 v18 原始口径。禁止将官方自带的非重叠 WikiText 或随机 C4 evaluator 数字与本协议横比。严格保持最终评测集不参与 M3 训练、超参/epoch 调整或失败后补救调参。

**归因边界：** M1/M2 是 symmetric、GPTQ 二阶误差补偿；M3 是 asymmetric affine、LWC/STE 梯度优化和 block reconstruction，且没有 GPTQ compensation。M2 vs M3 是**方法级**质量/成本对照，不能归因于“纯搜索 vs 梯度”或“谁的 clipping 更好”。如需孤立量化格式或 clipping 机制，需要将来另立控制消融，此轮不授权。

在 M3 baseline 完成前，暂停 adaptive search、robust/structure-aware objective、group-range search、proxy search 及新模型/位宽实验。
