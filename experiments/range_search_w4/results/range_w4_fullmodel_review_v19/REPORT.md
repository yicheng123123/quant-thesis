# v19 独立全模型结果审查

**最终 verdict：`V18_RESULT_AUDIT_PASS_M3_GO`。** 这只授权下一轮按冻结协议实施隔离的 OmniQuant Qwen2.5 port 与 gated smoke；本轮未运行 M3。v18 在当前冻结模型、校准和两条评测流上，M2 相对独立 native GPTQ M1 的模型级数值改善可信。尚无统计显著性、跨模型泛化或优于 OmniQuant 的证据。

## 审查方法与 correctness

本轮只读取 v16–v18 冻结产物和源码，运行 `audit_v19.py` 的 CPU 侧独立复算；没有 GPTQ 候选调用、模型 forward、PPL 调参或 v18 写入。`correctness_checks.json` 为 **PASS**。重新核对 48 个 layer checkpoint 的状态 hash 链、336 个模块状态、168 个 M2 候选损失矩阵、全部 336 个已提交 Linear 权重与 M1/M2 最终 safetensors 完全相等；75 个预选 q/k/o/up/down 行的 W4 G128 scale/zero/g_idx 整数格点反量化精确成立。M1/M2 各自 24 层、168 Linear、304,128 行。

**M1/M2 独立性：** `run_formal.run_method` 分别从原始 FP16 checkpoint 调用 `load_model`；M1 单独执行 native GPTQ，M2 单独从 pristine 模型起步。两法层 0 初始 qkv 输入与 k_proj Hessian hash 相同，但后续维护各自 checkpoint/activation 命名空间和模型状态链，最终 safetensors hash 分别为 `3678f867…` 与 `810d6329…`。M1 不是 M2 的 α=1 副产物。M2 在每个依赖边界使用自己已提交的量化前缀。

**原生语义：** α=1 在 v17.1/v17.3 接口没有 range hook；H、damping、原生 Cholesky/Hinv 路径、GPTQ compensation、量化器与 g_idx 保留。正式代码 bundle hash 与正式 manifest 和 48 个层 checkpoint 一致。35 次完整 Linear 原生 reference，覆盖固定层 0/6/12/18/23 × 七投影，每次 weight/scale/zero/g_idx exact；时间记录实数 35，另有 1,680 次 M2 full-Linear candidate、168 次 row stitch。这支持正式实现沿用认证语义，但不能替代对其余 133 个 Linear 的逐个 native reference 重跑。

**row stitch：** 当前 symmetric W4、G128、`desc_act=false`、`static_groups=false` 下，scale/zero 是 `[output row, input group]`，g_idx 仅由输入列确定 `[input column]`，没有跨输出行共享的量化参数。GPTQ 的 Hessian 对所有输出行共享，但补偿更新对行独立，因此 full-shape 候选按输出行 stitch 合法。v19 逐模块检查这些 shape、zero=8、G128 映射、候选 argmin/α=1 端点和最终 checkpoint；预选 75 行还重建了整数反量化。v18 保存全部候选**损失**，但只保存选中后的权重，未持久化每个 full candidate 的完整权重；因此 v19 无法仅从 v18 磁盘独立逐行比对“选中权重等于第 α 个候选权重”。v17.3 的真实 full-candidate cache 与独立逐行参考显示 weight/zero/g_idx 和 local output exact，少数 scale 仅在 `rtol=1e-6, atol=1e-9` 内相等；结合本轮正式源码/metadata/hash 构成支持。此限制保留在 `ROW_STITCH_FORMAL_AUDIT.json`。

**self-prefix：** 对 M1/M2 各自 24 层检查 q/k/v 同输入、up/gate 同输入、o 与 down 的依赖输入 hash，以及 23 个跨层 next-qkv hash 链，共 46 次跨层相等。源码顺序是 q/k/v 全部 commit 后 capture o；o commit 后 capture up/gate；up/gate commit 后 capture down；down commit 后进入下一层。没有 row-batch refresh、FP prefix 或 M1 prefix 复用给 M2 的源码路径。哈希和控制流检查不等同于重做 128×2048 GPU 激活数值，但与正式记录一致。

**数据分离与评测公平性：** v18 calibration、WT2 test、C4 固定预算 manifest 与 v16 对应文件 SHA 完全相同；128 篇校准文档的 source article ID/full article hash 均唯一，每份 token hash 复算通过。v16 在结果产生前完成全篇 hash 与最终评测文档分离检查。calibration 为 WT2 train，正式 WT2 为 test，C4 为另一数据集 validation 固定 token 预算。正式 manifest 显示量化在评测前完成，源码的选择/候选评分仅读取 calibration 激活；评测 token IDs 未进入候选选择。无法由这些 manifest 排除语义重复或模型预训练语料重叠，故不作更强声明。

**evaluator：** 六项均读取固定 token ID 流，同 revision tokenizer、FP16 SDPA、TF32 off、max_length 2048、stride 512、无额外 special tokens/chat template。v19 从六份 window CSV 独立重建每个 begin/end/new_targets/valid_tokens，按 `Σ window_loss×valid_tokens / Σvalid_tokens` 复算，而非平均 window loss。WT2 为 582 窗/299,077 targets，C4 为 253 窗/131,071 targets；所有 mean NLL 与 PPL 与 raw JSON 一致。M1/M2 保存前后 64-token logits max abs diff 都为 0，正式 PPL 加载**重载后的最终 checkpoint**，其 safetensors SHA 与逐层提交权重一致。两处 evaluator harness 修复限于 import path 和 stop-at-tail 窗口断言，未改 shared loss/掩码/聚合；修复前源码快照未单独归档，精确 before/after diff 无法从仓库独立生成，见 `EVALUATOR_FIX_AUDIT.md`。

**SDPA warning：** 正式日志有滑窗注意力提醒。当前三法同一 FP16 SDPA forward；Qwen2 `sliding_window=32768`，`use_sliding_window=false`，评测窗口最多 2048，因此没有看到该 warning 对 M1/M2 造成不同处理。它仍是绝对评测语义 caveat；不能宣称相对比较绝无数值交互。详见 `SDPA_WARNING_ASSESSMENT.md`。

## 独立数值复算

- **WT2 test：** M0/M1/M2 PPL 分别 `12.4634094959 / 13.8843695729 / 13.7605560510`。M2−M1 PPL `−0.1238135218`，相对 `−0.8917475235%`；NLL/token `−0.0089574739`。相对 M1→M0 的 PPL 劣化，M2 恢复 **8.713370898%**；若以 NLL 劣化为分母则恢复 **8.296522692%**。
- **C4 validation 固定预算：** M0/M1/M2 PPL `19.2485558180 / 22.2339244501 / 22.0328831641`。M2−M1 PPL `−0.2010412860`，相对 `−0.9042096300%`；NLL/token `−0.0090832242`。PPL 劣化恢复 **6.734219816%**，NLL 劣化恢复 **6.299779165%**。此数值不是全量/标准 C4 PPL。
- **代价：** 正式 M2 1,680 次 full-Linear GPTQ candidate，3,041,280 个 row-alpha score slot；M2 校准 wall `2115.9353341 s`，M1 `648.8761009 s`，比值 `3.2609235×`。正式进程 peak CUDA allocated/reserved `2,347,887,104 / 2,684,354,560 bytes`（约 2.19/2.50 GiB，含两法正式进程）；不能当成 int4 在线推理内存。v17.2 48.79 小时旧估计的**候选调用数本来正确**，错误在 capture/refresh 成本模型；v18 已使用纠正后的实际单位。

## α 与投影结构

从 `selected_alpha.csv` 独立重算 304,128 行：α=0.9 为 226,034，0.8 为 54,155，1.0 为 20,310；0.7/0.6/0.5/0.4 分别 3,044/554/30/1，0.1–0.3 为 0。均值 `0.8862821`，中位数 `0.9`。283,818 行 α<1 与严格 search loss 改善一一对应；20,310 行 α=1 与精确持平对应，无搜索损失变差行。局部平均相对输出 MSE gain 为 `12.826748%`。

七类投影的完整 α histogram、均值、α=1 比例和 α≤0.7 比例见 `PROJECTION_RECALC.csv`。例如 k_proj 的 α=1 比例约 29.33%，q_proj 约 19.81%，gate_proj 约 6.93%，down_proj 约 3.06%；这是结构描述，不能直接解释为因果机制。局部与整体结果方向一致，但一个 full-model 点不能证明局部收益幅度可预测 PPL，也不能证明广泛泛化。

## M3 决策与 STOP

当前 0.89% WT2 / 0.90% C4 相对 PPL 改善、8.71% / 6.73% PPL 劣化恢复，在 3.26× M1 校准成本下，足以把 M2 作为**值得比较的已实现 baseline**，进入 OmniQuant M3 的受控 port/smoke 阶段。M3 若成功，比较的是整体方法：M1/M2 为 symmetric GPTQ 二阶补偿；M3 官方 weight-only 配置为 asymmetric LWC、无 GPTQ compensation、block reconstruction。不能把两者差异归因于 clipping 算法、梯度与搜索的单一变量。正式 M3 需先通过 port 语义和成本烟测，按 `OMNIQUANT_COMPARISON_PROTOCOL.md` 执行共同数据与 evaluator。其他新方法实验继续暂停。

**本轮 STOP：** 未执行 OmniQuant、M2 重跑、PPL 调参或额外模型实验。审查结论只覆盖当前冻结 checkpoint 与两套冻结评测流。
