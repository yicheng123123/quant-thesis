# v38：纯搜索量化路线证据收束与毕设定位

审计日期：2026-10-05。只读现有代码、协议、原始表格和报告；未运行 GPU、模型前向、候选搜索、量化或 PPL。v38 仅新增本目录五个交付文件。

## 核心判断

最有力且可核验的正结果是 v18 M2 scalar10 动态 GPTQ 搜索：相对 M1 原生 GPTQ，WikiText-2 test PPL 下降 0.8917%，C4 固定预算 PPL 下降 0.9042%。这是单模型、单协议结果；量化模型仍逊于 FP16。M2 离线校准为 2115.935 秒，是 M1 的 3.261 倍。

稠密46点在局部 held SSE pooled 上改善 1.3845%，但 S0 改善、S1 退化，冻结 gate 失败。后续路线没有形成更稳健且通过门槛的结果。适合的毕设是对动态 GPTQ 范围搜索的质量、成本和迁移边界作可复现实证评估，不应包装成新算法。

## 证据分级

- **整模型已验证**：v18 M0/M1/M2 完整模型评测、量化正确性和保存重载。
- **局部已验证**：仅限记录中的层、行、文章、前缀和目标；不外推成 PPL。
- **回顾性诊断**：历史文章/held 已查看，不称盲测或正式泛化。
- **不同失败类型**：v35 是数据不足；v36 是 reference/capture 正确性问题；v37 是修复后 S0 质量门槛失败；M3 是训练数值停止。
- 数字以 EVIDENCE_TABLE 指向的原始 CSV/JSON 为准。v36 原 reference 下质量数字无效，不用于方法结论。

## A. 已验证的正结果

### 正式 M0/M1/M2 对照

| Dataset | M0 FP16 PPL | M1 native GPTQ PPL | M2 scalar10 PPL | M2 vs M1 |
|---|---:|---:|---:|---:|
| WikiText-2 test | 12.46341 | 13.88437 | 13.76056 | -0.8917% |
| C4 fixed-budget | 19.24856 | 22.23392 | 22.03288 | -0.9042% |

WT2 为 299,077 计分 token；C4 为固定 131,071 tokens/253 windows，不能称全量 C4。M2 覆盖 24 层、168 Linears、304,128 行；10 个 alpha 候选对应 1,680 次完整 Linear GPTQ 执行和 3,041,280 个 row-alpha score slots。M1 校准 648.876 秒，M2 2115.935 秒。量化模型均逊于 FP16。

### 评分阶段机会与边界

v27 在两条完整 L0 Linear 上选择和提交张量一致。v28 全模型双评分中，C 构建加二次型评分 73.704 秒，直接评分 718.524 秒；但 1/304,128 行改变 alpha，纯二次型不能替代原评分。v29 混合选择在双评分验证中匹配 direct，但触发器需要 direct scores；82/168 个完整 Linear 被复核，占 84.97% score slots。没有独立 direct-free hybrid M2。评分阶段时间不能称为完整校准加速或推理加速。

## B. 已验证的负结果

- **dense46 稳定性不足**：48 个固定 row-split units 的 pooled held SSE gain +1.3845%，S0 +3.0061%、S1 -0.4176%；W/T/L 20/12/16，最差退化 12.4467%。去掉 top-3 正贡献单位后 pooled 转为 -0.7213%，因此不扩到整模型。
- **纯二次型选择有浮点差异**：v28 的 L7 gate_proj row 884 将 direct alpha .8 选为 .9，W/scale 随之改变，选择等价 gate FAIL。
- **权重误差正则无收益**：v30 tau .1 和 1 分别使 pooled held SSE 增加 0.7364%/3.3767%；tau .01 复现原选择。
- **生成—选择分离不稳定**：v31 S0 独立选择 pooled held gain +2.978%，S1 -0.014%。这是看过的文章上的诊断，不是盲测，也不证明过拟合机制。
- **固定 D 与共享坐标尾部不稳**：v32 pooled 对照 +3.688%，但 25/48 行损失增加，中位逐行退化 2.404%，最差 -84.13%；剔除 L23/down 后总体退化 10.437%。v33 pooled +9.58%，但收益高度集中，移除 L23/down 只 +0.196%，且 3 行退化超过 10%，最差 -38.326%。均停止扩展。
- **AWQ selector + GPTQ 在修复后 S0 未过门槛**：v37 有效 S0 是已查看数据上的修复复核。G11 对 G01 held full-MLP MSE 增加 3.496%，不满足至少 1% 改善；projection 和 row-tail gate 也失败。仅适用当前 L12/S0 协议，不能否定 AWQ 或所有 scaling。
- **M3 未完成**：OmniQuant LWC 在 L21 出现非有限梯度，无完整模型和 PPL；这是训练/实现停止，不是质量失败。

## C. 尚未完成的工程或研究问题

1. M2 full-model 质量结果只有一个模型和一个冻结校准协议；跨模型/新独立 cohort 迁移未验证。
2. dense46 因局部 cross-split gate 失败，没有 own-prefix pilot/full-model PPL。
3. v28 pure scorer 有一行选择差异；v29 trigger 依赖 direct scores；没有独立 direct-free hybrid 校准运行。
4. AWQ+GPTQ 只有 v37 repaired S0 有效；S1 candidate/GPTQ、P3 held 和 PPL NOT_RUN。
5. M3 没有 complete model quality result。
6. 二次型 scorer 的全模型数字是双评分阶段对照，不能等同新 scorer 独立完成 M2 的端到端 wall。

### AWQ+GPTQ 证据链

v34 是 MIT AWQ pinned-source 静态审计和协议，不是运行结果。仓库 llmcompressor AWQModifier 的 W4A16 G64 full WT2 PPL 为 duo_scaling=false 14.48981、true 14.54975；它不是 MIT upstream，也没有 GPTQ，不能当作同一实验。v35 只剩 34 篇而冻结门槛是 64 篇，GPU 使用为零。v36 的旧 P1 held reference 受 norm 状态污染，fold-error reference 又漏折 gate/up 的 W*D；其质量数无效。v37 修复三项并从 pristine reference 复核锁定 S0；有效 S0 退化 3.496%，门槛失败；S1/P3/PPL 未运行。

## 结论

**A 已验证正结果**：v18 M2 scalar10 在现有单模型 WT2 test 与 C4 fixed-budget PPL 上小幅优于 M1；正确性和评测 PASS。  
**B 已验证负结果**：dense46 跨 split FAIL；纯 quadratic 一行选择不等价；正则无 pooled 收益；固定 D 尾部不稳；v37 AWQ+GPTQ S0 gate FAIL。  
**C 未完成**：M2 跨模型/独立数据迁移，dense46 模型级质量，direct-free hybrid 完整校准，AWQ 其他 split/PPL，M3 完整模型。

边界：M2 质量收益只适用于已测配置；局部 SSE、scorer-stage 省时、回顾性重选均不能替代新模型 PPL、独立盲测或在线推理性能证据。


