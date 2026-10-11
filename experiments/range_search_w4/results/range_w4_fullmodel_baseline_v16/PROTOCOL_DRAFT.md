# v16 Full-Model & OmniQuant Baseline Protocol Audit

**Verdict: FULLMODEL_BASELINE_PROTOCOL_READY**。这是协议设计可进入实现/compatibility-smoke的判定，不是port实现、正式量化或PPL通过。证据清单与执行配置已冻结，smoke必须先通过，formal costs还需实测校正。完成本轮后 **STOP**。

## 已有证据与研究边界

当前读取v12 VALIDATION_REPORT、v13 REPORT、v14 REPORT、v15 REPORT/SUMMARY/correctness/RUN_REPORT和dynamic源码：Gate v1 independent validation FAIL（severe reduction42.03%、gain retention23.82%）；coverage hypothesis NOT SUPPORTED；v15 correctness PASS、TRANSFER_SIGNAL_PRESERVED。v15 FP/QPREFIX pooled local gain13.8722/15.8111%、median12.1283/12.9196%、alpha exact33/48、within-step47/48、35/37 positive保留、QPREFIX无<-5%row。v15 prefix来自native D，**不是M2 self-prefix**。这些局部输出重建证据没有证明full-model PPL收益。v13诊断只解释历史失败，未建立可用Gate。

主方法不用Gate，不更换模型/位宽/grid，不做coverage/robust objective/PrefixQuant或96-row扩展；v8-v15原始文件树由HISTORICAL_FINGERPRINTS固定并在交付核查。既有仓库其他未提交修改保留；本轮只新增本目录。未找到适用AGENTS.md文件，不推断不存在的仓库约定。

## 对15个必答问题的冻结答案

1. **下一主matrix**：M0 FP16 vs M1 Native GPTQ vs M2 GPTQ+Pure Search vs M3 OmniQuant-Qwen2.5 port；M4不纳入必需矩阵。完整配置见BASELINE_MATRIX.csv/PROTOCOL.json。
2. **Omni native Qwen支持**：没有；官方commit `feffe8ea87d80f7bb57b6e25e7cff9dc950fcc14` 只有既有家族分派，AutoModel支持不能替代校准/quant支持。
3. **最小adapter**：新增native-decoder七Linear替换工厂；修改omniquant discovery/device及kwargs捕获/重放；新manifest入口和dense export/eval bridge。精确文件/类边界见QWEN25_PORT_PLAN。
4. **算法语义是否保持**：计划保持官方W4A16配置LWC-on/LET-off、量化器/20epochs/AdamW/AMP/block MSE/FP teacher和quant student不变。未实现，必须逐项smoke证明。若修改这些算法项，不能仍叫官方方法port。
5. **M2 sequential prefix**：原生 `[k,v,q]->[o]->[up,gate]->[down]`；组capture时此前组/层已装M2自身selected rows；每候选独立完整dynamic compensation，per-row shared alpha，经最终该block的quant output传播下一层。不是FP-input或v15 native-D prefix。
6. **candidate总数**：168 linears、304,128 rows、3,041,280 row-alpha candidates；B32 serialalpha约95,040 calls，所有行/所有十alpha保留。
7. **calibration wall预算**：极小timing外推M2 core约8.09h；规划total12–24h，M1规划20–60min；它们不是正式测量。M3总61,440steps，没有one-block训练时间，不虚构点估计，按COMPUTE_BUDGET情景/next-step smoke核定。M2>48h/M3>24h停止正式启动，不能缩水协议。
8. **显存**：8GiB设备，M1/M2 active-block流式设计规划2–4GiB，M3规划3–6GiB；本轮只有budget core约192MiB峰值，完整训练峰值尚未验证，不能搬用v15错位reset后的prefix peak。
9. **单独环境**：需要隔离Omni校准环境，目标相同torch2.6/cu124、transformers4.49、tokenizer0.21.4；完整依赖lock待import smoke。primary质量统一在已冻结GPTQ eval环境，禁止修改现有环境或未经审计安装real_quant kernel。
10. **共同cal**：CALIBRATION_MANIFEST.json，128篇未用train文章×2048tokens=262,144，seed20261005，排除v8-v15共280篇IDs/hash，候选cal-only；same exact token_ids/order给M1/M2/M3。不使用最终test/PPL筛选。
11. **E1**：完整WT2 test，revision固定，299,078 input/299,077 scored tokens，max2048/stride512，token-weighted共享计分器，留尾，无special/chat。
12. **E2**：独立C4validation固定131,072 input/131,071 scored，官方固定revision第0shard docs32..1055；不是standard full C4。清单和文本已生成，无模型forward。
13. **方法级公平指标**：共同checkpoint/cal/eval/precision/目标scope下，PPL、deltaNLL/PPL、总离线calibrationwall、GPU allocated/reserved、磁盘representation、candidate/epochs/steps/trainableparams/失败重试可以比较。M3默认asym与M1/M2sym不同且无GPTQ，因此只能说整体方法结果。
14. **机制比较**：enumeration-vs-LWC、row共享-vs组独立、projectionloss-vsblockloss、symmetric-vsaffine、compensation差异需要另设控制对照。完整M3不能单独归因优化器；官方weight-only本来LET-off，不能把它称为人为删减。M4如控制同quantizer/同objective/同prefix需另命名，不在本轮执行。
15. **Luna Max下一任务**：在独立新执行目录，先固定环境并实现QWEN25_PORT_PLAN白名单的官方weight-only port、共同calibration loader和dense export；只做SMOKE_TEST_PLAN的L0 two-sample×64token two-step、关量化native/port forward、7模块/梯度/LWC/finite/reload/tiny127tokens检查；再实现M2原生sequential接口并做first32row十alpha/new-cal cost checks。输出SMOKE_RESULTS、patch manifest、pip lock、actual step/candidate time、成本复算，**先停在实现+smoke gate，不跑正式四模型PPL表**。不自动升级为full-model训练。

## READY 判定与可验证边界

Qwen mapping由config+safetensors shapes及本机native forward代码明确；Omni port范围明确；M2 own-prefix/row independence/dead处理/scale/zero/表示策略明确；候选数量精确且存在B32计时与有边界预算；cal/E1/E2 exact IDs/hash冻结；disjoint和历史完整性由VERIFY_AUDIT.json检查。`READY`仅满足这些**设计门槛**，并不宣称环境安装、Qwen port forward、packed精度或full-model质量已通过。

主要未验证项：Omni兼容环境/训练step成本、port teacher/student接线、M2整个168模块self-prefix的实现、全grid真实X批量等价、packed与dense差异、最终PPL。primary统一dense-dequant避免将不同packing误差混入当前range研究；standard GPTQ layout能表达per-row scales，但真实packed导出仍是smoke门槛。

## 论文与工程方向

论文方向值得进入受控模型级验证，当前只支持“局部信号跨native量化prefix仍存在”，不支持PPL收益、算法优于OmniQuant或新颖性/发表承诺。下一证据是同cal/eval的完整M0-M3质量与成本，以及失败case；不能追正结果换目标。

工程亮点现阶段是动态GPTQ原生parity、候选状态隔离、版本/数据hash冻结、原始结果复算、负结果门槛和预算/恢复设计。full-model integration、标准packed导出、Omni port尚属于设计项，不能写成已完成能力；搜索工程耗时改进属于离线校准，不是在线推理加速。
