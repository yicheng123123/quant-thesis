# v52 REPORT

**结果 D：初始化复核通过；完整模型质量验证因真实前缀传播错误停止。**

## 1. 完成与停止

普通沙盒仍失败，已获授权的沙盒外执行入口可用。沿用该入口，未修改全局权限、未继续结束自动重启的辅助进程。Sol high 为请求设置，接口不能确认实际模型设置。

新增修改仅在原初始化跨精度检查失败时，对失败行做独立 raw-X FP64 重构损失复核；保留原 torch.isclose(rtol=2e-5, atol=1e-12)、C 构建、候选顺序、接受规则、真实FP16增量与预算。复核不替换原loss、p、码或缓存。见 LOCAL_REPAIR.diff、DATAFLOW.md、PROTOCOL.json。

row122：原检查误差4.8755117781560e-12，阈值2.9716946071256e-12，仍失败；独立FP64与二次型误差3.3748968992164e-21，通过。复核前后状态逐字节不变；8个CPU用例覆盖正常、错误C/输入、非有限值和原路径一致性。该回归复现的是历史失败输入，后续已证明它的跨block来源错误；不能将此回归当作真实前缀的验证。

共执行96依赖组、168 Linear、304128行；block0四组继承v51，不重复修复；layer1原未提交qkv从M2开始。记录接受1786990次、增量评价4802065669次、候选拒绝0次、原候选FP64复核17次。新增初始化复核触发126矩阵、2876行。这些是实际执行账本，layer1及之后不代表冻结完整模型目标下的有效修复。

## 2. 硬错误的独立证据

当前 Transformers5.16.1 decoder layer 返回 Tensor [1,2048,896]。冻结 runtime_v46.advance_block 使用旧 tuple 取法 [0][0]，实际得到 [896]，赋入 NumPy [2048,896] 时广播首token到全部位置。

有界真实block0重放（不修复、不评测）验证128/128文章受影响，且block0部署权重与v51继承结果一致：

- 错误广播边界hash：4e57d3e76ea68915e7efcf56cd5136991aff87dcc35b974b51657e9a3f3f2d19，同时匹配v51/v52原记录。
- 真实全序列边界hash：9488e56e397608a5d19291595f43645135ad6b6ed477cd6d97eeba7398bfeed6，不同。
- layer1已提交qkv的X hash：8564a0fb02f4b6fcb7ea82e184e13ec788cea964666974cdeddf5014ef1798cd，匹配错误广播路径。
- 真实layer1 qkv X hash：70cd41104db3769a3d434c27d4612a41becf4be98439c6328c6010e5b6eae7ac，不同。

hash匹配只证明同一错误路径可复现，不能证明其真实传播来源正确。来源包装使用明确run/block/依赖组、真实hash、shape/dtype；未以id(X)作为证据，C hash仅核验，无新增复用。这个包装未改变返回值或状态，但也不能替代前缀接口正确性核验。详见 PREFIX_COMPATIBILITY_DIAGNOSIS.json、diagnose_prefix.py。

该错误不在本轮仅允许的初始化失败处理修改内。停止而未自行改advance_block、修改库环境、重新修复或放宽数值规则。历史v51保持D、文件hash未变。

## 3. 部署与质量

全部矩阵局部终态 raw-X FP64核验及code/grid/pack/reload输出回归通过。保存模型SHA：bfe456fbe7135e5270b770665e8a994730276762af05cc2ea51a581656400480。它只作为诊断产物：MODEL_VALIDITY.json 和 CHECKPOINT_LOCK.json 标记为无效正式模型。具名部署张量与原R2存在差异，因此不能复用原R2 PPL。

原配置断言因Transformers序列化字段迁移而停止；原记录保留。三份配置经同一AutoConfig加载后，除原有_name_or_path/transformers_version外全部字段一致，未扩大忽略项。这仅说明配置语义字段一致，不能证明block返回接口一致。

收尾适配器两次接入错误发生在首个PPL前向前，记录、源码与开销全部保留。第三次WT2评测在发现前缀错误后停止；C4未启动。没有完整PPL、完整原始窗口或可用的新NLL结论。全部日志保留，进度日志每25窗口一条，不能重建缺失窗口。未删除窗口、重选模型、读取P3、复用旧PPL为新质量结果或进行第二次成功评测。

QUALITY_COMPARISON.csv列出已核查历史M1/M2/原R2/v49的PPL与valid-token加权NLL；本次两语料为空，不能回答是否保留R2质量。

## 4. 实际成本（秒）

- 历史v51：657.305；v52新增活跃作业：4621.148。
- 共同累计：5278.453/10800；剩余5521.547，未重置预算。
- 新增回归31.458；正式运行4406.961；三个部署收尾作业173.978（含两次失败27.718和中断PPL）；真实前缀诊断8.750。
- 实际执行24block局部修复阶段4541.045，包含v51 block0一次187.085及新block1–23的4353.960；恢复-only block0另计3.026。由于后续前缀错误，不称成功正式修复时间。
- 正式C构建timer1573.146；初始化必要独立复核（含检查/hash）197.491；来源hash/包装timer122.938。均为真实打点，timer可能嵌套，不相加冒充独立wall。
- 历史M2构建单列2115.935；加本轮验证链累计7394.388。

强停段采用身份记录写入时间＋完整Stop-Process命令实测wall作为保守结束上界，明确不是精确GPU kernel时间。中断PPL和具名张量比较的独立timer未落盘，标NOT_MEASURED，开销已计入活跃作业账本。CPU环境诊断独立总耗时NOT_MEASURED；当前58项来源哈希复核CPU3.912秒另列，不冒充GPU实验。应用关闭等待未计入。原R2净修复恢复链口径不完整，不计算精确加速比；也不与v49作无效质量/速度推广。

## 5. 交付与决定

冻结协议/源码：PROTOCOL.json、FROZEN_VALIDATION_LOCK.json、IMPLEMENTATION_SOURCES.json、LOCAL_REPAIR.diff。
数值与来源：CPU_TESTS.json、NUMERICAL_VALIDATION.json、INITIALIZATION_REVIEW_EVENTS.csv、SOURCE_REVERIFICATION.json、HISTORY_IMMUTABILITY_AFTER.json。
执行与成本：MODULE_SUMMARY.csv、逐行formal/rows、阶段STAGE_TIMINGS.csv、STAGE_COSTS.csv、COST_LEDGER.csv、ACTIVE_SEGMENTS.csv、ACTIVE_JOB_LEDGER.json。
停止证据：PREFIX_COMPATIBILITY_DIAGNOSIS.json、PREFIX_COMPATIBILITY_STOP.json、MODEL_VALIDITY.json、diagnostics下原配置停止及适配器失败。
复现入口与限制：REPRODUCE.md。

仅初始化数值机制得到本地验证。完整模型与PPL未有效验证，最终D。局部MSE下降不能替代PPL；经验规则不称理论安全保证、原创贡献或官方ReQuant复现。GPU已停止，不自动改协议、重新运行或创建新版本。
