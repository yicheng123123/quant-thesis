# v61：仓库 M3/LWC 完整校准成本与内存

状态：COMPLETE。本轮从原始 FP 模型开始；不加载历史训练参数，不运行 PPL。

## 实际方法及来源

采用历史 v21.4 实际 runner 及其 Qwen2 原生 decoder 组合适配器，依赖 OmniQuant commit feffe8ea87d80f7bb57b6e25e7cff9dc950fcc14 的量化/LWC代码。名称为“仓库 M3/LWC 实现”，不称完整官方 OmniQuant 复现。

模型 Qwen2.5-0.5B-Instruct，revision 7ae557604adf67be50417f59c2c2f167def9a775。正式 128×2048 WT2-train 冻结 token，原 manifest SHA 664419273f63df3c0b0746b18bb66b593a5384bbe6ba8afcd6f4905c9479aae3；逐篇和有序 token hash 已核验。W4A16、G128、每输出行连续输入组，非对称 affine [0,15]，zero 非固定8；最终为 dense dequant FP16，real_quant=False。

只更新每个权重组的 lowbound_factor/upbound_factor，初值4，sigmoid调节组内min/max；LET、aug_loss关闭，优化器不持有其他模型参数。目标为 FP 教师前缀的 block 输出与当前量化学生前缀 block 输出的 MSE。AdamW，LWC lr=0.01，weight_decay=0，betas=(0.9,0.999)，eps=1e-8；batch=1，20 epoch，固定篇次序，seed=20261005。每block新建AdamW和动态GradScaler，不补偿/重试overflow跳步。FP16模型/输入/teacher，优化前qlayer.float，原CUDA FP16 autocast及原GradScaler，提交前half。

历史完成链：v21.1提交block0–20，v21.4从block20恢复并完成21–23；共61440计划步，历史成功61438、跳步2。v21.5仅恢复最终摘要、导出/评测证据。历史0–20的成功步数来自有限owned梯度日志与scaler源码推断，缺少直接step hook/found_inf快照；本轮全部成功/跳步则由实际hook计数。旧内部总timer不可恢复，3052秒仅时间戳区间和，不能作为完整实测wall。本轮不缩减epoch，也不使用该旧区间替代测量。

本轮全程保留v21.4详细AMP/有限值/日志诊断；历史0–20采用较简诊断。优化时间包含这套实际执行检查，不能宣称是历史每段逐行执行的同一成本。仅新增阶段标记及外部采样，核心去除标记后的AST与历史源码一致。原末尾KeyError修复仅为返回已读manifest的摘要元数据；不改变模型/优化。IMPLEMENTATION.diff列出全部更改。

## 实测成本

- 纯优化循环及原检查：2349.203000 秒。包括优化器创建/训练、原有限值/AMP检查及epoch日志；裸GPU kernel/纯AdamW子项 NOT_MEASURED。
- 原omniquant完整调用：2380.7601884000032 秒，包括初始捕获、24个block教师/学生传播、训练、参数应用及组级检查点。
- 数据准备至最终化的完整校准流程：2384.344000 秒；data preparation, initial input capture, all teacher and student propagation, block setup, LWC optimization+existing diagnostics, quantization finalization, block checkpoints, cleanup/final checkpoint validation and dense finalization。不含加载、最终save及reload。
- 完整成功进程端到端（包括加载/检查、保存和reload输出回归）：2404.328000 秒。
- 最终保存：2.828000 秒；reload及64-token输出回归：1.687000 秒。
- 失败开销：38.547000 秒；所有正式尝试累计：2442.875000 秒。首次计量通知文件与采样读取发生Windows共享冲突，优化0步、提交0block；现场/账本保留，通知改为追加式日志，经200次并发读写检查后从FP重启。
- 主进程已覆盖阶段CPU时间（所有尝试）：2452.859375 秒；采样监督进程CPU：243.218750 秒。CPU时间不等于wall；阶段之外的Python导入/进程启动退出开销单列COST_ACCOUNTING，RSS时间线保留各进程CPU快照。
- CPU/NVML计量预检内部wall：22.312000 秒、CPU：13.312500 秒；不含Python导入和工具启动。后处理CPU账本另列。没有无计量对照，采样器对wall的额外影响NOT_MEASURED，不能拿采样进程CPU时间从wall中扣除。

阶段CSV同步计时，原训练代码已有逐步同步仍保留，新计量只在阶段边界同步，不称wall为GPU计算时间。calibration_cleanup合并cleanup、最终检查点读回及摘要IO，不按代码/运算量分摊。M1 648.8761秒、M2 2115.9353秒仅作背景，不给公平加速比。

## 内存

- 完整成功尝试PyTorch peak allocated：2,226,892,288 bytes / 2.073955 GiB。
- 完整成功尝试PyTorch peak reserved：2,321,547,264 bytes / 2.162109 GiB。
- GPU设备总显存采样峰值：3,722,502,144 bytes / 3.466850 GiB，含桌面及其他进程，不是本进程独占，也未把基线扣除后冒充allocated。
- 运行前设备显存基线：1,266,253,824 bytes / 1.179291 GiB。实际采样间隔中位数/P95/最大值：0.203 / 0.219 / 0.234 秒。
- 主模型进程RSS采样峰值：3,736,354,816 bytes / 3.479752 GiB。venv启动器及可见子进程分别记录；并发树RSS峰值与逐进程峰值见MEMORY_PEAKS，不将各独立峰值相加。
- GPU进程显存：NOT_AVAILABLE（当前Windows/WDDM返回不可用，空值不是0）。NVML设备总显存约200ms采样；CPU RSS同时采样。采样峰值不是精确瞬时峰值；短于采样间隔的子进程可能遗漏，明确记录该覆盖限制。
- 每阶段PyTorch峰值在reset前持久化，整程为所有阶段峰值最大值；失败阶段也保存。设备/RSS分阶段峰值和原始时间线独立保留。失败及正常退出均flush已有记录。
- 原runner摘要中的peak字段在新增分阶段reset后只代表末尾阶段，不作为整程峰值。原摘要保留，TRAINING_SUMMARY_WITH_MEASUREMENT_SCOPE补充口径，整程峰值以本报告和MEMORY_PEAKS的阶段累积为准。

## 环境、完整性及质量边界

实际进程使用 3.12.10 (tags/v3.12.10:0cc8128, Apr  8 2025, 12:21:36) [MSC v.1943 64 bit (AMD64)]；PyTorch 2.6.0+cu124；Transformers 4.49.0；CUDA 12.4；GPU NVIDIA GeForce RTX 4060 Laptop GPU。完整解释器、源码SHA、tokenizer、SDPA策略、TF32及有效配置见ACTUAL_RUNTIME。操作系统现为Win11 build26300，历史记录build26200；不抹去差异。接口无法确认Sol high设置。

COMPLETENESS记录24block/168Linear、全部61440计划位置的唯一性、成功/跳步、最终检查点SHA与原输出回归。最终模型为本轮新产物；即使局部/文件hash相同也不据此继承历史M3 PPL。本轮没有新质量结果，历史PPL不作为新模型测量。

独立CPU逐具名保存张量比较：168 个张量不同，详见HISTORICAL_M3_TENSOR_COMPARISON；本轮PPL未测量、不继承。正式输入文章/截断token及有序hash见CALIBRATION_SOURCE和CALIBRATION_SOURCE_ROWS；文章hash来自锁定manifest，本轮不重新解析原Arrow文章。实际Python工作进程映像与venv解释器路径分别见PROCESS_IMAGE_IDENTITY与ACTUAL_RUNTIME。

SOURCE_AUDIT、HISTORICAL_READONLY_CHECK与IMPLEMENTATION_LOCK_CHECK核对历史来源及冻结代码；不覆盖历史文件。成本只适用于这一次现有运行时下的完整校准及计量路径，不声称重复测量稳定性。

## 复现

已完成目录不能重新执行校准以覆盖结果。执行REPRODUCE_FRESH.ps1会在results下创建新的独立测量目录，复制生成器/计量源码，再从FP生成相同训练入口；不自行执行该脚本。现有目录/导出已存在时拒绝覆盖。

```powershell
& 'E:/quant-thesis/results/v61_m3_calibration_cost_memory/REPRODUCE_FRESH.ps1'
# 仅在有已提交block且运行未完成时恢复；账本保留全部PROCESS_ACCOUNTING文件：
& 'E:/quant-thesis/.venv-gptq/Scripts/python.exe' -B 'E:/quant-thesis/results/v61_m3_calibration_cost_memory/measure_once.py' --resume
# 仅CPU重算报告：
& 'E:/quant-thesis/.venv-gptq/Scripts/python.exe' -B 'E:/quant-thesis/results/v61_m3_calibration_cost_memory/summarize_measurement.py'
```

完整测量后停止，不启动其他方法重测或PPL。
