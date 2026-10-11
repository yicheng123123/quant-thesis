# v26：M2 阶段 profiling 与完整 upper factor 缓存决策

**结论 4：profiling 不支持继续优化此处。** v17.3 已缓存 damped-H lower Cholesky 之后的中间 inverse；每个候选仍计算 lower 和最终 upper Cholesky。在两种预先固定的 L0 完整矩阵上，这两项仅占完整候选耗时 **0.73% / 1.95%**。预计可省的时间低于预热后重复计时波动，并且还需首次构建 upper factor。因此按运行前冻结的双矩阵、双门槛规则，**没有实施新缓存**。本轮没有实测缓存速度收益，也没有修改 M2 的量化算法或历史产物。

## 冻结对象与正确性

输入为正式 v18 M2 checkpoint 的 layer 0 自身量化前缀，在冻结的 WikiText-2 train 128 篇完整文档 × 2048 token 上捕获 `up_proj` 和 `down_proj` 的真实输入。模型 Qwen2.5-0.5B-Instruct revision `7ae557604adf67be50417f59c2c2f167def9a775`，FP16、SDPA、RTX 4060 Laptop；W4A16、symmetric、G128、block128、damping 0.1、无 act-order。原 M2 是 10 个共享 alpha 候选、每行按全部校准输入的局部输出 MSE 选取；本轮只对 α=1 和 0.8 测速，没有进行新搜索。完整定义和消费文件 SHA256 在 [PROTOCOL.json](PROTOCOL.json)。

两个激活文件和新建 H 的 SHA256 均与 v18 layer 0 checkpoint 所记值一致；没有缩小输入。`up_proj` 为 `[4864,896]`，`down_proj` 为 `[896,4864]`，每次评分覆盖各自**全部输出行**和 262,144 个 token。α=1 的原包装路径与原生 GPTQ 的 FP16 权重、scale、zero、g_idx 在两矩阵上逐项精确相同。原路径的 α=1、0.8 各重复两次，权重、量化参数及全行评分逐项相同；pristine W/H 未变。独立 dead/damp/factor 分解与既有 prepared state 的 damped H、lower 和中间 inverse 精确相同。详见 [PARITY.json](PARITY.json)。

新 upper-factor 缓存、原版/缓存版对照、候选反序、整块传播均为 **NOT_RUN**；前两项因成本门槛失败而不实施。两候选的行选择及拼接仅为计时诊断，没有提交为新模型；不能据此推断模型质量。

## 阶段计时与成本判断

预热：每个矩阵先执行一次 α=0.8 完整候选。正式阶段计时：每 α 两次，原版共 **8 次完整 Linear GPTQ**；对应 **23,040 个逐行评分**，每个评分遍历 128×2048 输入。另有 2 次完整预热和分账诊断中的 2 次原版 α=1、2 次原生 α=1，共计本轮 14 次完成的 GPTQ 执行；分账诊断还各有一次全行评分，所以全轮已完成 28,800 个逐行评分。候选级同步只在操作段的 CUDA event 外层完成，没有逐列同步。嵌套 CUDA event 的不重叠操作时间与整候选 GPU 时间对账，剩余列为量化、误差补偿、Python/内核调度及其他未归因工作，**不是纯补偿计时**。H/dead/damp 的独立分解为另外一次忠实复现，不能与嵌套候选计时相加。

| 完整矩阵 | H 构建 | 首次共享 H/inverse 准备 | 完整 GPTQ 候选均值 | 全行评分均值 | 候选+评分均值 | 候选 lower+upper | 候选重复波动范围 |
|---|---:|---:|---:|---:|---:|---:|---:|
| L0 up_proj | 0.485 s | 0.139 s | 0.387 s | 0.683 s | 1.070 s | 0.00282 s (0.73%) | 0.0131 s¹ |
| L0 down_proj | 2.722 s | 0.352 s | 2.165 s | 0.986 s | 3.151 s | 0.04222 s (1.95%) | 0.1795 s¹ |

¹ 同一 α 的两次预热后计时差中取最大值；表中数值是候选耗时波动，非统计置信区间。完整候选四次的总体范围分别为 0.0216 s、0.1954 s。冻结门槛要求两个矩阵 lower+upper 占比均至少 5%，且扣掉首次 upper 构建后的预计净节省超过两倍同 α 重复波动；两者均未通过。

原有中间 inverse 命中每候选 1 次，候选仍各调用一次 lower、upper；动态 `find_params` 各调用 8/39 次。独立阶段分解显示初次 inverse 约 12/170 ms，准备结果已在既有 M2 中复用。全行评分另一次细分：up 的 host memmap 拷贝 0.133 s、H2D CPU 调用包络 0.473 s、GEMM GPU event 0.355 s；down 为 0.297/0.339/0.346 s。GPU event 与 CPU 调用包络不是同一时间轴，传输 event 可包含调度间隙，不能相加声称占比。其余逐项原始计时、峰值显存见 [CANDIDATE_TIMING.csv](CANDIDATE_TIMING.csv)、[STAGE_TIMING.csv](STAGE_TIMING.csv)、[PROFILE_ORIGINAL.json](PROFILE_ORIGINAL.json)、[PROFILE_DETAIL.json](PROFILE_DETAIL.json)。记录的评分峰值 allocated 分别约 1.79/1.73 GiB；这是原版峰值，**不是新增缓存显存**。

本轮有首次捕获约 7.60 s、H 构建、候选预热、评分、诊断与文件哈希/IO；保守累计 GPU 作业墙钟 **84.59 s / 600 s**，包含两次失败尝试的预算扣除。激活文件哈希读取耗时 up 0.449 s、down 1.992 s，属于诊断 IO，不应逐候选外推。阶段和运行墙钟受模型加载、缓存命中、Python 开销影响，单项 GPU event 不加总为整个作业。失败一：原始 safetensors 是 BF16，正式模型加载会转 FP16，初次 pristine 读取遗漏转换；失败二：profiling 回调局部变量遮蔽。这两次均未产生正式候选结果，已在 v26 内修复并留痕 [失败 1](PROFILE_ATTEMPT_FAILURE_01.json)、[失败 2](PROFILE_ATTEMPT_FAILURE_02.json)。

历史 v18 整模型 M2 校准真实墙钟为 2,115.94 s、1,680 次完整 Linear GPTQ、3,041,280 个 row-alpha 评分位，不能把评分数当执行数。若仅对全模型中 48 个 `[4864,896]` 和 24 个 `[896,4864]` 的同形状模块，把本轮平均 lower+upper 时间乘原 10 点网格，乐观毛节省约 11.48 s；扣除每个模块一次平均 upper 构建约 10.90 s，仅相当历史全模型墙钟的 0.52%。这个**仅涵盖 72/168 个同形状 Linear 的粗估**，其余形状、prefix、IO 与实际缓存访问开销没有实测，不能当完整运行 benchmark 或已实现的加速比。只测了两个 α，未实测完整 10 点搜索。

## 范围与复现

缓存为何停止、作用域与失效条件见 [CACHE_SCOPE.md](CACHE_SCOPE.md)，复现命令见 [REPRODUCE.md](REPRODUCE.md)。没有新增 held-out、PPL、整模型量化或 checkpoint。当前证据指向候选主体和全行评分成本，不能据此自动启动其他优化方向。所讨论的只是离线校准成本，与在线推理速度无关。
