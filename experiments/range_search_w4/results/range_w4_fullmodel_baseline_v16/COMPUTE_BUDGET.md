# 计算和存储预算：实测与外推分开

精确计数来自checkpoint config和168个实际safetensors weight shapes，详见MODEL_INVENTORY.json/MODULE_INVENTORY.csv。24blocks ×7=168 Linears；每block12,672 output rows；共304,128 rows，**3,041,280个行-alpha候选**。量化范围357,826,560 weights；2,795,520 row-group scale/zero pairs。lm_head tied embedding排除。不能把row-alpha逻辑次数与GPU调用数混报。

## 已有计时

v14的K4/8/16/32各480候选分别201.07/200.75/199.76/201.19秒；不同输入规模在该小row runner下没有比例增长，不能按token倍率推candidate core。v15三行批量、960 unique candidates、320 module-alpha calls，core+per-article search-loss133.842秒，native full-prefix133.076秒(16×128tokens)，FP capture2.640、Qfullforward3.674、target replay3.643、32H builds2.053秒。unique calls已对重复三行JSON去重；明细及来源hash见HISTORICAL_TIMING_EVIDENCE.json。

v15 `cost_summary.csv` 的native-prefix peak606,409,216bytes是在后续QPREFIX forward reset之后读取，**不能作为native量化峰值**。候选peak约256MiB也只覆盖旧小批量。v14约1.75–2.08GiB峰值属于旧完整capture/候选流程，也不等于新的全模型+Omni训练峰值。

## 本轮极小预算smoke

只用历史L2 q/down H和pristine前32行，各alpha.9，B3/B32，总70 slots，脚本内wall 3.857秒（包含cache读取与检查，不含Python/torch启动），四次同步candidate区间合计3.401秒。B32宽896 core=0.238441s，宽4864=1.198685s，候选分阶段peak最大191.7MiB。无新calibration搜索/PPL/模型级实验。单冷调用不能声称加速比或代表steady-state。

## M2估算

B32、一次一个alpha：每block宽896 batches=28q+4k+4v+28o+152up+152gate=368，宽4864 down=28；全模型95,040次GPU完整candidate调用。

核心外推=24×10×(368×t896+28×t4864)=29114.2s≈**8.09h**，假设其余同宽projection同吞吐，batch32端尾不更慢，使用该GPU/同core。它不包含262,144-token loss、capture、H、序列化，也不是观测到的full-model时间。三行旧runner按projection/行数推约74.1h，同宽k/v/gate用q/up替代，是未经该projection测量的估算，不能直接作为正式执行路线。

直接loss乘法总约2×N×10×357,826,560≈1.876×10^15 FLOPs；若用同目标Gram约2×10×Σ_rows d_in²≈1.47×10^13 FLOPs，另计一次Gram构建。二者浮点排序必须验证，不能将理论FLOPs比写成实测加速。预留capture/H约20–60min（v15按token数、H数外推上界，cholesky不随N线性；不精确）；候选loss/导出/metadata/I/O和冷启动另加。**规划M2 12–24h**，不是可信置信区间；formal前在共同新cal数据的固定32行q/down做全部十alpha含真实loss测量，更新预算并验证不越48h执行上限。越界停止，禁止删grid或挑重要row。可校验Gram/增加row batch/候选vectorization，不改变算法；任何工程吞吐收益独立报告。

M1：已有native133秒是在2048 calibration tokens；新协议N为其128倍，capture/H增加而原生全row GPTQ列循环并不同比增加。规划20–60min，待同协议one-block计时核定。M0只加载和评测，无校准。不能把这些规划当作已运行成本。

## M3成本单独算

20 epochs ×128 samples/batch1=2,560 AdamW steps/block，总61,440steps。LWC总5,591,040个scalar参数，每次只训练当前block232,960个；每block梯度/FP32params/两个moments约3.56MiB，原权重冻结，但forward/backward激活和temporary quant weights还需显存。没有可核验的Omni-Qwen block training timing，不虚构wall。若one-step实测 t=.05/.1/.5/1s，则纯optimizer-loop分别约0.85/1.71/8.53/17.07h，**仅为敏感性情景，不是测量**；另加每层teacher生成、quant传播、初始化、export。下一轮one-block smoke先测forward/backward/GradScaler实际step及warmup，给正式预算；若OOM/非finite/超过预设24h上限，不启正式训练。不得为赶时删epochs或改变loss。

## VRAM和磁盘

本机8188MiB可见GPU总显存。checkpoint unique FP16 tensor载荷约942.3MiB；模型数据header确认，未读取全tensor。128×2048×896 FP16 block-boundary cache=448MiB，Omni FP/quant两份约896MiB；不开aug_loss。最大H/Gram/Hinv单份4864²FP32=90.25MiB；Qsmall896²=3.06MiB。最大projection输入全CPU FP16约2.375GiB，GPU只流式2048-token块，约38MiB FP32；不把所有X移GPU。B32 serialalpha工作权重不到数MiB，但全模型/teacher/student/attention/AMP workspace另计。

设计M1/M2仅active block在GPU、embedding/norm按需移动，projection capture/H/loss流式；**规划2–4GiB**，不是完整流程实测峰值。M3 block训练+两个boundary cache+SDPA规划3–6GiB；8GiB能否满足必须实测，禁用会分配大[heads,L,L]tensor的旧自定义attention。VRAM阶段内reset，立即读取peak，记录allocated和reserved；不得在下一阶段reset后取前阶段peak。

标准GPTQ payload约438.0MiB（int4+FP16scale+4bitzero+g_idx+未量化FP16权重，不计headers），是布局预算而非已导出checkpoint。primary dense每方法约942.3MiB。十候选全部FP16 tensor约6.67GiB；本方案只永久保存selected tensor和十loss/params，避免全候选盘积。所有168模块X若全存约120GiB，禁止这样cache；每组消费后释放，boundary/prefix只保留当前和恢复所需。

保留96组Hinv和undampedGram约4.7GiB，可按恢复策略逐组删除/仅保留当前；当前down X约2.4GiB，加selected weights、两个dense prefix recovery checkpoints、三方法导出/metadata，规划**18GiB新磁盘空间上限**，cal/evaltoken JSON额外几MiB。所有预计算、H/cache、转移、load/save、optimizer和resume重试计入end-to-end成本，不把离线效率称为在线推理速度。
