# v40 — 逐 Linear 赢家并集独立运行

## 结论

**正确性：PASS。** 完整的 B 缩减候选运行与原 M2 在全部 168 个 Linear 的候选分数、逐行选择、量化权重、scale、zero、g_idx、输入/Hessian 和依赖边界上匹配。A 与 B 的最终 290 个具名模型张量及推理配置都与历史 v18 M2 相同。

**成本：配对实测校准 wall 从 2255.67s 降到 1134.99s，节省 1120.68s（49.68%），加速比 1.987×。** 这是本轮各一次完整独立运行的配对实测，不是由候选数推算。按监督进程计时，A=2268.83s、B=1148.39s，加速比 1.976×。

## 冻结比较

| 项目 | A：原 scalar10 | B：历史赢家并集 |
|---|---:|---:|
| 全部 Linear | 168 | 168 |
| 主候选完整 Linear GPTQ 次数 | 1,680 | 675 |
| 相对 A 的候选执行减少 | — | 59.82% |
| row-alpha score slots | 3,041,280 | 1,287,040 |
| 相对 A 的 score slots 减少 | — | 57.68% |
| 全运行 alpha=1 parity 辅助调用 | 35 | 35 |
| 方法计时 wall（含模型加载、capture/replay、保存/重载检查） | 2255.670s | 1134.987s |
| 观测峰值显存 allocated / reserved | 3.691 / 4.488 GiB | 2.159 / 2.707 GiB |

B 的 675 个候选来自运行前冻结的 candidate_budget_precheck 逐 Linear 赢家并集，强制包含 alpha=1，并遵守原 scalar10 顺序；没有在本轮根据结果补充候选。预检集合本身依赖此前已经完成的 10 点搜索，因此 675 次只表示在候选集合已知时的重执行量，不消除首次发现赢家并集的成本。

## 正确性证据

- 完整 block0 pilot 完成 7 个模块，27 个候选 GPTQ、49,792 个 row-alpha score slots，另有 7 个 alpha=1 原生 parity 调用；逐行 alpha/分数、量化状态及 5 个边界通过。
- A 与 B 分别从新加载的 pristine 模型启动；两者不共享候选张量、Hessian、activation cache 或 checkpoints。共同使用的只有冻结的原始 calibration manifest。
- A 复现 v18 scalar10 全部 3,041,280 个直接候选分数；B 复核本轮全部 1,287,040 个赢家并集候选分数。两臂 168/168 模块的 alpha、直接损失、W、scale、zero、g_idx、输入及 Hessian 对应值均精确匹配 v18。
- 两臂各检查 24 个 layer state、119 个依赖边界；`BOUNDARY_PARITY.csv` 中 286 条比较全部相等。
- v18、A、B 均为 290 个具名张量；规范化聚合 SHA256=`dd4803a72696d25593547d4d34429ccc24d37dd28bcd1420863609126611340d`。A/B 最终模型文件 SHA256=`810d6329cdc51d995543dbb701e2368568b19459a44f340df1a8a0019d447eed`。config、generation config、模型 revision、校准 manifest 与量化推理语义相同。元数据文件自身包含不同的 v40 来源记录，其元数据文件 hash 不同，但规范化张量和推理配置一致。
- A/B 保存后重载的 64-token logits 检查最大及平均绝对差均为 0（各自相对运行内参考）。未运行新的 held、P3 或 PPL。因为最终模型张量和推理配置精确等同，历史 M2 的质量评测可对应此模型；本轮没有产生新的质量评测或质量提升结论。

## 成本拆分

| 阶段（阶段时间为 instrumentation 累计，非端到端加和项） | A | B |
|---|---:|---:|
| 完整 Linear 候选 GPTQ | 1056.988s | 392.470s |
| 原直接全行评分 | 726.383s | 281.477s |
| 唯一组 Hessian 构建 | 84.258s | 84.128s |
| 共享原生 H-inverse 因子准备 | 5.610s | 5.534s |
| qkv → o 输入刷新 | 22.617s | 22.266s |
| o → up/gate 输入刷新 | 21.451s | 21.396s |
| up/gate → down 输入刷新 | 58.595s | 57.084s |
| down → 下一层输入刷新 | 29.552s | 29.151s |
| 额外 alpha=1 parity | 21.745s | 21.644s |
| 层 checkpoint 写入 / 最终模型保存 | 1.113s / 0.790s | 1.066s / 0.797s |

阶段累计值来自内层同步计时，`STAGE_TIMING.csv` 明确标注为 non-additive，不应相加替代 run wall。B 的主要节省来自候选 GPTQ 与直接评分：Hessian/因子准备、前缀输入刷新和提交开销基本不变，因此端到端 wall 降幅小于候选调用降幅。

本轮没有单独追加 kernel warm-up 调用；顺序在冻结前确定为 pilot → A → B。每个完整运行 wall 包含新模型加载、输入捕获/传播和最终保存/重载检查；全局预算还保守计入 phase supervisor wall 与 CPU parity/hash 审计。日志记录 `expandable_segments not supported` 的 PyTorch allocator 警告和 SDPA sliding-window 提示；没有异常退出或 OOM，最终张量与边界校验通过。

## 预算与结论边界

- pilot/A/B phase supervisor wall 总计 **3481.281s**；CPU correctness/hash 审计 **44.274s**；预算账本累计 **3525.555s / 5,400s**。失败尝试为 0。
- B 在冻结的历史模型 revision、calibration、GPTQ 配置和数据协议上执行正确且复现 M2。
- 节省适用于“逐 Linear 赢家并集已由历史 10 点搜索获得、配置与输入协议保持相同、需要再次执行校准”的复用情形。它不证明该集合迁移到新数据、模型或 revision，也没有抵扣首次搜索成本。
- 不读新 held/P3，不运行新 PPL；不把离线校准节省表述为在线推理加速。

## 文件索引

- 冻结协议：`FROZEN_PROTOCOL.json`
- 候选清单及来源：`CANDIDATE_MANIFEST.json`
- 任务核算：`JOB_TABLE.csv`
- pilot：`PILOT_PARITY.json`
- 逐模块核验：`MODULE_PARITY.csv`、`A_RUN_VALIDITY.json`、`B_RUN_VALIDITY.json`
- 激活边界：`BOUNDARY_PARITY.csv`
- 最终张量与配置 hash：`FINAL_TENSOR_HASHES.json` / `.csv`
- 阶段计时：`STAGE_TIMING.csv`；端到端计时：`RUN_WALLS.csv`
- 预算：`BUDGET.json`
- 汇总：`SUMMARY.json`
- 复现步骤：`REPRODUCE.md`
