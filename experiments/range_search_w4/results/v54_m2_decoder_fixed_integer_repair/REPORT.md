# v54 — M2 初始化的 decoder 接口修复后整数码修复

**结论：D，完整模型和 PPL 未完成。** 在 10,800 秒共同预算下，预算门于完整提交 block1 后停止：只继承已验证的 block0 四组，并新修复 layer1 七个 Linear（12,672 行）。其余 22 个 block、完整模型保存/reload 和两项正式 PPL 均未运行。保留的依赖组 checkpoint 可以用于以后经授权的恢复，但不构成可部署整模型，也不能据此回答修复质量是否迁移。

用户要求的 Luna Max 通过界面请求；当前接口不能确认实际模型设置。GPU 记录为本地 NVIDIA RTX 4060、PyTorch 2.6.0+cu124。

## 冻结来源和恢复审计

- M2 初始化：正式 v18 checkpoint，SHA256 `810d6329cdc51d995543dbb701e2368568b19459a44f340df1a8a0019d447eed`；模型 revision `7ae557604adf67be50417f59c2c2f167def9a775`。
- 正式 128×2048 calibration manifest SHA256 `664419273f63df3c0b0746b18bb66b593a5384bbe6ba8afcd6f4905c9479aae3`。候选算法/预算、固定量化网格、真实 FP16 网格增量、v52 初始化 raw-X FP64 失败分支和 v49/v48 接受规则均按冻结副本运行；v50 混合 Gram 未使用。
- v53 runtime SHA256 `360e323e8f868347d7bbfaada2c6cee6daa45c52326d35516b1db537461d16ad`；decoder 提取器 SHA256 `80b796ba6ed85910a36314b7a2d9a8857b50ad56d74e5a6cb5242c84c82c4f2c`；fast acceptance core SHA256 `f155a1f26dd65b67d16e531f438f85b720b6df05910e1dda84082f6b1201ef7a`；初始化复核 SHA256 `4f6972f70c3babe055d3e686a70fb7c765f809f3a3391c57625caf0a322bc4e5`。运行分派审计为 `v49.bound_fast → v48_fast_accept_v49.batch_repair`，没有走旧 V42 分派。
- 仅从原 M2 重建模型并装入 v53 允许的 v51 block0 四组；源 checkpoint 哈希、码、scale、zero、g_idx、packing 和原协议均通过。v52 layer1 及以后参数、激活和 C 均未复制。重放得到正式 128 篇 block0→layer1/qkv 全 token 输入哈希 `70cd41104db3769a3d434c27d4612a41becf4be98439c6328c6010e5b6eae7ac`，与 v53 独立核验值相同；隐藏状态哈希为 `9488e56e397608a5d19291595f43645135ad6b6ed477cd6d97eeba7398bfeed6`。
- 三篇固定文章（0、64、127）逐 block 与原生完整 forward + 独立 hook 对照。block0 和 block1 共 12/12 个 hidden/下一层 qkv 输入比较通过，全部 `[1,2048,896]` FP16、逐位一致、最大绝对误差 0。参考路径没有调用 decoder 提取器。只有 block0、block1 完成了此传播覆盖；不能外推到未运行的 block2–23。

## 实际修复和数值检查

| 覆盖项 | 结果 |
|---|---:|
| 已有有效 block0 | 4 组、7 Linear、12,672 行；继承后重放，不重修 |
| 本轮新修复 | layer1 四组、7 Linear、12,672 行 |
| 新修复接受更新 | 402,449 |
| 最终变化码位 | 387,291 |
| 候选增量评价 | 938,615,990 |
| 候选拒绝 / 候选 FP64 复核 / 运行中初始化 FP64 fallback | 0 / 0 / 0 |
| 固定预算内完整覆盖 | 14/168 Linear；25,344/304,128 行（其中本轮新修 12,672 行） |

每个 layer1 矩阵均通过全输出行 raw-X FP64 终态重构核验、合法码 [0,15]、更新/候选预算、固定 scale/zero/g_idx 与 packed-code 检查；最大终态绝对误差 `4.94e-17`，最大误差/冻结容差比 `6.93e-8`，最大 p 漂移 `1.08e-15`。没有发现非有限值、码越界、网格变化、状态污染或目标上升。CPU 初始化复核测试集通过；本次运行没有触发该 fallback。

layer1 固定输入局部 search MSE 汇总 `14.0505426945 → 12.4722725983`（变化 `−1.5782700962`）。这只是七个矩阵各自固定输入的局部目标汇总，不是整模型质量证据，也不预测 PPL。

## 预算和成本

预算停止依据不是测试结果：block1 提交后，冻结预测要求剩余 4,851.937 秒，内部共同账本余 4,834.407 秒，预测短缺 17.530 秒；预测已包含后续传播估算、120 秒不确定性余量和 360 秒保存/reload/PPL收尾预留。随后将已记录的 14.442 秒 GPU 前模型加载为零的 CPU 启动失败也记入共同成本，保守累计为约 5,980.040/10,800 秒，余约 4,819.960 秒。CPU 失败时间单列，不称 GPU 运行时间。工具等待不计入。

| 阶段/口径 | 秒 | 说明 |
|---|---:|---|
| 共享账本进入 v54 前 | 5,677.428 | 用户给定值；底层账本精确值见 `FINAL_BUDGET_ACCOUNTING.json` |
| 首次 runner 错误 | 14.442 | CPU 进程 wall；锁文件名不匹配，未加载模型、无 GPU 调用；保守计入共同总成本 |
| 第二次启动故障 | 0.028 | 已在共享活动账本；缺少 `budget.elapsed`，模型加载前退出 |
| 本次正式进程预算段 | 288.141 | 包含 warmup、M2 load/零修改检查、恢复、layer1 修复、终态和传播核验、写入停止状态 |
| block0 恢复与检查 block wall | 4.768 | `new_groups=0`，不含重复修复 |
| layer1 完整 block wall | 238.934 | 七个新矩阵；含各矩阵终态检查、打包提交、前缀刷新及三篇原生传播核验 |
| 其中三篇原生完整 forward | 2.293 | layer1→layer2 边界，包含在 block wall 内 |
| layer1 C 构建阶段 timer 合计 | 86.724 | 矩阵级 timer 汇总，嵌套，不可与 block wall 相加 |
| layer1 全矩阵独立 raw-X FP64终态 timer 合计 | 103.306 | 包含在 block wall 内，不是单独外加 wall |
| layer1 候选评分 timer 合计 | 5.427 | 包含在 block wall 内；时间不能由评价次数替代 |
| 完整保存/reload、完整模型锁、PPL | 未运行 | 因预算门停止；无整模型 checkpoint hash |

分项 timer 会嵌套；完整 stage 明细见 `STAGE_TIMINGS.csv`，block 实测见 `BLOCK_TIMINGS.csv`。历史 M2 构建成本 `2,115.935` 秒另列，不属于 v54 新增成本。

## 正式 PPL 与历史结果

| 模型 | WT2 test PPL / valid-token NLL | C4 fixed validation PPL / valid-token NLL | 来源与传播有效性 |
|---|---:|---:|---|
| 新 v54 | 未运行 / 未运行 | 未运行 / 未运行 | 无完整模型，不能生成质量结论 |
| M1 | 13.884370 / 2.630764 | 22.233924 / 3.101619 | v18 实际 checkpoint 测量；source/window hashes 已审计；记录的 v18 Transformers 4.49 runtime 支持 tuple 接口 |
| M2 | 13.760556 / 2.621806 | 22.032883 / 3.092536 | v18 实际 checkpoint 测量；source/window hashes 已审计；记录的 v18 Transformers 4.49 runtime 支持 tuple 接口 |
| v45 R2 | 13.553100 / 2.606615 | 21.941981 / 3.088402 | 历史 checkpoint 的两语料实际测量；校准传播有效性 UNKNOWN |
| v46 R1 | 13.662793 / 2.614676 | 22.092247 / 3.095227 | 历史 checkpoint 的两语料实际测量；校准传播有效性 UNKNOWN |
| v49 fast R1 | 13.628060 / 2.612131 | 22.077789 / 3.094572 | 历史 checkpoint 的两语料实际测量；校准传播有效性 UNKNOWN |

完整 PPL、窗口、result 和模型 hash 映射见 `HISTORICAL_QUALITY_BACKGROUND.csv`。M1/M2 的历史结果可按已审计 source 保留；v45/v46/v49 分数确实是各自已保存 checkpoint 的实际测量，但 v53 将其历史 decoder 返回类型与校准传播判为 UNKNOWN。当前 v53/v54 重放不能证明旧运行当时使用了 Tensor 返回，也不能证明历史传播有效/无效。因此这些旧分数不能替代 v54 PPL，不能把历史模型差值归因于接口修复。

## 结论与有效恢复边界

保留 v54 D 状态和全部成功/失败日志。当前有效可复用边界是：原 M2 + 经 v53 验证的 v51 block0 四组 + v54 layer1 四组，并以 v54 保存的完整 token hidden/input hashes 和 group commit 为准。layer1 以后仍是原 M2；没有复制或使用 v52 下游状态。部分 group 文件不是 final checkpoint；没有部署 reload 检查和新 PPL。不得报告 A/B/C质量分类，也不得宣称 v54 与历史 R2 等价。

本轮只是既有整数码修复思路的应用执行；没有新增搜索算法、跨 seed 稳定性或原创贡献主张。

## 主要产物

- 冻结规则及来源：`PROTOCOL.json`、`FROZEN_RUN_LOCK.json`、`SOURCE_AUDIT.json`、`RUNTIME_BINDING.json`。
- 输入/恢复/传播：`BLOCK0_GATE.json`、`PROPAGATION_RESTORE_CHECKS.csv`、`PROPAGATION_NATIVE_RESULTS.csv`、`PROPAGATION_NATIVE_TIMINGS.csv`、`formal/NATIVE_PROPAGATION_block_00.json`、`formal/NATIVE_PROPAGATION_block_01.json`。
- 逐模块/行与数值：`MODULE_RESULTS.csv`、`formal/rows/`、`MATRIX_TIMINGS.csv`、`NUMERICAL_VALIDATION.json`、`CPU_TESTS.json`。
- 预算/状态：`ACTIVE_JOB_LEDGER.json`、`FINAL_BUDGET_ACCOUNTING.json`、`ONLINE_BUDGET_STATUS.json`、`RUN_STATUS.json`、`FAILED_ATTEMPTS.jsonl`、`PARTIAL_CHECKPOINT_LOCK.json`。
- 正式 PPL 未产生；历史比较见 `HISTORICAL_QUALITY_BACKGROUND.csv`。本目录不包含完整模型 checkpoint lock。
