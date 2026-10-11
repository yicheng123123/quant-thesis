# v49：M1 初始化的 v48 快速整数修复完整模型验证

日期：2026-10-08。结果目录独立于 v46/v48；历史 checkpoint、实现与 PPL 原始结果均只读。当前接口不能读取或切换模型代号，故无法确认实际使用了用户指定的 Luna Max。计算记录中的 GPU 为 NVIDIA GeForce RTX 4060 Laptop GPU，PyTorch 2.6.0+cu124。

## 结论

冻结的 v48 `fast_accept.batch_repair` 路径完成了 M1 全模型修复、打包/reload 校验和两项正式 PPL。新模型与 v46 R1 在 143/168 个量化 Linear 权重张量上不同，推理配置相同，因此没有复用 R1 分数；WT2 与固定预算 C4 各实际评测一次。两项 PPL 都略低于 R1，也低于正式 M1。相对 R1 的变化很小；这是单次运行，不能据此声称跨 seed 稳定或统计显著，也不证明数值规则具有理论安全保证。

局部重构误差下降只描述固定前缀输入下的搜索目标，不能代替完整模型 PPL。v48 快速接受路径在本次运行通过了规定的终态数值检查；这不等于全程数学单调性的证明，也不是 ReQuant 官方复现或原创性结论。

## 冻结来源与实际执行

- 协议 SHA256：`6c44516cb14921873903c3c3185442460bdef423c0ee272479bc44a5192b3512`。
- v48 `fast_accept.py` SHA256：`c3a760c0f48723b3784774263af71bc9e469115c2ab5d0c35ba3145671e079a2`；规则 SHA256：`7c6dd06068f46b0b9efdead69055c711356588f4700d751cfc744018505631d6`。运行时绑定记录确认 `runtime.batch_repair -> v49.bound_fast -> v48_fast_accept_v49.batch_repair`，没有回落到 v45 旧分派。
- M1 初始化 checkpoint SHA256：`3678f867d064abc9730d392f279516223ef17de16351c577e0f1977e9b7bba33`；模型 revision `7ae557604adf67be50417f59c2c2f167def9a775`；校准清单 SHA256：`664419273f63df3c0b0746b18bb66b593a5384bbe6ba8afcd6f4905c9479aae3`。
- 运行时直接委派给冻结 v48 算法。三个记录在案的 v49 launcher 修订只处理导入别名、packed-code 检查点序列化/恢复、以及 block0 行数审计门槛；未改搜索、候选顺序/预算、接受规则或 v48 文件。修订证据见 `RUNNER_AMENDMENT_LOG.json`。
- GPU 总耗时计时从 `GPU_START.json` 开始；单独的 GPU 前导导入失败发生在 GPU 初始化前，记录为 0 GPU 秒、约 26 秒进程 wall。

## 覆盖与正确性

- 完整覆盖 24/24 blocks、168/168 Linear、304,128/304,128 输出行，并保存 96 个依赖组 checkpoint。实际传播使用本次修复模型自身的 qkv → o → up/gate → down → next block 前缀；输入哈希与组级提交状态保存在 `PROPAGATION_RESTORE_CHECKS.csv` 和 `formal/`。
- 所有 code 均通过合法范围与打包/reload 检查；M1 原始 scale、zero、g_idx 和固定网格保持不变。完成最终保存、全量 reload 与样例输出回归。
- 9,523,428 次接受更新，9,312,444 个最终被修改坐标，22,458,309,805 次候选评价；0 次候选拒绝、0 次 raw-X FP64 候选复核。每个矩阵均有独立完整终态 FP64 重构核验。
- 全模块最大 `p` 漂移 `2.62e-14`；最大终态误差 `7.27e-15`，最大终态误差比 `7.29e-7`。无非法码、非有限值或状态正确性失败。
- `MODULE_RESULTS.csv` 中逐模块固定输入 raw64 局部误差的报告总和为 882.900 → 771.271（−12.64%）。这是不同模块局部目标值的描述性求和，并非一个 pooled 模型目标，也不预测 PPL。

## R1 张量比较与正式 PPL

v49 最终模型 SHA256 为 `8a4e2aed53ed0f72c20780ca80ad7a2f751c8babb1c509d8f532054f150ecaf8`；配置 SHA256 为 `dbcd7d5a5cc90a32e149bf91a3a83dec8eacbd0474033c4030ac1cc8a14ef1d9`。290 个具名模型张量均参与比较，配置与 R1 相同。相对 v46 R1，有 143/168 个量化 Linear 的 code/FP16 网格权重不同，25 个相同；全部 168 个 Linear 的 scale、zero、g_idx 相同。权重差异最大绝对值为 0.2421875，因此按冻结流程分别评测新模型一次。

正式 M1 与 R1 结果来自 v46 `PPL_SOURCE_AUDIT.json` 指定并验过哈希的历史 JSON/窗口文件；v49 未重跑基线。新模型结果和原始窗口保存在本目录。两个数据集使用相同的 2048 最大长度、512 stride、无额外 special token 的 evaluator；PPL/NLL 按 valid tokens 汇总。PPL 相对变化均以表中比较对象为分母。

| 数据集 | 模型 | PPL | valid-token 加权 NLL | valid tokens / windows |
|---|---|---:|---:|---:|
| WT2 test | M1（历史） | 13.884370 | 2.630764 | 299,077 / 582 |
| WT2 test | R1（历史） | 13.662793 | 2.614676 | 299,077 / 582 |
| WT2 test | v49 | 13.628060 | 2.612131 | 299,077 / 582 |
| C4 fixed validation | M1（历史） | 22.233924 | 3.101619 | 131,071 / 253 |
| C4 fixed validation | R1（历史） | 22.092247 | 3.095227 | 131,071 / 253 |
| C4 fixed validation | v49 | 22.077789 | 3.094572 | 131,071 / 253 |

- v49 对 M1：WT2 PPL −0.256310（−1.8460%），NLL −0.018633；C4 PPL −0.156135（−0.7022%），NLL −0.007047。
- v49 对 R1：WT2 PPL −0.034733（−0.2542%），NLL −0.002545；C4 PPL −0.014458（−0.0654%），NLL −0.000655。
- 两个语料方向相同，但 v49 相对 R1 的增益较小。没有从局部 MSE 推导 PPL，也没有按 PPL 调参或生成第二个模型。

## 实测成本

以下是 v49 本轮同步 GPU wall 及其阶段记录，不含 v48 先前有界验证成本。24 个完整 block 的 `BLOCK_TIMINGS.csv` 合计是正式修复 wall；进程级 elapsed 包含失败尝试、恢复、保存、评测与收尾。

| 阶段/口径 | 实测时间 |
|---|---:|
| 24-block 完整矩阵修复（block wall 合计） | 4,543.08 秒（75.72 分钟） |
| checkpoint 锁定/模型就绪（GPU_START 起） | 4,981.02 秒（83.02 分钟） |
| 保存 + 最终打包/reload 检查 + reload | 13.65 秒；其中最终 save 0.93、pack/reload 全网格检查 11.56、reload 1.17 秒 |
| WT2 与 C4 evaluation | 138.65 秒；WT2 95.78、C4 42.87 秒 |
| 完整 GPU_START 至评测和最终状态写完 | 5,126.39 秒（85.44 分钟），低于 10,800 秒上限 |

正式阶段计时器汇总（同步单项阶段计时，不把其和替代 block wall）：C 构建 1,755.01 秒；独立完整终态 FP64 复核 2,124.07 秒；候选评分 102.50 秒；proposal/settle 40.59 秒；`p` 检查/刷新 88.06 秒；直接重算 96.39 秒；依赖输入捕获 105.36 秒；输入哈希 103.42 秒；初始化 17.06 秒；最终 cache/grid/pack 检查 11.41 秒；检查点 packing/IO 13.90 秒；依赖提交 1.24 秒；down 提交及前缀刷新 30.97 秒。终态 FP64 检查与独立候选复核是不同阶段。

峰值显存：allocated 4,745,180,160 bytes，reserved 6,482,296,832 bytes。`FAILED_ATTEMPTS.jsonl` 记录了三次 runner 失败（输出目录缺失、检查点 schema 缺少 `packed_u4`、block0 行数审计值错误）；它们及恢复间隔计入 GPU_START 到完成的总 elapsed。已提交 block0 未重复修复，packed-code schema 迁移保持整数码和反量化权重不变。另有一次 GPU 前导导入失败，发生于 GPU_START 之前，0 GPU 秒。失败记录里的 elapsed 是共同计时起点的时间戳，不能相加为独立 attempt 时长。

与 v46 的相同净正式口径相比：v46 24-block 正式修复 5,485.68 秒，v49 为 4,543.08 秒，单次测得少 942.60 秒（17.19%）。这是一次运行的工程测量，不是重复稳定加速结论。v46 GPU_START 至其收尾为 6,842.92 秒，含约 785.95 秒错误分派失败；v49 全轮也是含其失败/恢复的 GPU_START gross wall 5,126.39 秒。两轮失败组成不同，因此 gross wall 不单独解释为算法加速。

## 结果文件与复现

- 冻结协议：`PROTOCOL.json`；最终执行绑定与实现哈希：`FINAL_RUNNER_LOCK.json`、`RUNTIME_BINDING.json`。
- 逐模块和 block 计时：`MODULE_RESULTS.csv`、`BLOCK_TIMINGS.csv`、`MATRIX_TIMINGS.csv`、`STAGE_TIMINGS.csv`。
- 组级传播与检查点：`PROPAGATION_RESTORE_CHECKS.csv`、`CHECKPOINT_LOCK.json`、`formal/groups/`。
- 部署比较：`DEPLOYMENT_COMPARISON.json`、`reports/MODULE_VS_R1.csv`、`reports/TENSOR_DIFFERENCES_VS_R1.csv`。
- 正式评测与原始窗口：`R1_wikitext.json`、`R1_wikitext_WINDOWS.csv`、`R1_c4_fixed_budget.json`、`R1_c4_fixed_budget_WINDOWS.csv`。名称 `R1_*` 是冻结 evaluator 的历史兼容命名；文件中的 `model_sha256` 指向本次 v49 最终模型。

在本目录、同一锁定依赖和本地输入/权重缓存下复现：

```powershell
python .\prepare_v49.py
python .\run_v49.py --resume
```

完成状态、模型哈希、总 elapsed 和失败恢复链见 `RUN_STATUS.json`、`COMPLETE.json`、`RECOVERY_HISTORY.json` 与 `FAILED_ATTEMPTS.jsonl`。P3 未读取。
