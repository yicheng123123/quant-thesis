# v56 报告：正确传播下的 M1 快速整数码修复与四臂比较

## 结论

v56 从正式 M1 的 layer0 独立完成修复、传播核验、部署检查和两项实际 PPL。M1+修复相对 M1 在 WT2 与固定预算 C4 都有较低 NLL/PPL。观察到的 v55 M2+修复在两项指标上又低于 v56 M1+修复，scalar10 的数值优势没有缩小，按四臂点估计反而略扩大。

但 v55 新评测使用 PyTorch 2.5.1+cu121，v56 与 v18 M1/M2 基线使用 PyTorch 2.6.0+cu124。窗口、token、manifest 和 evaluator 逻辑相同，R2 相关差值仍混有运行时版本差异；按用户冻结要求没有重测 M1、M2 或 v55，因此不能把小幅差异严格归因于初始化或修复。结果是单次固定模型测量，不代表跨 seed 稳定或统计显著。

## 冻结范围与来源

- 模型：Qwen2.5-0.5B-Instruct revision `7ae557604adf67be50417f59c2f167def9a775`；M1 checkpoint SHA256 `3678f867d064abc9730d392f279516223ef17de16351c577e0f1977e9b7bba33`；从 layer0 新建运行，未加载 v55/v46 修复组、输入或 C，也未用 v46 修复参数。M2 checkpoint SHA256 `810d6329cdc51d995543dbb701e2368568b19459a44f340df1a8a0019d447eed`，只作既有对照。
- 校准：128×2048 manifest SHA256 `664419273f63df3c0b0746b18bb66b593a5384bbe6ba8afcd6f4905c9479aae3`；固定 M1 原始 scale/zero/g_idx，W4A16 symmetric G128；v49 快速修复核心、v52 初始化 FP64 复核和 v53 decoder 接口按冻结文件复用。`SOURCE_REUSE_AND_DIFF.json` 记录算法、候选顺序、预算、网格与数值接受规则没有变化；仅做 M1 初始化、路径、边界与标签适配。
- v56 协议 SHA256 `2745b349bf93a24624ab63dc4bbe47f678e8829f745d4be6b47b4826243a9aa9`；最终配置 SHA256 `7101cfb3475d42d7e81a9e5a75aabf5f9cfd39308ffdc367d35a6b27242fd5bb`。冻结源码哈希见 `FROZEN_RUN_LOCK.json`；快速接受实现 SHA256 `f155a1f26dd65b67d16e531f438f85b720b6df05910e1dda84082f6b1201ef7a`，decoder 提取 SHA256 `80b796ba6ed85910a36314b7a2d9a8857b50ad56d74e5a6cb5242c84c82c4f2c`，运行时 SHA256 `bbe1e4c22200f1ccb3d85f35a274c44cf5d46cd606160b59b0cd9afc346bae11`。
- M1 与 M2 有效推理配置相同；M2 的逐行量化网格与 M1 不同。四臂比较是不同初始化流程加同一修复算法，不是只改变整数码的严格参数对照。
- v18 的 M1/M2 PPL 来源、checkpoint/window/result 哈希在 v55 `HISTORICAL_QUALITY_BACKGROUND.csv` 中核验。v53 历史审计指出 v18 记录 Transformers 4.49.0，支持 tuple decoder 接口，未发现同类 Tensor 误用证据；但没有逐调用类型日志，也未穷尽重验全部历史前缀。因此沿用实际 checkpoint 测量，同时不把它当作逐层输入完全等价证明。v55 是 v53 正确传播后新测的模型。
- v45/v46/v49 历史模型仅作背景，其传播有效性为 UNKNOWN，不进入四臂表。v55 共同历史账本 11037.312983 秒单独列示，不计入 v56 cap。

## 完整修复和正确性

- 覆盖 24/24 blocks、168/168 Linear、304128/304128 输出行，96 个依赖组提交；接受更新 9,520,836，修改坐标 9,309,752，候选评价 22,453,814,293；拒绝 0，FP64 复核 0。
- 全部模块通过终态 raw-X FP64 直接重构。记录的 search MSE sum 从 882.437340482 降到 771.044208297；这是模块行均方误差汇总，不是 PPL。最大 p 漂移 4.383e-14；最大终态误差比 1.110e-06。
- 文章 0、64、127 的独立原生 full-forward/hook 参考覆盖 24 层。CSV 有 141 条 next-hidden/qkv 输入比较，141/141 通过且逐位一致；保留完整 `[1,2048,896]` token 轴，参考路径不调用待验证的 decoder 提取函数。
- 码、固定网格、packing/reload、输出回归通过。最终模型 SHA256 `9d768808bf1fd3581406df3538aebc00ed177eafc6e46372da679e9db85ccac0`；290 个具名张量参与对照，与 M1/M2/v55 分别有 168 个张量不同，故 v56 不等价于 v55，按协议完成了新评测。有效配置与四臂相同。

## 四臂质量结果

NLL 按 valid_tokens 加权；WT2 为 299077 tokens/582 windows，C4 固定预算为 131071 tokens/253 windows。数值 CSV 保留全精度及来源哈希。

| 数据集 | 模型 | 加权 NLL | PPL | valid_tokens |
|---|---|---:|---:|---:|
| WikiText-2 test | M1 原生 | 2.630764 | 13.884370 | 299077 |
| WikiText-2 test | M2 scalar10 | 2.621806 | 13.760556 | 299077 |
| WikiText-2 test | v56 M1+修复 | 2.612334 | 13.630824 | 299077 |
| WikiText-2 test | v55 M2+修复 | 2.602639 | 13.499319 | 299077 |
| C4 fixed-budget validation | M1 原生 | 3.101619 | 22.233924 | 131071 |
| C4 fixed-budget validation | M2 scalar10 | 3.092536 | 22.032883 | 131071 |
| C4 fixed-budget validation | v56 M1+修复 | 3.097449 | 22.141401 | 131071 |
| C4 fixed-budget validation | v55 M2+修复 | 3.086242 | 21.894642 | 131071 |

相对变化以左边模型减右边模型计算；负值表示左边 NLL/PPL 更低。

| 数据集 | 对比 | ΔNLL (相对%) | ΔPPL (相对%) |
|---|---|---:|---:|
| WikiText-2 test | v56 M1+修复 − M1 | -0.018430 (-0.7006%) | -0.253546 (-1.8261%) |
| WikiText-2 test | v55 M2+修复 − M2 | -0.019167 (-0.7311%) | -0.261237 (-1.8985%) |
| WikiText-2 test | v55 M2+修复 − v56 M1+修复 | -0.009694 (-0.3711%) | -0.131505 (-0.9648%) |
| WikiText-2 test | M2 − M1（修复前） | -0.008957 (-0.3405%) | -0.123814 (-0.8917%) |
| C4 fixed-budget validation | v56 M1+修复 − M1 | -0.004170 (-0.1344%) | -0.092524 (-0.4161%) |
| C4 fixed-budget validation | v55 M2+修复 − M2 | -0.006294 (-0.2035%) | -0.138241 (-0.6274%) |
| C4 fixed-budget validation | v55 M2+修复 − v56 M1+修复 | -0.011207 (-0.3618%) | -0.246759 (-1.1145%) |
| C4 fixed-budget validation | M2 − M1（修复前） | -0.009083 (-0.2929%) | -0.201041 (-0.9042%) |

### Scalar10 优势与描述性交互

- WikiText-2 test：M2 相对 M1 的 NLL 优势（M1−M2）从修复前 0.008957 变为修复后 0.009694，优势扩大 0.000737；PPL 差（M1−M2）从 0.123814 变为 0.131505。
- C4 fixed-budget validation：M2 相对 M1 的 NLL 优势（M1−M2）从修复前 0.009083 变为修复后 0.011207，优势扩大 0.002124；PPL 差（M1−M2）从 0.201041 变为 0.246759。
- WikiText-2 test 描述性交互 `I=(NLL_R2−NLL_M2)−(NLL_R1−NLL_M1)` = -0.000737；负值表示本次 M2 修复的 NLL 下降幅度较大。它不是显著性或协同机制检验。
- C4 fixed-budget validation 描述性交互 `I=(NLL_R2−NLL_M2)−(NLL_R1−NLL_M1)` = -0.002124；负值表示本次 M2 修复的 NLL 下降幅度较大。它不是显著性或协同机制检验。

## 成本账本

| 范围 | 秒 | 口径 |
|---|---:|---|
| 24 个完整 block timer 合计 | 5317.371 | BLOCK_TIMINGS.csv 的逐 block wall 合计 |
| 24 次原生传播核验 | 47.062 | 文章 0/64/127 full-forward/hook 对照 |
| 有效初始化构建 | 2.845 | v56 加载既有 M1；未重新 GPTQ |
| C 构建 | 1981.684 | 96 个依赖组，FP64 Gram |
| 全矩阵独立终态 FP64 重构 | 2325.674 | 168 个 Linear |
| 最终模型保存 + grid/reload 检查 | 16.482 | 部署阶段同步 timer 合计 |
| 成功的 WT2+C4 evaluator stage | 2278.982 | 两个完整语料的同步 evaluator timer |
| 峰值 GPU 显存记录 | 4525.3 MiB allocated / 5754.0 MiB reserved | STAGE_TIMINGS.csv 所记录的最大值 |
| 冻结续跑进程 wall | 2283.578 | 包含模型 reload、两项评测与落盘 |
| v56 原始账本总计 | 8327.706 | 含 preflight、CPU诊断、首轮失败/部分WT2和续跑 |
| 保守 cap 核算 | 8590.489 | reconciled upper-bound accounting; 9000 秒内 |
| v56 cap 剩余（保守） | 409.511 | 360 秒收尾预算要求满足 |

`STAGE_COSTS.csv`列出各同步阶段。阶段总计可能与 block/process wall 有嵌套，不应再次相加形成总 wall。首轮已完成修复后的 WT2 评测中断于窗口 200；它没有合格的结果文件，窗口损失未纳入质量表，可能耗时上界 256.790 秒。原 ledger 将首轮 process 记为 5693.778 秒，而修复完成计时另记 5699.771 秒；`FINAL_BUDGET_ACCOUNTING.json` 用较大的 repair elapsed 加部分评测上界作保守替代，不把这部分伪装成精确实测。

原正式校准构建成本单独参考：M1 648.876 秒，M2 scalar10 2115.935 秒，差 1467.059 秒，约 3.261×。这是 v18 历史完整构建 wall，不计入 v56 cap。v55 恢复链新增 layer2–23 block wall 为 4240.102 秒，但 block0/1 源自此前有效工作；不能与 v56 从 layer0 的完整 wall 直接计算加速比。

共同历史 v55 账本 11037.312983 秒保持只读、未并入 9000 秒 v56 cap。原始 ledger total 为 8327.706 秒；为保守处理第一段边界记录差异，报告使用 8590.489 秒的 upper-bound cap accounting，剩余 409.511 秒。

## 运行环境与限制

- v18 M1/M2：PyTorch `2.6.0+cu124` / Transformers `4.49.0`；v56：PyTorch `2.6.0+cu124`；v55 fresh PPL：PyTorch `2.5.1+cu121`。因此 R2 涉及的差值可能包含软件/内核差异。
- Luna Max 是用户指定配置；执行接口不能确认本轮实际模型设置。普通沙盒此前仍失败，本轮使用已获授权的沙盒外执行入口；没有声称普通沙盒恢复。
- 没有读取 P3、没有按 PPL 调参，没有重新评测 M1/M2/v55；所有修复选择仅来自冻结 search 目标。局部重构改善不替代 PPL，快速接受规则是经验数值流程，不是理论安全证书或官方 ReQuant 复现。

## 交付索引与复现

- 冻结/来源：`PROTOCOL.json`、`FROZEN_RUN_LOCK.json`、`SOURCE_AUDIT.json`、`SOURCE_REUSE_AND_DIFF.json`。
- 模型锁与传播：`CHECKPOINT_LOCK.json`、`DEPLOYMENT_COMPARISON.json`、`NAMED_TENSOR_HASHES.csv`、`PROPAGATION_NATIVE_RESULTS.csv`。
- 质量原始窗口：`evaluation/v56_wikitext_WINDOWS.csv`、`evaluation/v56_c4_fixed_budget_WINDOWS.csv`；摘要和锁：`QUALITY_EVALUATION_SOURCE.json`、`EVALUATION_RESUME_COMPLETE.json`。
- 四臂数值：`reports/FOUR_ARM_QUALITY.csv`、`reports/FOUR_ARM_CONTRASTS.csv`、`reports/SCALAR10_ADVANTAGE.csv`。成本：`reports/STAGE_COSTS.csv`、`FINAL_BUDGET_ACCOUNTING.json`、`ACTIVE_JOB_LEDGER.json`。
- 只重算报告与 CSV/JSON 聚合（CPU only）：
  ```powershell
  & 'E:\quant-thesis\.venv\Scripts\python.exe' 'E:\quant-thesis\experiments\range_search_w4\results\v56_m1_fast_integer_repair\finalize_v56_report.py'
  ```
- 一次性完整运行的冻结入口记录于 `REPRODUCE.md`；现有完成目录已有 checkpoint 和一次性评测结果，不应在其上重启正式运行或重复评测。
