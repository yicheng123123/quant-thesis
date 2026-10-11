# v55 报告：M2 初始化整数码修复与正式 PPL

## 结论

完整运行覆盖24个block、168个Linear和304,128个输出行。v55使用修复后的decoder接口传播；每个block的固定文章原生forward/hook对照均通过。最终checkpoint通过码/网格、packing/reload和固定样例输出回归。WT2与固定预算C4均由冻结evaluator对v55 checkpoint新测一次。

质量变化以如下实际测量为准；历史v45/v46/v49校准传播有效性仍为UNKNOWN，不能将v55与它们的差异全部归因于decoder接口修复。

| 数据集 | 模型 | PPL | 按valid_tokens加权NLL | valid_tokens |
|---|---:|---:|---:|---:|
| WikiText-2 test | M1 | 13.884370 | 2.630764 | 299077 |
| WikiText-2 test | M2 | 13.760556 | 2.621806 | 299077 |
| WikiText-2 test | v55 | 13.499319 | 2.602639 | 299077 |
| C4 fixed-budget validation | M1 | 22.233924 | 3.101619 | 131071 |
| C4 fixed-budget validation | M2 | 22.032883 | 3.092536 | 131071 |
| C4 fixed-budget validation | v55 | 21.894642 | 3.086242 | 131071 |

相对变化以v55减对应基线计算。

| 数据集 | 比较 | NLL差值（相对%） | PPL差值（相对%） |
|---|---|---:|---:|
| WikiText-2 test | v55 vs M1 | -0.028124 (-1.0691%) | -0.385051 (-2.7733%) |
| WikiText-2 test | v55 vs M2 | -0.019167 (-0.7311%) | -0.261237 (-1.8985%) |
| C4 fixed-budget validation | v55 vs M1 | -0.015377 (-0.4958%) | -0.339283 (-1.5260%) |
| C4 fixed-budget validation | v55 vs M2 | -0.006294 (-0.2035%) | -0.138241 (-0.6274%) |

## 来源与恢复

- 从正式M2 checkpoint恢复，SHA256为 810d6329cdc51d995543dbb701e2368568b19459a44f340df1a8a0019d447eed。v53有效恢复边界通过核对后，复用block0/1的8个组；这些组未在v55重复修复。layer2至23从本次模型自身前缀修复，未读取v52 layer1之后的无效产物。
- 128篇×2048校准manifest SHA256为 664419273f63df3c0b0746b18bb66b593a5384bbe6ba8afcd6f4905c9479aae3。全token轴输入严格检查shape和hash；传播对照覆盖24个block与文章0/64/127。
- 数值流程、decoder实现、初始化复核和fast repair核心依照v54/v53/v52/v49原件复用。v55唯一修订在收尾脚本：原始config JSON比较受到Transformers序列化版本差异阻断，故保留原FAILED_STOP证据，改为同一当前loader解析后的有效配置核对，再独立完成张量审计和新评测。未改量化代码、接受规则或checkpoint。
- 普通沙盒未确认恢复；本轮使用此前获授权的沙盒外执行入口。请求的Luna Max模型设置无法由执行接口确认。

## 修复、传播与部署检查

- 修复：168 Linear；304128输出行；接受更新 9582903；修改坐标 9417181；候选评价 22390919274；拒绝0；FP64复核0。
- 终态检查：每个模块全行raw-X FP64直接重构与冻结二次型检查通过；固定网格、码合法、scale/zero/g_idx保持、pack/reload检查和固定样例logit回归通过。
- 原生传播：24/24个block通过；141条逐边界对照均通过，覆盖文章0、64、127并保留完整token轴。
- 完整模型SHA256：0fa87105e36deac2ead4c871e6e0a802590418e6ccacd616929cdf91ecc8ec0c。具名张量SHA清单包含290个张量，SHA256 148edfee1108593cd62d160cf8c227c5205f2a8a027acebfbb30c8f05a5fb745。与M2及历史R2的逐具名张量比较见DEPLOYMENT_COMPARISON.json与reports/TENSOR_DIFFERENCES_VS_R2_AND_M2.csv。
- M2/v55/历史R2配置原始JSON因保存端序列化版本不同而不完全相同；当前同一loader有效配置比较：True，见CONFIG_COMPARISON.json。新checkpoint张量不同于历史R2，所以没有复用历史PPL。

## 成本

- 新增v55共同账本成本：5052.272秒；其中CPU-only 180.483秒，非CPU-only进程wall 4871.788秒（包含CPU工作，不等于纯GPU kernel时间）。
- 本轮有效正式修复增量成本（新修layer2至23的完整依赖组块wall）：4240.102秒；新接受更新 8801399。有效复用的block0/1组未在v55重复修复，其历史生成时间已经包含于继承账本。本次完整formal runner进程wall（含模型构建、恢复前缀、修复、传播与保存/reload至原配置序列化断言）：4383.618秒。
- 原M2构建成本另计 2115.935秒。共同累计：11032.312/12600秒，剩余 1567.688秒；最终收尾前预留满足360秒。
- 分阶段测量见STAGE_COST_SUMMARY.csv、BLOCK_TIMINGS.csv及STAGE_TIMINGS_EVAL_CONTINUATION.csv。共享C是否新做或共享情况由原冻结实现决定；此轮未改动。

## 评测与解释

- WT2/C4均实际运行一次，窗口NLL与窗口valid_tokens原始记录保存在evaluation目录；NLL按valid_tokens加权后汇总。没有读取P3或依据PPL调参。
- 局部search重构误差不替代PPL。本轮是已有离散修复思路在M2上的应用验证，不是官方ReQuant复现，也不构成原创性声明；一次运行不支持跨seed稳定性。
- v45/v46/v49历史PPL只作为各自checkpoint的实际测量背景；其校准传播有效性仍是UNKNOWN，未作为严格公平因果对照。
- 历史最终化断言失败被保留在FAILED_ATTEMPTS.jsonl：raw config JSON不相等导致停止；本轮通过有效配置解析和独立新PPL完成了v55交付，没有改写为“历史runner当时成功”。

## 复现与交付索引

- 冻结协议与来源锁：PROTOCOL.json、RESUME_SOURCE_LOCK.json、SOURCE_AUDIT.json。
- 数值和恢复：NUMERICAL_VALIDATION.json、PROPAGATION_RESTORE_CHECKS.csv、PROPAGATION_NATIVE_RESULTS.csv、CORRECTNESS.jsonl。
- 模型：R1/checkpoint/final/model.safetensors、CHECKPOINT_LOCK.json、FINAL_CHECKPOINT_LOCK.json、NAMED_TENSOR_HASHES.csv。
- 质量：QUALITY_COMPARISON.csv、QUALITY_EVALUATION_SOURCE.json、evaluation/v55_wikitext_WINDOWS.csv、evaluation/v55_c4_fixed_budget_WINDOWS.csv。
- 成本：ACTIVE_JOB_LEDGER.json、FINAL_BUDGET_ACCOUNTING.json、STAGE_COST_SUMMARY.csv。
- 复现命令与失败边界见REPRODUCE.md。所有历史checkpoint、报告与公共实现保持只读。
