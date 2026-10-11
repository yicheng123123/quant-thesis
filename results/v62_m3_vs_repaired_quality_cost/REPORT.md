# v62 — v61 仓库 M3/LWC 与 v55 M2+整数修复同进程质量/成本比较

## 结果

两套 checkpoint 在同一 Python 进程按 M3→v55 顺序、同一冻结 evaluator 完成四项评分。窗口共 582 WT2 test 和 253 C4 fixed-budget validation，每项计分 token 分别为 299,077 与 131,071；窗口、输入切片和 labels mask 均逐项核对，最终质量只使用本轮完整窗口。

| 语料 | v61 M3 NLL / PPL | v55 M2+修复 NLL / PPL | ΔNLL（M3−v55） | M3 相对 PPL 变化 |
|---|---:|---:|---:|---:|
| WT2 test | 2.607287326 / 13.562211055 | 2.602638921 / 13.499314693 | +0.004648406 nats/token | +0.4659% |
| C4 fixed-budget validation | 3.067665235 / 21.491666064 | 3.086241946 / 21.894641925 | -0.018576711 nats/token | -1.8405% |

M3 在 C4 fixed-budget 上更低，WT2 上 v55 更低，出现语料间取舍；没有一个方案在两项都占优。负 ΔNLL 表示 M3 更低。C4 是冻结的固定预算验证流，不能称标准 full C4。两者使用相同模型 revision、token manifest 与有效 Qwen2 配置。M3 导出 tokenizer 与锁定 revision 在固定探针编码、词表长度和 special-token id 上一致；正式 evaluator 直接使用冻结 token-ID manifest。两个 calibration manifest 的 SHA 相同，128×2048 的逐篇 token 哈希逐项匹配。v61 与 M2 的 ordered-token 总 hash 字段采用不同构造（per-sample hash 列表 vs 扁平 int64 token 数组），所以摘要值不同但底层 token 顺序相同，详见 `CALIBRATION_HASH_RECONCILIATION.json`。量化方法不同：v61 为仓库 M3/LWC W4A16 非对称 affine G128；v55 为 M2 scalar10 加 W4A16 对称 G128 整数码修复。这里比较的是两套完整方案，不能把差异单独归因于 LWC、scalar10 或修复。没有官方 OmniQuant 完整复现声明。

## 评测与正确性

进程 PID `67320`，Python `3.12.10`、PyTorch `2.6.0+cu124`、Transformers `5.16.1`、CUDA `12.4`、GPU `NVIDIA GeForce RTX 4060 Laptop GPU`；FP16、SDPA 请求、TF32 off、`use_cache=False`。两个有效配置指纹相同；全部评测窗口来自同一进程。SDPA 后端策略已记录，底层具体 kernel 未插桩。实际服务型号接口不可见，因此不能确认 Luna Max 是否被执行。

v61 的 61,440 个计划位置均有日志，实际优化器步为 61,438。两次跳步为：layer21 epoch0 sample0 (iteration 53761, found_inf=1.0, scale 65536.0→32768.0); layer21 epoch0 sample72 (iteration 53833, found_inf=1.0, scale 32768.0→16384.0)。冻结实现的动态 GradScaler 在 `found_inf=1` 时跳过优化器更新并降低 scale，不重试，也不补充样本；loss 有限，但日志中的 LWC gradient 与 gradient norm 标记非有限。更低层的 overflow 成因没有进一步拆解。v61 新模型未继承任何旧 M3 PPL。

每个窗口用 `window_loss × valid_tokens` 重算 NLL sum，除以该语料有效 token 总数得加权 NLL，再取 `exp(NLL)` 得 PPL。完整原始窗口、token 切片压缩档、mask 与逐窗哈希和独立复算在 `evaluation/`。

## 成本与内存

v62 四项评测阶段合计 898.384 秒；包含加载/哈希/卸载的正式 runner wall 923.755 秒。含 CPU 预检、两次失败诊断、首次 M3 加载核验失败、成功运行、保守计入的 30 秒预锁输入/来源工作及 12 秒 CPU 收尾的本轮预算计费为 1145.755 / 5400 秒。首个 GPU 尝试在前向前停止，原因是清单 dtype 文字 `torch.float16`/`float16` 表示形式不同、数值哈希与形状相同；修正仅标准化清单字段，未改权重、评测器或模型。该失败已留存并纳入计费。

历史成本与本轮评测成本分开：M1 原生 GPTQ build 648.876 秒，M2 scalar10 build 2115.935 秒；v61 M3 data-to-finalized calibration 2384.344 秒、包含加载保存/reload 的成功调用 2404.328 秒，连同失败 attempt 共 2442.875 秒；v56 M1 修复全模型正式阶段 5699.771 秒。v55 新修复的 block2–23 为 4240.102 秒；完整 24-block 净修复时间因 block0/1 继承恢复无法拆分，保留 NOT_IDENTIFIABLE。v55 的 11037.313 秒是多版本恢复链累计，不当作算法修复时间。未给出跨方法严格加速比。

内存的测量范围不同。v61 校准峰值来自既有 allocator 与资源采样；v62 推理峰值按模型单独记录 PyTorch allocated/reserved、0.5 秒设备总显存采样和主进程 RSS。设备总显存包含桌面及其他进程，WDDM 下单进程 GPU 显存不可得；采样峰值不是精确瞬时峰值。M1/M2/v55/v56 历史记录没有设备采样或 RSS 时在表内标 `NOT_MEASURED`。

## 边界与结论

WT2 与 C4 fixed-budget 是长期使用的 benchmark，不是新盲测。一次固定模型比较不代表统计显著或跨 seed 稳定；统一重评也不修复历史校准传播证据中的 UNKNOWN。按本轮结果，M3 与 M2+修复呈语料取舍。成本记录显示 M3 完整校准约 2,384 秒；M2+修复只可准确给出已有 block2–23 净修复及恢复链账本，缺少完整 24-block 净时长，不能做公平总成本胜负判断。是否继续修复路线没有跨语料质量或可比完整成本优势支持；本轮不启动后续实验。

## 产物

- checkpoint 来源：`CHECKPOINT_SOURCE_AUDIT.csv`
- v61 步数差异：`V61_STEP_COUNT_AUDIT.json`
- 校准 manifest 与 token hash 口径：`CALIBRATION_HASH_RECONCILIATION.json`
- 原始窗口、输入 token 切片、mask 来源和复算：`evaluation/`
- 统一环境：`PROCESS_ENVIRONMENT.json`
- 成本与内存：`COST_ACCOUNTING.json`、`QUALITY_COST_MEMORY_TABLE.csv`、`COST_SOURCES.csv`
- 协议、冻结锁、失败 attempt、账本：`FROZEN_PROTOCOL.json`、`RUN_LOCK.json`、`attempts/`
- 文件 SHA：`OUTPUT_MANIFEST.csv`
