# v58 — v55 M2 修复模型统一评测运行时重评

## 结论

v55 checkpoint 在本轮冻结输入上完成 WT2 test 与固定预算 C4 validation 两项实际评测。相对 v56 的 M1＋修复历史评测，v55 M2＋修复的本轮点估计在两项数据上都更低（WT2 NLL 低 -0.009694774 nats/token，C4 低 -0.011207248 nats/token）。

但这不是可确认的同运行时主对照：v56 成功评测进程没有记录 Python 和 Transformers 版本，且未记录实际选择的底层 SDPA kernel。按本轮冻结规则，**不能据此断言 M2 初始化修复模型在两项指标上已被证明优于 M1 初始化修复模型**。观察值方向一致，比较资格不足。v56 数值只作上下文参考；未计算四臂交互量。

## 完整评测结果与上下文比较

本轮按完整窗口的 `sum(window_nll) / sum(valid_tokens)` 计算加权 NLL，再以 `exp(NLL)` 计算 PPL。两个本轮数据集全部窗口均完成，独立 CSV 汇总通过。

| 数据集 | M2＋修复 v55/v58 NLL | M2＋修复 PPL | M1＋修复 v56 NLL（上下文） | M1＋修复 PPL（上下文） | NLL 差（M2−M1） | PPL 相对差 |
|---|---:|---:|---:|---:|---:|---:|
| wikitext | 2.602638921 | 13.499314693 | 2.612333695 | 13.630823945 | -0.009694774 | -0.9648% |
| c4_fixed_budget | 3.086241946 | 21.894641925 | 3.097449194 | 22.141400770 | -0.011207248 | -1.1145% |

本轮 v55 实测窗口：WT2 为 2.602638921 NLL、13.499314693 PPL、299,077 valid tokens、582 窗口；C4 为 3.086241946 NLL、21.894641925 PPL、131,071 valid tokens、253 窗口。

### v55 checkpoint 历史评测差异

| 数据集 | v55 历史 PPL | v58 当前 PPL | 绝对变化 | 相对变化 |
|---|---:|---:|---:|---:|
| wikitext | 13.499318701 | 13.499314693 | -0.000004008 | -0.000030% |
| c4_fixed_budget | 21.894641925 | 21.894641925 | +0.000000000 | +0.000000% |

差异只描述运行时/执行结果变化，不归因于某一软件组件。历史校准传播有效性仍为 `UNKNOWN`；本轮统一评测不补充该证据。

## 来源与运行时

- v55 `model.safetensors` SHA256：`0fa87105e36deac2ead4c871e6e0a802590418e6ccacd616929cdf91ecc8ec0c`；checkpoint lock、config SHA 和 290 个具名张量来源清单均核对通过。v57 已对 290 个张量逐项重算并与 v55 清单匹配；本轮再次核对 checkpoint 与清单 SHA。
- 模型与 tokenizer revision 均为 `7ae557604adf67be50417f59c2c2f167def9a775`；实际评分直接使用冻结 token-ID manifest，不重新分词。
- loader：Transformers `AutoModelForCausalLM.from_pretrained`，`local_files_only=True`、FP16、`attn_implementation="sdpa"`、`eval()`、CUDA。加载后 dtype/device 为 `torch.float16` / `cuda:0`，模型处于 eval，归一化配置 SHA 为 `a33901fd787a1385efbdd5574d88012774c97110064a3f6a2cb05d070cba9dd2`，与预检一致。evaluator 显式使用 `use_cache=False`，TF32 关闭。
- 当前评测进程：Python 3.12.10、PyTorch 2.6.0+cu124、CUDA build 12.4、Transformers 5.16.1、tokenizers 0.23.2、设备 NVIDIA GeForce RTX 4060 Laptop GPU。
- v56 协议记录 PyTorch 2.6.0+cu124，并请求 SDPA；v56 成功评测进程的 Python/Transformers 版本缺失。v56 保存 config 内 `transformers_version` 属于保存时元数据，不能证明评测进程版本。底层 SDPA kernel 两边都未实测。因此完整 runtime identity 为未证实。
- v56 wrapper evaluator SHA 与本轮相同：`bbe1e4c22200f1ccb3d85f35a274c44cf5d46cd606160b59b0cd9afc346bae11`。本轮当前 core evaluator `benchmark/eval/ppl.py` SHA 为 `5062225fc58d6d1272ba7c828f697ff9540407db5409340d2a94662ff32841a1`；v56 未单独保存当时 core 文件 SHA。
- tokenizer 不重新分词；使用锁定 manifest 中的原始 token ID。WT2 manifest/token SHA 与 C4 manifest/token SHA均匹配冻结记录，窗口边界为 max_length=2048、stride=512，mask 仅计新增目标 token，不添加 special token，不使用 chat template。
- v18 的 M1/M2 历史测量使用 Transformers 4.49.0；v55 历史 M2＋修复使用 PyTorch 2.5.1+cu121。四臂统一运行时证据不成立，不计算交互量。

## 成本与失败

- v58 预算上限：3600 秒；runner 实测活动时间：476.123 秒；再保守计入报告与审计 CPU 时间 5.0 秒，总预算计费 481.123 秒，余额 3118.877 秒。历史账本未重置、未混算。
- 本轮实际模型加载：4.665 秒；WT2 评测：300.344 秒；C4 评测：132.899 秒。另有约束准备、写盘、聚合等进程时间；完整 runner process wall 445.342 秒。
- 另保守计入预运行只读核查 30 秒；首次 CPU 预检遇到 v57 CSV UTF-8 BOM 表头，发生在模型加载及前向前，失败尝试约 30.780 秒（其中包含上述 30 秒保守核查额）；修正 BOM 读取后成功完成。未发生 GPU 模型前向失败。
- v56 实测 eval-only 成功 continuation wall 为 2283.576 秒（含加载），只用于预估；v58 本轮实测活动时间为 476.122 秒。两者差异不据此归因于特定软件或硬件因素。

## 交付和复现

- 原始窗口：`evaluation/v58_wikitext_WINDOWS.csv`、`evaluation/v58_c4_fixed_budget_WINDOWS.csv`；逐窗口可恢复 journal 位于同名数据集子目录。
- 结果与独立聚合：`evaluation/v58_wikitext.json`、`evaluation/v58_c4_fixed_budget.json`。
- 来源、环境、loader、预算、适格性：`SOURCE_AUDIT.json`、`ENVIRONMENT_LOCK.json`、`LOADER_CHECK.json`、`COST_ACCOUNTING.json`、`FOUR_ARM_ADMISSIBILITY.json`。
- 对照明细：`SOURCE_AUDIT.csv`、`RUNTIME_COMPARISON.csv`、`INPUT_SOURCE_AUDIT.csv`。
- 两模型数值上下文表：`UNIFIED_COMPARISON.csv`（明确标为不适格的上下文比较）；v55 历史差异：`HISTORICAL_V55_DIFFERENCE.csv`。
- 复现命令：`.venv\Scripts\python.exe experiments\range_search_w4\results\v58_m2_repaired_unified_eval\run_v58_eval.py`，工作目录为 `E:\quant-thesis`。断点续跑时保持相同 lock 与 manifest，完成窗口将通过输入/掩码/lock 哈希校验后跳过。

本轮没有量化、修复、候选搜索、新校准或 P3 读取；不声称统计显著、跨 seed 稳定、传播正确性已由本轮证明，或算法原创性。
