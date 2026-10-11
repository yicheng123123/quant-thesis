# v59 — 同一进程四 checkpoint 正式质量对照

## 结果状态

八项完整评测均完成，进程号 `32696`；582 个 WT2 窗口和 253 个 C4 窗口分别覆盖四个模型。窗口来源与计分 mask 独立重核通过；四份 checkpoint 的 290 项具名张量哈希均匹配。
本轮四臂数值全部来自 v59 这一个 Python 进程，不混入历史 PPL。历史校准传播有效性仍是 `UNKNOWN`；统一评测不补充传播证据。

## 统一四臂结果

NLL 是 `sum(window_nll) / sum(valid_tokens)`，单位 nats/valid token；PPL 为 `exp(NLL)`。PPL 变化百分比以对应对照模型为分母。

| 数据集 | M1 NLL / PPL | M2 NLL / PPL | M1+修复 NLL / PPL | M2+修复 NLL / PPL |
|---|---:|---:|---:|---:|
| wikitext | 2.630756906 / 13.884275011 | 2.621826617 / 13.760836419 | 2.612333695 / 13.630823945 | 2.602638921 / 13.499314693 |
| c4_fixed_budget | 3.101590568 / 22.233286732 | 3.092519681 / 22.032523022 | 3.097449194 / 22.141400770 | 3.086241946 / 21.894641925 |

### 差值与交互量

- **wikitext**：M1 修复 ΔNLL `-0.018423211`，PPL `-1.8255%`；M2 修复 ΔNLL `-0.019187696`，PPL `-1.9005%`。修复前 M2−M1 ΔNLL `-0.008930289`，修复后 M2+修复−M1+修复 ΔNLL `-0.009694774`。描述性交互量 `I=(NLL_R2−NLL_M2)−(NLL_R1−NLL_M1)=-0.000764485` nats/valid token。
- **c4_fixed_budget**：M1 修复 ΔNLL `-0.004141374`，PPL `-0.4133%`；M2 修复 ΔNLL `-0.006277735`，PPL `-0.6258%`。修复前 M2−M1 ΔNLL `-0.009070887`，修复后 M2+修复−M1+修复 ΔNLL `-0.011207248`。描述性交互量 `I=(NLL_R2−NLL_M2)−(NLL_R1−NLL_M1)=-0.002136361` nats/valid token。

交互量为单次固定 checkpoint 的描述性差值；负值表示本轮 M2 修复臂的 NLL 变化相对更有利。不表示统计显著性、跨 seed 稳定性或协同机制。

## 运行时和来源

- Python `3.12.10` (`E:\quant-thesis\.venv\Scripts\python.exe`)；PyTorch `2.6.0+cu124`；PyTorch CUDA build/runtime `12.4`；Transformers `5.16.1`；tokenizers `0.23.2`。GPU：`NVIDIA GeForce RTX 4060 Laptop GPU`。
- 同一 loader：FP16、CUDA、`attn_implementation=sdpa`、eval、`use_cache=False`、TF32 关闭、FP32 matmul precision `highest`。SDPA backend 开关见 `PROCESS_ENVIRONMENT.json`；实际底层 kernel 未插桩。
- Qwen2 model code SHA256 `34390fea7648b444d5727abef0b85acceba28f856aedec6c335d51253806bfee`；PPL core `benchmark/eval/ppl.py` SHA256 `5062225fc58d6d1272ba7c828f697ff9540407db5409340d2a94662ff32841a1`；v58 runner/evaluator wrapper SHA256 `c27fb3a5dbf2f940a7ed0fd465ac02a07da8de94c9dc2a060adea962a63ed2df`。v57 来源清单和逐 checkpoint/张量核验见 `CHECKPOINT_SOURCE_AUDIT.csv`、`MODEL_LOAD_AUDIT.csv`。
- 模型标识与 revision 来源于 v18 冻结协议，见 `MODEL_SOURCE_AUDIT.json`。
- WT2 manifest/token hashes：`acd46b20a6b84b534d5625bfab505d599001679a775d619960a40e8da4f62f46` / `8ffcee84dd4b684ec631cb52845e18c9c181ce6fdb8b26d65a4ffdc3d35fbda3`；C4：`0b0d70d70d3c1e0ba62126c81a3f3e97531e58d6701236fb8b3315450b7188d4` / `bc9a7575527abbc41352352a0439c6bee062ea265b7bd03120704e83f4af7053`。WT2 582 窗口、299,077 valid tokens；C4 253 窗口、131,071 valid tokens。窗口起止、输入切片哈希和 labels mask 哈希逐行复核。
- Luna Max 是用户指定设置；本地评测进程的模型选择接口不可查询，不能独立确认当前会话模型身份。

## 成本

- v59 上限 3600 秒；包括目录建立以来的预检/准备计费及正式 runner，本轮保守总计 `3163.730` 秒，余 `436.270` 秒。目录创建至 runner 入口计入 `679.793` 秒；其中独立 CPU 预检脚本实测 `4.416` 秒。GPU 评测与其他 CPU wall 分开列于 `COST_ACCOUNTING.json`。
- 正式运行前记录并修正一次 CPU 来源哈希断言混淆（v56 helper 与 v58 runner 是不同文件）；发生在模型加载/前向前，见 `PRE_RUN_DIAGNOSTIC.json`，已包含在预运行时间计费中。
- 另一次只读路径定位最初把版本目录假定在错误层级，随后改用项目实际嵌套目录；两项均为 CPU 侧预检，见 `PRE_RUN_DIAGNOSTIC.json`。
- 一次最终 CPU 汇总启动因 Python codec 名称拼写失败，未写出结果表；更正后完成，见 `POST_RUN_DIAGNOSTICS.json`，包含在 60 秒收尾保守计费中。
- 正式 runner 活动 wall（含来源核验、加载、张量校验、八项评分及卸载）为 `2423.937` 秒；八项同步 evaluator 阶段合计 `2383.035` 秒，模型加载 `4.488` 秒，290 张量核验 `3.875` 秒，卸载 `0.921` 秒。v58 的四对投影加准备/收尾预估为 `2851.161` 秒，实际计费高 `312.569` 秒；其中 M1+修复 WT2 实测 `694.514` 秒。收尾另保守计 `60.0` 秒。预测只作为预估，实际总成本以上述计费为准。
- 期间 GPU telemetry 曾显示约 98% 利用率、7.5–7.9 GiB/8 GiB 显存占用；WDDM 未提供进程归属，见 `GPU_TELEMETRY_SAMPLES.csv`。模型自身 CUDA 峰值 allocation/reservation 见 `MODEL_LOAD_AUDIT.csv`。

## 解释边界

本轮只对四个现有 checkpoint 做加载、哈希核验和正式评测，没有量化、整数修复、候选搜索、新校准或 P3。比较回答的是这些固定 checkpoint 在同一进程/运行时下的质量差异。它不能把历史校准传播有效性从 UNKNOWN 改为 PASS，也不提供统计显著、跨 seed 稳定或原创贡献证据。

## 文件与复现

- 八项窗口和聚合：`evaluation/<arm>_<dataset>_WINDOWS.csv`、`evaluation/<arm>_<dataset>.json`。独立复算：`INDEPENDENT_WINDOW_RECALC.csv`、`CSV_JSON_WINDOW_AUDIT.csv`。
- 四臂主表：`ARM_RESULTS.csv`、`FOUR_ARM_COMPARISON.csv`。环境/来源/成本：`PROCESS_ENVIRONMENT.json`、`MODEL_SOURCE_AUDIT.json`、`CHECKPOINT_SOURCE_AUDIT.csv`、`INPUT_SOURCE_AUDIT.csv`、`MODEL_LOAD_AUDIT.csv`、`STAGE_TIMINGS.csv`、`COST_ACCOUNTING.json`。
- 正式评测命令（工作目录 `E:\quant-thesis`）：`.venv\Scripts\python.exe experiments\range_search_w4\results\v59_same_process_four_checkpoint_eval\run_v59_eval.py`。本 v59 目录已完成且预算锁定；复跑需使用空白的独立输出目录并改写 OUT 路径，不要覆盖或重用本次窗口。CPU 独立复算脚本是 `finalize_v59.py`。

