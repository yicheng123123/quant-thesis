# v23 E1 — Block/Downstream-Aware Objective Diagnostic

## 执行状态与协议

- Verdict: **`E1_BLOCK_OBJECTIVE_SIGNAL_NOT_SUPPORTED`**。
- Correctness: **PASS**；alpha=1 native GPTQ parity: **True**。
- 10 个 alpha candidate tensor 在 local 与 block selector 间共享；held 数据的选择锁 `c62e902ef4bd9e22135cfe78b397cbe5ecf7eb4ea472e25399ca33de95d55c1a` 在 held block loss 计算前写入并校验。
- 模型：Qwen2.5-0.5B-Instruct，revision `7ae557604adf67be50417f59c2c2f167def9a775`，FP16、SDPA、TF32 off；W4A16 symmetric G128，native dynamic GPTQ，damp=0.1、block=128、desc_act=false、static_groups=false。
- search/held 各 16 篇完整 WT2 train 文档，每篇前 2048 tokens；manifest SHA256 `0d7b213349a6b04d9f9e59c703ab54ec8ad7beaf7f4eac7a61e4052a270f05c1`。WT2 test、C4 eval、PPL 均未运行。
- 目标为 L0/L12/L23 的 q_proj/up_proj，各预冻结 4 行、2 splits，共 48 row-split units。held oracle 只用于 retrospective ranking/regret。

## 实际工作量与成本

- 完成 full-Linear GPTQ candidate slots：120/120；事件记录的 full-Linear GPTQ 调用尝试：120，完成事件：120。
- 完成 candidate block-score 文件：960/960；单文档 block forward 调用尝试：15360，冻结唯一评分槽位为 15,360（search 7,680 + held 7,680）。若有中断重试，attempts 会高于唯一槽位数。
- 同时记录 345600 个 search all-row local score positions 和 127 个实际 held selected local score positions（上限 144）；它们不是 GPTQ execution 数。
- 量化前缀 capture/teacher/block-count：1536 个 M2 decoder-layer forward + 192 个 fixed teacher decoder-layer forward。
- 预注册 1.5× 成本闸门：中心估算 0.093 h，保守估算 0.140 h，状态 **E1_COST_GATE_PASS**（6 h 上限）。正式会话墙钟 0.060 h；本机峰值显存 allocated/reserved 为 1.852/1.918 GiB。
- 成本拆分见 `E1_COST_GATE.json` 和 `RUN_REPORT.json`，含候选 GPTQ、local scoring、H/Hinv、search/held block forwards、prefix/teacher、stitch、candidate checkpoint IO。

## Primary held block metric

- Block-selected vs local-selected win/tie/loss：14/17/17。
- `delta = block-selected held block MSE - local-selected held block MSE`：median 0，mean 9.4842767e-09；负值有利于 block selector。
- Pooled SSE delta (block − local)：13.36606；pooled relative improvement：-0.000121%。median/mean per-unit relative improvement：0.000000%/0.000119%。
- Worst case：`L23/up_proj/R2432/S0`，relative improvement -0.006468%（负值代表 block selector 更差）。每单位数值 tie epsilon 按冻结规则计算，见 `SELECTOR_COMPARISON.csv`。
- alpha disagreement：37/48 = 77.08%。只看 disagreement 且 held 非平局单位，block selector win fraction=0.45161290322580644。

## Ranking、tail 与结构差异

- Search→held block Spearman mean：local selector 0.72929；block selector 0.69697。held oracle regret 的中位数：local 2.6991422e-09，block 2.2338694e-09。
- 相对 native alpha=1，local selector 严重负尾（<-5%/<-10%/<-20%）：0/0/0；block selector：0/0/0。最差 native-relative gain 分别为 -0.008258%（L0/up_proj/R2432/S1）和 -0.006468%（L23/up_proj/R2432/S0）。
- Projection descriptive mean block-vs-local relative improvement：q_proj -0.000165%；up_proj 0.000403%。
- L0: n=16, block-vs-local win/tie/loss=6/4/6, mean relative improvement=0.000485%.
- L12: n=16, block-vs-local win/tie/loss=2/10/4, mean relative improvement=0.000015%.
- L23: n=16, block-vs-local win/tie/loss=6/3/7, mean relative improvement=-0.000144%.
- 分解是小样本描述结果，不作普遍因果结论。完整 alpha 频数、row/document 损失、actual scale/zero/g_idx 见 CSV/JSON 原始文件。

## 结论与下一步

本轮只检验同一动态 GPTQ candidate 下更换 selection objective 的局部信号，不构成整模型量化结果。Verdict `E1_BLOCK_OBJECTIVE_SIGNAL_NOT_SUPPORTED` 对应的下一步：停止 PRIMARY block-objective 路线；下一轮只按 v22 冻结 BACKUP 设计 2-region structured group-alpha 诊断，不在本轮执行。

未验证：整模型 PPL、下游任务质量、full-model block-aware stitch 收益、部署或推理速度变化。离线搜索成本不代表在线推理加速。

## Selection leakage 与复现

独立产物 `HELD_SELECTION_LEAKAGE_AUDIT.json` 验证：search 选择锁在 UTC `2026-10-01T18:36:47.618253+00:00` 写入；首个 held block score 于 `2026-10-01T18:36:47.621257+00:00` 开始。480 个 search 与 480 个 held landscape 的文档索引分离，选择值可仅由 search landscape 重算得到。该审计检查了 12 个 Hessian 的 search-only 文档索引、120 个 candidate 的文档索引及 weight/scale/zero/g_idx/local-score hashes，以及 24 次 v18 M2 target 恢复。

从仓库根目录 PowerShell 复现（继续使用本目录已冻结的 manifest）：

```powershell
.\.venv-gptq\Scripts\python.exe -B .\experiments\range_search_w4\results\range_w4_block_objective_e1_v23\run_e1_v23.py prepare
.\.venv-gptq\Scripts\python.exe -B .\experiments\range_search_w4\results\range_w4_block_objective_e1_v23\run_e1_v23.py preflight
.\.venv-gptq\Scripts\python.exe -B .\experiments\range_search_w4\results\range_w4_block_objective_e1_v23\run_e1_v23.py run
```

正式执行前要求 preflight correctness PASS 且 1.5× conservative estimate 不超过 6 小时。候选、Hessian、逐文档 block score 和 held selection lock 均有缓存及 hash 检查，可从中断处恢复。

Runner source SHA256: `976bf4e051a4223532270985bb88193ac33e288e3b7d686790999623f4ea8ba7` (recorded in `RUN_REPORT.json`).
