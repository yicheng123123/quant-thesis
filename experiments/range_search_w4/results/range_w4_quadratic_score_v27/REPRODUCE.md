# v27 复现

在 `E:\quant-thesis` 的 PowerShell 中，以现有 `.venv-gptq` 运行。输入依赖 v26 已保存的正式 128×2048 L0 up/down 激活，以及冻结的 v18 模型和校准 manifest。所有新写入均在本 v27 目录。

```powershell
& .\.venv-gptq\Scripts\python.exe experiments\range_search_w4\results\range_w4_quadratic_score_v27\freeze_v27.py
& .\.venv-gptq\Scripts\python.exe experiments\range_search_w4\results\range_w4_quadratic_score_v27\run_v27.py
& .\.venv-gptq\Scripts\python.exe experiments\range_search_w4\results\range_w4_quadratic_score_v27\verify_wrapper_v27.py
& .\.venv-gptq\Scripts\python.exe experiments\range_search_w4\results\range_w4_quadratic_score_v27\finalize_v27.py
```

已完成目录再次执行时，`run_v27.py` 读取 `RUN_REPORT.json` 并跳过已完成矩阵，`verify_wrapper_v27.py` 复用 spot 结果，`finalize_v27.py` 只读取原始数据重建 CSV/JSON。候选由既有原生动态 G128 GPTQ 每个 alpha 生成一次，缓存于 `candidates/`，共享给两个评分器；候选缓存记录源码、H、W 哈希。不要删除已有结果以触发重复实验。

`RUN_STATUS.json` 记录本轮 GPU 累计墙钟并在新操作前检查 600 秒限制。原始逐候选、逐行分数保存在 `RAW_SCORES_*.npz` 和 `RAW_SCORE_COMPARISON.csv`；两个评分器的选行及最优/次优间隔在 `ROW_SELECTION.csv`。这不是新 held-out、PPL 或整模型量化复现命令。
