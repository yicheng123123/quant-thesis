# v59 复现

正式评测使用命令（工作目录 `E:\quant-thesis`）：

```powershell
.venv\Scripts\python.exe experiments\range_search_w4\results\v59_same_process_four_checkpoint_eval\run_v59_eval.py
```

本 v59 输出目录已完成并锁定预算；不要在此目录重新启动 runner。新的完整重放应使用空白的独立输出目录，将 runner 中的 `OUT` 指向新目录，并保留相同来源、运行时及冻结配置。runner 按 M1→M2→M1+修复→M2+修复顺序，在一个 Python 进程中调用冻结的 `benchmark.eval.ppl.evaluate_ppl`。它不量化、修复、搜索、校准或访问 P3。

`finalize_v59.py` 是 CPU-only 独立核验脚本；它逐项检查已存窗口、token 和 mask 哈希、进程 ID、有效 token 数、加权 NLL 与 PPL。跨进程恢复会被标为不满足严格同进程比较。
