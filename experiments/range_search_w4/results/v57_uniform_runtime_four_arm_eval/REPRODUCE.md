从 PowerShell 运行 CPU-only 预检：

  & 'E:\quant-thesis\.venv\Scripts\python.exe' 'E:\quant-thesis\experiments\range_search_w4\results\v57_uniform_runtime_four_arm_eval\preflight_v57.py'

脚本只读 v18/v55/v56 checkpoint、config、manifest 与历史窗口；只在 v57 写入哈希、算术复核、来源审计和预算预检。不运行完整 HF forward、CUDA 评测或任何量化/修复。
