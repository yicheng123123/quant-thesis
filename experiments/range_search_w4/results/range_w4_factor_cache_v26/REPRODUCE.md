# v26 复现

在 `E:\quant-thesis` 的 PowerShell 中运行；依赖、模型、正式校准数据和历史 checkpoint 沿用仓库现有文件，所有新增写入仅在本目录。

```powershell
& .\.venv-gptq\Scripts\python.exe experiments\range_search_w4\results\range_w4_factor_cache_v26\freeze_v26.py
& .\.venv-gptq\Scripts\python.exe experiments\range_search_w4\results\range_w4_factor_cache_v26\profile_v26.py
& .\.venv-gptq\Scripts\python.exe experiments\range_search_w4\results\range_w4_factor_cache_v26\detail_v26.py
& .\.venv-gptq\Scripts\python.exe experiments\range_search_w4\results\range_w4_factor_cache_v26\finalize_v26.py
```

`freeze_v26.py` 校验已有冻结协议和源文件哈希；`profile_v26.py` 在完整结果已存在时复用结果；`detail_v26.py` 同样复用完整分账。若从无产物的新 v26 目录复现，先冻结再运行 GPU profiling。`finalize_v26.py` 仅读取原始结果生成 JSON/CSV/报告数据，不重新运行 GPTQ。请不要删除现有原始结果以触发重复实验。

本轮没有新 held-out、PPL、整模型量化；完整 block 传播未重跑。GPU 累计预算硬上限 600 s；`RUN_STATUS.json` 保存续跑账本。`PROFILE_ATTEMPT_FAILURE_01/02.json` 记录两次在正式候选结果产生前的实现故障及保守预算扣除。
