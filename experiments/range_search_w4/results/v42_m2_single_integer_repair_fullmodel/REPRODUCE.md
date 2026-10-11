# 复现

历史来源全部只读；当前 v42 产物已经存在，禁止原地重新执行新实验。以下新实验命令只可在人工指定的全新独立结果目录中使用，目录须仍位于本 results 父目录，以保持来源定位。复制 audit_sources.py、freeze_protocol.py、run_v42.py、audit_outputs.py、finalize_report.py 到该目录。历史 v41/v18 及原 pristine snapshot 必须保留且哈希一致。

PowerShell，在 E:/quant-thesis 执行，$freshDir 为指定的新目录（本轮不自动创建或运行）：

```powershell
& .venv-gptq/Scripts/python.exe -B -X utf8 "$freshDir/audit_sources.py"
& .venv-gptq/Scripts/python.exe -B -X utf8 "$freshDir/freeze_protocol.py"
& .venv-gptq/Scripts/python.exe -B -X utf8 "$freshDir/run_v42.py"
& .venv-gptq/Scripts/python.exe -B -X utf8 "$freshDir/audit_outputs.py"
& .venv-gptq/Scripts/python.exe -B -X utf8 "$freshDir/finalize_report.py"
```

先确认没有其他 GPU 计算作业。本轮从首次 CUDA 预热计时，硬上限5400秒；pilot 通过后仍须满足冻结成本门控，保留360秒保存/reload/PPL预算。最终R先锁定再各一次正式评测，不按评测结果重做模型。

中断恢复仅在原5400秒截止前，且实现/协议哈希完全一致、尚未锁定/评测最终模型时：

```powershell
& .venv-gptq/Scripts/python.exe -B -X utf8 "$freshDir/run_v42.py" --resume
```

恢复重新加载原M2并重放已提交组的权重和前缀，跳过已提交组的修复；不会重新给5400秒。预算门控停止或最终模型锁定是终态，不恢复以追加搜索/再次PPL。

现有结果可仅做CPU复核（不运行模型或PPL）：

```powershell
& .venv-gptq/Scripts/python.exe -B -X utf8 experiments/range_search_w4/results/v42_m2_single_integer_repair_fullmodel/audit_outputs.py
& .venv-gptq/Scripts/python.exe -B -X utf8 experiments/range_search_w4/results/v42_m2_single_integer_repair_fullmodel/finalize_report.py
```
