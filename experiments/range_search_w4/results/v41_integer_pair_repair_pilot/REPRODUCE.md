# 复现

当前结果目录不覆盖。将源文件复制到下一个独立空闲版本目录：prepare.py、cpu_checks.py、run_pilot.py、summarize.py、audit_outputs.py、plot_curves.py、budget_guard.ps1、PROTOCOL.json。
在 E:/quant-thesis 工作目录依次运行（将 NEW_DIR 替换为实际新目录）：

```powershell
.\.venv-gptq\Scripts\python.exe -B -X utf8 NEW_DIR\prepare.py
.\.venv-gptq\Scripts\python.exe -B -X utf8 NEW_DIR\cpu_checks.py
# 确认其他 GPU 研究进程全部结束；不得与 v40 并发。
.\.venv-gptq\Scripts\python.exe -B -X utf8 NEW_DIR\run_pilot.py
# 在另一个终端同时启动外部全轮墙钟预算监督器：
powershell -NoProfile -File NEW_DIR\budget_guard.ps1
# 两个进程结束后，在主终端生成报告：
.\.venv-gptq\Scripts\python.exe -B -X utf8 NEW_DIR\summarize.py
.\.venv-gptq\Scripts\python.exe -B -X utf8 NEW_DIR\audit_outputs.py
# 独立 CPU 图，使用本机已有 matplotlib，无需安装包：
C:/Users/gaoyi/AppData/Local/Programs/Python/Python312/python.exe -B -X utf8 NEW_DIR\plot_curves.py
```

prepare 为 CPU 来源审计；run_pilot 包含 900 秒全轮强制退出定时器、760 秒 search 截止及 880 秒评分/收尾截止。不同机器的截止覆盖可能不同，未完成目标应保留。源码不导入模型或 GPTQ、不改变公用实现。所有选择只使用 search；历史 epsilon 读取 held 汇总的冻结阈值，未用 held 损失构建配对图或选择码。报告程序只汇总冻结结果。

正常复现只需一次尝试。发生错误时先归档失败源码/产物和账单；若仍允许重试，外部 budget_guard.ps1 以第一次尝试的 UTC 起点严格计入中间诊断墙钟，重试也不得延长 900 秒。首次失败字段命名问题已修正，无需人为重现该失败。v41 的失败成本单独保存并纳入实测总账，不当作正常运行时间预测。
