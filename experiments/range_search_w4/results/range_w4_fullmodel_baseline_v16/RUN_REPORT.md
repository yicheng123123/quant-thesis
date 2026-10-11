# v16 audit run record

实际执行：读取用户附件/历史报告与源代码；官方固定commit浅克隆；CPU读取本地Arrow及tokenizer冻结cal/E1；bounded官方C4读取并tokenize；4次极小GPTQ core调用共70row-alpha slots；生成协议/计划/预算与检查。没有formal full-model quantization、Omni training、PPL、PrefixQuant或Gate实验。

命令（repo根目录，已有文件不可覆盖；completed结果应复用）：

```powershell
.venv-gptq\Scripts\python.exe -B experiments\range_search_w4\results\range_w4_fullmodel_baseline_v16\collect_audit.py
.venv-gptq\Scripts\python.exe -B experiments\range_search_w4\results\range_w4_fullmodel_baseline_v16\freeze_c4.py
.venv-gptq\Scripts\python.exe -B experiments\range_search_w4\results\range_w4_fullmodel_baseline_v16\budget_smoke.py
.venv-gptq\Scripts\python.exe -B experiments\range_search_w4\results\range_w4_fullmodel_baseline_v16\build_documents.py
.venv-gptq\Scripts\python.exe -B experiments\range_search_w4\results\range_w4_fullmodel_baseline_v16\verify_audit.py
```

文档生成是本轮交付过程，前四个命令不是formal实验入口；clean reproduction需空的v16目录及同commit official_source。已冻结目录只运行verify，避免重复smoke/更换cal。C4脚本第一次sandbox网络失败后经限定审批成功；这次审批只针对固定1056doc数据读取，不等于formal实验授权。软件准备错误详情见SMOKE_RESULTS。

历史文件前后SHA由verify核查；官方code blob核查排除已披露case-colliding images。原repo存在的README/benchmark修改均保留，本轮不新增对它们的修改。最终verdict以PROTOCOL_DRAFT/VERIFY_AUDIT为准，STOP后不启动大跑。
