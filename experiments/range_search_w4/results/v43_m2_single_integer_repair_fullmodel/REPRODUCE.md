# v43 复现与恢复

本轮为v42冻结算法的预算扩展。历史v42/v18目录和公共量化实现只读。当前v43已有产物时不得重新初始化覆盖。

从 E:/quant-thesis 执行的本轮命令：

```powershell
& .venv-gptq/Scripts/python.exe -B -X utf8 experiments/range_search_w4/results/v43_m2_single_integer_repair_fullmodel/prepare_v43.py
& .venv-gptq/Scripts/python.exe -B -X utf8 experiments/range_search_w4/results/v43_m2_single_integer_repair_fullmodel/run_v43.py
& .venv-gptq/Scripts/python.exe -B -X utf8 experiments/range_search_w4/results/v43_m2_single_integer_repair_fullmodel/finalize_v43.py
```

`prepare_v43.py`仅用于全新独立目录：预建输出/checkpoint/reports目录并检查可写，检查v42冻结源码/协议/pilot和原始校准/参数/hash，逐字节复制原运行函数及原始M2整数码。若手动复现须复制三个v43入口到results下另一个人工指定空目录；不自动创建下一版本或再跑模型。

已提交组checkpoint包含当前R输入哈希、C哈希、整数码、原网格参数、逐步修改、候选评价数和提交状态。以下恢复命令适用于外部中断：尚未写最终部署锁、未发生已定位的冻结接受规则终止，且原180分钟截止时间仍有余额时，可用原字节/协议恢复：

```powershell
& .venv-gptq/Scripts/python.exe -B -X utf8 experiments/range_search_w4/results/v43_m2_single_integer_repair_fullmodel/run_v43.py --resume
```

恢复不获得新180分钟；原GPU_START时间不变。模型从原M2重新加载后复制已提交组参数，重放本次R前缀，已提交组不再修复。最终DEPLOYMENT_LOCK为终态，不再次修复或追加PPL。本次 NUMERICAL_STOP_FROZEN_RULE 已定位到FP32批量近零下降判据冲突，属终态，不使用相同恢复命令反复执行已知失败，不自动替换评分/放宽接受条件。

模型使用原dense FP16/SDPA部署；原M2没有packed推理kernel，辅助4位packing只保存整数码。WT2/C4使用正式冻结token ID manifest、max_length2048、stride512、valid-target加权原共享evaluator，每语料各一次。

运行完成后的CPU只读结果复核/报告可再次执行 finalize_v43.py；不会模型前向、GPU评分或PPL。
