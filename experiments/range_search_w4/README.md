# W4 固定 GPTQ 范围诊断

本目录只研究 W4A16 均匀权重量化的量化范围。无梯度、反向传播、学习参数、旋转、激活或 KV 量化。历史 checkpoint 和 benchmark 文件均保持只读。

## 运行

在 `E:\quant-thesis` 下使用现有 `.venv-gptq`：

```powershell
.\.venv-gptq\Scripts\python.exe -m pytest experiments\range_search_w4\test_core.py -q
.\.venv-gptq\Scripts\python.exe -m experiments.range_search_w4.run --phase smoke --run-name my_smoke_v7
.\.venv-gptq\Scripts\python.exe -m experiments.range_search_w4.run --phase representative --run-name my_reproduction_v7
.\.venv-gptq\Scripts\python.exe -m experiments.range_search_w4.summarize --run-name my_reproduction_v7
```

`--run-name` 必须使用新的目录名。重复运行同一已完成的阶段会直接命中行级 JSON 缓存，不重新执行候选。中断后重启相同目录会跳过已完成行；请保留 `manifest.json` 和各行 JSON。默认目录是 `results/range_w4_local_v7/`，保存本轮结果。`protocol.json` 是冻结配置；新研究定义应使用新的协议 ID 和运行目录。

## 文件

- `core.py`：GPTQModel 2.2.0 的 affine 量化网格、固定范围下的 GPTQ 误差补偿、分组交叉项。
- `run.py`：按完整 WikiText 文章分割数据，捕获 FP 输入，执行代表行搜索与组诊断，保存原始 JSON/CSV。
- `summarize.py`：仅从原始产物生成汇总，不调用模型。
- `test_core.py`：6 个必要正确性检查，包括与原 GPTQModel 对齐及外部指定参数确实生效。
- `REPORT.md`：协议审计、数值解释、成本和推进判断。

历史 `benchmark/results/raw/gptq_g128.json` 的动态组重估 GPTQ 不是本目录的固定外部范围基线。两者不作直接因果比较。
