# v18 evaluator harness 两处修复

审查当前 `evaluate_v18.py`、共享 `benchmark/eval/ppl.py`、六份 raw JSON/window CSV、正式 `eval_v18.log`。失败尝试的独立源码快照和 SHA 未保存，因此**无法从仓库独立生成精确的 before/after Git diff**；下面的 before 行是前轮执行记录中明确观察到的代码，after 行以当前文件核验。此溯源限制不影响对正式六份输出的独立重算。

1. standalone import：原先在仓库路径进入 `sys.path` 前导入 `benchmark.eval.ppl`；当前先以 `Path(__file__).resolve().parents[4]` 插入 repo root，随后导入同一个共享 evaluator。只是 Python 导入定位，未改模型、token IDs 或 loss 公式。
2. 窗口断言：原先比较 `len(range(0, N, stride))`，这会把已覆盖尾部后的冗余窗口算入；当前断言 `max(1, ceil((N-max_length)/stride)+1)`。共享 evaluator 原本就在 `end == sequence_length` 时停止。WT2 582、C4 固定预算 253 个窗口；v19 从逐窗口 CSV 独立重建 begin/end/new_targets/valid_tokens 和尾部窗口，均一致。

当前代码先运行并核验 token 总数、窗口数及加权 NLL，之后才写 window CSV/raw JSON。正式日志显示 M0/M1/M2 × 两套数据全部 PASS，六份 raw 文件的模型/数据 hash 和窗口数学一致。失败尝试的原始 stdout/stderr 未作为单独归档保存，所以“它没有产生任何临时文件”的绝对说法无法独立证明；可以确认它未进入当前六项正式 raw 结果，且修复点不触及 `benchmark.eval.ppl.evaluate_ppl` 的 mask、loss 或加权聚合。
