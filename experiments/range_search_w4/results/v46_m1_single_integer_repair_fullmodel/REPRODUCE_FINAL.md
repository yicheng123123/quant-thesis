# v46 final reproduction

The original launcher stopped because its formal runtime still resolved the legacy V42 repair function. That attempt and its 14 committed but invalid groups are retained under diagnostics/attempt_001_legacy_v42_dispatch and excluded from R1. The correction only binds runtime.batch_repair to the byte-identical v45 repair_v45.batch_repair.

After verifying PRE_GPU_FREEZE.json and EXECUTION_AMENDMENT_001.json, run the corrected single-clock restart:

    .venv-gptq/Scripts/python.exe -B -X utf8 experiments/range_search_w4/results/v46_m1_single_integer_repair_fullmodel/run_v46_retry.py

The restart reloads original M1, recomputes all dependency groups from its own prefix, preserves the original GPU_START timestamp and charges the failed attempt against the same 180-minute limit.
