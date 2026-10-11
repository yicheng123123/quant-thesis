# v45 reproduction

PowerShell, working directory E:/quant-thesis. All outputs are in this independent v45 directory; historical v42/v43/v44 and shared quantization code are read-only. No P3.

## Commands actually used

    .venv-gptq/Scripts/python.exe -B -X utf8 experiments/range_search_w4/results/v45_m2_single_integer_repair_fullmodel/prepare_v45.py
    .venv-gptq/Scripts/python.exe -B -X utf8 experiments/range_search_w4/results/v45_m2_single_integer_repair_fullmodel/preflight_v45.py
    .venv-gptq/Scripts/python.exe -B -X utf8 experiments/range_search_w4/results/v45_m2_single_integer_repair_fullmodel/cpu_tests.py
    .venv-gptq/Scripts/python.exe -B -X utf8 experiments/range_search_w4/results/v45_m2_single_integer_repair_fullmodel/run_v45.py
    .venv-gptq/Scripts/python.exe -B -X utf8 experiments/range_search_w4/results/v45_m2_single_integer_repair_fullmodel/finalize_v45.py

CPU tests ran before the formal implementation lock. Their controlled synthetic scoring fixtures are correctness tests, not model quality results. The separate raw-X FP64 failure diagnosis runs before any formal recovery.

Do not rerun preparation, tests, preflight or repair in a completed locked delivery: they would overwrite hashes or attempt a second run. Reproduce independently by copying implementations to a new named experiment directory, keeping all historical source paths unchanged, running preparation there once, and executing the sequence. No automatic reproduction/new version is authorized by this report.

Historical baseline provenance was checked in a second CPU process during formal repair, before the new R PPL files existed:

    .venv-gptq/Scripts/python.exe -B -X utf8 experiments/range_search_w4/results/v45_m2_single_integer_repair_fullmodel/baseline_audit.py

This command is only for the pre-evaluation phase. The final audit below repeats the baseline checks alongside completed R windows.

## Safe read-only verification

The final CPU audit replays persisted integer moves, rejection blacklist states, acceptance branch evidence, budgets and auxiliary u4 packing. It verifies inherited v44 group and v44→v43 provenance hashes and aggregates the historical M1/M2 and new R PPL windows. Running finalize_v45.py writes only the reports in this directory and creates no CUDA model.

    .venv-gptq/Scripts/python.exe -B -X utf8 experiments/range_search_w4/results/v45_m2_single_integer_repair_fullmodel/finalize_v45.py

## Numerical process

precision_check.independent_delta takes only raw X, current E and actual dense D. It computes a=X.float64@E.T, d=X.float64@D.T and sum(2*a*d+d*d)/N in4096-token chunks. The predictor/cache are passed only to the separate verification wrapper. PROTOCOL.json contains the fixed empirical absolute/relative criteria, dimensions, accumulation-length rationale, original FP32 scorer/strict-decline gate, grid, budgets and unchanged formal PPL protocol.

Original FP32 full-output-row matmul determines strict decline. FP64 verification only handles FP32-descending candidates that fail the old cross-precision discrepancy check; it never overrides a FP32 tie/rise. Formula/cache mismatches are hard failures. All reevaluations count against128*d and32 accepted updates; blacklist resets only after that row accepts.

## Recovery and final evaluation

GPU_START.json records one180-minute clock starting initial GPU diagnosis. IMPLEMENTATION_LOCK.json and NUMERICAL_PROCESS_LOCK.json freeze implementation and policy before formal recovery. An interrupted, unsealed run can use run_v45.py --resume only with the same hashes/clock and no final CHECKPOINT_LOCK. It loads committed groups without repairing them again. A correctness failure must be diagnosed rather than blindly resumed.

Original M2 +15 trusted v44 groups rebuilds the prefix. Uncommitted block3 down starts original M2 codes. Recovery input/C SHA are checked. New groups commit inqkv→o→up/gate→down→next-block order; X always comes from the current R prefix.

Final code/grid, configuration, saved/reloaded weights and64-token output regression must pass before DEPLOYMENT_LOCK. Only then are WikiText-2 test and fixed-budget C4 validation each evaluated once with the inherited max_length2048,stride512,token-weighted valid-label rules. Raw windows, hashes, stage timings and decision remain in the delivery.
