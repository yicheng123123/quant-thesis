# v46 reproduction

Working directory: E:/quant-thesis

1. Verify M1 source and recover exact effective codes from the saved native grid:

    .venv-gptq/Scripts/python.exe -B -X utf8 experiments/range_search_w4/results/v46_m1_single_integer_repair_fullmodel/audit_sources.py

2. Verify M1/M2/R2 checkpoints and historical evaluation windows:

    .venv-gptq/Scripts/python.exe -B -X utf8 experiments/range_search_w4/results/v46_m1_single_integer_repair_fullmodel/baseline_audit_v46.py

3. After all source files and output folders are frozen and writable, run the single bounded preflight + formal + save/reload + WT2/C4 sequence:

    .venv-gptq/Scripts/python.exe -B -X utf8 experiments/range_search_w4/results/v46_m1_single_integer_repair_fullmodel/run_v46.py

The runner writes one fresh R1 and evaluates it once per frozen corpus. It refuses a second start when GPU_START.json exists. No shared quantization source or historical checkpoint is modified. The interface cannot confirm the requested Luna Max setting.
