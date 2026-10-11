# v60 CPU-only reproduction

Run from E:\quant-thesis in PowerShell:

    $env:CUDA_VISIBLE_DEVICES=''
    & 'E:\quant-thesis\.venv\Scripts\python.exe' 'E:\quant-thesis\experiments\range_search_w4\results\v60_quality_cost_low_budget_preanalysis\reproduce_v60.py'

The script reads the saved v55/v56 group move traces and row summaries on CPU, verifies row/update/evaluation counts and endpoint records, independently recalculates the v59 weighted NLL/PPL table, and writes the derived CSV/JSON artifacts under this directory. It never calls a CUDA API, model forward, quantizer, repair runner, evaluator, or PPL process.

Outputs:

- FOUR_ARM_RESULTS.csv and FOUR_ARM_CONTRASTS.csv
- TRACE_RECOVERY_AUDIT.csv and per-version row update distributions
- BUDGET_TRAJECTORY.csv and layer/projection/input-dimension breakdowns
- LATE_MARGINAL_RETURNS.csv and LOCAL_FULL_PATH_ENDPOINTS.csv
- STAGE_COSTS.csv, CONDITIONAL_COST_SCENARIO.csv, COST_SCOPE_NOTES.json
- VERSION_EVIDENCE.csv, SOURCE_AUDIT.csv, and group checkpoint trace manifests
- CPU_COST.json records the two completed analysis passes; one reproduction invocation overwrites that file with its own actual interpreter timing.

The raw source directories are under experiments/range_search_w4/results/v55_m2_decoder_fixed_integer_repair, v56_m1_fast_integer_repair, and v59_same_process_four_checkpoint_eval. They are read-only inputs to this analysis.
