# v28 Reproduction

Run from `E:/quant-thesis` in PowerShell with the existing `.venv-gptq` environment and local model/cache present.

```powershell
$env:V28_RUN_LABEL = 'validation'
$env:V28_SCORE_MODE = 'dual'
& .\.venv-gptq\Scripts\python.exe experiments/range_search_w4/results/range_w4_quadratic_full_m2_v28/run_v28.py
& .\.venv-gptq\Scripts\python.exe experiments/range_search_w4/results/range_w4_quadratic_full_m2_v28/analyze_v28.py
& .\.venv-gptq\Scripts\python.exe experiments/range_search_w4/results/range_w4_quadratic_full_m2_v28/finalize_v28_report.py
```

The completed artifacts include a post-run checkpoint finalization recovery. To reproduce that artifact from the already-generated validation checkpoints, use `recovery_finalize_v28.py`; it verifies the recorded historical checkpoint bundle plus every module/layer/alpha-chain hash, performs no GPTQ candidates, saves/reloads the direct-selected model, and preserves the cumulative budget ledger.

The conditional `quadratic_only` execution is intentionally not run because `CONSISTENCY_SUMMARY.json` records one selection-changing row. Do not bypass this gate to claim equivalence. PPL and downstream task evaluation were not run.
