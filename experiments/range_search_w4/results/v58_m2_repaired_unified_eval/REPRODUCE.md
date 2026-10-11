# Reproduce v58

PowerShell, from `E:\quant-thesis`:

```powershell
.venv\Scripts\python.exe experiments\range_search_w4\results\v58_m2_repaired_unified_eval\run_v58_eval.py
```

The runner verifies the v55 lock and frozen manifests, uses the unchanged `benchmark.eval.ppl.evaluate_ppl`, and atomically journals each complete window. On resume it validates the same run lock, token slice and label mask before skipping a completed window. It does not read P3 or modify any checkpoint. `finalize_v58.py` performs CPU-only CSV aggregation and report generation.
