# v62 reproduction

The completed v62 output is immutable. The exact successful evaluation command was:

```powershell
Set-Location E:\quant-thesis
.\.venv\Scripts\python.exe -B .\results\v62_m3_vs_repaired_quality_cost\run_v62_eval.py
```

The runner refuses to overwrite a `COMPLETE` result. To reproduce into a fresh independent directory, copy `preflight_v62.py`, `run_v62_eval.py`, and the locked protocol/source audit into a new directory under `results/`, update the two scripts' `OUT` constants only, rerun the CPU preflight, then run the evaluation runner once. Keep source checkpoint paths, evaluator, token manifests, model order and all model/evaluation settings unchanged. Do not reuse any partial windows across process ids. The CPU-only audit command for this completed output is:

```powershell
.\.venv\Scripts\python.exe -B .\results\v62_m3_vs_repaired_quality_cost\finalize_v62.py
```
