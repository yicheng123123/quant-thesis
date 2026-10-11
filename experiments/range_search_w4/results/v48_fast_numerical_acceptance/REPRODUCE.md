# Reproduction

```powershell
Set-Location E:/quant-thesis
& ./.venv-gptq/Scripts/python.exe -B -X utf8 experiments/range_search_w4/results/v48_fast_numerical_acceptance/run_v48.py
& ./.venv-gptq/Scripts/python.exe -B -X utf8 experiments/range_search_w4/results/v48_fast_numerical_acceptance/additional_cpu_validation.py
& ./.venv-gptq/Scripts/python.exe -B -X utf8 experiments/range_search_w4/results/v48_fast_numerical_acceptance/finalize_v48.py
```

The first command refuses an existing GPU_START.json to prevent overwriting evidence/resetting the1200s budget. To reproduce, copy run_v48.py, fast_accept.py, additional_cpu_validation.py, finalize_v48.py and ACCEPTANCE_RULES.md into a NEW independently named result directory under the same results parent; run those copied scripts. Leave v48 and historical evidence untouched. GPU budget includes diagnostics, not just timed matrices. Exact frozen v46 runtime/repair/precision helpers are imported read-only; Python -B suppresses historical pycache writes. No network, held, PPL, commit or deployable model save. The second command checks additional settlement branches on CPU only (it does not feed back into selection or alter rules); the third is CPU-only summary regeneration. Frozen source/manifest hashes are SOURCE_AUDIT.json and GPU_START.json.
