# v54 artifact reproduction

This directory contains a budget-stopped D run. The existing frozen runner has already committed and validated groups through block1. No complete model or new PPL result exists.

To regenerate the CPU-only audit summaries, checkpoint hashes, quality provenance table, and budget accounting from the preserved files:

```powershell
& .\.venv\Scripts\python.exe -B .\experiments\range_search_w4\results\v54_m2_decoder_fixed_integer_repair\finalize_v54_artifacts.py
```

The formal entry point and frozen hashes are `run_v54.py` and `FROZEN_RUN_LOCK.json`. Do not launch it under the already-recorded cap expecting completion: `ONLINE_BUDGET_STATUS.json` records that the conservative remaining-work projection exceeds available budget while preserving the 360-second closeout reserve. Any further GPU execution requires a separately authorized budget decision. The current partial groups are not a deployable model and must not be evaluated as one.

The run requested Luna Max, but the interface cannot verify the active model selection. No P3 data was read.
