# v53 reproduction (PowerShell at E:/quant-thesis)

All current validations have completed. No selection/rerun is performed. Historical files remain read-only; all new output remains inside this directory.

CPU deterministic tests (21 cases; no CUDA):
    .\.venv\Scripts\python.exe -B experiments/range_search_w4/results/v53_decoder_interface_propagation_audit/validate_cpu.py

Executed single bounded native propagation validator (already passed; do not silently reset/rerun budget):
    .\.venv\Scripts\python.exe -B experiments/range_search_w4/results/v53_decoder_interface_propagation_audit/validate_propagation.py
It refuses a second completed run. budget_v53.py carries the immutable v52 ledger into the new local ledger; newGPU cap600, sharedcap10800. Native reference hooks do not use decoder_interface.py. Calibrated prefix uses originalM2 plus four verified block0 states only.

CPU history audit sequence (does not execute historical models):
    .\.venv\Scripts\python.exe -B experiments/range_search_w4/results/v53_decoder_interface_propagation_audit/audit_history.py
    .\.venv\Scripts\python.exe -B experiments/range_search_w4/results/v53_decoder_interface_propagation_audit/expand_callsite_impact.py
    .\.venv\Scripts\python.exe -B experiments/range_search_w4/results/v53_decoder_interface_propagation_audit/complete_norm_alias_inventory.py

CPU final delivery/source verification:
    .\.venv\Scripts\python.exe -B experiments/range_search_w4/results/v53_decoder_interface_propagation_audit/finalize_v53.py

No full-model repair or PPL command is supplied/started. Future authorized driver must load the corrected local runtime as run_v42, then unchanged v52 fast core/initialization review, retain full terminal checks/provenance, restore only the listedblock0 groups, replay its own fulltoken prefix, and start originalM2 layer1/qkv. See VALID_RESUME_BOUNDARY.json and LEGACY_CALLSITE_HANDLING.md. Do not execute archived legacy drivers unadapted under5.x.
