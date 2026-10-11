v52 frozen reproduction commands (PowerShell, E:/quant-thesis)

CPU local checks:
    .\.venv\Scripts\python.exe experiments/range_search_w4/results/v52_m2_fast_integer_repair_fp64_initial_review/validate_cpu.py

Original GPU regression entry (already run; do not repeat to select results):
    .\.venv\Scripts\python.exe experiments/range_search_w4/results/v52_m2_fast_integer_repair_fp64_initial_review/validate_row122.py

Historical frozen formal entry (currently blocked by confirmed prefix incompatibility; do not execute):
    .\.venv\Scripts\python.exe experiments/range_search_w4/results/v52_m2_fast_integer_repair_fp64_initial_review/run_v52.py

The following historical recovery command is NOT valid for the current layer1+ checkpoints; retained for source reproduction only:
    .\.venv\Scripts\python.exe experiments/range_search_w4/results/v52_m2_fast_integer_repair_fp64_initial_review/run_v52.py --resume

Confirmed D stop is immutable. Do not use the diagnostic final checkpoint for inference-quality claims. No more GPU calls without explicit handling of the prefix incompatibility. Do not recreate protocol, stage inherited groups again, reset active-job ledger, or rerun PPL. Check frozen lock and inherited source hashes first.

CPU final evidence aggregation:
    .\.venv\Scripts\python.exe experiments/range_search_w4/results/v52_m2_fast_integer_repair_fp64_initial_review/finalize_v52.py

Prefix source diagnostic already executed: diagnose_prefix.py (do not repeat for result selection). Deployment adapters: finish_v52.py; their failures are preserved and no completed evaluation exists.

Active GPU budget starts in budget.start() for each GPU process. Its durable segments carry prior657.305s and all new work; idle applications and CPU tool diagnosis are excluded. See ACTIVE_JOB_LEDGER.json.
Final D report and manifest generation (CPU only, after aggregation):
    .\.venv\Scripts\python.exe experiments/range_search_w4/results/v52_m2_fast_integer_repair_fp64_initial_review/final_delivery.py

Do not rerun prepare_finish.py: it is the retained historical adapter builder, not an approved resume path. The executed attempt3 source is finish_v52.py with DEPLOYMENT_ADAPTER_LOCK.json; prior attempt1/2 sources and failures remain in diagnostics. No further GPU or PPL calls are authorized by this stopped result. PREFIX_COMPATIBILITY_DIAGNOSIS.json proves the layer1+ propagation source mismatch.
