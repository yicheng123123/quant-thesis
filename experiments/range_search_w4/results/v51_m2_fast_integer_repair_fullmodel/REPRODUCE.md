# v51 evidence and reproduction

This run stopped at the frozen v48 initial direct/quadratic guard. Do not resume or bypass the guard. `RUN_STATUS.json` and failure logs remain the original stop records.

## Reproduce the CPU evidence audit

```powershell
cd E:\quant-thesis
& .\.venv\Scripts\python.exe experiments/range_search_w4/results/v51_m2_fast_integer_repair_fullmodel/finalize_incomplete_v51.py
```

## Original run entry

The frozen run entry was:

```powershell
& .\.venv\Scripts\python.exe experiments/range_search_w4/results/v51_m2_fast_integer_repair_fullmodel/run_v51.py
```

The attempt then resumed once after the read-only cost-projection path correction:

```powershell
& .\.venv\Scripts\python.exe experiments/range_search_w4/results/v51_m2_fast_integer_repair_fullmodel/run_v51.py --resume
```

Both calls, the block0 checkpoints, failure attempts, and the no-search raw-X diagnostic are recorded. Re-running the stopped search is not part of this deliverable.

## Evidence

- `PROTOCOL.json`, `PROTOCOL_AMENDMENT_001.json`, `RUNNER_DERIVATION.json`, `RUNNER_AMENDMENT_LOG.json`, `RUNTIME_PREFLIGHT.json`
- `SOURCE_AUDIT.json`, `M2_MODULE_GRID_AUDIT.csv`, `PRE_RESUME_AUDIT.json`
- `RUNTIME_BINDING.json`, `formal/groups/layer_00_*.pt`, `PROPAGATION_RESTORE_CHECKS.csv`
- `diagnostics/INITIAL_QUAD_FAILURE.json`, `diagnostics/INITIAL_QUAD_FAILURE_ROWS.csv`, `diagnostics/INITIAL_QUAD_FAILURE_REPLAY2.log`
- `PARTIAL_STAGE_COSTS.csv`, `COST_LEDGER.csv`, `HISTORICAL_QUALITY_COMPARISON.csv`, `FINAL_DECISION.json`
