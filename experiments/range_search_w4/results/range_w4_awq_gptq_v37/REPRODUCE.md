# v37 reproduction and artifact map

## Frozen sources

- Protocol: `FROZEN_DIAGNOSTIC_PROTOCOL.json`, SHA-256 `e2b5a6b604b28a187aeb81f40aaf2a1298d73657ae9e446555f9c9b7f27ded0d`.
- Frozen source/checkpoint hashes: `manifests/SOURCE_ARTIFACT_HASHES.json` and `raw/INPUT_HASH_VERIFICATION.json`.
- v36 model/prefix, dataset groups, candidate banks, and selected tensors are read-only sources.
- S0 P1 had already been opened in v36. S1 P2 is used only for search-capture correctness. P3 is not read.

## Commands

Run from `E:\quant-thesis` using the frozen GPTQ environment:

```powershell
& '.venv-gptq/Scripts/python.exe' 'experiments/range_search_w4/results/range_w4_awq_gptq_v37/run_v37_correctness.py'
& '.venv-gptq/Scripts/python.exe' 'experiments/range_search_w4/results/range_w4_awq_gptq_v37/diagnose_s0_p1_norm.py'
& '.venv-gptq/Scripts/python.exe' 'experiments/range_search_w4/results/range_w4_awq_gptq_v37/repair_s0_held.py'
python 'experiments/range_search_w4/results/range_w4_awq_gptq_v37/audit_s0_selection_replay.py'
python 'experiments/range_search_w4/results/range_w4_awq_gptq_v37/finalize_v37.py'
```

The completed v37 output directory already contains the final results. The GPU commands append to the cumulative budget ledger and should only be re-run in a fresh, hash-verified diagnostic copy if independently reproducing. The original runner is restart-protected after finalization. `finalize_v37.py` is CPU-only and packages the recorded evidence; it does not run model or quantization work.

The commands above reproduce the saved-candidate S0 re-evaluation; they do not generate new alpha candidates. No S1 candidate, P3 held, PPL, or full-model quantization command is part of this run.

## Key outputs

- `correctness/v37_correctness_results.json`: final consolidated status.
- `correctness/v37_correctness_results_INITIAL_DIAGNOSTIC.json`: initial v37 checker output, retained to show intermediate false checks and the original bad reference.
- `correctness/S1_P2_CLEAN_RECAPTURE.json`: clean-process P2 capture and alias/state diagnosis.
- `correctness/G00_DOWN_ALPHA1_PARITY_REPAIRED.json`: G00/down alpha=1 direct score evidence.
- `correctness/S0_SELECTION_REPLAY.json`: CPU replay of all frozen selected row alpha values.
- `raw/S0_HELD_REPAIRED_OUTPUTS.npz`: four exact-replayed full-block arm outputs and corrected pristine P1 reference.
- `raw/S0_HELD_REPAIRED_RESULT.json`: repaired full-MLP, fold, projection, and row-tail metrics.
- `raw/S0_HELD_REPAIRED_ARTICLES.csv` and `raw/S0_HELD_REPAIRED_PROJECTION_ROWS.csv`: per-article and per-row data.
- `S0_VALIDITY.json`: final repaired validity and quality gate decision.
- `raw/BUDGET_LEDGER.csv` and `raw/BUDGET_SUMMARY.json`: all accounted stages and failures.
- `FIX_DIFF.patch`: isolated proposed corrections against v36; not applied to v36 or public code.

