# v47 reproduction

## CPU-only aggregation

Run from `E:\quant-thesis` with the environment that can import PyTorch:

```powershell
python experiments/range_search_w4/results/v47_integer_repair_cost_audit/analyze_v47.py
python experiments/range_search_w4/results/v47_integer_repair_cost_audit/finalize_v47.py
```

The analyzer reads committed v46 result checkpoints and timing CSVs. It does not load a model or use CUDA. It verifies 96 committed groups and reconciles row/update/evaluation totals.

## Bounded full-matrix GPU profile

The already completed profile command was:

```powershell
.venv-gptq/Scripts/python.exe -B -X utf8 experiments/range_search_w4/results/v47_integer_repair_cost_audit/profile_v46_l0_matrices.py
```

This reconstructs the exact L0 up_proj/down_proj inputs from locked R1, verifies v46 input and C hashes, and replays full rows under a 600-second hard cap. It adds only a synchronized timer around p update. It writes timing/status diagnostics in v47, does not commit weights, save a quantized checkpoint, score held data, or run PPL. Re-running it incurs a new GPU profiling run; it is not needed to read the saved results.

The v47 p timer is nested inside `accepted_update_and_rejection_restore`; never add parent and child timer totals. `profile_input_hash` is a no-op timer in this recorded profile; actual hashes were verified and included in total process wall, while full-model hash cost is measured from v46's `input_hash` timer.
