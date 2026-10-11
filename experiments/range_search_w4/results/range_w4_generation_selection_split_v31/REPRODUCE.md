# v31 reproduction

Run from repository root `E:/quant-thesis` in the existing `.venv-gptq` environment. This protocol performs no model forward and no GPTQ candidate generation.

```powershell
& '.venv-gptq/Scripts/python.exe' 'experiments/range_search_w4/results/range_w4_generation_selection_split_v31/audit_v31.py'
& '.venv-gptq/Scripts/python.exe' 'experiments/range_search_w4/results/range_w4_generation_selection_split_v31/run_cross_group_scores_v31.py'
& '.venv-gptq/Scripts/python.exe' 'experiments/range_search_w4/results/range_w4_generation_selection_split_v31/analyze_v31.py'
```

The score runner is resumable from per-candidate JSON checkpoints and enforces the frozen 600-second cumulative GPU wall budget. It reuses v24 candidate tensors and v23/v24 activation caches. Do not rerun `audit_v31.py` after scoring unless deliberately regenerating its frozen audit outputs. The score runner refuses any protocol status other than `FROZEN_BEFORE_CROSS_GROUP_SCORING`.

Key raw artifacts: `CROSS_GROUP_ROW_ALPHA_SCORES.csv` (all 1,920 candidate-row-alpha-group values, saved and new), `ROW_EVALUATION_COMPARISONS.csv`, `SELECTIONS_BY_ROW.csv`, and `raw/cross_group_scores/`.
