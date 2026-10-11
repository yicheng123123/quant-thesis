# v11 Stage 1 run report

Correctness: **PASS** (10/10 checks passed).
Projection alpha=1 parity: q_proj=PASS, o_proj=PASS, up_proj=PASS, down_proj=PASS.
Rows/splits: 96 rows × 3 frozen splits = 288 records; S00–S02 reused from v9.
Candidates: 2850 newly evaluated, 30 reused from v10, 2880 nominal.
Measured Stage 1 row-loop runtime: 2311.6 s; candidate time sum 1880.9 s; unique Hessian-build time sum 3.7 s; activation capture 2.5 s.
Peak observed process CUDA reserved: 1052.0 MiB.

## Stage 1 row labels

- Candidate-positive: 47
- Mixed-risk: 48
- Neutral/negative: 1

These are descriptive labels within the preregistered layer/projection strata, not model-wide prevalence or stable classifications.

## Projection strata

| Projection | Rows | Candidate-positive | Mixed-risk | Neutral/negative | Median split held gain | Worst split held gain | Selected alpha counts (split-level) |
|---|---:|---:|---:|---:|---:|---:|
| q_proj | 24 | 13 | 11 | 0 | 6.58% | -50.82% | `{"0.6": 4, "0.7": 9, "0.8": 10, "0.9": 26, "1.0": 23}` |
| o_proj | 24 | 13 | 11 | 0 | 18.09% | -53.95% | `{"0.6": 2, "0.7": 15, "0.8": 17, "0.9": 27, "1.0": 11}` |
| up_proj | 24 | 9 | 15 | 0 | 10.79% | -115.19% | `{"0.7": 1, "0.8": 19, "0.9": 44, "1.0": 8}` |
| down_proj | 24 | 12 | 11 | 1 | 7.02% | -91.98% | `{"0.6": 2, "0.7": 3, "0.8": 27, "0.9": 27, "1.0": 13}` |

## Layer strata

| Layer | Rows | Candidate-positive | Mixed-risk | Neutral/negative | Median split held gain | Worst split held gain | Selected alpha counts (split-level) |
|---:|---:|---:|---:|---:|---:|---:|---|
| 0 | 32 | 15 | 17 | 0 | 7.32% | -50.82% | `{"0.7": 1, "0.8": 22, "0.9": 39, "1.0": 34}` |
| 12 | 32 | 15 | 16 | 1 | 5.36% | -49.76% | `{"0.6": 2, "0.7": 4, "0.8": 24, "0.9": 54, "1.0": 12}` |
| 23 | 32 | 17 | 15 | 0 | 14.66% | -115.19% | `{"0.6": 6, "0.7": 23, "0.8": 27, "0.9": 31, "1.0": 9}` |

## Selected alpha and held-out tails

Selected alpha split counts: `{"0.6": 8, "0.7": 28, "0.8": 73, "0.9": 124, "1.0": 55}`.
- held_gain < −5%: 46/288
- held_gain < −10%: 33/288
- held_gain < −20%: 19/288
- held_gain < −40%: 8/288

## Worst 20 held-out regressions

| Layer | Projection | Row | Split | Alpha | Search gain | Held-out gain |
|---:|---|---:|---|---:|---:|---:|
| 23 | up_proj | 3814 | S00 | 0.9 | 29.21% | -115.19% |
| 23 | down_proj | 647 | S00 | 0.7 | 14.39% | -91.98% |
| 23 | up_proj | 3814 | S02 | 0.9 | 16.79% | -61.30% |
| 23 | o_proj | 597 | S02 | 0.7 | 39.71% | -53.95% |
| 0 | q_proj | 700 | S00 | 0.9 | 4.90% | -50.82% |
| 12 | q_proj | 75 | S01 | 0.9 | 13.98% | -49.76% |
| 12 | o_proj | 587 | S01 | 0.9 | 3.49% | -45.94% |
| 23 | up_proj | 711 | S00 | 0.9 | 15.67% | -42.38% |
| 23 | down_proj | 537 | S01 | 0.8 | 22.95% | -38.11% |
| 12 | up_proj | 1605 | S02 | 0.9 | 12.74% | -36.58% |
| 0 | down_proj | 203 | S02 | 0.9 | 7.41% | -35.02% |
| 23 | down_proj | 537 | S02 | 0.9 | 17.36% | -28.75% |
| 12 | down_proj | 170 | S02 | 0.8 | 10.51% | -28.29% |
| 12 | up_proj | 1923 | S01 | 0.8 | 10.06% | -27.81% |
| 12 | q_proj | 448 | S01 | 0.9 | 6.66% | -27.70% |
| 0 | down_proj | 203 | S00 | 0.9 | 0.15% | -27.68% |
| 23 | up_proj | 2896 | S01 | 0.9 | 14.99% | -25.03% |
| 12 | q_proj | 327 | S02 | 0.9 | 10.27% | -24.79% |
| 12 | up_proj | 1923 | S02 | 0.8 | 14.80% | -21.85% |
| 23 | o_proj | 308 | S01 | 0.9 | 9.86% | -19.86% |

## Pooled held-out SSE

D SSE=756.173101; selected C SSE=629.417974; pooled gain=16.76%.
Top-row shares of net reduction: {"1": 0.4426153715435689, "5": 0.7593635340953532, "10": 0.9830942032758327}.
Contribution shares are reported only when net held-out SSE reduction is positive; they can exceed 100% when some rows regress.

## Correctness checks

- `row_manifest_frozen_sha256`: PASS
- `v9_split_manifest_sha256`: PASS
- `S00_S02_exact_article_and_token_reuse`: PASS
- `four_projection_alpha1_native_parity`: PASS
- `pristine_weight_hashes_match_manifest`: PASS
- `candidate_state_and_weight_hash_isolation`: PASS
- `heldout_not_used_for_selection_search_argmin_verified`: PASS
- `existing_v10_row_splits_exactly_reused`: PASS
- `raw_to_csv_independent_gain_and_selection_recompute`: PASS
- `v8_v9_v10_historical_trees_unchanged`: PASS
- `status`: PASS

No Stage 2, PPL, quantized-prefix, or downstream evaluation was run.

## Reproduction commands

Run from the repository root in PowerShell:

```powershell
& '.venv-gptq\Scripts\python.exe' 'experiments/range_search_w4/run_96row_v11.py' --preflight
& '.venv-gptq\Scripts\python.exe' 'experiments/range_search_w4/run_96row_v11.py' --run-stage1
& '.venv-gptq\Scripts\python.exe' 'experiments/range_search_w4/summarize_96row_v11.py'
```
