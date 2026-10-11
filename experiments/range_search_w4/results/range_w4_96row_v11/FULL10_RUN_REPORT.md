# v11 full 10-split W4 dynamic GPTQ range search run report

## Scope and protocol

The landscape covers all 96 preregistered rows across S00–S09. Labels and prevalence describe only the preregistered layer/projection strata; they are not an unbiased model-wide prevalence estimate.
Model: `Qwen/Qwen2.5-0.5B-Instruct` revision `7ae557604adf67be50417f59c2c2f167def9a775`; FP16 weights, FP32 GPTQ/loss; symmetric W4A16 G128; block 128; damping 0.1; `desc_act=False`; FP model projection inputs.
D is native dynamic GPTQ at alpha=1.0. C selects among ten shared-alpha dynamic GPTQ candidates by search-split output MSE. Held-out data are not used for selection.
Frozen row manifest SHA-256 `c56eeaa1d1af6e4ed8dd78f811c811c66758b215659a98bf8550052c786e6106`; v9 split manifest SHA-256 `34a24ef8602b8ce1e38e7a284f112acf7f766962d87d737f69148b78bcfaba78`.

## Correctness

Status: **PASS**.
- `row_manifest_sha256_frozen`: PASS
- `v9_manifest_sha256_frozen_and_all_splits_match`: PASS
- `96_rows_by_10_splits_complete`: PASS
- `9600_alpha_candidates_complete`: PASS
- `S00_S02_and_stage1_outputs_unchanged`: PASS
- `all_frozen_search_heldout_splits_disjoint`: PASS
- `pristine_weight_hashes_match_manifest`: PASS
- `all_search_and_heldout_activation_hashes_match`: PASS
- `candidate_state_isolation_and_saved_weight_hashes`: PASS
- `search_only_argmin_selection_and_raw_losses_match`: PASS
- `stage2_alpha1_native_parity_four_projections`: PASS
- `v10_legacy_candidates_exactly_reused`: PASS
- `raw_to_csv_independent_recomputation`: PASS
- `v8_v9_v10_historical_trees_unchanged`: PASS
- `stage1_preservation_fingerprint_pass`: PASS

## Benefit stability

- Robust-positive: 26/96
- Mixed-risk: 61/96
- Neutral: 5/96
- Unclassified by the frozen label rules: 4/96.
Unclassified rows remain separate because the frozen positive/mixed/neutral definitions do not exhaust every possible gain pattern; no outcome-dependent threshold was added.

### Projection strata

| Projection | Rows | Positive | Mixed-risk | Neutral | Unclassified | Median row median gain | Worst row gain | Selected alpha counts |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| q_proj | 24 | 10 | 8 | 5 | 1 | 8.54% | -50.82% | `{"0.6": 13, "0.7": 25, "0.8": 37, "0.9": 83, "1.0": 82}` |
| o_proj | 24 | 2 | 21 | 0 | 1 | 19.25% | -59.92% | `{"0.6": 5, "0.7": 48, "0.8": 66, "0.9": 86, "1.0": 35}` |
| up_proj | 24 | 6 | 18 | 0 | 0 | 10.12% | -115.19% | `{"0.7": 1, "0.8": 80, "0.9": 132, "1.0": 27}` |
| down_proj | 24 | 8 | 14 | 0 | 2 | 9.84% | -91.98% | `{"0.6": 2, "0.7": 22, "0.8": 82, "0.9": 102, "1.0": 32}` |

### Layer strata

| Layer | Rows | Positive | Mixed-risk | Neutral | Unclassified | Median row median gain | Worst row gain | Selected alpha counts |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 0 | 32 | 9 | 19 | 3 | 1 | 7.21% | -89.40% | `{"0.7": 2, "0.8": 72, "0.9": 142, "1.0": 104}` |
| 12 | 32 | 7 | 21 | 2 | 2 | 10.67% | -49.76% | `{"0.6": 5, "0.7": 18, "0.8": 87, "0.9": 168, "1.0": 42}` |
| 23 | 32 | 10 | 21 | 0 | 1 | 12.22% | -115.19% | `{"0.6": 15, "0.7": 76, "0.8": 106, "0.9": 93, "1.0": 30}` |

## Stage 1 screening diagnostic

| Stage 1 label | Rows | Final robust-positive | Final mixed-risk | Final neutral | Unclassified | Robust-positive fraction |
|---|---:|---:|---:|---:|---:|---:|
| Stage1 candidate-positive | 47 | 25 | 19 | 0 | 3 | 53.19% |
| Stage1 mixed-risk | 48 | 1 | 41 | 5 | 1 | 2.08% |
| Stage1 neutral/negative | 1 | 0 | 1 | 0 | 0 | 0.00% |

## Held-out tail risk

Counts over all 960 row/split pairs: gain_lt_-5%: 132, gain_lt_-10%: 97, gain_lt_-20%: 51, gain_lt_-40%: 22, gain_lt_-50%: 10. Worst gain: -115.19%.

### Worst 30 cases

| Layer | Projection | Row | Split | Selected alpha | Held-out gain |
|---:|---|---:|---|---:|---:|
| 23 | up_proj | 3814 | S00 | 0.9 | -115.19% |
| 23 | down_proj | 647 | S00 | 0.7 | -91.98% |
| 0 | down_proj | 700 | S08 | 0.9 | -89.40% |
| 23 | up_proj | 1605 | S06 | 0.8 | -66.11% |
| 23 | up_proj | 3814 | S02 | 0.9 | -61.30% |
| 23 | down_proj | 45 | S04 | 0.9 | -60.78% |
| 0 | down_proj | 203 | S07 | 0.9 | -60.30% |
| 23 | o_proj | 546 | S07 | 0.7 | -59.92% |
| 23 | o_proj | 597 | S02 | 0.7 | -53.95% |
| 0 | q_proj | 700 | S00 | 0.9 | -50.82% |
| 12 | q_proj | 75 | S01 | 0.9 | -49.76% |
| 23 | o_proj | 546 | S05 | 0.7 | -48.34% |
| 12 | up_proj | 104 | S08 | 0.8 | -46.57% |
| 23 | up_proj | 1605 | S05 | 0.8 | -46.56% |
| 23 | o_proj | 308 | S08 | 0.8 | -45.97% |
| 12 | o_proj | 587 | S01 | 0.9 | -45.94% |
| 0 | o_proj | 517 | S04 | 0.9 | -45.44% |
| 23 | up_proj | 3413 | S06 | 0.9 | -44.24% |
| 23 | down_proj | 370 | S07 | 0.8 | -44.16% |
| 12 | q_proj | 448 | S07 | 0.9 | -43.47% |
| 23 | up_proj | 711 | S00 | 0.9 | -42.38% |
| 23 | o_proj | 597 | S04 | 0.8 | -41.06% |
| 23 | down_proj | 537 | S01 | 0.8 | -38.11% |
| 12 | up_proj | 1605 | S02 | 0.9 | -36.58% |
| 23 | down_proj | 537 | S09 | 0.8 | -36.12% |
| 0 | o_proj | 838 | S06 | 0.9 | -35.44% |
| 0 | down_proj | 203 | S02 | 0.9 | -35.02% |
| 23 | up_proj | 238 | S07 | 0.9 | -34.16% |
| 0 | o_proj | 94 | S05 | 0.9 | -33.80% |
| 12 | o_proj | 371 | S04 | 0.9 | -30.98% |

## Pooled SSE and concentration

D SSE `2504.99727`, search-selected C SSE `2059.80407`, pooled held-out gain 17.77%.

| Top rows | SSE reduction | Share of net reduction | Share of D SSE |
|---:|---:|---:|---:|
| 1 | 157.645523 | 35.41% | 6.29% |
| 5 | 278.570996 | 62.57% | 11.12% |
| 10 | 351.448886 | 78.94% | 14.03% |
| 20 | 420.229977 | 94.39% | 16.78% |
- Removing top1: pooled gain 13.67%; removed rows `L23_q_proj_R106`.
- Removing top5: pooled gain 9.61%; removed rows `L23_down_proj_R309, L23_down_proj_R754, L23_down_proj_R793, L23_q_proj_R106, L23_q_proj_R303`.

## Fixed-alpha descriptive comparison

These values are descriptive only; no universal alpha is selected from held-out outcomes.

| Alpha | Positive / regression / zero splits | Median gain | Pooled gain | Worst regression |
|---:|---:|---:|---:|---:|
| 0.7 | 287 / 673 / 0 | -27.39% | -70.53% | -1644.84% |
| 0.8 | 556 / 404 / 0 | 6.86% | -12.68% | -915.50% |
| 0.9 | 669 / 291 / 0 | 12.49% | 6.63% | -279.63% |

## Anomalies and limitations

- Stage 2 loop measured 10605.8s versus the preregistered planning estimate with margin 6583.1s (1.61x).
- Model input capture emitted the Transformers warning: Sliding Window Attention is enabled but not implemented for sdpa; no alternate attention-backend comparison was run.
- 4 rows fall in a gap between the frozen robust-positive, mixed-risk and neutral rules; they are reported separately without post-hoc reassignment.
- Largest selected-C held-out regression: L23_up_proj_R3814 / S00 = -1.151905.
- No full-model quantization, PPL, quantized-prefix or downstream evaluation was run in this stage.
- Frozen-rule gap rows: L0_down_proj_R517, L12_down_proj_R496, L12_o_proj_R827, L23_q_proj_R696.

## Runtime and cost

Stage 2 invocation wall time: 10605.8 s (2.95 h); summed candidate timer: 8700.6 s; unique Hessian construction: 6.66 s; activation capture: 8.22 s.
Preflight estimate with its 1.5× margin was 6583.1 s; measured runner loop was 10605.8 s (1.61× that estimate). The estimate underpredicted observed loop time; measured time is reported as the result.
New candidates: 6650; exact v10 reused: 70; maximum candidate CUDA reserved: 246.0 MiB; maximum across activation capture, native parity, and candidate evaluation: 1052.0 MiB.
Stage 2 saved activation cache size: 649,803,998 bytes. GPU timing synchronized: `True`.
The earlier estimate was a planning estimate. This report uses measured runtime and does not claim inference-speed improvement.

## Alpha stability

Per-row alpha histograms, mode/frequency, mean/median/min/max, and best three-grid-point band are in `full10_row_summary.csv`; alpha-band stability does not affect benefit labels.

## Artifacts and reproduction

All new experiment outputs are under this result directory. No PPL, full-model quantization, quantized-prefix, downstream task, classifier, or predictor run was performed.
Reproduction commands from the repository root (PowerShell):

```powershell
& '.venv-gptq\Scripts\python.exe' 'experiments/range_search_w4/run_full10_stage2_v11.py' --preflight
& '.venv-gptq\Scripts\python.exe' 'experiments/range_search_w4/run_full10_stage2_v11.py' --run-stage2
& '.venv-gptq\Scripts\python.exe' 'experiments/range_search_w4/summarize_full10_v11.py'
```
