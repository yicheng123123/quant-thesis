# v14 — Calibration Coverage Scaling

## Protocol and correctness

Frozen experiment: Qwen/Qwen2.5-0.5B-Instruct, revision `7ae557604adf67be50417f59c2c2f167def9a775`, FP16 model weights, FP32 GPTQ/Hessian/output loss, symmetric W4A16, G128, block 128, damping 0.1, desc_act disabled, shared alpha grid 0.1–1.0. Search inputs are FP-model projection activations; no quantized prefix is used.

Correctness status: **PASS**. Four projection alpha=1 native parity checks passed; 1920 new candidate results completed and independently recomputed. Historical v8–v13 result trees are unchanged.

Every K has its own H_K and D_K. C_K selects alpha only from mean search output MSE over the first K calibration articles, then D_K and C_K are evaluated on the same 32 held-out articles. Candidate weights are separate per alpha and the raw candidate tensors passed their hashes.

## Coverage scaling

| K | positive rows | <0 | <−5% | <−10% | <−20% | <−40% | median row gain | mean row gain | worst row gain | pooled D SSE/MSE | pooled C SSE/MSE | pooled C vs D | pooled D improvement vs D4 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | 35 | 4 | 4 | 1 | 0 | 0 | 12.37% | 13.07% | -12.99% | 485.415 / 0.002468946 | 403.684 / 0.0020532454 | 16.84% | 0.00% |
| 8 | 33 | 9 | 6 | 4 | 3 | 0 | 12.78% | 9.27% | -30.18% | 419.693 / 0.0021346677 | 361.599 / 0.0018391869 | 13.84% | 13.54% |
| 16 | 38 | 3 | 3 | 1 | 0 | 0 | 14.62% | 12.94% | -10.88% | 355.039 / 0.0018058235 | 297.765 / 0.0015145092 | 16.13% | 26.86% |
| 32 | 38 | 7 | 1 | 1 | 1 | 0 | 12.26% | 10.76% | -22.31% | 313.039 / 0.0015921996 | 275.712 / 0.0014023427 | 11.92% | 35.51% |

The row is the unit of analysis: 48 fixed rows per K, 192 row×K observations total. Article-level measurements are diagnostics within each row and are not treated as independent rows.

## Q1–Q5: severe regressions, worst case, pooled and median gain

- K4: severe (<−10%) 1/48; catastrophic (<−40%) 0/48; worst -12.99%; pooled C_K vs D_K gain 16.84%; median row gain 12.37%.
- K8: severe (<−10%) 4/48; catastrophic (<−40%) 0/48; worst -30.18%; pooled C_K vs D_K gain 13.84%; median row gain 12.78%.
- K16: severe (<−10%) 1/48; catastrophic (<−40%) 0/48; worst -10.88%; pooled C_K vs D_K gain 16.13%; median row gain 14.62%.
- K32: severe (<−10%) 1/48; catastrophic (<−40%) 0/48; worst -22.31%; pooled C_K vs D_K gain 11.92%; median row gain 12.26%.

The user-prespecified Coverage Support Gate requires K32 severe cases to be no more than half of K4 (unless K4 has zero, when catastrophic and worst-case values are compared), K32 pooled gain positive, K32 median row gain nonnegative, and both K16 and K32 severe counts below K4. Verdict: **COVERAGE_HYPOTHESIS_NOT_SUPPORTED**. Criteria: `{"A_severe_reduction": false, "B_K32_pooled_gain_positive": true, "C_K32_median_row_gain_nonnegative": true, "D_K16_and_K32_severe_below_K4": false}`.
The conditions came from the v14 task instructions before any candidate run. Their exact source hash is preserved in `coverage_support_gate_provenance.json`; the already-frozen `PROTOCOL.json` omitted this field, so the provenance note documents that serialization gap without editing the frozen protocol.

K4-versus-K32 comparisons separate two effects: D_K itself changes because each K builds its own Hessian, and the row-level clipping gain compares C_K only against that same K's D_K. The `coverage_baseline_D.csv` records per-row D_K changes relative to D4.

## Q6–Q8: alpha stability, landscape agreement, held-out concentration

- Across 48 rows, exact selected-alpha agreement across all four K values occurs for 18/48; median |α*_32−α*_4| is 0.00. α*_32 is closer to 1 than α*_4 for 6/48, and K16/K32 differ by at most one grid step for 47/48.
- Search landscape K4_K8: median Spearman 0.988, best-alpha agreement 28/48, median top-2 Jaccard 1.000.
- Search landscape K8_K16: median Spearman 0.988, best-alpha agreement 31/48, median top-2 Jaccard 1.000.
- Search landscape K16_K32: median Spearman 1.000, best-alpha agreement 34/48, median top-2 Jaccard 1.000.
- K4 held-out C_K article concentration: median top1/top3/top5 SSE share 6.18%/15.83%/24.01%; median worst single-article gain -58.49%.
- K8 held-out C_K article concentration: median top1/top3/top5 SSE share 5.86%/16.04%/24.70%; median worst single-article gain -84.12%.
- K16 held-out C_K article concentration: median top1/top3/top5 SSE share 6.08%/16.19%/24.35%; median worst single-article gain -62.98%.
- K32 held-out C_K article concentration: median top1/top3/top5 SSE share 6.29%/16.17%/24.59%; median worst single-article gain -80.43%.

Article concentration is an outcome diagnostic only. It did not enter alpha selection.

## Q9: calibration cost

Per-K capture cost attributes actual per-article forward time for C01..CK plus the common H01..H32 pool, then prorates the measured model-load and cache-write overhead by document count. Hessian, candidate, and held-out evaluation seconds are measured. The full 64-article capture was one shared pass, so per-K capture totals are attributed costs, not four repeated executions.

| K | activation capture (s) | Hessian (s) | candidate (s) | held-out eval (s) | total calibration (s) | multiplier vs K4 | peak allocated (MiB) | peak reserved (MiB) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | 2.75 | 1.70 | 201.07 | 0.35 | 205.86 | 1.00× | 1789.5 | 1838.0 |
| 8 | 3.01 | 0.68 | 200.75 | 0.35 | 204.80 | 0.99× | 1889.2 | 1950.0 |
| 16 | 3.51 | 0.77 | 199.76 | 0.34 | 204.39 | 0.99× | 1999.5 | 2062.0 |
| 32 | 4.59 | 1.03 | 201.19 | 0.37 | 207.19 | 1.01× | 2129.7 | 2194.0 |

Formal runner wall time including Hessian builds was 16.0 minutes. These are offline calibration/evaluation costs, not inference costs.

## Q10: conclusion and limits

Under the preregistered diagnostic support gate, this run is **COVERAGE_HYPOTHESIS_NOT_SUPPORTED**. Read the full K4→K8→K16→K32 sequence: severe/catastrophic counts, worst row, pooled gain, median gain, D_K baseline changes, and cost move independently. No claim of PPL or final-model improvement follows; no PPL, Gate, robust objective, predictor, or new model/bitwidth was run.

Projection- and layer-level counts are in `projection_summary.csv` and `layer_summary.csv`; all negative row×K cases are in `regression_cases.csv`. Raw JSON and candidate tensors preserve all 1,920 candidate records and per-article held-out metrics.

## Reproduction

```powershell
& '.venv-gptq\Scripts\python.exe' 'experiments/range_search_w4/results/range_w4_calibration_coverage_v14/prepare_v14.py'
& '.venv-gptq\Scripts\python.exe' 'experiments/range_search_w4/results/range_w4_calibration_coverage_v14/run_v14.py'
& '.venv-gptq\Scripts\python.exe' 'experiments/range_search_w4/results/range_w4_calibration_coverage_v14/summarize_v14.py'
```
