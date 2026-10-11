# Frozen Gate v1 independent validation report

Final result: **GATE_V1_VALIDATION_FAIL**. Correctness: **PASS**.

1. Gate SHA256 exact match: **True** (`7943b8f26e5f94456b7b00144f6d47fc775d0224ccca4a6a1eff38e5740999c9`).
2. Frozen new rows: **48**, from layers L4/L8/L20 and all four projections; these layers are disjoint from prior v11 discovery layers, with four equal-index buckets per projection/layer.
3. Validation documents: **80** globally unique full articles, 4 search + 4 held-out per split; disjoint from historical v8-v11 article IDs and content hashes. Split hash `8aa736fc321667f086f20c05f998f70d246bbee52fda96fe96e6a58265eb4274`.
4. Four-projection native alpha=1 parity: **PASS** for q_proj, o_proj, up_proj and down_proj.
5. Gate decisions: **25 ACCEPT / 23 FALLBACK** rows.
6. Uniform C pooled held-out SSE gain vs D: **10.9891%**.
7. Gate pooled held-out SSE gain vs D: **2.6172%**.
8. Gate retained C gain: **0.23816627106639646**.
9. C severe failures (held gain < -10%): **69 / 480**.
10. Gate severe failures (held gain < -10%): **40 / 480**.
11. Severe failures reduced by at least 50%: **False** (reduction 42.03%).
12. Gate pooled gain is strictly positive: **True**.
13. Gate retains at least half C gain when applicable: **False**.
14. All applicable primary conditions pass: **False**.
15. Per-projection severe-failure reduction: down_proj: severe failures 12→8 (33.3% reduction); o_proj: severe failures 28→20 (28.6% reduction); q_proj: severe failures 10→6 (40.0% reduction); up_proj: severe failures 19→6 (68.4% reduction). Only up_proj reaches 50%; this is descriptive and does not alter the global Gate.
16. Worst held-out regression: **C L20_o_proj_R287 / S08 -412.09%; Gate L20_o_proj_R287 / S08 -412.09%**.

## Discovery vs validation (descriptive only)

Gate acceptance, search-gain median and search-margin median by row; this comparison does not retune the frozen thresholds.

| Dataset | Rows | Accept/Fallback | Accept rate | Metric | Mean | Median | Q25–Q75 | Min–max |
|---|---:|---:|---:|---|---:|---:|---:|---:|
| discovery_v11 | 96 | 52/44 | 54.2% | search_gain_median | 0.1784 | 0.1745 | 0.0872–0.2692 | 0.0000–0.4764 |
| discovery_v11 | 96 | 52/44 | 54.2% | search_margin_median | 0.1074 | 0.0837 | 0.0620–0.1092 | 0.0274–0.6752 |
| validation_v12 | 48 | 25/23 | 52.1% | search_gain_median | 0.1610 | 0.1449 | 0.1131–0.1965 | 0.0000–0.3763 |
| validation_v12 | 48 | 25/23 | 52.1% | search_margin_median | 0.0757 | 0.0630 | 0.0525–0.0894 | 0.0240–0.2285 |

## D / C / Gate results

| Method | Pooled gain vs D | <0 | <-5% | <-10% | <-20% | <-40% | <-50% | Median row-median gain | Worst |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| D | 0.000% | 0 | 0 | 0 | 0 | 0 | 0 | 0.000% | 0.00% |
| C | 10.989% | 115 | 86 | 69 | 42 | 20 | 17 | 10.632% | -412.09% |
| G | 2.617% | 68 | 51 | 40 | 25 | 10 | 9 | 0.000% | -412.09% |

## Projection breakdown

| Projection | Rows | Accept/Fallback | C pooled | Gate pooled | C severe | Gate severe | Worst C | Worst Gate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| down_proj | 12 | 7/5 | 11.346% | 2.519% | 12 | 8 | -56.22% | -56.22% |
| o_proj | 12 | 9/3 | 5.592% | 4.722% | 28 | 20 | -412.09% | -412.09% |
| q_proj | 12 | 5/7 | 10.301% | 1.538% | 10 | 6 | -137.37% | -28.55% |
| up_proj | 12 | 4/8 | 12.351% | 4.125% | 19 | 6 | -129.11% | -45.97% |

## Layer breakdown

| Layer | Rows | Accept/Fallback | C pooled | Gate pooled | C severe | Gate severe | Worst C | Worst Gate |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| L4 | 16 | 8/8 | 9.391% | 5.684% | 17 | 3 | -164.65% | -30.23% |
| L8 | 16 | 9/7 | 17.041% | 5.588% | 12 | 10 | -27.25% | -27.25% |
| L20 | 16 | 8/8 | 10.373% | 1.069% | 40 | 27 | -412.09% | -412.09% |

## Search-best alpha counts across 480 row-splits

```json
{
  "0.1": 0,
  "0.2": 0,
  "0.3": 0,
  "0.4": 0,
  "0.5": 0,
  "0.6": 1,
  "0.7": 23,
  "0.8": 156,
  "0.9": 247,
  "1.0": 53
}
```

## Pooled gain concentration

- C: top1/top5/top10 row shares = 25.9%/59.2%/79.4%; pooled gains after removing top1/top5 = 9.189%/6.327%.
- G: top1/top5/top10 row shares = 27.1%/83.9%/107.5%; pooled gains after removing top1/top5 = 1.990%/0.484%.
Shares use net SSE reduction as denominator; they can exceed 100% when negative row contributions offset positive reductions.

## Cost

- Candidate slots: 4800/4800; actual candidate attempts including retries 4800; reused from completed cache 0.
- Candidate time sum 2093.3s; Hessian time sum 10.8s; preflight 11.1s.
- End-to-end wall time including preflight and input-cache build 2719.9s.
- Peak CUDA allocated/reserved 585.0/1052.0 MiB.

## Correctness, anomaly and limits

Correctness: 16/16 checks; details in `VALIDATION_CORRECTNESS.json`.
The frozen execute_validation_v12.py --summarize entry point raised AttributeError because its search-only table lacked row metadata; no raw experiment artifact was affected. This companion summarizer independently regenerated all reports from frozen raw JSON and candidate tensors.
## Reproducibility commands

These are the commands used to produce the artifacts; the frozen validation has now failed its preregistered criteria and must not be rerun on the same split.

```powershell
.\.venv-gptq\Scripts\python.exe experiments\range_search_w4\results\range_w4_gate_v12\execute_validation_v12.py --preflight
.\.venv-gptq\Scripts\python.exe experiments\range_search_w4\results\range_w4_gate_v12\execute_validation_v12.py --run
.\.venv-gptq\Scripts\python.exe experiments\range_search_w4\results\range_w4_gate_v12\summarize_validation_v12_postrun.py
```

The 480 row-split outcomes are repeated measurements clustered within 48 rows, three layers and four projections. No PPL, quantized-prefix, downstream task or other model was run. This report ends at the frozen Gate v1 decision; preserve any negative result and await independent Sol review.
