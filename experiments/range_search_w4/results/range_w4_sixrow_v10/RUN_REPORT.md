# W4 dynamic GPTQ six-row prevalence run v10

## Run

Ran the five remaining preregistered rows across the exact ten article-level splits from v9: 500 fresh full dynamic GPTQ alpha candidates. Imported all 100 L23 R448 alpha records across ten splits from v9 without re-quantization (`source=v9_existing`). No other experiment was run.

Frozen manifest SHA-256: `34a24ef8602b8ce1e38e7a284f112acf7f766962d87d737f69148b78bcfaba78`. Native dynamic alpha=1 spot parity on L0 R0 / S00 passed with exactly equal FP16 weights, scales, zero-points, and search/held-out SSE. All correctness checks: **PASS**.

## Per-row results

| Row | Source | Selected alpha count | Held-out outcome | Median gain | Min | Max | Pooled SSE gain | Diagnostic label |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| L0R0 | v10_run | 0.7:4, 0.8:4, 0.9:2 | 10/10 +, 0/10 tie, 0/10 - | 28.38% | 19.40% | 37.75% | 29.03% | Strong stable clipping row |
| L0R448 | v10_run | 0.8:1, 0.9:9 | 6/10 +, 0/10 tie, 4/10 - | 3.95% | -58.49% | 37.54% | 0.91% | Moderate / unstable |
| L12R0 | v10_run | 0.8:3, 0.9:5, 1.0:2 | 5/10 +, 2/10 tie, 3/10 - | 3.61% | -17.99% | 39.53% | 9.05% | Weak / neutral |
| L12R448 | v10_run | 0.9:8, 1.0:2 | 4/10 +, 2/10 tie, 4/10 - | 0.00% | -43.47% | 34.47% | 2.68% | Weak / neutral |
| L23R0 | v10_run | 0.6:2, 0.7:4, 0.8:4 | 9/10 +, 0/10 tie, 1/10 - | 21.57% | -0.14% | 43.48% | 22.62% | Strong stable clipping row |
| L23R448 | v9_existing | 0.6:4, 0.7:4, 0.8:2 | 10/10 +, 0/10 tie, 0/10 - | 40.33% | 20.20% | 56.45% | 39.63% | Strong stable clipping row |

Across the six row surfaces (60 row-split pairs), the combined held-out SSE gain is 25.13%. It is concentrated: L23 R448 contributes 82.06% of net SSE reduction, and the top two positive rows contribute 93.52%. Removing L23 R448 alone leaves a pooled gain of 9.40%. Row-level gains and D denominators are in `summary.json`.

L0 R448 is the clearest regression case: its search-selected alpha is 0.9 in 9/10 splits, but held-out gain is positive in only 6/10 and includes a -58.49% split. L12 R0 and L12 R448 also show unstable or near-neutral per-split outcomes despite positive pooled SSE gains. These negatives are retained in the raw files.

V8 single-split labels are not uniformly repeated. L0 R0's positive result repeats across 10/10 splits; L0 R448's slight v8 regression becomes a mixed result with large downside cases; L12 R0 and L12 R448 remain near-neutral/unstable across splits; L23 R0 and imported L23 R448 show more consistent positive results. Exact v8 alpha reuse counts are recorded in `summary.json`.

## Cost and data integrity

New candidates: 500; imported from v9: 100; total landscape points: 600. GPU-synchronized formal invocation: 128.68s; candidate evaluations: 126.16s; shared Hessian builds: 0.70s; activation capture: 5.90s. Observed preflight plus formal time: 145.94s. Peak allocated CUDA during capture: 996.5 MiB.

The v9 manifest hash matches the frozen value; all 80 article/token records were reused. Spot parity and the L23 R448 v9 import checks passed. V8 and v9 result trees match their pre-run file hashes, sizes and modification times. Per-row raw JSON, 60-row CSV, 600-point alpha landscape, and correctness checks are included here. No PPL, quantized-prefix, full-layer or expanded-row run was performed.
