# Gate v1 discovery report — v11 only

Status: **DISCOVERY_PASS**. Independent validation: **NOT RUN**. Pre-frozen grid: 6×6 = 36 pairs; 11 passed A/B/C.

Frozen protocol: `GATE_DISCOVERY_PROTOCOL.json` SHA-256 `04f0523a97ed6aa35db2c1d690fc81060a5ffe7c4a38f89ecbab81b2e42dd7c0`.
Frozen Gate: `GATE_V1.json` SHA-256 `7943b8f26e5f94456b7b00144f6d47fc775d0224ccca4a6a1eff38e5740999c9`. Script SHA-256 `97448c3cd399785ad6e0fee4280e47691d06adfcb81a223084b3ac3aa07d366e`.

## Chosen row-level rule

ACCEPT iff `search_gain_median >= 0.10` AND `search_margin_median >= 0.06`. Otherwise FALLBACK to native dynamic GPTQ alpha=1. Both medians use only the ten search landscapes for that row; ACCEPT uses each split's own search-best alpha. No held-out input enters the Gate.

Accepted 52/96 rows; fallback 44/96. Uniform C severe failures (<−10%): 97/960; Gate: 46/960, reduction 52.58%.
Uniform C pooled gain versus D: 17.77%. Gate: 15.75%. Retained gain: 88.64%.

## Tail and stratum diagnostics

D is all zero gain. C: negative 178, <−5% 132, <−10% 97, <−20% 51, <−40% 22, worst -115.19%.
Gate: negative 91, <−5% 63, <−10% 46, <−20% 24, <−40% 10, worst -115.19%.

Projection breakdown below reports paired pooled gain, severe count, and worst gain; complete projection/layer metrics for every threshold are JSON fields in `gate_grid_results.csv`.

- q_proj: pooled 22.00%, severe 7/240, worst -24.79%.
- o_proj: pooled 16.71%, severe 12/240, worst -53.95%.
- up_proj: pooled 4.88%, severe 18/240, worst -115.19%.
- down_proj: pooled 16.19%, severe 9/240, worst -91.98%.

Layer breakdown:

- L0: pooled 8.92%, severe 5/320, worst -89.40%.
- L12: pooled 8.01%, severe 16/320, worst -46.57%.
- L23: pooled 17.36%, severe 25/320, worst -115.19%.

## Interpretation and limits

The threshold pair maximized retained pooled gain among PASS candidates under the ranking rule frozen before evaluation. This is selection on the same 96-row discovery held-out outcomes; no unbiased effect size, safety guarantee, classifier accuracy, or model-level/PPL improvement follows. Search metrics are known only after the ten candidate GPTQ runs; Gate is not an offline candidate-search speedup. Stratum tails and concentrated SSE contributions remain relevant even if the aggregate PASS criteria hold.

The frozen independent design is in `VALIDATION_PLAN_FROZEN.md`. Its rows/articles have not been quantized or evaluated. Do not alter `GATE_V1.json` after this run; validate its SHA-256 before any subsequent work.
