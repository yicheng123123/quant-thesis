# Gate v1 independent validation protocol — FROZEN, NOT EXECUTED

Gate identity: SHA-256 `GATE_V1.json` = `7943b8f26e5f94456b7b00144f6d47fc775d0224ccca4a6a1eff38e5740999c9`. Use exactly tau_gain=0.10 and tau_margin=0.06; no retuning.

## Scope and row manifest rule

Use the same Qwen2.5-0.5B-Instruct checkpoint/revision and the same symmetric W4A16 dynamic GPTQ G128, block 128, damping 0.1, desc_act=False, FP-model input protocol as v11. New layers only: L4, L8, L20. Projections in fixed order q_proj, o_proj, up_proj, down_proj. Four rows per layer/projection, 48 total.

Split each output dimension into four index buckets `[floor(k*d/4), floor((k+1)*d/4))` for k=0..3. Use one `random.Random(20261001)` instance, iterate layers (4,8,20), projections (q,o,up,down), then buckets (low,low-mid,high-mid,high), drawing `randrange(start, stop)` once per bucket. The following IDs are fixed by that rule; none were selected by outcome:

```text
L4 q_proj: 60, 306, 540, 887
L4 o_proj: 135, 372, 631, 833
L4 up_proj: 802, 1470, 2739, 4033
L4 down_proj: 194, 411, 467, 845
L8 q_proj: 34, 233, 495, 790
L8 o_proj: 97, 278, 535, 721
L8 up_proj: 1156, 1764, 2952, 4448
L8 down_proj: 206, 418, 500, 752
L20 q_proj: 6, 444, 507, 878
L20 o_proj: 140, 287, 639, 739
L20 up_proj: 982, 2142, 2811, 4385
L20 down_proj: 99, 352, 626, 881
```

## Independent articles and split freezing

Use only `Salesforce/wikitext:wikitext-2-raw-v1:train`, the existing v7/v9 full-article parser and the same checkpoint tokenizer. Enumerate eligible complete train articles in original source order; retain those with at least 128 tokens. Exclude every old v8–v11 source article index AND article/token hash. From the remaining articles, take the first 80 with distinct source index, article hash, and first-128-token hash. If fewer than 80 exist, stop and document the blocker; do not substitute WikiText test or fragments.

Shuffle this fixed 80-article list once with `random.Random(20261001)`; partition sequentially into ten groups of eight; in each group the first four whole articles are search and the last four held-out. Use the first 128 tokens of each article. Before any GPTQ run, save a manifest with source indices, hashes, token IDs, seed, exact parser/tokenizer identities and split membership, then hash that manifest. Require global uniqueness of all 80 articles/tokens and no overlap with v8–v11. No search or held-out article may be selected after seeing any quantization outcome.

## Three paired methods and Gate inputs

For every new row and every frozen split, evaluate the same ten shared-alpha dynamic GPTQ candidates (0.1–1.0). Each candidate starts from pristine W and GPTQ initial state; alpha=1 receives native parity spot checks in each projection before formal work. D uses alpha=1, C uses the search-output-MSE-minimizing alpha for that split. Gate computes each row's ten search gains and ten top1/top2 search margins from search SSE only, takes the two medians, then accepts **all ten C split choices** iff both frozen thresholds pass; otherwise all ten splits use D. The Gate reads no validation held-out value, projection-specific rule, layer-specific rule, or third feature. Gate is a selection/safety experiment, not an offline-search acceleration.

## Frozen validation decision

A: Gate `held_gain < -0.10` count is at most half of C's count (`2*G_severe <= C_severe`). B: Gate pooled SSE gain versus D is strictly >0. C: if C pooled gain is >0, Gate retains at least 50% of it (`G_gain/C_gain >= 0.5`); if C gain is <=0, C is inapplicable and must be reported as such. Validation PASS requires all applicable conditions. Thresholds, rows, articles, splits, candidate grid, Gate structure and PASS conditions cannot change after seeing validation held-out values. If FAIL, retain the negative result and do not tune on these data.

## Required secondary reporting and cost gate

Report each method's pooled SSE gain, per-row median gain, counts for held_gain <0, <−5%, <−10%, <−20%, <−40%, worst gain, accepted/fallback rows, projection and layer breakdown, and top1/top5/top10 shares of net SSE reduction. Report exact failed and degenerate cases. Treat the 480 row/split values as repeated measurements, not 480 independent model samples; show row and layer/projection clustering. Record candidate evaluations, synchronized total wall time including precomputation/cache build, and peak allocated/reserved GPU memory. Do a small correctness/cost preflight before formal execution. Expected candidate slots: `48 rows × 10 splits × 10 alpha = 4,800`; this is a slot count, not a runtime guarantee.

No PPL, quantized-prefix, other model, extra bitwidth, predictor training, or threshold adaptation belongs to this validation.
