# v29 — Quadratic Fast Scoring with Original-Score Recheck

## Result

- Full v29 dual-score replay: **PASS** for 168 Linears and 304,128 output rows. This is not an independent hybrid-only run.
- The v28 pure quadratic result remains **FAIL** with its one known selection-changing row. v29 is a separate hybrid validation.
- During this dual-scoring validation, the original direct-score selection was committed to preserve the reference prefix. The separately computed hybrid selection matches it in 168/168 Linears and 304,128/304,128 rows. Full named tensors and config match v18: **True**.
- New PPL: **NOT_RUN**. Existing v18 M2 quality results describe the same normalized named tensors/config only because those hashes match.

## Frozen sensitivity rule and execution unit

- The v28 saved-score rule `2 × max_alpha(abs(quadratic − direct)) >= direct best/second gap` covers row 884 in L7 gate_proj. It flags 201 rows across 82 Linears in this v29 run.
- v18 `score_rows` sends the entire Linear output dimension through each GEMM; it does not have output-row chunks. A single flagged row therefore causes a full-Linear recheck for all 10 candidates.
- Actual rechecks: 820 full-Linear score calls, 258,432 output rows per alpha summed across triggered Linears, and 2,584,320 row-alpha score values. All 820/820 full-shape repeats exactly matched the initial direct score vectors.
- This rule is an empirical numerical guard, not a general error guarantee. It was computed with direct scores available during dual validation; standalone sensitivity prediction without direct scores has not been validated.
- The frozen trigger itself needs direct candidate scores (`abs(quadratic − direct)` and the direct best/second gap). Therefore the dual run validates the mixed result against the reference, but does **not** establish an operational quadratic-only trigger. The current `quadratic` mode cannot evaluate that trigger, and no replacement rule was introduced.

## Costs and cache

- Cumulative v29 budget ledger: **3448.00s (57.47 min)**, including spot/retries and checkpoint recovery. Hard cap: 5400s; remaining: 1952.00s.
- Main validation generated 1,680 full-Linear GPTQ candidates once and evaluated 3,041,280 row-alpha values per scorer. Direct scoring: 740.14s; C construction plus quadratic scoring: 76.09s; sensitivity detection: 0.23s; rechecks: 477.96s.
- Although only 201/304,128 rows (0.0661%) triggered, the scorer's indivisible unit caused rechecks for 82/168 full Linears (48.81%), covering 2,584,320/3,041,280 row-alpha values (84.97%). These 820 calls repeated direct scoring already performed in dual validation; they are not a standalone hybrid benchmark.
- Stage-only accounting gives C build + quadratic score + sensitivity detection + rechecks = 554.27s, compared with 740.14s for direct scoring (arithmetic difference 185.87s). This is not a measured saving: the observed trigger set depends on direct scores and is not available to a quadratic-only run.
- Candidate cache remains GPU resident for one Linear at a time. Peak logical Q/scale/zero cache: 89,886,720 bytes; candidate-cache disk and CPU-transfer bytes: 0. Stage-record peak GPU memory: 2.800 GiB allocated, 3.125 GiB reserved.
- Standalone hybrid-only M2: **NOT_RUN**. The v28-stage-based full-run gate estimate is 2607.63s versus 1952.00s remaining, so it exceeds the remaining budget; this is a budget estimate only, not a measured speedup. In addition, the frozen trigger requires direct scores, so an operational quadratic-only hybrid needs a separately specified trigger before it can run.

## Recovery and limits

- The first full pass completed all candidates and 24 layer checkpoints; finalization then hit a missing v29 output directory. The run resumed from layer 24 with zero candidate reruns. See `RECOVERY_NOTE.json`.
- No new held-out evaluation or PPL was run. No online inference speed claim is made.

## Artifacts

- `PROTOCOL.json`, `IMPLEMENTATION_LOCK.json`, `SPOT_RECHECK.json`
- `MODULE_CONSISTENCY.csv`, `SENSITIVE_RECHECKS.csv`, `HYBRID_SELECTION_DIFFERENCES.csv`
- `RECHECK_DECISIONS.csv` (per triggered Linear: reason, before/after alpha histogram, full recheck scope, and extra wall time)
- `CONSISTENCY_SUMMARY.json`, `COST_SUMMARY.json`, `STAGE_TIMING_SUMMARY.json`, `NORMALIZED_TENSOR_HASHES.json`
- `MIXED_OPERATIONAL_LIMITATION.json`
- `FORMAL_TIMING.csv`, `M2_HYBRID_ROW_RESULTS.csv`, `raw/M2_candidate_losses/layer_XX.npz`
- `M2_HYBRID_ROW_RESULTS_WITH_HEADER.csv`, `M2_HYBRID_MODULE_SUMMARY_WITH_HEADER.csv` (headered exports; the runner's raw hybrid CSVs are retained unchanged)

See `REPRODUCE.md` for commands.
