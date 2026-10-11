# v28 — Full M2 Quadratic Scoring Audit

## Outcome

The direct-score validation completed for all 168 Linear modules. The quadratic scorer did **not** pass the frozen selection-equivalence gate: one output row selects a different alpha. The conditional quadratic-only M2 run and PPL evaluation were therefore not run.

## Frozen scope and formula

- Qwen2.5-0.5B-Instruct revision `7ae557604adf67be50417f59c2c2f167def9a775`, FP16, RTX 4060 Laptop GPU, SDPA, TF32 disabled.
- Same v18 M2: 168 Linear modules, W4A16, symmetric dynamic GPTQ G128, block 128, damping 0.1, original error compensation, 10 alpha values 0.1–1.0, 128 WikiText-2 train documents × 2048 tokens.
- Score target remains the unweighted local Linear output MSE over 262,144 tokens. 72 modules have bias and each stored bias hash remained unchanged; 96 have no bias. Any unchanged common bias cancels in the output difference. The frozen path has no padding mask or sample weighting. Inputs are the actual M2 prefix activations.
- Quadratic scorer: `C = X.T @ X / N` with every entry retained, `e = Q.float() - W.float()`, score `e.T @ C @ e`. C was built from each exact dependency-group input, not GPTQ's damped/dead-column H or inverse.
- Frozen alpha tie handling: strict minimum; exact ties choose the larger alpha. No tolerance or fallback changed the selection.

## Full-model consistency

- Direct v28 path vs v18: input hashes 168/168; H hashes 168/168; direct candidate score arrays exact 168/168; direct selected alpha/W/scale/zero/g_idx exact 168/168; activation boundary hashes exact 119/119.
- Normalized named tensor digest: `dd4803a72696d25593547d4d34429ccc24d37dd28bcd1420863609126611340d`; all 290 named tensors match v18 exactly. `config.json`, model revision and quantization format also match. Existing v18 M2 quality results refer to this same direct-selected model; no new PPL result or quality gain is claimed.
- Dual scorer: 3041280 row-alpha pairs per scorer across all 10 candidates. Score max absolute error `0.00190054287`, max relative error `0.000829214644`, pooled median absolute error `5.45958007e-09`. The diagnostic bound `2 × max absolute score error >= direct best/second gap` flags 201 rows; only 1 row changes selection. This bound is descriptive and never changes selection.
- Whole-model selected-alpha counts (direct vs quadratic) are in `FINAL_SUMMARY.json`; exact ties were 0 and 0 rows, respectively.
- Alpha 0.8/0.9 counts shift by exactly the changed row: direct `54,155` / `226,034`, quadratic `54,154` / `226,035`. Other alpha counts are unchanged.

### Selection-changing row

`model.layers.7.mlp.gate_proj`, output row 884: direct score selects alpha **0.8**, quadratic score selects **0.9**.
- Direct best/second gap: `1.87509791505e-09`; quadratic best/second gap: `2.55139398324e-10`.
- Max absolute score error for this row: `1.56024793752e-07`. Full direct scoring ranks alpha 0.8 ahead of 0.9 by `1.87509791505e-09`; quadratic summation reverses them.
- Selected W differs in 792/896 elements (max absolute FP16 difference `0.0103302`); scale differs; zero and G128 `g_idx` match.
- Same-run direct recheck rescored the retained alpha 0.8 and 0.9 candidate rows. Subset GEMM differed from full-row direct scores by at most `4.03e-12` and `1.39e-12`, respectively, below the full-row direct winner gap; it was not bitwise-identical, so the original full-row direct result remains the reference. The recheck did not alter alpha selection.
- Therefore the numerical formula is mathematically equivalent, but the implemented FP32 Gram/matmul accumulation is not selection-equivalent at this near tie. The pure quadratic scorer cannot replace direct scoring under the frozen exact-choice rule.

The full arrays and all flagged rows are preserved in `audit/raw_scores/layer_XX_scores.npz`, `SENSITIVE_ROWS.csv`, and `SELECTION_DIFFERENCES.csv`; the expanded affected-row values and W/scale/zero/g_idx evidence are in `NUMERICAL_SELECTION_DIFFERENCE.json`.

## Sharing and costs

- 1680 full-Linear alpha GPTQ executions generated both scorers' candidate scores once each; 35 additional alpha=1 native parity references were run. Per scorer, there were 3,041,280 row-alpha slots.
- C was built 96 times, once for each of 96 dependency groups (q/k/v; o; up/gate; down in each layer). All 96 group hashes confirm exact input/H/C sharing only within the group.
- Scoring-only measurement over all rows/candidates: direct scorer `718.524s`; C construction plus quadratic scorer `73.704s` (`C=70.122s`, quadratic scoring `3.582s`). Operator-stage difference is `644.820s` (89.74%). This is a scoring-stage comparison inside the dual-validation run, not a full calibration speedup.
- Candidate GPTQ `1393.721s`; H preparation `96.651s`; activation capture/refresh `154.378s`; alpha=1 parity `31.313s`; checkpoint I/O `6.035s`.
- End-to-end dual-validation wall, including finalization recovery: `2790.652s` (46.51 min). Targeted stage timers sum to `2482.510s`; `308.142s` remains outside those stage counters, so stage sums are not substituted for end-to-end wall.
- Peak recorded GPU memory: `2.807 GiB` allocated, `3.156 GiB` reserved. Budget timer counts `perf_counter` process wall (CPU and GPU), conservatively; cumulative is `2790.652/5400s`.
- Three setup retries stopped before model load and recorded zero GPU time/candidates. There was no separate warm-up stage; model loading, first captures, alpha=1 parity references, and first candidate calls are timed. CPU-only source freeze and post-run analysis are identified separately in `EXECUTION_LOG.json`.

## Gate and unrun items

- Conditional quadratic-only M2: **NOT_RUN** because one row changed the selected alpha and committed W/scale. It was not hidden by a fallback.
- New WikiText-2/C4 PPL: **NOT_RUN**. The saved direct-selected validation model is tensor-identical to v18 M2, so existing v18 PPL can be attributed to that direct path. No PPL is attributed to an unrun quadratic-only model.
- End-to-end calibration acceleration for an exclusively quadratic scorer: **NOT_MEASURED**. No offline scoring-stage saving is described as inference acceleration.
- First full pass reached all 24 layers and 1,680 candidate executions, then final report writing failed because `validation/M2` had not been created. Finalization resumed from the immutable layer checkpoints without rerunning candidates; see `RECOVERY_AUDIT.json`. Three earlier setup attempts failed before model load and recorded zero GPU time/candidates.

## Artifacts

- `PROTOCOL.json` — frozen source/protocol hashes, formula, gates, and scope.
- `validation/FORMAL_TIMING.csv` and `validation/audit/STAGE_TIMING_SUMMARY.csv` — raw and aggregated phase timings.
- `validation/audit/MODULE_CONSISTENCY.csv`, `C_SHARING_AUDIT.csv`, `BOUNDARY_ACTIVATION_COMPARISON.csv` — full module/group evidence.
- `validation/audit/raw_scores/`, `SENSITIVE_ROWS.csv`, `SELECTION_DIFFERENCES.csv`, `NUMERICAL_SELECTION_DIFFERENCE.json` — scores and row-level diagnostics.
- `validation/audit/NORMALIZED_TENSOR_HASHES.json`, `FINAL_SUMMARY.json` — normalized final tensor/config comparison and run summary.
- `run_v28.py`, `analyze_v28.py`, `recovery_finalize_v28.py`, `finalize_v28_report.py` — runner and analysis tools.
