# v24 E2 — Structured 2-Region Group-Alpha Diagnostic

**Verdict: `E2_STRUCTURED_CAPACITY_SIGNAL_NOT_SUPPORTED`**

## Frozen protocol

- Qwen2.5-0.5B-Instruct, revision `7ae557604adf67be50417f59c2c2f167def9a775`, FP16, SDPA, RTX 4060 Laptop GPU; TF32 off.
- Symmetric W4A16 G128; native dynamic GPTQ, block 128, damping 0.1, `desc_act=False`, `static_groups=False`; alpha 1 is the native no-hook point.
- Layers 0/12/23, `up_proj` and `down_proj`, rows 0/224/448/672; 48 row/split units.
- Reused v22/v23's 64 complete WT2 train articles and frozen S0/S1 search/held split without resampling. The v23 prefix cache is read-only; down inputs are captured by replaying each target v18 M2 block once per document.
- Search objective is local projection output MSE. Held inputs are scored only after the selection lock. No block loss, PPL, C4, second model, AWQ scaling, or model-level experiment was run.

## Selection and execution budget

- Search-score positions: original scalar 10-point = 480; dense scalar 46-point = 2208; structured candidate set = 1409 unique positions (maximum 2,208). The two-sweep schedule had 1728 slots, with 799 exact tuple reuses.
- GPTQ candidate executions: 552 full-Linear runs generated the 46-point shared-alpha bank; 929 fresh per-row GPTQ executions evaluated unequal region tuples. These are separate from score-position counts.
- Held landscapes: 2208 dense scalar row scores and 1409 structured tuple scores, scored after selection lock; held-oracle choices were retrospective only.

## Held local reconstruction

- Structured vs original scalar: W/T/L 15/23/10, pooled improvement 0.128651%, median row gain 0.000000%, mean row gain 0.528350%.
- Structured vs dense scalar: W/T/L 17/11/20, pooled improvement -1.273512%, median row gain 0.000000%, mean row gain -1.012115%.
- Native α=1 held losses are the reference. Severe tail counts (<−10% vs native): M2 0, dense 0, structured 0; structured worst gain -0.454137%.
- Structured selected different A/B alphas in 25/48 units (52.08%); same-alpha rate 47.92%. Mean |αA−αB| 0.0521.
- E2 v22 gate: both splits improve over original and dense beyond epsilon = False; ≥4/6 targets improve over both = False; no new <−10% native tail = True; structured explored-set oracle advantage over dense oracle in both splits = False; αA/αB differ in a majority of units = True. Overall gate: False.

## Region assignment and interpretation

Region assignment follows the concrete v22 group-index split: 7 groups use A=0–2/B=3–6, and 38 groups use A=0–18/B=19–37. The v22 plan already froze this assignment, so no M3- or held-derived statistic/threshold was used. Weight range, native α=1 group error, and search-H diagonal summaries are descriptive only; see `REGION_DIAGNOSTICS.csv`.

Held scalar and structured oracles are upper bounds over the finite candidate sets actually scored. The structured oracle is not exhaustive over all 100 α-pairs. Similar structured and dense oracles would indicate little observed capacity gap under these budgets; an oracle gap alone does not show that search selection can generalize.

## Cost and correctness

- Final recovery-process wall: 31.60s; prior failed formal-process wall: 2513.02s; aggregate formal-process wall: 2544.62s; total including preflight: 2692.44s (0.7479h). Peak CUDA allocated/reserved: 2.176/2.381 GiB.
- Correctness: PASS; row/shared identity used v17.3 tolerance (weight/zero/g_idx/local loss exact, scale `rtol=1e-6, atol=1e-9`); held leakage audit: PASS.
- Cost components, synchronized GPTQ/scoring time, cache work, and checkpoint I/O are in `COST_ANALYSIS.csv` and `RUN_REPORT.json`.
- Execution history: six resumable process failures were preserved: two candidate path/lifetime bugs and four report-wiring errors. All unique GPTQ and held-score sidecars were reused for recovery; the aggregate wall above includes failed-attempt time. No held result changed during recovery.

## Verdict and next step

The frozen two-region search did not establish a stable capacity signal beyond the dense scalar control and the held-oracle comparison. Stop the structured-capacity line here; do not add more regions or tune the assignment.

No full-model, PPL, C4, AWQ-style scaling, proxy search, additional regions, or second-model work was started. Historical v8–v23 artifacts were read-only inputs.

## Reproduction

```powershell
.\.venv-gptq\Scripts\python.exe -B experiments\range_search_w4\results\range_w4_structured_group_e2_v24\prepare_e2_v24.py
.\.venv-gptq\Scripts\python.exe -B experiments\range_search_w4\results\range_w4_structured_group_e2_v24\run_e2_v24.py preflight
.\.venv-gptq\Scripts\python.exe -B experiments\range_search_w4\results\range_w4_structured_group_e2_v24\run_e2_v24.py run
```

The first command verifies and reuses the frozen document manifest; rerunning a completed stage validates existing hashes/checkpoints instead of resampling or overwriting them.
