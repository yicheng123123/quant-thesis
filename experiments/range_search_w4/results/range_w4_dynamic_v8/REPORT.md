# Dynamic W4 G128 range search: six-row audit

## Scope and parity

Qwen2.5-0.5B-Instruct, frozen revision in protocol.json, FP16 weights and FP32 GPTQ math; symmetric W4A16 G128, block 128, damp 0.1, desc_act false. Eight distinct WikiText-2 train articles supply four search and four held-out 128-token sequences; inputs are FP model activations. Native GPTQModel 2.2.0 and custom dynamic alpha=1 match exactly on the Task 0 full layer and all six representative rows across three native full-layer runs. No test data or PPL used.

D is native dynamic GPTQ (alpha=1). A freezes MinMax at pristine W. B uses whole-row RTN output MSE to choose one shared alpha, then runs dynamic GPTQ. C evaluates all ten shared-alpha dynamic GPTQ candidates from pristine W and selects on search output MSE. The fixed-range v7 per-channel 43.26% result has different grouping and range timing; its contribution from weak fixed range cannot be numerically decomposed by this G128 study.

## Main comparison

C vs D: 3 improve, 2 tie, 1 regress on held-out; median relative gain 11.90%, pooled SSE gain 15.08%, worst row gain -0.64%. Full row data and all actual scales/zero-points are in six_row_results.csv and raw JSON.

- D_native_dynamic vs A_fixed_minmax: held-out median 1.46%, pooled SSE 3.81%, 3 improve / 0 tie / 3 regress.
- B_rtn_then_dynamic vs A_fixed_minmax: held-out median 6.66%, pooled SSE 15.51%, 4 improve / 0 tie / 2 regress.
- C_gptq_dynamic_grid vs A_fixed_minmax: held-out median 20.73%, pooled SSE 18.31%, 4 improve / 0 tie / 2 regress.
- B_rtn_then_dynamic vs D_native_dynamic: held-out median 3.71%, pooled SSE 12.16%, 3 improve / 1 tie / 2 regress.
- C_gptq_dynamic_grid vs D_native_dynamic: held-out median 11.90%, pooled SSE 15.08%, 3 improve / 2 tie / 1 regress.

Held-out output MSE by preregistered row:

- L0 R0: D 1.46108e-05; A 1.55732e-05; B alpha=1.0 1.46108e-05; C alpha=0.8 9.59063e-06.
- L0 R448: D 3.81458e-05; A 2.75486e-05; B alpha=0.9 3.83881e-05; C alpha=0.9 3.83881e-05.
- L12 R0: D 0.00551092; A 0.0054934; B alpha=0.8 0.00510167; C alpha=1.0 0.00551092.
- L12 R448: D 0.00120908; A 0.0017158; B alpha=0.9 0.00205905; C alpha=1.0 0.00120908.
- L23 R0: D 0.000940659; A 0.00081378; B alpha=0.7 0.000716694; C alpha=0.7 0.000716694.
- L23 R448: D 0.0075484; A 0.00780029; B alpha=0.7 0.00547583; C alpha=0.7 0.00547583.

The ten-point search guarantees no worse **search** loss than D because D is alpha=1 in the same flow; it gives no held-out or PPL guarantee.

## Proxy ranking and cost

Shared-alpha RTN output/GPTQ Spearman median 0.927; one-group intervention median 0.661. See ranking_summary.json for p25/p75, min/max, weight proxy, all top-k hits, and per-group regrets. Interventions rerun all downstream GPTQ compensation; they are diagnostics, not the C method.

Top-2 rerank uses 16 GPTQ evaluations across six rows vs 60 for C; top-3 uses 21. Both match C's alpha on 5/6 rows; their worst held-out regret relative to C is 3.78%. Both count ten RTN evaluations per row and a fresh GPTQ rerun of every shortlisted candidate. C's 60 candidate evaluations took 12.8s; the full six-row invocation including 420 group interventions took 133.1s. These are offline calibration costs, not online inference speed.

## Group interaction and failure cases

Layer23 row448 legacy coordinate replacement search full SSE 1.83311, held SSE 3.61604; cross terms -1.25332 and 0.405559. This RTN-only one-group replacement can lose cancellation on held-out, so its search gains cannot be attributed to dynamic GPTQ. All six rows' independent/full/cross SSE, including dynamic D/C, are in group_interaction_diagnostics.csv.

The legacy RTN coordinate diagnostic improves held-out full SSE relative to independent-group RTN selection on 5/6 rows, but layer23 row448 regresses by 35.04%. Its group-local vs whole-row candidate rankings disagree in 7/7 groups on search. These are effects of group interaction under RTN replacement, with no demonstrated transfer to a GPTQ coordinate method.

Layer23 FP-input shift: global RMS ratio held/search 0.994, median per-feature RMS relative change 9.4%, Gram Frobenius relative change 48.5%. These describe the split, without isolating activation shift as the cause of loss changes.

Worst C-vs-D held row: layer 0 row 448, relative gain -0.64%, C alpha 0.9. Its exact losses remain in the raw JSON.

Worst B-vs-D held row: layer 12 row 448, relative gain -70.30%, B alpha 0.9; the proxy can choose a candidate that GPTQ search would reject. This is a negative result kept in the raw JSON.

The dynamic D/A differences change sign across rows, so fixed MinMax is not uniformly weaker. For layer23 row448, v7's RTN coordinate search lowered search error but its cross-term changed from negative on search to positive on held-out. This supports a cancellation/generalization concern for that RTN diagnostic; it does not establish the causal mechanism for dynamic GPTQ or final model quality.

## Interpretation and next evidence

The six rows use one projection type (q_proj), so projection-type differences are untested. Search/held-out article counts are small; alpha stability across other calibration splits is untested. No quantized-prefix activation, full-layer reconstruction, full-model PPL, or downstream quality has been measured here. Gate B fails the fixed 5/6 consistency rule, so no 96-row or PPL expansion follows from this run. The thesis hypothesis remains open but lacks broader held-out stability and model-level evidence. The current engineering result is an auditable native-parity dynamic GPTQ range search with resume, measured candidate budgets, and independent raw-result checks. No originality or publication claim is made.

## Reproduction

From E:\quant-thesis with the existing .venv-gptq environment:

```powershell
$env:PYTHONPATH='E:\quant-thesis'
.\.venv-gptq\Scripts\python.exe -m experiments.range_search_w4.run_task0_parity
.\.venv-gptq\Scripts\python.exe -m experiments.range_search_w4.run_dynamic_v8 --smoke
.\.venv-gptq\Scripts\python.exe -m experiments.range_search_w4.run_dynamic_v8
.\.venv-gptq\Scripts\python.exe -m experiments.range_search_w4.verify_native_rows_v8
.\.venv-gptq\Scripts\python.exe -m experiments.range_search_w4.diagnose_activation_v8
.\.venv-gptq\Scripts\python.exe -m experiments.range_search_w4.summarize_dynamic_v8
```

The parity command requires a fresh v8 output directory; the existing parity artifacts are already saved. The main runner resumes completed per-row JSON files without overwriting them. The summarizer regenerates CSV and reports from all six raw rows.
