# v17.2 M2 full-model smoke resume

Verdict: **M2_FULLMODEL_SMOKE_NOT_READY**.

The certified implementation path and SHA matched v17.1. The alpha=1 preflight exactly matched native GPTQ for weight, scale, zero, g_idx and output MSE. Actual pre-hooks on all seven L0 projections captured the Linear inputs. q/k/v inputs were bitwise equal, and the GPTQ order `[k,v,q] -> [o] -> [up,gate] -> [down]` is documented separately from the forward dependency graph.

The L0 k_proj experiment completed 320 row-alpha scores from the frozen v16 calibration samples 0 and 1, first 64 tokens each. Ten full 128x896 native-core calls scored the 32 fixed rows. Alpha counts were `{"0.1": 0, "0.2": 0, "0.3": 0, "0.4": 0, "0.5": 0, "0.6": 0, "0.7": 0, "0.8": 10, "0.9": 12, "1.0": 10}`. Candidate-order reversal, embedded alpha=1, direct native range-hook comparison on four rows across all ten alphas, selected commit hashes, and absence of nonselected-state leakage passed.

Self-prefix passed. Committing row 0 at alpha 0.9 changed the downstream o_proj input: relative L2 drift 0.000153637, maximum absolute change 0.00012207. The module transition committed the k/v/q group, captured a refreshed o_proj input, built its Hessian from that activation, committed native alpha=1 o_proj, then captured current up/gate/down inputs.

**Cost gate: M2_COST_NO_GO, conservative estimate 48.79 hours.** This exceeds the frozen 48-hour limit. The run therefore ends at `M2_FULLMODEL_SMOKE_NOT_READY`; no M3 smoke, full-model search, or PPL was run. See `UPDATED_M2_COMPUTE_BUDGET.md` for measured timings and explicit capture/Hessian/refresh/checkpoint assumptions.

No held-out or final-model improvement is claimed; the landscape uses only the frozen search subset. Runner issues encountered before successful completion are itemized in `EXECUTION_ISSUES.json`.
