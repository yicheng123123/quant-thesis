# v17 smoke results

## Verdict

**FORMAL_EXECUTION_NOT_READY**

The M2 `alpha=1` parity gate failed. The run stopped before the 320 row-alpha candidate grid, commit/prefix checks, and all OmniQuant/PPL smokes. No full-model experiment was launched.

## What ran

- Independent OmniQuant environment installation completed and CUDA was available.
- The Qwen2.5 architecture adapter, modified OmniQuant dispatcher/scaler, and M2 script passed Python syntax compilation.
- The native Qwen2 GPTQ sequential group order was read from GPTQModel and recorded.
- M2 loaded the v16 frozen calibration manifest, captured two L0 calibration inputs of length 64, built a Hessian, and attempted the strict alpha=1 comparison for the first 32 `k_proj` rows.
- The alpha=1 gate failed. Diagnostic artifacts localize the earliest recorded compensated-input drift to group 1 and show an FP16 quantized-weight mismatch by group 4.

## Not run

The 320-candidate search, selection, selected-row commit, self-prefix activation check, serial-vs-batch grid comparison, Omni L0 optimization, dense export/reload, PPL smoke, WT2/C4 manifest restoration, and formal cost extrapolation did not run.

## Evidence and limits

See `M2_PARITY.json`, `M2_PARITY_DIAGNOSTIC.json`, and `M2_PARITY_TRACE.json`. Local output MSEs are close, but native scale/weight parity is part of the correctness gate, so closeness does not turn this into a pass. The exact operator-level origin of the compensated-state drift remains unresolved. No range-search benefit, OmniQuant quality, PPL, or formal-run cost is established in v17.
