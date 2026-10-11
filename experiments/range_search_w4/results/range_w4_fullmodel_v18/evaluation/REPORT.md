# v18 model-level evaluation

Evaluation correctness: **PASS**. All six M0/M1/M2 corpus runs used the frozen token-ID manifests, the shared `benchmark.eval.ppl.evaluate_ppl` function, FP16 SDPA, TF32 disabled, `max_length=2048`, and `stride=512`. Exact per-window loss and valid-token counts are saved under `evaluation/windows/`; aggregate NLL was cross-checked against those exact window values.

## WikiText-2 test

- M0 FP16 PPL: 12.46340950 (299077 targets)
- M1 native GPTQ G128 PPL: 13.88436957 (299077 targets)
- M2 pure range search PPL: 13.76055605 (299077 targets)
- M2 vs M1: -0.8917% PPL.

## C4 validation fixed token budget

- M0 FP16 PPL: 19.24855582 (131071 targets)
- M1 native GPTQ G128 PPL: 22.23392445 (131071 targets)
- M2 pure range search PPL: 22.03288316 (131071 targets)
- M2 vs M1: -0.9042% PPL.

C4 is the frozen **validation fixed token budget PPL**, not standard/full C4 PPL. The two datasets are reported separately.

## Frozen decision

Verdict: **FULLMODEL_M2_QUALITY_IMPROVED**. The primary decision uses only the strict frozen WikiText-2 test comparison M2 vs M1. No method, alpha grid, calibration data, or evaluation settings were selected after seeing these scores.
