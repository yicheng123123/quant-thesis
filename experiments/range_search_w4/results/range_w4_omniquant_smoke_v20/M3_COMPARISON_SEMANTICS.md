# M1/M2/M3 comparison boundary

- **M1:** native symmetric GPTQ W4A16 G128, second-order compensation.
- **M2:** M1 plus discrete per-output-row range search, retaining GPTQ compensation.
- **M3:** official OmniQuant weight-only W4A16 G128, asymmetric affine quantizer, LWC bound optimization, block reconstruction, FP teacher and quantized student, without GPTQ compensation; LET off.

The eventual comparison shares the model revision, frozen calibration IDs and order, W4A16/G128 target, WT2/C4 frozen token streams and evaluator, FP16 dense-dequant quality representation. It does not share the optimizer, quantizer semantics, reconstruction objective or GPTQ compensation. Results therefore compare complete methods and cannot isolate clipping strength, discrete search versus gradients, or quantizer format without separate controlled ablations.

The v20 two-sample/two-step output is compatibility evidence only. It is not an M3 quality score. Formal calibration and PPL were not run.
