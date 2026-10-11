# Formal M3 cost estimate (low confidence)

Frozen official semantics require 24 blocks × 20 epochs × 128 samples / batch 1 = **61,440 optimizer updates**. The only step measurements are L0, batch 1, 64 tokens. The median disposable-profile step is **0.022346 s**. No 2048-token optimization step was run.

Scaling scenarios apply the measured 64-token median to sequence factors 32 (all work linear), 32^1.5 (central interpolation), and 32^2 (attention-dominant upper scenario). They omit model/block handoff, initial/final calibration capture, save/export and any additional host overhead, so they are not performance guarantees.

- optimistic_linear_sequence: factor 32.000; compute-only estimate **12.20 h**
- central_effective_exponent_1_5: factor 181.019; compute-only estimate **69.03 h**
- conservative_quadratic_attention: factor 1024.000; compute-only estimate **390.52 h**

The conservative estimate is above the frozen 24-hour gate. Because the smoke/full sequence ratio is 32× and the conservative estimate is an unvalidated extrapolation, report it as low confidence. Formal 20-epoch training was not run. The measured 64-token smoke reserved VRAM is reported separately; 2048-token formal-training VRAM feasibility was not measured and does not pass its readiness gate.
