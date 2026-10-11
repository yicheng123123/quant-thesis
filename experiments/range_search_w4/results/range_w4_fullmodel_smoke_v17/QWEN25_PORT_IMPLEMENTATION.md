# Qwen2.5 OmniQuant port status

The v17 working copy is based on the v16 audited OpenGVLab/OmniQuant commit. The adapter deep-copies the native Transformers Qwen2 decoder and wraps only the seven intended projection `Linear` modules with the upstream `QuantLinear`. The native decoder retains its attention, RMSNorm, RoPE, grouped-query attention, and forward implementation.

The OmniQuant dispatcher now recognizes `model_type=qwen2`, caches the decoder forward arguments used by Qwen2, forwards them to teacher/student calls, and exposes optional timing callbacks. The original quantizer, LWC factors, optimizer, AMP scaler update sequence, and loss were not intentionally changed. The three source edits and their stated semantics are recorded in `OMNIQUANT_PATCH_MANIFEST.csv`.

Python syntax compilation passed. The adapter was **not** run through native-off forward equivalence, seven-linear discovery, or the tiny OmniQuant optimization smoke. Those checks were not started after M2 `alpha=1` parity failed. Therefore this is implementation plumbing in progress, not a validated OmniQuant-Qwen2.5 port. No quality conclusion follows.

One launch guard is still needed before any future L0-only call: the upstream loop defaults to all decoder layers. v17 did not execute that loop, and did not add or validate a layer-limit guard. Future smoke code must explicitly cap processing at layer 0 before it can be run safely under the L0-only scope.
