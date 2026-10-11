# v20 OmniQuant Qwen2.5 compatibility smoke

**Verdict: `M3_FORMAL_EXECUTION_NOT_READY`.** This means the formal full calibration gate is not met; no formal training or PPL was run.

- Upstream commit/hash gate: PASS; quantizer/LWC/optimizer math sources are byte-identical to upstream; the driver has only declared Qwen plumbing and smoke-only controls/observers.
- Quant-off native/port parity on L0/mid/last and logits: PASS, 64 frozen tokens.
- Qwen target modules: 168 across 24 layers; mapping PASS.
- Attention semantics and native kwargs: PASS; LWC map: PASS (232,960 scalars).
- Functional LWC smoke: 2 optimizer steps; gradients PASS, updates PASS, reconstruction PASS.
- Dense L0 export/reload: PASS; shared evaluator: PASS (127 scored targets; smoke only).
- Calibration wall for one L0/two-sample smoke: 0.86s. Disposable timing repeats: 5 after one excluded warmup; L0 64-token median forward 0.0073s, loss 0.0003s, backward 0.0129s, optimizer 0.0013s, total 0.0223s.
- Runtime extrapolation: optimistic 12.20h, central 69.03h, conservative 390.52h; 64-to-2048 scaling is low-confidence.
- Actual smoke peak reserved VRAM: 1.35 GiB (<8 GiB); full 2048-token formal-training feasibility is unproven and its readiness gate is not passed.

M3 remains a method-level comparator to M1/M2 because OmniQuant uses asymmetric affine quantization, LWC block reconstruction and no GPTQ compensation. Smoke losses/PPL are not model-quality results. STOP.
