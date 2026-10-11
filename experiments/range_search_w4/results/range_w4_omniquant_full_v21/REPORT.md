# v21 Local OmniQuant M3: preflight stopped

**Verdict: `LOCAL_M3_NOT_READY_USE_KAGGLE`.** The local run stopped before model loading. No formal M3 calibration, M3 checkpoint, dense export, WT2 evaluation, or C4 evaluation was produced.

## Gate evidence

- The v20.1 bundle/source verification passed; official OmniQuant source hashes and the frozen calibration manifest matched.
- Calibration identity passed: 128 ordered WikiText-2 train samples × 2048 tokens, seed 20261005, manifest SHA-256 `664419273f63df3c0b0746b18bb66b593a5384bbe6ba8afcd6f4905c9479aae3`.
- Local runtime pins passed: Windows native, Python 3.12.10, PyTorch 2.6.0+cu124, Transformers 4.49.0, tokenizers 0.21.4, CUDA runtime 12.4, RTX 4060 Laptop GPU with 7.996 GiB reported VRAM, driver 555.97.
- Exact model-file identity failed before loading: the frozen snapshot lacks `model.safetensors` (expected 988,097,824 bytes, SHA-256 `fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe`) and `generation_config.json` (expected 242 bytes). The cache contains a 524,288,000-byte `.incomplete` model blob.
- Because the model identity gate failed, quant-off parity, LWC gradient/update checks, 2048-token timings, and memory checks were not run. Local runtime cost is therefore **not computable**; the Kaggle/T4 timings were not substituted for RTX 4060 timings.
- The v20.1 audited formal update count is preserved as **61,440**, but no optimizer updates were executed in this run.

## Frozen M0/M1/M2 context

| Method | WT2 PPL | C4 fixed-budget PPL | Calibration wall |
|---|---:|---:|---:|
| M0 FP16 | 12.46340950 | 19.24855582 | — |
| M1 native GPTQ | 13.88436957 | 22.23392445 | 648.876 s |
| M2 pure search | 13.76055605 | 22.03288316 | 2115.935 s |
| M3 OmniQuant | Not run | Not run | Not measured |

The frozen v18 result reported M2/M1 calibration wall ratio 3.261×. This is offline calibration cost. It is unrelated to inference speed. M2 and M3 remain a method-level comparison; no M3 quality conclusion is available.

## Next route

The local path stopped at the frozen identity gate. Use the already audited v20.1 Kaggle formal route in the next execution. Preserve the 24-hour gate and all frozen OmniQuant semantics.

## Artifacts

- Full protocol and parent artifact hashes: `PROTOCOL.json`
- Local environment, bundle/calibration checks, model mismatch details, and stopped gate: `LOCAL_PREFLIGHT/`
- Machine-readable gate summary: `correctness_checks.json` and `SUMMARY.json`
- Partial frozen baseline table: `FINAL_COMPARISON/M0_M1_M2_M3.csv`
