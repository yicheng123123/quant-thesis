# v21.1 M3 formal run report

## Outcome

**STOPPED; M3 is incomplete.** The frozen runner stopped on the first sample of decoder layer 21 because the LWC gradient finite check failed. The run must not be treated as a completed OmniQuant M3 baseline.

## What passed before formal training

- Exact local model revision, tokenizer, and frozen calibration identity passed.
- Native Hugging Face versus OmniQuant port quantization-off parity passed at the frozen check points.
- Matched 2048-token L0/L12/L23 timing and memory preflight passed on the RTX 4060 Laptop GPU.
- Conservative preflight cost estimate was 0.83699 hours, below the frozen 24-hour gate.
- Resume and atomic checkpoint smoke checks passed.

## Formal training evidence

The required workload was 61,440 optimizer updates across 24 decoder blocks. The latest complete checkpoint is `checkpoints/block_20`: 21 blocks and 53,760 updates (87.5%). At layer 21, epoch 0, sample 0, the recorded loss was finite (`0.15326565504074097`), but at least one of 14 LWC parameter gradients was non-finite. The runner's separate `torch.any(flat != 0)` flag was true; because NaN/Inf compare unequal to zero, this is not evidence of a valid nonzero update. The runner raised `FloatingPointError` and stopped.

Attempt 1 trained 2,560 steps but failed before committing its first block because checkpoint `fsync` used an invalid read-only handle. Those steps were not included in the final checkpoint. After the narrowly scoped checkpoint I/O repair, attempt 2 restarted from pristine frozen weights and committed blocks 0–20. Its final CSV row is the failing layer 21 sample and is not counted as a committed update.

Block 20 passed its checkpoint health checks: all 40 checked floating tensors were finite, all quantizer scales and zeros were finite, and seven quantized linear modules were present. The layer 21 failure therefore marks the first observed numerical gate failure in the second attempt; this run alone does not establish its underlying cause.

## Evaluation boundary

No complete M3 model was exported. Dense-dequant export, export/reload validation, WikiText-2 PPL, and fixed-budget C4 evaluation were not run. M0/M1/M2 values in `M0_M1_M2_M3_STATUS.csv` are reused from the frozen v18 comparison; M3 metric cells are intentionally blank.

The formal log printed an upstream allocated-memory readout of 2122.84 MiB. The run stopped before the runner wrote a final peak-reserved summary, so this value must not be reported as formal peak reserved VRAM. Preflight representative layers had a measured maximum peak-reserved value of 1.5625 GiB.

## Preserved evidence

- `STOP_DIAGNOSTIC.json`
- `RUN_STATUS.json`
- `checkpoints/LATEST.json` and `checkpoints/block_20/`
- `logs/training_steps.csv`, `logs/block_health.csv`, and both attempt logs
- `ATTEMPT_01_MANIFEST.json`, `RUN_STATUS_ATTEMPT_01.json`, and `RUNNER_RECOVERY_NOTE.json`

No training setting was changed after the non-finite-gradient stop, and no subsequent training or evaluation was launched.
