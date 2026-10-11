# v21.1 — local frozen model repair and M3 execution

## Final status

`M3_FORMAL_STOP_NONFINITE_LWC_GRADIENT_LAYER21`

The frozen model repair and all preflight gates passed. Formal M3 training then stopped at the registered numerical integrity check: a non-finite LWC gradient on decoder layer 21, epoch 0, calibration sample 0. The required 61,440 updates were not completed; the last valid committed checkpoint contains 53,760 updates through layer 20. This is a stopped partial run, not a completed M3 baseline.

## Protocol and preflight

The frozen run used `Qwen/Qwen2.5-0.5B-Instruct` at revision `7ae557604adf67be50417f59c2c2f167def9a775`, upstream OmniQuant commit `feffe8ea87d80f7bb57b6e25e7cff9dc950fcc14`, W4A16 asymmetric affine G128, LWC on, LET off, and the unchanged 128 × 2048 WikiText-2 calibration manifest (SHA256 `664419273f63df3c0b0746b18bb66b593a5384bbe6ba8afcd6f4905c9479aae3`). The local isolated snapshot and tokenizer checks passed. Native and port quantization-off outputs passed the frozen parity checks. Matched 2048-token L0/L12/L23 preflight passed; maximum representative peak reserved VRAM was 1.5625 GiB. The local conservative projected full-run cost was 0.83699 h against the 24 h gate.

These are preflight results; they do not override the later formal training failure.

## M3 stop evidence

| Item | Observed value |
|---|---|
| Last complete checkpoint | `M3/checkpoints/block_20` |
| Complete blocks / total | 21 / 24 |
| Committed updates / required | 53,760 / 61,440 (87.5%) |
| Failed location | layer 21, epoch 0, sample 0 |
| Recorded loss | 0.15326565504074097 (finite) |
| LWC parameters with gradients | 14 |
| Gradient finite / runner's `flat != 0` flag | false / true |
| Final model export and reload | Not run |
| WT2 / C4 evaluation | Not run |

Attempt 1 logged 2,560 training steps but did not commit a checkpoint because its first checkpoint `fsync` failed. Those updates were discarded; attempt 2 restarted from pristine weights. Attempt 2 committed 53,760 updates before its layer 21 check failed. The failed sample is recorded in the CSV but is not counted as a committed update.

The runner computes its nonzero flag with `torch.any(flat != 0)`, for which NaN/Inf values compare unequal to zero. The `true` flag is therefore not evidence of a valid finite update. The failure is evidence about this run under this frozen configuration. The run does not identify the root cause, and no AMP, optimizer, loss, calibration, or quantization setting was changed to work around it.

## Model-level results

There is no M3 WT2 PPL, C4 PPL, completed calibration wall time, or final-model peak VRAM result. The preflight predicted cost is not a completed calibration time. The upstream log showed 2122.84 MiB in its allocated-memory readout, but the run stopped before the final peak-reserved summary was produced.

For reference, the frozen v18 metrics remain M0 WT2/C4 12.46340950/19.24855582, M1 13.88436957/22.23392445, and M2 13.76055605/22.03288316. These are reused historical results, not rerun here. See `M3/M0_M1_M2_M3_STATUS.csv`; M3 metric fields remain empty.

## Disposition

The v21.1 local M3 execution did not produce a completed baseline. The block 20 checkpoint and complete failure evidence are preserved for audit. No dense export or evaluation was run after the stop. v20/v21 and historical experiment outputs were not modified.
