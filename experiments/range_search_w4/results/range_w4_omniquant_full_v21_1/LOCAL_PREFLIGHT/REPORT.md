# v20.1-Kaggle 2048-token matched-shape preflight

**Verdict: `LOCAL_M3_FORMAL_GO`**

No formal 20-epoch training, WT2 PPL, or C4 PPL was run.

## Measured environment and identity

1. GPU: `NVIDIA GeForce RTX 4060 Laptop GPU`; GPU count `1`; selected device `cuda:0`.
2. GPU total VRAM: `7.99560546875` GiB.
3. OmniQuant commit: `feffe8ea87d80f7bb57b6e25e7cff9dc950fcc14`; v20 patch files verified by bundle hashes.
4. Model: `Qwen/Qwen2.5-0.5B-Instruct` at `7ae557604adf67be50417f59c2c2f167def9a775`; config commit `None`.
5. Calibration hash: `664419273f63df3c0b0746b18bb66b593a5384bbe6ba8afcd6f4905c9479aae3`; 128 samples × 2048 tokens, in frozen order.
6. Formal update count: `61,440` = 24 × 20 × (128 / 1).

## 2048-token layer timings

| Layer | Measured updates | Median total (s) | P95 total (s) | Median student (s) | Median backward (s) | Peak allocated (GiB) | Peak reserved (GiB) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| L0 | 5 | 0.0338 | 0.0363 | 0.0110 | 0.0246 | 1.463 | 1.562 |
| L12 | 3 | 0.0350 | 0.0359 | 0.0100 | 0.0259 | 1.446 | 1.547 |
| L23 | 3 | 0.0338 | 0.0353 | 0.0108 | 0.0245 | 1.446 | 1.547 |

- Slowest representative layer by measured mean total step: **L12**.
- LWC correctness: **PASS**; finite nonzero gradients, an observed parameter update, and finite reconstruction losses are checked on real 2048-token inputs.
- Quant-off native/port 2048 parity: **PASS**.
- OOM: **no** for completed target runs.
- L0 eight-step allocated growth: `0.0` GiB; monotonic non-decreasing `True`; unexplained-growth warning `False`.

## Cost and limits

- L0 official capture boundary: 0.0473 s/sample; full 128-sample extrapolation is in `FORMAL_M3_COST_BREAKDOWN.csv`.
- Central total estimate: **0.641 h**.
- Conservative total estimate: **0.837 h** with ×1.25 measured-step safety multiplier.
- 24-hour gate: **PASS**.
- Kaggle single-session status: **KAGGLE_SINGLE_SESSION_REQUIRES_RESUME**. No Kaggle runtime-budget API was available; retain block-level checkpoint/resume for a formal run.
- Layer weights in the cost projection are three equal eight-layer strata (L0, L12, L23); unsampled within-stratum variation is not measured.
- L12/L23 inputs use native FP16 full-model prefix activations. The preflight does not claim they match a formal run's sequentially quantized prefixes.
- Dense export cost uses a measured FP16 state serialization/reload proxy; per-layer W4-to-dense conversion itself was not run. The stated safety margin covers this unmeasured operation but does not turn it into a direct measurement.

## Scope decision

This preflight supports a runtime/memory readiness decision only. It contains no formal M3 checkpoint and no M3 quality result. A READY verdict authorizes a later formal calibration request; it does not establish quality relative to M2.
