# v18 matched-shape precheck and cost gate

Correctness: **PASS** (all 10 frozen precheck conditions true).

## Work units

- 168 Linear modules, 304,128 output rows, and 10 alpha values.
- **3,041,280 row-alpha score slots** are produced by **1,680 full-Linear GPTQ executions**.
- One candidate execution returns every output row for that Linear. Per-row argmin and final stitch happen after scoring; they do not launch GPTQ per row.
- 96 unique input/H groups: per layer `[qkv, o, up/gate, down]`.
- 95 actual dependency transitions per method: 24 qkv→o, 24 o→up/gate, 24 up/gate→down, and 23 down→next layer.

## Matched measurements

All candidate and direct output-scoring timings use full output rows at 128×2048 calibration shape, with separate row update, stitch, and commit measurements. q/up/down data are in `MATCHED_SHAPE_TIMING.csv` and `MATCHED_SHAPE_PRECHECK.json`; the `[128,896]` k/v addendum is in `KV_SHAPE_TIMING.csv` and `KV_SHAPE_PRECHECK.json`.

The observed refresh calls are one full decoder-block replay at each actual dependency edge. The qkv and up/gate siblings share the same X/H. The next boundary is not refreshed for each row batch.

## Cost result

- M2 central: **0.595 h**.
- M2 conservative: **1.338 h**; frozen 48 h gate: **M2_COST_GO**.
- M1 conservative: **0.236 h**.
- Combined M1+M2 conservative: **1.574 h**.
- Evaluation time is outside the M2 calibration gate and will be recorded from actual WT2/C4 runs.

The old v17.3 **2.135 h** estimate is superseded because it used different conservative assumptions and did not have the present matched-shape score/stitch/commit and 2048-token boundary timings. The v17.2 48.79 h estimate remains invalid for the reasons recorded in the v17.3 cost audit; the 48 h gate itself is unchanged.

Cost components, shape counts, P95 selection, safety factors, and I/O throughput pilot are in `FINAL_COST_MODEL.json`. The I/O pilot used a durable flush and its temporary file was removed.

## Scope

No formal model quantization or PPL evaluation had started at precheck completion. The estimated decision is a cost gate only, not a quality result.
