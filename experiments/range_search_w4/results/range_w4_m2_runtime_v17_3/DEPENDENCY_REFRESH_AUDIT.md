# M2 activation dependency and refresh audit

## Fixed execution units

- 24 decoder layers, 7 quantized Linear modules per layer: 168 Linears.
- 304,128 output rows and 3,041,280 row-alpha scores.
- 1,680 full-Linear candidate GPTQ executions: 10 full-shape alpha calls for each Linear. Each call emits candidates for every output row simultaneously.
- The 32-row v17.2 landscape came from 10 full `[128,896]` calls. The 32 rows were a scoring subset, not 32 separate GPTQ invocations.
- A Linear is searched completely, each output row selects its alpha, Q/scale/zero are row-wise stitched, the Linear is committed once, then dependent activations advance.
- Output-row score or stitch chunks cannot trigger a forward refresh.

## Row separability check

The installed GPTQ source uses a shared column Hessian/Hinv. `find_params` uses per-channel reductions; compensation updates `W1[:, i:]` with a row vector times an Hinv row and has no output-row reduction. This is mathematically separable across output rows under the frozen settings.

The existing ten full-shape candidates were rescored for all 128 output rows. They produced 1,280 row-alpha scores and a complete selected row-wise Linear. Four rows fixed by the earlier v17.2 protocol (0, 1, 17, 31) were also independently quantized for all 10 alphas (40 row calls). In all 40 comparisons, FP16 weights, zero points, g_idx, and local output deltas matched bitwise; selected alpha matched on all four rows. FP32 scales were not bitwise identical in every case, but all were within rtol `1e-6`, atol `1e-9` (largest relative difference about `2.5e-7`). The first small weight divergence appears at a later group after error compensation and is at most `1.2e-7` in the traced row; this is shape-dependent FP32 arithmetic, not cross-row coupling. The formal shared implementation must retain full-Linear shape, so this one-row kernel difference is not introduced by row-wise stitching.

## Per-layer dependency schedule

| Boundary | Shared input / operation | Required action |
|---|---|---|
| Layer input to q/k/v | One post-input-LayerNorm activation | Capture once; build one H/Hinv source; run 10 candidates per each of three Linears; stitch and commit q/k/v as one group |
| q/k/v to o | Current Q/K/V jointly determine attention context | Advance the attention subgraph once with committed q/k/v; capture o input |
| o to up/gate | o output changes the residual and post-attention normalization | Recompute o output and normalization once; capture one input shared by up and gate |
| up/gate to down | Both outputs jointly determine the gated MLP product | Commit both, compute activation product once; capture down input |
| down to next layer | down output changes the residual stream | Commit down and advance once to the next layer input; this same stream supplies the next layer's q/k/v input |

There are 96 distinct group-input/Hessian states (four per layer), not 168. There are 95 inter-group/layer dependency transitions through the final quantized block (the last block needs no post-commit forward for calibration selection). These are activation-boundary computations in a layer-wise sequential pass. They are not 95 full-model replays. The required full-model implementation is not present yet, so the exact runtime of its local boundary computations remains an estimate.

Sibling processing order `[k,v,q]` and `[up,gate]` is an implementation order. Forward dependencies require shared inputs for each sibling group and only require refresh after the whole sibling group is committed.

## v17.2 cost correction

The v17.2 candidate count was already correct: 1,680 full-Linear alpha calls. Its 48.79-hour estimate is superseded because the capture source was mislabeled: `133.076s` came from v15 `prefix_native_quantization_seconds_measured`, which measures native GPTQ quantization, not activation capture. The v17.2 loader found no `native_full_prefix_seconds` key in its historical evidence and therefore used that 133.076-second fallback. It then token-scaled this quantization time as if it were activation capture and charged four additional full-model passes. The corrected estimate uses the v15 activation capture, quantized-prefix forward, target-block replay, and Hessian measurements with explicit token/shape factors and safety multipliers. It keeps the candidate core estimate and 48-hour gate unchanged.

The resulting conservative estimate is recorded in `CORRECTED_FULLMODEL_COST.json`. It is a cost estimate, not measured full-model runtime; the 2048-token and layer-local replay terms still need matched-shape validation before a formal run.
