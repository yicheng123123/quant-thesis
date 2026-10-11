# Corrected full-model M2 cost estimate

Status: **M2_COST_GO** against the frozen 48-hour gate. This is a conservative extrapolation, not a measured full-model run.

The work units are 168 Linears, 304,128 output rows, 3,041,280 row-alpha score slots, and **1,680 full-Linear candidate GPTQ executions**. Each of the ten alpha calls produces every output row for its Linear. Row selection and stitching happen after the candidate grid; they do not multiply GPTQ calls.

The formula is a sum over Linear/module groups of input capture/preparation, one Hessian setup per distinct shared input, ten full-Linear candidate GPTQ runs per Linear, all-row scoring and selection, one row-wise stitch/commit, and only dependency-required activation advances. q/k/v share one input/H state; up/gate share one. This gives 96 distinct group input/H states (four per layer) and 95 inter-group/layer transitions through the final block. Sibling projection processing order does not create an additional forward dependency.

| Cost component | Conservative seconds | Basis |
|---|---:|---|
| Full-Linear candidate GPTQ | 2,066.10 | v17.2 per-shape measurements with the v17.3 local delta applied only to 48 matching `[128,896]` targets and doubled conservatism |
| Unique group Hessian preparation | 788.47 | 96 shared-input groups, scaled from measured v15 H builds, 2× safety |
| Initial input activation capture | 281.62 | v15 activation capture, token/sequence scaled, 2× safety |
| Sequential quantized-prefix forward | 391.93 | v15 quantized-prefix forward, token/sequence scaled, 2× safety |
| Dependency-boundary replay proxy | 2,331.79 | 96-group conservative proxy; may overlap sequential-forward work and is retained as an upper-cost allowance |
| All-row scoring, selection, stitch, commit | 1,800.00 | v17.2 15-minute central assumption doubled; still awaiting matched-shape measurement |
| Model load | 3.90 | v15 measured load doubled |
| Checkpoint and I/O | 20.94 | two 942 MiB writes at the measured assumed rate, doubled |
| **Total** | **7,684.75 s = 2.135 h** | Sum of the listed conservative components |

## v17.2 estimate correction

The v17.2 **48.79 h** estimate used the right 1,680 candidate execution count. Its capture term was wrong: `133.076 s` came from `prefix_native_quantization_seconds_measured`, which is GPTQ quantization time. The requested `native_full_prefix_seconds` key was absent, so the estimate fell back to that value and token-scaled it as activation capture. It also charged four within-layer boundaries as four additional complete full-model passes and counted Hessian builds per projection instead of per shared-input group. The old estimate is superseded for the full-model decision; historical artifacts remain unchanged.

## Limits

- No formal full-model M2 or PPL run was performed.
- Capture and replay terms at 2,048 tokens are extrapolated from 128-token evidence; matched-shape validation is still needed.
- The 96-group replay proxy is intentionally conservative and can overlap the sequential-forward term.
- The 30-minute scoring/stitch/commit allowance is an assumption, not a measurement.
- v17.3's local candidate timing gain applies only to the tested L0 `[128,896]` shape in the cost update; no speedup credit is given to other shapes.
