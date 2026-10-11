# v52 initialization data flow

Q is formed from current exact integer codes and M2 FP16 scale/g_idx; E=Q64-W64; p=E@C64. initial_quad=(E*p).sum and st.loss=initial_quad.clone are established before the consistency check. The original FP32-output direct loss (FP64 SSE accumulation) is initial. It remains the input to threshold=max(initial*1e-8,1e-14) and the reported initial_mse. The independent fallback never replaces either loss, p, codes, E, Q, B, counters, rejection mask, or candidate threshold.

Only original isclose(initial, initial_quad, rtol=2e-5, atol=1e-12) failing rows receive independent raw-X FP64 loss recomputation. Fresh row error is formed from actual FP16 grid and pristine W, not C or p. FP64 raw-X SSE/N is compared to initial_quad using the SAME isclose and loss-scale formula. Source binding, finite values, state versions and failed-row bytes are checked. A passing fallback records an initialization cross-precision difference and returns to unchanged candidate scoring/settle. It never accepts a candidate or modifies candidate acceptance conditions. Failing raw-X/quad or source/finite/state checks stop.

Provenance keys are explicit run_id, phase, block and dependency_group. X hash, shape and dtype identify content; C hash/shape/dtype only verify provenance. The original v49 builder is always executed on each new dependency group. No id(X) content proof, activation substitution or Gram reuse is added. Wrappers return the original X/C values untouched; their hashing and checks have named measured timers.

This is an empirical numerical check, not a theoretical safety guarantee. Normal-path candidate trajectories are tested for exact equality; broader modified-initialization execution is not declared stepwise equal to old runs.
