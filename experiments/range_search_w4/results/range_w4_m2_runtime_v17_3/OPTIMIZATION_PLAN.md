# v17.3 bounded runtime optimization

## Profile finding

The primary cost unit is a full-shape native GPTQ call per alpha per Linear. The 32-row smoke uses ten full `[128,896]` calls, not 320 GPTQ calls. Full model therefore has 1,680 full-Linear candidate executions and 3,041,280 row-alpha score slots.

The reference profile instruments one full-shape candidate and is attribution-only because profiler hooks inflate wall time. The synchronized five-repeat benchmark is the timing source. Reference reset/clone work was separately probed. The optimized path removes the wrapper's redundant FP16 weight clone and prepares the alpha-independent native block-inverse once per full-grid, while leaving the installed native `GPTQ.quantize()` and compensation loop intact.

## Optimization invariants

1. Preserve symmetric W4A16, G128, `desc_act=False`, `static_groups=False`, `true_sequential=True`, damping 0.1, and the alpha grid 0.1–1.0.
2. Keep alpha=1 identical to native GPTQ.
3. Use pristine `W` and `H` for every alpha; clone mutable candidate state.
4. Reuse only the inverse output that is mathematically independent of alpha; retain the native final Cholesky and all GPTQ compensation.
5. Score with the frozen same `X`; stitch and commit all selected rows before a dependency-required forward refresh.
6. Do not refresh between row batches. Share q/k/v and up/gate input/H captures.

## Bounded verification sequence

- Run alpha=1 parity against the certified v17.2 native implementation.
- Compare all tensors and scores for 32 rows × 10 alphas against saved v17.2 full-shape results.
- Reverse candidate order and require exact result tensors.
- Check direct output-MSE arithmetic, selected row commit, and pristine unselected rows.
- Replay the v17.2 self-prefix and complete q/k/v-to-o dependency transition; require exact activation and next-target Hessian hashes.
- Measure five synchronized full-grid repetitions per implementation after one excluded warmup; include optimized preparation time.

## Stop/continue rule

Do not expand to a formal full-model search or PPL in this runtime audit. Cost eligibility is evaluated against the frozen 48-hour gate using the corrected work-unit and dependency model. Matched 2048-token capture/replay measurements remain the next high-value evidence because they dominate remaining estimate uncertainty.
