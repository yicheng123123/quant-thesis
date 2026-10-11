# v17.3 M2 runtime and cost audit

## Result

Correctness: **PASS**. The isolated implementation reuses an alpha-independent native inverse input while preserving installed `GPTQ.quantize()`, dynamic group range estimation, and GPTQ compensation. No historical v8–v17.2 source, checkpoint, or result was modified.

For the L0 `self_attn.k_proj` full shape `[128,896]`, all ten optimized alpha candidates matched the v17.2 saved reference tensors and losses. Alpha=1 matched native GPTQ for weight, scale, zero, and `g_idx`. Reversing candidate order gave exact tensors; the direct output-MSE calculation matched exactly; the selected commit matched reference hashes and left unselected rows pristine. The complete q/k/v dependency replay reproduced the v17.2 downstream activation and next-target Hessian hashes exactly.

The first self-prefix/Hessian attempt failed because it compared the next-target Hessian after only a single K-row commit, while v17.2 had built that Hessian after committing the full searched K subset and native alpha=1 V/Q siblings. The verification was corrected to reproduce the frozen q/k/v transition; no algorithm change was made. The final correctness record is PASS.

## Timing

Five synchronized repeats per implementation were measured after one excluded warmup. A full grid is ten full-shape native calls and scores 32 fixed rows from each candidate:

- Reference median: **4.342 s** per 32-row × 10-alpha grid.
- Optimized median, including one prepared-state build: **3.919 s**.
- Local measured ratio: **1.108×** for this shape and smoke workload.
- Candidate medians: 0.415 s reference and 0.385 s optimized excluding preparation; preparation median 0.0101 s per optimized grid.

This is a local GPTQ calibration-search timing result. It is not a full-model wall-time measurement or an online inference speedup. The operator profile is instrumented and used for attribution only; it is not added to benchmark wall time.

## Corrected units and dependency schedule

The full-model grid has 3,041,280 row-alpha score slots but **1,680 full-Linear GPTQ executions**. A full call computes every output row. All rows for one Linear share one input activation and Hessian. After the entire Linear is searched, row-wise selection and stitching complete, and the Linear commits once.

Per layer, q/k/v use one shared input/H; after all three commit, advance once to capture o input. o commits before one refresh to the shared up/gate input. up/gate commit before one refresh to down input. Down commits before advancing to the next layer. This produces 96 distinct group input/H states and 95 dependency transitions through the final quantized block. GPTQ processing order is recorded separately from forward dependencies. No row batch causes a refresh.

## Cost decision

The corrected conservative estimate is **2.135 h**, below the unchanged **48 h** gate. It uses 1,680 as the full-candidate execution count and separately tracks capture, Hessian setup, sequential forward, dependency-boundary replay, scoring/stitch/commit, load, and I/O. The full-model run was not performed. The cost estimate still has unverified 2,048-token capture/replay extrapolations, an intentionally conservative replay proxy that may overlap sequential forward, and a 30-minute scoring/stitch assumption.

The old v17.2 48.79 h estimate is superseded because its capture source and refresh units were wrong. Its candidate call count was correct. See `UPDATED_FULLMODEL_COST.md` and `CORRECTED_FULLMODEL_COST.json` for source values and components.

## Next evidence

The candidate reuse is correct and gives a modest local timing improvement. The estimated cost remains below the gate, but a formal run should first validate 2,048-token capture and dependency-boundary runtime and replace the scoring/stitch assumption with a matched-shape measurement. This audit does not authorize or claim full-model search or PPL completion.
