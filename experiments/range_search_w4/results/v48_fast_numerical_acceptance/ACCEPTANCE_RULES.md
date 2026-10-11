# v48 — frozen before GPU execution

This is an empirical FP64 numerical implementation, not a safety certificate or a proved rounding-error bound. No constants are fitted to historical errors, rejected proposals, held data, or final results. Sol high was requested; the interface does not expose a verifiable setting switch.

## Shared search

Import v46's byte-identical v45 `choose`, `propose`, `score_and_retire`, and `retire_caps`. Keep output chunks 256, coordinate ascending then offset -1/+1, ±1 legal integer neighbors, FP16 grid differences, 32 accepts, 128*d evaluations, blacklist state semantics, and initial FP32 loss threshold max(loss*1e-8,1e-14). No screening, new neighborhood, or shortened budget. The old path remains `batch_repair` unchanged.

## Error model and decision

u=2^-53; gamma(k)=k*u/(1-k*u). C is the unchanged streamed FP64 undamped Gram, chunk4096, divided by N and symmetrized. TF32 disabled. Stored FP16 X is converted exactly to FP64; pristine target W and effective grid weights are FP16.

C formation: use gamma(N+d+ceil(N/4096)) as a deliberately conservative serial accumulation length proxy. A proper elementwise Gram error bound would require abs(X).T@abs(X), kernel-specific reduction assumptions and division/symmetrization errors. We do not compute that extra Gram or claim those assumptions establish a theorem. The absolute-Gram proxy below can underestimate cancellation in individual off-diagonal entries. Independent raw-X historical, trajectory and full-terminal checks are therefore essential; empirical pass only covers these inputs/states.

Initialize B=abs(e)@abs(C); update B += abs(delta)*abs(C[j,:]) after each accept. This overestimates the absolute product for subsequent e, in exact arithmetic. For a selected move define T=2*abs(delta)*B[j]+delta^2*abs(C[j,j]). C/p/delta arithmetic uncertainty envelope U=64*gamma(N+d+ceil(N/4096)+32)*T + 2*abs(delta)*K[j] + 32*u*T, where K is the original v45 cache tolerance: (64*gamma(d+32)+64*u)*sum(abs(e))*max(abs(C_column)). This is a magnitude-aware empirical envelope, not a rigorously propagated raw-X bound.

If prediction < -U: accept by FP64 quadratic increment. Otherwise: check p against fresh e@C using the unchanged v45 cache tolerance; refresh p, recompute the selected scalar prediction, and independently verify raw-X a=Xe, b=Xdelta, mean(2ab+b²) using frozen v45 `verify`. Accept only if direct64 < -allowed_error (strict negativity beyond that empirical comparison tolerance). Otherwise reject/restore, blacklist the coordinate+offset for this state, do not reset evaluations; clear blacklist only after a legal accept.

At accepted-update milestones 1/8/16/24/32, check the complete p cache with v45's criterion, then refresh p and B from current e. Final cache check is mandatory. A cache mismatch, nonfinite, illegal code, actual-grid delta mismatch or failed same-precision formula/raw-X comparison is a hard error; never handled as an ordinary rejection.

Fast loss bookkeeping uses initial quadratic loss plus accepted increments; the FP32 initial direct loss is retained only for the unchanged search threshold. Final independent raw-X FP64 initial/final loss, for every output row, must show descent and agree with quadratic loss within the predeclared terminal criterion: abs(raw64-quad) <= 64*gamma(N+d+ceil(N/4096)+32)*(abs(e)@abs(C)*abs(e)).sum + 64*u*raw64. This criterion is empirical too. Both control and fast costs include this additional full terminal check.

## Coverage and cost

Before complete L0 matrices: replay v43 3 failure rows, v44 failed down row211, all 9 formal v46 rejected rows; validate original full FP32 scoring shape and independent raw-X FP64 increments. Reconstruct their authentic prefix inputs and check saved hashes. Historical FP32 non-descent may be FP64 descent and thus accepted by v48; this is an explicit policy change, not historical PASS rewriting.

Pre-fixed shadow rows [0,224,448,672] for each complete L0 up/down matrix, both paths: independently check every accepted step using raw-X FP64 and frozen v45 comparison tolerances. Shadow computations are after execution and separately timed; no shadow result influences selection. Full terminal checks cover every row, not every intermediate state.

1200s new-process hard cap includes startup, warmup, failure fixtures, diagnostics, C, both complete paths, terminal checks and shadow. Prior v47 two-matrix replay was149.94s; planning allowance: historical reconstruction/fixtures250s, matrices/C/control200s, fast100s, terminal/shadow350s, IO/setup100s =1000s (prediction only). No whole-model repair, held, PPL, model commit or deployable checkpoint. One timed run per path, control first; warm-cache/order bias and single-run uncertainty reported.
