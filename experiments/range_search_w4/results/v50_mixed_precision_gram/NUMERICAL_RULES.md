# v50 frozen mixed-Gram numerical workflow

Frozen before the first GPU comparison. Only one new construction path is tested. No constants depend on observed C64 errors, historical failed candidates, held/PPL, or comparison results. Sol high is requested; the interface cannot verify/switch the model setting.

## Construction and assumptions

X is the exact stored FP16 calibration input from locked v49 block0. N=262144, token chunk=4096, TF32 off, float32 matmul precision highest, autocast disabled. The baseline is the unchanged v49 streamed FP64 builder. The new builder computes each full xx.T@xx in FP32, converts that result to FP64, accumulates in FP64, divides by the same N and symmetrizes. Casting to FP64 does not recover lost block precision. All off-diagonal and cross-G128 terms are retained.

The new builder also accumulates each column's sum of squares in FP64, in the same input stream. Define gamma_p(k)=k*u_p/(1-k*u_p), u32=2^-24, u64=2^-53, K=ceil(N/4096), g64=gamma64(N+K+8), h_i=sqrt(s_i/(1-g64)), where s_i is the computed column mean square. Set rho=gamma32(min(N,4096))+8*g64*(1+gamma32(min(N,4096))). This statistic and its cost belong to the independent production path; no FP64 Gram is needed.

Under ordinary FP32 rounded accumulation with at most 4096 terms and no overflow/underflow, block Gram error is bounded by gamma32(4096)*sum(abs(x_i*x_j)). Products of finite FP16 inputs are representable in FP32, and Cauchy-Schwarz bounds the absolute-product sum/N by h_i*h_j. The FP64 proxy includes energy, chunk accumulation, division and symmetrization. These assumptions are not a verified theorem about every CUDA kernel/reduction path. rho*h_i*h_j is therefore used as an assumption-based empirical numerical uncertainty workflow, not a safety certificate or theoretical monotonicity guarantee. A validation failure is not grounds to change rho.

## Search and acceptance

Call the original v48 batch_repair, v45 choose/propose/score_and_retire/retire_caps and deterministic coordinate/-1/+1 order. Output chunk256, code[0,15], zero8, G128, actual FP16 grid delta, 32 accepts, 128*d evaluations, rejection blacklist and initial FP32 search threshold max(loss*1e-8,1e-14) remain unchanged. The initial old FP32/quadratic regression remains an extra diagnostic; it is not claimed to cover new Gram error.

For a single selected coordinate j, augment the original v48 envelope by
U_G = rho*(2*abs(delta)*h_j*sum(abs(e_i)*h_i) + delta^2*h_j^2).
The current e and independent h determine this term; it does not use C64. Clearly negative prediction below -(U48+U_G) follows v48's quadratic acceptance. Otherwise retain v48 cache check/refresh and invoke independent raw-X FP64 increment a=Xe, b=Xdelta, mean(2ab+b^2).

The new verifier keeps the old same-C FP64 cache and selected/full-formula checks. Approximate-C prediction versus raw-X increment must satisfy abs(pred-direct) <= raw_rounding_allowance+U_G. The acceptance sign test uses only direct < -raw_rounding_allowance, not the Gram comparison allowance. Failure of the predicted/raw comparison, same-C formula/cache check, finite/code/grid checks is a hard correctness error. A non-descending raw candidate is rejected/restored/blacklisted by the unchanged v48 flow. No budget reset or fallback to the original C64 construction.

At the original 1/8/16/24/32 acceptance milestones, refresh/check p=E@C and B with unchanged v48 rules. New Gram error does not excuse stale p or corrupted bookkeeping.

## Independent validation and stopping

All output rows of both matrices/paths get independent raw-X FP64 initial and final reconstruction. For approximate C, the terminal quadratic/raw comparison adds rho*(sum(abs(e)*h))^2 to the original terminal rounding criterion. Final quadratic loss bookkeeping must still pass the original same-C criterion. Every row's final direct loss must be <= its own initial direct loss. Confirmed ascent, nonfinite, illegal code, grid/source pollution or unexplained tolerance failure stops the experiment.

Before running, fix shadow rows [0,224,448,672] for both projections and both paths. Check every accepted update on these rows by independent raw-X FP64 increment; require direct increment below its raw rounding allowance, and prediction discrepancy within raw allowance plus the applicable Gram uncertainty. This is sampled full-trajectory evidence, not all-row stepwise proof. The baseline C64 may be reused only for validation after the mixed repair has finished; it never participates in mixed selection/thresholds. Shadow and C-error validation are separately timed.

One timed run per path, baseline first; no repeated stability claim. The 1200-second cumulative GPU wall cap includes warmup, input replay, failures, checks and IO. No model submission/checkpoint, whole-model repair, held or PPL is performed. LOCAL evidence files are matrix states only.
