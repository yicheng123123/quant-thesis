# v17.1 Native GPTQ parity audit

Verdict: **M2_NATIVE_PARITY_RESTORED**. The audit stopped after the five-level parity ladder. No range-search grid, selected-weight commit, self-prefix experiment, OmniQuant, or PPL was run.

1. The first numerical divergence is **the FP32 diagonal mean reduction before damping**, not a range or compensation operation. Native computes `torch.mean(torch.diag(H))`; v17 computes `H.diagonal().mean()`. The diagonal values agree bitwise, but the native reduction input is contiguous and the other is a stride-897 view. The means are 0.028906324878334999 and 0.02890632301568985, differing by 1.862645149230957e-09. Exact float bit patterns and full tensors are saved in FIRST_DIVERGENCE.json / FIRST_DIVERGENCE_TENSORS.pt.

2. The mismatch is upstream H/Hinv preprocessing. The external v17 factor hash was reproduced exactly. Capturing the actual native factor showed it was already different before group0. The original starting-state check therefore fails; a compensation comparison under that unequal starting state was stopped. HINV_AUDIT.json records the damping, lower Cholesky, inverse and upper-Cholesky comparisons.

3. Group0 range/scale comes from W, independently of Hinv, so it can agree while the divisor and compensation use a different factor. These small update changes alter the group1 current weights and therefore subsequent dynamic ranges.

4. Rounding decisions and repeated compensation turn initial FP32 differences into some different final FP16 weights. Under a captured common native Hinv, the group0 trace compares 903 tensor operations; all exact: True. This controlled check removes the initial factor mismatch before testing the quantizer/error/product/update path.

5. Native-vs-native five-run bitwise repeatability: True.

6. The original custom path's five-run bitwise repeatability: True; the restored native-core path: True. These controls provide no evidence that backend nondeterminism explains this failure. No tolerance amendment is proposed.

7. The implementation mismatch is repairable. Controlled variants show that keeping the native mean but changing the diagonal write syntax reproduces native H/Hinv, whereas retaining the view reduction does not. The new isolated `native_m2.native_dynamic_gptq` accepts pristine FP16 W plus undamped FP32 H and runs the actual native core. Alpha=1 has no range modification. Other alpha values are an implemented range-hook branch, unexecuted in this alpha=1-only audit.

8. The restored interface meets exact weight/scale/zero/output-MSE parity at levels 1-5: reported failing group4/row17 (plus group0 control), complete row17, first32 original smoke rows, fixed k/v/q/o first-middle-last controls, and complete original k_proj. PARITY_LEVEL1..5.json contain exact checks. All native-core calls preserve the original full projection geometry; row subsets are taken from outputs. This audit does not qualify an independently sliced row-batch implementation for a later search.

9. No unresolved mismatch is attributed to CUDA. Runtime TF32, autocast, deterministic settings and matmul precision are recorded. All source hashes and runtime dtype/storage/view observations are available in SOURCE_MAP, DTYPE_TIMELINE and operator trace artifacts. The historical source and v17 results are unchanged.

10. One additional implementation issue was found: the old v17 smoke used the **raw decoder-entry embedding input** as k_proj calibration input. Native k_proj receives RMSNorm output. Both sides used the same old input, so this does not explain parity failure. It must be corrected when re-entering v17 sequential smoke. Root-cause levels 1-3/5 deliberately retain the old input for exact reproduction; level4 captures real projection inputs from two native L0-only forwards.

Reproduce from the repository root with `.\.venv-gptq\Scripts\python.exe experiments/range_search_w4/results/range_w4_m2_parity_v17_1/upstream_probe.py`, then the same Python executable with `run_audit.py`. Read the completed artifacts before rerunning; these commands are diagnostic only. The v17 smoke launcher remains frozen and still imports the old core. A future explicitly requested smoke must import the new interface and pass undamped H; this audit does not resume that smoke.

## Final artifact checks

Original full128-row failure reproduced against v17: True. Original FP16 mismatches: 317; scale mismatches: 692. The row17/col513 values match the saved v17 diagnostic. ORIGINAL_FAILURE_REPRODUCTION.json includes group entry/scale checks and the two original local MSEs.

Correcting damping alone yields exact full128-row weight and scale, while the separate legacy32-row execution still differs at 24 scale elements (max 7.450580596923828e-09). The restored interface retains full native projection geometry and does not qualify independent row slicing.

Reported failing group4/row17 Level1: True; group0 control: True; g_idx and boundaries: True. Native/custom trace tensor coverage matches in both directions: True. v16 and recorded source checksum checks pass: True. No pre-turn hash manifest exists for v17 result files; they were read only in this audit. Final verdict after every required check: **M2_NATIVE_PARITY_RESTORED**. STOP.

Supplemental verification command: `.\.venv-gptq\Scripts\python.exe experiments/range_search_w4/results/range_w4_m2_parity_v17_1/finish_checks.py`. It runs two bounded alpha=1 calls on the same k_proj and validates existing artifacts; it does not resume any experiment.
