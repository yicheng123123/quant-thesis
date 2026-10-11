# v36: L12 AWQ scaling selector + dynamic GPTQ (32 new articles)

## Outcome

**Stopped on correctness failure.** This is not a successful four-arm validation and it does not pass the v34 progression gate.

The protocol used the first 32 entries of v35's already-verified 34-article manifest in the frozen order: P0 ranks 0–7, P1 ranks 8–15, P2 ranks 16–23, P3 ranks 24–31. P0/P1 form S0; P2/P3 form S1. Ranks 32–33 were unused. Each group contains 8 complete articles × 2,048 tokens (16,384 token positions). The only protocol-scale change from v34 is 8 search + 8 held articles per split instead of 16 + 16. These two smaller splits do not provide the evidence strength of the original 16/16 plan.

The frozen model/protocol is Qwen2.5-0.5B-Instruct revision `7ae557604adf67be50417f59c2c2f167def9a775`, FP16, L12 complete MLP; symmetric W4/G128 dynamic GPTQ, damping 0.1, native error compensation, scalar alpha 0.1–1.0; official AWQ selector ratio grid 0–0.95 by 0.05. The four arms were G00 original/native, G01 original/scalar10, G10 selected D/native, and G11 the same D/scalar10. D and alpha are selected only on that split's search articles. This is the AWQ scaling selector joined to GPTQ; it is not complete AWQ and does not add AWQ clipping.

The frozen article and token hashes are in [ARTICLE_MANIFEST.json](manifests/ARTICLE_MANIFEST.json), group token-file hashes in [GROUP_TOKEN_CACHE_MANIFEST.json](manifests/GROUP_TOKEN_CACHE_MANIFEST.json), and method settings in [FROZEN_PROTOCOL.json](manifests/FROZEN_PROTOCOL.json). v35 remains STOP and is unchanged.

## Correctness and stop reason

S0 passed its D=I regression, gate/up native alpha=1 parity, down parameter/group parity, G01/G11 isolation replays, selected row stitch and actual module commit, and selection lock before P1 was opened. The four complete L12 MLP arms were evaluated on P1, with the actual L13 boundary captured.

There is one S0 parity-reporting gap: the G00 down native alpha=1 parity record omits the direct-row-output-score comparison. Its weight, scale, zero, g_idx, and group-entry comparisons passed. The other recorded S0 alpha=1 score comparisons passed.

S1 stopped at `D=I norm output differs from frozen prefix capture`, before its AWQ selector, any S1 GPTQ candidates, selection lock, or P3 held access. The failing check uses an exact tensor comparison in [run_v36.py:1007](run_v36.py:1007); the frozen threshold was not changed. A CPU-only replay after the stop found that the saved P2 pre/post-norm capture pairs are inconsistent with a Qwen2 RMSNorm replay: across its eight documents, max absolute differences range from 7.57 to 8.30 and mean absolute differences from 0.194 to 0.199. The corresponding P0 CPU replay is close (per-document max absolute difference 0.00195–0.00781). Base and prefix checkpoint norm weights compare equal. This narrows the failure to the P2 capture/recompute path but does not identify its underlying cause. See [STOP_DIAGNOSIS.json](correctness/STOP_DIAGNOSIS.json) and [S1_P2_capture.json](raw/S1_P2_capture.json).

A static audit found that the recorded FP16 fold-error reference is not the specified fold: [selected_fold_correctness in run_v36.py:1045](run_v36.py:1045) and [eval_held in run_v36.py:1117](run_v36.py:1117) divide the norm weight by D but leave pristine gate/up weights unchanged. The required fold also applies `W′=WD`. Therefore the stored S0 search/held fold-error values are invalid for the v34 fold-error gate. No replacement model forward was run after the stop. Projection-row tail CSV finalization requires both splits; the per-projection gate/up/down tail table is consequently unavailable. The saved S0 full-MLP output arrays do support CPU-only row and article summaries.

## S0 only: descriptive held results

These are full-MLP output MSE values over P1's 8 whole articles / 16,384 positions, recomputed from the saved FP16 outputs and common FP16 reference. They are partial diagnostics, not a valid overall v36 quality result.

| Arm | P1 full-MLP MSE |
|---|---:|
| G00 original + native GPTQ | 0.0246144712 |
| G01 original + scalar10 | 0.0251397523 |
| G10 selected D + native GPTQ | 0.0246392447 |
| G11 same D + scalar10 | 0.0251121095 |

| Paired contrast | Pooled loss change | Row W/T/L | Median row improvement | Worst row degradation |
|---|---:|---:|---:|---:|
| G10 vs G00 (scaling transfer) | **+0.101%** loss | 458/0/438 | +0.043% | +5.875% |
| G11 vs G10 (scalar search after scaling) | **+1.919%** loss | 69/0/827 | −1.935% | +13.024% |
| G11 vs G01 (combined vs existing scalar10) | **−0.110%** loss | 525/0/371 | +0.287% | +4.303% |

Lower loss is better; positive loss change means degradation. For G11 vs G10, 3 of 896 rows degrade by more than 10%; none exceeds 50%. For G11 vs G01, no row degrades by more than 10%. Article-level G11 vs G01 is 5 wins / 3 losses; article 295 contributes 57.6% of the net positive difference, and the net remains positive after removing it. Tokens within an article are not treated as independent replicates.

S0 selected AWQ ratio 0.30. In G01, alpha 0.9 was selected for 3,904/4,864 gate rows, 3,911/4,864 up rows, and 819/896 down rows. In G11, alpha 0.9 was selected for 4,351 gate, 4,340 up, and 824 down rows. Full histograms are in [S0_SELECTION_LOCK.json](final/S0_SELECTION_LOCK.json).

The frozen primary gate requires G11 to improve over G01 by at least 1% in **both** splits. S0's partial contrast improves by only 0.110%, so S0 does not meet the threshold. S1 was not evaluated. The fold-error gate is invalid for this run, and the projection tail gate is unavailable. Overall quality gate: **NOT PASS / correctness-stopped**. These S0 held figures do not support stable scaling transfer, an extra scalar-search benefit after scaling, or the combination gate.

Raw derived outputs: [partial analysis JSON](raw/PARTIAL_ANALYSIS.json), [per-row full-MLP MSE](raw/S0_FULL_MLP_ROW_HELD_PARTIAL.csv), [per-row paired contrasts](raw/S0_FULL_MLP_ROW_CONTRASTS_PARTIAL.csv), and [per-article results](raw/S0_ARTICLE_HELD_CONTRASTS.csv). The saved original held outputs are [S0_held_mlp_outputs.npz](raw/S0_held_mlp_outputs.npz). No S1 quality result or held oracle exists.

## Workload and cost

| Unit | Frozen plan | Actually executed before stop |
|---|---:|---:|
| Main full-Linear GPTQ candidate calls | 124 | 62 (all S0; 0 S1) |
| Independent alpha=1 parity GPTQ calls | 16 | 9 (8 passed records + 1 call whose first device comparison failed) |
| Down isolation replay GPTQ calls | 4 | 4, all placed in S0 after resume; 0 in S1 |
| Total full-Linear GPTQ executions | 144 | 75 |
| Main row-alpha score slots | 428,544 | 214,272 (S0) |
| Independent parity direct-score row slots | 46,080 planned | 22,144 completed; G00 down direct-score parity missing |
| Replay score row slots | 3,584 planned | 3,584 completed |
| AWQ RTN selector ratios / full MLP selector forwards | 40 planned | 20 in S0; 0 in S1 |

The failed S0/G00 down parity call did run native GPTQ; its CPU/CUDA equality comparison raised before the direct parity score. Recovery reran and passed the parity call, so it adds one full-Linear execution beyond the 144-call plan. Resume also placed the two S0 isolation replays twice in S0 rather than allocating the planned pair to each split. All such time is charged in [BUDGET_LEDGER.csv](raw/BUDGET_LEDGER.csv) and listed in [ATTEMPT_LOG.csv](raw/ATTEMPT_LOG.csv).

The cumulative experiment wall charged to the 1,800-second cap is **180.336 seconds** (including the 5.2-second external CUDA probe, failed launches, and final tail charge). Peak allocated memory was **5.04 GB**; peak reserved memory was **6.89 GB**. The recorded stage durations sum to 148.869 seconds; the difference is uninstrumented/runner overhead, including down-H construction before its damping-preparation timer. Do not interpret the stage sum as the end-to-end wall. Detailed counts and timings are in [COST_AUDIT.json](raw/COST_AUDIT.json) and [ACTUAL_EXECUTION_AUDIT.json](raw/ACTUAL_EXECUTION_AUDIT.json).

The v34 numeric quality thresholds were left unchanged; the reduced 8+8 article sample is explicitly lower-scale evidence. No PPL, full-model quantization, other layer, or extra candidate was run. No conclusion about model-wide benefit is supported.

## Answers

1. **Was the selector/scaling integration correct?** S0's transform, parity, stitching, full-MLP and boundary checks passed with the noted G00 down score-parity gap. Global correctness did not pass: S1 failed D=I. The fold-error implementation also does not measure the frozen full FP16 fold.
2. **Did RTN-selected scaling transfer to GPTQ?** In S0, G10 is 0.101% worse than G00. S1 is not run, so stable transfer is unverified.
3. **Did scalar10 add benefit after scaling?** In S0, G11 is 1.919% worse than G10 on the held full MLP. It does not show an extra benefit there.
4. **Did the combination pass the frozen gate?** No. S0's G11 vs G01 improvement is 0.110%, below 1%; S1 is missing. Stop the specific v34 combination; do not expand its grid or claim broader rejection of scaling methods.
5. **What remains unverified?** S1 D selection/candidates/lock/P3, both-split quality, projection tails, valid selected-D fold error, full-model quality/PPL, and any broader transfer.
