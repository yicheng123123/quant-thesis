# v46 — M1 initialization plus v45 single-coordinate integer repair

## Result

Category A for repair migration: R1 improved on native M1 on both corpora. The scalar10 flow still adds a quality gain after repair: R2 is also better than R1 on both corpora. These are single-run descriptive comparisons, not statistical claims.

| Split | M1 PPL / NLL | M2 PPL / NLL | R1 PPL / NLL | R2 PPL / NLL | R1 vs M1 | R2 vs R1 | NLL interaction I |
|---|---:|---:|---:|---:|---:|---:|---:|
| WikiText-2 test | 13.884370 / 2.630764 | 13.760556 / 2.621806 | 13.662793 / 2.614676 | 13.553100 / 2.606615 | -1.596% | -0.803% | +0.000896 |
| C4 fixed-budget validation | 22.233924 / 3.101619 | 22.032883 / 3.092536 | 22.092247 / 3.095227 | 21.941981 / 3.088402 | -0.637% | -0.680% | +0.002258 |

R1 vs M1 PPL change: WikiText-2 -1.596%, C4 -0.637%. R2 vs M2: WikiText-2 -1.508%, C4 -0.413%. R2 vs R1: WikiText-2 -0.803%, C4 -0.680%. M2 vs M1: WikiText-2 -0.892%, C4 -0.904%.

Positive I on both splits means the scalar10 repair contrast is less favorable after repair on the NLL scale; R2 nevertheless remains below R1. This interaction is descriptive and does not establish significance or a mechanism.

## Coverage and numerical checks

Completed all 168 Linear modules and 304,128 output rows. Accepted updates: 9,521,779; changed coordinates: 9,310,879; candidate evaluations: 22,455,520,788. Aggregate search reconstruction MSE sum changed -12.629%.

V45 numeric flow recorded 9 FP32 non-descent candidates, all restored and rejected, plus 4 same-precision/independent-FP64 rechecks, all accepted after the frozen checks. No hard numerical or state error occurred in the corrected formal run. Final code/grid packing, reload, fixed-sample logits, and all 96 group hashes passed.

## Cost and execution history

R1's full v46 clock from original GPU_START to final status was 6842.9s (114.05 min) of the 10,800s cap. It includes preflight, the failed first dispatch, diagnosis/continuation, repair, save/reload, and both PPL evaluations. The 24 measured formal block walls sum to 5485.7s (91.43 min). Save/grid/reload/output regression was 11.04s; continuation checkpoint reload 1.25s; WT2+C4 evaluation 131.32s. Nested numeric-recheck timing is listed separately in COST_COMPARISON.csv and is not additive to block wall.

Historical M1 construction was 648.9s and M2 scalar10 construction 2115.9s, a measured 1467.1s (24.45 min) difference. R2's complete v42-v45 validation chain cost 7064.8s; plus M2 construction the lineage total is 9180.7s. That chain includes pilot, failure/diagnostics, recovered work, save/reload, and PPL. The unique clean repair-only cost across restored R2 groups is not reliably reconstructable; v45 separately reports 4717.0s for its 141 newly repaired modules, excluding 27 inherited modules. Do not read R1/R2 wall totals as an acceleration comparison.

v40's already-known winner union measured 1135.0s for 675 full-Linear candidates versus 2255.7s for its 1,680-candidate scalar10 arm. This is an engineering reference for executing a known union, not the cost of discovering it.

The first v46 launcher resolved the legacy V42 function and failed at an expected non-descent. Its status and 14 invalid group files remain archived/excluded. The corrected run started from the original M1 and completed all groups with the byte-identical v45 function. After save/reload, the original runner hit `KeyError('v18')` before scoring any PPL window. The separate continuation verified historical manifest hashes from BASELINE_AUDIT and ran the unchanged evaluator once per corpus. RUN_STATUS.json and FAILED_ATTEMPTS.jsonl preserve the failed invocation; FINAL_STATUS.json and FINAL_DECISION.json describe the completed chain.

## Sources and limits

See PROTOCOL.json, SOURCE_AUDIT.json, BASELINE_AUDIT.json, PPL_SOURCE_AUDIT.json, FOUR_MODEL_COMPARISON.csv, MODULE_RESULTS.csv, MODULE_AGGREGATES.csv, NUMERICAL_RECHECKS.csv, REJECTION_EVENTS.csv, COST_COMPARISON.csv, CHECKPOINT_LOCK.json, DEPLOYMENT_LOCK.json, and FINAL_STATUS.json. P3 was not read. No PPL-based tuning or second model was performed. Interface state could not confirm the requested Luna Max selection. R1 is an application of existing discrete repair ideas, not an official ReQuant reproduction or an originality claim.
