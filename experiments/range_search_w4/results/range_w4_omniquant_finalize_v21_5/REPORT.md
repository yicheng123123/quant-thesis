# v21.5 — M3 evaluation resume and baseline finalization

## Verdict

**M3_FORMAL_BASELINE_COMPLETED**  
Primary frozen-WT2 verdict: **M3_BEATS_M2_ON_FROZEN_WT2**. C4 fixed-budget points in the same direction; it remains secondary and does not set the primary verdict.

This is a method-level comparison. M1 is native GPTQ; M2 is GPTQ plus per-row discrete range search with native second-order compensation; M3 is OmniQuant with LWC gradient optimization and block reconstruction. M2 versus M3 does not isolate clipping from LWC as a single variable.

## Training and final artifact

- Blocks: 24/24; scheduled iterations: 61,440; successful optimizer steps: 61,438; native GradScaler skips: 2.
- Final block23 checkpoint hash/manifest checks: PASS.
- The frozen v21.4 runner raised `KeyError: 'manifest'` while assembling its summary after block23 was committed. The original failure is preserved; v21.5 reconstructs the summary from the committed checkpoint and logs, then verifies the dense export/reload.
- Dense-dequant export: 168 Linear modules replaced. Reload probe logits were bitwise equal; max absolute difference 0.
- This v21.5 pass performed zero optimizer steps and did not rerun either evaluation.
- The completed WT2 and C4 runs both loaded the final v21.4 dense export; model weight, model revision, frozen evaluator, and tokenizer revisions were hash checked.

## Independent evaluation aggregation

| Dataset | Scope | Windows | Scored targets | Total NLL | Mean NLL/token | PPL | Recompute difference vs evaluator |
|---|---|---:|---:|---:|---:|---:|---:|
| WT2 | frozen test | 582 | 299077 | 779405.562548637 | 2.606036447298 | 13.545256974944 | mean 0; PPL 0 |
| C4 | fixed-budget validation, not full C4 PPL | 253 | 131071 | 401799.654414415 | 3.065511474044 | 21.445427962301 | mean 0; PPL 0 |

The final windows and token totals agree with the frozen manifests. All required window indices are present and finite. Aggregation was independently recomputed from the persisted per-window CSV using total window NLL divided by total scored targets, followed by `exp(mean NLL)`.

## Frozen quality comparison

### WT2 primary

- M0 FP16: NLL 2.522797111231, PPL 12.463409495878.
- M1 native GPTQ: NLL 2.630763716261, PPL 13.884369572858.
- M2 GPTQ + range search: NLL 2.621806242375, PPL 13.760556051038.
- M3 OmniQuant: NLL 2.606036447298, PPL 13.545256974944.
- M3 vs M1: PPL delta -0.339112598 (-2.4424%), NLL/token delta -0.024727269.
- M3 vs M2: PPL delta -0.215299076 (-1.5646%), NLL/token delta -0.015769795.

### C4 secondary cross-corpus check

- M0/M1/M2/M3 PPL: 19.248555817998 / 22.233924450068 / 22.032883164065 / 21.445427962301.
- M3 vs M1: PPL delta -0.788496488 (-3.5464%), NLL/token delta -0.036107777.
- M3 vs M2: PPL delta -0.587455202 (-2.6663%), NLL/token delta -0.027024553.
- WT2 and C4 M3-vs-M2 directions agree. C4 is a 131,071-target fixed-budget validation, not standard/full C4 PPL.

## Recovery of native-GPTQ degradation

Using `(PPL_M1 - PPL_method) / (PPL_M1 - PPL_M0)`:

- WT2: M2 recovered 8.713% of M1-to-M0 PPL degradation; M3 recovered 23.865%.
- C4 fixed budget: M2 recovered 6.734%; M3 recovered 26.412%.

## AMP accounting

The canonical final checkpoint accounts for exactly two skips in 61,440 scheduled positions (skip fraction 0.0032552083%): L21/epoch0/sample0, scale 65,536→32,768; and L21/epoch0/sample74, scale 32,768→16,384. The v21.4 attempt log has 7,680 rows: 7,678 optimizer steps and two matching `found_inf` skips. Prefix blocks0–20 contribute 53,760 successful updates by historical inference from the v21.3 audit, because direct historical optimizer hooks/found_inf snapshots were not saved. An earlier interrupted L21/sample0 boundary row is disclosed separately and is not counted again in the canonical final-model schedule.

## Cost and measurement limits

- M1 calibration wall: 648.876101 s. M2: 2115.935334 s; M2/M1 = 3.2609×.
- M3's exact internal `perf_counter` calibration wall was not persisted. Timestamp-derived intervals are 2089 s for the v21.1 committed prefix and 963 s for the v21.4 continuation, sum 3052 s. This is not equivalent to the v18 wall measurement, so no M3/M1 or M3/M2 cost ratio is reported.
- The formal v21.4 training allocated peak was 1842344960 bytes; reserved peak was not persisted. Post-checkpoint reconstruction took 59.070 s and is excluded as post-hoc replay. Dense export/reload took 12.604 s and is reported separately.
- The first v21.1 checkpoint-fsync attempt executed 2,560 uncommitted updates and is excluded from final-artifact calibration cost. v21.2/v21.3 diagnostics, smoke runs, and debugging are also excluded.

## What this evidence supports

On this frozen Qwen2.5-0.5B-Instruct checkpoint and these two frozen evaluation streams, M3 completes a valid full-model OmniQuant baseline and has lower NLL/PPL than M1 and M2 on WT2 and C4 fixed-budget. It recovers more of the measured M1-to-M0 PPL degradation than M2 on these runs. It does not establish statistical significance, cross-model generalization, universal superiority, a causal advantage of one clipping mechanism, or inference speed gains. Cost superiority versus M2 is unresolved because exact M3 calibration wall time is unavailable.

## Reproduction and evidence files

This finalization used the frozen M3 raw JSON/window CSV and v18/v19 audited baseline artifacts; it ran no model forward, training, or evaluation. Re-run the standard-library finalizer from the repository root with:

```powershell
& 'E:\quant-thesis\.venv-gptq\Scripts\python.exe' -B 'experiments\range_search_w4\results\range_w4_omniquant_finalize_v21_5\finalize_v21_5.py'
```

The final checkpoint identity is in `FINAL_M3_ARTIFACT_AUDIT.json`; raw-data reaggregation is in `WT2_M3_FINAL.json` and `C4_M3_FINAL.json`; the machine-readable final table is `M0_M1_M2_M3_FINAL.csv`.
