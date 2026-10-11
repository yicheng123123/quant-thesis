# v11 Stage 2 protocol amendment: complete S03–S09 follow-up

Amendment frozen before any v11 S03–S09 activation capture or candidate evaluation: 2026-09-29.

## Rationale

Stage 1 covered the preregistered layer/projection strata with 96 frozen rows and three frozen splits. Its row labels were 47 `Stage1 candidate-positive`, 48 `Stage1 mixed-risk`, and 1 `Stage1 neutral/negative`: 95/96 rows therefore require follow-up under the prior conditional plan. Screening out the lone neutral row would save only 7 × 10 = 70 candidate slots before legacy reuse, while conditioning the follow-up set on Stage 1 outcomes would complicate prevalence interpretation. This amendment includes all 96 rows so that no row is omitted based on Stage 1 performance.

This amendment was made before any S03–S09 run, from the Stage 1 class counts and measured compute cost. It is not a response to S03–S09 outcomes and does not drop poor-performing rows. The original frozen row IDs, Stage 1 artifacts, and S00–S02 results remain unchanged.

## Frozen scope and candidate accounting

- Frozen rows: 96; manifest SHA-256: `c56eeaa1d1af6e4ed8dd78f811c811c66758b215659a98bf8550052c786e6106`.
- Split source: original v9 manifest SHA-256 `34a24ef8602b8ce1e38e7a284f112acf7f766962d87d737f69148b78bcfaba78`; follow-up splits are exactly S03–S09.
- Completed before this amendment: 2,880 v11 candidate evaluations for S00–S02.
- Remaining nominal landscape: 96 × 7 × 10 = 6,720 candidates.
- Expected legacy reuse: 70 candidates for any exact matching v10 row/split; reuse is permitted only after article/token, activation, pristine-weight, candidate-grid, and candidate-weight hashes match. If an exact check fails, that candidate is new work and the new count may rise to 6,720.
- Expected new candidates after the 70-candidate reuse gate: 6,650; formal Stage 2 code records the verified existing/reusable/new counts before running candidates.
- Total full 10-split landscape: 9,600 row-split candidate evaluations, counting already completed and exactly reused candidates.

## Quantization protocol (unchanged)

Qwen/Qwen2.5-0.5B-Instruct revision `7ae557604adf67be50417f59c2c2f167def9a775`; FP16 model weights and FP32 Hessian/GPTQ/loss; W4A16 symmetric uniform quantization; G128; block size 128; damping 0.1; `desc_act=False`; alpha grid 0.1 through 1.0 in steps of 0.1. D is native dynamic GPTQ at alpha=1.0. C selects the lowest search-split output MSE among ten shared-alpha dynamic GPTQ runs. Every new candidate starts from pristine W and GPTQ state and retains normal compensation. Held-out inputs are evaluation-only and are not used to select alpha.

## Final benefit labels (frozen before S03–S09)

For each row, compute its ten held-out gains, median, minimum, positive/negative signs, and counts above +2%.

1. `Robust-positive`: at least 8/10 gains are greater than 0; median gain is greater than +5%; worst gain is at least −10%.
2. `Mixed-risk`: not robust-positive and either gains include both a positive and a negative value, or the worst gain is below −10%.
3. `Neutral`: not robust-positive or mixed-risk; absolute median gain is at most 2%; and no more than 2/10 gains exceed +2%.

These tie-ordered rules give mixed-risk precedence over neutral. Alpha stability is reported separately and is not part of the robust-positive criterion.

For alpha-band stability, report each row's selected-alpha histogram, mode, mode frequency, mean, median, min/max, and the largest count in any contiguous three-point grid band. The band flag is true only when at least 8/10 selections fall in that band.

## Cost estimate from completed Stage 1

The Stage 1 row-loop measured 2311.6 s for 1880.89372620012 newly evaluated candidates; the measured loop/candidate-time ratio was 1.229. Applying observed per-projection candidate times to the anticipated Stage 2 new-candidate counts gives 4388.8 candidate-seconds and approximately 5393.8 s row-loop wall time. A 1.5× planning margin gives approximately 8090.7 s (2.25 h). These are planning estimates, not measured Stage 2 results. Per-projection estimates: `{"q_proj":{"stage1_measured_mean_candidate_seconds":0.3199823123189518,"stage2_new_candidate_estimate":1610,"estimated_candidate_seconds":515.1715228335124},"o_proj":{"stage1_measured_mean_candidate_seconds":0.3296953831945252,"stage2_new_candidate_estimate":1680,"estimated_candidate_seconds":553.8882437668024},"up_proj":{"stage1_measured_mean_candidate_seconds":0.3324205205554992,"stage2_new_candidate_estimate":1680,"estimated_candidate_seconds":558.4664745332386},"down_proj":{"stage1_measured_mean_candidate_seconds":1.6435867777778135,"stage2_new_candidate_estimate":1680,"estimated_candidate_seconds":2761.2257866667264}}`.

## Output and preservation

New records belong under `raw/stage2/` with Stage 2 activation/Hessian caches under their own `stage2/` paths. S00–S02 JSON, candidate weights, CSVs, summaries, manifests, and reports are protected by `stage1_preservation_fingerprint.json`, captured before this amendment. No PPL, Stage 2 model-wide quantization, quantized-prefix evaluation, downstream tasks, or feature predictor is authorized in this stage.
