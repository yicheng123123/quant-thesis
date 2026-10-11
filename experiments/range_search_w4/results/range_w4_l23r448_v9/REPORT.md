# L23 R448 W4 dynamic GPTQ: article split stability

## Frozen scope

Qwen/Qwen2.5-0.5B-Instruct at revision `7ae557604adf67be50417f59c2c2f167def9a775`; layer 23 `self_attn.q_proj` row 448 only. FP16 model weight, FP32 Hessian/GPTQ/output error, symmetric W4A16 G128, 128-column blocks, damping 0.1, `desc_act=False`. Inputs come from the FP model. Candidate α is shared across all seven groups; each candidate reruns the v8 parity-validated dynamic GPTQ compensation from pristine W. D is α=1, and C is the ten-point search-MSE minimizer. No test split, PPL, other rows, or model-level inference was used.

The frozen [split manifest](splits_manifest.json) has SHA-256 `34a24ef8602b8ce1e38e7a284f112acf7f766962d87d737f69148b78bcfaba78`. The pool is the next 80 eligible WikiText-2 train articles after the eight v8 reference articles, not a random sample of the entire corpus. The 80 articles are globally disjoint across v9 splits and disjoint from v8. Every split has four search and four held-out full articles, with the first 128 tokens per article. The manifest records source IDs, article/token hashes, token IDs and deterministic sub-seeds; it was written before any α candidate run.

## Per-split D versus search-selected C

| Split | Selected α | Search gain | Held-out gain | D held-out MSE | C held-out MSE |
| --- | ---: | ---: | ---: | ---: | ---: |
| S00 | 0.7 | 47.52% | 34.69% | 0.0083595 | 0.00545981 |
| S01 | 0.7 | 43.37% | 46.32% | 0.0076108 | 0.00408513 |
| S02 | 0.6 | 35.46% | 20.20% | 0.00806111 | 0.00643269 |
| S03 | 0.8 | 35.73% | 38.30% | 0.00794964 | 0.00490455 |
| S04 | 0.6 | 31.00% | 44.03% | 0.00775109 | 0.00433798 |
| S05 | 0.7 | 39.66% | 29.80% | 0.00497825 | 0.00349464 |
| S06 | 0.6 | 51.99% | 23.48% | 0.0078167 | 0.00598161 |
| S07 | 0.8 | 32.59% | 42.35% | 0.00883371 | 0.0050922 |
| S08 | 0.7 | 49.33% | 56.45% | 0.00877488 | 0.00382182 |
| S09 | 0.6 | 50.20% | 51.36% | 0.0108039 | 0.00525523 |

Held-out gains: 10/10 positive, 0 ties, 0 regressions under relative tolerance 0.0001. Median 40.33%, mean 38.70%, p25–p75 31.02–45.75%, min–max 20.20–56.45%. There is no own-split held-out regression; the weakest own split is S02 at 20.20% gain. Search gain is nonnegative by construction because α=1 is in the candidate set; it is not generalization evidence.

## α stability and the v8 reference

α=0.6: 4/10, α=0.7: 4/10, α=0.8: 2/10.

Modes tie at α=0.6 and α=0.7, each 4/10; exact selection agreement is 40%. Mean α 0.68, median 0.7, range 0.6–0.8. All ten selections lie in the adjacent 0.6–0.8 band, with eight in 0.6–0.7. This supports a stable *band*, not a uniquely stable α.

The separate v8 reference selected α=0.7: D held-out MSE 0.0075483983, C 0.0054758275, gain 27.46%. Only 2/10 v9 gains are at or below it. Thus v8's ≈27% is below the v9 median, not an upper-tail success. Different train articles and Hessians mean the absolute MSE values should not be treated as paired across v8/v9.

## Fixed α and cross-split transfer

The α=0.7 value was already selected in v8, before the new disjoint splits: fixed α=0.7 beats α=1 on 10/10 own held-out sets, with pooled SSE gain 38.59%. In the post hoc full grid, fixed α=0.8 has pooled gain 40.07% and 10/10 wins; its slight edge is descriptive and must not be used to retune the frozen method on held-out data.

Transfer cell `M[i,j]` uses the quantized row built with search split *i* and its search-selected α, tested on held-out split *j*. Its denominator uses the **same source split i GPTQ state** with α=1 on that target held-out set. This measures transfer of the source-calibrated quantized row, including its Hessian; it is not an α-only intervention with target-specific recalibration.

Transfer diagonal median 40.33%; off-diagonal median 36.33%, mean 34.56%. Off-diagonal outcomes are 88/90 positive and 2/90 regressions. Both regressions are small: S02 (α=0.6) → S06: -0.46%; S02 (α=0.6) → S08: -0.60%. Mean off-diagonal gains by source-selected α are 0.6: 27.98%, 0.7: 39.80%, 0.8: 37.25%. The 90 cells reuse ten source weight states and ten held-out sets, so they are correlated diagnostics, not 90 independent replications. For the four source splits selecting α=0.7, all 36 off-diagonal cells improve over their source-matched D.

## Search predictiveness and activation shift

Across ten splits, search gain versus held-out gain has Pearson r=0.147 and tie-aware Spearman ρ=0.139. Search-gain magnitude does not rank held-out-gain magnitude well in this small set; no significance claim is made.

No own held-out split regresses, so there is no own-regression subgroup to explain. S02 has the weakest own gain and is the source of both slight transfer regressions; its search/held-out Gram relative change is 64.6%, while the ten-split median is 54.5%. Gram shift versus held-out gain has descriptive Pearson r=-0.377. Global RMS, per-feature RMS, mean/std and percentile changes are saved in `activation_split_diagnostics.csv`. These observations do not establish a causal explanation.

## Gate, cost and limits

The requested diagnostic rubric classifies this as **strong_stable_local_signal** for L23 R448: 10/10 own held-out wins, median gain above 5%, no own regression, selected α in a narrow adjacent band, and the previously observed fixed α=0.7 has positive pooled gain. Exact α agreement remains only 40%, and two off-diagonal transfers are slightly negative. This is local FP-input evidence, not a statistical significance test or a claim about other rows or full-model quality.

Measured work: 100 full dynamic GPTQ candidate evaluations, ten native-timing Hessian builds, 80 FP-model forwards. Model capture took 3.84s (peak allocated CUDA 996.5 MiB); candidate evaluations took 20.98s, Hessians 0.27s. Full invocation took 25.69s. These are offline calibration costs, not inference speedups.

All 1100 candidate/search and cross-held-out losses were recomputed directly from cached FP inputs and saved candidate weights; max relative difference 2.37e-07. `correctness_checks.json` is PASS. The full landscape, transfer cells and split metrics remain in raw per-split JSON plus CSV; cached activations and candidate weights allow independent rechecking.

The result supports a stable clipping preference for **this row under FP projection inputs and this article pool**. It does not show that the same α helps other rows, quantized-prefix inputs, or final PPL. No experiment beyond the requested one-row multi-split diagnostic was launched.

## Reproduce

From `E:\quant-thesis` with the existing `.venv-gptq`:

```powershell
$env:PYTHONPATH='E:\quant-thesis'
.\.venv-gptq\Scripts\python.exe -m experiments.range_search_w4.prepare_v9_splits
.\.venv-gptq\Scripts\python.exe -m experiments.range_search_w4.run_l23r448_v9
.\.venv-gptq\Scripts\python.exe -m experiments.range_search_w4.summarize_l23r448_v9
```

The preparation command verifies an existing manifest byte-for-byte instead of replacing it. The run command resumes complete split JSON/weight pairs; the summarizer rebuilds all tables and this report from frozen inputs and raw results.
