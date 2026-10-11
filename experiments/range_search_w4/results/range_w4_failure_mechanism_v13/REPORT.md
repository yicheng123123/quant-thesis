# v13 Failure Mechanism Audit

## Scope and provenance

This is a cached-data audit of v12 Gate v1. It invokes no GPTQ quantization, creates no new candidate, and performs no PPL or selector experiment. The frozen Gate, row and article manifests, raw JSON, candidate tensors, pristine weights, and FP projection inputs are read from `../range_w4_gate_v12/`. Source hashes, definitions, and tolerances are recorded in `PROTOCOL.json`. The original v11 full10 study is the Gate's discovery source; v13 analyzes the independent v12 validation only. Thus all associations below are **post-outcome diagnostics**, not prospective selection evidence.

Reproduce from the repository root in PowerShell:

```powershell
& '.venv-gptq\Scripts\python.exe' 'experiments/range_search_w4/results/range_w4_failure_mechanism_v13/analyze_v13.py'
& '.venv-gptq\Scripts\python.exe' 'experiments/range_search_w4/results/range_w4_failure_mechanism_v13/summarize_v13.py'
```

The first command independently rechecks v12 summary statistics from 480 raw row/split JSON records, then validates two representative cases before analyzing all cases. It checks all cached candidate tensor hashes and preserves the frozen source files. `preflight_checks.json` also compares the low-rank Gram norm formula to explicit FP64 Gram matrices in two cases (absolute differences below `1.7e-9`). `quadratic_identity_checks.json` reports PASS for 480 cases, 1,920 raw-MSE comparisons, 1,920 direct-projection versus quadratic comparisons, 960 directional side-difference comparisons, and 480 shift identities. Maximum absolute numerical discrepancy is `7.45e-9`; tolerance is `1e-8 + 1e-5 × |reference|`. Frozen `GATE_V1.json` SHA-256 remains `7943b8f26e5f94456b7b00144f6d47fc775d0224ccca4a6a1eff38e5740999c9`. The final analyzer invocation took 6.13 s inside the script, including cache load; this is offline audit cost, separate from v12 calibration. No inference speed claim follows.

## Independent v12 FAIL audit

- C pooled held-out gain against native dynamic GPTQ D: **10.989%**; Gate: **2.617%**. Gate retained **23.82%** of C's pooled gain.
- C severe regressions below −10%: **69/480**; Gate: **40/480**. The reduction is **42.03%**, below the frozen 50% requirement. Gate accepts 25/48 rows; 40 severe and 10/20 catastrophic C cases remain in accepted rows.
- C catastrophic regressions below −40%: **20/480**. Worst is `L20_o_proj_R287/S08`, C=Gate **−412.09%**.
- Gate's positive pooled gain passes its second criterion, but severe reduction and retained gain fail. Overall v12 status remains **GATE_V1_VALIDATION_FAIL**. These numbers are reconstructed from raw SSE and the frozen row-level decision rule; they are not copied as conclusions from `VALIDATION_SUMMARY.json`.

## Quadratic identity and mechanism

For each case the cached FP32 pristine row `w`, FP16 candidate weights converted to FP32, and cached 512-token FP32 inputs define `e_D = w_D − w`, `e_C = w_C − w`, and `H = XᵀX/512`. Both contraction orders—`mean((Xe)²)` and `eᵀHe` without materializing H—agree with v12 output MSE. Define `A_s=q_D,s−q_C,s` and `A_h=q_D,h−q_C,h`. Every case satisfies

`A_h−A_s = e_Dᵀ(H_h−H_s)e_D − e_Cᵀ(H_h−H_s)e_C`.

This identity isolates the observed failure: a positive search advantage can become negative when held-out inputs amplify C's error direction more than D's. It is an **exact retrospective decomposition** of the observed losses. The directional terms use held-out inputs and cannot serve as an available search-time selector.

The global relative Gram shift `||H_h−H_s||_F/||H_s||_F` does not distinguish outcomes well: its median is 0.536 in the 241 cases with held-out gain ≥10%, 0.577 in the 69 severe cases, and 0.484 in the 20 catastrophic cases; descriptive Spearman with held-out gain is −0.052. In 20 deterministic layer/projection-matched failure–success pairs, the respective median global shifts are 0.484 and 0.484. By contrast, median C-direction shifts are 0.00290 versus 0.000598 in those pairs. The directional difference `shift_D−shift_C` correlates strongly with held-out gain (Spearman 0.874), but that correlation is partly algebraic because it is a component of the outcome identity, not independent predictive validation.

The same `L20/o_proj/S08` input matrices are used for four rows. `R287` regresses −412.09%, while `R140`, `R639`, and `R739` gain 37.53%, 62.35%, and 73.51%. Their common global Gram shift is 0.382. Different quantization error directions under the same input geometry produce very different outcomes; global shift alone is insufficient.

## Worst case: L20_o_proj_R287/S08

Search chose α=0.7. Search output MSE fell from `1.39364e-4` (D) to `9.90719e-5` (C), a 28.91% gain. Held-out MSE rose from `5.53481e-4` to `2.83430e-3`, a −412.09% gain. The search advantage was `+4.02922e-5`; held-out advantage was `−2.28081e-3`; generalization shift was `−2.32111e-3`. C's directional input shift (`+2.73522e-3`) exceeded D's (`+4.14117e-4`) by `2.32111e-3`.

All four search articles improved under α=0.7 (individual gains 29.40%, 7.83%, 34.74%, 41.38%); their ten-alpha rankings have mean pairwise Spearman 0.970, and three articles choose the aggregate best alpha. The failure therefore cannot be explained by one search article forcing an otherwise rejected alpha. A search-only worst-article objective would choose α=0.8 for this case, but its held-out consequence is **not** assessed as a new method here.

On held-out inputs, the second article (source article 105) contributes 75.13% of C SSE versus 19.49% of D SSE. Its C and D SSE are `1.09018` and `0.05523`; it accounts for 88.63% of the total C−D excess SSE. This is article-level concentration rather than a single-token spike: C's top 1% of held-out tokens account for 14.01% of SSE, slightly below D's 15.10%. These observations locate the realized regression; they do not establish why that article's activation geometry differs.

## Article ranking and token concentration across cases

Search article rank agreement is high in both outcome groups. Mean pairwise ten-alpha Spearman has median 0.970 in robust successes (held-out gain ≥10%), 0.976 in severe regressions, and 0.972 in catastrophic regressions. No case in either group has mean pairwise Spearman below 0.5. The selected alpha improves at least three search articles in 207/241 robust successes, 60/69 severe cases, and 17/20 catastrophic cases. Thus search article ranking instability is **not supported as the main failure mechanism** in this cache. The worst case is an explicit counterexample to a simple `k/4 articles improve` safeguard.

One or more search articles still regress for 124/241 robust successes and 35/69 severe cases, so heterogeneous search articles exist, but not predominantly among failures. The median share of positive search-side improvement supplied by the best single article is 0.499 in robust successes and 0.495 in severe cases; ≥75% occurs in 19/241 and 5/69 respectively. Aggregate search selection is not generally dominated by one article in the severe group.

Held-out C token concentration at the top 1% has nearly identical medians: 11.3% in robust successes, 11.5% in severe cases, 11.8% in catastrophic cases. The top article's median C SSE share is 31.5%, 36.7%, and 46.1% respectively. In the 20 matched pairs, catastrophic failures have median top-article share 46.1% versus 32.6% in matched successes (15/20 failures higher). Article concentration is associated with failures; a few individual tokens are not a general explanation. The worst case is more concentrated than the group median.

## Why Gate v1 failed, and what remains uncertain

The Gate checks the median search gain and top1/top2 search margin for each row. Those quantities measure local search preference, not how the chosen error vector is weighted by unseen inputs. In v12 they allowed 40 severe regressions while giving up most of C's pooled gain. Search signal is **not simply weak**: median search gain is 18.10% in severe cases versus 17.84% in robust successes; the worst case improves 28.91% on search data and all four articles support clipping. Its failure is exactly accounted for by held-out directional amplification. Across cases, high article rank stability and similar global Gram shifts in matched successes/failures argue against unstable search rankings or a global covariance norm as the primary explanation.

This does not prove a unique causal source. Candidate error direction, article composition, and activations are jointly involved; cached observations do not reveal whether a larger calibration set would have sampled the missing direction often enough, nor whether any search-only robust objective would generalize. The 480 row/split cases share rows, article splits, and projection inputs; descriptive correlations and counts are not independent-sample significance tests. Held-out directional quantities and token concentration are outcome-aware diagnostics.

## Next research route: larger calibration coverage, not a v13 method result

Among Gate v2, robust search objective, larger calibration set, and stop, **larger calibration coverage is the most defensible future test** if this local direction is continued. The failure evidence is unseen directional sensitivity, including a held-out article that dominates the worst regression while all search articles agree. More independently sampled full articles can test whether the selected alpha stabilizes when that direction appears in search. This is a research hypothesis, not a validated fix, and its calibration cost must be budgeted before execution. This v13 audit does **not** authorize or run that experiment.

A robust pure-search objective (mean+λ·std, worst article, upper-tail/CVaR, or `k/4` support) is less directly supported by the current evidence. It might change some choices, but ranking is already stable across four search articles and the worst case improves on all four. Gate v2 threshold tuning on v12 held-out would contaminate validation and is excluded. Stopping the method direction remains reasonable if a separately frozen coverage experiment fails to reduce severe regressions at acceptable cost. No v12 held-out value was used to fit λ, k, a threshold, or a selector.

## Artifacts and limits

`quadratic_decomposition.csv` gives each case's errors and advantage decomposition; `quadratic_identity_checks.json` gives correctness evidence. `catastrophic_cases.csv` contains all 20 cases below −40%; `matched_cases.csv` records the predeclared deterministic same-layer/projection comparison. `article_level_landscapes.csv` contains 4,800 ten-alpha search landscape rows, and `article_ranking_stability.csv` gives case-level ranks. `directional_shift_analysis.csv`, `token_error_concentration.csv`, `feature_outcome_summary.csv`, and `analysis_summary.json` provide the descriptive diagnostics. The analysis scripts and `PROTOCOL.json` specify exact replay steps and source hashes.

No final-model quality, PPL transfer, method advantage, causal selector, or originality claim follows from v13. **STOP after this audit.**
