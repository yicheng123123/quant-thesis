# Reproduction record

All experiment artifacts and new code are isolated in this v36 directory. Historical v34/v35/v18 code, reports, checkpoints, and shared quantization implementations were read-only.

## Frozen protocol and data

The CPU preflight sequence used the repository's pinned local artifacts:

```powershell
Set-Location E:\quant-thesis
python experiments/range_search_w4/results/range_w4_awq_gptq_v36/prepare_v36.py
python experiments/range_search_w4/results/range_w4_awq_gptq_v36/split_v36_token_cache.py
python experiments/range_search_w4/results/range_w4_awq_gptq_v36/finalize_v36_freeze.py
```

`prepare_v36.py` re-tokenizes and verifies all 34 v35-eligible articles, then freezes ranks 0–31 into P0–P3 and leaves ranks 32–33 unused. The finalized inputs, job table, and source hashes are recorded in `manifests/` and `JOB_TABLE*`. The two preflight scripts do not load a model or initialize CUDA.

## GPU runner actually used

The correctness-first runner was invoked with the local frozen environment:

```powershell
Set-Location E:\quant-thesis
.\.venv-gptq\Scripts\python.exe -B experiments/range_search_w4/results/range_w4_awq_gptq_v36/run_v36.py
```

The command resumed from hash-checked v36 caches, executed S0, then stopped at S1 D=I with `CORRECTNESS_FAIL`. The cumulative charged runtime is 180.336 seconds against the 1,800-second cap. **Do not use a rerun of this command as a quality experiment:** correctness failed, the frozen stop rule was applied, and no tolerance was relaxed.

## CPU-only post-stop analysis

This command recomputes only summaries from the saved S0 FP16 output arrays and checks the saved norm-capture pairs on CPU. It performs no CUDA/model forward/GPTQ/PPL work:

```powershell
Set-Location E:\quant-thesis
$env:CUDA_VISIBLE_DEVICES = ''
.\.venv-gptq\Scripts\python.exe -B experiments/range_search_w4/results/range_w4_awq_gptq_v36/analyze_partial_v36.py
```

Outputs are `raw/PARTIAL_ANALYSIS.json`, `raw/ACTUAL_EXECUTION_AUDIT.json`, `raw/COST_AUDIT.json`, `raw/S0_FULL_MLP_ROW_HELD_PARTIAL.csv`, `raw/S0_FULL_MLP_ROW_CONTRASTS_PARTIAL.csv`, `raw/S0_ARTICLE_HELD_CONTRASTS.csv`, and `correctness/STOP_DIAGNOSIS.json`.

## Provenance and resume

`raw/RUN_STATE.json` contains the frozen hashes and final stopped state. `raw/BUDGET_LEDGER.csv` includes the device probe, failed launches, retries, stage timings, held evaluation, and final wall-tail charge. `raw/FAILED_ATTEMPTS.csv` and `raw/ATTEMPT_LOG.csv` distinguish saved work from retries. Candidate banks and captures are under `raw/`, `cache/`, and `checkpoints/`; none are inputs for a new quality claim without first resolving the correctness failure and freezing a new protocol.
