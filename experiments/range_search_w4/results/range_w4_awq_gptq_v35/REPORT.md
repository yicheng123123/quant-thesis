# v35 — L12 AWQ scaling selector × dynamic GPTQ bounded diagnostic

Status: **STOP — insufficient new disjoint whole articles**.

## Pre-execution result

- Frozen source: v34 protocol SHA256 `2b3b7d417fca04ec898fb890dd645953f694557f58efd38ef78b744f82b34662`; model/tokenizer revision `7ae557604adf67be50417f59c2c2f167def9a775`.
- Dataset: `Salesforce/wikitext:wikitext-2-raw-v1:train`, source parquet SHA256 `e83889baabc497075506f91975be5fac0d45c5290b6b20582c8cd1e853d0c9f7`; tokenizer has `add_special_tokens=false`, 2048 tokens/article.
- v24 frozen audit recorded 98 eligible articles after the shared v14/v15/v18 blacklist and selected 64 complete articles for v24; v31/v32/v33 reuse those same documents. Independent v35 scan after excluding those documents and all prior IDs/full/token hashes found **34** complete articles.
- Required: 64; shortfall: 30. No search/held split was assigned. Article-level identities and token hashes are in `NEW_ARTICLE_MANIFEST.json`; excluded article identities are in `HISTORICAL_EXCLUSION_LIST.csv`.
- `EVALUATION_SOURCE_EXCLUSION_AUDIT.json` records the frozen WT2-test and C4 evaluation source/split identities. v18 stores stream-level hashes rather than per-document test hashes; v35 only reads WT2 train. The C4 source is a separate corpus. No evaluation source is used for selection, and no claim of full-article cross-corpus hash comparison is made.
- The v24 arithmetic predicts 98-64=34 articles after additionally excluding its own 64; recomputation matches that count.

## Job-count audit

The independently generated `JOB_TABLE.csv` reconciles exactly with v34: **124 main full-Linear GPTQ executions**, **144 including 16 alpha=1 parity and 4 down-state isolation replays**, and **428,544 row-alpha score slots**. These are planned counts only. The 40 AWQ RTN ratios (20/split), 80 RTN Linear pseudo-quantizations, and 40 full-MLP selector forwards are separate from GPTQ counts.

## Stop decision

The frozen v34 data gate requires 64 newly eligible, unique whole articles, 16 each for S0-search, S0-held, S1-search, and S1-held. Only 34 remain. Following the protocol, execution stops before GPU initialization, correctness/model checks, scaling search, GPTQ, selection lock, or held evaluation. No split size, token length, exclusion list, target, or method was changed to fill the gap.

## Execution and status

- D=I/native parity, fold correctness, alpha=1 parity, candidate isolation, row stitch/commit, MLP propagation: **NOT_RUN**.
- Four-arm candidates and search lock: **NOT_RUN**.
- Held evaluation and quality gate: **NOT_RUN**.
- GPU cumulative time: **0/1800 seconds**; PPL/full-model quantization: **NOT_RUN**.
- Existing history/checkpoints/public implementations: read-only. New files are confined to v35.

## Reproduction

```powershell
.\.venv-gptq\Scripts\python.exe -B experiments\range_search_w4\results\range_w4_awq_gptq_v35\prepare_v35_preflight.py
```

This CPU-only command verifies source hashes and article exclusion, writes the immutable preflight manifests and job counts, and refuses to overwrite differing outputs. Since the frozen gate fails, there is no command to start the GPU experiment.
