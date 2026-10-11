# Local RTX 4060 preflight report

**Status: `LOCAL_M3_NOT_READY_USE_KAGGLE`.** The run stopped before model loading at the exact frozen model-file hash check. No optimizer steps ran.

- Calibration manifest identity: PASS (128 × 2048, seed 20261005, frozen order and SHA-256 match).
- Bundle/source and formal update count audit: PASS; the 61,440 update count is copied from v20.1 and its source hash matches.
- Runtime package/CUDA gate: PASS (native Windows, RTX 4060 Laptop GPU, 8 GiB class).
- Model identity: FAIL. `model.safetensors` and `generation_config.json` are absent from the exact snapshot; only a 500 MiB `.incomplete` model blob exists.
- Quant-off parity, LWC checks, representative 2048-token timing/memory, and local cost projection: NOT RUN.

No CPU/offload/sequence-length/algorithm fallback was applied. See `MODEL_HASH_DIAGNOSTIC.json`, `LOCAL_ENVIRONMENT.json`, and `LOCAL_COST_RECALC.json`. The measured T4 preflight is not used as a local RTX 4060 estimate.

## Reproduction command

```powershell
& .\.venv-gptq\Scripts\python.exe experiments\range_search_w4\results\range_w4_omniquant_full_v21\LOCAL_PREFLIGHT\run_local_preflight.py --bundle-root experiments\range_search_w4\results\range_w4_omniquant_kaggle_v20_1\kaggle_bundle --config experiments\range_search_w4\results\range_w4_omniquant_full_v21\LOCAL_PREFLIGHT\FROZEN_PREFLIGHT_CONFIG.json --output-dir experiments\range_search_w4\results\range_w4_omniquant_full_v21\LOCAL_PREFLIGHT --model-path C:\Users\gaoyi\.cache\huggingface\hub\models--Qwen--Qwen2.5-0.5B-Instruct\snapshots\7ae557604adf67be50417f59c2c2f167def9a775
```
