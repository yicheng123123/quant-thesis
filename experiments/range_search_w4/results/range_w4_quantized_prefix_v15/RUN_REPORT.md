# v15 Run Report

## Reproduction

For a clean reproduction, first run `prepare_v15.py` once in an empty v15 output directory, then run the other stages from the repository root with the frozen `.venv-gptq` environment:

```powershell
.venv-gptq\Scripts\python.exe -B experiments\range_search_w4\results\range_w4_quantized_prefix_v15\prepare_v15.py
.venv-gptq\Scripts\python.exe -B experiments\range_search_w4\results\range_w4_quantized_prefix_v15\run_v15.py --stage preflight
.venv-gptq\Scripts\python.exe -B experiments\range_search_w4\results\range_w4_quantized_prefix_v15\run_v15.py --stage candidates
.venv-gptq\Scripts\python.exe -B experiments\range_search_w4\results\range_w4_quantized_prefix_v15\summarize_v15.py
```

`prepare_v15.py` refuses to replace existing frozen manifests. For the completed run, v14 rows and new article splits were frozen before preflight; all later stages read those saved manifests.

## Execution record

- Preflight: PASS. Eight native alpha=1 parity runs passed, four for each input regime.
- Runtime: GPTQModel 2.2.0, PyTorch 2.6.0+cu124, NVIDIA GeForce RTX 4060 Laptop GPU. Transformers emitted an SDPA/sliding-window warning; the frozen Qwen2 config has `use_sliding_window=false` (its `sliding_window` field is present but inactive), all 48 quantized-prefix forwards were finite, and alpha=1 parity passed.
- Candidate workload: 960 unique row-level candidates. The first candidate invocation wrote three complete FP_INPUT row records (30 candidates), then stopped on a Python variable-lifetime error before QPREFIX candidates. Resume verified all three raw records and candidate tensor files against hashes, reused those 30 slots, and computed the remaining 930. No partial candidate record was accepted.
- Candidate estimate frozen before search: 401.39 seconds. Synchronized candidate and search-loss time across all unique slots: 133.84 seconds. Held-out candidate evaluation: 0.66 seconds.
- Timed protocol components, including the successful native prefix quantization, FP/QPREFIX capture, Hessians, search, held-out evaluation, and cross-alpha diagnostics: 281.39 seconds. The sum excludes script startup, data/file serialization outside timed sections, and the failed software-debugging retries below.
- The prefix GPTQ pass was repeated three times while correcting runtime integration: GPTQModel 2.2.0 returned a single log dictionary rather than the anticipated two-value tuple; the freshly packed TorchQuantLinear needed its device-dependent helper tensors initialized before forward; the regular initialization path attempted `torch.compile/inductor`, which is unavailable because Triton is not installed in this Windows environment. The final run used GPTQModel's same eager `PackableQuantLinear.dequantize_weight` path with the required helper tensors initialized, then completed the full quantized-prefix forward and parity checks. Logs from all attempts are retained in `logs/`.
- No PPL, downstream task, self-method prefix, or additional model/bitwidth experiment was run.

## Final artifacts

`REPORT.md` contains results and interpretation. `correctness_checks.json` records final PASS status and source-tree integrity. Raw per-row records and candidate tensors are under `raw/`; the frozen protocol and manifests are beside this file. Summary CSV/JSON artifacts were regenerated from raw results, including direct held-out SSE recomputation from saved candidate weights and cached inputs.
