# v32 reproduction

From `E:/quant-thesis`, using the existing `.venv-gptq` and a clean v32 output copy:

```powershell
& '.venv-gptq/Scripts/python.exe' 'experiments/range_search_w4/results/range_w4_fixed_channel_scale_scalar_v32/prepare_v32.py'
& '.venv-gptq/Scripts/python.exe' 'experiments/range_search_w4/results/range_w4_fixed_channel_scale_scalar_v32/run_v32.py' preflight
& '.venv-gptq/Scripts/python.exe' 'experiments/range_search_w4/results/range_w4_fixed_channel_scale_scalar_v32/run_v32.py' formal
& '.venv-gptq/Scripts/python.exe' 'experiments/range_search_w4/results/range_w4_fixed_channel_scale_scalar_v32/check_scoring_v32.py'
& '.venv-gptq/Scripts/python.exe' 'experiments/range_search_w4/results/range_w4_fixed_channel_scale_scalar_v32/analyze_v32.py'
```

The protocol must be frozen before the first GPU command. Existing D/H/candidate/lock/held checkpoints are validated and reused on resume; cumulative wall is capped at 1200 seconds. Re-running completed GPU commands adds their real time to the cumulative budget, so use the saved outputs for analysis. No model forward, whole-model quantization, or PPL is invoked.

`raw/stitched/` contains only the four selected rows and quantizer tensors in scaled coordinates. It is a local diagnostic artifact and not a deployable checkpoint.
