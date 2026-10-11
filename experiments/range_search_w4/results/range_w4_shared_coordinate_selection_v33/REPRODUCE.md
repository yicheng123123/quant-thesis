# v33 reproduction

From `E:/quant-thesis`, with the existing `.venv-gptq` CPU-capable environment:

```powershell
& '.venv-gptq/Scripts/python.exe' 'experiments/range_search_w4/results/range_w4_shared_coordinate_selection_v33/prepare_v33.py'
& '.venv-gptq/Scripts/python.exe' 'experiments/range_search_w4/results/range_w4_shared_coordinate_selection_v33/run_v33.py' freeze-selection
& '.venv-gptq/Scripts/python.exe' 'experiments/range_search_w4/results/range_w4_shared_coordinate_selection_v33/run_v33.py' analyze-held
```

The selection phase reads the frozen v24 original scalar10 search scores, the v32 scaled-coordinate search score table, search locks, and selected candidate tensors/parameters. It writes `SEARCH_SELECTION_LOCK.json` before the held phase is allowed to open any held-score source. The held phase verifies that lock, then reads v32's saved per-target held results and the frozen v24 epsilon table. No GPU, model forward, GPTQ, PPL, or whole-model quantization runs. The combined CPU budget is 300 seconds.
