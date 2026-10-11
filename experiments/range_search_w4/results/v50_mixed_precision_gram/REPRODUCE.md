# v50 reproduction

CPU preparation (only before the first GPU start):

```powershell
& E:/quant-thesis/.venv-gptq/Scripts/python.exe -B -X utf8 E:/quant-thesis/experiments/range_search_w4/results/v50_mixed_precision_gram/prepare_v50.py
```

Frozen bounded GPU entry:

```powershell
& E:/quant-thesis/.venv-gptq/Scripts/python.exe -u -B -X utf8 E:/quant-thesis/experiments/range_search_w4/results/v50_mixed_precision_gram/run_v50.py
```

The existing evidence directory refuses another launch because GPU_START.json exists. Do not delete the start marker, reset the budget or overwrite this attempt to make a fresh run appear completed. CPU closeout uses finalize_v50.py; it refuses overwriting an existing STATUS.json.
