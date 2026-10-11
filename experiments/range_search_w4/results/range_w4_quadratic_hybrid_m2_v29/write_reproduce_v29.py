from pathlib import Path


def write(run_dir: Path, root_dir: Path) -> None:
    text = r'''# v29 reproduction

## Audit the delivered run

From `E:/quant-thesis` in the existing `.venv-gptq` PowerShell environment, regenerate summaries and headered CSV exports from saved raw results (no GPU experiment is rerun):

```powershell
& .\.venv-gptq\Scripts\python.exe experiments/range_search_w4/results/range_w4_quadratic_hybrid_m2_v29/analyze_v29.py
```

## Fresh full replay

The delivered v29 directory is a completed immutable run; its protocol freezer and implementation lock refuse to overwrite or relock it. For a fresh replay, create an empty sibling directory named `range_w4_quadratic_hybrid_m2_v29_replay`, copy the five scripts listed below, and run:

```powershell
$dst = 'experiments/range_search_w4/results/range_w4_quadratic_hybrid_m2_v29_replay'
$src = 'experiments/range_search_w4/results/range_w4_quadratic_hybrid_m2_v29'
New-Item -ItemType Directory -Path $dst
Copy-Item -Path @("$src/freeze_v29.py", "$src/spot_recheck_v29.py", "$src/lock_and_preflight_v29.py", "$src/run_v29.py", "$src/analyze_v29.py", "$src/write_reproduce_v29.py") -Destination $dst
'{"max_seconds":5400.0,"gpu_wall_elapsed_seconds":0.0,"last_recorded_stage":"fresh replay"}' | Set-Content -Encoding ascii "$dst/BUDGET.json"
& .\.venv-gptq\Scripts\python.exe "$dst/freeze_v29.py"
& .\.venv-gptq\Scripts\python.exe "$dst/spot_recheck_v29.py"
& .\.venv-gptq\Scripts\python.exe "$dst/lock_and_preflight_v29.py"
$env:V29_RUN_LABEL='validation'
$env:V29_SCORE_MODE='dual'
& .\.venv-gptq\Scripts\python.exe "$dst/run_v29.py"
& .\.venv-gptq\Scripts\python.exe "$dst/analyze_v29.py"
```

The completed v29 run recovered finalization from 24 verified layer checkpoints after creating the missing `validation/M2` output directory; candidate GPTQ was not repeated during recovery. It used the frozen local model/calibration cache and wrote new artifacts only below v29. Hybrid-only M2 and new PPL were not run.
'''
    (run_dir / "REPRODUCE.md").write_text(text, "utf-8")
    (root_dir / "REPRODUCE.md").write_text(text, "utf-8")
