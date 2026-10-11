# Reproduction

Run from the repository root in PowerShell:

```powershell
.\.venv-gptq\Scripts\python.exe -B experiments\range_search_w4\results\range_w4_awq_gptq_v35\prepare_v35_preflight.py
```

The command is CPU-only. It checks the v34 parent protocol, dataset/tokenizer hashes, complete-article exclusions, and planned execution counts. It refuses to overwrite differing v35 artifacts. The frozen 64-article gate fails, so no GPU-launch command is provided.
