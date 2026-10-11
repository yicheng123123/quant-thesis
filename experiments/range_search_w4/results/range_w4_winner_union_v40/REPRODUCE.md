# v40 reproduction and audit commands

Run from `E:\quant-thesis` with the frozen GPTQ environment.

The new GPU runs are complete. Do not rerun them over this finished result directory. The following read-only check validates the frozen manifest and saved source hashes:

```powershell
.\.venv-gptq\Scripts\python.exe experiments\range_search_w4\results\range_w4_winner_union_v40\verify_v40_freeze.py
```

The recorded GPU-run command for each phase was the supervisor invocation below. If repeating the experiment, first make a clean sibling copy of this frozen v40 directory and run from that copy, preserving this completed output:

```powershell
.\.venv-gptq\Scripts\python.exe experiments\range_search_w4\results\range_w4_winner_union_v40\run_v40_stage.py PILOT
.\.venv-gptq\Scripts\python.exe experiments\range_search_w4\results\range_w4_winner_union_v40\run_v40_stage.py A
.\.venv-gptq\Scripts\python.exe experiments\range_search_w4\results\range_w4_winner_union_v40\run_v40_stage.py B
```

Each phase is resumable at completed decoder-layer checkpoints, and the supervisor enforces the frozen cumulative execution cutoff. Phase B uses only the immutable `CANDIDATE_MANIFEST.json`; candidate order must remain the original scalar10 order and alpha=1 must remain included. Do not rebuild the manifest from current-run outcomes.

Saved parity/hash tables were generated with:

```powershell
.\.venv-gptq\Scripts\python.exe experiments\range_search_w4\results\range_w4_winner_union_v40\analyze_v40.py pilot
.\.venv-gptq\Scripts\python.exe experiments\range_search_w4\results\range_w4_winner_union_v40\analyze_v40.py A
.\.venv-gptq\Scripts\python.exe experiments\range_search_w4\results\range_w4_winner_union_v40\analyze_v40.py B
```

The executed command lines, environments, phase logs, source hashes, raw module records, checkpoints and final model files are preserved under this v40 directory. The current artifact review does not run held evaluation, P3, or PPL.
