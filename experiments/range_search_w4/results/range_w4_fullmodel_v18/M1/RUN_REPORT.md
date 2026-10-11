# M1 native GPTQ run report

Status: **COMPLETE / correctness PASS / reload PASS**.

- Model revision: `7ae557604adf67be50417f59c2c2f167def9a775`; W4A16 symmetric G128; block 128; damp 0.1; native sequential GPTQ.
- Started from the original FP checkpoint and used the frozen 128 × 2048 WikiText-2 train calibration manifest (SHA256 `664419273f63df3c0b0746b18bb66b593a5384bbe6ba8afcd6f4905c9479aae3`).
- Processed 24 layers / 168 linears / 304,128 rows. This is an independent native GPTQ run; it was not assembled from M2 candidates.
- Actual wall including load, layer replay, checkpointing, final save/reload: 648.876 s. Final save 0.772 s. Peak allocated/reserved CUDA memory for the combined formal process: 2.19/2.50 GiB.
- Final checkpoint SHA256: `3678f867d064abc9730d392f279516223ef17de16351c577e0f1977e9b7bba33`. Reload output max/mean absolute difference on 64 fixed tokens: 0.0/0.0.
- WT2: PPL 13.88436957, NLL/token 2.630763716, 299,077 scored tokens. C4 fixed-budget: PPL 22.23392445, NLL/token 3.101619251, 131,071 scored tokens.

Reproduce both formal methods in frozen order with `& '.venv-gptq\Scripts\python.exe' -X utf8 'experiments\range_search_w4\results\range_w4_fullmodel_v18\run_formal.py'`; this runner has no per-method CLI and resumes completed layer checkpoints. Evaluate the frozen matrix with the corresponding `evaluate_v18.py` command. Detailed artifacts: `QUANTIZATION_METADATA.json`, `QUANTIZATION_CORRECTNESS.json`, `RELOAD_CORRECTNESS.json`, `checkpoint/final/`, and `../FORMAL_TIMING.csv`.
