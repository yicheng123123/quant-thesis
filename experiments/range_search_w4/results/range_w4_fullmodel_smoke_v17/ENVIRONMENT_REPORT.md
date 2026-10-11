# v17 isolated environment

- Python: 3.12.10, 64-bit Windows.
- PyTorch: 2.6.0+cu124, CUDA runtime 12.4.
- Transformers: 4.49.0; tokenizers: 0.21.4; datasets: 5.0.1; accelerate: 1.15.0.
- Device: NVIDIA GeForce RTX 4060 Laptop GPU, 8,188 MiB reported by NVML; driver 555.97 (driver CUDA 12.5).
- Official source: OpenGVLab/OmniQuant commit `feffe8ea87d80f7bb57b6e25e7cff9dc950fcc14`, cloned into this v17 result directory. The v16 upstream source hash list is preserved as `OFFICIAL_SOURCE_SHA256.json`; source provenance and local patch hashes are in `OFFICIAL_SOURCE_PROVENANCE.json`.
- The isolated `.venv-omni` was created under v17 and installed the official source as an editable project plus the frozen target versions. Full transitive versions are in `pip_freeze.txt`.
- The existing `.venv-gptq` was used only to run the bounded M2 parity gate and was not modified.
- `py_compile` passed for the adapter, modified OmniQuant dispatcher/scaler, and the M2 smoke script.

The v17 run stopped before loading the OmniQuant adapter in a model forward. Thus environment installation and source syntax are verified, while source import and Qwen forward compatibility remain unverified. A first import probe used the incorrect top-level name `omniquant` (the repository exposes modules such as `quantize.omniquant`); no claim of an import failure of the official entrypoint is made.
