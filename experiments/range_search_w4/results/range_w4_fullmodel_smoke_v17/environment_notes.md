# Environment notes

This is an isolated Python 3.12 virtual environment at `.venv-omni`; the existing `.venv-gptq` remained unchanged. The installed stack pins PyTorch 2.6.0+cu124, Transformers 4.49.0, and tokenizers 0.21.4. The official OmniQuant source is editable from the v17-local clone at commit `feffe8ea87d80f7bb57b6e25e7cff9dc950fcc14`.

The official project has no complete dependency lock, so `pip_freeze.txt` records the resolved Windows dependency set. GPU availability and device information are in `cuda_environment.txt`. CUDA runtime is 12.4; the installed NVIDIA driver reports CUDA 12.5. The device reports 8,188 MiB total memory under WDDM.

Dependency installation and Python syntax compilation succeeded. No OmniQuant model-forward, LWC optimization, export, or PPL smoke ran because the preceding GPTQ parity gate failed. The first attempt to import a top-level module named `omniquant` was not the repository's documented module layout; the correct `quantize.omniquant` import was not smoke-tested in this stopped run.
