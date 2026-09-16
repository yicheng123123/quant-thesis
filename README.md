# Qwen2.5 Quantization Benchmark

A reproducible, adapter-based framework for comparing quantized variants of
`Qwen/Qwen2.5-0.5B-Instruct` across language-model quality, downstream
accuracy, storage, memory, and generation performance.

The checked-in numerical matrix remains the validated **benchmark v1.2.1**
result set. The v1.2.2 code introduces a quantization-adapter architecture
without changing those frozen raw results or evaluation protocols.

## Add another quantization method

Quantization identity and model loading are separate configuration fields:

```yaml
quantization:
  method: hqq

loader:
  adapter: hf_auto
```

If the checkpoint works with `AutoModelForCausalLM.from_pretrained`, copy the
[generic HF config template](benchmark/configs/_templates/hf_auto.yaml) and no
benchmark code changes are required. A checkpoint that needs a special API
requires one adapter plugin; PPL, HellaSwag, system evaluation, summaries, and
figures remain algorithm-agnostic.

`runtime.python` documents and validates the environment. It does not switch
interpreters. Invoke the runner explicitly with the environment named there.

## Validated configurations

- FP16 baseline
- GPTQ W4A16 symmetric, group sizes 32, 64, and 128
- AWQ W4A16 symmetric G64 with `duo_scaling=false`
- AWQ W4A16 symmetric G64 with `duo_scaling=true`

The formal HellaSwag comparison covers FP16, GPTQ G64, and both AWQ G64
duo-scaling variants. The checked-in results use zero-shot
`lm-evaluation-harness==0.4.13` with a fixed protocol.

## Inspect an experiment without running it

Every runner supports `--dry-run`. It validates the config and checkpoint
metadata, then prints the full experiment manifest without loading a model,
initializing CUDA, or running evaluation.

```powershell
python -m benchmark.runners.eval_downstream `
  --config benchmark/configs/gptq_g64.yaml `
  --task hellaswag `
  --scope full `
  --dry-run
```

The manifest reports the base model, checkpoint, quantization method,
precision, group size, symmetry, backend, AWQ duo-scaling setting, task,
scope, dataset/split, and output paths.

## Results

- [Human-readable summary](benchmark/results/summary/benchmark_v1_summary.md)
- [Complete JSON summary](benchmark/results/summary/benchmark_v1_summary.json)
- [Interactive HTML report](benchmark/results/summary/benchmark_v1_summary.html)
- [Figures](benchmark/results/figures)
- [Raw results and immutable run snapshots](benchmark/results/raw)

## Lightweight validation

These checks do not load a model or rerun an expensive benchmark:

```powershell
python -m benchmark.check_v121
python -m benchmark.check_v122
python -m benchmark.check_v1
python -m benchmark.summarize
git diff --check
```

See [the benchmark documentation](benchmark/README.md) for the fixed evaluation
protocol, environment details, runner commands, comparability boundaries, and
reproduction workflow.
