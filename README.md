# Qwen2.5 Quantization Benchmark

A reproducible benchmark for comparing FP16, GPTQ, and AWQ variants of
`Qwen/Qwen2.5-0.5B-Instruct` across language-model quality, downstream
accuracy, storage, memory, and generation performance.

The latest validated release is **benchmark v1.2.1**. It adds explicit
experiment identity, checkpoint/config cross-validation, startup manifests,
dry-run inspection, raw-result provenance, and unambiguous summary labels.

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
python -m benchmark.check_v1
python -m benchmark.summarize
git diff --check
```

See [the benchmark documentation](benchmark/README.md) for the fixed evaluation
protocol, environment details, runner commands, comparability boundaries, and
reproduction workflow.
