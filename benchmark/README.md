# Unified quantization benchmark v1

## Status: validated

```yaml
benchmark: v1
status: validated

validated_configurations:
  - GPTQ W4A16 SYM G64: quality + system
  - AWQ W4A16 SYM G64: quality

quality_reproducibility:
  gptq_ppl: 13.4928375
  awq_ppl: 14.4898112

system_protocol:
  process: independent
  warmup: 3
  measure: 5
  generation: fixed deterministic protocol
```

Validated v1 results:

- GPTQ G64 quality PPL: `13.492837547663`
- GPTQ G64 system throughput: `9.6287 ± 0.7263 tok/s`
- AWQ G64 quality PPL: `14.489811246371`
- AWQ system metrics remain `comparable=false` because the current
  Transformers path decompresses the AWQ checkpoint.

The framework is now frozen for v1 acceptance. The next step is to use this
framework for new experiments; do not expand the benchmark framework unless a
new experiment requires a separately justified protocol change.

The v1.1 reporting layer adds immutable run metadata, environment fingerprints,
validation gates, and standardized figures without changing quantization or
checkpoint generation.

This framework evaluates existing FP16, GPTQ, and AWQ artifacts. It does not
quantize models. Version 1 standardizes four things:

1. YAML experiment configs.
2. One WikiText-2 sliding-window PPL implementation.
3. Isolated-process CUDA memory and repeated generation benchmarks.
4. One JSON schema with JSON/CSV/Markdown summaries and a visual HTML dashboard.

## Structure

```text
benchmark/
├── configs/                 # model × method × configuration
├── eval/                    # reusable PPL, generation, memory, and loaders
├── runners/                 # parent process → isolated worker
├── results/raw/             # latest JSON per experiment_id
├── results/raw/runs/        # immutable v1.1 run snapshots
├── results/summary/         # generated JSON, CSV, Markdown, and visual HTML
├── results/figures/         # standardized PNG and SVG figures
├── config.py                # schema and result persistence
├── figures.py               # unified figure pipeline
├── validation.py            # regression and protocol gates
├── freeze_current_results.py
├── migrate_v11.py
├── check_v1.py
└── summarize.py
```

## Freeze and validate the current experiments

Run from `E:\quant-thesis`:

```powershell
& .venv\Scripts\python.exe -X utf8 -m benchmark.freeze_current_results
& .venv\Scripts\python.exe -X utf8 -m benchmark.check_v1
& .venv\Scripts\python.exe -X utf8 -m benchmark.summarize
```

The frozen rows preserve their original protocol. The old system benchmark
used one warmup, three measurements, 96 generated tokens, and did not save a
throughput standard deviation. It is not presented as a v1 repeated result.

## Run quality in an isolated process

The lightweight parent reads the YAML and starts a worker using the config's
own Python environment. The worker loads one model, writes its result, and
exits before another configuration starts.

```powershell
& .venv\Scripts\python.exe -X utf8 -m benchmark.runners.eval_quality `
  --config benchmark\configs\gptq_g64.yaml

& .venv\Scripts\python.exe -X utf8 -m benchmark.runners.eval_quality `
  --config benchmark\configs\awq_g64.yaml
```

Each config contains a frozen reference PPL. A new run must match it within
the configured tolerance.

## Run system metrics in an isolated process

Every invocation creates one fresh worker. The standardized generation
protocol is three warmups, five measured runs, and 128 maximum new tokens.

```powershell
& .venv\Scripts\python.exe -X utf8 -m benchmark.runners.eval_system `
  --config benchmark\configs\gptq_g64.yaml

& .venv\Scripts\python.exe -X utf8 -m benchmark.runners.eval_system `
  --config benchmark\configs\awq_g64.yaml
```

The current AWQ Transformers path decompresses the checkpoint. Its PPL is
valid, but its system metrics are saved with `system_metrics_comparable=false`
and must not be ranked against the GPTQ quantized kernel.

Regenerate summaries after any run:

```powershell
& .venv\Scripts\python.exe -X utf8 -m benchmark.summarize
```

For an existing v1 checkout, migrate the current latest results once before
running the v1.1 summary gate:

```powershell
& .venv\Scripts\python.exe -X utf8 -m benchmark.migrate_v11
```

Every summary run also regenerates the standalone visual report:

```text
benchmark/results/summary/benchmark_v1_summary.html
```

It also regenerates these paired PNG/SVG figures:

```text
benchmark/results/figures/
├── ppl_comparison.{png,svg}
├── ppl_delta.{png,svg}
├── vram_comparison.{png,svg}
├── throughput.{png,svg}
├── quality_vs_memory_tradeoff.{png,svg}
└── quality_vs_checkpoint_tradeoff.{png,svg}
```

If validation fails, `summarize` stops before writing new summaries or figures
and prints the failed regression gate. AWQ system rows are excluded from
system comparison whenever `system_metrics_comparable=false`.
