# Unified quantization benchmark v1.2.2

## Status: validated

```yaml
benchmark: v1.2
status: validated

validated_configurations:
  - FP16: PPL + HellaSwag
  - GPTQ W4A16 SYM G64: PPL + system + HellaSwag
  - AWQ W4A16 SYM G64 duo_scaling=false: PPL + HellaSwag
  - AWQ W4A16 SYM G64 duo_scaling=true: PPL + HellaSwag

downstream_protocol:
  harness: lm-evaluation-harness 0.4.13
  task: HellaSwag
  mode: zero-shot continuation likelihood
  num_fewshot: 0
  apply_chat_template: false
  formal_runs: offline/cache-only
```

Version 1.2 adds one controlled capability to v1.1: downstream HellaSwag
quality evaluation through `lm-evaluation-harness`. Existing PPL, system,
quantizer, checkpoint, and system-comparability logic is unchanged.

The four-configuration downstream matrix is FP16, GPTQ G64, AWQ G64
`duo_scaling=false`, and AWQ G64 `duo_scaling=true`. G32 and G128 are not part
of the v1.2 downstream acceptance matrix.

## Transparency patch (v1.2.1)

Every config now contains an explicit `experiment` identity block and a
`quantization` block covering method, W/A precision, group size, symmetry,
backend, and AWQ `duo_scaling`. Each runner prints the same startup manifest,
including the task, smoke/full scope, dataset/split, checkpoint identity, and
result paths. Checkpoint-side JSON metadata is cross-checked before model
loading when the format exposes method, bits, group size, or symmetry.

Use `--dry-run` to validate a config and inspect the manifest without importing
Torch, initializing CUDA, loading a model, or evaluating anything:

```powershell
& .\.venv-gptq\Scripts\python.exe -X utf8 `
  -m benchmark.runners.eval_quality `
  --config .\benchmark\configs\gptq_g64.yaml `
  --dry-run
```

The lightweight regression suite is `python -m benchmark.check_v121`; it does
not run any model or expensive benchmark.

## Quantization adapter architecture (v1.2.2)

The benchmark now separates four concerns:

```text
quantization.method   algorithm identity, such as hqq or myquant
quantization.*        physical representation, such as W4A16 G64 symmetric
loader.adapter        loading mechanism, such as hf_auto or a native plugin
runtime.*             process environment and experiment-level comparability
```

`quantization.method` never implicitly chooses a loader. For example, an HQQ
checkpoint that Transformers can load directly uses:

```yaml
quantization:
  method: hqq
  weight_bits: 4
  activation_bits: 16
  activation_precision: A16
  group_size: 64
  symmetric: true
  duo_scaling: null

loader:
  adapter: hf_auto
  backend: transformers_hqq
```

Start from [`configs/_templates/hf_auto.yaml`](configs/_templates/hf_auto.yaml).
The generic adapter supports arbitrary method names. It reports checkpoint
identity as `MATCH`, `MISMATCH`, or `NOT_DETECTABLE`; absent metadata is never
treated as proof of identity.

For a checkpoint requiring a custom API, add one dependency-light module that
subclasses `QuantizationAdapter`, decorate it with
`@register_adapter("my_adapter")`, and set both fields explicitly:

```yaml
quantization:
  method: myquant
loader:
  adapter: my_adapter
  module: benchmark.adapters.myquant
```

Adapter modules must defer Torch and backend imports until `load()`. They return
a single `LoadedModel` containing the model, tokenizer, detected checkpoint
metadata, and capabilities. Evaluators consume that object without inspecting
the quantization method.

`runtime.python` is provenance and an environment-consistency contract only.
It cannot switch the active interpreter. Run with the configured environment:

```powershell
& .\.venv-hqq\Scripts\python.exe -X utf8 `
  -m benchmark.runners.eval_quality `
  --config .\benchmark\configs\hqq_w4.yaml `
  --dry-run
```

Workers use that already-active interpreter when creating their isolated child
process. `runtime.packed_execution` and `runtime.system_metrics_comparable`
describe this particular checkpoint/backend combination, not an intrinsic
property of the algorithm. Existing AWQ configs therefore remain non-comparable
because their current path executes decompressed weights.

The v1.2.2 offline architecture regression is:

```powershell
python -m benchmark.check_v122
```

## Structure

```text
benchmark/
├── adapters/                # plugin interface, registry, built-in loaders
├── configs/_templates/      # config-only onboarding templates
├── configs/                 # model × method × controlled protocol
├── eval/
│   ├── model.py             # algorithm-neutral adapter dispatch
│   ├── ppl.py               # existing WikiText-2 PPL
│   └── downstream.py        # lm-eval adapter + provenance capture
├── runners/
│   ├── eval_quality.py      # existing PPL runner
│   ├── eval_system.py       # existing isolated system runner
│   └── eval_downstream.py   # isolated downstream runner
├── results/raw/             # latest JSON per experiment_id
├── results/raw/runs/        # immutable run snapshots
├── results/summary/         # JSON, CSV, Markdown, and HTML
├── results/figures/         # paired PNG/SVG scientific figures
├── config.py                # schema and persistence
├── validation.py            # regression + comparability gates
├── figures.py               # unified figure pipeline
└── summarize.py             # validate → summarize → visualize
```

## Fixed harness dependency

The formal v1.2 protocol fixes `lm_eval==0.4.13` in every runtime used by the
four configurations:

```powershell
& .\.venv\Scripts\python.exe -m pip install "lm_eval==0.4.13"
& .\.venv-gptq\Scripts\python.exe -m pip install "lm_eval==0.4.13"
& .\.venv-awq\Scripts\python.exe -m pip install "lm_eval==0.4.13"
```

Do not upgrade Torch, Transformers, GPTQModel, llmcompressor, or the existing
quantization environments as part of downstream evaluation.

## Prepare once, evaluate offline

Dataset preparation is intentionally separate from formal runs. Cache
`Rowan/hellaswag` under the project `.hf_cache` once while online, then run the
benchmark worker with the cache in offline mode. The runner sets
`HF_HUB_OFFLINE=1` and `HF_DATASETS_OFFLINE=1` before loading the dataset.

Each raw downstream result records:

- `acc`, `acc_norm`, and both standard errors;
- effective and original sample counts;
- `num_fewshot`, chat-template setting, batch size, and smoke/full scope;
- harness and task versions plus the task YAML SHA-256;
- dataset name, split, and fingerprint;
- tokenizer class, vocabulary size, and behavior fingerprint.

## Smoke then full

Run from `E:\quant-thesis`. A smoke run checks 32 examples but is marked
`quality_comparable=false` and cannot enter formal comparison figures.

```powershell
& .\.venv-gptq\Scripts\python.exe -X utf8 `
  -m benchmark.runners.eval_downstream `
  --config .\benchmark\configs\gptq_g64.yaml `
  --task hellaswag `
  --scope smoke
```

After smoke passes, run the full matrix sequentially so GPU workloads never
overlap:

```powershell
& .\.venv\Scripts\python.exe -X utf8 -m benchmark.runners.eval_downstream --config .\benchmark\configs\fp16.yaml --task hellaswag --scope full
& .\.venv-gptq\Scripts\python.exe -X utf8 -m benchmark.runners.eval_downstream --config .\benchmark\configs\gptq_g64.yaml --task hellaswag --scope full
& .\.venv-awq\Scripts\python.exe -X utf8 -m benchmark.runners.eval_downstream --config .\benchmark\configs\awq_g64_duo_false.yaml --task hellaswag --scope full
& .\.venv-awq\Scripts\python.exe -X utf8 -m benchmark.runners.eval_downstream --config .\benchmark\configs\awq_g64_duo_true.yaml --task hellaswag --scope full
```

## Validate, summarize, and visualize

```powershell
& .\.venv\Scripts\python.exe -X utf8 -m benchmark.summarize
```

`summarize` first applies per-run and cross-run gates. All four formal rows must
use the same harness/task definition, dataset fingerprint, sample count,
zero-shot protocol, no chat template, and tokenizer behavior. A failure stops
summary and figure generation.

The report layer generates JSON, CSV, Markdown, HTML, and paired PNG/SVG
figures. Version 1.2 adds only:

```text
hellaswag_acc.{png,svg}
hellaswag_acc_norm.{png,svg}
ppl_vs_hellaswag_acc_norm.{png,svg}
```

Accuracy bars include standard-error bars. Automatic prose reports PPL changes
and downstream changes in percentage points; it describes observed association
without claiming causality.

AWQ system metrics remain `system_metrics_comparable=false` because the current
Transformers path decompresses those checkpoints. They are excluded from all
cross-method system figures and rankings.
