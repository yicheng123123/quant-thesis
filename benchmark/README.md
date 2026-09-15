# Unified quantization benchmark v1.2

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

## Structure

```text
benchmark/
├── configs/                 # model × method × controlled protocol
├── eval/
│   ├── model.py             # existing method-aware loader
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
