# Quantization experiment archive (v1–v62)

This entry point indexes the actual experiment directories under
`experiments/range_search_w4/results/` plus the separate v61 and v62 records
under `results/`. The index lists the report and reproduction notes that exist
for each directory; directory names and version gaps are not inferred to mean
that a run occurred. Read each original report for its frozen protocol,
execution status, failures, and limitations.

## Browse the archive

1. Start with [`EXPERIMENT_INDEX.csv`](EXPERIMENT_INDEX.csv).
2. Open the linked report and reproduction entry for the experiment of
   interest. The shared range-search entry points are
   [`experiments/range_search_w4/README.md`](../../experiments/range_search_w4/README.md),
   [`REPORT.md`](../../experiments/range_search_w4/REPORT.md), and
   [`protocol.json`](../../experiments/range_search_w4/protocol.json).
3. Use [`ARCHIVE_SCOPE.md`](ARCHIVE_SCOPE.md) and
   [`LARGE_ARTIFACTS_MANIFEST.csv`](LARGE_ARTIFACTS_MANIFEST.csv) to identify
   local-only checkpoints, tensor caches, and excluded tokenized evaluation
   inputs. `DATA_INPUT_MANIFESTS.csv` contains metadata and hashes without the
   token-ID arrays.

The artifact manifest preserves hashes that were already computed earlier in
this archive attempt. Those values are marked with their source and are not
re-verified by the final staging pass. Entries without an existing hash are
marked `NOT_COMPUTED`; the catalog builder reuses the existing manifest by
default and does not read excluded artifact contents. An explicit
`--refresh-artifact-inventory` option updates path/size inventory only and
still does not calculate hashes.

The archive preserves reports, positive and negative results, failure/stop
records, JSON/JSONL/CSV measurements, source scripts, frozen configuration,
data-split references, input hashes, and reproduction notes. It does not
publish full model weights, local environments, activation/Hessian caches, or
duplicate source corpora. Those exclusions do not remove the local files.

## Dependencies and reproduction

The repository-level `requirements.txt` is a lightweight starting point, not
a single environment lock for all historical runs. Each run's report or
`PROCESS_ENVIRONMENT.json` is authoritative for its actual Python, PyTorch,
Transformers, CUDA, attention, and precision settings. Use the exact
`REPRODUCE.md` linked by an index row; do not assume that commands or package
versions are interchangeable across versions.

CPU-only audits and report/table reconstruction can be run from the archived
scripts without model execution when their local inputs are available. GPTQ,
LWC calibration, integer-code repair, activation capture, and full PPL
evaluation require the documented GPU/runtime and local or reproducibly
downloaded model/data artifacts. The large-artifact manifest records whether
an item is rebuildable from a frozen revision and protocol; it does not invent
download URLs. Evaluation token inputs are deliberately not uploaded.

Typical setup for a CPU-only archive inspection is:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python --version
```

Then follow the selected run's reproduction note. Install only the packages
that note requires; installing the current `requirements.txt` alone does not
recreate historical GPU environments. Do not run a reproduction command that
would write into a historical results directory; choose a new output directory
first.

## Interpretation notes

- v59 is the same-process four-checkpoint evaluation. v62 is a fresh same-
  process comparison of v61 M3 and v55 M2 plus integer repair.
- v55 and v56 include independent propagation validation for their respective
  repair runs. The historical v45/v46/v49 repair propagation status remains
  `UNKNOWN`; later evaluation does not change that evidence state.
- v61 is a new repository M3/LWC calibration output and does not inherit an
  earlier M3 PPL measurement. It is not described as an official OmniQuant
  reproduction.
- v62 shows a corpus tradeoff: v55 is lower on WT2 test and v61 M3 is lower on
  the fixed-budget C4 validation stream. Neither wins both.
- Integer-coordinate repair follows existing discrete repair ideas; this
  archive does not claim the repair algorithm as original.
- Calibration/quantization cost is not online inference speedup.
- WT2 and C4 are established benchmarks here, not new blind test data. No
  single run establishes statistical significance or cross-seed stability.
