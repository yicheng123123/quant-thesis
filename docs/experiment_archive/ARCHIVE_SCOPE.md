# Archive scope and exclusions

## Included

- Actual experiment directories present under
  `experiments/range_search_w4/results/`, including numbered runs, variants,
  unversioned prechecks, negative outcomes, numerical stops, failures, and
  recovery records.
- The shared range-search implementation and protocol at
  `experiments/range_search_w4/`.
- `results/v61_m3_calibration_cost_memory/` and
  `results/v62_m3_vs_repaired_quality_cost/` reports, configuration, timing,
  memory, environment, hash audits, score windows, raw score tables, and
  reproduction notes.
- Project-level ignore rules needed to keep nested environments and generated
  tensor/checkpoint caches out of Git while preserving local files.
- This archive index and a local-only artifact manifest with byte sizes,
  purposes, rebuild notes, and hash provenance where available. Existing
  hashes in the manifest were computed earlier in this archive attempt from
  local bytes; they were not re-verified after the no-rescan instruction.
  Items not covered by that completed pass are marked `NOT_COMPUTED`.

## Excluded from Git, retained locally

- Full model checkpoints/dense exports, candidate banks, optimizer states,
  activation/Hessian/Gram/factor caches, NumPy tensor dumps, and local Python
  environments. The manifest lists scoped excluded items by relative path,
  byte size, experiment, purpose, and rebuild/download status. Hashes are
  reused only where already present; otherwise the value is `NOT_COMPUTED`.
- Full WT2/C4 token-ID evaluation streams and duplicate dataset-derived token
  caches. `DATA_INPUT_MANIFESTS.csv` preserves source manifest hashes, dataset
  revision/split, token hashes, window settings, and counts without embedding
  the token IDs. The original files remain in their historical locations.
- Duplicate v20.1 Kaggle bundle ZIPs, which package local source/data material;
  their paths, sizes, and hashes are listed with the other local-only artifacts.
- Downloaded model/data caches and unrelated root-level artifacts that are
  outside the v1–v62 experiment result trees.
- Vendored upstream source clones and nested site-package/virtual-environment
  copies. The experiment reports retain their source/revision/hash references;
  this archive does not present a local clone as an official reproduction.

## Worktree changes not included

The pre-existing changes under `benchmark/`, root `README.md`, and unrelated
untracked project directories were left unstaged and untouched. They are not
part of this range-search v1–v62 archive. No historical report or checkpoint
was rewritten, deleted, or changed to `PASS`. No global Git configuration or
Git LFS policy was changed.

## Evidence limits

- A result directory is indexed only when it exists. Version labels identify
  the directory name, including numbered variants; missing numbers do not
  imply a run.
- Some early result folders have no standalone report. Those entries say so
  and point to available raw artifacts rather than inventing a result.
- Original output manifests may reference local-only tensor or token-window
  files. The manifests are preserved as evidence; a GitHub checkout does not
  contain those excluded files, and their hashes/paths remain in the local-only
  artifact manifest.
- The large `REFERENCE_PROFILE_TRACE.json` remains on disk and is excluded
  from Git; a lossless gzip copy is archived instead. Its content hash is
  `NOT_COMPUTED`.
- v59 is the same-process four-arm formal evaluation. v55/v56 have their own
  independent propagation checks. Historical v45/v46/v49 propagation remains
  `UNKNOWN`.
- v61 M3 is a newly calibrated repository M3/LWC checkpoint, with its own
  measured PPL; it does not inherit an older M3 PPL.
- v62 is a two-checkpoint same-process comparison and shows a WT2/C4 corpus
  tradeoff. The methods use different quantization/repair schemes, so this is
  a comparison of complete solutions.
- Existing integer-coordinate repair is not claimed as an original algorithm.
  Calibration/quantization time is not online inference acceleration.
- No raw corpus is uploaded. Reproduction requires the exact frozen dataset
  revisions/manifests and local or reproducibly sourced model artifacts named
  by each protocol. Where no verified external download location is known, the
  large-artifact manifest says `仅本地保存`.
