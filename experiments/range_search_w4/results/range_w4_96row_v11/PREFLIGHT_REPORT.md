# v11 Stage 1 projection preflight

Status: **PASS**

Frozen 96-row manifest SHA-256: `c56eeaa1d1af6e4ed8dd78f811c811c66758b215659a98bf8550052c786e6106`.
v9 split manifest SHA-256: `34a24ef8602b8ce1e38e7a284f112acf7f766962d87d737f69148b78bcfaba78`; splits: S00–S02.
Planned Stage 1: 2880 candidates; reused: 30; new: 2850.

| Projection | In × out | Parity | H build (s) | H peak alloc / reserved (MiB) | candidate mean (s) | candidate peak alloc / reserved (MiB) | estimated formal (s, 1.5×) |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| q_proj | 896 × 896 | PASS | 0.226 | 30.9 / 44.0 | 0.378 | 20.0 / 42.0 | 393.8 |
| o_proj | 896 × 896 | PASS | 0.009 | 41.8 / 66.0 | 0.388 | 28.3 / 44.0 | 419.1 |
| up_proj | 896 × 4864 | PASS | 0.006 | 46.6 / 66.0 | 0.377 | 28.6 / 44.0 | 407.5 |
| down_proj | 4864 × 896 | PASS | 0.275 | 583.9 / 614.0 | 2.049 | 155.4 / 246.0 | 2216.0 |

## Gate checks

- `frozen_row_manifest_96_rows`: PASS
- `v9_split_manifest_hash`: PASS
- `S00_S02_reused`: PASS
- `all_four_projection_preflights_completed`: PASS
- `four_projection_native_parity_pass`: PASS
- `no_preflight_cuda_oom_or_numerical_failure`: PASS
- `existing_v10_row_splits_verified`: PASS
- `projected_formal_runtime_within_4h_gate`: PASS
- `peak_process_reserved_within_80pct_gpu`: PASS
- `v8_v9_v10_unchanged_during_preflight`: PASS

Observed successful preflight wall time: 18.52 s.
Estimated end-to-end including preflight and activation capture: 3457.4 s.
Maximum process CUDA reserved: 1052.0 MiB / 8187.5 MiB.
