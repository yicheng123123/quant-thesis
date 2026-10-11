# Validation v12 preflight and native parity

Status: **PASS**

Gate SHA256: `7943b8f26e5f94456b7b00144f6d47fc775d0224ccca4a6a1eff38e5740999c9`; thresholds: gain ≥ 0.10 AND margin ≥ 0.06.
Frozen manifests: rows `0e26c120f805772ea7619a4891979272afa0db43e619e46a35acfb133efcd91d`, splits `8aa736fc321667f086f20c05f998f70d246bbee52fda96fe96e6a58265eb4274`.
Workload: 4800 nominal, 0 reusable, 4800 new candidate evaluations.
Model-input capture: 3.9s; CUDA allocated/reserved peak 996.5/1052.0 MiB.

## Alpha=1 native parity on preregistered S00 rows

| Projection | Row | Input dim | Result | H build (s) | Native GPTQ (s) | Custom alpha=1 (s) | Max CUDA reserved (MiB) |
|---|---:|---:|---|---:|---:|---:|---:|
| q_proj | L4_R60 | 896 | PASS | 0.224 | 0.515 | 0.208 | 64.0 |
| o_proj | L4_R135 | 896 | PASS | 0.006 | 0.358 | 0.204 | 64.0 |
| up_proj | L4_R802 | 896 | PASS | 0.005 | 0.373 | 0.202 | 64.0 |
| down_proj | L4_R194 | 4864 | PASS | 0.182 | 1.871 | 1.092 | 522.0 |

## Candidate cost estimate

Measured alpha=1 candidate samples project {'q_proj': 0.20838630000071134, 'o_proj': 0.2039065999997547, 'up_proj': 0.2019058000005316, 'down_proj': 1.0920101000010618} to 1,200 evaluations per projection. Estimated candidate timer sum: 2047.5s; Hessian construction: 12.5s; capture: 3.9s.
With 1.5× execution margin, estimated formal wall time is 3093.8s (0.86h). This is an estimate from one row per projection and may vary by split/layer.
Disk free before formal run: 505.52 GiB; minimum planned artifact reservation: 1.23 GiB.
Peak observed CUDA reserved across capture/parity: 1052.0 MiB.

Formal evaluation is authorized only when status is PASS. No candidate landscape is produced during preflight.
