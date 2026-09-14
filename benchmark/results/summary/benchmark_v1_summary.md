# Unified quantization benchmark v1

| Method | G | PPL | ΔPPL | CKPT MiB | Resident MiB | Peak MiB | System | tok/s mean ± std |
|---|---:|---:|---:|---:|---:|---:|---|---:|
| FP16 | — | 12.4636 | 0.0000 | 953.28 | — | 961.98 | comparable | 43.20 ± — |
| GPTQ | 32 | 13.3063 | 0.8426 | 473.38 | 458.50 | 494.23 | comparable | 15.37 ± — |
| GPTQ | 64 | 13.4928 | 1.0292 | 460.05 | 445.18 | 481.23 | comparable | 9.63 ± 0.73 |
| GPTQ | 128 | 13.8463 | 1.3827 | 453.39 | 438.55 | 474.19 | comparable | 12.43 ± — |
| AWQ | 64 | 14.4898 | 2.0262 | 452.05 | — | — | N/C | — |
| AWQ | 64 | 14.4898 | 2.0262 | 452.05 | — | — | N/C | — |
| AWQ | 64 | 14.5498 | 2.0861 | 452.05 | — | — | N/C | — |

`N/C` means the runtime representation is not comparable to the quantized GPTQ kernel.
Legacy frozen speed rows may not have a standard deviation; rerun `eval_system` for v1 mean/std.
