# Unified quantization benchmark v1.2

| Configuration | G | PPL | ΔPPL | CKPT MiB | Resident MiB | Peak MiB | System | tok/s mean ± std |
|---|---:|---:|---:|---:|---:|---:|---|---:|
| FP16 | — | 12.4636 | 0.0000 | 953.28 | — | 961.98 | comparable | 43.20 ± — |
| GPTQ G32 | 32 | 13.3063 | 0.8426 | 473.38 | 458.50 | 494.23 | comparable | 15.37 ± — |
| GPTQ G64 | 64 | 13.4928 | 1.0292 | 460.05 | 445.18 | 481.23 | comparable | 9.63 ± 0.73 |
| GPTQ G128 | 128 | 13.8463 | 1.3827 | 453.39 | 438.55 | 474.19 | comparable | 12.43 ± — |
| AWQ G64 | 64 | 14.4898 | 2.0262 | 452.05 | — | — | N/C | — |
| AWQ G64 duo=False | 64 | 14.4898 | 2.0262 | 452.05 | — | — | N/C | — |
| AWQ G64 duo=True | 64 | 14.5498 | 2.0861 | 452.05 | — | — | N/C | — |

`N/C` means the runtime representation is not comparable to the quantized GPTQ kernel.
Legacy frozen speed rows may not have a standard deviation; rerun `eval_system` for v1 mean/std.

## HellaSwag downstream quality

| Configuration | PPL | ΔPPL | HS acc ± stderr | HS acc_norm ± stderr | n |
|---|---:|---:|---:|---:|---:|
| FP16 | 12.4636 | 0.0000 | 40.63% ± 0.49 pp | 52.48% ± 0.50 pp | 10042 |
| GPTQ G64 | 13.4928 | 1.0292 | 39.46% ± 0.49 pp | 50.27% ± 0.50 pp | 10042 |
| AWQ G64 duo=False | 14.4898 | 2.0262 | 39.29% ± 0.49 pp | 49.42% ± 0.50 pp | 10042 |
| AWQ G64 duo=True | 14.5498 | 2.0861 | 38.92% ± 0.49 pp | 49.91% ± 0.50 pp | 10042 |

## Automatic factual conclusions

- GPTQ G64 increased PPL by 8.26% relative to FP16; HellaSwag acc_norm changed by -2.21 percentage points.
- AWQ G64 duo=False increased PPL by 16.26% relative to FP16; HellaSwag acc_norm changed by -3.06 percentage points.
- AWQ G64 duo=True increased PPL by 16.74% relative to FP16; HellaSwag acc_norm changed by -2.57 percentage points.
- Higher PPL was accompanied by lower HellaSwag acc_norm in these evaluated configurations.
