# v46 source reuse

- repair_v45.py and precision_check.py are byte-identical copies of v45 and define candidate selection, thresholds, restoration and numerical checks.
- runtime_v46.py is the v45 legacy runtime with only the initialization checkpoint path switched from M2 to M1 and final artifact names switched from R to R1.
- audit_sources.py reuses v45 exact bitwise grid inversion with M1 layer checkpoints and M1 model hash, plus alpha=1 validation. It does not infer codes from approximate weights.
- No shared quantization code or historical artifact is written.
