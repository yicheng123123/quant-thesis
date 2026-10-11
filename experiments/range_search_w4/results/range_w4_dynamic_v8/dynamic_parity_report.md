# Dynamic G128 parity

Status: **PASS**. Native GPTQModel 2.2.0 vs custom dynamic alpha=1.

Max FP16 weight difference: 0; mean difference: 0; mismatched elements: 0.

Max group-entry difference: 0; scale difference: 0; zero difference: 0.

Native/custom search MSE: 8.99372473e-05 / 8.99372473e-05.

Native/custom held-out MSE: 0.000321639835 / 0.000321639835.

The native call uses GPTQModel's real `GPTQ.quantize` with a read-only find_params trace. `parity_group_entries.pt` saves each compensated group-entry weight and scale/zero. `parity_weights.pt` saves the final weights and H/Hinv. Search continues only if all checks pass.
