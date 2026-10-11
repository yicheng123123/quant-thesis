# M2 state and mutation boundaries

## Immutable across the ten alpha candidates for one Linear

- Pristine FP16 weight `W` from the frozen model checkpoint.
- Raw FP32 Hessian `H` built from the current dependency-correct input activation.
- Prepared dead-column mask, damped Hessian, its Cholesky factor, and the native `block_cholesky_inverse` output. These depend on `H` and frozen GPTQ settings, not on alpha.
- Frozen activation `X` used to score every candidate for this Linear.

The prepared inverse is the native method output immediately before GPTQ's final Cholesky. The optimized path returns that immutable tensor from an instance hook. The installed `GPTQ.quantize()` and final Cholesky remain in use.

## Fresh candidate-local state for each alpha

- A new GPTQ object and its FP32 working weight copy.
- A fresh clone of `H`, which native GPTQ may modify while preparing `Hinv`.
- Quantizer group parameters, zero points, `g_idx`, error buffers, and all compensation updates.
- The candidate's final quantized weight and metadata.

Candidates run from pristine `W` and `H`; no candidate receives a prior candidate's compensated weights or mutated Hessian. Candidate order reversal produced exact matching tensors.

## Commit boundary

After all ten full-Linear candidates have produced every output row, select alpha independently per row using the frozen search loss. Stitch the selected row of `Q`, scale, zero, and `g_idx`, then commit the complete Linear once. Only then advance the dependent activation graph. Score or stitch chunking does not trigger a forward refresh.

The small smoke only commits rows 0–31 in its selected-row check; all other rows are verified pristine. Full-model work units and cost use all output rows.
