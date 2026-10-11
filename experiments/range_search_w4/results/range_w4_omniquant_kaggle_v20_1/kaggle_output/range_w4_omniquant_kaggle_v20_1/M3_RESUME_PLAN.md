# M3 formal block checkpoint and resume plan

This is a plan for the later formal run. It is not an executed resume test.

## Commit boundary

Commit only after a complete decoder block has finished the frozen 20-epoch loop, its final W4 parameters/scales/zero points have been registered, and its 128 calibration inputs have been propagated to the next block. A partial block is not marked complete.

## Atomic checkpoint contents

Write to a temporary file, flush, then atomically rename a manifest that records:

- completed decoder block ID and next block ID;
- the accumulated `omni_state_dict` values for all completed blocks, including LWC `bound_factor` parameters;
- optimizer state for the current block only when a mid-block checkpoint is explicitly supported; the default safe resume boundary is block-level, so no mid-block optimizer resume is claimed;
- frozen protocol/config JSON and its SHA-256;
- official OmniQuant commit, v20 port hashes, runtime harness hash, and environment JSON hash;
- model ID/revision and model safetensor SHA-256;
- calibration manifest SHA-256, ordered token ID hashes, sample count, and sequence length;
- Python, PyTorch, Transformers, tokenizers, CUDA, GPU name/capability, TF32 and determinism flags;
- Python, NumPy, CPU torch, and CUDA RNG states at the boundary;
- output file hashes and save completion marker.

## Resume checks

Before loading optimizer parameters or continuing:

1. Verify every checkpoint file hash and the manifest's atomic completion marker.
2. Require exact model revision/file hashes, protocol/config hash, port hash, calibration hash/order, and compatible optimizer settings.
3. Require `completed_block_id` to be contiguous from zero; reject a missing or duplicate block.
4. Restore model and RNG states, then capture/refresh the next block's inputs using the frozen data order.
5. Refuse resume on any mismatch. Do not silently restart with new samples or different hyperparameters.

The formal run should keep a last-known-good block checkpoint and write the next checkpoint to a new temporary path before replacing the pointer. The preflight in v20.1 does not implement or test this lifecycle.
