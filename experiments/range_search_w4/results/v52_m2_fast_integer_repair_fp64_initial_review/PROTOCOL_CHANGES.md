# v51 -> v52

Only the failure handling of the initial FP32-direct/FP64-quadratic check changes. On failure, independently recompute raw-X FP64 loss only for failed rows and apply the pre-existing 2e-5 relative + 1e-12 absolute loss-scale criterion against initial_quad. Keep original FP32 direct initial and quadratic state loss. All candidate/scoring/settle logic, C builder, FP16 grid, refreshes, 32 updates, 128*d evaluations and ordering are inherited without alteration.

The four committed v51 block0 groups are copied into v52 containers with explicit source path/hash/protocol/inheritance metadata. Integer/grid/move tensors are preserved, replay inputs must match, and no repair repeats. Uncommitted layer1 qkv starts at original M2. Historical v51 remains D/read-only.

Budget cap is 10800 seconds shared with v51. v51 is conservatively charged 657.305 seconds (ledger measured 657.3046090602875). Actual active v52 job segments, including regression, failures, recovery, IO and evaluation, are added; application idle waits and CPU environment diagnosis are separate. Finish reserve remains 600 seconds.

Protocol and local core were frozen before validation; passing CPU/GPU regression artifacts unlock production. Later errors never authorize new rules or tolerances. No v50 path is imported.
