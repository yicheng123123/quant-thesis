# v44 reproducibility

All commands use PowerShell from E:/quant-thesis. Historical v42/v43 sources remain read-only. No P3 is accessed.

## Read-only CPU reproduction of the delivered audits

The synthetic tests already ran before the GPU freeze. Re-running CPU tests would rewrite CPU_TESTS.json, which is hashed by GPU_START.json. To inspect the delivered eight checks, read CPU_TESTS.json and cpu_tests.py. For independent test reruns, copy the entire v44 directory to another named experiment directory and run cpu_tests.py there; do not modify the frozen delivery.

The final persisted replay audit is CPU-only, starts no CUDA model, and writes only reports in v44:

    .venv-gptq/Scripts/python.exe -B -X utf8 experiments/range_search_w4/results/v44_m2_single_integer_repair_fullmodel/finalize_v44.py

OUTPUT_AUDIT checks the unchanged runtime/protocol hashes, original source hashes, v43 failure state and committed groups, every accepted move, state-specific rejection blacklist, actual candidate evaluation counts, caps, integer packing and grid equality. It also reaggregates the historical formal baseline windows.

## Exact GPU command used (do not restart this completed stop)

    .venv-gptq/Scripts/python.exe -B -X utf8 experiments/range_search_w4/results/v44_m2_single_integer_repair_fullmodel/run_v44.py

The coordinator enforces one GPU_START clock of 10800 seconds. It does not rerun v41 trajectories or block0 pilot, reuses 14 committed v43 groups, and repairs the failed up/gate group from the original M2 codes. The first invocation validated the saved failure fixture, then started formal repair. It stopped on a material discrepancy in block3 down_proj. run.log and FAILED_ATTEMPTS.jsonl retain the attempt.

Do not run --resume against the same material failure. It would encounter the same check again without an authorized change. No change or automatic continuation is part of this delivery.

## Bounded diagnostic command used

    .venv-gptq/Scripts/python.exe -B -X utf8 experiments/range_search_w4/results/v44_m2_single_integer_repair_fullmodel/diagnose_v44.py

It used the original GPU_START clock, an additional local 240-second ceiling, recreated the committed prefix and input/C hashes, verified the actual code/grid E state, reproduced original full-batch FP32 scores, and compared diagnostic FP64 paired output with the predictor. It neither searches candidates nor commits/accepts a move nor evaluates PPL. Its alternate precision is not an acceptance fallback.

## Source and result locks

- GPU_START.json / IMPLEMENTATION_LOCK.json: implementation, CPU test, source audit and frozen protocol hashes.
- SOURCE_REUSE_AUDIT.json: original calibration/M2 hashes, committed v43 checkpoint hashes, preserved v43 NUMERICAL_STOP/protocol/failure/diagnosis hashes.
- reports/OUTPUT_AUDIT.json: v44 group checkpoint, input and undamped C hashes.
- CRITICAL_FAILURE_formal_3_mlp_down_proj.pt: current pre-proposal state, counted evaluations, tentative proposal and direct loss; no group commit.
- NUMERICAL_DIAGNOSIS.json: exact reproduction and diagnosis only.
- DELIVERY_MANIFEST.json: delivered files and hashes.

The final model save/reload/output/PPL path was not reached. There is no final R checkpoint or WT2/C4 result to reproduce. An eventual complete run would require another explicitly authorized revision; this version stops here.
