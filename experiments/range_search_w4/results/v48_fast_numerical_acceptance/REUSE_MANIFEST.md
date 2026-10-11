# Read-only implementation reuse

- `v46/repair_v45.py` SHA256 `ede71ff8abe69927196b01ca5c2d755fff11154c5e918f68d172cf2508dbab37`: control `batch_repair`, and shared `choose`, `score_and_retire`, `propose`, `retire_caps`. Fast path calls these original functions; no rewritten candidate enumeration/argmin/budget logic.
- `v46/precision_check.py` SHA256 `ff743bdaa2f862ea87e6605d6f788404418c7e6bf933d78edd6fa7449d83b0cd`: unchanged empirical same-precision cache and raw-X independent verification criterion. New envelope is specified separately before GPU in ACCEPTANCE_RULES.md.
- `v46/runtime_v46.py`: original FP32 complete-output direct scorer, FP64 full undamped Gram, calibration embedding/capture, block replay, pack/unpack and synchronized timers. Imported as `run_v42`; logging directory and deadline redirected to v48 before use. No model commit/evaluation functions called.
- v46 `original_codes/layer_00.pt` and `layer_03.pt`: trusted M1 codes/scales/zero/g_idx, not M2 grid or approximate reverse inference.
- v46 locked `R1/checkpoint/final`: prefix input replay only. Both local searches initialize from original M1 grids/codes, not final R1 codes. Hashes checked against official committed input and C hashes.
- pristine target: pinned Qwen revision `7ae557604adf67be50417f59c2c2f167def9a775`, BF16 checkpoint storage converted to historical FP16 target; official v18 calibration128×2048 manifest hash `664419273f63df3c0b0746b18bb66b593a5384bbe6ba8afcd6f4905c9479aae3`.
- v43/v44 stopped states: validation only. Historical M2 prefix is rebuilt from original M2 plus committed v44 groups; state code/Q/e/p, actual FP16 delta and X/C provenance checked. No formal repair resumed or historical status changed.
- v47 original stage records and stage coverage: source of the54.03% proposal direct-score bottleneck and prior planning costs; not substituted for v48 timings. Candidate slots removed are not converted into claimed runtime savings.

New fast engine duplicates only orchestration and replaces settlement/maintenance with the explicit v48 acceptance revision. Fixed p refresh is part of that numerical mechanism; C construction is unchanged. New code and protocol hashes are frozen in GPU_START.json. Historical source hashes before/after are SOURCE_AUDIT.json / SOURCE_IMMUTABILITY.json. Python -B prevents historical bytecode writes. All new output files are in v48.
