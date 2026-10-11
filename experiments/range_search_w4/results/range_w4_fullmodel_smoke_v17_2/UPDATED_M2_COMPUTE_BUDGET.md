# Updated M2 compute budget

Frozen scale: 24 decoder blocks, 168 Linear modules, 304,128 output rows and 3,041,280 row-alpha scores.
The output-shape inventory separates 48 k/v modules at 128x896, 48 q/o modules at 896x896, 48 up/gate modules at 4864x896, and 24 down modules at 896x4864.
Each full-target alpha call produces scores for every output row of that Linear. The projected 1,680 GPTQ executions are not 3,041,280 separate GPU launches. Candidate executions rebuild native Hinv; shared Hessian construction is estimated separately.

| Scenario | GPTQ candidates | Input capture | Shared Hessian | Commit/refresh | Selection/export | Checkpoints | Total |
|---|---:|---:|---:|---:|---:|---:|---:|
| optimistic | 0.29 h | 4.73 h | 0.19 h | 18.93 h | 0.25 h | 0.003 h | 24.40 h |
| central | 0.44 h | 7.10 h | 0.29 h | 28.39 h | 0.38 h | 0.004 h | 36.59 h |
| conservative | 0.59 h | 9.46 h | 0.38 h | 37.85 h | 0.50 h | 0.006 h | 48.79 h |

Conservative estimate: **48.79 hours**; registered gate is **M2_COST_NO_GO** (48 hours maximum). The gate fails, so M2 is not ready for formal full-model search.

Certified measurements: k/v 128x896 median 0.334582s; q/o 896x896 representative q 0.364013s; up/gate 4864x896 one run 0.456453s; down 896x4864 one run 2.078868s. Large-shape timing is a single pilot, and all candidate timing includes native Hinv reconstruction.

Shared-H estimate scales the frozen v15 32-H build time 2.053s by 336 to 689.808s. Sequential capture uses frozen 133.076s full-prefix timing scaled from 16x128 to 128x2048 tokens; refreshed prefixes use four GPTQ module groups per layer. Those terms dominate and are extrapolated. Checkpoint writes assume two 942.3 MiB snapshots at 180 MiB/s. No cost assumption was changed to cross the gate.

Calibration smoke alpha counts for 32 k_proj rows: {"0.1": 0, "0.2": 0, "0.3": 0, "0.4": 0, "0.5": 0, "0.6": 0, "0.7": 0, "0.8": 10, "0.9": 12, "1.0": 10}. These are search-subset outcomes only, not held-out or PPL results.
