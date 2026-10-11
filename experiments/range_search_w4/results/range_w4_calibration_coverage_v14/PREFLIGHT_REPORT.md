# v14 parity and cost preflight

Status: **PASS**.

Frozen rows SHA-256: `9d78172e59cf4979dd57bfa4dd5d2349d98b2de923b0fd599925c4c637faf809`.
Frozen articles SHA-256: `08123d206f603bcc489e53193263abd50a28cbf9a2e915f2bdc980c3bdb83daf`.

## Native alpha=1 parity by projection (K4)

- `q_proj` L2/R82: PASS; checks={'group_count_and_boundaries': True, 'compensated_group_entry_weights': True, 'scale_equal': True, 'zero_equal': True, 'final_fp16_weight_equal': True, 'search_sse_equal': True, 'heldout_sse_equal': True, 'native_hessian_matches_helper': True}; max scale delta=0; group-entry max delta=0; weight mismatches=0; native/custom search SSE=0.0293757878/0.0293757878; held-out SSE=1.10265326/1.10265326.
- `o_proj` L2/R189: PASS; checks={'group_count_and_boundaries': True, 'compensated_group_entry_weights': True, 'scale_equal': True, 'zero_equal': True, 'final_fp16_weight_equal': True, 'search_sse_equal': True, 'heldout_sse_equal': True, 'native_hessian_matches_helper': True}; max scale delta=0; group-entry max delta=0; weight mismatches=0; native/custom search SSE=0.00324758771/0.00324758771; held-out SSE=0.131523132/0.131523132.
- `up_proj` L2/R191: PASS; checks={'group_count_and_boundaries': True, 'compensated_group_entry_weights': True, 'scale_equal': True, 'zero_equal': True, 'final_fp16_weight_equal': True, 'search_sse_equal': True, 'heldout_sse_equal': True, 'native_hessian_matches_helper': True}; max scale delta=0; group-entry max delta=0; weight mismatches=0; native/custom search SSE=0.391436428/0.391436428; held-out SSE=14.2411432/14.2411432.
- `down_proj` L2/R162: PASS; checks={'group_count_and_boundaries': True, 'compensated_group_entry_weights': True, 'scale_equal': True, 'zero_equal': True, 'final_fp16_weight_equal': True, 'search_sse_equal': True, 'heldout_sse_equal': True, 'native_hessian_matches_helper': True}; max scale delta=0; group-entry max delta=0; weight mismatches=0; native/custom search SSE=0.170571417/0.170571417; held-out SSE=1.88169098/1.88169098.

## Hessian build sample by projection and K

- `q_proj`: K4=0.10s, alloc=28.3 MiB, reserved=44.0 MiB; K8=0.01s, alloc=33.1 MiB, reserved=46.0 MiB; K16=0.01s, alloc=39.6 MiB, reserved=50.0 MiB; K32=0.01s, alloc=49.7 MiB, reserved=58.0 MiB
- `o_proj`: K4=0.01s, alloc=31.3 MiB, reserved=44.0 MiB; K8=0.01s, alloc=33.1 MiB, reserved=46.0 MiB; K16=0.01s, alloc=39.6 MiB, reserved=50.0 MiB; K32=0.01s, alloc=49.7 MiB, reserved=58.0 MiB
- `up_proj`: K4=0.01s, alloc=31.3 MiB, reserved=44.0 MiB; K8=0.01s, alloc=33.1 MiB, reserved=46.0 MiB; K16=0.01s, alloc=39.6 MiB, reserved=50.0 MiB; K32=0.02s, alloc=49.7 MiB, reserved=58.0 MiB
- `down_proj`: K4=0.37s, alloc=562.9 MiB, reserved=594.0 MiB; K8=0.16s, alloc=659.6 MiB, reserved=686.0 MiB; K16=0.16s, alloc=769.8 MiB, reserved=798.0 MiB; K32=0.22s, alloc=900.1 MiB, reserved=930.0 MiB

## Workload estimate

- Nominal / reusable / new formal candidates: 1920 / 0 / 1920.
- Estimated formal compute total (preflight formula, no inference): 13.7 min, before analysis/file overhead.
- Actual shared 64-article activation-capture wall: 0.1 min.
- Candidate runtime samples (10 alphas, one L2 row at K4): q_proj 1.94s, o_proj 1.90s, up_proj 1.92s, down_proj 10.77s.

Candidate timings cover custom dynamic GPTQ and search-side loss only. In the preflight timing sample, held-out inputs are also used to time post-selection evaluation; those held-out losses do not select alpha or affect the cost estimate. If any projection parity fails, the formal candidate stage stops.
