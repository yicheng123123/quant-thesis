# M2 pure range-search run report

Status: **COMPLETE / correctness PASS / reload PASS**.

- Frozen method: dynamic native symmetric GPTQ G128 with shared alpha grid `0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0`. Each alpha ran a full Linear from pristine candidate state; all rows were scored, selected independently, stitched once per Linear, and committed before dependency refresh.
- Processed 24 layers / 168 linears / 304,128 rows. Formal counts: 1,680 full-Linear candidate GPTQ runs, 3,041,280 row-alpha score slots, 35 alpha=1 native spot references. Frozen calibration manifest SHA256 `664419273f63df3c0b0746b18bb66b593a5384bbe6ba8afcd6f4905c9479aae3`.
- Actual wall including load, prefix capture/replay, search, checkpoint, final save/reload: 2115.935 s (35.27 min); conservative pre-run estimate 1.338 h under the unchanged 48 h gate. M2/M1 measured calibration wall ratio: 3.261×.
- Final checkpoint SHA256: `810d6329cdc51d995543dbb701e2368568b19459a44f340df1a8a0019d447eed`; reload max/mean absolute logit difference on 64 fixed tokens: 0.0/0.0.
- WT2: PPL 13.76055605, NLL/token 2.621806242; vs M1 -0.8917% PPL. C4 fixed-budget: PPL 22.03288316, NLL/token 3.092536027; vs M1 -0.9042% PPL.
- Final alpha counts: `{"0.4": 1, "0.5": 30, "0.6": 554, "0.7": 3044, "0.8": 54155, "0.9": 226034, "1.0": 20310}`. Local mean/median relative output-MSE gain: 12.827%/12.662%; 283,818 improved rows, 20,310 tied rows.

## Measured stage wall totals

- `full_linear_alpha_candidate_gptq`: 938.43 s
- `all_row_direct_output_scoring`: 718.77 s
- `unique_group_hessian_build`: 85.40 s
- `up_gate_commit_to_down_input`: 56.93 s
- `down_commit_to_next_layer_input`: 29.64 s
- `qkv_commit_to_o_input`: 22.52 s
- `o_commit_to_up_gate_input`: 21.89 s
- `alpha1_native_parity_reference`: 20.17 s
- `shared_native_hinv_prepare`: 4.61 s
- `streaming_per_row_argmin_update`: 1.47 s
- `model_load`: 1.31 s
- `atomic_layer_checkpoint_io`: 1.10 s
- `final_dense_model_save`: 0.76 s
- `initial_embedding_capture`: 0.55 s
- `qkv_shared_input_capture`: 0.34 s
- `full_linear_row_stitch`: 0.09 s
- `whole_linear_commit`: 0.01 s

Reproduce both formal methods in frozen order with `& '.venv-gptq\Scripts\python.exe' -X utf8 'experiments\range_search_w4\results\range_w4_fullmodel_v18\run_formal.py'`; this runner has no per-method CLI and resumes completed layer checkpoints. Evaluate with `& '.venv-gptq\Scripts\python.exe' -X utf8 'experiments\range_search_w4\results\range_w4_fullmodel_v18\evaluate_v18.py'`; regenerate derived tables and reports with the same command using `finalize_reports_v18.py`. Detailed row/module tables, alpha distribution, candidate landscapes, metadata, and checkpoints are retained in v18.
