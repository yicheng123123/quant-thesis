v52 deployment serialization adaptation, frozen before continuation.

The original run stopped on raw JSON equality after all repair, terminal checks, pack/grid checks, reload, and sample output regression passed. The exact stop is retained in diagnostics/CONFIG_ASSERTION_STOP.json; no repair is repeated.

Transformers 5.16.1 maps the legacy torch_dtype, rope_theta/rope_scaling and disabled sliding_window configuration to dtype, rope_parameters and 24 full_attention layer_types. It also serializes default pad_token_id=null. All three source configurations, when loaded by the same actual AutoConfig, have exactly equal to_dict fields after excluding ONLY the original _name_or_path and transformers_version metadata. No new field is ignored and no source/config/model is changed. See CONFIG_SCHEMA_DIAGNOSTIC.json and local installed configuration_qwen2.py.

finish_v52.py repeats that full loaded-config comparison, then continues the original tensor/grid comparisons and unchanged evaluator. This is a deployment schema adapter, not a numerical protocol amendment. Frozen run_v52.py, repair core, numerical thresholds and output checkpoint bytes remain unchanged. If loaded fields differ, stop. If named model tensors differ from R2, run each frozen evaluator exactly once after locking. The original failed run and all continuation costs remain in the ledger.
