"""Write the v17.3 cost-unit and activation-boundary audit artifacts."""
import csv
import hashlib
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent
V16 = OUT.parent / "range_w4_fullmodel_baseline_v16"
V17 = OUT.parent / "range_w4_fullmodel_smoke_v17"
V172 = OUT.parent / "range_w4_fullmodel_smoke_v17_2"
V15 = OUT.parent / "range_w4_quantized_prefix_v15"


def sha_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    old = json.loads((V172 / "M2_COST_ESTIMATE.json").read_text(encoding="utf-8"))
    v15 = json.loads((V15 / "COST_ESTIMATE.json").read_text(encoding="utf-8"))
    stitched = json.loads((OUT / "ROW_STITCH_AUDIT.json").read_text(encoding="utf-8"))
    layers = 24
    modules = 168
    scores = 304_128 * 10
    executions = modules * 10
    unique_group_inputs = 4 * layers
    tokens_v15 = 48 * 128
    tokens_formal = 128 * 2048
    token_scale = tokens_formal / tokens_v15
    sequence_shape_safety = 1.25
    global_safety = 2.0

    # Work-unit audit: v17.2's native candidate call count already used 1680.
    # Its 48.79h cost failure came from an invalid capture source and refresh
    # multiplier, not from treating every row-alpha score as a GPTQ call.
    assert old["native_candidate_core_calls_extrapolated"] == executions
    assert old["row_alpha_candidates"] == scores

    core_base = old["cost_scenarios"]["optimistic"]["core_candidate_seconds"]
    core_conservative = old["cost_scenarios"]["conservative"]["core_candidate_seconds"]
    h_source = v15["hessian_build_seconds_measured"]
    h_source_hessians = 32
    h_source_tokens = 4096
    h_base = h_source * (unique_group_inputs / h_source_hessians) * (tokens_formal / h_source_tokens)
    h_conservative = h_base * global_safety

    fp_capture_source = v15["fp_activation_capture_seconds_measured"]
    qprefix_forward_source = v15["qprefix_full_forward_seconds_measured"]
    target_replay_source = v15["qprefix_target_replay_seconds_measured"]
    replay_source_groups = 16
    capture_base = fp_capture_source * token_scale
    sequential_forward_base = qprefix_forward_source * token_scale
    group_boundary_replay_base = target_replay_source * (unique_group_inputs / replay_source_groups) * token_scale
    capture_conservative = capture_base * sequence_shape_safety * global_safety
    sequential_forward_conservative = sequential_forward_base * sequence_shape_safety * global_safety
    boundary_replay_conservative = group_boundary_replay_base * sequence_shape_safety * global_safety

    selection_base = old["cost_scenarios"]["optimistic"]["selection_export_seconds"]
    selection_conservative = old["cost_scenarios"]["conservative"]["selection_export_seconds"]
    checkpoint_conservative = old["cost_scenarios"]["conservative"]["checkpoint_seconds"]
    model_load_conservative = 2 * v15["model_load_seconds_measured"]

    components = {
        "candidate_full_linear_gptq": core_conservative,
        "unique_group_hessian_build": h_conservative,
        "initial_input_activation_capture": capture_conservative,
        "sequential_m2_full_model_forward": sequential_forward_conservative,
        "dependency_group_boundary_replay_proxy": boundary_replay_conservative,
        "all_row_output_scoring_selection_stitch_commit": selection_conservative,
        "model_load": model_load_conservative,
        "checkpoint_and_io": checkpoint_conservative,
    }
    total = sum(components.values())
    cost = {
        "status": "M2_COST_GO" if total / 3600 <= 48 else "M2_COST_NO_GO",
        "model": "Qwen/Qwen2.5-0.5B-Instruct",
        "calibration_documents": 128,
        "sequence_length": 2048,
        "formal_tokens": tokens_formal,
        "layers": layers,
        "linear_modules": modules,
        "output_rows": 304128,
        "alpha_count": 10,
        "row_alpha_scores": scores,
        "full_linear_alpha_executions": executions,
        "execution_unit": "one full-shape GPTQ execution per alpha per Linear; all output rows are produced by each call",
        "unique_activation_and_hessian_groups": unique_group_inputs,
        "group_layout": "24 x [qkv shared input, o, up/gate shared input, down]",
        "formula": "sum over 168 Linear: 10 full-Linear candidate GPTQ calls + all-row score/select/stitch/commit; add 96 unique input/H groups; sequential layer execution with dependency-local group activations; one set of checkpoint writes",
        "conservative_components_seconds": components,
        "conservative_total_seconds": total,
        "conservative_total_hours": total / 3600,
        "gate_hours": 48,
        "assumptions": {
            "candidate_core": "retains v17.2 per-shape measured full-Linear alpha timings with the existing 2x conservative scenario; no v17.3 speedup credit",
            "capture": f"v15 FP activation capture {fp_capture_source:.6f}s at 48x128 tokens, scaled by token ratio {token_scale:.6f}, sequence-shape factor {sequence_shape_safety}, and 2x global safety",
            "sequential_forward": f"v15 quantized-prefix full forward {qprefix_forward_source:.6f}s at 48x128 tokens, token-scaled, sequence-shape factor {sequence_shape_safety}, and 2x global safety",
            "boundary_replay": f"v15 replay of 16 target blocks scaled to 96 unique activation groups and formal token count, sequence-shape factor {sequence_shape_safety}, and 2x global safety; deliberately conservative proxy",
            "hessian": f"v15 measured {h_source:.6f}s for {h_source_hessians} H builds at {h_source_tokens} token rows; scaled to 96 shared-input H groups and formal tokens, 2x safety",
            "scoring_stitch": "retains v17.2 15-minute central assumption, doubled; covers direct all-row score selection, rowwise stitch and commit pending matched-shape measurement",
            "io": "retains v17.2 two 942.3 MiB checkpoint writes at 180 MiB/s with 2x safety",
        },
        "limitations": [
            "No full-model M2 run was performed.",
            "Full-sequence capture/replay estimates are extrapolated from v15 128-token runs, not matched 2048-token measurements.",
            "The 96-group boundary replay proxy is intentionally conservative and may overlap the sequential forward work; it is retained as a separate upper-cost allowance.",
            "The cost model does not claim measured full-model wall time.",
        ],
        "supersedes": "v17.2 M2_COST_ESTIMATE.json for the full-model decision only; v17.2 files remain immutable",
        "v17_2_cost_status": old["status"],
        "v17_2_conservative_hours": old["conservative_hours"],
        "v17_2_invalid_terms": [
            "133.076 seconds was copied from v15 prefix_native_quantization_seconds_measured, which is native GPTQ quantization time, not activation capture time.",
            "HISTORICAL_TIMING_EVIDENCE.json contains no native_full_prefix_seconds key, so the v17.2 fallback hard-coded 133.076 seconds.",
            "The 4x full-prefix refresh term treated four within-layer dependency boundaries as four extra complete full-model forward passes.",
            "Hessian build scale counted 168 projections separately despite qkv and up/gate sharing their input and H; 96 unique group inputs are required.",
        ],
        "source_hashes": {
            "v17_2_M2_COST_ESTIMATE.json": sha_file(V172 / "M2_COST_ESTIMATE.json"),
            "v17_2_run_m2_smoke.py": sha_file(V172 / "run_m2_smoke.py"),
            "v15_COST_ESTIMATE.json": sha_file(V15 / "COST_ESTIMATE.json"),
            "v16_FULLMODEL_SEARCH_PLAN.md": sha_file(V16 / "FULLMODEL_SEARCH_PLAN.md"),
            "row_stitch_audit": sha_file(OUT / "ROW_STITCH_AUDIT.json"),
        },
    }
    (OUT / "CORRECTED_FULLMODEL_COST.json").write_text(json.dumps(cost, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    flow = """# M2 activation dependency and refresh audit

## Fixed execution units

- 24 decoder layers, 7 quantized Linear modules per layer: 168 Linears.
- 304,128 output rows and 3,041,280 row-alpha scores.
- 1,680 full-Linear candidate GPTQ executions: 10 full-shape alpha calls for each Linear. Each call emits candidates for every output row simultaneously.
- The 32-row v17.2 landscape came from 10 full `[128,896]` calls. The 32 rows were a scoring subset, not 32 separate GPTQ invocations.
- A Linear is searched completely, each output row selects its alpha, Q/scale/zero are row-wise stitched, the Linear is committed once, then dependent activations advance.
- Output-row score or stitch chunks cannot trigger a forward refresh.

## Row separability check

The installed GPTQ source uses a shared column Hessian/Hinv. `find_params` uses per-channel reductions; compensation updates `W1[:, i:]` with a row vector times an Hinv row and has no output-row reduction. This is mathematically separable across output rows under the frozen settings.

The existing ten full-shape candidates were rescored for all 128 output rows. They produced 1,280 row-alpha scores and a complete selected row-wise Linear. Four rows fixed by the earlier v17.2 protocol (0, 1, 17, 31) were also independently quantized for all 10 alphas (40 row calls). In all 40 comparisons, FP16 weights, zero points, g_idx, and local output deltas matched bitwise; selected alpha matched on all four rows. FP32 scales were not bitwise identical in every case, but all were within rtol `1e-6`, atol `1e-9` (largest relative difference about `2.5e-7`). The first small weight divergence appears at a later group after error compensation and is at most `1.2e-7` in the traced row; this is shape-dependent FP32 arithmetic, not cross-row coupling. The formal shared implementation must retain full-Linear shape, so this one-row kernel difference is not introduced by row-wise stitching.

## Per-layer dependency schedule

| Boundary | Shared input / operation | Required action |
|---|---|---|
| Layer input to q/k/v | One post-input-LayerNorm activation | Capture once; build one H/Hinv source; run 10 candidates per each of three Linears; stitch and commit q/k/v as one group |
| q/k/v to o | Current Q/K/V jointly determine attention context | Advance the attention subgraph once with committed q/k/v; capture o input |
| o to up/gate | o output changes the residual and post-attention normalization | Recompute o output and normalization once; capture one input shared by up and gate |
| up/gate to down | Both outputs jointly determine the gated MLP product | Commit both, compute activation product once; capture down input |
| down to next layer | down output changes the residual stream | Commit down and advance once to the next layer input; this same stream supplies the next layer's q/k/v input |

There are 96 distinct group-input/Hessian states (four per layer), not 168. There are 95 inter-group/layer dependency transitions through the final quantized block (the last block needs no post-commit forward for calibration selection). These are activation-boundary computations in a layer-wise sequential pass. They are not 95 full-model replays. The required full-model implementation is not present yet, so the exact runtime of its local boundary computations remains an estimate.

Sibling processing order `[k,v,q]` and `[up,gate]` is an implementation order. Forward dependencies require shared inputs for each sibling group and only require refresh after the whole sibling group is committed.

## v17.2 cost correction

The v17.2 candidate count was already correct: 1,680 full-Linear alpha calls. Its 48.79-hour estimate is superseded because the capture source was mislabeled: `133.076s` came from v15 `prefix_native_quantization_seconds_measured`, which measures native GPTQ quantization, not activation capture. The v17.2 loader found no `native_full_prefix_seconds` key in its historical evidence and therefore used that 133.076-second fallback. It then token-scaled this quantization time as if it were activation capture and charged four additional full-model passes. The corrected estimate uses the v15 activation capture, quantized-prefix forward, target-block replay, and Hessian measurements with explicit token/shape factors and safety multipliers. It keeps the candidate core estimate and 48-hour gate unchanged.

The resulting conservative estimate is recorded in `CORRECTED_FULLMODEL_COST.json`. It is a cost estimate, not measured full-model runtime; the 2048-token and layer-local replay terms still need matched-shape validation before a formal run.
"""
    (OUT / "DEPENDENCY_REFRESH_AUDIT.md").write_text(flow, encoding="utf-8")

    cost_md = "# Corrected full-model M2 cost estimate\n\n"
    cost_md += f"Conservative estimate: **{total / 3600:.3f} hours**; fixed gate: 48 hours; status: **{cost['status']}**. No formal full-model quantization or PPL was run.\n\n"
    cost_md += "| Component | Conservative seconds | Basis |\n|---|---:|---|\n"
    basis = {
        "candidate_full_linear_gptq": "1,680 measured-shape calls with v17.2 conservative 2x factor",
        "unique_group_hessian_build": "96 input/H groups; v15 Hessian time token-scaled, 2x factor",
        "initial_input_activation_capture": "v15 FP capture, token-scaled, sequence factor 1.25, 2x factor",
        "sequential_m2_full_model_forward": "v15 quantized full forward, token-scaled, sequence factor 1.25, 2x factor",
        "dependency_group_boundary_replay_proxy": "v15 replay scaled from 16 to 96 groups and formal tokens, sequence factor 1.25, 2x factor",
        "all_row_output_scoring_selection_stitch_commit": "v17.2 15-minute central assumption, doubled",
        "model_load": "v15 measured model load, doubled",
        "checkpoint_and_io": "v17.2 two model-sized writes at 180 MiB/s, doubled",
    }
    for key, seconds in components.items():
        cost_md += f"| {key.replace('_', ' ')} | {seconds:.2f} | {basis[key]} |\n"
    cost_md += "\nWork units: 3,041,280 per-row alpha scores and 1,680 full-Linear GPTQ executions. No per-row GPTQ timing is multiplied by 304,128 rows.\n\n"
    cost_md += "The old v17.2 value is superseded. It used the v15 native GPTQ quantization time as an activation-capture source and charged four extra full-model forward passes for module-group boundaries. The revised schedule shares q/k/v and up/gate inputs and advances the model layer by layer. The revised estimate retains a conservative group-replay allowance; the precise 2048-token replay time is still extrapolated.\n"
    (OUT / "UPDATED_FULLMODEL_COST.md").write_text(cost_md, encoding="utf-8")

    patch = [
        {"artifact": "audit_work_units_and_stitch.py", "purpose": "Re-score all cached full-Linear candidates and compare fixed spot rows with independent rowwise GPTQ"},
        {"artifact": "diagnose_row_scale_rounding.py", "purpose": "Trace FP32 scale ULP differences to compensated group entry"},
        {"artifact": "write_cost_unit_audit.py", "purpose": "Correct execution units and activation cost source/refresh model"},
    ]
    with (OUT / "PATCH_MANIFEST_AUDIT.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["artifact", "purpose"])
        writer.writeheader()
        writer.writerows(patch)
    print(json.dumps({"cost_hours": total / 3600, "gate": cost["status"], "candidate_execs": executions, "row_scores": scores}, indent=2))


if __name__ == "__main__":
    main()
