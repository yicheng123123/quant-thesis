"""Build the v18 formal cost model from matched-shape timing evidence."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import statistics
import time
from pathlib import Path


OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[3]
TIMING = OUT / "MATCHED_SHAPE_TIMING.csv"
PRECHECK = OUT / "MATCHED_SHAPE_PRECHECK.json"
PROTOCOL = OUT / "PROTOCOL.json"
GATE_HOURS = 48.0


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def nearest_rank(values: list[float], q: float) -> float:
    values = sorted(float(x) for x in values)
    if not values:
        raise ValueError("empty timing group")
    return values[max(0, math.ceil(q * len(values)) - 1)]


def timings():
    rows = []
    for path in (TIMING, OUT / "KV_SHAPE_TIMING.csv"):
        with path.open("r", newline="", encoding="utf-8") as f:
            rows.extend(csv.DictReader(f))
    grouped: dict[tuple[str, str], list[float]] = {}
    for row in rows:
        key = (row["stage"], row["module"])
        grouped.setdefault(key, []).append(float(row["wall_seconds"]))
    return rows, grouped


def stat(grouped, stage: str, module: str, q: float) -> float:
    values = grouped[(stage, module)]
    return nearest_rank(values, q)


def min_stat(grouped, stage: str, module: str) -> float:
    return min(grouped[(stage, module)])


def durable_io_probe(source: Path, target: Path) -> float:
    if target.exists():
        raise RuntimeError(f"I/O probe target already exists: {target}")
    start = time.perf_counter()
    with source.open("rb", buffering=0) as inp, target.open("xb", buffering=0) as out:
        while True:
            chunk = inp.read(8 << 20)
            if not chunk:
                break
            out.write(chunk)
        out.flush()
        os.fsync(out.fileno())
    elapsed = time.perf_counter() - start
    target.unlink()
    return elapsed


def main():
    protocol = read_json(PROTOCOL)
    pre = read_json(PRECHECK)
    if pre.get("status") != "PASS" or not all(pre["checks"].values()):
        raise SystemExit("matched-shape correctness is not PASS")

    rows, grouped = timings()
    alpha_n = len(protocol["m2"]["alpha_grid"])
    layers = 24
    shape_counts = {"square_896": 48, "kv_128x896": 48,
                    "up_4864x896": 48, "down_896x4864": 24}
    if sum(shape_counts.values()) != 168:
        raise AssertionError("projection inventory mismatch")
    full_execs = 168 * alpha_n
    row_alpha_scores = 304_128 * alpha_n
    transitions = {"qkv_commit_to_o": 24, "o_commit_to_up_gate": 24,
                   "up_gate_commit_to_down": 24, "down_commit_to_next_layer": 23}
    if sum(transitions.values()) != 95:
        raise AssertionError("forward dependency transition count mismatch")

    # Candidate and direct output-score measurements were repeated ten times
    # per exact Linear shape. Exclude the separately recorded warmups.
    candidate_keys = {
        "square_896": "L0 q_proj",
        "kv_128x896": "L0 self_attn.k_proj",
        "up_4864x896": "L0 up_proj",
        "down_896x4864": "L0 down_proj",
    }
    candidate_count = {"square_896": 48, "kv_128x896": 48,
                       "up_4864x896": 48, "down_896x4864": 24}
    rows_per_linear = {"square_896": 896, "kv_128x896": 128,
                       "up_4864x896": 4864, "down_896x4864": 896}
    shape_cost = {}
    for shape, module in candidate_keys.items():
        c_med = stat(grouped, "optimized_full_linear_candidate", module, .50)
        c_p95 = stat(grouped, "optimized_full_linear_candidate", module, .95)
        s_med = stat(grouped, "all_row_direct_scoring", module, .50)
        s_p95 = stat(grouped, "all_row_direct_scoring", module, .95)
        u_med = stat(grouped, "streaming_argmin_best_row_update", module, .50)
        u_p95 = stat(grouped, "streaming_argmin_best_row_update", module, .95)
        stitch = stat(grouped, "streaming_final_row_stitch", module, .50)
        commit = stat(grouped, "full_linear_commit", module, .50)
        shape_cost[shape] = {
            "linear_count": candidate_count[shape],
            "candidate_seconds_each": {"median": c_med, "p95_nearest_rank": c_p95},
            "score_seconds_each": {"median": s_med, "p95_nearest_rank": s_p95},
            "row_update_seconds_each": {"median": u_med, "p95_nearest_rank": u_p95},
            "stitch_seconds_each_measured": stitch,
            "commit_seconds_each_measured": commit,
            "candidate_execution_count": candidate_count[shape] * alpha_n,
            "output_rows_per_linear": rows_per_linear[shape],
            "row_alpha_score_slots": candidate_count[shape] * rows_per_linear[shape] * alpha_n,
            "candidate_seconds_each_min": min_stat(grouped, "optimized_full_linear_candidate", module),
            "score_seconds_each_min": min_stat(grouped, "all_row_direct_scoring", module),
            "row_update_seconds_each_min": min_stat(grouped, "streaming_argmin_best_row_update", module),
        }

    # Four unique activation/H groups per layer: qkv, o, up/gate, down.
    # Hessian times are one matched 128 x 2048 observation per group type;
    # use a 2x cross-layer factor in the conservative scenario.
    h_keys = {"qkv": "L0 q_proj", "o": "L0 o_proj",
              "up_gate": "L0 up_proj", "down": "L0 down_proj"}
    h_build = {key: stat(grouped, "hessian_build_128x2048", module, 1.0)
               for key, module in h_keys.items()}
    hinv = {
        "qkv": stat(grouped, "shared_native_hinv_prepare", "L0 q_proj", 1.0),
        "o": stat(grouped, "shared_native_hinv_prepare", "L0 q_proj", 1.0),
        # Both use d_in=896. Use the slower matched measurement for all 896-wide inputs.
        "up_gate": max(stat(grouped, "shared_native_hinv_prepare", "L0 q_proj", 1.0),
                       stat(grouped, "shared_native_hinv_prepare", "L0 up_proj", 1.0)),
        "down": stat(grouped, "shared_native_hinv_prepare", "L0 down_proj", 1.0),
    }
    group_counts = {"qkv": 24, "o": 24, "up_gate": 24, "down": 24}

    refresh_map = {
        "qkv_commit_to_o": ("dependency_forward_capture", "qkv_commit_to_o_input"),
        "o_commit_to_up_gate": ("dependency_forward_capture", "o_commit_to_up_gate_input"),
        "up_gate_commit_to_down": ("dependency_forward_capture", "up_gate_commit_to_down_input"),
        "down_commit_to_next_layer": ("down_commit_to_next_layer_input", "L0 down_proj to L1 qkv"),
    }
    refresh_measurements = {}
    for name, (stage, module) in refresh_map.items():
        refresh_measurements[name] = stat(grouped, stage, module, 1.0)

    # Native M1 alpha=1 uses its own pristine model and self-propagated prefix.
    native_by_projection = {
        "q": stat(grouped, "M1_native_alpha1_gptq", "L0 self_attn.q_proj", 1.0),
        "k": stat(grouped, "M1_native_alpha1_gptq", "L0 self_attn.k_proj", 1.0),
        "v": stat(grouped, "M1_native_alpha1_gptq", "L0 self_attn.v_proj", 1.0),
        "o": stat(grouped, "M1_native_alpha1_gptq", "L0 self_attn.o_proj", 1.0),
        "up": stat(grouped, "M1_native_alpha1_gptq", "L0 mlp.up_proj", 1.0),
        "gate": stat(grouped, "M1_native_alpha1_gptq", "L0 mlp.gate_proj", 1.0),
        "down": stat(grouped, "M1_native_alpha1_gptq", "L0 mlp.down_proj", 1.0),
    }
    native_counts = {name: 24 for name in native_by_projection}

    # Measure a durable, uncached-throughput proxy on the actual checkpoint volume.
    # We use a 4x slower sustained-rate allowance and 2x bytes for serialization/
    # small-file overhead. Two methods each write one final model plus all rows
    # as 24 layer checkpoints (estimated from the frozen architecture's linears).
    model_file = ROOT / ".hf_cache" / "hub" / "models--Qwen--Qwen2.5-0.5B-Instruct" / "snapshots" / protocol["model"]["revision"] / "model.safetensors"
    model_bytes = model_file.stat().st_size
    layer_linear_bytes = 24 * (2 * 896 * 896 + 2 * 128 * 896 + 3 * 4864 * 896) * 2
    final_models_bytes = 2 * model_bytes
    layer_checkpoints_bytes = 2 * layer_linear_bytes
    io_probe_target = OUT / "checkpoints" / "io_probe_flushed.safetensors"
    durable_copy_seconds = durable_io_probe(model_file, io_probe_target)
    durable_copy_bytes = model_bytes
    observed_bytes_per_second = durable_copy_bytes / durable_copy_seconds
    io_payload_bytes = final_models_bytes + layer_checkpoints_bytes
    io_base_seconds = io_payload_bytes / (observed_bytes_per_second / 4.0)
    io_safety_seconds = io_base_seconds * 2.0

    # Native alpha=1 candidate timings and capture measurements have only one
    # layer's observations. Conservative scenario applies a 2x cross-layer factor.
    m1_native_base = sum(native_by_projection[k] * native_counts[k] for k in native_by_projection)
    m1_score_base = sum(
        shape_cost[k]["score_seconds_each"]["median"] * candidate_count[k]
        for k in shape_cost
    )
    m2_parity_base = 5 * sum(native_by_projection.values())
    m1_h_base = sum(h_build[k] * group_counts[k] for k in group_counts)
    m1_capture_base = (
        stat(grouped, "calibration_input_embedding", "embedding", 1.0)
        + stat(grouped, "shared_qkv_activation_capture", "L0 q/k/v shared input", 1.0)
        + sum(refresh_measurements[k] * count for k, count in transitions.items())
    )

    m2_candidate_med = sum(shape_cost[k]["candidate_seconds_each"]["median"] * shape_cost[k]["candidate_execution_count"] for k in shape_cost)
    m2_candidate_p95 = sum(shape_cost[k]["candidate_seconds_each"]["p95_nearest_rank"] * shape_cost[k]["candidate_execution_count"] for k in shape_cost)
    m2_candidate_min = sum(shape_cost[k]["candidate_seconds_each_min"] * shape_cost[k]["candidate_execution_count"] for k in shape_cost)
    m2_score_med = sum(shape_cost[k]["score_seconds_each"]["median"] * shape_cost[k]["candidate_execution_count"] for k in shape_cost)
    m2_score_p95 = sum(shape_cost[k]["score_seconds_each"]["p95_nearest_rank"] * shape_cost[k]["candidate_execution_count"] for k in shape_cost)
    m2_score_min = sum(shape_cost[k]["score_seconds_each_min"] * shape_cost[k]["candidate_execution_count"] for k in shape_cost)
    m2_update_med = sum(shape_cost[k]["row_update_seconds_each"]["median"] * shape_cost[k]["candidate_execution_count"] for k in shape_cost)
    m2_update_p95 = sum(shape_cost[k]["row_update_seconds_each"]["p95_nearest_rank"] * shape_cost[k]["candidate_execution_count"] for k in shape_cost)
    m2_update_min = sum(shape_cost[k]["row_update_seconds_each_min"] * shape_cost[k]["candidate_execution_count"] for k in shape_cost)
    m2_stitch_commit_base = sum(
        (shape_cost[k]["stitch_seconds_each_measured"] + shape_cost[k]["commit_seconds_each_measured"])
        * shape_cost[k]["linear_count"] for k in shape_cost
    )
    m2_h_base = sum(h_build[k] * group_counts[k] for k in group_counts)
    m2_hinv_base = sum(hinv[k] * group_counts[k] for k in group_counts)
    m2_refresh_base = sum(refresh_measurements[k] * count for k, count in transitions.items())
    input_prep_base = (stat(grouped, "calibration_input_embedding", "embedding", 1.0)
                       + stat(grouped, "shared_qkv_activation_capture", "L0 q/k/v shared input", 1.0))
    model_load = float(pre["model_load_seconds"])
    m1_commit_base = sum(
        stat(grouped, "full_linear_commit", candidate_keys[shape], 1.0) * candidate_count[shape]
        for shape in candidate_keys
    )

    safety = {
        "candidate_scoring_selection_cross_layer_factor": 2.0,
        "single_observation_hessian_hinv_native_gptq_factor": 2.0,
        "activation_transition_factor": 2.0,
        "stitch_commit_factor": 2.0,
        "durable_io_factor": 2.0,
    }
    m1_central = {
        "input_embedding_and_initial_qkv_capture": input_prep_base,
        "model_load": model_load,
        "unique_group_hessian_build": m1_h_base,
        "native_alpha1_full_linear_gptq": m1_native_base,
        "native_m1_all_row_output_scoring": m1_score_base,
        "whole_linear_commits": m1_commit_base,
        "dependency_boundary_refresh": sum(refresh_measurements[k] * count for k, count in transitions.items()),
        "commit_and_resume_checkpoint_io": io_base_seconds / 2.0,
    }
    m1_conservative = {
        "input_embedding_and_initial_qkv_capture": input_prep_base * 2,
        "model_load": model_load * 2,
        "unique_group_hessian_build": m1_h_base * 2,
        "native_alpha1_full_linear_gptq": m1_native_base * 2,
        "native_m1_all_row_output_scoring": m1_score_base * 2,
        "whole_linear_commits": m1_commit_base * 2,
        "dependency_boundary_refresh": m1_capture_base - input_prep_base,
        "commit_and_resume_checkpoint_io": io_safety_seconds / 2.0,
    }
    # The capture term above includes each measured transition and applies 2x.
    m1_conservative["dependency_boundary_refresh"] *= 2

    m2_central = {
        "input_embedding_and_initial_qkv_capture": input_prep_base,
        "model_load": model_load,
        "unique_group_hessian_build": m2_h_base,
        "shared_native_hinv_prepare": m2_hinv_base,
        "full_linear_alpha_candidates": m2_candidate_med,
        "all_row_output_scoring": m2_score_med,
        "streaming_row_argmin_updates": m2_update_med,
        "row_stitch_and_single_linear_commit": m2_stitch_commit_base,
        "alpha1_native_parity_spot_checks": m2_parity_base,
        "dependency_boundary_refresh": m2_refresh_base,
        "checkpoint_and_final_model_io": io_base_seconds / 2.0,
    }
    m2_conservative = {
        "input_embedding_and_initial_qkv_capture": input_prep_base * 2,
        "model_load": model_load * 2,
        "unique_group_hessian_build": m2_h_base * 2,
        "shared_native_hinv_prepare": m2_hinv_base * 2,
        "full_linear_alpha_candidates": m2_candidate_p95 * 2,
        "all_row_output_scoring": m2_score_p95 * 2,
        "streaming_row_argmin_updates": m2_update_p95 * 2,
        "row_stitch_and_single_linear_commit": m2_stitch_commit_base * 2,
        "alpha1_native_parity_spot_checks": m2_parity_base * 2,
        "dependency_boundary_refresh": m2_refresh_base * 2,
        "checkpoint_and_final_model_io": io_safety_seconds / 2.0,
    }
    optimistic = {
        "M1": sum(m1_central.values()),
        "M2": (input_prep_base + model_load + m2_h_base + m2_hinv_base + m2_candidate_min
               + m2_score_min + m2_update_min + m2_stitch_commit_base + m2_parity_base
               + m2_refresh_base + io_base_seconds / 2.0),
        "basis": "minimum observed candidate/score/update timings; all single-observation costs use measured values; no discount factor",
    }
    m1_central_total = sum(m1_central.values())
    m1_conservative_total = sum(m1_conservative.values())
    m2_central_total = sum(m2_central.values())
    m2_conservative_total = sum(m2_conservative.values())
    gate = "M2_COST_GO" if m2_conservative_total / 3600 <= GATE_HOURS else "M2_COST_NO_GO"
    payload = {
        "status": gate,
        "v17_3_estimate_hours_superseded": 2.135,
        "v17_3_estimate_note": "Replaced because matched full-shape candidate, scoring, selection, stitch, commit and 2048-token dependency timings are now available.",
        "gate_hours_unchanged": GATE_HOURS,
        "work_units": {
            "layers": layers,
            "linears": 168,
            "output_rows": 304_128,
            "alpha_count": alpha_n,
            "row_alpha_score_slots": row_alpha_scores,
            "full_linear_alpha_gptq_executions": full_execs,
            "unique_input_hessian_groups": 96,
            "dependency_refresh_transitions_per_method": 95,
            "dependency_transition_counts": transitions,
            "linear_shape_counts": shape_counts,
        },
        "matched_shape_measurements": shape_cost,
        "hessian_build_seconds_per_unique_group": h_build,
        "hinv_prepare_seconds_per_unique_group": hinv,
        "native_m1_full_linear_seconds_by_projection": native_by_projection,
        "m1_projection_counts": native_counts,
        "dependency_transition_seconds_each": refresh_measurements,
        "io_measurement": {
            "method": "actual model.safetensors copied on checkpoint volume; destination flushed with os.fsync",
            "durable_copy_seconds": durable_copy_seconds,
            "observed_bytes_per_second": observed_bytes_per_second,
            "model_safetensors_bytes": model_bytes,
            "estimated_layer_linear_checkpoint_bytes_per_method": layer_linear_bytes,
            "two_final_model_bytes": final_models_bytes,
            "two_methods_layer_checkpoint_bytes": layer_checkpoints_bytes,
            "payload_bytes_total": io_payload_bytes,
            "sustained_rate_divisor": 4,
            "conservative_io_multiplier": 2,
            "conservative_seconds_all_methods": io_safety_seconds,
            "conservative_seconds_per_method": io_safety_seconds / 2.0,
            "probe_file_removed": not io_probe_target.exists(),
        },
        "scenario_seconds": {
            "optimistic": optimistic,
            "central": {"M1": m1_central, "M2": m2_central,
                        "M1_total": m1_central_total, "M2_total": m2_central_total,
                        "combined_total": m1_central_total + m2_central_total},
            "conservative": {"M1": m1_conservative, "M2": m2_conservative,
                             "M1_total": m1_conservative_total, "M2_total": m2_conservative_total,
                             "combined_total": m1_conservative_total + m2_conservative_total},
        },
        "scenario_hours": {
            "optimistic_M1": optimistic["M1"] / 3600,
            "optimistic_M2": optimistic["M2"] / 3600,
            "central_M1": m1_central_total / 3600,
            "central_M2": m2_central_total / 3600,
            "conservative_M1": m1_conservative_total / 3600,
            "conservative_M2": m2_conservative_total / 3600,
            "conservative_M1_plus_M2": (m1_conservative_total + m2_conservative_total) / 3600,
        },
        "safety_factors": safety,
        "calculation_notes": [
            "Each of 10 alpha candidate runs is a full Linear GPTQ execution producing all output rows.",
            "Candidate and direct score timings use measured per-shape P95 from 10 repeats and a 2x cross-layer factor for the conservative scenario.",
            "Hessian, Hinv, native M1 GPTQ and dependency timings have one layer observation; conservative scenario applies 2x.",
            "The 128x896 k/v linears were separately measured at full shape; no square-projection proxy is used for them.",
            "All 128 calibration documents have length 2048; capture refresh timings are already end-to-end over the 128 documents.",
            "q/k/v and up/gate share inputs/H. Refresh happens only after whole-Linear row stitch and commit.",
            "The 95 transitions are 24 qkv-to-o, 24 o-to-up/gate, 24 up/gate-to-down, and 23 down-to-next-layer.",
            "M2 conservative runtime includes 35 full native alpha=1 parity reference executions across layers 0, 6, 12, 18, and 23.",
            "The 48-hour gate applies to conservative M2 calibration cost; the combined M1+M2 figure is also reported.",
            "Evaluation time is outside the calibration gate and will be measured in the formal evaluator run.",
            "This is an extrapolation from matched L0 shapes, not an observed full-model runtime.",
        ],
    }
    (OUT / "FINAL_COST_MODEL.json").write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    report = f"""# v18 matched-shape precheck and cost gate

Correctness: **PASS** (all {len(pre['checks'])} frozen precheck conditions true).

## Work units

- 168 Linear modules, 304,128 output rows, and 10 alpha values.
- **3,041,280 row-alpha score slots** are produced by **1,680 full-Linear GPTQ executions**.
- One candidate execution returns every output row for that Linear. Per-row argmin and final stitch happen after scoring; they do not launch GPTQ per row.
- 96 unique input/H groups: per layer `[qkv, o, up/gate, down]`.
- 95 actual dependency transitions per method: 24 qkv→o, 24 o→up/gate, 24 up/gate→down, and 23 down→next layer.

## Matched measurements

All candidate and direct output-scoring timings use full output rows at 128×2048 calibration shape, with separate row update, stitch, and commit measurements. q/up/down data are in `MATCHED_SHAPE_TIMING.csv` and `MATCHED_SHAPE_PRECHECK.json`; the `[128,896]` k/v addendum is in `KV_SHAPE_TIMING.csv` and `KV_SHAPE_PRECHECK.json`.

The observed refresh calls are one full decoder-block replay at each actual dependency edge. The qkv and up/gate siblings share the same X/H. The next boundary is not refreshed for each row batch.

## Cost result

- M2 central: **{m2_central_total/3600:.3f} h**.
- M2 conservative: **{m2_conservative_total/3600:.3f} h**; frozen 48 h gate: **{gate}**.
- M1 conservative: **{m1_conservative_total/3600:.3f} h**.
- Combined M1+M2 conservative: **{(m1_conservative_total+m2_conservative_total)/3600:.3f} h**.
- Evaluation time is outside the M2 calibration gate and will be recorded from actual WT2/C4 runs.

The old v17.3 **2.135 h** estimate is superseded because it used different conservative assumptions and did not have the present matched-shape score/stitch/commit and 2048-token boundary timings. The v17.2 48.79 h estimate remains invalid for the reasons recorded in the v17.3 cost audit; the 48 h gate itself is unchanged.

Cost components, shape counts, P95 selection, safety factors, and I/O throughput pilot are in `FINAL_COST_MODEL.json`. The I/O pilot used a durable flush and its temporary file was removed.

## Scope

No formal model quantization or PPL evaluation had started at precheck completion. The estimated decision is a cost gate only, not a quality result.
"""
    (OUT / "PRECHECK_REPORT.md").write_text(report, encoding="utf-8")

    # Keep machine-readable correctness checks short and tied to precheck evidence.
    checks = dict(pre["checks"])
    (OUT / "correctness_checks.json").write_text(json.dumps({
        "precheck_status": pre["status"], "checks": checks,
        "cost_gate": gate,
        "row_alpha_score_slots": row_alpha_scores,
        "full_linear_alpha_gptq_executions": full_execs,
        "dependency_refresh_transitions_per_method": 95,
        "conservative_m2_hours": m2_conservative_total / 3600,
        "gate_hours": GATE_HOURS,
    }, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    manifest_path = OUT / "FORMAL_RUN_MANIFEST.json"
    manifest = read_json(manifest_path)
    manifest["status"] = "FORMAL_M1_M2_READY" if gate == "M2_COST_GO" else "FORMAL_M2_COST_NO_GO"
    manifest["results"]["matched_shape_precheck"] = "PASS"
    manifest["results"]["cost_gate"] = gate
    manifest["cost_model"] = "FINAL_COST_MODEL.json"
    if gate == "M2_COST_GO":
        manifest["results"]["m1"] = "READY"
        manifest["results"]["m2"] = "READY"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": gate, "m2_central_hours": m2_central_total / 3600,
                      "m2_conservative_hours": m2_conservative_total / 3600,
                      "m1_conservative_hours": m1_conservative_total / 3600,
                      "combined_conservative_hours": (m1_conservative_total + m2_conservative_total) / 3600,
                      "row_alpha_score_slots": row_alpha_scores,
                      "full_linear_alpha_gptq_executions": full_execs}, indent=2))


if __name__ == "__main__":
    main()
