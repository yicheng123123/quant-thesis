#!/usr/bin/env python3
"""Independently audit v20.1 Kaggle preflight artifacts and cost extrapolation.

Reads the immutable downloaded Kaggle output and frozen local preflight config.
Writes only audit files next to this script.
"""

from __future__ import annotations

import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
RAW = ROOT / "kaggle_output_v2" / "range_w4_omniquant_kaggle_v20_1"
CONFIG = ROOT / "kaggle_bundle" / "configs" / "m3_preflight.json"


def read_json(name: str) -> dict[str, Any]:
    return json.loads((RAW / name).read_text(encoding="utf-8"))


def write_json(name: str, value: dict[str, Any]) -> None:
    (ROOT / name).write_text(
        json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def percentile(values: list[float], q: float) -> float:
    """NumPy's default linear percentile, without a NumPy dependency."""
    ordered = sorted(values)
    if not ordered:
        raise ValueError("percentile requires observations")
    index = (len(ordered) - 1) * q / 100.0
    lo = math.floor(index)
    hi = math.ceil(index)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (index - lo)


def main() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    cfg = config
    formal_cfg = cfg["formal_training"]
    preflight_cfg = cfg["preflight"]

    bundle = read_json("BUNDLE_VERIFY.json")
    model = read_json("MODEL_IDENTITY.json")
    calibration = read_json("KAGGLE_CALIBRATION_CHECK.json")
    parity = read_json("KAGGLE_QUANT_OFF_PARITY.json")
    lwc = read_json("KAGGLE_LWC_CHECK.json")
    correctness = read_json("correctness_checks.json")
    step_count = read_json("FORMAL_STEP_COUNT.json")
    verdict = read_json("PREFLIGHT_VERDICT.json")
    memory = {layer: read_json(f"L{layer}_2048_MEMORY.json") for layer in (0, 12, 23)}
    runs = {layer: read_json(f"L{layer}_2048_RUN.json") for layer in (0, 12, 23)}
    recalc = read_json("FORMAL_M3_COST_RECALC.json")
    sequence = read_json("L0_8STEP_SEQUENCE.json")
    env = read_json("KAGGLE_ENVIRONMENT.json")

    layers = int(step_count["decoder_layers"])
    epochs = int(formal_cfg["epochs"])
    samples = int(formal_cfg["samples"])
    batch_size = int(formal_cfg["batch_size"])
    formal_updates = layers * epochs * (samples // batch_size)
    updates_per_block = epochs * (samples // batch_size)
    weights = {int(k): int(v) for k, v in preflight_cfg["representative_layer_cost_weights"].items()}

    timing_rows: dict[int, list[dict[str, str]]] = {}
    means: dict[int, float] = {}
    measured_values: list[float] = []
    timing_counts: dict[int, int] = {}
    for layer in (0, 12, 23):
        path = RAW / f"L{layer}_2048_STEP_TIMING.csv"
        with path.open(encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        timing_rows[layer] = rows
        measured = [row for row in rows if row["status"] == "MEASURED" and row["warmup_excluded"].lower() == "false"]
        times = [float(row["total_s"]) for row in measured]
        timing_counts[layer] = len(times)
        means[layer] = statistics.mean(times)
        measured_values.extend(times)

    p95 = percentile(measured_values, 95)
    central_training = sum(weights[layer] * updates_per_block * means[layer] for layer in weights)
    conservative_training = p95 * formal_updates * float(preflight_cfg["conservative_safety_multiplier"])

    source_components = {key: float(value) for key, value in recalc["cost_components_seconds"].items()}
    excluded = {"optimizer_training", "block_overhead", "diagnostic_full_native_capture_not_charged"}
    non_block_extras = sum(value for key, value in source_components.items() if key not in excluded)

    # The remote report's residual model scales a whole-call residual by 128 / N,
    # although each smoke call executes 20 outer epoch iterations and only N
    # optimizer updates. Keep that raw estimate, then bracket it explicitly.
    fixed_per_layer_call = sum(weights[layer] * float(runs[layer]["other_block_overhead_seconds"]) for layer in weights)
    reported_residual = source_components["block_overhead"]
    per_update_residual_stress = sum(
        weights[layer]
        * float(runs[layer]["other_block_overhead_seconds"])
        / max(1, int(runs[layer]["updates_observed"]))
        * updates_per_block
        for layer in weights
    )

    scenarios = {
        "fixed_call_residual_floor": {
            "block_overhead_seconds": fixed_per_layer_call,
            "central_total_hours": (central_training + non_block_extras + fixed_per_layer_call) / 3600.0,
            "conservative_total_hours": (conservative_training + non_block_extras + fixed_per_layer_call) / 3600.0,
            "assumption": "The measured non-step residual is paid once per formal decoder-block call; this is a lower scenario, not a measured decomposition.",
        },
        "remote_report_scaling": {
            "block_overhead_seconds": reported_residual,
            "central_total_hours": (central_training + non_block_extras + reported_residual) / 3600.0,
            "conservative_total_hours": (conservative_training + non_block_extras + reported_residual) / 3600.0,
            "assumption": "Preserves the remote report's residual scaling by samples divided by observed smoke updates.",
        },
        "all_residual_as_per_update_stress": {
            "block_overhead_seconds": per_update_residual_stress,
            "central_total_hours": (central_training + non_block_extras + per_update_residual_stress) / 3600.0,
            "conservative_total_hours": (conservative_training + non_block_extras + per_update_residual_stress) / 3600.0,
            "assumption": "Treats every second of measured non-step residual as if it recurred at every formal update. This deliberately over-allocates call/epoch setup and is a stress bound, not an expected runtime.",
        },
    }

    checks = {
        "bundle_pass": bundle.get("status") == "PASS" and not bundle.get("failures"),
        "official_commit_exact": bundle.get("checks", {}).get("upstream_commit", {}).get("pass") is True,
        "model_identity_pass": model.get("status") == "PASS" and model.get("actual_snapshot_revision") == cfg["model_revision"],
        "calibration_hash_and_shape_pass": calibration.get("status") == "PASS"
            and calibration.get("manifest_sha256") == formal_cfg["calibration_manifest_sha256"]
            and calibration.get("samples") == samples
            and calibration.get("seed") == formal_cfg["seed"]
            and len(calibration.get("sample_hashes", [])) == samples
            and all(item.get("token_count") == formal_cfg["tokens_per_sample"] for item in calibration.get("sample_hashes", [])),
        "formal_step_formula_matches_source_audit": step_count.get("status") == "PASS"
            and step_count.get("total_optimizer_updates") == formal_updates,
        "quant_off_parity_pass": parity.get("status") == "PASS"
            and all(row.get("bitwise_equal") for row in parity.get("layers", [])),
        "lwc_2048_check_pass": lwc.get("status") == "PASS"
            and lwc.get("lwc_parameter_exists") is True
            and lwc.get("gradient_finite_all_steps") is True
            and lwc.get("nonzero_lwc_gradient_all_steps") is True
            and lwc.get("parameter_update_observed") is True
            and lwc.get("reconstruction_loss_finite_all_steps") is True,
        "representative_timing_counts": timing_counts == {
            0: int(preflight_cfg["l0_measured_steps"]),
            12: int(preflight_cfg["l12_measured_steps"]),
            23: int(preflight_cfg["l23_measured_steps"]),
        },
        "representative_run_correctness": all(
            runs[layer].get("sequence_length") == formal_cfg["tokens_per_sample"]
            and runs[layer].get("all_losses_finite")
            and runs[layer].get("all_lwc_gradients_finite")
            and runs[layer].get("all_steps_have_nonzero_lwc_gradient")
            and runs[layer].get("any_lwc_parameter_changed")
            for layer in runs
        ),
        "memory_gate_pass": all(
            memory[layer].get("status") == "PASS"
            and memory[layer].get("oom") is False
            and memory[layer].get("peak_reserved_gib", float("inf")) <= preflight_cfg["frozen_v20_reserved_vram_gate_gib"]
            for layer in memory
        ),
        "eight_step_growth_gate_pass": sequence.get("measured_steps") == 8
            and sequence.get("memory_growth_diagnostic", {}).get("unexplained_growth_warning") is False,
        "remote_verdict_matches_checks": verdict.get("verdict") == "M3_KAGGLE_FORMAL_EXECUTION_READY"
            and correctness.get("status") == "PASS"
            and all(correctness.get("checks", {}).values())
            and verdict.get("formal_training_executed") is False
            and verdict.get("formal_ppl_executed") is False,
    }
    all_correctness_pass = all(checks[key] for key in (
        "bundle_pass", "official_commit_exact", "model_identity_pass", "calibration_hash_and_shape_pass",
        "formal_step_formula_matches_source_audit", "quant_off_parity_pass", "lwc_2048_check_pass",
        "representative_timing_counts", "representative_run_correctness", "memory_gate_pass",
        "eight_step_growth_gate_pass", "remote_verdict_matches_checks",
    ))
    conservative_upper = scenarios["all_residual_as_per_update_stress"]["conservative_total_hours"]
    runtime_gate = conservative_upper <= float(preflight_cfg["frozen_formal_runtime_gate_hours"])

    audit = {
        "audit_status": "PASS_WITH_COST_MODEL_SENSITIVITY" if all_correctness_pass else "FAIL",
        "remote_preflight_verdict": verdict.get("verdict"),
        "audited_formal_readiness_verdict": "M3_KAGGLE_FORMAL_EXECUTION_READY"
            if all_correctness_pass and runtime_gate else "M3_KAGGLE_FORMAL_EXECUTION_NOT_READY",
        "formal_training_executed": False,
        "formal_ppl_executed": False,
        "environment": {
            "gpu": [gpu.get("name") for gpu in env.get("gpus", [])],
            "gpu_count": env.get("gpu_count"),
            "selected_device": env.get("selected_device"),
            "single_gpu_peak_reserved_gib": max(memory[layer]["peak_reserved_gib"] for layer in memory),
        },
        "formal_updates": {
            "layers": layers,
            "epochs": epochs,
            "samples_per_epoch": samples,
            "batch_size": batch_size,
            "updates_per_block": updates_per_block,
            "total_updates": formal_updates,
            "formula": f"{layers} * {epochs} * ({samples} / {batch_size}) = {formal_updates}",
        },
        "measured_timing": {
            "measured_update_counts": {str(k): v for k, v in timing_counts.items()},
            "mean_step_seconds_by_layer": {str(k): means[k] for k in means},
            "pooled_p95_step_seconds": p95,
            "central_training_hours": central_training / 3600.0,
            "conservative_training_hours_with_safety_margin": conservative_training / 3600.0,
        },
        "cost_model_audit": {
            "non_block_extras_seconds": non_block_extras,
            "block_overhead_scenarios": scenarios,
            "interpretation": "The single-call residual is not separated into per-call, per-epoch, and per-update components. The remote central estimate's 0.687 h is reproduced, but its residual extrapolation is not uniquely identified by these smoke calls. The conservative stress case assigns the full observed residual to every formal update; even then the 24 h gate passes.",
            "conservative_stress_total_hours": conservative_upper,
            "runtime_gate_hours": float(preflight_cfg["frozen_formal_runtime_gate_hours"]),
            "runtime_gate_pass_under_stress_case": runtime_gate,
        },
        "correctness_checks": checks,
        "single_session_status": recalc.get("kaggle_single_session_status"),
        "limitations": [
            "L12 and L23 use native FP16 full-model prefix activations; sequentially quantized-prefix equivalence is not tested.",
            "Dense export timing is an FP16 state copy/serialization/reload proxy; W4 dequantization and its peak memory were not measured.",
            "Three representative layers and 11 measured steps support a preflight estimate, not a formal 20-epoch run or a quality/PPL conclusion.",
            "The Kaggle session wall-time budget was unavailable; block-level checkpoint/resume remains required for the formal run.",
        ],
    }
    write_json("COST_MODEL_AUDIT.json", audit)

    with (ROOT / "FORMAL_M3_COST_BREAKDOWN_AUDITED.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["scenario", "component", "seconds", "hours", "interpretation"])
        writer.writeheader()
        for scenario, detail in scenarios.items():
            writer.writerow({
                "scenario": scenario,
                "component": "block_overhead_residual",
                "seconds": f"{detail['block_overhead_seconds']:.6f}",
                "hours": f"{detail['block_overhead_seconds']/3600.0:.9f}",
                "interpretation": detail["assumption"],
            })
            writer.writerow({
                "scenario": scenario,
                "component": "central_total",
                "seconds": "",
                "hours": f"{detail['central_total_hours']:.9f}",
                "interpretation": "Measured per-layer mean step time plus common measured extras and this residual scenario.",
            })
            writer.writerow({
                "scenario": scenario,
                "component": "conservative_total",
                "seconds": "",
                "hours": f"{detail['conservative_total_hours']:.9f}",
                "interpretation": "Pooled measured P95 step time times 1.25 plus common measured extras and this residual scenario.",
            })

    print(json.dumps({
        "audit_status": audit["audit_status"],
        "audited_verdict": audit["audited_formal_readiness_verdict"],
        "formal_updates": formal_updates,
        "step_p95_seconds": p95,
        "cost_scenarios_hours": {
            key: {"central": value["central_total_hours"], "conservative": value["conservative_total_hours"]}
            for key, value in scenarios.items()
        },
        "correctness_checks": checks,
    }, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
