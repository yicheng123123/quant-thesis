from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open


OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[3]
V18 = ROOT / "experiments/range_search_w4/results/range_w4_fullmodel_v18"
MANIFEST = json.loads((OUT / "CANDIDATE_MANIFEST.json").read_text(encoding="utf-8"))
FULL_GRID = [float(x) for x in MANIFEST["full_grid"]]


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def f64(t: torch.Tensor) -> np.ndarray:
    return t.detach().cpu().numpy().astype(np.float64, copy=False)


def exact_tensor(left: torch.Tensor, right: torch.Tensor) -> bool:
    return left.dtype == right.dtype and left.shape == right.shape and torch.equal(left, right)


def max_abs(left: torch.Tensor, right: torch.Tensor) -> float:
    if left.shape != right.shape:
        return float("inf")
    a, b = f64(left), f64(right)
    if a.size == 0:
        return 0.0
    return float(np.max(np.abs(a - b)))


def named_tensor_hashes(path: Path) -> tuple[dict[str, str], str]:
    result: dict[str, str] = {}
    with safe_open(str(path), framework="pt", device="cpu") as model:
        for name in sorted(model.keys()):
            tensor = model.get_tensor(name).contiguous()
            h = hashlib.sha256()
            h.update(name.encode("utf-8") + b"\0")
            h.update(str(tensor.dtype).encode("ascii") + b"\0")
            h.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode("ascii") + b"\0")
            h.update(tensor.numpy().tobytes())
            result[name] = h.hexdigest()
    aggregate = hashlib.sha256()
    for name in sorted(result):
        aggregate.update(f"{name}:{result[name]}\n".encode("ascii"))
    return result, aggregate.hexdigest()


def compare_configs(base_quant_dir: Path, target_quant_dir: Path) -> dict:
    final_names = ["config.json", "generation_config.json"]
    config_checks = {}
    for name in final_names:
        a, b = base_quant_dir / "checkpoint/final" / name, target_quant_dir / "M2/checkpoint/final" / name
        if not a.exists() or not b.exists():
            config_checks[name] = {"equal": False, "status": "MISSING", "base_exists": a.exists(), "target_exists": b.exists()}
            continue
        config_checks[name] = {"equal": read_json(a) == read_json(b), "base_sha256": hashlib.sha256(a.read_bytes()).hexdigest(),
                               "target_sha256": hashlib.sha256(b.read_bytes()).hexdigest()}
    base_meta_path = base_quant_dir / "QUANTIZATION_METADATA.json"
    target_meta_path = target_quant_dir / "M2/QUANTIZATION_METADATA.json"
    base_meta, target_meta = read_json(base_meta_path), read_json(target_meta_path)
    semantic_keys = ["representation", "model_revision", "format", "calibration_manifest_sha256", "protocol_sha256"]
    semantic = {key: {"equal": base_meta.get(key) == target_meta.get(key), "base": base_meta.get(key), "target": target_meta.get(key)}
                for key in semantic_keys}
    semantic_equal = all(v["equal"] for v in semantic.values())
    return {"config_files": config_checks, "quantization_metadata_semantics": semantic,
            "equal": all(v.get("equal", False) for v in config_checks.values()) and semantic_equal}


def compare_run(arm: str, *, pilot: bool = False) -> dict:
    run_name = "PILOT" if pilot else arm
    run_dir = OUT / run_name
    expected_layers = 1 if pilot else 24
    module_rows = []
    boundary_rows = []
    all_equal = True
    layers_compared = 0
    for layer in range(expected_layers):
        run_path = run_dir / f"checkpoints/M2/layer_{layer:02d}.pt"
        ref_path = V18 / f"checkpoints/M2/layer_{layer:02d}.pt"
        if not run_path.exists() or not ref_path.exists():
            raise FileNotFoundError(f"missing v40 or v18 layer checkpoint L{layer}: {run_path} / {ref_path}")
        current = torch.load(run_path, map_location="cpu", weights_only=False)
        reference = torch.load(ref_path, map_location="cpu", weights_only=False)
        if set(current["modules"]) != set(reference["modules"]):
            raise RuntimeError(f"module inventory mismatch in L{layer}")
        layer_module_equal = True
        for submodule in sorted(reference["modules"]):
            full_name = f"model.layers.{layer}.{submodule}"
            cur = current["modules"][submodule]
            ref = reference["modules"][submodule]
            allowed = FULL_GRID if arm == "A" else [float(x) for x in MANIFEST["modules"][full_name]["candidate_alphas"]]
            alpha_equal = exact_tensor(cur["selected_alpha"], ref["selected_alpha"])
            chosen_loss_equal = exact_tensor(cur["best_loss"], ref["best_loss"])
            alpha1_loss_equal = exact_tensor(cur["alpha1_loss"], ref["alpha1_loss"])
            tensors = {key: exact_tensor(cur[key], ref[key]) for key in ("weight", "scales", "zeros", "g_idx")}
            hashes = {key: cur.get(key) == ref.get(key) for key in ("input_sha256", "hessian_sha256")}
            if not all(float(a) in allowed for a in ref["selected_alpha"].tolist()):
                raise RuntimeError(f"winner-union omitted a historical winning alpha: {full_name}")
            cur_scores = cur["candidate_losses"].detach().cpu().numpy()
            ref_scores = ref["candidate_losses"].detach().cpu().numpy()
            if cur_scores.shape != (len(allowed), ref_scores.shape[1]):
                raise RuntimeError(f"candidate score shape mismatch: {full_name} {cur_scores.shape} vs {len(allowed)}")
            score_equal = True
            score_max_abs = 0.0
            score_slots = 0
            for i, alpha in enumerate(allowed):
                ref_index = FULL_GRID.index(float(alpha))
                candidate_ref = ref_scores[ref_index]
                candidate_cur = cur_scores[i]
                if candidate_cur.shape != candidate_ref.shape:
                    score_equal = False
                    score_max_abs = float("inf")
                else:
                    difference = np.abs(candidate_cur.astype(np.float64) - candidate_ref.astype(np.float64))
                    score_max_abs = max(score_max_abs, float(difference.max(initial=0.0)))
                    score_equal = score_equal and bool(np.array_equal(candidate_cur, candidate_ref))
                    score_slots += candidate_cur.size
            selected_loss_max_abs = max_abs(cur["best_loss"], ref["best_loss"])
            scalar_checks = {"selected_alpha": alpha_equal, "selected_loss": chosen_loss_equal,
                             "alpha1_loss": alpha1_loss_equal, "all_candidate_scores": score_equal}
            module_equal = all(tensors.values()) and all(hashes.values()) and all(scalar_checks.values())
            layer_module_equal = layer_module_equal and module_equal
            all_equal = all_equal and module_equal
            module_rows.append({
                "arm": arm, "layer": layer, "module": full_name, "shape_out_in": "x".join(map(str, cur["weight"].shape)),
                "rows": int(cur["weight"].shape[0]), "candidate_alphas": ";".join(f"{a:.1f}" for a in allowed),
                "candidate_count": len(allowed), "candidate_score_slots_compared": score_slots,
                "candidate_score_exact": score_equal, "candidate_score_max_abs_error": score_max_abs,
                "selected_alpha_exact": alpha_equal, "selected_loss_exact": chosen_loss_equal,
                "selected_loss_max_abs_error": selected_loss_max_abs, "alpha1_loss_exact": alpha1_loss_equal,
                "weight_exact": tensors["weight"], "weight_max_abs_error": max_abs(cur["weight"], ref["weight"]),
                "scale_exact": tensors["scales"], "zero_exact": tensors["zeros"], "g_idx_exact": tensors["g_idx"],
                "input_hash_exact": hashes["input_sha256"], "hessian_hash_exact": hashes["hessian_sha256"],
                "module_pass": module_equal,
            })
        cur_boundaries = current["metadata"].get("boundary_activation_hashes", {})
        ref_boundaries = reference["metadata"].get("boundary_activation_hashes", {})
        for key in sorted(set(cur_boundaries) | set(ref_boundaries)):
            same = cur_boundaries.get(key) == ref_boundaries.get(key)
            boundary_rows.append({"arm": arm, "layer": layer, "boundary": key,
                                  "v40_sha256": cur_boundaries.get(key), "v18_sha256": ref_boundaries.get(key),
                                  "equal": same})
            all_equal = all_equal and same
        state_hash_equal = current["metadata"].get("layer_quant_state_sha256") == reference["metadata"].get("layer_quant_state_sha256")
        if not state_hash_equal:
            all_equal = False
        boundary_rows.append({"arm": arm, "layer": layer, "boundary": "layer_quant_state_sha256",
                              "v40_sha256": current["metadata"].get("layer_quant_state_sha256"),
                              "v18_sha256": reference["metadata"].get("layer_quant_state_sha256"),
                              "equal": state_hash_equal})
        layers_compared += 1
        del current, reference
    if pilot:
        runtime = read_json(run_dir / "FORMAL_RUNTIME.json")
        counts = runtime["m2"]["formal_counts"]
        expected_calls, expected_slots, expected_parity = 27, 49792, 7
        result_path = OUT / "PILOT_PARITY.json"
    else:
        report = read_json(run_dir / "M2/RUN_REPORT.json")
        counts = report["formal_counts"]
        if arm == "A":
            expected_calls, expected_slots, expected_parity = 1680, 3041280, 35
        else:
            expected_calls, expected_slots, expected_parity = 675, 1287040, 35
        result_path = OUT / f"{arm}_RUN_VALIDITY.json"
    call_count_ok = counts["full_linear_alpha_candidate_executions"] == expected_calls
    score_slot_count_ok = counts["row_alpha_score_slots"] == expected_slots
    parity_count_ok = counts["alpha1_native_parity_reference_executions"] == expected_parity
    status = "PASS" if all_equal and call_count_ok and score_slot_count_ok and parity_count_ok else "FAIL"
    result = {"status": status, "arm": arm, "reference": "v18 M2 full-grid direct scorer",
              "pilot_only": pilot, "modules_compared": len(module_rows), "layers_compared": layers_compared,
              "boundary_hashes_compared": sum(r["boundary"] != "layer_quant_state_sha256" for r in boundary_rows),
              "layer_state_hashes_compared": layers_compared,
              "counts": counts, "expected_counts": {"candidate_gptq_executions": expected_calls,
                  "row_alpha_score_slots": expected_slots, "alpha1_native_parity_gptq_executions": expected_parity},
              "candidate_count_pass": call_count_ok, "score_slot_count_pass": score_slot_count_ok,
              "native_parity_call_count_pass": parity_count_ok,
              "selected_alpha_scores_W_scale_zero_g_idx_input_H_and_boundary_hashes_exact": all_equal,
              "p3_read": False, "held_run": False, "ppl_run": False}
    write_json(result_path, result)
    if not pilot:
        existing = OUT / "MODULE_PARITY.csv"
        previous = list(csv.DictReader(existing.open(encoding="utf-8-sig", newline=""))) if existing.exists() else []
        previous = [r for r in previous if r["arm"] != arm]
        previous.extend(module_rows)
        with existing.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(module_rows[0]))
            writer.writeheader(); writer.writerows(previous)
        boundary_path = OUT / "BOUNDARY_PARITY.csv"
        prior_boundaries = list(csv.DictReader(boundary_path.open(encoding="utf-8-sig", newline=""))) if boundary_path.exists() else []
        prior_boundaries = [r for r in prior_boundaries if r["arm"] != arm]
        prior_boundaries.extend(boundary_rows)
        with boundary_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(boundary_rows[0]))
            writer.writeheader(); writer.writerows(prior_boundaries)
    return result


def final_hashes() -> dict:
    candidates = {"v18_M2": V18 / "M2/checkpoint/final", "A": OUT / "A/M2/checkpoint/final",
                  "B": OUT / "B/M2/checkpoint/final"}
    summary = {}
    tensor_rows = []
    hash_maps = {}
    config_results = {}
    for label, directory in candidates.items():
        model_path = directory / "model.safetensors"
        if not model_path.exists():
            summary[label] = {"status": "NOT_RUN", "path": str(directory)}
            continue
        hashes, aggregate = named_tensor_hashes(model_path)
        hash_maps[label] = hashes
        quant_dir = V18 / "M2" if label == "v18_M2" else OUT / f"{label}/M2"
        summary[label] = {"status": "COMPLETE", "model_file_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
                          "named_tensor_count": len(hashes), "normalized_named_tensor_sha256": aggregate,
                          "quantization_metadata_sha256": hashlib.sha256((quant_dir / "QUANTIZATION_METADATA.json").read_bytes()).hexdigest()}
    base_dir = V18 / "M2"
    for label in ("A", "B"):
        if label in hash_maps:
            config = compare_configs(base_dir, OUT / label)
            config_results[label] = config
            summary[label]["inference_configuration_equal_to_v18"] = config["equal"]
            summary[label]["named_tensors_equal_to_v18"] = hash_maps[label] == hash_maps.get("v18_M2", {})
    labels = sorted(hash_maps)
    if "v18_M2" in hash_maps:
        names = sorted(hash_maps["v18_M2"])
        for name in names:
            row = {"tensor_name": name, "v18_sha256": hash_maps["v18_M2"][name]}
            for label in ("A", "B"):
                row[f"{label}_sha256"] = hash_maps.get(label, {}).get(name, "")
                row[f"{label}_equal"] = label in hash_maps and hash_maps[label].get(name) == row["v18_sha256"]
            tensor_rows.append(row)
    overall_equal = all(summary.get(label, {}).get("named_tensors_equal_to_v18") is True and
                        summary.get(label, {}).get("inference_configuration_equal_to_v18") is True for label in ("A", "B"))
    payload = {"status": "PASS" if overall_equal else "INCOMPLETE_OR_DIFFERENT",
               "model_inference_quality_reusable_from_v18": overall_equal,
               "tensor_hash_method": "v28 named_tensor_hashes: name, dtype, JSON shape and contiguous tensor bytes; sorted-name aggregate",
               "runs": summary, "configuration_comparisons": config_results,
               "ppl_run": False, "p3_read": False, "held_run": False}
    write_json(OUT / "FINAL_TENSOR_HASHES.json", payload)
    if tensor_rows:
        with (OUT / "FINAL_TENSOR_HASHES.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(tensor_rows[0]))
            writer.writeheader(); writer.writerows(tensor_rows)
    return payload


def aggregate_stage_timing() -> None:
    output = []
    for phase in ("PILOT", "A", "B"):
        path = OUT / phase / "FORMAL_TIMING.csv"
        if not path.exists():
            continue
        rows = list(csv.DictReader(path.open(encoding="utf-8-sig", newline="")))
        by = {}
        for row in rows:
            stage = row["stage"]
            x = by.setdefault(stage, {"seconds": 0.0, "count": 0, "peak_allocated": 0, "peak_reserved": 0})
            x["seconds"] += float(row["wall_seconds"])
            x["count"] += 1
            x["peak_allocated"] = max(x["peak_allocated"], int(row["gpu_peak_allocated_bytes"]))
            x["peak_reserved"] = max(x["peak_reserved"], int(row["gpu_peak_reserved_bytes"]))
        for stage, values in sorted(by.items()):
            output.append({"arm": phase, "stage": stage, "count": values["count"],
                           "stage_wall_seconds_sum_nonadditive": values["seconds"],
                           "gpu_peak_allocated_bytes": values["peak_allocated"],
                           "gpu_peak_reserved_bytes": values["peak_reserved"]})
    with (OUT / "STAGE_TIMING.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["arm", "stage", "count", "stage_wall_seconds_sum_nonadditive",
                                               "gpu_peak_allocated_bytes", "gpu_peak_reserved_bytes"])
        writer.writeheader(); writer.writerows(output)


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["pilot", "A", "B", "finalize"])
    args = parser.parse_args()
    if args.mode == "pilot":
        result = compare_run("B", pilot=True)
    elif args.mode in ("A", "B"):
        result = compare_run(args.mode)
        if args.mode == "A":
            runtime = read_json(OUT / "A/M2/RUN_REPORT.json")
            result["full_run_status"] = runtime["status"]
            result["final_reload_correctness"] = runtime["reload_correctness"]
            result["status"] = "PASS" if result["status"] == "PASS" and runtime["status"] == "COMPLETE" and runtime["reload_correctness"]["pass"] else "FAIL"
            write_json(OUT / "A_RUN_VALIDITY.json", result)
        else:
            runtime = read_json(OUT / "B/M2/RUN_REPORT.json")
            result["full_run_status"] = runtime["status"]
            result["final_reload_correctness"] = runtime["reload_correctness"]
            result["status"] = "PASS" if result["status"] == "PASS" and runtime["status"] == "COMPLETE" and runtime["reload_correctness"]["pass"] else "FAIL"
            write_json(OUT / "B_RUN_VALIDITY.json", result)
    else:
        result = {"status": "FINALIZING"}
    tensor_result = final_hashes()
    aggregate_stage_timing()
    budget = read_json(OUT / "BUDGET.json")
    summary = {"status": result.get("status"), "run_result": result, "final_tensor_hashes": tensor_result,
               "budget_elapsed_seconds": budget.get("gpu_experiment_elapsed_seconds"),
               "budget_status": budget.get("status"), "stages": str((OUT / "STAGE_TIMING.csv").relative_to(OUT))}
    write_json(OUT / f"ANALYSIS_{args.mode.upper()}.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
