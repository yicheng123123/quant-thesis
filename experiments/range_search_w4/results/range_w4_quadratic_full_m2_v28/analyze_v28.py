from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open


ROOT = Path(__file__).resolve().parent
V18 = ROOT.parent / "range_w4_fullmodel_v18"
GRID = np.array([0.1 * i for i in range(1, 11)], dtype=np.float64)
MODULES = [
    "self_attn.k_proj", "self_attn.v_proj", "self_attn.q_proj",
    "self_attn.o_proj", "mlp.up_proj", "mlp.gate_proj", "mlp.down_proj",
]


def sha256_tensor(t: torch.Tensor) -> str:
    raw = t.detach().cpu().contiguous().numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def score_gap(a: np.ndarray) -> np.ndarray:
    s = np.sort(a, axis=0)
    return s[1] - s[0]


def load_layer(base: Path, layer: int) -> dict:
    path = base / "checkpoints" / "M2" / f"layer_{layer:02d}.pt"
    if not path.is_file():
        raise FileNotFoundError(path)
    return torch.load(path, map_location="cpu", weights_only=False)


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def named_tensor_hashes(model_file: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    with safe_open(str(model_file), framework="pt", device="cpu") as model:
        for name in sorted(model.keys()):
            tensor = model.get_tensor(name).contiguous()
            h = hashlib.sha256()
            h.update(name.encode("utf-8") + b"\0")
            h.update(str(tensor.dtype).encode("ascii") + b"\0")
            h.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode("ascii") + b"\0")
            h.update(tensor.numpy().tobytes())
            result[name] = h.hexdigest()
    return result


def main() -> None:
    run = ROOT / "validation"
    if not (run / "FORMAL_RUNTIME.json").exists():
        raise RuntimeError("validation run is not complete")
    output = run / "audit"
    raw_dir = output / "raw_scores"
    raw_dir.mkdir(parents=True, exist_ok=True)
    module_rows: list[dict] = []
    sensitive_rows: list[dict] = []
    selection_diffs: list[dict] = []
    direct_vs_v18_diffs: list[dict] = []
    boundary_rows: list[dict] = []
    c_sharing_rows: list[dict] = []

    global_scores = 0
    total_rows = 0
    module_count = 0
    exact_direct_score_arrays = 0
    exact_input_hashes = 0
    exact_hessian_hashes = 0
    exact_direct_alpha = 0
    exact_direct_tensors = 0
    exact_quad_tensors = 0
    total_sensitive = 0
    total_selection_diffs = 0
    bias_present_count = 0
    bias_absent_count = 0
    max_abs = 0.0
    max_rel = 0.0
    pooled_abs_errors: list[np.ndarray] = []
    pooled_rel_errors: list[np.ndarray] = []

    for layer in range(24):
        new = load_layer(run, layer)
        old = load_layer(V18, layer)
        nm = new["metadata"].get("boundary_activation_hashes", {})
        om = old["metadata"].get("boundary_activation_hashes", {})
        for boundary in sorted(set(nm) | set(om)):
            boundary_rows.append({
                "layer": layer, "boundary": boundary,
                "v18_sha256": om.get(boundary), "v28_direct_sha256": nm.get(boundary),
                "exact": nm.get(boundary) == om.get(boundary),
            })
        for group, names in {
            "qkv": ["self_attn.k_proj", "self_attn.v_proj", "self_attn.q_proj"],
            "o": ["self_attn.o_proj"],
            "up_gate": ["mlp.up_proj", "mlp.gate_proj"],
            "down": ["mlp.down_proj"],
        }.items():
            states = [new["modules"][name] for name in names]
            input_hashes = sorted({state["input_sha256"] for state in states})
            h_hashes = sorted({state["hessian_sha256"] for state in states})
            c_input_hashes = sorted({state["C_input_sha256"] for state in states})
            c_hashes = sorted({state["C_sha256"] for state in states})
            c_sharing_rows.append({
                "layer": layer, "dependency_group": group, "modules": ";".join(names),
                "module_count": len(names), "distinct_input_hashes": len(input_hashes),
                "distinct_H_hashes": len(h_hashes), "distinct_C_input_hashes": len(c_input_hashes),
                "distinct_C_hashes": len(c_hashes), "one_shared_C_for_exact_group":
                    len(input_hashes) == len(h_hashes) == len(c_input_hashes) == len(c_hashes) == 1
                    and input_hashes[0] == c_input_hashes[0],
                "input_sha256": input_hashes[0], "hessian_sha256": h_hashes[0],
                "C_input_sha256": c_input_hashes[0], "C_sha256": c_hashes[0],
            })
        layer_npz: dict[str, np.ndarray] = {}
        for name in MODULES:
            a = new["modules"][name]
            b = old["modules"][name]
            d = a["candidate_losses"].numpy().astype(np.float64, copy=False)
            q_tensor = a.get("candidate_losses_quadratic")
            if q_tensor is None:
                raise RuntimeError(f"missing quadratic scores L{layer} {name}")
            q = q_tensor.numpy().astype(np.float64, copy=False)
            if d.shape != q.shape or d.shape[0] != len(GRID):
                raise RuntimeError(f"candidate score shape mismatch L{layer} {name}: {d.shape}/{q.shape}")
            abs_err = np.abs(q - d)
            rel_err = abs_err / np.maximum(np.abs(d), 1e-30)
            dg, qg = score_gap(d), score_gap(q)
            direct_alpha = a["selected_alpha"].numpy()
            quad_alpha = a["quadratic_selected_alpha"].numpy()
            v18_alpha = b["selected_alpha"].numpy()
            diff = np.flatnonzero(direct_alpha != quad_alpha)
            sensitive = np.flatnonzero(2 * abs_err.max(axis=0) >= dg)
            direct_w = a["weight"].numpy()
            quad_w = a["quadratic_stitched_weight"].numpy()
            direct_s = a["scales"].numpy()
            quad_s = a["quadratic_stitched_scales"].numpy()
            direct_z = a["zeros"].numpy()
            quad_z = a["quadratic_stitched_zeros"].numpy()
            direct_g = a["g_idx"].numpy()
            old_direct_exact = all([
                np.array_equal(direct_alpha, v18_alpha),
                np.array_equal(direct_w, b["weight"].numpy()),
                np.array_equal(direct_s, b["scales"].numpy()),
                np.array_equal(direct_z, b["zeros"].numpy()),
                np.array_equal(direct_g, b["g_idx"].numpy()),
            ])
            quad_exact = all([
                np.array_equal(direct_alpha, quad_alpha),
                np.array_equal(direct_w, quad_w),
                np.array_equal(direct_s, quad_s),
                np.array_equal(direct_z, quad_z),
                np.array_equal(direct_g, np.arange(direct_g.shape[-1]) // 128),
            ])
            direct_scores_v18_exact = np.array_equal(d, b["candidate_losses"].numpy())
            input_equal = a["input_sha256"] == b["input_sha256"]
            hessian_equal = a["hessian_sha256"] == b["hessian_sha256"]
            bias_hash = a.get("unchanged_bias_sha256")
            bias_present_count += int(bias_hash is not None)
            bias_absent_count += int(bias_hash is None)
            max_abs = max(max_abs, float(abs_err.max(initial=0)))
            max_rel = max(max_rel, float(rel_err.max(initial=0)))
            pooled_abs_errors.append(abs_err.reshape(-1))
            pooled_rel_errors.append(rel_err.reshape(-1))
            global_scores += int(d.size)
            total_rows += int(d.shape[1])
            module_count += 1
            exact_direct_score_arrays += int(direct_scores_v18_exact)
            exact_input_hashes += int(input_equal)
            exact_hessian_hashes += int(hessian_equal)
            exact_direct_alpha += int(np.array_equal(direct_alpha, v18_alpha))
            exact_direct_tensors += int(old_direct_exact)
            exact_quad_tensors += int(quad_exact)
            total_sensitive += int(sensitive.size)
            total_selection_diffs += int(diff.size)

            layer_npz[name.replace(".", "_")] = d
            layer_npz[name.replace(".", "_") + "_quadratic"] = q
            layer_npz[name.replace(".", "_") + "_abs_error"] = abs_err
            layer_npz[name.replace(".", "_") + "_relative_error"] = rel_err
            layer_npz[name.replace(".", "_") + "_selected_alpha_direct"] = direct_alpha
            layer_npz[name.replace(".", "_") + "_selected_alpha_quadratic"] = quad_alpha
            layer_npz[name.replace(".", "_") + "_best_second_gap_direct"] = dg
            layer_npz[name.replace(".", "_") + "_best_second_gap_quadratic"] = qg
            layer_npz[name.replace(".", "_") + "_selected_score_direct"] = np.min(d, axis=0)
            layer_npz[name.replace(".", "_") + "_selected_score_quadratic"] = np.min(q, axis=0)
            for row in sensitive.tolist():
                sensitive_rows.append({
                    "layer": layer, "module": name, "row": row,
                    "direct_alpha": float(direct_alpha[row]),
                    "quadratic_alpha": float(quad_alpha[row]),
                    "direct_gap": float(dg[row]), "quadratic_gap": float(qg[row]),
                    "max_abs_score_error": float(abs_err[:, row].max()),
                    "max_rel_score_error": float(rel_err[:, row].max()),
                    "selection_diff": bool(direct_alpha[row] != quad_alpha[row]),
                })
            for row in diff.tolist():
                selection_diffs.append({
                    "layer": layer, "module": name, "row": row,
                    "direct_alpha": float(direct_alpha[row]),
                    "quadratic_alpha": float(quad_alpha[row]),
                    "direct_gap": float(dg[row]), "quadratic_gap": float(qg[row]),
                    "candidate_scores_direct": json.dumps(d[:, row].tolist()),
                    "candidate_scores_quadratic": json.dumps(q[:, row].tolist()),
                    "candidate_score_abs_error": json.dumps(abs_err[:, row].tolist()),
                    "direct_recheck": json.dumps(a["comparison"].get("direct_recheck", [])),
                })
            if not old_direct_exact:
                direct_vs_v18_diffs.append({"layer": layer, "module": name,
                                            "input_equal": input_equal,
                                            "hessian_equal": hessian_equal,
                                            "scores_equal": direct_scores_v18_exact,
                                            "alpha_equal": np.array_equal(direct_alpha, v18_alpha),
                                            "weight_equal": np.array_equal(direct_w, b["weight"].numpy()),
                                            "scale_equal": np.array_equal(direct_s, b["scales"].numpy()),
                                            "zero_equal": np.array_equal(direct_z, b["zeros"].numpy()),
                                            "g_idx_equal": np.array_equal(direct_g, b["g_idx"].numpy())})
            module_rows.append({
                "layer": layer, "module": name, "rows": int(d.shape[1]),
                "score_count": int(d.size), "max_abs_error": float(abs_err.max()),
                "median_abs_error": float(np.median(abs_err)),
                "max_relative_error": float(rel_err.max()),
                "median_relative_error": float(np.median(rel_err)),
                "direct_best_second_gap_min": float(dg.min()),
                "direct_best_second_gap_median": float(np.median(dg)),
                "quadratic_best_second_gap_min": float(qg.min()),
                "quadratic_best_second_gap_median": float(np.median(qg)),
                "direct_exact_tie_rows": int(a["comparison"].get("direct_tie_rows", 0)),
                "quadratic_exact_tie_rows": int(a["comparison"].get("quadratic_tie_rows", 0)),
                "direct_selected_alpha_histogram": json.dumps({f"{alpha:.1f}": int(np.count_nonzero(direct_alpha == alpha)) for alpha in GRID}, sort_keys=True),
                "quadratic_selected_alpha_histogram": json.dumps({f"{alpha:.1f}": int(np.count_nonzero(quad_alpha == alpha)) for alpha in GRID}, sort_keys=True),
                "sensitive_rows": int(sensitive.size), "selection_difference_rows": int(diff.size),
                "selection_differences": json.dumps(diff.tolist()),
                "input_hash_v18_equal": input_equal, "H_hash_v18_equal": hessian_equal,
                "bias_present": bias_hash is not None, "unchanged_bias_sha256": bias_hash,
                "direct_scores_v18_exact": direct_scores_v18_exact,
                "direct_selected_alpha_v18_exact": np.array_equal(direct_alpha, v18_alpha),
                "direct_selected_W_scale_zero_g_idx_v18_exact": old_direct_exact,
                "direct_vs_quadratic_selected_W_scale_zero_g_idx_exact": quad_exact,
                "C_input_sha256": a["C_input_sha256"], "C_sha256": a["C_sha256"],
                "direct_weight_sha256": sha256_tensor(a["weight"]),
                "quadratic_selected_weight_sha256": sha256_tensor(a["quadratic_stitched_weight"]),
            })
        np.savez_compressed(raw_dir / f"layer_{layer:02d}_scores.npz", **layer_npz)

    write_csv(output / "MODULE_CONSISTENCY.csv", module_rows)
    write_csv(output / "SENSITIVE_ROWS.csv", sensitive_rows)
    write_csv(output / "SELECTION_DIFFERENCES.csv", selection_diffs)
    write_csv(output / "DIRECT_V18_DIFFERENCES.csv", direct_vs_v18_diffs)
    write_csv(output / "BOUNDARY_ACTIVATION_COMPARISON.csv", boundary_rows)
    write_csv(output / "C_SHARING_AUDIT.csv", c_sharing_rows)

    # Preserve an expanded record for the single decision-changing row.
    difference_detail = {"selection_differences": []}
    for record in selection_diffs:
        layer, name, row = int(record["layer"]), record["module"], int(record["row"])
        state = load_layer(run, layer)["modules"][name]
        w0, w1 = state["weight"][row], state["quadratic_stitched_weight"][row]
        s0, s1 = state["scales"][row], state["quadratic_stitched_scales"][row]
        z0, z1 = state["zeros"][row], state["quadratic_stitched_zeros"][row]
        g = state["g_idx"]
        delta = (w0.float() - w1.float()).abs()
        direct = state["candidate_losses"][:, row].double().numpy()
        quad = state["candidate_losses_quadratic"][:, row].double().numpy()
        direct_order = np.argsort(direct, kind="stable")
        quad_order = np.argsort(quad, kind="stable")
        per_row = {
            "layer": layer, "module": name, "row": row,
            "alpha_grid": GRID.tolist(),
            "direct_scores": direct.tolist(), "quadratic_scores": quad.tolist(),
            "direct_order_best_to_worst": GRID[direct_order].tolist(),
            "quadratic_order_best_to_worst": GRID[quad_order].tolist(),
            "direct_best_second_gap": float(direct[direct_order[1]] - direct[direct_order[0]]),
            "quadratic_best_second_gap": float(quad[quad_order[1]] - quad[quad_order[0]]),
            "max_abs_score_error": float(np.abs(direct - quad).max()),
            "direct_recheck": state["comparison"]["direct_recheck"],
            "direct_recheck_interpretation": "the original direct scorer rescored the retained same-run alpha 0.8 and 0.9 candidate rows; subset GEMM changed the last few FP bits, with recorded max deviations below the full-row direct winner gap",
            "selected_weight_row_equal": bool(torch.equal(w0, w1)),
            "selected_weight_differing_elements": int(torch.count_nonzero(w0 != w1)),
            "selected_weight_max_abs_difference": float(delta.max()),
            "selected_weight_mean_abs_difference": float(delta.mean()),
            "selected_direct_weight_row_sha256": sha256_tensor(w0),
            "selected_quadratic_weight_row_sha256": sha256_tensor(w1),
            "selected_scale_row_equal": bool(torch.equal(s0, s1)),
            "selected_direct_scale_row_sha256": sha256_tensor(s0),
            "selected_quadratic_scale_row_sha256": sha256_tensor(s1),
            "selected_direct_scale_values": s0.double().tolist(),
            "selected_quadratic_scale_values": s1.double().tolist(),
            "selected_zero_row_equal": bool(torch.equal(z0, z1)),
            "selected_direct_zero_values": z0.double().tolist(),
            "selected_quadratic_zero_values": z1.double().tolist(),
            "g_idx_equal": bool(torch.equal(g, torch.arange(g.numel()) // 128)),
            "g_idx_group_boundary_sha256": sha256_tensor(g),
        }
        difference_detail["selection_differences"].append(per_row)
    (output / "NUMERICAL_SELECTION_DIFFERENCE.json").write_text(
        json.dumps(difference_detail, indent=2) + "\n", encoding="utf-8")

    # Compare the saved phase-1 direct-selected model against the frozen v18
    # M2 named tensors, independent of safetensors file-level metadata/order.
    old_model = V18 / "M2" / "checkpoint" / "final" / "model.safetensors"
    new_model = run / "M2" / "checkpoint" / "final" / "model.safetensors"
    old_named = named_tensor_hashes(old_model)
    new_named = named_tensor_hashes(new_model)
    tensor_names = sorted(set(old_named) | set(new_named))
    tensor_differences = [name for name in tensor_names if old_named.get(name) != new_named.get(name)]
    old_cfg = json.loads((old_model.parent / "config.json").read_text(encoding="utf-8"))
    new_cfg = json.loads((new_model.parent / "config.json").read_text(encoding="utf-8"))
    old_meta = json.loads((V18 / "M2" / "QUANTIZATION_METADATA.json").read_text(encoding="utf-8"))
    new_meta = json.loads((run / "M2" / "QUANTIZATION_METADATA.json").read_text(encoding="utf-8"))
    tensor_hash_result = {
        "hash_definition": "SHA256 over each sorted tensor name, dtype, JSON shape, and contiguous tensor bytes; model digest over sorted name + per-tensor digest pairs",
        "v18_tensor_count": len(old_named), "v28_direct_tensor_count": len(new_named),
        "same_tensor_names": set(old_named) == set(new_named),
        "per_tensor_exact_match_count": sum(old_named.get(n) == new_named.get(n) for n in tensor_names),
        "differing_tensor_names": tensor_differences,
        "v18_named_tensor_hash": hashlib.sha256("".join(f"{n}:{old_named[n]}\n" for n in sorted(old_named)).encode()).hexdigest(),
        "v28_direct_named_tensor_hash": hashlib.sha256("".join(f"{n}:{new_named[n]}\n" for n in sorted(new_named)).encode()).hexdigest(),
        "config_json_exact": old_cfg == new_cfg,
        "quantization_format_exact": old_meta.get("format") == new_meta.get("format"),
        "model_revision_exact": old_meta.get("model_revision") == new_meta.get("model_revision"),
        "calibration_manifest_exact": old_meta.get("calibration_manifest_sha256") == new_meta.get("calibration_manifest_sha256"),
        "v18_quantization_metadata": old_meta,
        "v28_quantization_metadata": new_meta,
    }
    (output / "NORMALIZED_TENSOR_HASHES.json").write_text(
        json.dumps(tensor_hash_result, indent=2) + "\n", encoding="utf-8")
    summary = {
        "status": "PASS" if total_selection_diffs == 0 and exact_quad_tensors == module_count
        and exact_direct_tensors == module_count and all(r["exact"] for r in boundary_rows) else "FAIL",
        "modules": module_count, "rows": total_rows, "row_alpha_scores": global_scores,
        "selection_difference_rows": total_selection_diffs, "sensitive_rows": total_sensitive,
        "modules_direct_selected_tensors_exact_vs_v18": exact_direct_tensors,
        "modules_quadratic_selected_tensors_exact_vs_direct": exact_quad_tensors,
        "modules_direct_scores_exact_vs_v18": exact_direct_score_arrays,
        "modules_input_hash_exact_vs_v18": exact_input_hashes,
        "modules_hessian_hash_exact_vs_v18": exact_hessian_hashes,
        "modules_direct_alpha_exact_vs_v18": exact_direct_alpha,
        "boundary_hashes_exact": sum(bool(r["exact"]) for r in boundary_rows),
        "boundary_hashes_total": len(boundary_rows),
        "modules_with_bias": bias_present_count, "modules_without_bias": bias_absent_count,
        "max_abs_score_error": max_abs, "max_relative_score_error": max_rel,
        "median_abs_score_error_all_row_alpha_pairs": float(np.median(np.concatenate(pooled_abs_errors))),
        "median_relative_score_error_all_row_alpha_pairs": float(np.median(np.concatenate(pooled_rel_errors))),
        "normalized_final_model_named_tensors_exact_vs_v18": not tensor_differences and tensor_hash_result["same_tensor_names"],
        "inference_config_and_quant_format_exact_vs_v18": tensor_hash_result["config_json_exact"]
            and tensor_hash_result["quantization_format_exact"]
            and tensor_hash_result["model_revision_exact"],
        "C_dependency_groups": len(c_sharing_rows),
        "C_groups_one_correctly_shared_matrix": sum(bool(r["one_shared_C_for_exact_group"]) for r in c_sharing_rows),
        "alpha_grid_tie_rule": "strict minimum; exact score ties resolve to larger alpha, matching existing M2 streaming update",
        "raw_score_artifacts": "audit/raw_scores/layer_XX_scores.npz",
        "selection_rule": "existing exact argmin/tie rule; alpha in frozen ascending grid; no tolerance",
        "sensitivity_rule": "2 * max_abs_score_error(row) >= direct best-second gap(row)",
    }
    (output / "CONSISTENCY_SUMMARY.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
