"""Complete-matrix v29 recheck of the known v28 selection-difference Linear."""
from __future__ import annotations

import csv
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
V28 = ROOT / "experiments/range_search_w4/results/range_w4_quadratic_full_m2_v28"
V28_RUN = V28 / "validation"
V29_RUNNER_PATH = HERE / "run_v29.py"
sys.path.insert(0, str(HERE))
os.environ["V29_RUN_LABEL"] = "spot"
os.environ["V29_SCORE_MODE"] = "dual"
import run_v29 as rv  # noqa: E402


def load_v28_runner():
    spec = importlib.util.spec_from_file_location("v28_readonly_runner", V28 / "run_v28.py")
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def assert_checkpoint_group(path: Path, layer: int, group: str, expected_names: list[str]):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    meta = payload["metadata"]
    if meta.get("layer") != layer or meta.get("group") != group:
        raise RuntimeError(f"wrong v28 checkpoint at {path}")
    if set(payload["modules"]) != set(expected_names):
        raise RuntimeError(f"v28 checkpoint module inventory mismatch at {path}")
    for name, state in payload["modules"].items():
        if rv.module_state_hash(state) != state.get("state_sha256"):
            raise RuntimeError(f"v28 checkpoint state hash mismatch {path.name} {name}")
    return payload


def pick(scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    best = np.full(scores.shape[1], np.inf, dtype=np.float64)
    alpha = np.full(scores.shape[1], -1.0, dtype=np.float64)
    grid = np.asarray(rv.GRID, dtype=np.float64)
    for i, a in enumerate(grid):
        better = (scores[i] < best) | ((scores[i] == best) & (a > alpha))
        best[better] = scores[i, better]
        alpha[better] = a
    return alpha, best


def stitch(alpha: np.ndarray, qs, scales, zeros, g_idx):
    weight = torch.empty_like(qs[0])
    scale = torch.empty_like(scales[0])
    zero = torch.empty_like(zeros[0])
    for i, a in enumerate(rv.GRID):
        rows = torch.from_numpy(alpha == a).to(device=weight.device)
        if torch.any(rows):
            weight[rows] = qs[i][rows]
            scale[rows] = scales[i][rows]
            zero[rows] = zeros[i][rows]
    return weight, scale, zero, g_idx.clone()


def main():
    if not torch.cuda.is_available() or torch.cuda.get_device_name(0) != "NVIDIA GeForce RTX 4060 Laptop GPU":
        raise RuntimeError("Frozen GPU is unavailable")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(20261005)
    np.random.seed(20261005)
    started = time.perf_counter()
    prior_budget = json.loads(rv.BUDGET_PATH.read_text("utf-8")) if rv.BUDGET_PATH.exists() else {}
    rv.GPU_BUDGET_PREVIOUS = float(prior_budget.get("gpu_wall_elapsed_seconds", 0.0))
    rv.GPU_BUDGET_START = time.perf_counter()
    v28 = load_v28_runner()
    recovery_audit = json.loads((V28_RUN / "RECOVERY_AUDIT.json").read_text("utf-8"))
    v28_code = recovery_audit["checkpoint_code_bundle_sha256"]
    if recovery_audit.get("recovery_compatibility", "").find("exact old hash stored in every checkpoint") < 0:
        raise RuntimeError("v28 checkpoint recovery provenance is not the frozen one")
    complete, v28_layers = v28.list_completed_layers(
        "M2", v28_code, v28.sha_file(v28.CAL_PATH), v28.sha_file(v28.PROTOCOL_PATH))
    if complete != rv.LAYERS:
        raise RuntimeError(f"v28 M2 checkpoint chain incomplete: {complete}/{rv.LAYERS}")

    rec = rv.Recorder()
    token_samples = rv.token_rows()
    model = rv.load_model(rv.MODEL_DIR, rec, "M2_spot")
    for layer_index in range(7):
        rv.restore_layer(model, layer_index, v28_layers[layer_index])
    hidden, qkv_x, hidden_path, _ = rv.replay_completed_prefix(model, "M2", token_samples, 7, rec)

    ckpt_dir = V28_RUN / "checkpoints" / "M2"
    qkv_group = assert_checkpoint_group(
        ckpt_dir / "layer_07_group_qkv.pt", 7, "qkv", rv.MODULE_ORDERS["qkv"])
    o_group = assert_checkpoint_group(
        ckpt_dir / "layer_07_group_o.pt", 7, "o", rv.MODULE_ORDERS["o"])
    gate_group = assert_checkpoint_group(
        ckpt_dir / "layer_07_group_up_gate.pt", 7, "up_gate", rv.MODULE_ORDERS["up_gate"])
    qkv_hash = rv.rp.sha_memmap(qkv_x)
    if qkv_hash != qkv_group["metadata"]["input_sha256"]:
        raise RuntimeError("replayed L7 qkv activation does not match v28 frozen hash")
    layer = model.model.layers[7]
    for payload in (qkv_group, o_group):
        for name, state in payload["modules"].items():
            target = layer.get_submodule(name).weight
            with torch.no_grad():
                target.copy_(state["weight"].to(device=target.device, dtype=target.dtype))

    x_path = rv.ACT_DIR / "M2_L7_up_gate_spot.f16"
    x_mm, input_hash = rv.run_block_capture(
        model, layer, hidden, rv.MODULE_ORDERS["up_gate"], x_path,
        "M2_spot", 7, "spot_qkv_o_commit_to_up_gate_input", rec)
    expected_gate_state = gate_group["modules"]["mlp.gate_proj"]
    expected_up_meta = gate_group["metadata"]
    if input_hash != expected_up_meta["input_sha256"] or input_hash != expected_gate_state["input_sha256"]:
        raise RuntimeError("replayed L7 up_gate activation does not match v28 checkpoint input hash")

    gate = layer.get_submodule("mlp.gate_proj")
    pristine = gate.weight.detach().clone()
    pristine_hash = rv.sha_tensor(pristine)
    h_logs = []
    hessian, nsamples, hstats = rv.rp.build_hessian(
        gate, x_mm, h_logs, "v29 spot L7 gate_proj exact up_gate input")
    if nsamples != rv.N_DOCS or hstats["sha256"] != expected_gate_state["hessian_sha256"]:
        raise RuntimeError("reconstructed L7 gate H does not match v28 checkpoint H")
    h_hash_before = rv.sha_tensor(hessian)
    prepared = rv.prepare_native_state(hessian, nsamples=nsamples)
    c_mat = rv.build_quadratic_C(x_mm)
    c_hash_before = rv.sha_tensor(c_mat)
    x_hash_before = rv.rp.sha_memmap(x_mm)

    direct_rows, quadratic_rows = [], []
    q_candidates, s_candidates, z_candidates = [], [], []
    g_idx_ref = None
    candidate_s = []
    direct_s = []
    quadratic_s = []
    alpha1_native_parity = None
    for i, alpha in enumerate(rv.GRID):
        rv.check_gpu_budget(f"spot_candidate_{i}")
        rv.peak_reset(); t0 = time.perf_counter()
        result = rv.optimized_dynamic_gptq(pristine, prepared, alpha=alpha, check_cholesky=(alpha == 1.0))
        rv.sync(); candidate_elapsed = time.perf_counter() - t0
        candidate_s.append(candidate_elapsed)
        rec.record("M2_spot", "full_linear_alpha_candidate_gptq", 7,
                   "model.layers.7.mlp.gate_proj", candidate_elapsed,
                   alpha=f"{alpha:.1f}", detail="full output-row candidate; shared v28 H/Hinv; fresh candidate state")
        q = result.weight_fp16.detach().clone()
        scale = result.scales.detach().clone()
        zero = result.zeros.detach().clone()
        g_idx = result.g_idx.detach().clone()
        if g_idx_ref is None:
            g_idx_ref = g_idx.clone()
        elif not torch.equal(g_idx, g_idx_ref):
            raise RuntimeError(f"candidate g_idx changed at alpha={alpha}")
        if alpha == 1.0:
            native = rv.native_dynamic_gptq(pristine, hessian, nsamples=nsamples, alpha=1.0)
            alpha1_native_parity = {
                "weight": torch.equal(native.weight_fp16, q),
                "scale": torch.equal(native.scales, scale),
                "zero": torch.equal(native.zeros, zero),
                "g_idx": torch.equal(native.g_idx, g_idx),
            }
            rv.sync()
            del native
            if not all(alpha1_native_parity.values()):
                raise RuntimeError(f"L7 alpha=1 native GPTQ parity failed: {alpha1_native_parity}")
        rv.peak_reset(); t0 = time.perf_counter()
        direct = rv.rp.score_rows(x_mm, pristine, q)
        rv.sync(); direct_elapsed = time.perf_counter() - t0
        direct_s.append(direct_elapsed)
        rec.record("M2_spot", "all_row_direct_output_scoring", 7,
                   "model.layers.7.mlp.gate_proj", direct_elapsed,
                   alpha=f"{alpha:.1f}", detail="full Linear; v18 score_rows unchanged")
        rv.peak_reset(); t0 = time.perf_counter()
        quadratic = rv.quadratic_score_rows(c_mat, pristine, q)
        rv.sync(); quad_elapsed = time.perf_counter() - t0
        quadratic_s.append(quad_elapsed)
        rec.record("M2_spot", "all_row_quadratic_output_scoring", 7,
                   "model.layers.7.mlp.gate_proj", quad_elapsed,
                   alpha=f"{alpha:.1f}", detail="same candidate tensor as direct scorer")
        direct_rows.append(direct.detach().cpu().numpy().astype(np.float64, copy=True))
        quadratic_rows.append(quadratic.detach().cpu().numpy().astype(np.float64, copy=True))
        q_candidates.append(q); s_candidates.append(scale); z_candidates.append(zero)
        del result, q, scale, zero, g_idx, direct, quadratic

    direct_scores = np.stack(direct_rows)
    quadratic_scores = np.stack(quadratic_rows)
    v28_scores_path = V28_RUN / "audit" / "raw_scores" / "layer_07_scores.npz"
    with np.load(v28_scores_path, allow_pickle=False) as saved:
        saved_direct = saved["mlp_gate_proj"]
        saved_quad = saved["mlp_gate_proj_quadratic"]
    direct_exact_v28 = np.array_equal(direct_scores, saved_direct)
    quad_exact_v28 = np.array_equal(quadratic_scores, saved_quad)
    if not direct_exact_v28 or not quad_exact_v28:
        raise RuntimeError("L7 candidate scores do not reproduce v28 full-matrix score arrays")

    selection_started = time.perf_counter()
    abs_error = np.abs(quadratic_scores - direct_scores)
    direct_gap = np.partition(direct_scores, 1, axis=0)[1] - np.partition(direct_scores, 1, axis=0)[0]
    sensitive = 2.0 * abs_error.max(axis=0) >= direct_gap
    sensitive_rows = np.flatnonzero(sensitive)
    target_row = 884
    if target_row not in set(sensitive_rows.tolist()):
        raise RuntimeError("frozen v28 criterion failed to mark the known L7 row 884")
    rechecked_rows = []
    recheck_s = []
    if sensitive_rows.size:
        t0 = time.perf_counter()
        for i, alpha in enumerate(rv.GRID):
            rv.check_gpu_budget(f"spot_full_linear_recheck_{i}")
            rescored = rv.rp.score_rows(x_mm, pristine, q_candidates[i])
            rv.sync()
            rescored_np = rescored.detach().cpu().numpy().astype(np.float64, copy=False)
            recheck_s.append(time.perf_counter() - t0 if i == 0 else 0.0)
            rechecked_rows.append(rescored_np.copy())
            if not np.array_equal(rescored_np, direct_scores[i]):
                raise RuntimeError(f"full-Linear repeat direct score mismatch alpha={rv.GRID[i]}")
            del rescored
        recheck_elapsed = time.perf_counter() - t0
        rec.record("M2_spot", "sensitive_full_linear_direct_recheck", 7,
                   "model.layers.7.mlp.gate_proj", recheck_elapsed,
                   detail=json.dumps({"trigger_rows": sensitive_rows.tolist(),
                                      "recheck_rows_per_candidate": int(pristine.shape[0]),
                                      "candidate_calls": len(rv.GRID),
                                      "exact_to_initial_direct": True}))
        hybrid_scores = np.stack(rechecked_rows)
    else:
        recheck_elapsed = 0.0
        hybrid_scores = quadratic_scores.copy()
    direct_alpha, direct_best_loss = pick(direct_scores)
    hybrid_alpha, hybrid_best_loss = pick(hybrid_scores)
    direct_stitched = stitch(direct_alpha, q_candidates, s_candidates, z_candidates, g_idx_ref)
    hybrid_stitched = stitch(hybrid_alpha, q_candidates, s_candidates, z_candidates, g_idx_ref)
    saved_direct = (expected_gate_state["selected_alpha"].numpy(), expected_gate_state["weight"],
                    expected_gate_state["scales"], expected_gate_state["zeros"], expected_gate_state["g_idx"])
    direct_match_v28 = (
        np.array_equal(direct_alpha, saved_direct[0]) and
        torch.equal(direct_stitched[0].cpu(), saved_direct[1]) and
        torch.equal(direct_stitched[1].cpu(), saved_direct[2]) and
        torch.equal(direct_stitched[2].cpu(), saved_direct[3]) and
        torch.equal(direct_stitched[3].cpu(), saved_direct[4]))
    if not direct_match_v28:
        raise RuntimeError("L7 direct re-created selected W/scale/zero/g_idx differs from v28 checkpoint")
    tensor_equal = [torch.equal(a, b) for a, b in zip(direct_stitched, hybrid_stitched)]
    sensitivity_selection_stitch_seconds = time.perf_counter() - selection_started
    alpha_equal = np.array_equal(direct_alpha, hybrid_alpha)
    recheck_exact = bool(all(np.array_equal(rechecked_rows[i], direct_scores[i]) for i in range(len(rv.GRID)))) if sensitive_rows.size else True
    checks = {
        "input_hash_matches_v28_up_gate": input_hash == expected_up_meta["input_sha256"],
        "hessian_hash_matches_v28": hstats["sha256"] == expected_gate_state["hessian_sha256"],
        "direct_scores_match_v28_all_rows_all_alphas": direct_exact_v28,
        "quadratic_scores_match_v28_all_rows_all_alphas": quad_exact_v28,
        "alpha1_native_gptq_parity": alpha1_native_parity is not None and all(alpha1_native_parity.values()),
        "frozen_rule_includes_row_884": target_row in set(sensitive_rows.tolist()),
        "repeated_full_linear_scores_exact_to_initial_direct": recheck_exact,
        "direct_selected_state_matches_v28_W_scale_zero_g_idx": direct_match_v28,
        "hybrid_selection_equal_direct": alpha_equal,
        "hybrid_W_scale_zero_g_idx_equal_direct": all(tensor_equal),
        "original_weight_unchanged": rv.sha_tensor(gate.weight) == pristine_hash,
        "H_unchanged": rv.sha_tensor(hessian) == h_hash_before,
        "C_unchanged": rv.sha_tensor(c_mat) == c_hash_before,
        "input_memmap_unchanged": rv.rp.sha_memmap(x_mm) == x_hash_before,
    }
    np.savez_compressed(HERE / "spot" / "SPOT_CANDIDATE_SCORES.npz",
                        direct=direct_scores, quadratic=quadratic_scores,
                        hybrid=hybrid_scores, sensitive=sensitive)
    row = {
        "alpha_grid": rv.GRID,
        "row_884": {
            "direct_scores": direct_scores[:, target_row].tolist(),
            "quadratic_scores": quadratic_scores[:, target_row].tolist(),
            "max_abs_score_error": float(abs_error[:, target_row].max()),
            "direct_gap": float(direct_gap[target_row]),
            "sensitivity_lhs": float(2 * abs_error[:, target_row].max()),
            "direct_alpha": float(direct_alpha[target_row]),
            "quadratic_alpha": float(pick(quadratic_scores)[0][target_row]),
            "hybrid_alpha": float(hybrid_alpha[target_row]),
        },
        "sensitive_rows": sensitive_rows.tolist(),
        "sensitive_rows_count": int(sensitive_rows.size),
    }
    payload = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "module": "model.layers.7.mlp.gate_proj", "shape": list(pristine.shape),
        "v28_candidate_scores_reproduced": {"direct": direct_exact_v28, "quadratic": quad_exact_v28},
        "score_recheck_unit": "full Linear output rows x full 10-alpha grid; original all-row GEMM shape",
        "full_linear_direct_recheck_calls": len(rv.GRID) if sensitive_rows.size else 0,
        "full_linear_rows_rechecked_per_alpha": int(pristine.shape[0]) if sensitive_rows.size else 0,
        "candidate_cache": {
            "gpu_logical_bytes_q_scale_zero": sum(t.numel() * t.element_size()
                for seq in (q_candidates, s_candidates, z_candidates) for t in seq),
            "disk_io_bytes": 0, "cpu_transfer_bytes": 0,
        },
        "timing_seconds": {
            "spot_elapsed_including_model_load_prefix_capture_H_C_and_output": time.perf_counter()-started,
            "candidate_gptq_10_total": sum(candidate_s),
            "initial_direct_score_10_total": sum(direct_s),
            "quadratic_score_10_total": sum(quadratic_s),
            "full_linear_recheck_10_total": recheck_elapsed,
            "sensitivity_and_selection_stitch_seconds": sensitivity_selection_stitch_seconds,
        },
        "row_884_evidence": row,
        "checks": checks,
        "alpha1_native_parity": alpha1_native_parity,
        "v28_direct_selected_state_hash": expected_gate_state["state_sha256"],
        "candidate_pristine_weight_sha256": pristine_hash,
        "H_sha256": h_hash_before, "C_sha256": c_hash_before,
        "input_sha256": input_hash,
        "actual_cuda_peak_allocated_bytes": rec.peak_allocated,
        "actual_cuda_peak_reserved_bytes": rec.peak_reserved,
    }
    rv.write_json_atomic(HERE / "spot" / "SPOT_RECHECK.json", payload)
    rv.save_global_budget("v29_full_linear_spot_recheck_complete")
    rec.close()
    rv.close_mm(qkv_x); rv.close_mm(hidden); rv.close_mm(x_mm)
    del model
    gc.collect()
    if payload["status"] != "PASS":
        raise RuntimeError("v29 known-difference spot recheck failed")
    print(json.dumps({"status": payload["status"], "checks": checks,
                      "sensitive_rows": len(sensitive_rows), "direct_recheck_calls": payload["full_linear_direct_recheck_calls"],
                      "elapsed_seconds": payload["timing_seconds"]["spot_elapsed_including_model_load_prefix_capture_H_C_and_output"]}, indent=2))


if __name__ == "__main__":
    import gc
    main()
