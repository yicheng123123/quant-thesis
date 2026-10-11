"""Freeze the v29 sensitivity rule from immutable v28 score evidence."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from datetime import datetime, timezone

import numpy as np


HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent
ROOT = HERE.parents[3]
V18 = RESULTS / "range_w4_fullmodel_v18"
V28 = RESULTS / "range_w4_quadratic_full_m2_v28"
RUN28 = V28 / "validation"
AUDIT28 = RUN28 / "audit"
GRID = np.asarray([i / 10 for i in range(1, 11)], dtype=np.float64)
MODULES = [
    "self_attn.k_proj", "self_attn.v_proj", "self_attn.q_proj",
    "self_attn.o_proj", "mlp.up_proj", "mlp.gate_proj", "mlp.down_proj",
]


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    if (HERE / "PROTOCOL.json").exists():
        raise RuntimeError("v29 PROTOCOL.json already exists; frozen protocol is immutable")
    v28_protocol = json.loads((V28 / "PROTOCOL.json").read_text("utf-8"))
    v28_sources = v28_protocol["source_sha256"]
    for rel, expected in v28_sources.items():
        path = ROOT / rel
        if not path.is_file() or sha(path) != expected:
            raise RuntimeError(f"v28 frozen source hash mismatch: {rel}")

    rows: list[dict] = []
    sensitive: list[dict] = []
    module_rows: list[dict] = []
    score_count = 0
    for layer in range(24):
        score_path = AUDIT28 / "raw_scores" / f"layer_{layer:02d}_scores.npz"
        with np.load(score_path, allow_pickle=False) as saved:
            for module in MODULES:
                key = module.replace(".", "_")
                direct = saved[key]
                quadratic = saved[f"{key}_quadratic"]
                if direct.shape != quadratic.shape or direct.shape[0] != len(GRID):
                    raise RuntimeError(f"v28 score shape mismatch: L{layer} {module}")
                abs_error = np.abs(quadratic - direct)
                gap = np.partition(direct, 1, axis=0)[1] - np.partition(direct, 1, axis=0)[0]
                triggered = 2.0 * abs_error.max(axis=0) >= gap
                # Frozen M2 scans alpha in ascending order and updates on <=.
                direct_best = np.full(direct.shape[1], -1.0, dtype=np.float64)
                best = np.full(direct.shape[1], np.inf, dtype=np.float64)
                quad_best = np.full(direct.shape[1], -1.0, dtype=np.float64)
                quad_loss = np.full(direct.shape[1], np.inf, dtype=np.float64)
                for i, alpha in enumerate(GRID):
                    replace = (direct[i] < best) | ((direct[i] == best) & (alpha > direct_best))
                    best[replace] = direct[i, replace]
                    direct_best[replace] = alpha
                    replace_q = (quadratic[i] < quad_loss) | ((quadratic[i] == quad_loss) & (alpha > quad_best))
                    quad_loss[replace_q] = quadratic[i, replace_q]
                    quad_best[replace_q] = alpha
                score_count += direct.size
                mod_name = f"model.layers.{layer}.{module}"
                module_rows.append({
                    "layer": layer, "module": mod_name, "output_rows": int(direct.shape[1]),
                    "sensitive_rows": int(triggered.sum()), "sensitive": bool(triggered.any()),
                    "direct_selected_vs_quadratic_differences": int(np.count_nonzero(direct_best != quad_best)),
                })
                for row in np.flatnonzero(triggered):
                    sensitive.append({
                        "layer": layer, "module": mod_name, "row": int(row),
                        "direct_best_second_gap": float(gap[row]),
                        "max_abs_score_error": float(abs_error[:, row].max()),
                        "rule_lhs_2x_error": float(2.0 * abs_error[:, row].max()),
                        "direct_alpha": float(direct_best[row]), "quadratic_alpha": float(quad_best[row]),
                        "selection_diff": bool(direct_best[row] != quad_best[row]),
                    })

    # Compare independently rederived flags to the frozen v28 tabulation.
    old_sensitive = set()
    with (AUDIT28 / "SENSITIVE_ROWS.csv").open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            old_sensitive.add((int(row["layer"]), f"model.layers.{row['layer']}.{row['module']}", int(row["row"])))
    new_sensitive = {(r["layer"], r["module"], r["row"]) for r in sensitive}
    if old_sensitive != new_sensitive:
        raise RuntimeError(f"frozen sensitivity table mismatch: old={len(old_sensitive)} new={len(new_sensitive)}")
    target = (7, "model.layers.7.mlp.gate_proj", 884)
    if target not in new_sensitive:
        raise RuntimeError("frozen v28 empirical criterion does not cover the known selection difference")

    selection_diffs = json.loads((AUDIT28 / "NUMERICAL_SELECTION_DIFFERENCE.json").read_text("utf-8"))["selection_differences"]
    if len(selection_diffs) != 1 or (
        selection_diffs[0]["layer"], f"model.layers.{selection_diffs[0]['layer']}.{selection_diffs[0]['module']}",
        selection_diffs[0]["row"],
    ) != target:
        raise RuntimeError("v28 known selection-difference evidence changed")

    triggered_modules = [r for r in module_rows if r["sensitive"]]
    full_score_calls = len(triggered_modules) * len(GRID)
    rechecked_rows_per_alpha = sum(r["output_rows"] for r in triggered_modules)
    timing_path = RUN28 / "FORMAL_TIMING.csv"
    score_by_module: dict[str, float] = {}
    with timing_path.open(newline="", encoding="utf-8") as f:
        for item in csv.DictReader(f):
            if item["stage"] == "all_row_direct_output_scoring":
                score_by_module[item["module"]] = score_by_module.get(item["module"], 0.0) + float(item["wall_seconds"])
    triggered_names = {r["module"] for r in triggered_modules}
    est_recheck = sum(score_by_module.get(name, 0.0) for name in triggered_names)
    direct_total = sum(score_by_module.values())
    v28_timing = json.loads((AUDIT28 / "FINAL_SUMMARY.json").read_text("utf-8"))["timing_seconds"]
    trigger_path = HERE / "FROZEN_RECHECK_MODULES.csv"
    with trigger_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(module_rows[0]))
        writer.writeheader(); writer.writerows(module_rows)
    with (HERE / "FROZEN_SENSITIVE_ROWS.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(sensitive[0]))
        writer.writeheader(); writer.writerows(sensitive)

    protocol = {
        "name": "v29 quadratic fast score with original-score numerical recheck",
        "status": "FROZEN_BEFORE_V29_GPU_EXECUTION",
        "v18_protocol": v28_protocol["v18_protocol"],
        "inherit_from": {
            "v18_protocol_sha256": sha(V18 / "PROTOCOL.json"),
            "v28_protocol_sha256": sha(V28 / "PROTOCOL.json"),
            "v28_pure_quadratic_verdict": "FAIL: 1 output row differs; retained as-is",
            "model_checkpoint": "v18 M2; same immutable revision and all non-search settings",
        },
        "scope": {
            "model": "Qwen/Qwen2.5-0.5B-Instruct@7ae557604adf67be50417f59c2c2f167def9a775",
            "quantization": "W4A16 symmetric dynamic GPTQ G128, block=128, damp_percent=0.1, desc_act=false, original compensation",
            "alpha_grid": GRID.tolist(), "linear_modules": 168, "output_rows": 304128,
            "calibration": "v18 frozen WikiText-2 train 128 documents x 2048 tokens; exact M2 own-prefix activations",
            "selection": "per-row exact argmin; exact ties choose larger alpha",
            "heldout_ppl": "NOT_RUN",
        },
        "frozen_hybrid_rule": {
            "default_scorer": "v27/v28 complete quadratic scorer C=X.T@X/N and e@C then FP64 row reduction",
            "sensitivity_criterion": "2 * max_alpha(abs(quadratic_score-direct_score)) >= direct_best_second_gap, using v28 full-score empirical errors and exact frozen direct tie rule",
            "empirical_only": True,
            "row_allowlist": None,
            "trigger_unit": "one full output-row Linear because v18 score_rows forms delta for all rows and one GEMM with the full output dimension; no output-row chunking",
            "recheck": "for any Linear with >=1 sensitive row, run all 10 candidates through unchanged v18 score_rows over the full row matrix; mixed scorer uses rechecked direct scores for every row in that Linear, quadratic scores elsewhere",
            "selection": "same frozen exact argmin and larger-alpha exact-tie rule; no fixed alpha and no threshold relaxation",
            "claim_limit": "experience-based numerical guard only; no general error guarantee",
        },
        "frozen_evidence_audit": {
            "v28_sensitive_rows": len(sensitive), "v28_triggered_full_linears": len(triggered_modules),
            "total_full_linears": 168, "rechecked_full_score_calls": full_score_calls,
            "rechecked_output_rows_per_candidate": rechecked_rows_per_alpha,
            "rechecked_row_alpha_values": rechecked_rows_per_alpha * len(GRID),
            "rechecked_linear_fraction": len(triggered_modules) / 168,
            "direct_score_time_v28_all_s": direct_total,
            "direct_score_time_v28_trigger_subset_s": est_recheck,
            "trigger_subset_fraction_of_direct_score_time": est_recheck / direct_total,
            "row_884_covered": True,
            "row_884_direct_best_second_gap": next(r["direct_best_second_gap"] for r in sensitive if (r["layer"], r["module"], r["row"]) == target),
            "row_884_max_abs_score_error": next(r["max_abs_score_error"] for r in sensitive if (r["layer"], r["module"], r["row"]) == target),
            "v28_full_run_wall_s": v28_timing["first_dual_formal_pass_wall"],
            "v28_hybrid_recheck_subset_historical_timing_estimate_s": est_recheck,
            "original_scoring_output_row_chunks": 1,
            "direct_score_calls_for_triggered_linear_alpha_grid": 10,
            "sensitivity_set_matches_saved_v28_csv": True,
        },
        "validation": {
            "phase1": "full v29 score-mode dual run: one candidate per alpha; direct and quadratic share Q; recheck all 10 full-matrix direct scores for every trigger Linear; commit direct selection; save quadratic and hybrid results and compare tensors/selections",
            "phase2_gate": "standalone hybrid-only M2 is optional and may run only if phase1 passes and remaining cumulative 90-minute budget covers measured/projected full run; never derive a claimed wall by subtracting score stages",
            "candidate_cache": "retain each current Linear's ten Q/scale/zero candidate tensors on GPU until sensitivity decision and any recheck finish; record logical bytes and observed CUDA peak; no disk candidate cache",
            "recoverability": "atomic checkpoints after dependency groups and layers; resume only verified contiguous groups; budget includes load, capture, warmup, candidate scoring, rechecks, checkpoint and final save",
            "gpu_wall_budget_seconds": 5400,
            "ppl": "NOT_RUN",
        },
        "source_sha256": {
            "v18_PROTOCOL.json": sha(V18 / "PROTOCOL.json"),
            "v18_run_precheck.py": sha(V18 / "run_precheck.py"),
            "v18_run_formal.py": sha(V18 / "run_formal.py"),
            "v28_PROTOCOL.json": sha(V28 / "PROTOCOL.json"),
            "v28_run_v28.py": sha(V28 / "run_v28.py"),
            "v28_sensitive_rows.csv": sha(AUDIT28 / "SENSITIVE_ROWS.csv"),
            "v28_known_difference.json": sha(AUDIT28 / "NUMERICAL_SELECTION_DIFFERENCE.json"),
            "v28_raw_scores_layer_00_23": {f"layer_{i:02d}": sha(AUDIT28 / "raw_scores" / f"layer_{i:02d}_scores.npz") for i in range(24)},
        },
        "new_artifacts": "v29 directory only; v18/v28 read-only",
    }
    (HERE / "PROTOCOL.json").write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + "\n", "utf-8")
    (HERE / "FROZEN_TRIGGER_AUDIT.json").write_text(json.dumps(protocol["frozen_evidence_audit"], ensure_ascii=False, indent=2) + "\n", "utf-8")
    (HERE / "RUN_MANIFEST.json").write_text(json.dumps({
        "experiment": protocol["name"], "status": "FROZEN_NOT_STARTED",
        "protocol_sha256": sha(HERE / "PROTOCOL.json"),
        "source_evidence_sha256": protocol["source_sha256"],
        "run_label": "validation", "score_mode": "dual",
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }, ensure_ascii=False, indent=2) + "\n", "utf-8")
    print(json.dumps(protocol["frozen_evidence_audit"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
