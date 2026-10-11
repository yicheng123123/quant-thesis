from __future__ import annotations

import csv
import hashlib
import json
import platform
import shutil
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import torch


OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[3]
RESULTS = ROOT / "experiments/range_search_w4/results"
V18 = RESULTS / "range_w4_fullmodel_v18"
PRECHECK = RESULTS / "range_w4_candidate_budget_precheck"
V171 = RESULTS / "range_w4_m2_parity_v17_1"
V173 = RESULTS / "range_w4_m2_runtime_v17_3"
SOURCE_FILES = {
    "v18_protocol": V18 / "PROTOCOL.json",
    "v18_runner": V18 / "run_formal.py",
    "v18_precheck_runner": V18 / "run_precheck.py",
    "v18_calibration_manifest": V18 / "CALIBRATION_MANIFEST.json",
    "v18_calibration_validation": V18 / "CALIBRATION_VALIDATION.json",
    "v18_row_results": V18 / "M2_ROW_RESULTS.csv",
    "v18_module_summary": V18 / "M2_MODULE_SUMMARY.csv",
    "candidate_precheck_summary": PRECHECK / "RUN_SUMMARY.json",
    "candidate_precheck_alpha_usage": PRECHECK / "ALPHA_USAGE.csv",
    "candidate_precheck_source_audit": PRECHECK / "SOURCE_AUDIT.csv",
    "native_m2": V171 / "native_m2.py",
    "optimized_native_m2": V173 / "optimized_native_m2.py",
}


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def copy_protocol_assets(run_dir: Path) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    for name in (
        "PROTOCOL.json", "CALIBRATION_MANIFEST.json", "CALIBRATION_VALIDATION.json",
        "UPSTREAM_HASH_AUDIT.json", "MATCHED_SHAPE_PRECHECK.json", "KV_SHAPE_PRECHECK.json",
        "FINAL_COST_MODEL.json", "FORMAL_RUNNER_SMOKE.json", "FORMAL_RUN_MANIFEST.json",
    ):
        shutil.copy2(V18 / name, run_dir / name)
    shutil.copy2(V18 / "run_precheck.py", run_dir / "run_precheck.py")


def replace_once(source: str, old: str, new: str, label: str) -> str:
    count = source.count(old)
    if count != 1:
        raise RuntimeError(f"expected one {label} patch anchor, found {count}")
    return source.replace(old, new, 1)


def patch_runner(run_dir: Path) -> None:
    source = (V18 / "run_formal.py").read_text(encoding="utf-8")
    source = replace_once(
        source,
        'OUT = Path(__file__).resolve().parent\nROOT = OUT.parents[3]\nRESULTS = ROOT / "experiments/range_search_w4/results"\n',
        'OUT = Path(__file__).resolve().parent\nROOT = OUT.parents[4]\nRESULTS = ROOT / "experiments/range_search_w4/results"\nV40_ROOT = OUT.parent\nRUN_CONFIG_PATH = OUT / "V40_RUN_CONFIG.json"\nCANDIDATE_MANIFEST_PATH = V40_ROOT / "CANDIDATE_MANIFEST.json"\nRUN_CONFIG = json.loads(RUN_CONFIG_PATH.read_text(encoding="utf-8"))\nCANDIDATE_MANIFEST = json.loads(CANDIDATE_MANIFEST_PATH.read_text(encoding="utf-8"))\n',
        "nested run root",
    )
    source = replace_once(
        source,
        'GRID = [float(x) for x in PROTOCOL["m2"]["alpha_grid"]]\n',
        'GRID = [float(x) for x in PROTOCOL["m2"]["alpha_grid"]]\n'
        'EXPECTED_CANDIDATE_EXECUTIONS = int(RUN_CONFIG["expected_candidate_executions"])\n'
        'EXPECTED_ROW_ALPHA_SCORE_SLOTS = int(RUN_CONFIG["expected_row_alpha_score_slots"])\n'
        '\n'
        'def candidate_grid(full_name: str) -> list[float]:\n'
        '    if RUN_CONFIG["candidate_mode"] == "full_grid":\n'
        '        return list(GRID)\n'
        '    item = CANDIDATE_MANIFEST["modules"].get(full_name)\n'
        '    if item is None:\n'
        '        raise RuntimeError(f"frozen candidate set missing {full_name}")\n'
        '    values = [float(x) for x in item["candidate_alphas"]]\n'
        '    if not values or values != [x for x in GRID if x in set(values)]:\n'
        '        raise RuntimeError(f"candidate order differs from original scalar10 grid: {full_name}")\n'
        '    if 1.0 not in values:\n'
        '        raise RuntimeError(f"frozen candidate set omits alpha=1: {full_name}")\n'
        '    return values\n',
        "frozen per-module grid",
    )
    source = replace_once(
        source,
        'CODE_BUNDLE = [Path(__file__), OUT / "run_precheck.py", V171 / "native_m2.py", V173 / "optimized_native_m2.py"]\n',
        'CODE_BUNDLE = [Path(__file__), OUT / "run_precheck.py", V171 / "native_m2.py",\n'
        '               V173 / "optimized_native_m2.py", RUN_CONFIG_PATH, CANDIDATE_MANIFEST_PATH]\n',
        "checkpoint code bundle",
    )
    source = replace_once(
        source,
        '        formal_counts = old_report.get("formal_counts") or {\n'
        '            "full_linear_alpha_candidate_executions": 1680 if method == "M2" else 0,\n'
        '            "row_alpha_score_slots": 3041280 if method == "M2" else 0,\n'
        '            "alpha1_native_parity_reference_executions": 35 if method == "M2" else 0,\n'
        '            "layers_completed_before_this_process": LAYERS,\n'
        '        }\n',
        '        formal_counts = old_report.get("formal_counts")\n'
        '        if not formal_counts:\n'
        '            formal_counts = {"full_linear_alpha_candidate_executions": 0, "row_alpha_score_slots": 0,\n'
        '                             "alpha1_native_parity_reference_executions": 0,\n'
        '                             "layers_completed_before_this_process": LAYERS}\n'
        '            for layer_i, payload in enumerate(payloads):\n'
        '                for submodule, saved in payload["modules"].items():\n'
        '                    n_candidates = len(saved["candidate_seconds"])\n'
        '                    formal_counts["full_linear_alpha_candidate_executions"] += n_candidates\n'
        '                    formal_counts["row_alpha_score_slots"] += int(saved["weight"].shape[0]) * n_candidates\n'
        '                    if layer_i in SPOT_LAYERS and 1.0 in candidate_grid(f"model.layers.{layer_i}.{submodule}"):\n'
        '                        formal_counts["alpha1_native_parity_reference_executions"] += 1\n',
        "recovery counts from checkpoints",
    )
    source = replace_once(
        source,
        '    for alpha in GRID:\n        peak_reset(); start = time.perf_counter()\n',
        '    module_grid = candidate_grid(name)\n'
        '    for alpha in module_grid:\n        peak_reset(); start = time.perf_counter()\n',
        "candidate loop",
    )
    source = replace_once(
        source,
        '                  f"candidates={formal_counts[\'full_linear_alpha_candidate_executions\']}/1680", flush=True)\n',
        '                  f"candidates={formal_counts[\'full_linear_alpha_candidate_executions\']}/"\n'
        '                  f"{EXPECTED_CANDIDATE_EXECUTIONS}", flush=True)\n',
        "progress count",
    )
    source = replace_once(
        source,
        '            if method == "M2" and not np.isin(alpha, np.asarray(GRID, dtype=np.float64)).all():\n'
        '                raise RuntimeError(f"{method} L{layer_index} {name} selected alpha outside frozen grid")\n'
        '            expected_candidates = len(GRID) if method == "M2" else 1\n'
        '            if state["candidate_losses"].shape != (expected_candidates, rows):\n',
        '            module_grid = candidate_grid(f"model.layers.{layer_index}.{name}") if method == "M2" else [1.0]\n'
        '            if method == "M2" and not np.isin(alpha, np.asarray(module_grid, dtype=np.float64)).all():\n'
        '                raise RuntimeError(f"{method} L{layer_index} {name} selected alpha outside frozen grid")\n'
        '            expected_candidates = len(module_grid)\n'
        '            if state["candidate_losses"].shape != (expected_candidates, rows):\n',
        "checkpoint alpha and score validation",
    )
    source = replace_once(
        source,
        '              "selected_alpha_grid": GRID if method == "M2" else [1.0],\n',
        '              "selected_alpha_grid": (GRID if RUN_CONFIG["candidate_mode"] == "full_grid" else\n'
        '                  sorted({float(a) for item in CANDIDATE_MANIFEST["modules"].values()\n'
        '                          for a in item["candidate_alphas"]})) if method == "M2" else [1.0],\n',
        "quantization correctness summary",
    )
    source = replace_once(
        source,
        '    for layer_index in range(completed, LAYERS):\n',
        '    max_layers = min(LAYERS, int(RUN_CONFIG.get("max_layers", LAYERS)))\n'
        '    last_completed_layer = completed - 1\n'
        '    for layer_index in range(completed, max_layers):\n',
        "resumable pilot layer bound",
    )
    source = replace_once(
        source,
        '        checkpoint_path, checkpoint_seconds = save_layer_checkpoint(method, layer_index, layer_payload)\n',
        '        checkpoint_path, checkpoint_seconds = save_layer_checkpoint(method, layer_index, layer_payload)\n'
        '        last_completed_layer = layer_index\n',
        "completed layer tracking",
    )
    source = replace_once(
        source,
        '    if modules_seen not in (0, 168):\n'
        '        raise RuntimeError(f"unexpected modules processed during this process: {modules_seen}")\n'
        '    quant_check = validate_layer_checkpoints(method, code_hash, cal_hash, protocol_hash)\n',
        '    expected_new_modules = max(0, max_layers - completed) * sum(len(x) for x in MODULE_ORDERS.values())\n'
        '    if modules_seen != expected_new_modules:\n'
        '        raise RuntimeError(f"unexpected modules processed during this process: {modules_seen}/{expected_new_modules}")\n'
        '    if max_layers < LAYERS:\n'
        '        n_complete = last_completed_layer + 1\n'
        '        if n_complete != max_layers:\n'
        '            raise RuntimeError(f"pilot stopped before frozen layer boundary: {n_complete}/{max_layers}")\n'
        '        state = {"method": method, "status": "PILOT_COMPLETE", "layers_completed": n_complete,\n'
        '                 "formal_counts": formal_counts, "code_bundle_sha256": code_hash,\n'
        '                 "candidate_mode": RUN_CONFIG["candidate_mode"],\n'
        '                 "last_completed_layer": last_completed_layer,\n'
        '                 "completed_utc": datetime.now(timezone.utc).isoformat()}\n'
        '        write_json_atomic(OUT / method / "PILOT_RUN_REPORT.json", state)\n'
        '        del model\n'
        '        gc.collect(); torch.cuda.empty_cache()\n'
        '        return state\n'
        '    quant_check = validate_layer_checkpoints(method, code_hash, cal_hash, protocol_hash)\n',
        "pilot complete block boundary",
    )
    source = replace_once(
        source,
        '        "model_revision": PROTOCOL["model"]["revision"],\n'
        '        "format": PROTOCOL["quantization"],\n',
        '        "model_revision": PROTOCOL["model"]["revision"],\n'
        '        "format": PROTOCOL["quantization"],\n'
        '        "v40_arm": RUN_CONFIG["arm"],\n'
        '        "candidate_mode": RUN_CONFIG["candidate_mode"],\n'
        '        "candidate_manifest_sha256": sha_file(CANDIDATE_MANIFEST_PATH),\n',
        "final metadata provenance",
    )
    source = replace_once(
        source,
        'def finalize_row_artifacts():\n    for method in ("M1", "M2"):\n',
        'def finalize_row_artifacts():\n    for method in ("M2",):\n',
        "M2-only artifact finalizer",
    )
    source = replace_once(
        source,
        'def main():\n    if not torch.cuda.is_available() or torch.cuda.get_device_name(0) != "NVIDIA GeForce RTX 4060 Laptop GPU":\n',
        'def main():\n    if sha_file(CANDIDATE_MANIFEST_PATH) != RUN_CONFIG["candidate_manifest_sha256"]:\n'
        '        raise RuntimeError("frozen candidate manifest hash mismatch")\n'
        '    if sha_file(PROTOCOL_PATH) != RUN_CONFIG["protocol_sha256"]:\n'
        '        raise RuntimeError("frozen v18 protocol hash mismatch")\n'
        '    if not torch.cuda.is_available() or torch.cuda.get_device_name(0) != "NVIDIA GeForce RTX 4060 Laptop GPU":\n',
        "frozen artifact runtime verification",
    )
    source = replace_once(
        source,
        '        m1_state = run_method("M1", token_samples, rec, code_hash, cal_hash, protocol_hash)\n'
        '        update_manifest(status="M1_COMPLETE_M2_RUNNING", results={\n'
        '            "matched_shape_precheck": "PASS", "cost_gate": "M2_COST_GO", "m1": "COMPLETE",\n'
        '            "m2": "RUNNING", "wikitext_evaluation": "NOT_STARTED", "c4_evaluation": "NOT_STARTED"})\n'
        '        m2_state = run_method("M2", token_samples, rec, code_hash, cal_hash, protocol_hash)\n'
        '        artifacts = finalize_row_artifacts()\n',
        '        m2_state = run_method("M2", token_samples, rec, code_hash, cal_hash, protocol_hash)\n'
        '        if RUN_CONFIG.get("pilot_only", False):\n'
        '            runtime = {"status": "PILOT_COMPLETE", "arm": RUN_CONFIG["arm"],\n'
        '                       "m2": m2_state, "gpu_peak_allocated_bytes": rec.peak_allocated,\n'
        '                       "gpu_peak_reserved_bytes": rec.peak_reserved,\n'
        '                       "timing_csv": "FORMAL_TIMING.csv",\n'
        '                       "completed_utc": datetime.now(timezone.utc).isoformat()}\n'
        '            write_json_atomic(FORMAL_RUNTIME, runtime)\n'
        '            update_manifest(status="PILOT_COMPLETE", results={"m2": "PILOT_COMPLETE",\n'
        '                "full_model": "NOT_RUN", "held_or_ppl": "NOT_RUN"})\n'
        '            print(json.dumps(runtime, indent=2))\n'
        '            return\n'
        '        artifacts = finalize_row_artifacts()\n',
        "independent M2-only execution",
    )
    source = replace_once(
        source,
        '            "m2_1680_full_linear_alpha_candidates": m2_state["formal_counts"]["full_linear_alpha_candidate_executions"] == 1680,\n'
        '            "m2_3041280_row_alpha_score_slots": m2_state["formal_counts"]["row_alpha_score_slots"] == 3041280,\n'
        '            "m2_35_alpha1_native_spot_references": m2_state["formal_counts"]["alpha1_native_parity_reference_executions"] == 35,\n'
        '            "m1_reload_output_match": m1_state["reload_correctness"]["pass"],\n'
        '            "m2_reload_output_match": m2_state["reload_correctness"]["pass"],\n',
        '            "m2_candidate_execution_count": m2_state["formal_counts"]["full_linear_alpha_candidate_executions"] == EXPECTED_CANDIDATE_EXECUTIONS,\n'
        '            "m2_row_alpha_score_slots": m2_state["formal_counts"]["row_alpha_score_slots"] == EXPECTED_ROW_ALPHA_SCORE_SLOTS,\n'
        '            "m2_35_alpha1_native_spot_references": m2_state["formal_counts"]["alpha1_native_parity_reference_executions"] == 35,\n'
        '            "m2_reload_output_match": m2_state["reload_correctness"]["pass"],\n',
        "formal run count gates",
    )
    source = replace_once(
        source,
        '            "m1_168_linears_304128_rows": m1_state["quantization_correctness"]["linear_modules"] == 168 and m1_state["quantization_correctness"]["output_rows"] == 304128,\n'
        '            "m2_168_linears_304128_rows": m2_state["quantization_correctness"]["linear_modules"] == 168 and m2_state["quantization_correctness"]["output_rows"] == 304128,\n',
        '            "m2_168_linears_304128_rows": m2_state["quantization_correctness"]["linear_modules"] == 168 and m2_state["quantization_correctness"]["output_rows"] == 304128,\n',
        "remove irrelevant M1 gate",
    )
    source = replace_once(
        source,
        '            "m1": m1_state, "m2": m2_state,\n'
        '            "formal_work_units": {"m2_full_linear_alpha_candidate_executions": 1680,\n'
        '                "m2_row_alpha_score_slots": 3041280,\n'
        '                "m2_auxiliary_alpha1_native_parity_references": 35,\n'
        '                "m1_native_full_linear_gptq_executions": 168,\n'
        '                "m2_output_rows": 304128},\n',
        '            "arm": RUN_CONFIG["arm"], "candidate_manifest_sha256": sha_file(CANDIDATE_MANIFEST_PATH),\n'
        '            "m2": m2_state, "formal_work_units": {"m2_full_linear_alpha_candidate_executions": EXPECTED_CANDIDATE_EXECUTIONS,\n'
        '                "m2_row_alpha_score_slots": EXPECTED_ROW_ALPHA_SCORE_SLOTS,\n'
        '                "m2_auxiliary_alpha1_native_parity_references": 35,\n'
        '                "m2_output_rows": 304128},\n',
        "M2-only runtime summary",
    )
    source = replace_once(
        source,
        '        print(json.dumps({"status": runtime["status"], "m1_seconds": m1_state["wall_seconds_including_load_replay_save"],\n'
        '                          "m2_seconds": m2_state["wall_seconds_including_load_replay_save"],\n'
        '                          "total_seconds": runtime["overall_quantization_wall_seconds"],\n'
        '                          "candidate_executions": 1680, "row_alpha_scores": 3041280,\n'
        '                          "peak_allocated_bytes": rec.peak_allocated,\n'
        '                          "peak_reserved_bytes": rec.peak_reserved}, indent=2))\n',
        '        print(json.dumps({"status": runtime["status"], "arm": RUN_CONFIG["arm"],\n'
        '                          "m2_seconds": m2_state["wall_seconds_including_load_replay_save"],\n'
        '                          "total_seconds": runtime["overall_quantization_wall_seconds"],\n'
        '                          "candidate_executions": EXPECTED_CANDIDATE_EXECUTIONS,\n'
        '                          "row_alpha_scores": EXPECTED_ROW_ALPHA_SCORE_SLOTS,\n'
        '                          "peak_allocated_bytes": rec.peak_allocated,\n'
        '                          "peak_reserved_bytes": rec.peak_reserved}, indent=2))\n',
        "M2-only console summary",
    )
    (run_dir / "run_formal.py").write_text(source, encoding="utf-8")

    precheck = (run_dir / "run_precheck.py").read_text(encoding="utf-8")
    if "ROOT = OUT.parents[3]" in precheck:
        precheck = replace_once(precheck, "ROOT = OUT.parents[3]", "ROOT = OUT.parents[4]", "precheck nested run root")
    elif "ROOT = OUT.parents[4]" not in precheck:
        raise RuntimeError("precheck runner root does not match the nested v40 layout")
    (run_dir / "run_precheck.py").write_text(precheck, encoding="utf-8")


def main() -> None:
    if (OUT / "FREEZE_LOCK.json").exists():
        raise RuntimeError("v40 freeze lock already exists; refusing to regenerate frozen artifacts")
    if not V18.exists() or not PRECHECK.exists():
        raise FileNotFoundError("required v18 and candidate precheck directories are missing")
    pre_summary = json.loads((PRECHECK / "RUN_SUMMARY.json").read_text(encoding="utf-8"))
    if pre_summary.get("status") != "COMPLETE_ARTIFACT_ONLY_CANDIDATE_BUDGET_PRECHECK":
        raise RuntimeError("candidate budget precheck is not complete")
    if pre_summary.get("gpu_initializations") != 0 or pre_summary.get("gptq_runs") != 0 or pre_summary.get("p3_read") is not False:
        raise RuntimeError("precheck provenance does not match requested artifact-only source")
    for name, path in SOURCE_FILES.items():
        if not path.is_file():
            raise FileNotFoundError(f"missing source {name}: {path}")
    v18_protocol = json.loads(SOURCE_FILES["v18_protocol"].read_text(encoding="utf-8"))
    full_grid = [float(x) for x in v18_protocol["m2"]["alpha_grid"]]
    if full_grid != [round(x, 1) for x in [i / 10 for i in range(1, 11)]]:
        raise RuntimeError(f"unexpected scalar10 order: {full_grid}")
    summary_rows = list(csv.DictReader(SOURCE_FILES["v18_module_summary"].open(encoding="utf-8-sig", newline="")))
    usage_rows = list(csv.DictReader(SOURCE_FILES["candidate_precheck_alpha_usage"].open(encoding="utf-8-sig", newline="")))
    if len(summary_rows) != 168 or len({r["module"] for r in summary_rows}) != 168:
        raise RuntimeError("v18 M2 module inventory is not 168 unique Linear modules")
    summary = {r["module"]: r for r in summary_rows}
    if sum(int(r["rows"]) for r in summary_rows) != 304128:
        raise RuntimeError("v18 M2 output row count is not 304128")
    usage: dict[tuple[str, str], int] = {}
    for r in usage_rows:
        if r["scope"] == "M2_FULL_PER_MODULE":
            key = (r["unit"], r["alpha"])
            usage[key] = int(r["selected_rows"])
    modules: dict[str, dict] = {}
    candidate_executions = 0
    score_slots = 0
    winner_total = Counter()
    for full_name, row in sorted(summary.items()):
        layer = int(row["layer"])
        suffix = full_name.split(".", 3)[-1]
        candidate_set = {a for a in full_grid if usage.get((full_name, f"{a:.1f}"), 0) > 0}
        winner_counts = {f"{a:.1f}": usage.get((full_name, f"{a:.1f}"), 0) for a in full_grid}
        for alpha, count in winner_counts.items():
            winner_total[alpha] += count
        candidate_set.add(1.0)
        candidates = [a for a in full_grid if a in candidate_set]
        rows = int(row["rows"])
        candidate_executions += len(candidates)
        score_slots += len(candidates) * rows
        modules[full_name] = {
            "layer": layer,
            "submodule": suffix,
            "rows": rows,
            "candidate_alphas": candidates,
            "candidate_count": len(candidates),
            "forced_alpha_1": winner_counts["1.0"] == 0,
            "source_winner_row_counts": winner_counts,
            "row_alpha_score_slots": rows * len(candidates),
        }
    if candidate_executions != 675:
        raise RuntimeError(f"candidate union count mismatch: {candidate_executions} != 675")
    if score_slots != 1287040:
        raise RuntimeError(f"row-alpha score count mismatch: {score_slots} != 1,287,040")
    expected_selected = {f"{float(k):.1f}": int(v) for k, v in pre_summary["m2_full_model"]["selected_alpha_counts"].items()}
    if dict(winner_total) != expected_selected:
        raise RuntimeError("ALPHA_USAGE module winner counts do not reproduce the precheck global alpha totals")
    # Independently validate the per-module winner population against the frozen full row table.
    row_counts: dict[str, Counter] = {}
    with SOURCE_FILES["v18_row_results"].open(encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            row_counts.setdefault(r["module"], Counter())[f"{float(r['selected_alpha']):.1f}"] += 1
    if set(row_counts) != set(modules):
        raise RuntimeError("v18 row table module inventory differs from candidate manifest")
    for name, counts in row_counts.items():
        expected = modules[name]["source_winner_row_counts"]
        if dict(counts) != {a: n for a, n in expected.items() if n}:
            raise RuntimeError(f"candidate source winner count mismatch in {name}")
        if any(float(r["selected_alpha"]) not in modules[name]["candidate_alphas"] for r in []):
            raise AssertionError("unreachable")

    # Obtain true matrix shapes from the completed v18 layer-0 checkpoint, without GPU/model work.
    old_layer0 = torch.load(V18 / "checkpoints/M2/layer_00.pt", map_location="cpu", weights_only=False)
    shapes = {}
    for full_name, item in modules.items():
        state = old_layer0["modules"][item["submodule"]]
        shape = list(state["weight"].shape)
        if shape[0] != item["rows"]:
            raise RuntimeError(f"shape/row mismatch for {full_name}")
        item["shape"] = shape
        item["in_features"] = shape[1]
        shapes[full_name] = shape
    del old_layer0

    sources = {name: {"relative_path": str(path.relative_to(ROOT)), "sha256": sha_file(path), "bytes": path.stat().st_size}
               for name, path in SOURCE_FILES.items()}
    candidate_manifest = {
        "schema": "v40-candidate-manifest-v1",
        "source_experiment": "v18 M2 per-row selected alpha winners + alpha=1, derived from candidate_budget_precheck ALPHA_USAGE.csv",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "candidate_order_rule": "original v18 scalar10 order; no re-ranking",
        "full_grid": full_grid,
        "linears": len(modules),
        "rows": 304128,
        "full_grid_candidate_executions": 1680,
        "winner_union_candidate_executions": candidate_executions,
        "candidate_execution_reduction_fraction": 1.0 - candidate_executions / 1680.0,
        "full_grid_row_alpha_score_slots": 3041280,
        "winner_union_row_alpha_score_slots": score_slots,
        "global_union_for_information_only": [a for a in full_grid if any(a in x["candidate_alphas"] for x in modules.values())],
        "module_candidate_count_histogram": {str(k): sum(v["candidate_count"] == k for v in modules.values()) for k in sorted({v["candidate_count"] for v in modules.values()})},
        "source_files": sources,
        "modules": modules,
        "construction_boundary": "No candidates generated here; row winner sets are copied from frozen v18/precheck evidence and cross-counted against all 304128 v18 row records.",
    }
    manifest_path = OUT / "CANDIDATE_MANIFEST.json"
    write_json(manifest_path, candidate_manifest)
    manifest_hash = sha_file(manifest_path)
    shutil.copy2(SOURCE_FILES["v18_protocol"], OUT / "PROTOCOL.json")
    v18_protocol_hash = sha_file(OUT / "PROTOCOL.json")
    frozen = {
        "experiment": "v40 independent per-Linear winner-union candidate execution",
        "frozen_utc": datetime.now(timezone.utc).isoformat(),
        "base_protocol": "v18 formal M2, copied unchanged to PROTOCOL.json",
        "base_protocol_sha256": v18_protocol_hash,
        "only_method_difference": "candidate alpha set per Linear: full scalar10 in A; previous M2 row-winner union plus alpha=1 in B",
        "model_and_quantization": v18_protocol["model"],
        "quantization": v18_protocol["quantization"],
        "m2": v18_protocol["m2"],
        "calibration": {"manifest_sha256": sources["v18_calibration_manifest"]["sha256"], **v18_protocol["calibration"]},
        "candidate_manifest_sha256": manifest_hash,
        "candidate_population": {"linears": 168, "rows": 304128, "A_candidate_executions": 1680,
                                 "B_candidate_executions": 675, "B_row_alpha_score_slots": score_slots,
                                 "candidate_execution_reduction_fraction": 1.0 - 675 / 1680},
        "counts_exclude": ["complete-block0 pilot", "native alpha=1 parity references", "preheat/model load", "any failed attempts"],
        "execution_design": {"order": ["PILOT_B_BLOCK0", "A_SCALAR10_FULL", "B_WINNER_UNION_FULL"],
                             "fresh_model_each_run": True, "candidate_activation_or_H_cache_shared_between_A_B": False,
                             "resume": "atomic completed decoder-layer checkpoints; an interrupted current layer is replayed from prior complete layer"},
        "evaluation_boundary": "No held, P3, or PPL is read or executed in v40.",
        "source_hashes": sources,
        "cost_budget": {"hard_cap_seconds": 5400, "execution_cutoff_seconds": 4700,
                        "wrap_up_reserve_seconds": 700, "new_gpu_experiment_only": True},
    }
    write_json(OUT / "FROZEN_PROTOCOL.json", frozen)

    # A separate local tree per run ensures activations, H state, candidates and checkpoints are not reused.
    assets = ["PROTOCOL.json", "CALIBRATION_MANIFEST.json", "CALIBRATION_VALIDATION.json", "UPSTREAM_HASH_AUDIT.json",
              "MATCHED_SHAPE_PRECHECK.json", "KV_SHAPE_PRECHECK.json", "FINAL_COST_MODEL.json", "FORMAL_RUNNER_SMOKE.json"]
    pilot_calls = sum(item["candidate_count"] for item in modules.values() if item["layer"] == 0)
    pilot_slots = sum(item["row_alpha_score_slots"] for item in modules.values() if item["layer"] == 0)
    if pilot_calls != 27 or pilot_slots != 49792:
        raise RuntimeError(f"block0 pilot count mismatch: {pilot_calls} GPTQ calls, {pilot_slots} score slots")
    run_specs = [
        ("PILOT", "PILOT_B_BLOCK0", "winner_union", True, 1, pilot_calls, pilot_slots),
        ("A", "A_SCALAR10_FULL", "full_grid", False, 24, 1680, 3041280),
        ("B", "B_WINNER_UNION_FULL", "winner_union", False, 24, 675, score_slots),
    ]
    for dirname, arm, mode, pilot_only, max_layers, calls, slots_expected in run_specs:
        run_dir = OUT / dirname
        copy_protocol_assets(run_dir)
        for subdir in ("raw/activations", "checkpoints/M2", "M2/checkpoint/final", "finalization"):
            (run_dir / subdir).mkdir(parents=True, exist_ok=True)
        config = {
            "schema": "v40-run-config-v1", "arm": arm, "candidate_mode": mode,
            "pilot_only": pilot_only, "max_layers": max_layers,
            "expected_candidate_executions": calls,
            "expected_row_alpha_score_slots": slots_expected,
            "expected_alpha1_native_parity_references": 7 if pilot_only else 35,
            "candidate_manifest_sha256": manifest_hash,
            "protocol_sha256": v18_protocol_hash,
            "calibration_manifest_sha256": sources["v18_calibration_manifest"]["sha256"],
            "independent_fresh_process_and_activation_root": True,
            "pilot_reference": str(V18 / "checkpoints/M2/layer_00.pt") if pilot_only else None,
        }
        write_json(run_dir / "V40_RUN_CONFIG.json", config)
        patch_runner(run_dir)
        init_manifest = {
            "run": f"v40_{dirname.lower()}", "status": "FROZEN_NOT_STARTED", "arm": arm,
            "candidate_manifest_sha256": manifest_hash, "protocol_sha256": v18_protocol_hash,
            "resume": {"layer_checkpoint_stride": 1, "atomic_layer_checkpoints": True,
                       "candidate_selection_repeated_for_completed_layers": False},
            "held": "NOT_RUN", "ppl": "NOT_RUN",
        }
        write_json(run_dir / "FORMAL_RUN_MANIFEST.json", init_manifest)

    # Job table has one row per module and run; parity and pilot overhead are separate rows.
    job_rows = []
    for run_name in ("PILOT", "A", "B"):
        for full_name, item in sorted(modules.items()):
            if run_name == "PILOT" and item["layer"] != 0:
                continue
            count = len(full_grid) if run_name == "A" else len(item["candidate_alphas"])
            if run_name == "PILOT":
                count = len(item["candidate_alphas"])
            job_rows.append({
                "phase": run_name, "module": full_name, "layer": item["layer"], "shape_out_in": "x".join(map(str, item["shape"])),
                "output_rows": item["rows"], "candidate_alphas": ";".join(f"{a:.1f}" for a in (full_grid if run_name == "A" else item["candidate_alphas"])),
                "candidate_gptq_executions": count, "row_alpha_score_slots": count * item["rows"],
                "fresh_pristine_weight_and_candidate_state": True, "fresh_group_input_H_Hinv": True,
                "activation_cache_shared_with_other_run": False,
                "cache_reuse": "none across runs; common immutable token manifest only",
            })
    for run_name, parity in (("PILOT", 7), ("A", 35), ("B", 35)):
        job_rows.append({"phase": run_name, "module": "AUX_ALPHA1_NATIVE_PARITY", "layer": "0,6,12,18,23",
                         "shape_out_in": "per full Linear", "output_rows": 0, "candidate_alphas": "1.0",
                         "candidate_gptq_executions": 0, "row_alpha_score_slots": 0,
                         "fresh_pristine_weight_and_candidate_state": True, "fresh_group_input_H_Hinv": True,
                         "activation_cache_shared_with_other_run": False,
                         "cache_reuse": "extra native parity GPTQ calls counted separately", "parity_gptq_executions": parity})
    with (OUT / "JOB_TABLE.csv").open("w", newline="", encoding="utf-8") as f:
        fields = list(dict.fromkeys(key for row in job_rows for key in row))
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="raise")
        writer.writeheader(); writer.writerows(job_rows)

    baseline_timing = list(csv.DictReader((V18 / "FORMAL_TIMING.csv").open(encoding="utf-8-sig", newline="")))
    per_module_stage = {}
    for stage in ("full_linear_alpha_candidate_gptq", "all_row_direct_output_scoring"):
        by_module: dict[str, list[float]] = {}
        for r in baseline_timing:
            if r["method"] == "M2" and r["stage"] == stage:
                by_module.setdefault(r["module"], []).append(float(r["wall_seconds"]))
        per_module_stage[stage] = by_module
    projected_b = 0.0
    for name, item in modules.items():
        for stage in per_module_stage:
            times = per_module_stage[stage].get(name, [])
            if len(times) != 10:
                raise RuntimeError(f"expected 10 timed alpha entries for {stage} {name}, got {len(times)}")
            projected_b += sum(times) / len(times) * len(item["candidate_alphas"])
    v18_wall = json.loads((V18 / "FORMAL_RUNTIME.json").read_text(encoding="utf-8"))["m2"]["wall_seconds_including_load_replay_save"]
    budget = {
        "status": "FROZEN_NOT_STARTED", "hard_cap_seconds": 5400.0,
        "execution_cutoff_seconds": 4700.0, "wrap_up_reserve_seconds": 700.0,
        "gpu_experiment_elapsed_seconds": 0.0, "completed_phases": [], "failed_attempts": [],
        "phase_ledger": [], "started_utc": None, "cutoff_utc": None,
        "reference": {"v18_m2_wall_seconds": v18_wall,
                      "v18_m2_candidate_gptq_seconds": sum(float(r["wall_seconds"]) for r in baseline_timing if r["method"] == "M2" and r["stage"] == "full_linear_alpha_candidate_gptq"),
                      "v18_m2_direct_scoring_seconds": sum(float(r["wall_seconds"]) for r in baseline_timing if r["method"] == "M2" and r["stage"] == "all_row_direct_output_scoring"),
                      "projected_B_candidate_and_score_seconds_from_per_module_v18_timing": projected_b,
                      "projected_B_total_seconds_with_v18_non_candidate_residual": projected_b + max(0.0, v18_wall - sum(float(r["wall_seconds"]) for r in baseline_timing if r["method"] == "M2" and r["stage"] in per_module_stage)),
                      "projection_is_estimate_not_benchmark": True},
    }
    write_json(OUT / "BUDGET.json", budget)
    environment = {
        "captured_utc": datetime.now(timezone.utc).isoformat(),
        "python_executable": sys.executable, "python": sys.version, "platform": platform.platform(),
        "torch": torch.__version__, "torch_cuda_build": torch.version.cuda,
        "cuda_device_query": "Deferred to pilot; no CUDA context initialized by build_v40.py",
        "v18_environment_reference": json.loads((V18 / "FORMAL_RUNTIME.json").read_text(encoding="utf-8")),
    }
    write_json(OUT / "ENVIRONMENT.json", environment)
    lock_files = [OUT / "FROZEN_PROTOCOL.json", OUT / "PROTOCOL.json", OUT / "CANDIDATE_MANIFEST.json", OUT / "JOB_TABLE.csv"]
    lock = {"status": "FROZEN", "created_utc": datetime.now(timezone.utc).isoformat(),
            "sha256": {p.name: sha_file(p) for p in lock_files},
            "candidate_manifest_sha256": manifest_hash,
            "source_hashes": {name: info["sha256"] for name, info in sources.items()},
            "counts": {"linears": 168, "full_grid_candidate_gptq_executions": 1680,
                       "winner_union_candidate_gptq_executions": 675, "winner_union_row_alpha_score_slots": score_slots,
                       "pilot_block0_candidate_executions": 27, "pilot_alpha1_parity_executions": 7,
                       "full_run_alpha1_parity_executions_per_arm": 35},
            "p3_read": False, "held_read": False, "ppl_run": False, "gpu_run": False}
    write_json(OUT / "FREEZE_LOCK.json", lock)

    # Freeze an executable source copy per run; unchanged protocol files remain byte-identical to v18.
    for run_name in ("PILOT", "A", "B"):
        run_dir = OUT / run_name
        if sha_file(run_dir / "PROTOCOL.json") != v18_protocol_hash:
            raise RuntimeError(f"copied v18 protocol changed in {run_name}")
    print(json.dumps({"status": "FROZEN", "candidate_calls": candidate_executions,
                      "row_alpha_score_slots": score_slots, "manifest_sha256": manifest_hash,
                      "projected_B_total_seconds": budget["reference"]["projected_B_total_seconds_with_v18_non_candidate_residual"]}, indent=2))


if __name__ == "__main__":
    main()
