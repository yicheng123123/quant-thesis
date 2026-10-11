"""CPU-only provenance and frozen-design audit for v31; no scoring or GPTQ."""
from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import torch
from safetensors import safe_open


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
RESULTS = ROOT / "experiments/range_search_w4/results"
V18 = RESULTS / "range_w4_fullmodel_v18"
V23 = RESULTS / "range_w4_block_objective_e1_v23"
V24 = RESULTS / "range_w4_structured_group_e2_v24"
V25 = RESULTS / "range_w4_dense_scalar_audit_v25"
V30 = RESULTS / "range_w4_weight_error_regularization_v30"
MODEL = ROOT / ".hf_cache/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775/model.safetensors"
TARGETS = [(0, "up_proj"), (0, "down_proj"), (12, "up_proj"), (12, "down_proj"), (23, "up_proj"), (23, "down_proj")]
ROWS = {"up_proj": [0, 224, 448, 672], "down_proj": [0, 224, 448, 672]}
GROUPS = {"A": list(range(0, 16)), "B": list(range(16, 32)), "C": list(range(32, 48)), "D": list(range(48, 64))}
SPLIT_TO_GROUPS = {"S0": {"search": "A", "held": "B"}, "S1": {"search": "C", "held": "D"}}


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha_tensor(t: torch.Tensor) -> str:
    return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def load_json(path: Path):
    return json.loads(path.read_text("utf-8"))


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", "utf-8")
    tmp.replace(path)


def main() -> None:
    checks: dict[str, bool] = {}
    v24_protocol = load_json(V24 / "PROTOCOL.json")
    v24_docs = load_json(V24 / "E2_DOCUMENT_MANIFEST.json")
    v24_down_manifest = load_json(V24 / "cache/DOWN_INPUT_CACHE_MANIFEST.json")
    v23_prefix_manifest = load_json(V23 / "cache/PREFIX_CACHE_MANIFEST.json")
    v18_cal = load_json(V18 / "CALIBRATION_MANIFEST.json")
    v25_protocol = load_json(V25 / "PROTOCOL.json")
    v25_provenance = load_json(V25 / "INPUT_PROVENANCE.json")
    v25_report = load_json(V25 / "RUN_REPORT.json")
    v30_summary = load_json(V30 / "RUN_SUMMARY.json")
    v30_checks = load_json(V30 / "AUDIT_CHECKS.json")
    v30_sources = list(csv.DictReader((V30 / "SOURCE_MANIFEST.csv").open(newline="", encoding="utf-8")))

    alpha_grid = [float(a) for a in v24_protocol["methods"]["M2_original_scalar"]["alpha_grid"]]
    samples = v24_docs["samples"]
    article_shas = [s["article_sha256"] for s in samples]
    v18_article_shas = {s["article_sha256"] for s in v18_cal["samples"]}
    v24_article_shas = set(article_shas)
    prefix_overlap = sorted(v18_article_shas & v24_article_shas)

    group_rows = []
    for group_name, positions in GROUPS.items():
        for pos in positions:
            sample = samples[pos]
            group_rows.append({
                "group": group_name, "manifest_position": pos,
                "source_article_index": sample["source_article_index"],
                "article_sha256": sample["article_sha256"],
                "token_sha256": sample["token_sha256"],
                "split_label_in_v24_manifest": sample.get("split"),
                "overlaps_v18_prefix_calibration_article": sample["article_sha256"] in v18_article_shas,
            })

    checks["64_unique_v24_complete_articles"] = len(samples) == 64 and len(v24_article_shas) == 64
    checks["v18_prefix_calibration_disjoint_from_v24_articles"] = len(prefix_overlap) == 0
    checks["frozen_groups_cover_each_v24_position_once"] = sorted(p for xs in GROUPS.values() for p in xs) == list(range(64))
    checks["S0_generation_and_original_search_are_A"] = v24_protocol["documents"]["split_ids"]["S0"]["search_positions"] == [0, 15]
    checks["S0_held_is_B"] = v24_protocol["documents"]["split_ids"]["S0"]["held_positions"] == [16, 31]
    checks["S1_generation_and_original_search_are_C"] = v24_protocol["documents"]["split_ids"]["S1"]["search_positions"] == [32, 47]
    checks["S1_held_is_D"] = v24_protocol["documents"]["split_ids"]["S1"]["held_positions"] == [48, 63]
    checks["v24_protocol_uses_original_scalar10"] = alpha_grid == [i / 10 for i in range(1, 11)]
    checks["v25_reuses_v24_fixed_scalar_bank"] = v25_protocol["grid_source"].endswith("range_w4_structured_group_e2_v24/PROTOCOL.json")
    checks["v25_run_completed"] = v25_report.get("status") in ("COMPLETE", "PASS", "COMPLETE_ARTIFACT_AUDIT")
    checks["v30_is_fixed_v24_candidate_artifact_audit"] = "original v24 scalar10 only" in v30_summary.get("scope", "")
    checks["v30_all_saved_checks_passed"] = v30_checks.get("all_passed") is True and v30_checks.get("passed") == v30_checks.get("total")

    # Current hashes of all input activation caches are checked against the
    # immutable manifests before they are permitted as v31 scoring inputs.
    input_rows = []
    for li, projection in TARGETS:
        if projection == "up_proj":
            path = V23 / "cache" / f"input_L{li}_up_proj.f16"
            entry = v23_prefix_manifest["files"][path.name]
            source_manifest_hash = entry["sha256"]
            expected_shape = [64, 2048, 896]
        else:
            path = V24 / "cache" / f"input_L{li}_down_proj.f16"
            entry = v24_down_manifest["files"][path.name]
            source_manifest_hash = entry["sha256"]
            expected_shape = list(entry["shape"])
        current_hash = sha_file(path)
        expected_bytes = math.prod(expected_shape) * 2
        input_rows.append({
            "layer": li, "projection": projection, "path": str(path.relative_to(ROOT)).replace("\\", "/"),
            "shape": expected_shape, "dtype": "float16", "size_bytes": path.stat().st_size,
            "manifest_sha256": source_manifest_hash, "current_sha256": current_hash,
            "hash_match": current_hash == source_manifest_hash,
            "origin": "v23 frozen v18-M2-prefix capture" if projection == "up_proj" else "v24 replay of v18-M2 block at the frozen target dependency boundary",
        })
    checks["all_six_activation_cache_hashes_and_shapes_match"] = all(
        x["hash_match"] and x["size_bytes"] == math.prod(x["shape"]) * 2 for x in input_rows
    )

    # Cross-check each saved Hessian and every candidate's recorded source
    # positions; load tensor sidecars only to verify the hashes already saved.
    candidate_dir = V24 / "raw/full_candidates"
    held_dir = V24 / "raw/held_full_scores"
    candidate_files = sorted(candidate_dir.glob("*.json"))
    expected_candidate_count = 6 * 2 * 46
    bank_audit = []
    hash_manifest_rows = []
    observed: dict[tuple[int, str, str], list[dict]] = defaultdict(list)
    for meta_path in candidate_files:
        meta = load_json(meta_path)
        li = int(meta["layer"])
        projection = meta["projection"]
        split = meta["split"]
        expected_search_group = SPLIT_TO_GROUPS[split]["search"]
        expected_doc_indices = GROUPS[expected_search_group]
        h_path = V24 / "cache/hessians" / f"L{li}_{projection}_{split}.json"
        h_meta = load_json(h_path)
        pt_path = meta_path.with_suffix(".pt")
        held_path = held_dir / meta_path.name
        held_meta = load_json(held_path)
        pt = torch.load(pt_path, map_location="cpu", weights_only=True)
        tensor_hashes = {key: sha_tensor(pt[key]) for key in ("weight_fp16", "scales", "zeros", "g_idx")}
        alpha = float(meta["alpha"])
        alpha_is_scalar10 = any(math.isclose(alpha, a, rel_tol=0.0, abs_tol=1e-12) for a in alpha_grid)
        good = (
            meta.get("candidate_status") == "COMPLETE"
            and meta.get("search_doc_indices") == expected_doc_indices
            and h_meta.get("doc_indices") == expected_doc_indices
            and meta.get("hessian_sha256") == h_meta.get("H_sha256")
            and meta.get("e2_manifest_sha256") == sha_file(V24 / "E2_DOCUMENT_MANIFEST.json")
            and tensor_hashes["weight_fp16"] == meta.get("weight_sha256") == held_meta.get("candidate_weight_sha256")
            and tensor_hashes["scales"] == meta.get("scales_sha256")
            and tensor_hashes["zeros"] == meta.get("zeros_sha256")
            and tensor_hashes["g_idx"] == meta.get("g_idx_sha256")
            and held_meta.get("held_doc_indices") == GROUPS[SPLIT_TO_GROUPS[split]["held"]]
        )
        record = {
            "layer": li, "projection": projection, "split": split, "alpha": alpha,
            "rows": meta["rows"], "search_group": expected_search_group,
            "search_doc_indices": meta["search_doc_indices"], "held_group": SPLIT_TO_GROUPS[split]["held"],
            "held_doc_indices": held_meta["held_doc_indices"],
            "hessian_sha256": meta["hessian_sha256"], "pristine_weight_sha256": meta["pristine_weight_sha256"],
            "weight_sha256": tensor_hashes["weight_fp16"], "scales_sha256": tensor_hashes["scales"],
            "zeros_sha256": tensor_hashes["zeros"], "g_idx_sha256": tensor_hashes["g_idx"],
            "search_mse_by_row": meta["search_mse_by_row"],
            "held_mse_by_row": held_meta["held_mse_by_row"],
            "candidate_pt_bytes": pt_path.stat().st_size, "candidate_meta_bytes": meta_path.stat().st_size,
            "held_sidecar_bytes": held_path.stat().st_size,
            "candidate_tensor_hashes_match_sidecars": good,
            "is_original_scalar10_candidate": alpha_is_scalar10,
        }
        bank_audit.append(record)
        hash_manifest_rows.append({k: record[k] for k in ("layer", "projection", "split", "alpha", "hessian_sha256", "pristine_weight_sha256", "weight_sha256", "scales_sha256", "zeros_sha256", "g_idx_sha256", "candidate_tensor_hashes_match_sidecars")})
        observed[(li, projection, split)].append(record)
    checks["all_552_candidate_pt_json_pairs_exist"] = len(candidate_files) == expected_candidate_count and all((p.with_suffix(".pt")).is_file() for p in candidate_files)
    checks["all_candidate_tensor_and_held_sidecar_hashes_match"] = len(bank_audit) == expected_candidate_count and all(r["candidate_tensor_hashes_match_sidecars"] for r in bank_audit)
    checks["all_candidate_banks_have_46_unique_alphas"] = all(
        len(v) == 46 and len({r["alpha"] for r in v}) == 46 for v in observed.values()
    ) and len(observed) == 12
    checks["all_candidate_Hessians_match_exact_article_groups"] = all(
        r["search_doc_indices"] == GROUPS[r["search_group"]]
        and r["held_doc_indices"] == GROUPS[r["held_group"]]
        for r in bank_audit
    )
    scalar_bank_audit = [r for r in bank_audit if r["is_original_scalar10_candidate"]]
    checks["both_10_alpha_candidate_banks_per_target_exist"] = len(scalar_bank_audit) == 6 * 2 * 10

    # The base W used to create GPTQ candidates is distinct from the fixed M2
    # prefix checkpoint used to form X. Check the FP16 target tensor hashes.
    pristine_by_target = {}
    with safe_open(str(MODEL), framework="pt", device="cpu") as f:
        for li, projection in TARGETS:
            key = f"model.layers.{li}.mlp.{projection}.weight"
            w = f.get_tensor(key).to(torch.float16).contiguous()
            pristine_by_target[(li, projection)] = sha_tensor(w)
    checks["candidate_pristine_weight_hashes_match_base_model_fp16"] = all(
        all(r["pristine_weight_sha256"] == pristine_by_target[(r["layer"], r["projection"])] for r in group)
        for group in observed.values()
    )

    # Existing raw scores must faithfully represent the frozen scalar10 path.
    search_rows = list(csv.DictReader((V24 / "SEARCH_RESULTS.csv").open(newline="", encoding="utf-8")))
    held_rows = list(csv.DictReader((V24 / "HELD_RESULTS.csv").open(newline="", encoding="utf-8")))
    search_lookup = {(int(r["layer"]), r["projection"], r["split_id"], int(r["row"]), float(r["alpha"])): r for r in search_rows
                     if r["candidate_method"] == "scalar_dense_bank"}
    held_selected_lookup = {(int(r["layer"]), r["projection"], r["split_id"], int(r["row"])): r for r in held_rows
                            if r["method"] == "M2_original_scalar10"}
    v30_candidate_rows = list(csv.DictReader((V30 / "CANDIDATE_ROW_TABLE.csv").open(newline="", encoding="utf-8")))
    v30_lookup = {(int(r["layer"]), r["projection"], r["split_id"], int(r["row"]), float(r["alpha"])): r for r in v30_candidate_rows}
    score_map_by_unit = defaultdict(list)
    for key, sr in search_lookup.items():
        li, projection, split, row, alpha = key
        if any(math.isclose(alpha, a, rel_tol=0.0, abs_tol=1e-12) for a in alpha_grid):
            score_map_by_unit[(li, projection, split, row)].append((float(sr["search_mse"]), alpha, sr))
    frozen_choice_by_unit = {
        unit: sorted(values, key=lambda x: (x[0], -x[1]))[0][1]
        for unit, values in score_map_by_unit.items()
    }
    score_crosschecks = []
    selected_original_checks = []
    for r in scalar_bank_audit:
        li, projection, split, alpha = r["layer"], r["projection"], r["split"], r["alpha"]
        for row in ROWS[projection]:
            sr = search_lookup.get((li, projection, split, row, alpha))
            v30r = v30_lookup.get((li, projection, split, row, alpha))
            unit = (li, projection, split, row)
            selected_held = held_selected_lookup.get(unit)
            saved_search = float(r["search_mse_by_row"][str(row)])
            saved_held = float(r["held_mse_by_row"][str(row)])
            is_selected_row = sr is not None and sr["selected_by_original_10"].lower() == "true"
            selected_alpha = frozen_choice_by_unit.get(unit)
            if is_selected_row:
                selected_original_checks.append({
                    "layer": li, "projection": projection, "split": split, "row": row,
                    "selected_alpha_from_v24_search": selected_alpha,
                    "selected_alpha_flagged_in_search_csv": alpha,
                    "held_csv_alpha": None if selected_held is None else float(selected_held["alpha_A"]),
                    "sidecar_held_mse": saved_held,
                    "held_csv_mse": None if selected_held is None else float(selected_held["held_local_mse"]),
                    "selected_alpha_reproduced": selected_alpha == alpha and selected_alpha == float(v30r["original_search_selected_alpha"]),
                    "selected_held_result_reproduced": selected_held is not None and float(selected_held["alpha_A"]) == alpha and float(selected_held["held_local_mse"]) == saved_held,
                })
            score_crosschecks.append({
                "layer": li, "projection": projection, "split": split, "row": row, "alpha": alpha,
                "sidecar_search_mse": saved_search,
                "search_csv_mse": None if sr is None else float(sr["search_mse"]),
                "sidecar_held_mse": saved_held,
                "v30_candidate_row_held_mse": None if v30r is None else float(v30r["held_output_mse"]),
                "is_original_v24_selected_alpha": is_selected_row,
                "selected_held_csv_mse": None if not is_selected_row or selected_held is None else float(selected_held["held_local_mse"]),
                "search_exact_match": sr is not None and saved_search == float(sr["search_mse"]),
                "held_exact_match_v30": v30r is not None and saved_held == float(v30r["held_output_mse"]),
            })
    checks["saved_original10_search_scores_reproduce_search_csv"] = all(r["search_exact_match"] for r in score_crosschecks)
    checks["saved_original10_held_scores_reproduce_v30_candidate_table"] = all(r["held_exact_match_v30"] for r in score_crosschecks)
    checks["all_48_original_v24_alpha_choices_reproduce"] = len(selected_original_checks) == 48 and all(r["selected_alpha_reproduced"] for r in selected_original_checks)
    checks["all_48_original_v24_selected_held_rows_reproduce"] = len(selected_original_checks) == 48 and all(r["selected_held_result_reproduced"] for r in selected_original_checks)

    source_paths = {
        "v18_protocol": V18 / "PROTOCOL.json",
        "v18_calibration_manifest": V18 / "CALIBRATION_MANIFEST.json",
        "v18_m2_prefix_weights": V18 / "M2/checkpoint/final/model.safetensors",
        "v23_protocol": V23 / "PROTOCOL.json",
        "v23_prefix_cache_manifest": V23 / "cache/PREFIX_CACHE_MANIFEST.json",
        "v24_protocol": V24 / "PROTOCOL.json",
        "v24_document_manifest": V24 / "E2_DOCUMENT_MANIFEST.json",
        "v24_down_input_manifest": V24 / "cache/DOWN_INPUT_CACHE_MANIFEST.json",
        "v24_search_selection_lock": V24 / "SEARCH_SELECTION_LOCK.json",
        "v24_search_results": V24 / "SEARCH_RESULTS.csv",
        "v24_held_results": V24 / "HELD_RESULTS.csv",
        "v24_held_repeat_noise": V24 / "HELD_REPEAT_NOISE.json",
        "v25_protocol": V25 / "PROTOCOL.json",
        "v25_input_provenance": V25 / "INPUT_PROVENANCE.json",
        "v25_run_report": V25 / "RUN_REPORT.json",
        "v30_readme": V30 / "README.md",
        "v30_run_summary": V30 / "RUN_SUMMARY.json",
        "v30_audit_checks": V30 / "AUDIT_CHECKS.json",
        "v30_source_manifest": V30 / "SOURCE_MANIFEST.csv",
        "v30_candidate_row_table": V30 / "CANDIDATE_ROW_TABLE.csv",
    }
    source_hashes = {k: sha_file(p) for k, p in source_paths.items()}
    source_hashes["base_model_weights"] = sha_file(MODEL)
    source_hashes["activation_inputs"] = {x["path"]: x["current_sha256"] for x in input_rows}
    source_hashes["candidate_state_sha256_manifest"] = hashlib.sha256(
        "".join(json.dumps(r, sort_keys=True, separators=(",", ":")) + "\n" for r in hash_manifest_rows).encode()
    ).hexdigest()

    checks["all_required_source_files_present"] = all(p.is_file() for p in source_paths.values())
    checks["all_source_artifact_files_hashed"] = len(source_hashes) == len(source_paths) + 3
    checks_passed = sum(bool(v) for v in checks.values())
    status = "PASS" if all(checks.values()) else "FAIL"

    with (HERE / "ARTICLE_SPLITS.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(group_rows[0]))
        writer.writeheader(); writer.writerows(group_rows)
    with (HERE / "CANDIDATE_BANK_AUDIT.csv").open("w", newline="", encoding="utf-8") as f:
        fields = ["layer", "projection", "split", "alpha", "rows", "search_group", "search_doc_indices", "held_group", "held_doc_indices",
                  "hessian_sha256", "pristine_weight_sha256", "weight_sha256", "scales_sha256", "zeros_sha256", "g_idx_sha256",
                  "candidate_pt_bytes", "candidate_meta_bytes", "held_sidecar_bytes", "candidate_tensor_hashes_match_sidecars", "is_original_scalar10_candidate"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in bank_audit:
            row = {k: r[k] for k in fields}
            row.update({"rows": json.dumps(r["rows"]), "search_doc_indices": json.dumps(r["search_doc_indices"]),
                        "held_doc_indices": json.dumps(r["held_doc_indices"])})
            writer.writerow(row)
    with (HERE / "SAVED_SCORE_CROSSCHECKS.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(score_crosschecks[0]))
        writer.writeheader(); writer.writerows(score_crosschecks)
    with (HERE / "ORIGINAL_SELECTION_REPRODUCTION.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(selected_original_checks[0]))
        writer.writeheader(); writer.writerows(selected_original_checks)
    with (HERE / "ACTIVATION_CACHE_AUDIT.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(input_rows[0]))
        writer.writeheader(); writer.writerows(input_rows)
    write_json(HERE / "SOURCE_HASHES.json", source_hashes)

    audit = {
        "status": status,
        "checks_passed": checks_passed, "checks_total": len(checks), "checks": checks,
        "candidate_files": len(candidate_files), "candidate_bank_records": len(bank_audit),
        "candidate_original10_records": len(scalar_bank_audit),
        "v24_candidate_hessian_lineage": {
            "S0": {"candidate_generation_articles": GROUPS["A"], "exact_saved_hessian_doc_indices": GROUPS["A"], "original_selection_group": "A", "independent_selection_group": "B", "evaluation_groups": ["C", "D"]},
            "S1": {"candidate_generation_articles": GROUPS["C"], "exact_saved_hessian_doc_indices": GROUPS["C"], "original_selection_group": "C", "independent_selection_group": "D", "evaluation_groups": ["A", "B"]},
        },
        "v18_prefix_calibration_article_overlap_count": len(prefix_overlap),
        "v18_prefix_calibration_overlap_article_sha256": prefix_overlap,
        "v18_prefix_checkpoint_sha256": v24_down_manifest["v18_m2_checkpoint_sha256"],
        "v18_calibration_documents": len(v18_cal["samples"]),
        "v24_complete_documents": len(samples),
        "original_selected_choices_checked": len(selected_original_checks),
        "original10_row_score_comparisons": len(score_crosschecks),
        "new_gpu_scoring_needed": True,
        "new_gpu_scoring_plan": "240 full-candidate score calls; each scores all 4 fixed rows for one original alpha on one missing 16-document group; no GPTQ or model forward",
        "historical_and_current_data_are_seen": True,
        "interpretation": "Exploratory retrospective generation-selection separation diagnostic; not a blind or formal generalization test.",
    }
    write_json(HERE / "DATA_RELATION_AUDIT.json", audit)

    if status != "PASS":
        print(json.dumps({"status": "AUDIT_FAIL_STOP_BEFORE_SCORING", "failed_checks": [k for k, v in checks.items() if not v]}, indent=2))
        return

    v24_tie_eps = load_json(V24 / "HELD_REPEAT_NOISE.json")["repeat_noise_mse_by_unit"]
    protocol = {
        "name": "v31 Fixed GPTQ Candidate Generation-Selection Data Separation Diagnostic",
        "status": "FROZEN_BEFORE_CROSS_GROUP_SCORING",
        "interpretation": "Exploratory retrospective diagnosis; all 64 articles and prior results were already inspected. Not blind, not formal generalization evidence.",
        "inheritance": {
            "v24_model": v24_protocol["model"], "v24_quantization": v24_protocol["quantization"],
            "v24_targets": v24_protocol["targets"], "v24_scalar_grid": alpha_grid,
            "v24_fixed_rows": ROWS, "v24_candidate_generator": "reuse all saved v24 scalar10 candidate tensors; no candidate GPTQ reruns",
            "prefix_checkpoint": "v18 frozen M2 dense checkpoint; same as used to capture v23 up inputs and v24 down inputs",
            "target_weight_source": "pristine base-model FP16 target W from the v24 candidate generation source, independently hash-matched",
        },
        "documents": {
            "manifest": "v24 E2_DOCUMENT_MANIFEST.json; 64 unique complete WikiText-2 train articles, 2048 tokens each",
            "groups": GROUPS,
            "candidate_generation_and_selection": {
                "S0_bank": {"candidate_H_articles": "A=0..15", "original_selector": "A=0..15", "independent_selector": "B=16..31", "evaluation": ["C=32..47", "D=48..63"]},
                "S1_bank": {"candidate_H_articles": "C=32..47", "original_selector": "C=32..47", "independent_selector": "D=48..63", "evaluation": ["A=0..15", "B=16..31"]},
            },
            "prefix_relationship": {
                "fixed_prefix": "v18 M2 quantized checkpoint, calibrated on its frozen 128 WikiText-2 train articles",
                "article_overlap_with_v24_64": len(prefix_overlap),
                "article_hash_intersection": prefix_overlap,
                "limit": "This only establishes article-level disjointness of v18 prefix calibration versus the 64 v24 articles; prior inspection makes all current v31 comparisons exploratory.",
            },
        },
        "scoring": {
            "method": "original v24 direct output scorer only; no v27/v28 quadratic scoring",
            "formula": "row_mse = sum over documents/tokens of (X @ (Q_FP16.float()-W_FP16.float())^T)^2 / (16*2048)",
            "implementation": "same v24 score_selected_rows GEMM over all 4 fixed rows; 2 complete documents per token chunk; FP32 inputs/delta/GEMM and FP64 squared-sum accumulation; division by 32768 tokens",
            "selection": "row-wise argmin over original scalar10 candidate losses; exact score ties select larger alpha, exactly as v24",
            "win_tie_loss": "reuse the frozen v24 per-unit repeated-alpha1 epsilon; no threshold adjustment",
            "oracle": "minimum among the same ten candidate scores on each evaluation group; post-hoc diagnostic only",
        },
        "data_reuse": {
            "saved_scores_reused": "S0 bank A search and B held; S1 bank C search and D held",
            "new_scoring_only": "S0 bank on C,D; S1 bank on A,B",
            "new_gpu_score_calls": 240,
            "new_row_alpha_scores": 960,
            "candidate_executions": 0,
            "new_model_forwards": 0,
            "candidate_cache_read_only": True,
        },
        "budget": {"new_gpu_wall_seconds_max": 600, "includes": ["CUDA init", "warmup", "candidate loads/transfers", "all new score GEMMs", "checkpoint writes", "retries/resume"]},
        "analysis": {
            "comparison": "independent-selector alpha vs original-selector alpha, paired by same candidate bank/target/row/evaluation article group",
            "pooled_metric": "sum row MSE * 32768 tokens; report each evaluation group, each target, each bank, and pooled bank/group descriptive totals",
            "alpha1_reference": "evaluate the saved alpha=1 candidate on the same group",
            "oracle_regret": "report selector SSE minus per-row finite-bank evaluation oracle SSE; never used for selection",
            "severity": "report row win/tie/loss using v24 frozen epsilon, median paired relative change, worst paired relative regression, gain concentration, and < -10% vs alpha1 count",
            "pairing": "the same 24 target-row identities recur in two evaluation groups per bank; group observations are paired and not treated as independent samples",
        },
        "prohibitions": ["new GPTQ candidate generation", "PPL", "whole-model quantization", "new held blind-test claim", "AWQ", "gradient optimization", "regularization", "grid expansion", "changing groups based on results"],
        "source_sha256": source_hashes,
    }
    write_json(HERE / "FROZEN_PROTOCOL.json", protocol)
    with (HERE / "CANDIDATE_BANK_AUDIT.csv").open("a", encoding="utf-8", newline="") as f:
        pass
    print(json.dumps({
        "status": "AUDIT_PASS_PROTOCOL_FROZEN",
        "candidate_banks": 12, "candidate_states": expected_candidate_count,
        "original10_states": len(scalar_bank_audit), "v18_v24_article_overlap": len(prefix_overlap),
        "new_score_calls": 240, "new_row_alpha_scores": 960,
        "saved_score_exact_search_rows": sum(r["search_exact_match"] for r in score_crosschecks),
        "saved_score_exact_held_rows": sum(r["held_exact_match_v30"] for r in score_crosschecks),
        "saved_score_rows": len(score_crosschecks),
        "checks_passed": checks_passed, "checks_total": len(checks),
    }, indent=2))


if __name__ == "__main__":
    main()
