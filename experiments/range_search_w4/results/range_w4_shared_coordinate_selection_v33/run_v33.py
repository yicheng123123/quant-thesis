"""CPU-only v33 shared-coordinate selection and post-lock held analysis."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import torch


HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent
V32 = RESULTS / "range_w4_fixed_channel_scale_scalar_v32"
V24 = RESULTS / "range_w4_structured_group_e2_v24"
PROTO = json.loads((HERE / "FROZEN_PROTOCOL.json").read_text("utf-8"))
P32 = json.loads((V32 / "FROZEN_PROTOCOL.json").read_text("utf-8"))
ALPHAS = [float(x) for x in PROTO["inherits"]["alpha_grid"]]
ROWS = {k: [int(v) for v in vals] for k, vals in PROTO["inherits"]["targets"]["rows"].items()}
N_TOKENS = 16 * 2048
V32_RAW = V32 / "raw"
V24_FULL = V24 / "raw/full_candidates"
SPLITS = ("S0", "S1")
LAYERS = (0, 12, 23)
PROJECTIONS = ("up_proj", "down_proj")
_ORIGINAL_CANDIDATE_MAP: dict[str, dict[float, tuple[Path, Path]]] = {}


def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def sha_tensor(t: torch.Tensor) -> str:
    return sha_bytes(t.detach().cpu().contiguous().numpy().tobytes())


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise RuntimeError(f"refusing empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def dump(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n", "utf-8")


def add_budget(stage: str, seconds: float) -> None:
    p = HERE / "BUDGET.json"
    b = json.loads(p.read_text("utf-8"))
    b["stages"][stage] = seconds
    b["cumulative_wall_seconds"] = math.fsum(b["stages"].values())
    if b["cumulative_wall_seconds"] > b["budget_seconds"]:
        dump(p, b)
        raise RuntimeError("v33 CPU wall budget exceeded")
    dump(p, b)


def verify_protocol_sources() -> None:
    for name, expected in PROTO["source_files_sha256"].items():
        paths = {
            "v32_protocol": V32 / "FROZEN_PROTOCOL.json",
            "v32_correctness": V32 / "CORRECTNESS.json",
            "v32_source_audit": V32 / "SOURCE_AUDIT.json",
            "v32_scaled_search_scores": V32 / "SCALED_CANDIDATE_SEARCH_SCORES.csv",
            "v24_protocol": V24 / "PROTOCOL.json",
            "v24_search_scores": V24 / "SEARCH_RESULTS.csv",
            "v24_search_lock": V24 / "SEARCH_SELECTION_LOCK.json",
        }
        if sha_file(paths[name]) != expected:
            raise RuntimeError(f"frozen source hash changed: {name}")


def stem(layer: int, projection: str, split: str) -> str:
    return f"L{layer}_{projection}_{split}"


def unit_id(layer: int, projection: str, row: int, split: str) -> str:
    return f"L{layer}_{projection}_R{row}_{split}"


def select_alpha(scores: dict[float, float]) -> float:
    best = min(scores.values())
    return max(a for a, score in scores.items() if score == best)


def read_search_banks() -> tuple[dict, dict]:
    # v24 SEARCH_RESULTS is search-only. Only its original scalar10 subset is used.
    original_rows = [r for r in read_csv(V24 / "SEARCH_RESULTS.csv")
                     if r["candidate_method"] == "scalar_dense_bank" and r["is_original_10"] == "True"]
    original = defaultdict(dict)
    for r in original_rows:
        key = (r["split_id"], int(r["layer"]), r["projection"], int(r["row"]))
        a = float(r["alpha"])
        if a in original[key]:
            raise RuntimeError(f"duplicate original search score: {key}/{a}")
        original[key][a] = float(r["search_mse"])
    if len(original_rows) != 480:
        raise RuntimeError(f"expected 480 original scalar10 search scores, got {len(original_rows)}")

    # The v32 table has only search scores; no held table is opened in this phase.
    scaled_rows = read_csv(V32 / "SCALED_CANDIDATE_SEARCH_SCORES.csv")
    scaled = defaultdict(dict)
    scaled_meta = {}
    for r in scaled_rows:
        key = (r["split"], int(r["layer"]), r["projection"], int(r["row"]))
        a = float(r["alpha"])
        if a in scaled[key]:
            raise RuntimeError(f"duplicate scaled search score: {key}/{a}")
        scaled[key][a] = float(r["search_mse"])
        scaled_meta[(key, a)] = r
    if len(scaled_rows) != 480:
        raise RuntimeError(f"expected 480 fixed-D search scores, got {len(scaled_rows)}")

    expected = set(ALPHAS)
    for key in original:
        if set(original[key]) != expected or set(scaled[key]) != expected:
            raise RuntimeError(f"alpha grid mismatch in search data: {key}")
    if set(original) != set(scaled) or len(original) != 48:
        raise RuntimeError("original and scaled search row units differ")
    return original, (scaled, scaled_meta)


def original_candidate(layer: int, projection: str, split: str, alpha: float) -> tuple[Path, Path]:
    st = stem(layer, projection, split)
    if st not in _ORIGINAL_CANDIDATE_MAP:
        bank = {}
        for jp in sorted(V24_FULL.glob(f"{st}_a*.json")):
            meta = json.loads(jp.read_text("utf-8"))
            a = float(meta["alpha"])
            if a in ALPHAS:
                if a in bank:
                    raise RuntimeError(f"duplicate original scalar candidate alpha: {st}/{a}")
                bank[a] = (jp.with_suffix(".pt"), jp)
        if set(bank) != set(ALPHAS):
            raise RuntimeError(f"original candidate bank lacks scalar10 points: {st}")
        _ORIGINAL_CANDIDATE_MAP[st] = bank
    return _ORIGINAL_CANDIDATE_MAP[st][alpha]


def scaled_candidate(layer: int, projection: str, split: str, alpha: float) -> tuple[Path, Path]:
    base = V32_RAW / "candidates" / f"{stem(layer, projection, split)}_a{alpha:.1f}"
    return Path(str(base) + ".pt"), Path(str(base) + ".json")


def load_and_verify_candidate(coord: str, layer: int, projection: str, split: str,
                               row: int, alpha: float, expected_search: float) -> tuple[dict, dict, dict]:
    pt, jp = (original_candidate(layer, projection, split, alpha) if coord == "identity" else
              scaled_candidate(layer, projection, split, alpha))
    if not pt.is_file() or not jp.is_file():
        raise RuntimeError(f"candidate missing: {pt}")
    meta = json.loads(jp.read_text("utf-8"))
    obj = torch.load(pt, map_location="cpu", weights_only=True)
    idx = ROWS[projection].index(row)
    expected_alpha = float(f"{alpha:.1f}")
    if float(meta["alpha"]) != expected_alpha or [int(x) for x in meta["rows"]] != ROWS[projection]:
        raise RuntimeError(f"candidate alpha/row metadata mismatch: {pt}")
    if abs(float(meta["search_mse_by_row"][str(row)]) - expected_search) > 1e-15:
        raise RuntimeError(f"candidate search score differs from selection input: {pt}/{row}")
    for key, hkey in (("weight_fp16", "weight_sha256"), ("scales", "scales_sha256"),
                      ("zeros", "zeros_sha256"), ("g_idx", "g_idx_sha256")):
        if sha_tensor(obj[key]) != meta[hkey]:
            raise RuntimeError(f"candidate tensor hash mismatch: {pt}/{key}")
    if obj["weight_fp16"].dtype != torch.float16 or obj["g_idx"].dtype != torch.int32:
        raise RuntimeError(f"candidate tensor dtype mismatch: {pt}")
    row_tensors = {"weight_fp16": obj["weight_fp16"][idx].contiguous(),
                   "scales": obj["scales"][idx].contiguous(),
                   "zeros": obj["zeros"][idx].contiguous(),
                   "g_idx": obj["g_idx"].contiguous()}
    rec = {
        "coordinate": coord, "alpha": alpha, "candidate_pt": pt.relative_to(RESULTS).as_posix(),
        "candidate_json": jp.relative_to(RESULTS).as_posix(), "candidate_pt_sha256": sha_file(pt),
        "candidate_json_sha256": sha_file(jp), "candidate_weight_sha256": meta["weight_sha256"],
        "candidate_scales_sha256": meta["scales_sha256"], "candidate_zeros_sha256": meta["zeros_sha256"],
        "candidate_g_idx_sha256": meta["g_idx_sha256"], "selected_row_index_in_candidate": idx,
        "selected_weight_row_sha256": sha_tensor(row_tensors["weight_fp16"]),
        "selected_scales_row_sha256": sha_tensor(row_tensors["scales"]),
        "selected_zeros_row_sha256": sha_tensor(row_tensors["zeros"]),
        "selected_g_idx_sha256": sha_tensor(row_tensors["g_idx"]),
    }
    if coord == "fixed":
        if meta["D_sha256"] != json.loads((V32_RAW / "scales" / f"{stem(layer, projection, split)}.json").read_text("utf-8"))["D_sha256"]:
            raise RuntimeError(f"scaled candidate D mismatch: {pt}")
        rec["D_sha256"] = meta["D_sha256"]
        rec["H_sha256"] = meta["H_sha256"]
    else:
        rec["D_sha256"] = sha_bytes(b"v33:D=I:FP32")
        rec["H_sha256"] = meta["hessian_sha256"]
    return obj, meta, {**rec, **{"_row_tensors": row_tensors}}


def freeze_selection() -> None:
    started = time.perf_counter()
    verify_protocol_sources()
    if (HERE / "SEARCH_SELECTION_LOCK.json").exists():
        raise RuntimeError("selection lock already exists; refusing to replace frozen choices")
    original, scaled_pack = read_search_banks()
    scaled, scaled_meta_rows = scaled_pack

    # Recompute and verify both existing scalar10 row selections using only search scores.
    dense_orig = {r["unit_id"]: r for r in read_csv(V24 / "DENSE_SCALAR_CONTROL.csv")}
    v32_locks = {}
    for li in LAYERS:
        for pr in PROJECTIONS:
            for sp in SPLITS:
                st = stem(li, pr, sp)
                v32_locks[st] = json.loads((V32_RAW / f"{st}_SEARCH_LOCK.json").read_text("utf-8"))
    row_choices = []
    unit_choices = []
    candidate_audit = []
    stitches = []
    identity_inclusion_pass = True

    for li in LAYERS:
        for pr in PROJECTIONS:
            row_ids = ROWS[pr]
            for sp in SPLITS:
                st = stem(li, pr, sp)
                by_coord = {"identity": {}, "fixed": {}}
                unit_details = {}
                for coord, bank in (("identity", original), ("fixed", scaled)):
                    for row in row_ids:
                        key = (sp, li, pr, row)
                        choices = bank[key]
                        alpha_star = select_alpha(choices)
                        min_score = choices[alpha_star]
                        by_coord[coord][row] = (alpha_star, min_score)
                        if coord == "identity":
                            dense = dense_orig[unit_id(li, pr, row, sp)]
                            if float(dense["original_10_alpha"]) != alpha_star or abs(float(dense["original_10_search_mse"]) - min_score) > 1e-15:
                                raise RuntimeError(f"G01 search choice/score does not reproduce v24: {st}/{row}")
                        else:
                            lock = v32_locks[st]
                            if float(lock["selected_alpha_by_row"][str(row)]) != alpha_star or abs(float(lock["selected_search_mse_by_row"][str(row)]) - min_score) > 1e-15:
                                raise RuntimeError(f"G11 search choice/score does not reproduce v32: {st}/{row}")
                        # Verify the frozen candidate, scales, zero point and group map from CPU artifacts.
                        obj, meta, audit = load_and_verify_candidate(coord, li, pr, sp, row, alpha_star, min_score)
                        row_tensors = audit.pop("_row_tensors")
                        candidate_audit.append({"stem": st, "split": sp, "layer": li, "projection": pr,
                                                "row": row, "rowwise_min_candidate_coordinate": coord,
                                                "selected_alpha": alpha_star, "selected_search_mse": min_score,
                                                **audit})
                        unit_details.setdefault(coord, {})[row] = (row_tensors, audit)
                        row_choices.append({"stem": st, "split": sp, "layer": li, "projection": pr,
                                            "row": row, "coordinate": coord, "alpha": alpha_star,
                                            "search_mse": min_score, "search_sse": min_score * N_TOKENS,
                                            "candidate_weight_sha256": meta["weight_sha256"],
                                            "candidate_scales_sha256": meta["scales_sha256"],
                                            "candidate_zeros_sha256": meta["zeros_sha256"],
                                            "candidate_g_idx_sha256": meta["g_idx_sha256"],
                                            "D_sha256": audit["D_sha256"], "H_sha256": audit["H_sha256"]})

                score_i = math.fsum(by_coord["identity"][r][1] * N_TOKENS for r in row_ids)
                score_fixed = math.fsum(by_coord["fixed"][r][1] * N_TOKENS for r in row_ids)
                chosen_coord = "identity" if score_i <= score_fixed else "fixed"
                chosen_score = min(score_i, score_fixed)
                identity_inclusion_pass &= chosen_score <= score_i
                selected_alphas = {str(r): by_coord[chosen_coord][r][0] for r in row_ids}
                selected_search_sse = math.fsum(by_coord[chosen_coord][r][1] * N_TOKENS for r in row_ids)
                if selected_search_sse != chosen_score:
                    raise RuntimeError(f"chosen coordinate score mismatch: {st}")
                unit_choices.append({"stem": st, "split": sp, "layer": li, "projection": pr,
                                     "row_count": len(row_ids), "D_identity_search_sse": score_i,
                                     "D_fixed_search_sse": score_fixed, "selected_coordinate": chosen_coord,
                                     "selected_D_sha256": (sha_bytes(b"v33:D=I:FP32") if chosen_coord == "identity" else
                                                            unit_details["fixed"][row_ids[0]][1]["D_sha256"]),
                                     "selected_search_sse": selected_search_sse,
                                     "search_score_le_G01": selected_search_sse <= score_i,
                                     "alpha_by_row_json": json.dumps(selected_alphas, sort_keys=True),
                                     "identity_alpha_by_row_json": json.dumps({str(r): by_coord["identity"][r][0] for r in row_ids}, sort_keys=True),
                                     "fixed_alpha_by_row_json": json.dumps({str(r): by_coord["fixed"][r][0] for r in row_ids}, sort_keys=True)})
                # Create an auditable four-row stitch solely from already saved candidates.
                entries = [unit_details[chosen_coord][r][0] for r in row_ids]
                g_indices = [item["g_idx"] for item in entries]
                if not all(torch.equal(g_indices[0], idx) for idx in g_indices[1:]):
                    raise RuntimeError(f"g_idx differs across chosen row candidates: {st}")
                stitch = {name: torch.stack([item[name] for item in entries]).contiguous()
                          for name in ("weight_fp16", "scales", "zeros")}
                stitch["g_idx"] = g_indices[0].clone()
                out = HERE / "raw/selected_stitches" / f"{st}.pt"
                torch.save(stitch, out)
                stitches.append({"stem": st, "coordinate": chosen_coord, "rows": row_ids,
                                 "selected_alpha_by_row": selected_alphas,
                                 "weight_sha256": sha_tensor(stitch["weight_fp16"]),
                                 "scales_sha256": sha_tensor(stitch["scales"]),
                                 "zeros_sha256": sha_tensor(stitch["zeros"]),
                                 "g_idx_sha256": sha_tensor(stitch["g_idx"]),
                                 "stitch_path": out.relative_to(HERE).as_posix(),
                                 "stitch_file_sha256": sha_file(out),
                                 "packability": ("original coordinates" if chosen_coord == "identity" else
                                                 "scaled coordinates only; not declared packable as original G128")})

    if not identity_inclusion_pass or len(unit_choices) != 12 or len(row_choices) != 96 or len(candidate_audit) != 96:
        raise RuntimeError(f"shared selection invariant/count failed: identity={identity_inclusion_pass}, units={len(unit_choices)}, row_choices={len(row_choices)}, candidate_audits={len(candidate_audit)}")
    write_csv(HERE / "SEARCH_ROW_CHOICES.csv", row_choices)
    write_csv(HERE / "SHARED_D_SELECTION.csv", unit_choices)
    write_csv(HERE / "SELECTED_CANDIDATE_AUDIT.csv", candidate_audit)
    dump(HERE / "SELECTED_STITCHES.json", {"status": "PASS", "scope": "four preselected rows per target x split only",
                                            "stitches": stitches})
    selection_files = ["SEARCH_ROW_CHOICES.csv", "SHARED_D_SELECTION.csv", "SELECTED_CANDIDATE_AUDIT.csv", "SELECTED_STITCHES.json"]
    lock = {
        "status": "LOCKED_BEFORE_HELD_ANALYSIS",
        "locked_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_sha256": sha_file(HERE / "FROZEN_PROTOCOL.json"),
        "search_source_hashes": {name: value for name, value in PROTO["source_files_sha256"].items()
                                  if name in ("v32_scaled_search_scores", "v24_search_scores", "v24_search_lock")},
        "selection_files_sha256": {name: sha_file(HERE / name) for name in selection_files},
        "unit_count": 12,
        "row_coordinate_choices": 96,
        "identity_inclusion_property_pass": True,
        "G01_search_choices_reproduced_from_v24": True,
        "G11_search_choices_reproduced_from_v32": True,
        "selected_candidate_tensor_and_quant_params_verified": True,
        "candidate_choice_audit_rows": len(candidate_audit),
        "stitched_target_split_tensors_verified": len(stitches),
        "identity_exact_ties_preferred": True,
        "held_score_values_used": False,
        "held_score_sources_opened": [],
        "selected_fixed_D_unit_count": sum(r["selected_coordinate"] == "fixed" for r in unit_choices),
        "selected_identity_unit_count": sum(r["selected_coordinate"] == "identity" for r in unit_choices),
        "cpu_elapsed_seconds": time.perf_counter() - started,
    }
    dump(HERE / "SEARCH_SELECTION_LOCK.json", lock)
    dump(HERE / "SEARCH_CORRECTNESS.json", {
        "status": "PASS", "protocol_sha256": lock["protocol_sha256"],
        "target_split_units": len(unit_choices), "per_coordinate_row_alpha_choices": len(row_choices),
        "selected_candidate_tensor_audits": len(candidate_audit),
        "selected_four_row_stitches": len(stitches),
        "G01_search_choices_reproduced_from_v24": True,
        "G11_search_choices_reproduced_from_v32": True,
        "identity_inclusion_property_pass": identity_inclusion_pass,
        "held_score_values_used": False,
        "CUDA_calls": 0, "model_forwards": 0, "GPTQ_executions": 0,
    })
    add_budget("freeze_search_selection_and_candidate_audit", time.perf_counter() - started)
    print(json.dumps({"status": lock["status"], "identity_units": lock["selected_identity_unit_count"],
                      "fixed_units": lock["selected_fixed_D_unit_count"],
                      "identity_inclusion_property_pass": lock["identity_inclusion_property_pass"],
                      "elapsed_seconds": lock["cpu_elapsed_seconds"]}, indent=2))


def held_outcome(base: float, new: float, epsilon: float) -> str:
    delta = base - new
    return "win" if delta > epsilon else "loss" if delta < -epsilon else "tie"


def verify_locked_search_artifacts(lock: dict) -> dict:
    for name, expected in lock["selection_files_sha256"].items():
        if sha_file(HERE / name) != expected:
            raise RuntimeError(f"selection artifact changed after lock: {name}")
    choices = read_csv(HERE / "SEARCH_ROW_CHOICES.csv")
    units = {r["stem"]: r for r in read_csv(HERE / "SHARED_D_SELECTION.csv")}
    audits = read_csv(HERE / "SELECTED_CANDIDATE_AUDIT.csv")
    if len(choices) != 96 or len(units) != 12 or len(audits) != 96:
        raise RuntimeError("locked selection artifact counts do not match frozen design")
    orig_dense = {r["unit_id"]: r for r in read_csv(V24 / "DENSE_SCALAR_CONTROL.csv")}
    orig_ok = True
    fixed_ok = True
    for r in choices:
        uid = unit_id(int(r["layer"]), r["projection"], int(r["row"]), r["split"])
        if r["coordinate"] == "identity":
            baseline = orig_dense[uid]
            orig_ok &= float(r["alpha"]) == float(baseline["original_10_alpha"])
            orig_ok &= abs(float(r["search_mse"]) - float(baseline["original_10_search_mse"])) <= 1e-15
        else:
            st = r["stem"]
            vlock = json.loads((V32_RAW / f"{st}_SEARCH_LOCK.json").read_text("utf-8"))
            fixed_ok &= float(r["alpha"]) == float(vlock["selected_alpha_by_row"][r["row"]])
            fixed_ok &= abs(float(r["search_mse"]) - float(vlock["selected_search_mse_by_row"][r["row"]])) <= 1e-15
    candidate_ok = True
    for r in audits:
        pt = RESULTS / r["candidate_pt"]
        jp = RESULTS / r["candidate_json"]
        candidate_ok &= sha_file(pt) == r["candidate_pt_sha256"]
        candidate_ok &= sha_file(jp) == r["candidate_json_sha256"]
    stitch_info = json.loads((HERE / "SELECTED_STITCHES.json").read_text("utf-8"))
    stitch_ok = stitch_info.get("status") == "PASS" and len(stitch_info.get("stitches", [])) == 12
    for entry in stitch_info.get("stitches", []):
        stitch_path = HERE / entry["stitch_path"]
        stitch_ok &= sha_file(stitch_path) == entry["stitch_file_sha256"]
        obj = torch.load(stitch_path, map_location="cpu", weights_only=True)
        stitch_ok &= sha_tensor(obj["weight_fp16"]) == entry["weight_sha256"]
        stitch_ok &= sha_tensor(obj["scales"]) == entry["scales_sha256"]
        stitch_ok &= sha_tensor(obj["zeros"]) == entry["zeros_sha256"]
        stitch_ok &= sha_tensor(obj["g_idx"]) == entry["g_idx_sha256"]
    inclusion_ok = True
    row_lookup = defaultdict(dict)
    for r in choices:
        row_lookup[(r["stem"], r["coordinate"])][int(r["row"])] = float(r["search_sse"])
    for st, u in units.items():
        score_i = math.fsum(row_lookup[(st, "identity")][row] for row in ROWS[u["projection"]])
        score_f = math.fsum(row_lookup[(st, "fixed")][row] for row in ROWS[u["projection"]])
        expected_coord = "identity" if score_i <= score_f else "fixed"
        inclusion_ok &= float(u["D_identity_search_sse"]) == score_i
        inclusion_ok &= float(u["D_fixed_search_sse"]) == score_f
        inclusion_ok &= u["selected_coordinate"] == expected_coord
        inclusion_ok &= float(u["selected_search_sse"]) == min(score_i, score_f)
        inclusion_ok &= json.loads(u["alpha_by_row_json"]) == {
            str(int(r["row"])): float(r["alpha"]) for r in choices
            if r["stem"] == st and r["coordinate"] == expected_coord
        }
    status = all((orig_ok, fixed_ok, candidate_ok, stitch_ok, inclusion_ok))
    checks = {"status": "PASS" if status else "FAIL", "protocol_sha256": sha_file(HERE / "FROZEN_PROTOCOL.json"),
              "target_split_units": len(units), "per_coordinate_row_alpha_choices": len(choices),
              "candidate_row_tensor_audits": len(audits), "selected_four_row_stitches": len(stitch_info.get("stitches", [])),
              "G01_search_choices_reproduced_from_v24": bool(orig_ok),
              "G11_search_choices_reproduced_from_v32": bool(fixed_ok),
              "candidate_tensor_files_and_params_present": bool(candidate_ok),
              "stitch_tensor_hashes_pass": bool(stitch_ok),
              "shared_coordinate_argmin_and_identity_inclusion_pass": bool(inclusion_ok),
              "held_score_values_used": False, "CUDA_calls": 0, "model_forwards": 0, "GPTQ_executions": 0}
    dump(HERE / "SEARCH_CORRECTNESS.json", checks)
    if not status:
        raise RuntimeError(f"locked search artifact verification failed: {checks}")
    return checks


def held_analysis() -> None:
    started = time.perf_counter()
    lock_path = HERE / "SEARCH_SELECTION_LOCK.json"
    if not lock_path.is_file():
        raise RuntimeError("search choices are not locked; held analysis is forbidden")
    lock = json.loads(lock_path.read_text("utf-8"))
    if lock["status"] != "LOCKED_BEFORE_HELD_ANALYSIS" or lock["protocol_sha256"] != sha_file(HERE / "FROZEN_PROTOCOL.json"):
        raise RuntimeError("selection lock/protocol mismatch")
    for name, expected in lock["selection_files_sha256"].items():
        if sha_file(HERE / name) != expected:
            raise RuntimeError(f"locked selection artifact changed: {name}")
    if (HERE / "HELD_ROW_RESULTS.csv").exists():
        raise RuntimeError("held result already exists; refusing to overwrite")
    search_checks = verify_locked_search_artifacts(lock)
    implementation_binding = {
        "selection_lock_sha256": sha_file(lock_path),
        "protocol_sha256": sha_file(HERE / "FROZEN_PROTOCOL.json"),
        "prepare_v33_py_sha256": sha_file(HERE / "prepare_v33.py"),
        "run_v33_py_sha256": sha_file(HERE / "run_v33.py"),
        "search_correctness_sha256": sha_file(HERE / "SEARCH_CORRECTNESS.json"),
        "bound_before_held_score_sources_opened": True,
    }
    dump(HERE / "IMPLEMENTATION_BINDING.json", implementation_binding)
    selections = {r["stem"]: r for r in read_csv(HERE / "SHARED_D_SELECTION.csv")}
    choice_rows = read_csv(HERE / "SEARCH_ROW_CHOICES.csv")
    chosen_alpha = {(r["stem"], int(r["row"])): float(r["alpha"])
                    for r in choice_rows if r["coordinate"] == selections[r["stem"]]["selected_coordinate"]}
    eps_rows = read_csv(V32 / "FOUR_ARM_ROW_RESULTS.csv")
    epsilon_by_unit = {r["unit_id"]: float(r["epsilon_mse_v24_frozen"]) for r in eps_rows}
    if len(epsilon_by_unit) != 48:
        raise RuntimeError("frozen v24 row epsilon table is incomplete")
    selected_rows = []
    held_hashes = {}
    for li in LAYERS:
        for pr in PROJECTIONS:
            for sp in SPLITS:
                st = stem(li, pr, sp)
                hp = V32_RAW / f"{st}_HELD_RESULTS.json"
                held_hashes[hp.relative_to(RESULTS).as_posix()] = sha_file(hp)
                raw = json.loads(hp.read_text("utf-8"))
                if raw.get("status") != "COMPLETE" or raw.get("protocol_sha256") != sha_file(V32 / "FROZEN_PROTOCOL.json"):
                    raise RuntimeError(f"v32 held artifact invalid: {st}")
                if raw["held_doc_indices"] != PROTO["inherits"]["split_documents"][sp]["held"]:
                    raise RuntimeError(f"held document mismatch: {st}")
                unit = selections[st]
                coord = unit["selected_coordinate"]
                selected_alpha_map = json.loads(unit["alpha_by_row_json"])
                for hr in raw["rows"]:
                    row = int(hr["row"])
                    a_selected = chosen_alpha[(st, row)]
                    if a_selected != float(selected_alpha_map[str(row)]):
                        raise RuntimeError(f"choice table mismatch at held evaluation: {st}/{row}")
                    if (float(hr["G01_alpha"]) != float(json.loads(unit["identity_alpha_by_row_json"])[str(row)]) or
                        float(hr["G11_alpha"]) != float(json.loads(unit["fixed_alpha_by_row_json"])[str(row)])):
                        raise RuntimeError(f"held artifact candidate alpha mapping mismatch: {st}/{row}")
                    baseline = float(hr["G01_held_mse"])
                    selected = float(hr["G01_held_mse"] if coord == "identity" else hr["G11_held_mse"])
                    # Epsilon is frozen in v32 per-row output. Read it only after the search lock.
                    epsilon = epsilon_by_unit[unit_id(li, pr, row, sp)]
                    gain = (baseline - selected) / baseline
                    v32_g11_gain = (baseline - float(hr["G11_held_mse"])) / baseline
                    outcome = held_outcome(baseline, selected, epsilon)
                    v32_outcome = held_outcome(baseline, float(hr["G11_held_mse"]), epsilon)
                    selected_rows.append({
                        "stem": st, "split": sp, "layer": li, "projection": pr, "row": row,
                        "unit_id": unit_id(li, pr, row, sp), "selected_coordinate": coord,
                        "selected_D_sha256": unit["selected_D_sha256"], "selected_alpha": a_selected,
                        "G01_alpha": float(hr["G01_alpha"]), "G11_alpha": float(hr["G11_alpha"]),
                        "G01_held_mse": baseline, "selected_held_mse": selected,
                        "selected_held_sse": selected * N_TOKENS,
                        "relative_gain_vs_G01": gain, "epsilon_mse_v24_frozen": epsilon,
                        "outcome_vs_G01": outcome, "regression_gt10pct_vs_G01": gain < -0.10,
                        "v32_G11_relative_gain_vs_G01": v32_g11_gain,
                        "v32_G11_regression_gt10pct": v32_g11_gain < -0.10,
                        "v32_G11_outcome": v32_outcome,
                        "v32_severe_regression_avoided": v32_g11_gain < -0.10 and gain >= -0.10,
                        "new_severe_regression_vs_v32_G11": gain < -0.10 and v32_g11_gain >= -0.10,
                        "new_loss_vs_v32_G11": outcome == "loss" and v32_outcome != "loss",
                        "held_coordinate_drift_mse": float(raw["held_coordinate_drift"]["unquantized_fp16_coordinate_drift_mse_by_row"][str(row)]),
                        "held_doc_indices": ";".join(map(str, raw["held_doc_indices"])),
                    })
    if len(selected_rows) != 48:
        raise RuntimeError(f"expected 48 selected held rows, got {len(selected_rows)}")

    summaries = summarize_rows(selected_rows)
    write_csv(HERE / "HELD_ROW_RESULTS.csv", selected_rows)
    write_csv(HERE / "HELD_SUMMARY.csv", summaries)
    negatives = [r for r in selected_rows if r["outcome_vs_G01"] == "loss" or r["regression_gt10pct_vs_G01"]]
    if negatives:
        write_csv(HERE / "NEGATIVE_CASES.csv", negatives)
    else:
        with (HERE / "NEGATIVE_CASES.csv").open("w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=list(selected_rows[0])).writeheader()
    alpha_counts = defaultdict(int)
    for r in choice_rows:
        coord = selections[r["stem"]]["selected_coordinate"]
        if r["coordinate"] == coord:
            alpha_counts[(r["split"], coord, float(r["alpha"]))] += 1
    alpha_rows = []
    for sp in SPLITS:
        for coord in ("identity", "fixed"):
            total_coord_rows = sum(1 for rr in choice_rows if rr["split"] == sp and selections[rr["stem"]]["selected_coordinate"] == coord and rr["coordinate"] == coord)
            for alpha in ALPHAS:
                count = alpha_counts[(sp, coord, alpha)]
                alpha_rows.append({"split": sp, "coordinate": coord, "alpha": alpha,
                                   "selected_row_count": count,
                                   "fraction_within_coordinate_rows": count / total_coord_rows if total_coord_rows else 0.0})
    write_csv(HERE / "SELECTED_ALPHA_DISTRIBUTION.csv", alpha_rows)
    units = summarize_units(selected_rows, selections)
    write_csv(HERE / "SHARED_UNIT_HELD.csv", units)
    selected_fixed = [u for u in units if u["selected_coordinate"] == "fixed"]
    fixed_improve = sum(float(u["selected_minus_G01_held_sse"]) < 0 for u in selected_fixed)
    fixed_degrade = sum(float(u["selected_minus_G01_held_sse"]) > 0 for u in selected_fixed)
    fixed_tie = sum(float(u["selected_minus_G01_held_sse"]) == 0 for u in selected_fixed)
    old_severe = sum(r["v32_G11_regression_gt10pct"] for r in selected_rows)
    avoided = sum(r["v32_severe_regression_avoided"] for r in selected_rows)
    new_severe = sum(r["new_severe_regression_vs_v32_G11"] for r in selected_rows)
    new_losses = sum(r["new_loss_vs_v32_G11"] for r in selected_rows)
    selected_unit_count = len(units)
    selection_summary = {
        "status": "COMPLETE_EXPLORATORY",
        "protocol_sha256": sha_file(HERE / "FROZEN_PROTOCOL.json"),
        "selection_lock_sha256": sha_file(lock_path),
        "selection_lock_was_written_before_held_sources_opened": True,
        "held_source_sha256": held_hashes,
        "held_rows": len(selected_rows),
        "selected_fixed_D_units": len(selected_fixed),
        "selected_identity_units": selected_unit_count - len(selected_fixed),
        "search_choice_inclusion_pass": lock["identity_inclusion_property_pass"],
        "G01_reproduction_pass": search_checks["G01_search_choices_reproduced_from_v24"],
        "G11_reproduction_pass": search_checks["G11_search_choices_reproduced_from_v32"],
        "candidate_tensor_and_quant_params_pass": search_checks["candidate_tensor_files_and_params_present"],
        "selected_stitches_pass": search_checks["stitch_tensor_hashes_pass"],
        "fixed_D_selected_units_held_improve_degrade_tie_pooled": [fixed_improve, fixed_degrade, fixed_tie],
        "v32_G11_severe_rows": old_severe,
        "v32_severe_regressions_avoided": avoided,
        "new_severe_regressions_vs_v32_G11": new_severe,
        "new_row_losses_vs_v32_G11": new_losses,
        "strict_tie_epsilon_used_from_v24": True,
        "epsilon_row_count": len(selected_rows),
        "PPL": "NOT_RUN",
        "GPU_calls": 0,
        "model_forwards": 0,
        "GPTQ_executions": 0,
        "held_analysis_cpu_seconds": time.perf_counter() - started,
    }
    dump(HERE / "HELD_ANALYSIS_SUMMARY.json", selection_summary)
    dump(HERE / "CORRECTNESS.json", {
        "status": "PASS" if (lock["identity_inclusion_property_pass"] and len(selected_rows) == 48 and len(units) == 12) else "FAIL",
        "protocol_sha256": selection_summary["protocol_sha256"],
        "selection_lock_sha256": selection_summary["selection_lock_sha256"],
        "search_selection_checks": search_checks,
        "held_rows": len(selected_rows), "held_units": len(units),
        "held_uses_only_locked_coordinates_and_alpha": True,
        "epsilon_source_v24_frozen": selection_summary["strict_tie_epsilon_used_from_v24"],
        "GPU_calls": 0, "model_forwards": 0, "GPTQ_executions": 0, "PPL_runs": 0,
    })
    write_report(selection_summary, summaries, units)
    dump(HERE / "RUN_REPORT.json", {"status": selection_summary["status"],
                                     "protocol_sha256": selection_summary["protocol_sha256"],
                                     "selection_lock_sha256": selection_summary["selection_lock_sha256"],
                                     "correctness": json.loads((HERE / "CORRECTNESS.json").read_text("utf-8")),
                                     "held_analysis": selection_summary,
                                     "summaries": summaries, "unit_results": units,
                                     "whole_model": "NOT_RUN", "PPL": "NOT_RUN"})
    write_reproduce()
    add_budget("post_lock_held_analysis_and_reporting", time.perf_counter() - started)
    write_manifest()
    print(json.dumps({"status": selection_summary["status"],
                      "selected_fixed_D_units": selection_summary["selected_fixed_D_units"],
                      "selected_identity_units": selection_summary["selected_identity_units"],
                      "overall": next(x for x in summaries if x["level"] == "overall"),
                      "S0_S1": [x for x in summaries if x["level"] == "split"],
                      "v32_severe_avoided": avoided, "new_severe": new_severe,
                      "elapsed_seconds": selection_summary["held_analysis_cpu_seconds"]},
                     ensure_ascii=False, indent=2))


def summary_levels(rows: list[dict]):
    for level, keys in (("overall", ()), ("split", ("split",)),
                        ("target_both_splits", ("layer", "projection")),
                        ("target_split", ("split", "layer", "projection"))):
        groups = defaultdict(list)
        for r in rows:
            groups[tuple(r[k] for k in keys)].append(r)
        for key, items in sorted(groups.items(), key=lambda kv: tuple(str(v) for v in kv[0])):
            yield level, dict(zip(keys, key)), items


def summarize_rows(rows: list[dict]) -> list[dict]:
    out = []
    for level, labels, items in summary_levels(rows):
        gains = [float(r["relative_gain_vs_G01"]) for r in items]
        base = math.fsum(float(r["G01_held_mse"]) * N_TOKENS for r in items)
        selected = math.fsum(float(r["selected_held_mse"]) * N_TOKENS for r in items)
        positive = sorted((max(0.0, (float(r["G01_held_mse"]) - float(r["selected_held_mse"])) * N_TOKENS)
                           for r in items), reverse=True)
        gross_positive = math.fsum(positive)
        gross_harm = math.fsum(max(0.0, (float(r["selected_held_mse"]) - float(r["G01_held_mse"])) * N_TOKENS)
                               for r in items)
        worst = min(gains)
        out.append({"level": level, "split": labels.get("split", "ALL"),
                    "layer": labels.get("layer", "ALL"), "projection": labels.get("projection", "ALL"),
                    "row_split_units": len(items), "G01_pooled_held_sse": base,
                    "selected_pooled_held_sse": selected, "net_sse_reduction": base - selected,
                    "pooled_relative_gain": (base - selected) / base,
                    "wins": sum(r["outcome_vs_G01"] == "win" for r in items),
                    "ties": sum(r["outcome_vs_G01"] == "tie" for r in items),
                    "losses": sum(r["outcome_vs_G01"] == "loss" for r in items),
                    "median_row_relative_gain": statistics.median(gains),
                    "worst_row_relative_gain": worst,
                    "worst_unit_id": items[gains.index(worst)]["unit_id"],
                    "regressions_gt10pct": sum(g < -0.10 for g in gains),
                    "gross_positive_savings_sse": gross_positive, "gross_harm_sse": gross_harm,
                    "top1_share_of_positive_savings": positive[0] / gross_positive if gross_positive else 0.0,
                    "top3_share_of_positive_savings": math.fsum(positive[:3]) / gross_positive if gross_positive else 0.0})
    return out


def summarize_units(rows: list[dict], selections: dict[str, dict[str, str]]) -> list[dict]:
    groups = defaultdict(list)
    for r in rows:
        groups[r["stem"]].append(r)
    out = []
    for st, items in sorted(groups.items()):
        base = math.fsum(float(r["G01_held_mse"]) * N_TOKENS for r in items)
        chosen = math.fsum(float(r["selected_held_sse"]) for r in items)
        delta = chosen - base
        out.append({"stem": st, "split": items[0]["split"], "layer": items[0]["layer"],
                    "projection": items[0]["projection"],
                    "selected_coordinate": selections[st]["selected_coordinate"],
                    "selected_D_sha256": selections[st]["selected_D_sha256"],
                    "G01_held_sse": base, "selected_held_sse": chosen,
                    "selected_minus_G01_held_sse": delta,
                    "pooled_unit_outcome": "improve" if delta < 0 else "degrade" if delta > 0 else "tie",
                    "row_wins": sum(r["outcome_vs_G01"] == "win" for r in items),
                    "row_ties": sum(r["outcome_vs_G01"] == "tie" for r in items),
                    "row_losses": sum(r["outcome_vs_G01"] == "loss" for r in items),
                    "v32_G11_severe_rows": sum(r["v32_G11_regression_gt10pct"] for r in items),
                    "v33_severe_rows": sum(r["regression_gt10pct_vs_G01"] for r in items)})
    if len(out) != 12:
        raise RuntimeError("expected 12 target-split held units")
    return out


def write_report(summary: dict, summaries: list[dict], units: list[dict]) -> None:
    overall = next(r for r in summaries if r["level"] == "overall")
    split_rows = [r for r in summaries if r["level"] == "split"]
    target_rows = [r for r in summaries if r["level"] == "target_both_splits"]
    positive_targets = [r for r in target_rows if r["net_sse_reduction"] > 0]
    max_target = max(target_rows, key=lambda r: r["net_sse_reduction"])
    remaining_base = overall["G01_pooled_held_sse"] - max_target["G01_pooled_held_sse"]
    remaining_reduction = overall["net_sse_reduction"] - max_target["net_sse_reduction"]
    lines = [
        "# v33 — 已有两种坐标下的共享缩放选择诊断", "",
        "> 只复用 v24/v32 已冻结的数据与候选；所有文章已被查看，本轮是探索性诊断，不是新质量验证。四行只是固定代表行，不能代表完整 Linear。", "",
        "## 冻结选择规则", "",
        "选择单位为 target×split。对 D=I 和 v32 已有固定 D，分别逐行按 search MSE 从原 10 点网格选 alpha，精确并列取较大 alpha，再将四行 MSE×32768 相加。总分较低的 D 获选；D 总分精确相同时取 D=I，不用容差。每个单位四行共用同一个 D，行可各自选 alpha。search 记录和候选提交审计先保存并锁定，随后才读取 held。", "",
        f"search 候选集合包含关系检查：{summary['search_choice_inclusion_pass']}；D=I 时重现 G01 alpha/score：{summary['G01_reproduction_pass']}；固定 D 时重现 v32 G11 alpha/score：{summary['G11_reproduction_pass']}。共选择固定 D {summary['selected_fixed_D_units']}/12 个 target×split 单位，D=I {summary['selected_identity_units']}/12 个。", "",
        "## Held 对 G01", "",
        "改善为 pooled SSE 下降；W/T/L 按冻结的 v24 per-row epsilon 分类，严重退化按相对 G01 严格低于 −10% 统计。", "",
        "| Split | G01 SSE | 共享 D 选择 SSE | 改善 | W/T/L | 中位逐行改善 | 最差 | >10% 退化 | Top-3 正节省占比 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in split_rows:
        lines.append(f"| {r['split']} | {r['G01_pooled_held_sse']:.6f} | {r['selected_pooled_held_sse']:.6f} | {100*r['pooled_relative_gain']:+.3f}% | {r['wins']}/{r['ties']}/{r['losses']} | {100*r['median_row_relative_gain']:+.3f}% | {100*r['worst_row_relative_gain']:+.3f}% ({r['worst_unit_id']}) | {r['regressions_gt10pct']} | {100*r['top3_share_of_positive_savings']:.2f}% |")
    lines.extend(["", f"Overall pooled gain {100*overall['pooled_relative_gain']:+.3f}%; W/T/L {overall['wins']}/{overall['ties']}/{overall['losses']}; median {100*overall['median_row_relative_gain']:+.3f}%; worst {100*overall['worst_row_relative_gain']:+.3f}% ({overall['worst_unit_id']}); >10% regressions {overall['regressions_gt10pct']}/{overall['row_split_units']}; top-3 share of positive savings {100*overall['top3_share_of_positive_savings']:.2f}%.", "",
                   "### Target 汇总（两个 split 配对）", "",
                   "| Target | Pooled gain | W/T/L | Median | Worst | >10%退化 |",
                   "|---|---:|---:|---:|---:|---:|"])
    for r in target_rows:
        lines.append(f"| L{r['layer']} {r['projection']} | {100*r['pooled_relative_gain']:+.3f}% | {r['wins']}/{r['ties']}/{r['losses']} | {100*r['median_row_relative_gain']:+.3f}% | {100*r['worst_row_relative_gain']:+.3f}% ({r['worst_unit_id']}) | {r['regressions_gt10pct']} |")
    lines.extend(["", "### 固定 D 被选中的 held 结果与尾部", "",
                  f"在被 search 选中固定 D 的 {summary['selected_fixed_D_units']} 个单位中，held pooled SSE 改善/退化/精确持平为 {summary['fixed_D_selected_units_held_improve_degrade_tie_pooled'][0]}/{summary['fixed_D_selected_units_held_improve_degrade_tie_pooled'][1]}/{summary['fixed_D_selected_units_held_improve_degrade_tie_pooled'][2]}。行级 W/T/L 与每个单位细节见 `SHARED_UNIT_HELD.csv`、`HELD_ROW_RESULTS.csv`。",
                  f"v32 G11 相对 G01 有 {summary['v32_G11_severe_rows']} 个 >10% 严重退化行；共享坐标选择避开 {summary['v32_severe_regressions_avoided']} 个，仍有 {summary['v32_G11_severe_rows']-summary['v32_severe_regressions_avoided']} 个；新出现的 >10% 严重退化 {summary['new_severe_regressions_vs_v32_G11']} 个，新增 epsilon 判定 loss {summary['new_row_losses_vs_v32_G11']} 个。", "",
                  f"最大净节省 target 为 L{max_target['layer']} {max_target['projection']}，净节省 {max_target['net_sse_reduction']:.3f} SSE。移除该 target 后，其余目标相对 G01 的 pooled 变化为 {100*remaining_reduction/remaining_base:+.3f}%。正收益目标 {len(positive_targets)}/6。", "",
                  "## 结论", "",
                  "仅当 S0 和 S1 都改善且尾部/集中度未明显恶化时才提出新文章冻结验证。若 split 增益方向不一致、目标贡献集中或尾部退化仍明显，则停止当前两种坐标的联合选择扩展。该诊断不构成新文章盲测，不是完整 Linear 的全行最优选择，也不支持 AWQ 新方法、正 interaction 或 PPL 改善结论。", "",
                  f"本轮结果：S0/S1 pooled gain 分别为 {100*split_rows[0]['pooled_relative_gain']:+.3f}% / {100*split_rows[1]['pooled_relative_gain']:+.3f}%；最大目标贡献占净节省 {max_target['net_sse_reduction']/overall['net_sse_reduction']*100 if overall['net_sse_reduction'] else float('nan'):.2f}%（净节省可能因其他目标退化而被抵消）。据此停止当前两坐标联合选择扩展，不增加第三种 D 或更多缩放强度。", "",
                  "GPU、模型前向、GPTQ、PPL 与整模型量化均未运行。完整逐行负例、搜索选择、alpha 和候选张量审计见 CSV/JSON；复现命令见 `REPRODUCE.md`。"])
    (HERE / "REPORT.md").write_text("\n".join(lines) + "\n", "utf-8")


def write_reproduce() -> None:
    (HERE / "REPRODUCE.md").write_text("""# v33 reproduction\n\nFrom `E:/quant-thesis`, with the existing `.venv-gptq` CPU-capable environment:\n\n```powershell\n& '.venv-gptq/Scripts/python.exe' 'experiments/range_search_w4/results/range_w4_shared_coordinate_selection_v33/prepare_v33.py'\n& '.venv-gptq/Scripts/python.exe' 'experiments/range_search_w4/results/range_w4_shared_coordinate_selection_v33/run_v33.py' freeze-selection\n& '.venv-gptq/Scripts/python.exe' 'experiments/range_search_w4/results/range_w4_shared_coordinate_selection_v33/run_v33.py' analyze-held\n```\n\nThe selection phase reads the frozen v24 original scalar10 search scores, the v32 scaled-coordinate search score table, search locks, and selected candidate tensors/parameters. It writes `SEARCH_SELECTION_LOCK.json` before the held phase is allowed to open any held-score source. The held phase verifies that lock, then reads v32's saved per-target held results and the frozen v24 epsilon table. No GPU, model forward, GPTQ, PPL, or whole-model quantization runs. The combined CPU budget is 300 seconds.\n""", "utf-8")


def write_manifest() -> None:
    rows = []
    for p in sorted(HERE.rglob("*")):
        if not p.is_file() or "__pycache__" in p.parts or p.name == "OUTPUT_MANIFEST.csv":
            continue
        rows.append({"relative_path": p.relative_to(HERE).as_posix(), "bytes": p.stat().st_size,
                     "sha256": sha_file(p)})
    write_csv(HERE / "OUTPUT_MANIFEST.csv", rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("freeze-selection", "analyze-held"))
    phase = parser.parse_args().phase
    if phase == "freeze-selection":
        freeze_selection()
    else:
        held_analysis()


if __name__ == "__main__":
    main()
