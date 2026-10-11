"""Independent raw-to-report audit for the completed v11 full 10-split run."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.range_search_w4 import run_96row_v11 as base
from experiments.range_search_w4.run_full10_stage2_v11 import (
    EXPECTED_ROW_HASH, EXPECTED_SPLIT_HASH, OUT, SPLIT_IDS, STAGE2, V9,
    actual_weight_hashes_match, check_split_separation, digest,
    load_tensor_cache, verify_stage1_preservation,
)
from experiments.range_search_w4.run_task0_parity import tensor_hash

V8, V10 = (base.HERE / "results" / n for n in ("range_w4_dynamic_v8", "range_w4_sixrow_v10"))
GRID = base.GRID
ALL_SPLITS = tuple(f"S{i:02d}" for i in range(10))
PROJECTIONS = base.PROJECTIONS
LAYERS = base.LAYERS
STAGE1_RAW = OUT / "raw"


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise RuntimeError(f"refusing to write empty artifact {path.name}")
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, ensure_ascii=False, separators=(",", ":"))
                             if isinstance(v, (dict, list, tuple)) else v for k, v in row.items()})


def save_json(path: Path, obj: object) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def number(values: list[float], q: float | None = None) -> float:
    if not values:
        return float("nan")
    if q is None:
        return float(statistics.mean(values))
    return float(torch.quantile(torch.tensor(values, dtype=torch.float64), q))


def quantile_dict(values: list[float], prefix: str) -> dict:
    result = {f"{prefix}_{name}": number(values, q) for name, q in (
        ("min", 0.0), ("mean", None), ("median", .5), ("p90", .9), ("max", 1.0))}
    result[f"{prefix}_std"] = float(statistics.pstdev(values)) if values else float("nan")
    return result


def key_tuple(record: dict) -> tuple[int, str, int, str]:
    return int(record["layer"]), record["projection_short"], int(record["row_index"]), record["split_id"]


def verify_history(reference: dict) -> dict:
    result = {}
    for name, path in (("v8", V8), ("v9", V9), ("v10", V10)):
        actual = base.tree_fingerprint(path)
        result[name] = actual == reference["historical_fingerprints_before_v11"][name]
    return result


def source_weight_path(record: dict, stage: int, slug: str) -> Path:
    if stage == 1:
        return OUT / "raw" / "weights" / f"{record['row_id']}_{record['split_id']}.pt"
    pointer = record.get("weights_path") or record.get("source_weights_path")
    if not pointer:
        raise RuntimeError(f"no candidate weights pointer in {slug}")
    return ROOT / pointer


def validate_v10_copy(record: dict, stage: int, candidate_metrics: list[dict]) -> bool:
    if record.get("source") != "existing_v10":
        return True
    raw_pointer = record.get("source_v10_raw")
    if stage == 1:
        raw_pointer = record.get("source_v10_raw")
        # The Stage 1 artifact stores a reserialized local candidate copy;
        # provenance byte-hash applies to the original v10 source file.
        weight_path = V10 / f"weights_L{record['layer']}R{record['row_index']}_{record['split_id']}.pt"
        expected_weight_sha = record.get("source_weights_sha256")
    else:
        weight_path = ROOT / record.get("source_weights_path", "")
        expected_weight_sha = record.get("source_weights_sha256")
    if not raw_pointer or not weight_path.is_file():
        return False
    src_path = ROOT / raw_pointer
    if not src_path.is_file():
        return False
    src = load_json(src_path)
    raw_sha = record.get("source_v10_raw_sha256")
    expected_raw_sha = digest(src_path.read_bytes())
    if raw_sha is not None and raw_sha != expected_raw_sha:
        return False
    if expected_weight_sha is not None and digest(weight_path.read_bytes()) != expected_weight_sha:
        return False
    old_candidates = src.get("candidates", [])
    if len(old_candidates) != len(candidate_metrics):
        return False
    fields = ("alpha", "search_sse", "heldout_sse", "search_output_mse", "heldout_output_mse",
              "weight_mse", "scale", "zero", "weight_sha256")
    return all(all(old.get(k) == new.get(k) for k in fields)
               for old, new in zip(old_candidates, candidate_metrics))


def load_all_raw(row_by_id: dict, split_map: dict) -> tuple[list[dict], dict, list[dict], dict]:
    stage1_files = sorted(STAGE1_RAW.glob("L*_S??.json"))
    stage2_files = sorted(STAGE2.glob("L*_S??.json"))
    if len(stage1_files) != 96 * 3 or len(stage2_files) != 96 * 7:
        raise RuntimeError(f"incomplete raw landscape: Stage 1={len(stage1_files)}, Stage 2={len(stage2_files)}")
    per_split, landscapes, raw_by_key = [], [], {}
    integrity = {
        "row_manifest_and_split_indices": True,
        "pristine_weight_hashes": True,
        "activation_input_hashes": True,
        "search_only_argmin_selection": True,
        "candidate_weight_files_match_record_hashes": True,
        "v10_records_byte_and_candidate_exact": True,
        "stage1_candidate_state_isolation_recorded": True,
        "stage2_candidate_state_isolation_asserted_by_runner": True,
    }
    source_row_counts = Counter()
    input_cache_1 = load_tensor_cache(OUT / "activation_cache.pt")
    input_cache_2 = load_tensor_cache(OUT / "stage2_activation_cache.pt")
    caches = [input_cache_1["inputs"], input_cache_2["inputs"]]
    source_weight_records = []
    for stage, files in ((1, stage1_files), (2, stage2_files)):
        for path in files:
            rec = load_json(path)
            rid = rec["row_id"]
            if rid not in row_by_id:
                raise RuntimeError(f"unknown row id in {path.name}: {rid}")
            row = row_by_id[rid]
            sid = rec["split_id"]
            if sid not in split_map or (stage == 1 and sid not in ALL_SPLITS[:3]) or (stage == 2 and sid not in ALL_SPLITS[3:]):
                raise RuntimeError(f"row/split stage mismatch: {path.name}")
            split = split_map[sid]
            slug = path.stem
            expected_search_ids = [d["source_article_index"] for d in split["search"]]
            expected_held_ids = [d["source_article_index"] for d in split["heldout"]]
            expected = {
                "row_id": rid, "layer": row["layer"], "projection_short": row["projection"],
                "row_index": row["row_index"], "split_id": sid,
                "row_manifest_sha256": EXPECTED_ROW_HASH, "v9_manifest_sha256": EXPECTED_SPLIT_HASH,
                "weight_sha256": row["weight_hash"],
                "search_article_ids": expected_search_ids, "heldout_article_ids": expected_held_ids,
            }
            if any(rec.get(k) != v for k, v in expected.items()):
                integrity["row_manifest_and_split_indices"] = False
            source_row_counts[rec["source"]] += 1
            candidates = rec.get("candidates", [])
            if [float(c["alpha"]) for c in candidates] != GRID:
                raise RuntimeError(f"candidate grid mismatch in {rid}/{sid}")
            selected = min(candidates, key=lambda c: float(c["search_output_mse"]))
            if float(selected["alpha"]) != float(rec["selected_alpha"]):
                integrity["search_only_argmin_selection"] = False
            alpha1 = next(c for c in candidates if float(c["alpha"]) == 1.0)
            chosen = selected
            selected_search_gain = 1.0 - float(chosen["search_sse"]) / float(alpha1["search_sse"])
            held_gain = 1.0 - float(chosen["heldout_sse"]) / float(alpha1["heldout_sse"])
            # Confirm the named D/C aggregate fields came from their raw candidates.
            field_pairs = (("search_D_sse", alpha1["search_sse"]), ("search_C_sse", chosen["search_sse"]),
                           ("heldout_D_sse", alpha1["heldout_sse"]), ("heldout_C_sse", chosen["heldout_sse"]))
            if any(not math.isclose(float(rec[k]), float(v), rel_tol=1e-12, abs_tol=1e-12)
                   for k, v in field_pairs):
                integrity["search_only_argmin_selection"] = False
            # Recompute FP-model X hashes from the saved activation caches.
            cache = caches[0] if stage == 1 else caches[1]
            act_key = base.input_key(row["layer"], row["projection"])
            for side, hash_field in (("search", "search_x_sha256"), ("heldout", "heldout_x_sha256")):
                x = torch.cat([cache[act_key][d["token_sha256"]] for d in split[side]])
                if tensor_hash(x) != rec.get(hash_field):
                    integrity["activation_input_hashes"] = False
            if tensor_hash(input_cache_1["weights"][rid]) != row["weight_hash"] or rec["weight_sha256"] != row["weight_hash"]:
                integrity["pristine_weight_hashes"] = False
            qpath = source_weight_path(rec, stage, slug)
            if not qpath.is_file():
                integrity["candidate_weight_files_match_record_hashes"] = False
                qweights = {}
            else:
                qobj = load_tensor_cache(qpath)
                qweights = qobj.get("weights", {})
                if any(str(float(c["alpha"])) not in qweights
                       or tensor_hash(qweights[str(float(c["alpha"]))]) != c["weight_sha256"]
                       for c in candidates):
                    integrity["candidate_weight_files_match_record_hashes"] = False
            if rec["source"] == "existing_v10":
                if not validate_v10_copy(rec, stage, candidates):
                    integrity["v10_records_byte_and_candidate_exact"] = False
            if stage == 1 and rec["source"] == "v11_run":
                # Stage 1 recorded one evaluation per alpha. The runner cloned W
                # in dynamic_gptq; verify all artifact hashes and pristine manifest.
                if rec.get("candidate_evaluations_this_run") != len(GRID):
                    integrity["stage1_candidate_state_isolation_recorded"] = False
            if stage == 2 and rec["source"] == "v11_stage2" and rec.get("candidate_evaluations_this_run") != len(GRID):
                integrity["stage2_candidate_state_isolation_asserted_by_runner"] = False
            source_weight_records.append((rec, qweights))
            row_obj = row
            per_item = {
                "row_id": rid, "layer": row["layer"], "projection": row["projection"],
                "row_index": row["row_index"], "split_id": sid, "source": rec["source"],
                "selected_alpha": float(chosen["alpha"]),
                "search_D_sse": float(alpha1["search_sse"]), "search_D_mse": float(alpha1["search_output_mse"]),
                "search_C_sse": float(chosen["search_sse"]), "search_C_mse": float(chosen["search_output_mse"]),
                "heldout_D_sse": float(alpha1["heldout_sse"]), "heldout_D_mse": float(alpha1["heldout_output_mse"]),
                "heldout_C_sse": float(chosen["heldout_sse"]), "heldout_C_mse": float(chosen["heldout_output_mse"]),
                "search_gain": selected_search_gain, "held_gain": held_gain,
                "weight_mse": float(chosen["weight_mse"]), "selected_scale": chosen["scale"],
                "selected_zero": chosen["zero"], "selected_weight_sha256": chosen["weight_sha256"],
                "selected_candidate_seconds": chosen.get("candidate_seconds"),
                "all_candidate_seconds": rec.get("candidate_seconds_total"),
                "candidate_evaluations_this_run": rec.get("candidate_evaluations_this_run"),
                "search_article_ids": expected_search_ids, "heldout_article_ids": expected_held_ids,
                "search_x_sha256": rec.get("search_x_sha256"), "heldout_x_sha256": rec.get("heldout_x_sha256"),
                "candidate_count": len(candidates),
            }
            per_split.append(per_item)
            raw_by_key[(row["layer"], row["projection"], row["row_index"], sid)] = rec
            for c in candidates:
                landscapes.append({
                    "row_id": rid, "layer": row["layer"], "projection": row["projection"],
                    "row_index": row["row_index"], "split_id": sid, "source": rec["source"],
                    "alpha": float(c["alpha"]), "search_sse": float(c["search_sse"]),
                    "search_output_mse": float(c["search_output_mse"]),
                    "heldout_sse": float(c["heldout_sse"]),
                    "heldout_output_mse": float(c["heldout_output_mse"]),
                    "weight_mse": float(c["weight_mse"]), "scale": c["scale"], "zero": c["zero"],
                    "candidate_seconds": c.get("candidate_seconds"), "weight_sha256": c["weight_sha256"],
                    "candidate_peak_allocated_bytes": c.get("candidate_peak_allocated_bytes"),
                    "candidate_peak_reserved_bytes": c.get("candidate_peak_reserved_bytes"),
                    "selected_by_search": float(c["alpha"]) == float(chosen["alpha"]),
                })
    return per_split, raw_by_key, landscapes, {"checks": integrity, "source_row_counts": dict(source_row_counts),
                                              "source_weights": source_weight_records,
                                              "activation_caches": caches}


def classify_final(gains: list[float]) -> tuple[str, dict]:
    median = float(statistics.median(gains))
    worst = min(gains)
    positives = sum(g > 0 for g in gains)
    gt2 = sum(g > .02 for g in gains)
    mixed_signs = any(g > 0 for g in gains) and any(g < 0 for g in gains)
    robust = positives >= 8 and median > .05 and worst >= -.10
    if robust:
        label = "Robust-positive"
    elif mixed_signs or worst < -.10:
        label = "Mixed-risk"
    elif abs(median) <= .02 and gt2 <= 2:
        label = "Neutral"
    else:
        # Preserve the protocol's threshold gap rather than moving a threshold
        # after observing outcomes. It is reported separately in all counts.
        label = "Unclassified (frozen-rule gap)"
    return label, {"median": median, "mean": float(statistics.mean(gains)), "worst": worst,
                   "best": max(gains), "positive_split_count": positives,
                   "negative_split_count": sum(g < 0 for g in gains), "gt2_split_count": gt2,
                   "mixed_signs": mixed_signs}


def alpha_stats(alphas: list[float]) -> dict:
    hist = Counter(f"{a:.1f}" for a in alphas)
    mode_count = max(hist.values())
    mode = min(float(a) for a, n in hist.items() if n == mode_count)
    band_counts = {}
    for start in range(len(GRID) - 2):
        band = GRID[start:start + 3]
        band_counts[f"{band[0]:.1f}-{band[-1]:.1f}"] = sum(a in band for a in alphas)
    best_band, best_band_count = sorted(band_counts.items(), key=lambda x: (-x[1], x[0]))[0]
    return {"selected_alpha_histogram": dict(sorted(hist.items(), key=lambda x: float(x[0]))),
            "selected_alpha_mode": mode, "selected_alpha_mode_frequency": mode_count,
            "selected_alpha_mean": float(statistics.mean(alphas)),
            "selected_alpha_median": float(statistics.median(alphas)),
            "selected_alpha_min": min(alphas), "selected_alpha_max": max(alphas),
            "best_contiguous_three_alpha_band": best_band,
            "best_contiguous_three_alpha_band_count": best_band_count,
            "alpha_band_stable_8_of_10": best_band_count >= 8}


def main() -> None:
    row_manifest = load_json(OUT / "row_manifest.json")
    protocol = load_json(OUT / "protocol.json")
    reference = load_json(OUT / "manifest_reference.json")
    amendment = (OUT / "STAGE2_PROTOCOL_AMENDMENT.md").read_text(encoding="utf-8")
    if "Robust-positive" not in amendment or "at least 8/10" not in amendment:
        raise RuntimeError("frozen Stage 2 label rules not found")
    row_by_id = {base.row_key(r): r for r in row_manifest["rows"]}
    split_manifest = load_json(V9 / "splits_manifest.json")
    split_map = {s["split_id"]: s for s in split_manifest["splits"]}
    all_split_separation = all(
        not ({d["source_article_index"] for d in split["search"]}
             & {d["source_article_index"] for d in split["heldout"]})
        and not ({d["token_sha256"] for d in split["search"]}
                 & {d["token_sha256"] for d in split["heldout"]})
        for split in split_manifest["splits"])
    per_split, raw_by_key, landscapes, loaded = load_all_raw(row_by_id, split_map)
    integrity = loaded["checks"]
    preservation = verify_stage1_preservation()
    history = verify_history(reference)
    split_check = check_split_separation(split_manifest)
    for frozen in reference["split_article_token_hashes"]:
        split = split_map[frozen["split_id"]]
        for side in ("search", "heldout"):
            live = [{k: d[k] for k in ("source_article_index", "source_row_index", "article_sha256", "token_sha256")}
                    for d in split[side]]
            if live != frozen[side]:
                integrity["row_manifest_and_split_indices"] = False

    per_split.sort(key=lambda x: (x["layer"], x["projection"], x["row_index"], x["split_id"]))
    landscapes.sort(key=lambda x: (x["layer"], x["projection"], x["row_index"], x["split_id"], x["alpha"]))
    write_csv(OUT / "full10_per_row_split_results.csv", per_split)
    write_csv(OUT / "full10_alpha_landscapes.csv", landscapes)

    by_row = defaultdict(list)
    for item in per_split:
        by_row[item["row_id"]].append(item)
    row_summary = []
    for rid, items in by_row.items():
        items.sort(key=lambda x: x["split_id"])
        gains, alphas = [x["held_gain"] for x in items], [x["selected_alpha"] for x in items]
        label, stats = classify_final(gains)
        row = row_by_id[rid]
        row_summary.append({"row_id": rid, "layer": row["layer"], "projection": row["projection"],
                            "row_index": row["row_index"], "final_benefit_label": label,
                            **stats, **alpha_stats(alphas),
                            "selected_alpha_by_split": alphas, "held_gain_by_split": gains,
                            "search_gain_by_split": [x["search_gain"] for x in items],
                            "alpha1_search_sse_total": sum(x["search_D_sse"] for x in items),
                            "selected_search_sse_total": sum(x["search_C_sse"] for x in items),
                            "alpha1_heldout_sse_total": sum(x["heldout_D_sse"] for x in items),
                            "selected_heldout_sse_total": sum(x["heldout_C_sse"] for x in items),
                            "mean_selected_weight_mse": float(statistics.mean(x["weight_mse"] for x in items)),
                            "worst_split_id": min(items, key=lambda x: x["held_gain"])["split_id"]})
    row_summary.sort(key=lambda x: (x["layer"], x["projection"], x["row_index"]))

    stage1_rows = {r["row_id"]: r for r in read_csv(OUT / "stage1_row_summary.csv")}
    for row in row_summary:
        row["stage1_label"] = stage1_rows[row["row_id"]]["stage1_label"]
    write_csv(OUT / "full10_row_summary.csv", row_summary)

    final_labels = ["Robust-positive", "Mixed-risk", "Neutral", "Unclassified (frozen-rule gap)"]
    label_counts = Counter(x["final_benefit_label"] for x in row_summary)
    projection_summary, layer_summary = [], []
    for field, values, target in (("projection", list(PROJECTIONS), projection_summary),
                                  ("layer", list(LAYERS), layer_summary)):
        for value in values:
            rows = [r for r in row_summary if r[field] == value]
            splits = [s for s in per_split if s[field] == value]
            alphas = Counter(f"{s['selected_alpha']:.1f}" for s in splits)
            medians = [r["median"] for r in rows]
            item = {field: value, "row_count": len(rows),
                    **{f"{lab.lower().replace('-', '_').replace(' ', '_')}_count": label_counts_for(rows, lab)
                       for lab in final_labels},
                    "median_row_median_held_gain": float(statistics.median(medians)),
                    "median_row_mean_held_gain": float(statistics.median([r["mean"] for r in rows])),
                    "worst_row_regression": min(r["worst"] for r in rows),
                    "median_split_held_gain": float(statistics.median(s["held_gain"] for s in splits)),
                    "selected_alpha_counts_10_splits": dict(sorted(alphas.items(), key=lambda x: float(x[0])))}
            target.append(item)
    write_csv(OUT / "full10_projection_summary.csv", projection_summary)
    write_csv(OUT / "full10_layer_summary.csv", layer_summary)

    confusion = Counter((r["stage1_label"], r["final_benefit_label"]) for r in row_summary)
    confusion_rows = []
    stage1_labels = ["Stage1 candidate-positive", "Stage1 mixed-risk", "Stage1 neutral/negative"]
    for slabel in stage1_labels:
        denominator = sum(confusion[(slabel, flabel)] for flabel in final_labels)
        for flabel in final_labels:
            n = confusion[(slabel, flabel)]
            confusion_rows.append({"stage1_label": slabel, "final_label": flabel,
                                   "row_count": n, "within_stage1_label_fraction": n / denominator if denominator else None,
                                   "stage1_row_count": denominator})
    write_csv(OUT / "stage1_vs_full10_confusion.csv", confusion_rows)

    regression_rows = sorted((dict(x) for x in per_split), key=lambda x: (x["held_gain"], x["row_id"], x["split_id"]))[:30]
    write_csv(OUT / "full10_regression_cases.csv", regression_rows)
    tail_counts = {f"gain_lt_{t}%": sum(x["held_gain"] < t / 100 for x in per_split)
                   for t in (-5, -10, -20, -40, -50)}

    # Pooled SSE with contribution ranks at the frozen row level.
    contribution_by_row = defaultdict(lambda: {"D": 0.0, "C": 0.0})
    for x in per_split:
        c = contribution_by_row[x["row_id"]]
        c["D"] += x["heldout_D_sse"]
        c["C"] += x["heldout_C_sse"]
    contribution = []
    for rid, values in contribution_by_row.items():
        row = row_by_id[rid]
        reduction = values["D"] - values["C"]
        contribution.append({"row_id": rid, "layer": row["layer"], "projection": row["projection"],
                             "row_index": row["row_index"], "heldout_D_sse": values["D"],
                             "heldout_selected_C_sse": values["C"], "heldout_sse_reduction": reduction})
    contribution.sort(key=lambda x: (-x["heldout_sse_reduction"], x["row_id"]))
    total_d = sum(x["heldout_D_sse"] for x in contribution)
    total_c = sum(x["heldout_selected_C_sse"] for x in contribution)
    total_reduction = total_d - total_c
    for rank, item in enumerate(contribution, 1):
        item["rank_by_sse_reduction"] = rank
        item["share_of_net_total_reduction"] = item["heldout_sse_reduction"] / total_reduction if total_reduction else None
        item["share_of_baseline_D_sse"] = item["heldout_sse_reduction"] / total_d if total_d else None
    write_csv(OUT / "full10_pooled_contribution.csv", contribution)
    top_contrib = {}
    for n in (1, 5, 10, 20):
        top_contrib[str(n)] = {
            "top_n_sse_reduction": sum(x["heldout_sse_reduction"] for x in contribution[:n]),
            "share_of_net_reduction": sum(x["heldout_sse_reduction"] for x in contribution[:n]) / total_reduction if total_reduction else None,
            "share_of_baseline_sse": sum(x["heldout_sse_reduction"] for x in contribution[:n]) / total_d if total_d else None,
        }
    after_remove = {}
    for n in (1, 5):
        removed = {x["row_id"] for x in contribution[:n]}
        d_left = sum(x["heldout_D_sse"] for x in contribution if x["row_id"] not in removed)
        c_left = sum(x["heldout_selected_C_sse"] for x in contribution if x["row_id"] not in removed)
        after_remove[f"top{n}"] = {"removed_rows": sorted(removed), "remaining_D_sse": d_left,
                                    "remaining_C_sse": c_left,
                                    "pooled_gain": 1 - c_left / d_left if d_left else None}

    # Fixed alpha descriptions use the same selected-vs-D SSE normalization;
    # they are not used to select a universal alpha.
    fixed_alpha_rows = []
    candidate_lookup = {(x["row_id"], x["split_id"], x["alpha"]): x for x in landscapes}
    for alpha in (.7, .8, .9):
        values = [candidate_lookup[(x["row_id"], x["split_id"], alpha)] for x in per_split]
        gains = []
        sum_d = sum_c = 0.0
        for x, candidate in zip(per_split, values):
            d_sse = x["heldout_D_sse"]
            c_sse = candidate["heldout_sse"]
            gains.append(1 - c_sse / d_sse)
            sum_d += d_sse
            sum_c += c_sse
        fixed_alpha_rows.append({"alpha": alpha, "row_split_count": len(values),
                                 "positive_gain_count": sum(g > 0 for g in gains),
                                 "regression_count": sum(g < 0 for g in gains),
                                 "zero_gain_count": sum(g == 0 for g in gains),
                                 "median_gain": float(statistics.median(gains)),
                                 "mean_gain": float(statistics.mean(gains)),
                                 "pooled_D_sse": sum_d, "pooled_fixed_alpha_sse": sum_c,
                                 "pooled_gain": 1 - sum_c / sum_d if sum_d else None,
                                 "worst_regression": min(gains)})
    write_csv(OUT / "full10_fixed_alpha_summary.csv", fixed_alpha_rows)

    # Aggregate the frozen Stage 1 row features with the remaining FP inputs
    # and H/Hinv diagonals. This is descriptive only; no feature ranking/model.
    stage1_feature_rows = read_csv(OUT / "row_features.csv")
    stage1_feature_by_row = defaultdict(list)
    for f in stage1_feature_rows:
        stage1_feature_by_row[f["row_id"]].append(f)
    stage1_cache, stage2_cache = loaded["activation_caches"]
    feature_outcomes = []
    alpha1_candidate_by_key = {}
    for x in per_split:
        rec = raw_by_key[(x["layer"], x["projection"], x["row_index"], x["split_id"])]
        alpha1_candidate_by_key[(x["row_id"], x["split_id"])] = next(c for c in rec["candidates"] if float(c["alpha"]) == 1.0)
    for summary_row in row_summary:
        rid = summary_row["row_id"]
        row = row_by_id[rid]
        first_feature = next(f for f in stage1_feature_by_row[rid] if f["split_id"] == "S00")
        feature = {"row_id": rid, "layer": row["layer"], "projection": row["projection"],
                   "row_index": row["row_index"]}
        for k, v in first_feature.items():
            if k.startswith("weight_") or k.startswith("group_"):
                try:
                    feature[k] = float(v)
                except (TypeError, ValueError):
                    feature[k] = v
        row_splits = [x for x in per_split if x["row_id"] == rid]
        feature.update({"alpha1_search_sse_total": sum(x["search_D_sse"] for x in row_splits),
                        "alpha1_heldout_sse_total": sum(x["heldout_D_sse"] for x in row_splits),
                        "alpha1_search_output_mse_mean": float(statistics.mean(x["search_D_mse"] for x in row_splits)),
                        "alpha1_heldout_output_mse_mean": float(statistics.mean(x["heldout_D_mse"] for x in row_splits)),
                        "selected_heldout_sse_total": sum(x["heldout_C_sse"] for x in row_splits),
                        "mean_selected_weight_mse": summary_row["mean_selected_weight_mse"],
                        "held_gain_median": summary_row["median"], "held_gain_mean": summary_row["mean"],
                        "held_gain_worst": summary_row["worst"], "held_gain_best": summary_row["best"],
                        "positive_split_count": summary_row["positive_split_count"],
                        "negative_split_count": summary_row["negative_split_count"],
                        "selected_alpha_histogram": summary_row["selected_alpha_histogram"],
                        "selected_alpha_mode": summary_row["selected_alpha_mode"],
                        "selected_alpha_mode_frequency": summary_row["selected_alpha_mode_frequency"],
                        "selected_alpha_mean": summary_row["selected_alpha_mean"],
                        "selected_alpha_median": summary_row["selected_alpha_median"],
                        "selected_alpha_min": summary_row["selected_alpha_min"],
                        "selected_alpha_max": summary_row["selected_alpha_max"],
                        "best_contiguous_three_alpha_band": summary_row["best_contiguous_three_alpha_band"],
                        "best_contiguous_three_alpha_band_count": summary_row["best_contiguous_three_alpha_band_count"],
                        "alpha_band_stable_8_of_10": summary_row["alpha_band_stable_8_of_10"],
                        "stage1_label": summary_row["stage1_label"],
                        "final_benefit_label": summary_row["final_benefit_label"]})
        cache = stage1_cache if row_splits[0]["split_id"] in ALL_SPLITS[:3] else stage2_cache
        act_key = base.input_key(row["layer"], row["projection"])
        activation_by_side = defaultdict(lambda: defaultdict(list))
        hinv_values = defaultdict(list)
        h_values = defaultdict(list)
        for sid in ALL_SPLITS:
            split = split_map[sid]
            act_store = (stage1_cache if sid in ALL_SPLITS[:3] else stage2_cache)[act_key]
            for side in ("search", "heldout"):
                x = torch.cat([act_store[d["token_sha256"]] for d in split[side]]).float()
                activation_by_side[side]["global_rms"].append(float(x.square().mean().sqrt()))
                activation_by_side[side]["mean"].append(float(x.mean()))
                activation_by_side[side]["std"].append(float(x.std(unbiased=False)))
                feature_rms = x.square().mean(dim=0).sqrt()
                feature_var = x.var(dim=0, unbiased=False)
                for label, tensor in (("feature_rms", feature_rms), ("feature_variance", feature_var)):
                    for stat, q in (("min", 0), ("mean", None), ("p50", .5), ("p90", .9), ("p99", .99), ("max", 1)):
                        value = float(tensor.mean()) if q is None else float(torch.quantile(tensor, q))
                        activation_by_side[side][f"{label}_{stat}"].append(value)
            hpath = (OUT / "hessian_cache" if sid in ALL_SPLITS[:3] else OUT / "stage2" / "hessian_cache") / f"hinv_L{row['layer']}_{row['projection']}_{sid}.pt"
            hobj = load_tensor_cache(hpath)
            for k, v in hobj["diag_h_summary"].items(): h_values[k].append(float(v))
            for k, v in hobj["diag_hinv_cholesky_summary"].items(): hinv_values[k].append(float(v))
        for side in ("search", "heldout"):
            for metric, vals in activation_by_side[side].items():
                feature.update(quantile_dict(vals, f"activation_{side}_{metric}"))
        for metric, vals in h_values.items():
            feature.update(quantile_dict(vals, f"hessian_diag_{metric}"))
        for metric, vals in hinv_values.items():
            feature.update(quantile_dict(vals, f"hinv_cholesky_diag_{metric}"))
        feature_outcomes.append(feature)
    write_csv(OUT / "row_feature_outcome_table.csv", feature_outcomes)

    # Ensure the emitted per-split CSV is an independent, correctly recomputed
    # serialization of raw selected losses and search-only alpha selection.
    reread_split = read_csv(OUT / "full10_per_row_split_results.csv")
    reread_landscape = read_csv(OUT / "full10_alpha_landscapes.csv")
    raw_to_csv = len(reread_split) == 960 and len(reread_landscape) == 9600
    if raw_to_csv:
        for row in reread_split:
            d, c = float(row["heldout_D_sse"]), float(row["heldout_C_sse"])
            expected_gain = 1 - c / d
            if not math.isclose(float(row["held_gain"]), expected_gain, rel_tol=1e-12, abs_tol=1e-12):
                raw_to_csv = False
            subset = [x for x in reread_landscape if x["row_id"] == row["row_id"] and x["split_id"] == row["split_id"]]
            selected = min(subset, key=lambda x: float(x["search_output_mse"])) if subset else None
            if selected is None or not math.isclose(float(selected["alpha"]), float(row["selected_alpha"]), abs_tol=1e-12):
                raw_to_csv = False

    stage2_preflight = load_json(OUT / "stage2_preflight.json")
    stage2_timing = load_json(OUT / "stage2_run_timing.json")
    stage1_timing = load_json(OUT / "run_timing.json")
    stage2_peak_reserved = max(
        int(stage2_preflight["capture"]["peak_reserved_bytes"]),
        int(stage2_timing["max_candidate_peak_reserved_bytes"]),
        *(int(p["native_parity_peak_reserved_bytes"]) for p in stage2_preflight["parity"]),
    )
    checks = {
        "row_manifest_sha256_frozen": digest((OUT / "row_manifest.json").read_bytes()) == EXPECTED_ROW_HASH == protocol["row_manifest_sha256"],
        "v9_manifest_sha256_frozen_and_all_splits_match": digest((V9 / "splits_manifest.json").read_bytes()) == EXPECTED_SPLIT_HASH == protocol["v9_manifest_sha256"],
        "96_rows_by_10_splits_complete": len(per_split) == 960 and len(by_row) == 96,
        "9600_alpha_candidates_complete": len(landscapes) == 9600 and all(x["candidate_count"] == 10 for x in per_split),
        "S00_S02_and_stage1_outputs_unchanged": preservation["status"] == "PASS",
        "all_frozen_search_heldout_splits_disjoint": split_check["status"] == "PASS" and all_split_separation,
        "pristine_weight_hashes_match_manifest": integrity["pristine_weight_hashes"],
        "all_search_and_heldout_activation_hashes_match": integrity["activation_input_hashes"],
        "candidate_state_isolation_and_saved_weight_hashes": integrity["candidate_weight_files_match_record_hashes"]
            and integrity["stage1_candidate_state_isolation_recorded"] and integrity["stage2_candidate_state_isolation_asserted_by_runner"]
            and load_json(OUT / "correctness_checks.json").get("candidate_state_and_weight_hash_isolation") is True,
        "search_only_argmin_selection_and_raw_losses_match": integrity["search_only_argmin_selection"],
        "stage2_alpha1_native_parity_four_projections": stage2_preflight["checks"]["alpha1_native_parity_all_projections"]
            and len(stage2_preflight["parity"]) == 4 and all(p["status"] == "PASS" for p in stage2_preflight["parity"]),
        "v10_legacy_candidates_exactly_reused": integrity["v10_records_byte_and_candidate_exact"] and
            sum(1 for r in raw_by_key.values() if r["source"] == "existing_v10") == 10,
        "raw_to_csv_independent_recomputation": raw_to_csv,
        "v8_v9_v10_historical_trees_unchanged": all(history.values()),
        "stage1_preservation_fingerprint_pass": preservation["status"] == "PASS",
    }
    checks["status"] = "PASS" if all(checks.values()) else "FAIL"
    if checks["status"] != "PASS":
        save_json(OUT / "FULL10_CORRECTNESS.json", {"status": "FAIL", "checks": checks,
                                                        "integrity_details": integrity,
                                                        "stage1_preservation": preservation,
                                                        "historical_tree_match": history})
        raise RuntimeError("full10 correctness check failed; stopping before summary/report")

    source_record_counts = Counter(r["source"] for r in raw_by_key.values())
    candidate_count_by_source = Counter()
    stage2_candidate_seconds_by_projection = defaultdict(float)
    for r in raw_by_key.values():
        if r["source"] in ("v11_run", "v11_stage2"):
            candidate_count_by_source["new"] += int(r["candidate_evaluations_this_run"])
        else:
            candidate_count_by_source["reused"] += len(r["candidates"])
        if r["source"] == "v11_stage2":
            stage2_candidate_seconds_by_projection[r["projection_short"]] += sum(
                float(c.get("candidate_seconds", 0.0)) for c in r["candidates"])
    pooled_gain = 1 - total_c / total_d if total_d else None
    fixed_counts = {str(row["alpha"]): row for row in fixed_alpha_rows}
    screen_predictivity = {}
    for stage_label in stage1_labels:
        subset = [r for r in row_summary if r["stage1_label"] == stage_label]
        screen_predictivity[stage_label] = {
            "row_count": len(subset), "final_robust_positive_count": sum(r["final_benefit_label"] == "Robust-positive" for r in subset),
            "final_mixed_risk_count": sum(r["final_benefit_label"] == "Mixed-risk" for r in subset),
            "final_neutral_count": sum(r["final_benefit_label"] == "Neutral" for r in subset),
            "final_unclassified_count": sum(r["final_benefit_label"] == "Unclassified (frozen-rule gap)" for r in subset),
            "final_robust_positive_fraction": sum(r["final_benefit_label"] == "Robust-positive" for r in subset) / len(subset) if subset else None,
        }
    rule_gap_rows = [{"row_id": r["row_id"], "median_held_gain": r["median"],
                      "worst_held_gain": r["worst"], "positive_split_count": r["positive_split_count"],
                      "negative_split_count": r["negative_split_count"]}
                     for r in row_summary if r["final_benefit_label"] == "Unclassified (frozen-rule gap)"]
    largest_regression = regression_rows[0]
    anomalies = [
        f"Stage 2 loop measured {stage2_timing['invocation_seconds']:.1f}s versus the preregistered planning estimate with margin {stage2_preflight['estimated_candidate_wall_seconds_with_1_5x_margin']:.1f}s ({stage2_timing['invocation_seconds']/stage2_preflight['estimated_candidate_wall_seconds_with_1_5x_margin']:.2f}x).",
        "Model input capture emitted the Transformers warning: Sliding Window Attention is enabled but not implemented for sdpa; no alternate attention-backend comparison was run.",
        f"{len(rule_gap_rows)} rows fall in a gap between the frozen robust-positive, mixed-risk and neutral rules; they are reported separately without post-hoc reassignment.",
        f"Largest selected-C held-out regression: {largest_regression['row_id']} / {largest_regression['split_id']} = {largest_regression['held_gain']:.6f}.",
        "No full-model quantization, PPL, quantized-prefix or downstream evaluation was run in this stage.",
    ]
    summary = {
        "status": "COMPLETE", "correctness_status": "PASS",
        "scope": "96 frozen rows × S00-S09; within the preregistered layer/projection strata; not model-wide unbiased prevalence",
        "protocol": {"row_manifest_sha256": EXPECTED_ROW_HASH, "v9_manifest_sha256": EXPECTED_SPLIT_HASH,
                     "model": protocol["model"], "revision": protocol["revision"],
                     "model_dtype": protocol["model_dtype"], "math_dtype": protocol["math_dtype"],
                     "quantization": protocol["weight_quantization"], "group_size": protocol["group_size"],
                     "block_size": protocol["block_size"], "damp_percent": protocol["damp_percent"],
                     "desc_act": protocol["desc_act"], "alpha_grid": GRID,
                     "input_semantics": protocol["input_semantics"]},
        "rows": 96, "splits": 10, "row_split_records": len(per_split), "candidate_landscape_rows": len(landscapes),
        "candidate_evaluations": {"nominal": 9600, "new_v11_total": int(candidate_count_by_source["new"]),
                                  "reused_legacy_total": int(candidate_count_by_source["reused"]),
                                  "stage1_new": stage1_timing["candidate_evaluations_durable_new"],
                                  "stage1_reused": stage1_timing["candidate_evaluations_reused"],
                                  "stage2_new": stage2_timing["new_candidate_evaluations_durable_total"],
                                  "stage2_reused": stage2_timing["reusable_v10_candidate_evaluations"]},
        "benefit_label_counts": {k: int(label_counts[k]) for k in final_labels},
        "benefit_label_rules": {"robust_positive": "at least 8/10 held_gain>0; median>+5%; worst>=-10%",
                                "mixed_risk": "not robust-positive and mixed positive/negative split signs or worst<-10%",
                                "neutral": "not robust/mixed; abs(median)<=2%; no more than 2/10 gains>+2%",
                                "unclassified": "frozen rule gap; reported without post-hoc reassignment"},
        "stage1_screening_predictivity": screen_predictivity,
        "heldout_gain_tail_counts": {k: int(v) for k, v in tail_counts.items()},
        "worst_heldout_gain": min(x["held_gain"] for x in per_split),
        "pooled_heldout_sse": {"alpha1_D_sse": total_d, "search_selected_C_sse": total_c,
                                "net_sse_reduction": total_reduction, "pooled_gain": pooled_gain,
                                "top_row_contribution": top_contrib,
                                "pooled_gain_after_removing_top_rows": after_remove,
                                "contribution_definition": "sum of heldout SSE across 10 splits by each row; rows ranked by D minus search-selected C; negative rows remain included"},
        "fixed_alpha_descriptive_only": fixed_alpha_rows,
        "frozen_rule_gap_rows": rule_gap_rows,
        "anomalies_and_limits": anomalies,
        "projection_strata": projection_summary, "layer_strata": layer_summary,
        "runtime": {
            "stage2_invocation_seconds": stage2_timing["invocation_seconds"],
            "stage2_candidate_seconds_sum": stage2_timing["candidate_seconds_this_invocation"],
            "stage2_unique_hessian_seconds": stage2_timing["hessian_seconds_this_invocation"],
            "stage2_activation_capture_seconds": stage2_timing["activation_capture_seconds"],
            "stage2_max_candidate_peak_allocated_bytes": stage2_timing["max_candidate_peak_allocated_bytes"],
            "stage2_max_candidate_peak_reserved_bytes": stage2_timing["max_candidate_peak_reserved_bytes"],
            "stage2_max_cuda_reserved_bytes_in_capture_parity_or_candidates": stage2_peak_reserved,
            "stage2_max_cuda_reserved_mib_in_capture_parity_or_candidates": stage2_peak_reserved / (1024 ** 2),
            "stage2_preflight_diagnostic_candidate_evaluations": stage2_preflight["diagnostic_evaluations_outside_formal_count"],
            "stage2_activation_cache_bytes": stage2_timing["stage2_activation_cache_bytes"],
            "stage2_gpu_synchronized": stage2_timing["gpu_synchronized"],
            "stage2_candidate_seconds_by_projection": dict(stage2_candidate_seconds_by_projection),
            "stage2_preflight_estimated_loop_seconds_with_margin": stage2_preflight["estimated_candidate_wall_seconds_with_1_5x_margin"],
            "stage2_observed_loop_over_estimate_ratio": stage2_timing["invocation_seconds"] /
                stage2_preflight["estimated_candidate_wall_seconds_with_1_5x_margin"],
            "stage1_invocation_seconds": stage1_timing["invocation_seconds"],
            "stage1_candidate_seconds_sum": stage1_timing["candidate_seconds_this_invocation"],
            "combined_stage1_stage2_invocation_seconds": stage1_timing["invocation_seconds"] + stage2_timing["invocation_seconds"],
            "stage2_elapsed_hours": stage2_timing["invocation_seconds"] / 3600},
        "diagnostics": {"stage1_vs_full10_confusion": screen_predictivity,
                        "tail_case_count": len(regression_rows),
                        "alpha_band_stability_is_separate_from_benefit_label": True},
        "checks": checks,
        "artifacts": {"row_split_csv": "full10_per_row_split_results.csv",
                      "alpha_landscape_csv": "full10_alpha_landscapes.csv",
                      "row_summary_csv": "full10_row_summary.csv",
                      "projection_summary_csv": "full10_projection_summary.csv",
                      "layer_summary_csv": "full10_layer_summary.csv",
                      "confusion_csv": "stage1_vs_full10_confusion.csv",
                      "regressions_csv": "full10_regression_cases.csv",
                      "pooled_contribution_csv": "full10_pooled_contribution.csv",
                      "fixed_alpha_csv": "full10_fixed_alpha_summary.csv",
                      "row_features_csv": "row_feature_outcome_table.csv"},
    }
    correctness_obj = {"status": "PASS", "checks": checks, "integrity_details": integrity,
                       "source_record_counts": dict(source_record_counts),
                       "stage1_preservation": preservation, "historical_tree_match": history,
                       "stage2_parity": stage2_preflight["parity"],
                       "split_separation": split_check}
    save_json(OUT / "FULL10_CORRECTNESS.json", correctness_obj)
    save_json(OUT / "FULL10_SUMMARY.json", summary)
    write_report(summary, regression_rows, confusion_rows, checks, stage2_timing)
    print(json.dumps({"status": summary["status"], "correctness": checks,
                      "labels": summary["benefit_label_counts"], "tails": summary["heldout_gain_tail_counts"],
                      "pooled_gain": pooled_gain, "runtime": summary["runtime"]}, ensure_ascii=False, indent=2))


def label_counts_for(rows: list[dict], label: str) -> int:
    return sum(r["final_benefit_label"] == label for r in rows)


def write_report(summary: dict, worst: list[dict], confusion: list[dict], checks: dict, timing: dict) -> None:
    labels = summary["benefit_label_counts"]
    pct = lambda x: f"{100*x:.2f}%" if x is not None else "n/a"
    lines = [
        "# v11 full 10-split W4 dynamic GPTQ range search run report", "",
        "## Scope and protocol", "",
        "The landscape covers all 96 preregistered rows across S00–S09. Labels and prevalence describe only the preregistered layer/projection strata; they are not an unbiased model-wide prevalence estimate.",
        f"Model: `{summary['protocol']['model']}` revision `{summary['protocol']['revision']}`; FP16 weights, FP32 GPTQ/loss; symmetric W4A16 G128; block 128; damping 0.1; `desc_act=False`; FP model projection inputs.",
        "D is native dynamic GPTQ at alpha=1.0. C selects among ten shared-alpha dynamic GPTQ candidates by search-split output MSE. Held-out data are not used for selection.",
        f"Frozen row manifest SHA-256 `{EXPECTED_ROW_HASH}`; v9 split manifest SHA-256 `{EXPECTED_SPLIT_HASH}`.", "",
        "## Correctness", "", f"Status: **{checks['status']}**.",
    ]
    lines.extend(f"- `{k}`: {'PASS' if v else 'FAIL'}" for k, v in checks.items() if k != "status")
    lines += ["", "## Benefit stability", "",
              f"- Robust-positive: {labels['Robust-positive']}/96",
              f"- Mixed-risk: {labels['Mixed-risk']}/96",
              f"- Neutral: {labels['Neutral']}/96",
              f"- Unclassified by the frozen label rules: {labels['Unclassified (frozen-rule gap)']}/96.",
              "Unclassified rows remain separate because the frozen positive/mixed/neutral definitions do not exhaust every possible gain pattern; no outcome-dependent threshold was added.", "",
              "### Projection strata", "", "| Projection | Rows | Positive | Mixed-risk | Neutral | Unclassified | Median row median gain | Worst row gain | Selected alpha counts |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---|"]
    for x in summary["projection_strata"]:
        lines.append(f"| {x['projection']} | {x['row_count']} | {x['robust_positive_count']} | {x['mixed_risk_count']} | {x['neutral_count']} | {x['unclassified_(frozen_rule_gap)_count']} | {pct(x['median_row_median_held_gain'])} | {pct(x['worst_row_regression'])} | `{json.dumps(x['selected_alpha_counts_10_splits'])}` |")
    lines += ["", "### Layer strata", "", "| Layer | Rows | Positive | Mixed-risk | Neutral | Unclassified | Median row median gain | Worst row gain | Selected alpha counts |",
              "|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    for x in summary["layer_strata"]:
        lines.append(f"| {x['layer']} | {x['row_count']} | {x['robust_positive_count']} | {x['mixed_risk_count']} | {x['neutral_count']} | {x['unclassified_(frozen_rule_gap)_count']} | {pct(x['median_row_median_held_gain'])} | {pct(x['worst_row_regression'])} | `{json.dumps(x['selected_alpha_counts_10_splits'])}` |")
    lines += ["", "## Stage 1 screening diagnostic", "", "| Stage 1 label | Rows | Final robust-positive | Final mixed-risk | Final neutral | Unclassified | Robust-positive fraction |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for label, v in summary["stage1_screening_predictivity"].items():
        lines.append(f"| {label} | {v['row_count']} | {v['final_robust_positive_count']} | {v['final_mixed_risk_count']} | {v['final_neutral_count']} | {v['final_unclassified_count']} | {pct(v['final_robust_positive_fraction'])} |")
    lines += ["", "## Held-out tail risk", "", "Counts over all 960 row/split pairs: " + ", ".join(f"{k}: {v}" for k, v in summary["heldout_gain_tail_counts"].items()) + f". Worst gain: {pct(summary['worst_heldout_gain'])}.",
              "", "### Worst 30 cases", "", "| Layer | Projection | Row | Split | Selected alpha | Held-out gain |",
              "|---:|---|---:|---|---:|---:|"]
    lines.extend(f"| {x['layer']} | {x['projection']} | {x['row_index']} | {x['split_id']} | {x['selected_alpha']:.1f} | {pct(x['held_gain'])} |" for x in worst)
    pooled = summary["pooled_heldout_sse"]
    lines += ["", "## Pooled SSE and concentration", "",
              f"D SSE `{pooled['alpha1_D_sse']:.9g}`, search-selected C SSE `{pooled['search_selected_C_sse']:.9g}`, pooled held-out gain {pct(pooled['pooled_gain'])}.",
              "", "| Top rows | SSE reduction | Share of net reduction | Share of D SSE |", "|---:|---:|---:|---:|"]
    for n, v in pooled["top_row_contribution"].items():
        lines.append(f"| {n} | {v['top_n_sse_reduction']:.9g} | {pct(v['share_of_net_reduction'])} | {pct(v['share_of_baseline_sse'])} |")
    for k, v in pooled["pooled_gain_after_removing_top_rows"].items():
        lines.append(f"- Removing {k}: pooled gain {pct(v['pooled_gain'])}; removed rows `{', '.join(v['removed_rows'])}`.")
    lines += ["", "## Fixed-alpha descriptive comparison", "", "These values are descriptive only; no universal alpha is selected from held-out outcomes.",
              "", "| Alpha | Positive / regression / zero splits | Median gain | Pooled gain | Worst regression |", "|---:|---:|---:|---:|---:|"]
    for x in summary["fixed_alpha_descriptive_only"]:
        lines.append(f"| {x['alpha']:.1f} | {x['positive_gain_count']} / {x['regression_count']} / {x['zero_gain_count']} | {pct(x['median_gain'])} | {pct(x['pooled_gain'])} | {pct(x['worst_regression'])} |")
    lines += ["", "## Anomalies and limitations", ""]
    lines.extend(f"- {item}" for item in summary["anomalies_and_limits"])
    if summary["frozen_rule_gap_rows"]:
        lines.append("- Frozen-rule gap rows: " + ", ".join(x["row_id"] for x in summary["frozen_rule_gap_rows"]) + ".")
    lines += ["", "## Runtime and cost", "",
              f"Stage 2 invocation wall time: {timing['invocation_seconds']:.1f} s ({timing['invocation_seconds']/3600:.2f} h); summed candidate timer: {timing['candidate_seconds_this_invocation']:.1f} s; unique Hessian construction: {timing['hessian_seconds_this_invocation']:.2f} s; activation capture: {timing['activation_capture_seconds']:.2f} s.",
              f"Preflight estimate with its 1.5× margin was {summary['runtime']['stage2_preflight_estimated_loop_seconds_with_margin']:.1f} s; measured runner loop was {summary['runtime']['stage2_invocation_seconds']:.1f} s ({summary['runtime']['stage2_observed_loop_over_estimate_ratio']:.2f}× that estimate). The estimate underpredicted observed loop time; measured time is reported as the result.",
              f"New candidates: {timing['new_candidate_evaluations_durable_total']}; exact v10 reused: {timing['reusable_v10_candidate_evaluations']}; maximum candidate CUDA reserved: {timing['max_candidate_peak_reserved_bytes']/(1024**2):.1f} MiB; maximum across activation capture, native parity, and candidate evaluation: {summary['runtime']['stage2_max_cuda_reserved_mib_in_capture_parity_or_candidates']:.1f} MiB.",
              f"Stage 2 saved activation cache size: {timing['stage2_activation_cache_bytes']:,} bytes. GPU timing synchronized: `{timing['gpu_synchronized']}`.",
              "The earlier estimate was a planning estimate. This report uses measured runtime and does not claim inference-speed improvement.", "",
              "## Alpha stability", "",
              "Per-row alpha histograms, mode/frequency, mean/median/min/max, and best three-grid-point band are in `full10_row_summary.csv`; alpha-band stability does not affect benefit labels.", "",
              "## Artifacts and reproduction", "",
              "All new experiment outputs are under this result directory. No PPL, full-model quantization, quantized-prefix, downstream task, classifier, or predictor run was performed.",
              "Reproduction commands from the repository root (PowerShell):", "", "```powershell",
              "& '.venv-gptq\\Scripts\\python.exe' 'experiments/range_search_w4/run_full10_stage2_v11.py' --preflight",
              "& '.venv-gptq\\Scripts\\python.exe' 'experiments/range_search_w4/run_full10_stage2_v11.py' --run-stage2",
              "& '.venv-gptq\\Scripts\\python.exe' 'experiments/range_search_w4/summarize_full10_v11.py'",
              "```", ""]
    (OUT / "FULL10_RUN_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
