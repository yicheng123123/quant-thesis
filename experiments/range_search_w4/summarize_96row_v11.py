"""Summarize completed v11 Stage 1 raw artifacts and independently audit them."""
from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
OUT = RESULTS / "range_w4_96row_v11"
V8, V9, V10 = (RESULTS / n for n in (
    "range_w4_dynamic_v8", "range_w4_l23r448_v9", "range_w4_sixrow_v10"))
GRID = [round(i / 10, 1) for i in range(1, 11)]
SPLITS = ("S00", "S01", "S02")
PROJECTIONS = ("q_proj", "o_proj", "up_proj", "down_proj")
LAYERS = (0, 12, 23)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, obj: object) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise RuntimeError(f"refusing to write an empty table: {path.name}")
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, ensure_ascii=False, separators=(",", ":"))
                             if isinstance(v, (dict, list, tuple)) else v for k, v in row.items()})


def tree_fingerprint(path: Path) -> dict:
    result = {}
    for file in sorted(p for p in path.rglob("*") if p.is_file()):
        result[str(file.relative_to(path))] = {
            "sha256": sha256_file(file), "mtime_ns": file.stat().st_mtime_ns,
            "size_bytes": file.stat().st_size,
        }
    return result


def summary_stats(x: torch.Tensor) -> dict:
    z = x.detach().float().reshape(-1).cpu()
    return {
        "min": float(z.min()), "mean": float(z.mean()), "std": float(z.std(unbiased=False)),
        "p50": float(torch.quantile(z, .50)), "p90": float(torch.quantile(z, .90)),
        "p99": float(torch.quantile(z, .99)), "max": float(z.max()),
    }


def key_for(row: dict) -> str:
    return f"L{row['layer']}_{row['projection']}_R{row['row_index']}"


def activation_key(layer: int, projection: str) -> str:
    return f"L{layer}/{projection}"


def load_local_cache(path: Path) -> dict:
    # Local runner cache may include torch.__version__ serialized as TorchVersion.
    with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
        return torch.load(path, map_location="cpu", weights_only=True)


def write_preflight_report(payload: dict, protocol: dict) -> None:
    estimates = protocol.get("candidate_evaluation_estimates_by_projection", {})
    lines = [
        "# v11 Stage 1 projection preflight", "",
        f"Status: **{payload['status']}**", "",
        f"Frozen 96-row manifest SHA-256: `{payload['row_manifest_sha256']}`.",
        f"v9 split manifest SHA-256: `{payload['v9_manifest_sha256']}`; splits: S00–S02.",
        f"Planned Stage 1: {payload['nominal_candidate_evaluations']} candidates; reused: {payload['reusable_candidate_evaluations']}; new: {payload['new_candidate_evaluations']}.", "",
        "| Projection | In × out | Parity | H build (s) | H peak alloc / reserved (MiB) | candidate mean (s) | candidate peak alloc / reserved (MiB) | estimated formal (s, 1.5×) |",
        "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in payload["projections"]:
        mib = lambda n: f"{n / (1024**2):.1f}"
        estimate = estimates.get(item["projection"], {}).get("estimated_formal_seconds_with_1_5x_safety_factor")
        est_text = f"{estimate:.1f}" if estimate is not None else "n/a"
        lines.append(
            f"| {item['projection']} | {item['input_dim']} × {item['output_dim']} | {item['parity_status']} | "
            f"{item['hessian_build_seconds']:.3f} | {mib(item['hessian_peak_allocated_bytes'])} / {mib(item['hessian_peak_reserved_bytes'])} | "
            f"{item['candidate_mean_seconds']:.3f} | {mib(item['candidate_peak_allocated_bytes'])} / {mib(item['candidate_peak_reserved_bytes'])} | {est_text} |"
        )
    lines.extend(["", "## Gate checks", ""])
    lines.extend(f"- `{name}`: {'PASS' if value else 'FAIL'}" for name, value in payload["checks"].items())
    total_est = (float(payload["estimated_formal_seconds_with_safety_factor"])
                 + float(payload["preflight_wall_seconds"])
                 + float(payload.get("capture", {}).get("capture_seconds", 0.0)))
    lines.extend(["", f"Observed successful preflight wall time: {payload['preflight_wall_seconds']:.2f} s.",
                  f"Estimated end-to-end including preflight and activation capture: {total_est:.1f} s.",
                  f"Maximum process CUDA reserved: {payload['max_process_reserved_bytes'] / (1024**2):.1f} MiB / {payload['cuda_total_memory_bytes'] / (1024**2):.1f} MiB.", ""])
    (OUT / "PREFLIGHT_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def classify_row(split_rows: list[dict]) -> tuple[str, dict]:
    gains = [float(x["held_gain"]) for x in split_rows]
    alphas = [float(x["selected_alpha"]) for x in split_rows]
    median = sorted(gains)[1]
    worst = min(gains)
    positive = sum(g > .02 for g in gains)
    candidate_positive = positive >= 2 and median > .05 and worst >= -.10
    mixed_signs = any(g > 0 for g in gains) and any(g < 0 for g in gains)
    alpha_span = max(alphas) - min(alphas)
    alpha_unstable = alpha_span >= .3 or len(set(alphas)) == 3
    risk = (mixed_signs or worst < -.10 or alpha_unstable or abs(median) < .02)
    label = "Stage1 candidate-positive" if candidate_positive else (
        "Stage1 mixed-risk" if risk else "Stage1 neutral/negative")
    return label, {
        "held_gain_median": median, "held_gain_worst": worst,
        "held_gain_best": max(gains), "held_gain_mean": sum(gains) / len(gains),
        "splits_held_gain_gt_2pct": positive, "selected_alpha_span": alpha_span,
        "distinct_selected_alphas": len(set(alphas)),
    }


def main() -> None:
    manifest = read_json(OUT / "row_manifest.json")
    reference = read_json(OUT / "manifest_reference.json")
    protocol = read_json(OUT / "protocol.json")
    preflight = read_json(OUT / "preflight_metrics.json")
    for projection in preflight.get("projections", []):
        parity_checks = projection.get("parity", {}).get("checks", {})
        if "scale_equal" in parity_checks:
            parity_checks["scale_allclose_task0_tolerance"] = parity_checks.pop("scale_equal")
    save_json(OUT / "preflight_metrics.json", preflight)
    write_preflight_report(preflight, protocol)
    v9_manifest = read_json(V9 / "splits_manifest.json")
    split_map = {s["split_id"]: s for s in v9_manifest["splits"][:3]}
    row_by_key = {key_for(r): r for r in manifest["rows"]}
    cache = load_local_cache(OUT / "activation_cache.pt")
    raw_dir = OUT / "raw"
    raw_files = sorted(raw_dir.glob("L*_S??.json"))
    raw = [read_json(p) for p in raw_files]
    if len(raw) != 96 * 3:
        raise RuntimeError(f"Stage 1 incomplete: expected 288 row-split JSON, found {len(raw)}")

    alpha_landscape, per_split, split_by_row = [], [], defaultdict(list)
    input_hashes_match = True
    source_records = Counter()
    candidate_hash_checks = []
    selection_checks = []
    for record in raw:
        rid = record["row_id"]
        row = row_by_key[rid]
        sid = record["split_id"]
        split = split_map[sid]
        source_records[record["source"]] += 1
        if record["weight_sha256"] != row["weight_hash"]:
            input_hashes_match = False
        expected_search_ids = [d["source_article_index"] for d in split["search"]]
        expected_held_ids = [d["source_article_index"] for d in split["heldout"]]
        if record["search_article_ids"] != expected_search_ids or record["heldout_article_ids"] != expected_held_ids:
            input_hashes_match = False
        candidates = record["candidates"]
        if [float(c["alpha"]) for c in candidates] != GRID:
            raise RuntimeError(f"candidate grid mismatch in {rid}/{sid}")
        selected = min(candidates, key=lambda c: float(c["search_output_mse"]))
        selection_checks.append(selected["alpha"] == record["selected_alpha"])
        alpha1 = next(c for c in candidates if float(c["alpha"]) == 1.0)
        chosen = next(c for c in candidates if float(c["alpha"]) == float(record["selected_alpha"]))
        for c in candidates:
            alpha_landscape.append({
                "row_id": rid, "layer": record["layer"], "projection": record["projection_short"],
                "row_index": record["row_index"], "split_id": sid, "source": record["source"],
                "alpha": c["alpha"], "search_sse": c["search_sse"],
                "search_output_mse": c["search_output_mse"], "heldout_sse": c["heldout_sse"],
                "heldout_output_mse": c["heldout_output_mse"], "weight_mse": c["weight_mse"],
                "scale": c["scale"], "zero": c["zero"], "candidate_seconds": c.get("candidate_seconds"),
                "weight_sha256": c["weight_sha256"], "candidate_peak_allocated_bytes": c.get("candidate_peak_allocated_bytes"),
                "candidate_peak_reserved_bytes": c.get("candidate_peak_reserved_bytes"),
                "selected_by_search": float(c["alpha"]) == float(record["selected_alpha"]),
            })
        held_gain = 1.0 - float(chosen["heldout_sse"]) / float(alpha1["heldout_sse"])
        search_gain = 1.0 - float(chosen["search_sse"]) / float(alpha1["search_sse"])
        second = sorted(candidates, key=lambda c: float(c["search_output_mse"]))[1]
        margin = (float(second["search_output_mse"]) - float(chosen["search_output_mse"])) / max(
            abs(float(second["search_output_mse"])), 1e-30)
        item = {
            "row_id": rid, "layer": record["layer"], "projection": record["projection_short"],
            "row_index": record["row_index"], "split_id": sid, "source": record["source"],
            "selected_alpha": record["selected_alpha"],
            "search_D_sse": alpha1["search_sse"], "search_D_mse": alpha1["search_output_mse"],
            "search_C_sse": chosen["search_sse"], "search_C_mse": chosen["search_output_mse"],
            "heldout_D_sse": alpha1["heldout_sse"], "heldout_D_mse": alpha1["heldout_output_mse"],
            "heldout_C_sse": chosen["heldout_sse"], "heldout_C_mse": chosen["heldout_output_mse"],
            "search_gain": search_gain, "held_gain": held_gain,
            "weight_mse": chosen["weight_mse"], "selected_scale": chosen["scale"],
            "selected_zero": chosen["zero"], "selected_weight_sha256": chosen["weight_sha256"],
            "selected_candidate_seconds": chosen.get("candidate_seconds"),
            "all_candidate_seconds": record.get("candidate_seconds_total"),
            "candidate_evaluations_this_run": record["candidate_evaluations_this_run"],
            "search_top1_top2_margin_fraction": margin,
            "search_article_ids": record["search_article_ids"], "heldout_article_ids": record["heldout_article_ids"],
            "search_x_sha256": record["search_x_sha256"], "heldout_x_sha256": record["heldout_x_sha256"],
        }
        per_split.append(item)
        split_by_row[rid].append(item)

        qweight_file = raw_dir / "weights" / f"{rid}_{sid}.pt"
        qobj = load_local_cache(qweight_file)
        qweights = qobj["weights"]
        candidate_hash_checks.append(all(
            str(float(c["alpha"])) in qweights
            and sha256_bytes(qweights[str(float(c["alpha"]))].detach().contiguous().cpu().numpy().tobytes()) == c["weight_sha256"]
            for c in candidates
        ))

    write_csv(OUT / "per_row_split_results.csv", per_split)
    write_csv(OUT / "alpha_landscapes.csv", alpha_landscape)

    # Descriptive row and calibration feature records: one row per frozen row x split.
    feature_rows = []
    for rid, row in row_by_key.items():
        w = cache["weights"][rid].float().reshape(-1)
        absw = w.abs()
        rms = float(torch.sqrt(torch.mean(w.square())))
        mean_w = w.mean()
        std_w = w.std(unbiased=False)
        safe_std = max(float(std_w), 1e-30)
        centered = w - mean_w
        skew = float((centered.pow(3).mean() / (safe_std ** 3)))
        kurtosis_excess = float(centered.pow(4).mean() / (safe_std ** 4) - 3.0)
        groups = list(w.split(128))
        group_max = torch.tensor([g.abs().max() for g in groups])
        group_rms = torch.tensor([g.square().mean().sqrt() for g in groups])
        static_weight = {
            "weight_max_abs": float(absw.max()), "weight_rms": rms, "weight_std": float(std_w),
            "weight_max_abs_over_rms": float(absw.max()) / max(rms, 1e-30),
            "weight_kurtosis_excess": kurtosis_excess, "weight_skewness": skew,
            "weight_p90_abs": float(torch.quantile(absw, .90)), "weight_p95_abs": float(torch.quantile(absw, .95)),
            "weight_p99_abs": float(torch.quantile(absw, .99)), "weight_p999_abs": float(torch.quantile(absw, .999)),
            "weight_outlier_ratio_2rms": float((absw > 2 * rms).float().mean()),
            "weight_outlier_ratio_3rms": float((absw > 3 * rms).float().mean()),
            "weight_outlier_ratio_4rms": float((absw > 4 * rms).float().mean()),
            "group_max_abs_mean": float(group_max.mean()), "group_max_abs_std": float(group_max.std(unbiased=False)),
            "group_max_abs_max": float(group_max.max()), "group_rms_mean": float(group_rms.mean()),
            "group_rms_std": float(group_rms.std(unbiased=False)),
        }
        for sid in SPLITS:
            split = split_map[sid]
            acts = cache["inputs"][activation_key(row["layer"], row["projection"])]
            feat = {**static_weight, "row_id": rid, "layer": row["layer"],
                    "projection": row["projection"], "row_index": row["row_index"], "split_id": sid}
            for side in ("search", "heldout"):
                x = torch.cat([acts[d["token_sha256"]] for d in split[side]]).float()
                feat[f"activation_{side}_global_rms"] = float(x.square().mean().sqrt())
                feat[f"activation_{side}_mean"] = float(x.mean())
                feat[f"activation_{side}_std"] = float(x.std(unbiased=False))
                feat[f"activation_{side}_feature_rms_distribution"] = summary_stats(x.square().mean(dim=0).sqrt())
                feat[f"activation_{side}_feature_variance_distribution"] = summary_stats(x.var(dim=0, unbiased=False))
            hpath = OUT / "hessian_cache" / f"hinv_L{row['layer']}_{row['projection']}_{sid}.pt"
            hobj = load_local_cache(hpath)
            for statname, stat in hobj["diag_h_summary"].items():
                feat[f"hessian_diag_{statname}"] = stat
            for statname, stat in hobj["diag_hinv_cholesky_summary"].items():
                feat[f"hinv_cholesky_diag_{statname}"] = stat
            gptq = next(x for x in split_by_row[rid] if x["split_id"] == sid)
            feat.update({
                "alpha1_search_sse": gptq["search_D_sse"], "best_search_sse": gptq["search_C_sse"],
                "search_gain": gptq["search_gain"], "selected_alpha": gptq["selected_alpha"],
                "search_top1_top2_margin_fraction": gptq["search_top1_top2_margin_fraction"],
            })
            feature_rows.append(feat)
    write_csv(OUT / "row_features.csv", feature_rows)

    row_summaries = []
    for rid, items in split_by_row.items():
        items.sort(key=lambda x: x["split_id"])
        label, stats = classify_row(items)
        row_summaries.append({
            "row_id": rid, "layer": items[0]["layer"], "projection": items[0]["projection"],
            "row_index": items[0]["row_index"], "stage1_label": label,
            "selected_alpha_by_split": [x["selected_alpha"] for x in items],
            "held_gain_by_split": [x["held_gain"] for x in items],
            "search_gain_by_split": [x["search_gain"] for x in items],
            "held_gain_median": stats["held_gain_median"], "held_gain_mean": stats["held_gain_mean"],
            "held_gain_worst": stats["held_gain_worst"], "held_gain_best": stats["held_gain_best"],
            "splits_held_gain_gt_2pct": stats["splits_held_gain_gt_2pct"],
            "selected_alpha_span": stats["selected_alpha_span"],
            "distinct_selected_alphas": stats["distinct_selected_alphas"],
        })
    row_summaries.sort(key=lambda x: (x["layer"], x["projection"], x["row_index"]))
    write_csv(OUT / "stage1_row_summary.csv", row_summaries)

    def stratified_summary(field: str, values: list):
        selected = [x for x in row_summaries if x[field] == values]
        split_values = [s["held_gain"] for s in per_split if s[field] == values]
        counts = Counter(x["stage1_label"] for x in selected)
        alpha_count = Counter(float(s["selected_alpha"]) for s in per_split if s[field] == values)
        return {
            field: values, "row_count": len(selected), "candidate_positive_count": counts["Stage1 candidate-positive"],
            "mixed_risk_count": counts["Stage1 mixed-risk"], "neutral_negative_count": counts["Stage1 neutral/negative"],
            "held_gain_split_median": float(torch.quantile(torch.tensor(split_values), .50)),
            "held_gain_split_mean": sum(split_values) / len(split_values),
            "held_gain_split_worst": min(split_values), "held_gain_split_best": max(split_values),
            "held_gain_split_p10": float(torch.quantile(torch.tensor(split_values), .10)),
            "held_gain_split_p90": float(torch.quantile(torch.tensor(split_values), .90)),
            "selected_alpha_split_counts": dict(sorted(alpha_count.items())),
        }

    projection_summary = [stratified_summary("projection", p) for p in PROJECTIONS]
    layer_summary = [stratified_summary("layer", l) for l in LAYERS]
    write_csv(OUT / "projection_summary.csv", projection_summary)
    write_csv(OUT / "layer_summary.csv", layer_summary)

    regression_cases = sorted(per_split, key=lambda x: x["held_gain"])[:20]
    write_csv(OUT / "regression_cases.csv", regression_cases)
    tails = {f"held_gain_lt_neg_{int(abs(t) * 100)}pct": sum(x["held_gain"] < t for x in per_split)
             for t in (-.05, -.10, -.20, -.40)}

    contribution_by_row = defaultdict(lambda: {"heldout_D_sse": 0.0, "heldout_C_sse": 0.0, "heldout_reduction": 0.0})
    for x in per_split:
        c = contribution_by_row[x["row_id"]]
        c["heldout_D_sse"] += x["heldout_D_sse"]
        c["heldout_C_sse"] += x["heldout_C_sse"]
        c["heldout_reduction"] += x["heldout_D_sse"] - x["heldout_C_sse"]
    net_reduction = sum(v["heldout_reduction"] for v in contribution_by_row.values())
    contribution = []
    for rid, v in contribution_by_row.items():
        row = row_by_key[rid]
        contribution.append({"row_id": rid, "layer": row["layer"], "projection": row["projection"],
                             "row_index": row["row_index"], **v,
                             "share_of_net_heldout_reduction": v["heldout_reduction"] / net_reduction if net_reduction > 0 else None})
    contribution.sort(key=lambda x: x["heldout_reduction"], reverse=True)
    write_csv(OUT / "pooled_contribution.csv", contribution)
    top_share = {}
    for n in (1, 5, 10):
        top_share[str(n)] = sum(x["heldout_reduction"] for x in contribution[:n]) / net_reduction if net_reduction > 0 else None

    # Reload generated result CSVs and independently recompute the selected gains.
    with (OUT / "per_row_split_results.csv").open(encoding="utf-8-sig", newline="") as f:
        csv_rows = list(csv.DictReader(f))
    raw_to_csv = len(csv_rows) == len(per_split) and all(
        abs(float(c["held_gain"]) - (1 - float(c["heldout_C_sse"]) / float(c["heldout_D_sse"]))) <= 1e-12
        and float(c["selected_alpha"]) == min(
            (float(a["alpha"]) for a in alpha_landscape if a["row_id"] == c["row_id"] and a["split_id"] == c["split_id"]),
            key=lambda a: next(float(z["search_output_mse"]) for z in alpha_landscape
                               if z["row_id"] == c["row_id"] and z["split_id"] == c["split_id"] and float(z["alpha"]) == a)
        )
        for c in csv_rows
    )

    # Exact integrity check for the one row overlapping v10: three splits, ten candidates each.
    reused_records = [r for r in raw if r["source"] == "existing_v10"]
    reuse_exact = len(reused_records) == protocol["reusable_candidate_evaluations"] // 10
    for rec in reused_records:
        src_path = ROOT / rec["source_v10_raw"]
        src = read_json(src_path)
        reuse_exact = reuse_exact and sha256_file(src_path) == rec["source_v10_raw_sha256"]
        reuse_exact = reuse_exact and sha256_file(V10 / f"weights_L{rec['layer']}R{rec['row_index']}_{rec['split_id']}.pt") == rec["source_weights_sha256"]
        fields = ("alpha", "search_sse", "heldout_sse", "search_output_mse", "heldout_output_mse", "weight_mse", "scale", "zero", "weight_sha256")
        old_candidates = src["candidates"]
        reuse_exact = reuse_exact and len(old_candidates) == len(rec["candidates"])
        reuse_exact = reuse_exact and all(all(old.get(f) == new.get(f) for f in fields)
                                          for old, new in zip(old_candidates, rec["candidates"]))

    split_reuse = sha256_file(V9 / "splits_manifest.json") == reference["v9_splits_manifest_sha256"]
    frozen_split_rows = reference["split_article_token_hashes"]
    for frozen in frozen_split_rows:
        live = split_map[frozen["split_id"]]
        for side in ("search", "heldout"):
            projected = [{k: d[k] for k in ("source_article_index", "source_row_index", "article_sha256", "token_sha256")} for d in live[side]]
            split_reuse = split_reuse and projected == frozen[side]
    parity = all(p.get("parity_status") == "PASS" for p in preflight["projections"]) and len(preflight["projections"]) == 4
    history = {name: tree_fingerprint(path) == reference["historical_fingerprints_before_v11"][name]
               for name, path in (("v8", V8), ("v9", V9), ("v10", V10))}
    checks = {
        "row_manifest_frozen_sha256": sha256_file(OUT / "row_manifest.json") == reference["row_manifest_sha256"] == protocol["row_manifest_sha256"],
        "v9_split_manifest_sha256": split_reuse,
        "S00_S02_exact_article_and_token_reuse": split_reuse,
        "four_projection_alpha1_native_parity": parity,
        "pristine_weight_hashes_match_manifest": input_hashes_match,
        "candidate_state_and_weight_hash_isolation": len(candidate_hash_checks) == 288 and all(candidate_hash_checks),
        "heldout_not_used_for_selection_search_argmin_verified": len(selection_checks) == 288 and all(selection_checks),
        "existing_v10_row_splits_exactly_reused": reuse_exact,
        "raw_to_csv_independent_gain_and_selection_recompute": raw_to_csv,
        "v8_v9_v10_historical_trees_unchanged": all(history.values()),
    }
    checks["status"] = "PASS" if all(checks.values()) else "FAIL"
    checks["evidence"] = {
        "row_split_records": len(raw), "candidate_rows_in_alpha_landscapes": len(alpha_landscape),
        "new_dynamic_gptq_candidates": sum(int(r["candidate_evaluations_this_run"]) for r in raw),
        "reused_candidate_count": sum(10 for r in reused_records),
        "reused_row_split_ids": [f"{r['row_id']}/{r['split_id']}" for r in reused_records],
        "source_record_counts": dict(source_records), "historical_tree_match": history,
        "scale_parity_tolerance": "torch.allclose(atol=1e-6, rtol=1e-5); FP16 output weights and SSE checked separately",
    }
    save_json(OUT / "correctness_checks.json", checks)

    run_timing_path = OUT / "run_timing.json"
    run_timing = read_json(run_timing_path) if run_timing_path.exists() else {}
    hessian_files = list((OUT / "hessian_cache").glob("hinv_*.pt"))
    hessian_seconds = sum(float(load_local_cache(p)["hessian_build_seconds"]) for p in hessian_files)
    candidate_seconds = sum(float(r.get("candidate_seconds_total", 0.0)) for r in raw)
    preflight_wall = float(preflight.get("preflight_wall_seconds", 0.0))
    capture_seconds = float(preflight.get("capture", {}).get("capture_seconds", 0.0))
    max_peak_reserved = max(
        [int(preflight.get("max_process_reserved_bytes", 0))]
        + [int(c.get("candidate_peak_reserved_bytes", 0)) for r in raw for c in r["candidates"] if c.get("candidate_peak_reserved_bytes")]
        + [int(load_local_cache(p).get("hessian_peak_reserved_bytes", 0)) for p in hessian_files]
    )
    labels = Counter(x["stage1_label"] for x in row_summaries)
    alpha_counts = Counter(str(float(x["selected_alpha"])) for x in per_split)
    summary = {
        "status": "COMPLETE" if len(raw) == 288 else "INCOMPLETE",
        "correctness_status": checks["status"], "scope": "v11 Stage 1 only; S00-S02; no Stage 2 or PPL",
        "row_manifest_sha256": protocol["row_manifest_sha256"], "v9_manifest_sha256": protocol["v9_manifest_sha256"],
        "rows": len(row_summaries), "row_split_records": len(raw),
        "nominal_candidate_count": protocol["nominal_candidate_evaluations"],
        "reused_candidate_count": sum(10 for r in reused_records),
        "new_candidate_count": sum(int(r["candidate_evaluations_this_run"]) for r in raw),
        "candidate_label_counts": dict(labels),
        "selected_alpha_split_counts": dict(sorted(alpha_counts.items(), key=lambda x: float(x[0]))),
        "heldout_gain_tail_counts": tails,
        "pooled_heldout_sse": {
            "D": sum(x["heldout_D_sse"] for x in per_split), "selected_C": sum(x["heldout_C_sse"] for x in per_split),
            "net_reduction": net_reduction,
            "gain": 1 - sum(x["heldout_C_sse"] for x in per_split) / sum(x["heldout_D_sse"] for x in per_split),
            "top_row_shares_of_net_reduction": top_share,
            "definition": "Each row contribution sums its three heldout D-SSE minus selected-C-SSE values; top-k sorted by reduction.",
        },
        "cost": {
            "measured_stage1_invocation_seconds": run_timing.get("invocation_seconds"),
            "measured_preflight_capture_plus_stage1_seconds": (float(run_timing.get("invocation_seconds", 0.0)) + preflight_wall + capture_seconds),
            "measured_candidate_seconds_sum": candidate_seconds,
            "measured_hessian_build_seconds_sum_unique_caches": hessian_seconds,
            "activation_capture_seconds": capture_seconds,
            "final_preflight_wall_seconds": preflight_wall,
            "candidate_evaluations_this_invocation": run_timing.get("candidate_evaluations_this_invocation"),
            "gpu_synchronized": run_timing.get("gpu_synchronized"),
            "max_observed_cuda_reserved_bytes": max_peak_reserved,
            "max_observed_cuda_reserved_mib": max_peak_reserved / (1024 ** 2),
            "timing_scope_note": "Stage 1 invocation timer starts after cache load and v10 reuse verification; preflight and activation capture are itemized separately.",
        },
        "preflight_projection_status": {p["projection"]: p["parity_status"] for p in preflight["projections"]},
        "row_summary_csv": "stage1_row_summary.csv", "projections_csv": "projection_summary.csv",
        "layers_csv": "layer_summary.csv", "worst_regressions_csv": "regression_cases.csv",
    }
    save_json(OUT / "summary.json", summary)

    def fmt_pct(x): return f"{100*x:.2f}%"
    report = [
        "# v11 Stage 1 run report", "",
        f"Correctness: **{checks['status']}** ({sum(bool(v) for k, v in checks.items() if k not in ('status', 'evidence'))}/{len(checks)-2} checks passed).",
        f"Projection alpha=1 parity: {', '.join(f'{k}={v}' for k, v in summary['preflight_projection_status'].items())}.",
        f"Rows/splits: {len(row_summaries)} rows × 3 frozen splits = {len(raw)} records; S00–S02 reused from v9.",
        f"Candidates: {summary['new_candidate_count']} newly evaluated, {summary['reused_candidate_count']} reused from v10, {summary['nominal_candidate_count']} nominal.",
        f"Measured Stage 1 row-loop runtime: {run_timing.get('invocation_seconds', float('nan')):.1f} s; candidate time sum {candidate_seconds:.1f} s; unique Hessian-build time sum {hessian_seconds:.1f} s; activation capture {capture_seconds:.1f} s.",
        f"Peak observed process CUDA reserved: {max_peak_reserved/(1024**2):.1f} MiB.", "",
        "## Stage 1 row labels", "",
        f"- Candidate-positive: {labels['Stage1 candidate-positive']}",
        f"- Mixed-risk: {labels['Stage1 mixed-risk']}",
        f"- Neutral/negative: {labels['Stage1 neutral/negative']}", "",
        "These are descriptive labels within the preregistered layer/projection strata, not model-wide prevalence or stable classifications.", "",
        "## Projection strata", "",
        "| Projection | Rows | Candidate-positive | Mixed-risk | Neutral/negative | Median split held gain | Worst split held gain | Selected alpha counts (split-level) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for x in projection_summary:
        report.append(f"| {x['projection']} | {x['row_count']} | {x['candidate_positive_count']} | {x['mixed_risk_count']} | {x['neutral_negative_count']} | {fmt_pct(x['held_gain_split_median'])} | {fmt_pct(x['held_gain_split_worst'])} | `{json.dumps(x['selected_alpha_split_counts'])}` |")
    report += ["", "## Layer strata", "", "| Layer | Rows | Candidate-positive | Mixed-risk | Neutral/negative | Median split held gain | Worst split held gain | Selected alpha counts (split-level) |", "|---:|---:|---:|---:|---:|---:|---:|---|"]
    for x in layer_summary:
        report.append(f"| {x['layer']} | {x['row_count']} | {x['candidate_positive_count']} | {x['mixed_risk_count']} | {x['neutral_negative_count']} | {fmt_pct(x['held_gain_split_median'])} | {fmt_pct(x['held_gain_split_worst'])} | `{json.dumps(x['selected_alpha_split_counts'])}` |")
    report += ["", "## Selected alpha and held-out tails", "", f"Selected alpha split counts: `{json.dumps(summary['selected_alpha_split_counts'])}`."]
    report += [f"- held_gain < −{int(k.rsplit('_', 1)[1][:-3])}%: {v}/{len(per_split)}" for k, v in tails.items()]
    report += ["", "## Worst 20 held-out regressions", "", "| Layer | Projection | Row | Split | Alpha | Search gain | Held-out gain |", "|---:|---|---:|---|---:|---:|---:|"]
    report += [f"| {x['layer']} | {x['projection']} | {x['row_index']} | {x['split_id']} | {x['selected_alpha']:.1f} | {fmt_pct(x['search_gain'])} | {fmt_pct(x['held_gain'])} |" for x in regression_cases]
    report += ["", "## Pooled held-out SSE", "", f"D SSE={summary['pooled_heldout_sse']['D']:.9g}; selected C SSE={summary['pooled_heldout_sse']['selected_C']:.9g}; pooled gain={fmt_pct(summary['pooled_heldout_sse']['gain'])}.",
               f"Top-row shares of net reduction: {json.dumps(top_share)}.",
               "Contribution shares are reported only when net held-out SSE reduction is positive; they can exceed 100% when some rows regress.", "",
               "## Correctness checks", ""]
    report += [f"- `{k}`: {'PASS' if v else 'FAIL'}" for k, v in checks.items() if k != "evidence"]
    report += ["", "No Stage 2, PPL, quantized-prefix, or downstream evaluation was run.", ""]
    report += ["## Reproduction commands", "", "Run from the repository root in PowerShell:", "", "```powershell",
               "& '.venv-gptq\\Scripts\\python.exe' 'experiments/range_search_w4/run_96row_v11.py' --preflight",
               "& '.venv-gptq\\Scripts\\python.exe' 'experiments/range_search_w4/run_96row_v11.py' --run-stage1",
               "& '.venv-gptq\\Scripts\\python.exe' 'experiments/range_search_w4/summarize_96row_v11.py'",
               "```", ""]
    (OUT / "RUN_REPORT.md").write_text("\n".join(report), encoding="utf-8")

    print(json.dumps({"summary": summary, "correctness": checks}, ensure_ascii=False, indent=2))
    if checks["status"] != "PASS":
        raise SystemExit(3)


if __name__ == "__main__":
    main()
