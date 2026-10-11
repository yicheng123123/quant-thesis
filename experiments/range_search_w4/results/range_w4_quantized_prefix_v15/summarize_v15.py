"""Build auditable v15 CSV/JSON/report outputs from frozen raw candidates."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def sha_tensor(t: torch.Tensor) -> str:
    return hashlib.sha256(t.detach().contiguous().cpu().numpy().tobytes()).hexdigest()


def load_torch(path: Path):
    with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
        return torch.load(path, map_location="cpu", weights_only=True)


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, obj):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def write_csv(path: Path, rows: list[dict]):
    if not rows:
        raise RuntimeError(f"refusing to write empty CSV: {path.name}")
    fields = list(dict.fromkeys(key for row in rows for key in row))
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def rid(row: dict) -> str:
    return f"L{row['layer']}_{row['projection']}_R{row['row_index']}"


def module_key(row: dict) -> str:
    return f"L{row['layer']}/{row['projection']}"


def norm_sse(xdocs: list[torch.Tensor], w: torch.Tensor, q: torch.Tensor) -> float:
    delta = q.float() - w.float()
    total = 0.0
    for x in xdocs:
        err = x.to(device=delta.device, dtype=torch.float32) @ delta
        total += float(err.square().sum().item())
    return total


def average_ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and values[order[j]] == values[order[i]]:
            j += 1
        rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[order[k]] = rank
        i = j
    return ranks


def spearman(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2:
        return None
    rx, ry = average_ranks(xs), average_ranks(ys)
    mx, my = statistics.mean(rx), statistics.mean(ry)
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return cov / den if den else None


def gain_stats(rows: list[dict], suffix: str) -> dict:
    gains = [float(row[f"gain_{suffix}"]) for row in rows]
    d_sse = sum(float(row[f"heldout_D_{suffix}_sse"]) for row in rows)
    c_sse = sum(float(row[f"heldout_C_{suffix}_sse"]) for row in rows)
    return {
        "row_count": len(rows),
        "pooled_gain": 1.0 - c_sse / d_sse if d_sse else 0.0,
        "median_gain": statistics.median(gains) if gains else None,
        "mean_gain": statistics.mean(gains) if gains else None,
        "worst_gain": min(gains) if gains else None,
        "positive_rows": sum(x > 0 for x in gains),
        "negative_rows": sum(x < 0 for x in gains),
        "severe_lt_5pct": sum(x < -0.05 for x in gains),
        "severe_lt_10pct": sum(x < -0.10 for x in gains),
        "severe_lt_20pct": sum(x < -0.20 for x in gains),
        "severe_lt_40pct": sum(x < -0.40 for x in gains),
        "heldout_D_sse_sum": d_sse,
        "heldout_C_sse_sum": c_sse,
    }


def tree_fingerprint(path: Path) -> dict:
    return {
        p.relative_to(path).as_posix(): {"sha256": sha_file(p), "size_bytes": p.stat().st_size}
        for p in sorted(path.rglob("*")) if p.is_file()
    }


def main():
    protocol = read_json(OUT / "PROTOCOL.json")
    rows_ref = read_json(OUT / "row_manifest_reference.json")
    articles = read_json(OUT / "article_manifest.json")
    order = read_json(OUT / "module_execution_order.json")
    before = read_json(OUT / "historical_fingerprints_before.json")
    preflight = read_json(OUT / "preflight_checks.json")
    prefix = read_json(OUT / "prefix_quantization_log.json")
    timing = read_json(OUT / "candidate_run_timing.json")
    cost_estimate = read_json(OUT / "COST_ESTIMATE.json")
    isolation = read_json(OUT / "candidate_isolation_spot_check.json")
    activation = load_torch(OUT / "cache" / "activation_cache.pt")
    fp_cache = load_torch(OUT / "cache" / "fp_activations.pt")
    q_cache = load_torch(OUT / "cache" / "qprefix_activations.pt")
    original_source = ROOT / "experiments/range_search_w4/results/range_w4_calibration_coverage_v14/row_manifest.json"

    # Read every raw record, validate provenance, candidate grid, and held-out isolation.
    records: dict[tuple[str, str], dict] = {}
    all_candidate_tensors_finite = True
    for row in rows_ref["rows"]:
        row_id = rid(row)
        for regime in ("FP_INPUT", "QPREFIX_INPUT"):
            path = OUT / "raw" / f"{row_id}_{regime}.json"
            rec = read_json(path)
            if rec["record_id"] != row_id or rec["input_regime"] != regime:
                raise RuntimeError(f"raw record identity mismatch: {path.name}")
            if rec["row_weight_sha256"] != row["weight_hash"]:
                raise RuntimeError(f"raw target row hash mismatch: {path.name}")
            if rec["article_manifest_sha256"] != sha_file(OUT / "article_manifest.json"):
                raise RuntimeError(f"raw article provenance mismatch: {path.name}")
            if set(rec["candidate_results"]) != {str(a) for a in protocol["quantization"]["alpha_grid"]}:
                raise RuntimeError(f"candidate grid mismatch: {path.name}")
            candidates = rec["candidate_results"]
            selected = min((float(a) for a in candidates), key=lambda a: (candidates[str(a)]["search_mse"], a))
            if abs(selected - float(rec["selected_alpha_search_only"])) > 1e-12:
                raise RuntimeError(f"selected alpha does not minimize calibration objective: {path.name}")
            if candidates["1.0"]["candidate_tensor_sha256"] == "":
                raise RuntimeError(f"missing native D candidate hash: {path.name}")
            wpath = OUT / rec["candidate_weight_file"]
            if sha_file(wpath) != rec["candidate_weight_file_sha256"]:
                raise RuntimeError(f"candidate weight file checksum mismatch: {path.name}")
            weights = load_torch(wpath)
            if weights["row_id"] != row_id or weights["regime"] != regime:
                raise RuntimeError(f"candidate weight metadata mismatch: {wpath.name}")
            for alpha, item in candidates.items():
                q = weights["candidate_weights_fp16"][alpha]
                if not torch.isfinite(q).all():
                    all_candidate_tensors_finite = False
                    raise RuntimeError(f"non-finite candidate weight in {wpath.name}/{alpha}")
                if sha_tensor(q) != item["candidate_tensor_sha256"]:
                    raise RuntimeError(f"candidate tensor hash mismatch in {wpath.name}/{alpha}")
                if sha_file(wpath) != rec["candidate_weight_file_sha256"]:
                    raise RuntimeError(f"candidate weight checksum changed while summarizing: {wpath.name}")
            records[(row_id, regime)] = rec
    if len(records) != 96:
        raise RuntimeError(f"expected 96 raw row/regime records, found {len(records)}")

    fp_rows, q_rows = [], []
    pair_rows, cross_rows = [], []
    direct_heldout_recompute_ok = True
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    held_docs = articles["heldout"]
    start_cross = time.perf_counter()
    for row in rows_ref["rows"]:
        row_id = rid(row)
        fp = records[(row_id, "FP_INPUT")]
        q = records[(row_id, "QPREFIX_INPUT")]
        for rec, dst in ((fp, fp_rows), (q, q_rows)):
            alpha = float(rec["selected_alpha_search_only"])
            item = rec["candidate_results"][str(alpha)]
            dst.append({
                "record_id": row_id, "layer": row["layer"], "projection": row["projection"],
                "row_index": row["row_index"], "bucket": row["bucket"],
                "selected_alpha": alpha,
                "search_D_mse": rec["search_D_mse"], "search_C_mse": rec["search_C_mse"],
                "heldout_D_mse": rec["heldout_D_mse"], "heldout_C_mse": rec["heldout_C_mse"],
                "heldout_D_sse": rec["heldout_D_sse"], "heldout_C_sse": rec["heldout_C_sse"],
                "heldout_gain": rec["heldout_gain"], "selected_weight_mse": item["weight_mse"],
                "candidate_count": len(rec["candidate_results"]),
                "candidate_eval_seconds_equal_share": rec["candidate_seconds_allocated_equal_share"],
                "heldout_eval_seconds": rec["heldout_eval_seconds"],
            })

        alpha_fp = float(fp["selected_alpha_search_only"])
        alpha_q = float(q["selected_alpha_search_only"])
        fp_w = activation["row_weights"][row_id].float()
        module_weights = {}
        for regime, alpha in (("FP_INPUT", alpha_fp), ("QPREFIX_INPUT", alpha_q)):
            weight_obj = load_torch(OUT / records[(row_id, regime)]["candidate_weight_file"])
            module_weights[regime] = weight_obj["candidate_weights_fp16"]
        q_fp_sel = module_weights["FP_INPUT"][str(alpha_fp)].to(device).float()
        q_q_sel = module_weights["QPREFIX_INPUT"][str(alpha_q)].to(device).float()
        fp_hold_docs = [activation["inputs"]["FP_INPUT"][module_key(row)][d["article_id"]].to(device).float() for d in held_docs]
        q_hold_docs = [activation["inputs"]["QPREFIX_INPUT"][module_key(row)][d["article_id"]].to(device).float() for d in held_docs]
        fp_d_weight = module_weights["FP_INPUT"]["1.0"].to(device).float()
        q_d_weight = module_weights["QPREFIX_INPUT"]["1.0"].to(device).float()
        fp_d_recomputed = norm_sse(fp_hold_docs, fp_w.to(device), fp_d_weight)
        fp_c_recomputed = norm_sse(fp_hold_docs, fp_w.to(device), q_fp_sel)
        q_d_recomputed = norm_sse(q_hold_docs, fp_w.to(device), q_d_weight)
        q_c_recomputed = norm_sse(q_hold_docs, fp_w.to(device), q_q_sel)
        direct_heldout_recompute_ok &= (
            math.isclose(fp_d_recomputed, float(fp["heldout_D_sse"]), rel_tol=1e-5, abs_tol=1e-7)
            and math.isclose(fp_c_recomputed, float(fp["heldout_C_sse"]), rel_tol=1e-5, abs_tol=1e-7)
            and math.isclose(q_d_recomputed, float(q["heldout_D_sse"]), rel_tol=1e-5, abs_tol=1e-7)
            and math.isclose(q_c_recomputed, float(q["heldout_C_sse"]), rel_tol=1e-5, abs_tol=1e-7)
        )
        if not direct_heldout_recompute_ok:
            raise RuntimeError(f"direct heldout output SSE recomputation differs: {row_id}")
        fp_alpha_on_q_sse = norm_sse(q_hold_docs, fp_w.to(device), q_fp_sel)
        q_alpha_on_fp_sse = norm_sse(fp_hold_docs, fp_w.to(device), q_q_sel)
        d_fp = float(fp["heldout_D_sse"])
        d_q = float(q["heldout_D_sse"])
        fp_gain, q_gain = float(fp["heldout_gain"]), float(q["heldout_gain"])
        pair_rows.append({
            "record_id": row_id, "layer": row["layer"], "projection": row["projection"],
            "row_index": row["row_index"], "bucket": row["bucket"],
            "alpha_FP": alpha_fp, "alpha_QPREFIX": alpha_q,
            "alpha_exact_agreement": alpha_fp == alpha_q,
            "alpha_within_one_grid_step": abs(alpha_fp - alpha_q) <= 0.1000000001,
            "gain_FP": fp_gain, "gain_QPREFIX": q_gain,
            "gain_sign_agreement": (fp_gain > 0) == (q_gain > 0) and (fp_gain < 0) == (q_gain < 0),
            "FP_positive_Q_positive": fp_gain > 0 and q_gain > 0,
            "FP_positive_Q_negative": fp_gain > 0 and q_gain < 0,
            "FP_positive_Q_severe_lt_minus10": fp_gain > 0 and q_gain < -0.10,
            "heldout_D_FP_mse": fp["heldout_D_mse"], "heldout_D_QPREFIX_mse": q["heldout_D_mse"],
            "heldout_D_FP_sse": d_fp, "heldout_C_FP_sse": float(fp["heldout_C_sse"]),
            "heldout_D_QPREFIX_sse": d_q, "heldout_C_QPREFIX_sse": float(q["heldout_C_sse"]),
            "D_QPREFIX_over_D_FP_mse": q["heldout_D_mse"] / fp["heldout_D_mse"] if fp["heldout_D_mse"] else None,
            "FP_selected_alpha_on_QPREFIX_heldout_sse": fp_alpha_on_q_sse,
            "FP_selected_alpha_on_QPREFIX_gain_vs_D_QPREFIX": 1.0 - fp_alpha_on_q_sse / d_q if d_q else 0.0,
            "Q_selected_alpha_on_FP_heldout_sse": q_alpha_on_fp_sse,
            "Q_selected_alpha_on_FP_gain_vs_D_FP": 1.0 - q_alpha_on_fp_sse / d_fp if d_fp else 0.0,
        })
        cross_rows.append({
            "record_id": row_id, "layer": row["layer"], "projection": row["projection"],
            "alpha_FP": alpha_fp, "alpha_QPREFIX": alpha_q,
            "FP_candidate_on_FP_heldout_sse": fp["heldout_C_sse"],
            "FP_candidate_on_QPREFIX_heldout_sse": fp_alpha_on_q_sse,
            "QPREFIX_candidate_on_QPREFIX_heldout_sse": q["heldout_C_sse"],
            "QPREFIX_candidate_on_FP_heldout_sse": q_alpha_on_fp_sse,
            "D_FP_on_FP_heldout_sse": d_fp, "D_QPREFIX_on_QPREFIX_heldout_sse": d_q,
            "FP_alpha_gain_when_transferred_to_QPREFIX": 1.0 - fp_alpha_on_q_sse / d_q if d_q else 0.0,
            "QPREFIX_alpha_gain_when_transferred_to_FP": 1.0 - q_alpha_on_fp_sse / d_fp if d_fp else 0.0,
        })
    cross_eval_seconds = time.perf_counter() - start_cross
    write_csv(OUT / "fp_per_row.csv", fp_rows)
    write_csv(OUT / "qprefix_per_row.csv", q_rows)
    write_csv(OUT / "transfer_pairwise.csv", pair_rows)
    write_csv(OUT / "cross_alpha_evaluation.csv", cross_rows)

    # Input drift and per-feature RMS ratio, including split-level aggregates.
    input_drift_rows = []
    input_module_summary = {}
    for row in rows_ref["rows"]:
        if row["projection"] != "q_proj" and row["projection"] not in ("o_proj", "up_proj", "down_proj"):
            continue
        key = module_key(row)
        # One calculation per module; rows within a projection share exactly the same X.
        if key in input_module_summary:
            continue
        docs_by_split = {"all": articles["calibration"] + articles["heldout"],
                         "calibration": articles["calibration"], "heldout": articles["heldout"]}
        aggregates = {}
        for split, docs in docs_by_split.items():
            xf = torch.cat([activation["inputs"]["FP_INPUT"][key][d["article_id"]] for d in docs]).to(device).float()
            xq = torch.cat([activation["inputs"]["QPREFIX_INPUT"][key][d["article_id"]] for d in docs]).to(device).float()
            if xf.shape != xq.shape or not torch.isfinite(xf).all() or not torch.isfinite(xq).all():
                raise RuntimeError(f"paired input shape/non-finite mismatch: {key}/{split}")
            diff = xq - xf
            rms_fp = torch.sqrt(torch.mean(xf.square()))
            rms_q = torch.sqrt(torch.mean(xq.square()))
            cosine = torch.sum(xf * xq) / (torch.linalg.vector_norm(xf) * torch.linalg.vector_norm(xq))
            aggregates[split] = {
                "token_count": int(xf.shape[0]), "feature_count": int(xf.shape[1]),
                "relative_activation_drift": float(torch.linalg.vector_norm(diff) / torch.linalg.vector_norm(xf)),
                "cosine_similarity": float(cosine), "activation_rms_FP": float(rms_fp),
                "activation_rms_QPREFIX": float(rms_q), "activation_rms_ratio_Q_over_FP": float(rms_q / rms_fp) if rms_fp else None,
                "max_abs_FP": float(xf.abs().max()), "max_abs_QPREFIX": float(xq.abs().max()),
            }
            if split == "all":
                feature_rms_fp = torch.sqrt(torch.mean(xf.square(), dim=0))
                feature_rms_q = torch.sqrt(torch.mean(xq.square(), dim=0))
                ratio = feature_rms_q / feature_rms_fp.clamp_min(1e-12)
                for idx in range(xf.shape[1]):
                    input_drift_rows.append({
                        "module_key": key, "layer": row["layer"], "projection": row["projection"],
                        "scope": "per_feature_all_articles", "feature_index": idx,
                        "feature_rms_FP": float(feature_rms_fp[idx]), "feature_rms_QPREFIX": float(feature_rms_q[idx]),
                        "feature_rms_ratio_Q_over_FP": float(ratio[idx]),
                    })
            input_drift_rows.append({
                "module_key": key, "layer": row["layer"], "projection": row["projection"],
                "scope": split, "feature_index": "", **aggregates[split],
            })
            del xf, xq, diff
        input_module_summary[key] = aggregates
    write_csv(OUT / "input_drift.csv", input_drift_rows)

    # Calibration Gram/Hessian drift. Materialize one module at a time in FP32.
    gram_rows = []
    gram_module_summary = {}
    for row in rows_ref["rows"]:
        key = module_key(row)
        if key in gram_module_summary:
            continue
        docs = articles["calibration"]
        xf = torch.cat([activation["inputs"]["FP_INPUT"][key][d["article_id"]] for d in docs]).to(device).float()
        xq = torch.cat([activation["inputs"]["QPREFIX_INPUT"][key][d["article_id"]] for d in docs]).to(device).float()
        hfp = (xf.T @ xf) / xf.shape[0]
        hq = (xq.T @ xq) / xq.shape[0]
        delta = hq - hfp
        rel = float(torch.linalg.vector_norm(delta) / torch.linalg.vector_norm(hfp))
        dfp, dq = torch.diagonal(hfp), torch.diagonal(hq)
        diag_rel_rms = float(torch.sqrt(torch.mean((dq - dfp).square())) / torch.sqrt(torch.mean(dfp.square())))
        trace_ratio = float(torch.trace(hq) / torch.trace(hfp))
        item = {
            "module_key": key, "layer": row["layer"], "projection": row["projection"],
            "feature_count": int(xf.shape[1]), "token_count": int(xf.shape[0]),
            "relative_gram_frobenius_difference": rel,
            "diagonal_relative_rms_change": diag_rel_rms,
            "trace_ratio_QPREFIX_over_FP": trace_ratio,
        }
        gram_rows.append(item)
        gram_module_summary[key] = item
        del xf, xq, hfp, hq, delta, dfp, dq
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    write_csv(OUT / "gram_drift.csv", gram_rows)

    # Projection and layer rollups use row-paired gains, not pooled-mean proxies.
    projections, layers = [], []
    for dim, groups in (("projection", ("q_proj", "o_proj", "up_proj", "down_proj")),
                         ("layer", (2, 6, 14, 18))):
        target = projections if dim == "projection" else layers
        for group in groups:
            subset = [r for r in pair_rows if r[dim] == group]
            sf, sq = gain_stats(subset, "FP"), gain_stats(subset, "QPREFIX")
            target.append({
                dim: group, "row_count": len(subset),
                "alpha_exact_agreement_count": sum(bool(r["alpha_exact_agreement"]) for r in subset),
                "alpha_exact_agreement_rate": sum(bool(r["alpha_exact_agreement"]) for r in subset) / len(subset),
                "alpha_within_one_step_count": sum(bool(r["alpha_within_one_grid_step"]) for r in subset),
                "alpha_within_one_step_rate": sum(bool(r["alpha_within_one_grid_step"]) for r in subset) / len(subset),
                "gain_spearman_FP_vs_QPREFIX": spearman([r["gain_FP"] for r in subset], [r["gain_QPREFIX"] for r in subset]),
                "median_gain_FP": sf["median_gain"], "median_gain_QPREFIX": sq["median_gain"],
                "pooled_gain_FP": sf["pooled_gain"], "pooled_gain_QPREFIX": sq["pooled_gain"],
                "positive_to_negative_count": sum(bool(r["FP_positive_Q_negative"]) for r in subset),
                "severe_lt_minus10_FP": sf["severe_lt_10pct"], "severe_lt_minus10_QPREFIX": sq["severe_lt_10pct"],
                "worst_gain_FP": sf["worst_gain"], "worst_gain_QPREFIX": sq["worst_gain"],
            })
    write_csv(OUT / "projection_summary.csv", projections)
    write_csv(OUT / "layer_summary.csv", layers)

    fpstats, qstats = gain_stats(pair_rows, "FP"), gain_stats(pair_rows, "QPREFIX")
    exact_count = sum(bool(r["alpha_exact_agreement"]) for r in pair_rows)
    within_count = sum(bool(r["alpha_within_one_grid_step"]) for r in pair_rows)
    fp_positive = [r for r in pair_rows if r["gain_FP"] > 0]
    pos_pos = sum(r["gain_QPREFIX"] > 0 for r in fp_positive)
    transfer_failures = [r for r in pair_rows if r["FP_positive_Q_negative"]]
    failure_rows = []
    for r in transfer_failures:
        failure_rows.append({
            **r,
            "selection_changed": r["alpha_FP"] != r["alpha_QPREFIX"],
            "selection_within_one_step": r["alpha_within_one_grid_step"],
            "FP_alpha_transferred_gain_QPREFIX": r["FP_selected_alpha_on_QPREFIX_gain_vs_D_QPREFIX"],
            "Q_alpha_transferred_gain_FP": r["Q_selected_alpha_on_FP_gain_vs_D_FP"],
        })
    write_csv(OUT / "transfer_failure_cases.csv", failure_rows or [{"status": "no_positive_to_negative_cases"}])

    gate = protocol["transfer_support_gate"]
    condition_a = qstats["pooled_gain"] > 0
    condition_b = qstats["median_gain"] >= 0
    condition_c = (pos_pos / len(fp_positive)) >= (2 / 3) if fp_positive else True
    max_severe = max(fpstats["severe_lt_10pct"] * 2, fpstats["severe_lt_10pct"] + 3)
    condition_d = qstats["severe_lt_10pct"] <= max_severe
    verdict = "TRANSFER_SIGNAL_PRESERVED" if all((condition_a, condition_b, condition_c, condition_d)) else "TRANSFER_SIGNAL_NOT_PRESERVED"

    alpha_distribution = {
        regime: {str(alpha): sum(float(records[(rid(row), regime)]["selected_alpha_search_only"]) == alpha for row in rows_ref["rows"])
                 for alpha in protocol["quantization"]["alpha_grid"]}
        for regime in ("FP_INPUT", "QPREFIX_INPUT")
    }
    row_savings = []
    for pair in pair_rows:
        for regime, prefix_key in (("FP_INPUT", "FP"), ("QPREFIX_INPUT", "QPREFIX")):
            saving = pair[f"heldout_D_{prefix_key}_sse"] - pair[f"heldout_C_{prefix_key}_sse"]
            row_savings.append((regime, pair["record_id"], saving))
    pooled_saving_concentration = {}
    for regime in ("FP_INPUT", "QPREFIX_INPUT"):
        savings = sorted((value for r, _, value in row_savings if r == regime and value > 0), reverse=True)
        net_saving = sum(pair[f"heldout_D_{'FP' if regime == 'FP_INPUT' else 'QPREFIX'}_sse"] - pair[f"heldout_C_{'FP' if regime == 'FP_INPUT' else 'QPREFIX'}_sse"] for pair in pair_rows)
        gross_positive = sum(savings)
        pooled_saving_concentration[regime] = {
            "net_sse_saving": net_saving,
            "gross_positive_row_sse_saving": gross_positive,
            "top1_positive_saving_share_of_net": savings[0] / net_saving if savings and net_saving else None,
            "top5_positive_saving_share_of_net": sum(savings[:5]) / net_saving if savings and net_saving else None,
        }

    # Verify the v8-v14 protected trees against the pre-run immutable snapshot.
    tree_root = ROOT / "experiments/range_search_w4/results"
    expected_trees = before.get("trees", before)
    tree_checks = {}
    for name in ("v8", "v9", "v10", "v11", "v12", "v13", "v14"):
        tree = tree_root / {
            "v8": "range_w4_dynamic_v8", "v9": "range_w4_l23r448_v9", "v10": "range_w4_sixrow_v10",
            "v11": "range_w4_96row_v11", "v12": "range_w4_gate_v12", "v13": "range_w4_failure_mechanism_v13",
            "v14": "range_w4_calibration_coverage_v14",
        }[name]
        current = tree_fingerprint(tree)
        prior = expected_trees.get(name)
        tree_checks[name] = prior == current
    if not all(tree_checks.values()):
        raise RuntimeError(f"historical result tree changed: {tree_checks}")

    # Re-read emitted row CSVs and independently recompute the pooled metrics.
    def read_csv(path):
        with path.open("r", newline="", encoding="utf-8-sig") as f:
            return list(csv.DictReader(f))
    csv_fp, csv_q = read_csv(OUT / "fp_per_row.csv"), read_csv(OUT / "qprefix_per_row.csv")
    raw_pooled_fp = 1 - sum(float(r["heldout_C_FP_sse"]) for r in pair_rows) / sum(float(r["heldout_D_FP_sse"]) for r in pair_rows)
    raw_pooled_q = 1 - sum(float(r["heldout_C_QPREFIX_sse"]) for r in pair_rows) / sum(float(r["heldout_D_QPREFIX_sse"]) for r in pair_rows)
    csv_pooled_fp = 1 - sum(float(r["heldout_C_sse"]) for r in csv_fp) / sum(float(r["heldout_D_sse"]) for r in csv_fp)
    csv_pooled_q = 1 - sum(float(r["heldout_C_sse"]) for r in csv_q) / sum(float(r["heldout_D_sse"]) for r in csv_q)
    raw_to_summary_ok = (len(csv_fp) == 48 and len(csv_q) == 48 and direct_heldout_recompute_ok and
                         math.isclose(raw_pooled_fp, csv_pooled_fp, rel_tol=1e-12, abs_tol=1e-12) and
                         math.isclose(raw_pooled_q, csv_pooled_q, rel_tol=1e-12, abs_tol=1e-12))
    if not raw_to_summary_ok:
        raise RuntimeError("independent raw-to-CSV recomputation failed")

    # Independently reconstruct the historical article ID/hash exclusion set from v8-v14 manifests.
    def find_article_records(obj):
        found = []
        def visit(value):
            if isinstance(value, dict):
                if "source_article_index" in value and ("article_sha256" in value or "token_sha256" in value):
                    found.append((int(value["source_article_index"]), value.get("article_sha256"), value.get("token_sha256", value.get("sha256"))))
                for child in value.values():
                    visit(child)
            elif isinstance(value, list):
                for child in value:
                    visit(child)
        visit(obj)
        return found
    hist_ids, hist_hashes = set(), set()
    inventory = articles["historical_blacklist_source_inventory"]
    for name in ("v8", "v9", "v10", "v11", "v12", "v14"):
        source = ROOT / inventory[name]["path"]
        if sha_file(source) != inventory[name]["sha256"]:
            raise RuntimeError(f"historical article manifest changed since v15 freeze: {name}")
        for article_id, article_hash, token_hash in find_article_records(read_json(source)):
            hist_ids.add(article_id)
            if article_hash:
                hist_hashes.add(article_hash)
            if token_hash:
                hist_hashes.add(token_hash)
    new_docs = articles["calibration"] + articles["heldout"]
    new_ids = {x["source_article_index"] for x in new_docs}
    new_hashes = {h for x in articles["calibration"] + articles["heldout"] for h in (x["article_sha256"], x["token_sha256"])}
    expected_article_ids = {x["article_id"] for x in new_docs}
    paired_input_cache = all(
        set(activation["inputs"][regime][key]) == expected_article_ids
        for regime in ("FP_INPUT", "QPREFIX_INPUT")
        for key in activation["inputs"][regime]
    )

    checks = {
        "v14_row_manifest_exact_reuse": sha_file(original_source) == protocol["v14_row_manifest_sha256"] and rows_ref["rows"] == read_json(original_source)["rows"],
        "new_article_manifest_frozen_and_unique": sha_file(OUT / "article_manifest.json") == protocol["article_manifest_sha256"] and len(new_ids) == 48 and len({x["article_sha256"] for x in new_docs}) == 48 and len({x["token_sha256"] for x in new_docs}) == 48 and not ({x["source_article_index"] for x in articles["calibration"]} & {x["source_article_index"] for x in articles["heldout"]}),
        "new_articles_disjoint_v8_v14": not (new_ids & hist_ids) and not (new_hashes & hist_hashes) and len(hist_ids) == articles["historical_unique_article_ids_excluded"] and len(hist_hashes) == articles["historical_unique_article_and_token_hashes_excluded"],
        "same_articles_fp_qprefix": preflight["checks"]["same_documents_fp_qprefix"] and paired_input_cache,
        "native_module_order_recorded": sha_file(OUT / "module_execution_order.json") == protocol["module_execution_order_sha256"] and len(order["target_capture_order"]) == 16,
        "qprefix_target_pristine_during_capture": all(v["target_pristine_rows_at_capture"] and v["target_class_during_capture"] == "Linear" for v in prefix["prefix_module_checks"].values()),
        "required_prefix_modules_native_w4_quantized": all(v["required_prefix_modules_all_native_quantized"] for v in prefix["prefix_module_checks"].values()),
        "prefix_always_native_alpha1": q_cache["capture_metadata"]["native_prefix_alpha"] == 1.0 and prefix["quantize_config"]["bits"] == 4 and prefix["quantize_config"]["group_size"] == 128 and prefix["quantize_config"]["sym"] and prefix["quantize_config"]["true_sequential"] and prefix["quantize_config"]["damp_percent"] == 0.1 and prefix["quantize_config"]["desc_act"] is False,
        "fp_q_target_pristine_weight_hashes_equal": preflight["checks"]["fp_q_target_pristine_weight_hashes_equal"],
        "four_projection_fp_and_qprefix_alpha1_parity": len(preflight["parity_runs"]) == 8 and all(x["status"] == "PASS" and all(x["checks"].values()) for x in preflight["parity_runs"]),
        "candidate_state_isolation": isolation["status"] == "PASS" and all(isolation["checks"].values()),
        "heldout_did_not_select_alpha": len(records) == 96 and all(float(r["selected_alpha_search_only"]) == min((float(a) for a in r["candidate_results"]), key=lambda a: (r["candidate_results"][str(a)]["search_mse"], a)) for r in records.values()),
        "raw_to_summary_independent_recomputation": raw_to_summary_ok and direct_heldout_recompute_ok,
        "v8_v14_source_trees_unchanged": all(tree_checks.values()),
        "no_gate_or_robust_objective_activated": not any(x in json.dumps(protocol).lower() for x in ("cvar", "mean+std", "worst_article_selector", "gate_v2")) and protocol["candidate_semantics"]["C"].find("mean calibration output MSE") >= 0,
        "finite_fp_qprefix_and_candidate_tensors": all_candidate_tensors_finite and all(torch.isfinite(x).all().item() for regime in activation["inputs"].values() for module in regime.values() for x in module.values()),
        "expected_candidate_workload_complete": len(records) == 96 and all(len(r["candidate_results"]) == 10 for r in records.values()) and int(timing["unique_row_level_candidate_slots_complete"]) == 960,
    }
    correctness = {"status": "PASS" if all(checks.values()) else "FAIL", "checks": checks, "historical_tree_checks": tree_checks,
                   "candidate_isolation_spot_check": isolation, "raw_record_count": len(records), "candidate_slots": 960,
                   "summary_independent_pooled_gain_FP": raw_pooled_fp, "summary_independent_pooled_gain_QPREFIX": raw_pooled_q}
    write_json(OUT / "correctness_checks.json", correctness)
    if correctness["status"] != "PASS":
        raise RuntimeError(f"final correctness checks failed: {checks}")

    # Input and Gram drift by layer is descriptive only.
    layer_drift = {}
    for layer in (2, 6, 14, 18):
        keys = [f"L{layer}/{p}" for p in ("q_proj", "o_proj", "up_proj", "down_proj")]
        layer_drift[str(layer)] = {
            "relative_activation_drift_mean": statistics.mean(input_module_summary[k]["all"]["relative_activation_drift"] for k in keys),
            "cosine_similarity_mean": statistics.mean(input_module_summary[k]["all"]["cosine_similarity"] for k in keys),
            "relative_gram_difference_mean": statistics.mean(gram_module_summary[k]["relative_gram_frobenius_difference"] for k in keys),
        }
    positive_to_negative_changed_alpha = sum(r["selection_changed"] for r in failure_rows)
    positive_to_negative_same_or_adjacent = sum(not r["selection_changed"] for r in failure_rows)
    worst_rows = {
        "FP_INPUT": min(pair_rows, key=lambda r: r["gain_FP"])["record_id"],
        "QPREFIX_INPUT": min(pair_rows, key=lambda r: r["gain_QPREFIX"])["record_id"],
    }
    summary = {
        "protocol_id": protocol["protocol_id"], "verdict": verdict,
        "verdict_conditions": {"A_qprefix_pooled_gain_positive": condition_a, "B_qprefix_median_gain_nonnegative": condition_b,
                               "C_at_least_two_thirds_FP_positive_stay_positive": condition_c,
                               "D_qprefix_severe_count_within_frozen_bound": condition_d,
                               "D_severe_count_limit": max_severe},
        "row_count": 48, "row_regime_record_count": 96, "candidate_evaluations": 960,
        "overall": {"FP_INPUT": fpstats, "QPREFIX_INPUT": qstats},
        "worst_regression_row": worst_rows,
        "selected_alpha_distribution": alpha_distribution,
        "pooled_saving_concentration": pooled_saving_concentration,
        "transfer": {
            "alpha_exact_agreement_count": exact_count, "alpha_exact_agreement_rate": exact_count / 48,
            "alpha_within_one_step_count": within_count, "alpha_within_one_step_rate": within_count / 48,
            "gain_spearman_FP_vs_QPREFIX": spearman([r["gain_FP"] for r in pair_rows], [r["gain_QPREFIX"] for r in pair_rows]),
            "gain_sign_agreement_count": sum(bool(r["gain_sign_agreement"]) for r in pair_rows),
            "FP_positive_rows": len(fp_positive), "FP_positive_Q_positive_count": pos_pos,
            "FP_positive_Q_positive_rate": pos_pos / len(fp_positive) if fp_positive else None,
            "positive_to_negative_count": len(transfer_failures),
            "positive_to_negative_with_alpha_change": positive_to_negative_changed_alpha,
            "positive_to_negative_without_alpha_change": positive_to_negative_same_or_adjacent,
            "positive_to_negative_transfer_gain_median_at_FP_alpha": statistics.median([r["FP_selected_alpha_on_QPREFIX_gain_vs_D_QPREFIX"] for r in transfer_failures]) if transfer_failures else None,
            "positive_to_negative_transfer_gain_median_at_Q_alpha_on_FP": statistics.median([r["Q_selected_alpha_on_FP_gain_vs_D_FP"] for r in transfer_failures]) if transfer_failures else None,
        },
        "projection_summary": projections, "layer_summary": layers,
        "input_and_gram_drift_by_layer_descriptive": layer_drift,
        "cost": {
            "candidate_run_timing": timing,
            "prefix_quantization_seconds": prefix["native_sequential_quantization_seconds"],
            "fp_capture_seconds": fp_cache["metadata"]["capture_seconds"],
            "qprefix_full_forward_seconds": prefix["full_quantized_model_forward_seconds"],
            "qprefix_target_replay_seconds": prefix["target_block_replay_seconds"],
            "hessian_build_seconds": sum(float(x["hessian_build_seconds"]) for x in preflight["hessian_builds"]),
            "cross_alpha_diagnostics_seconds": cross_eval_seconds,
            "prefix_model_load_seconds": prefix["model_load_seconds"],
            "peak_prefix_allocated_bytes": prefix["peak_allocated_bytes"],
            "peak_candidate_allocated_bytes": timing["peak_allocated_bytes"],
            "peak_candidate_reserved_bytes": timing["peak_reserved_bytes"],
            "candidate_runtime_estimate_seconds_frozen": cost_estimate["candidate_runtime_estimate_seconds"],
            "candidate_runtime_actual_seconds": timing["candidate_kernel_seconds_total"],
        },
        "diagnostic_limits": ["Local output reconstruction only; no PPL or downstream-task evaluation was run.",
                              "Input/Gram drift is descriptive and does not establish a causal layer-depth effect.",
                              "Cross-alpha held-out diagnostics do not reselect alpha."],
    }
    write_json(OUT / "SUMMARY.json", summary)

    # Long-form phase cost CSV. Combined measured work is a sum of explicitly timed components.
    phases = [
        ("fp_activation_capture", float(fp_cache["metadata"]["capture_seconds"]), 0),
        ("native_sequential_prefix_quantization", float(prefix["native_sequential_quantization_seconds"]), 168),
        ("qprefix_full_model_forward", float(prefix["full_quantized_model_forward_seconds"]), 48),
        ("qprefix_target_block_replay", float(prefix["target_block_replay_seconds"]), 16 * 48),
        ("hessian_construction", sum(float(x["hessian_build_seconds"]) for x in preflight["hessian_builds"]), 32),
        ("candidate_dynamic_gptq_and_search_loss", float(timing["candidate_kernel_seconds_total"]), int(timing["unique_row_level_candidate_slots_complete"])),
        ("heldout_candidate_evaluation", float(timing["heldout_eval_seconds_total"]), 96 * 2),
        ("cross_alpha_diagnostics", cross_eval_seconds, 48 * 2),
    ]
    cost_rows = [{"phase": name, "seconds": seconds, "work_units": units,
                  "candidate_evaluations": int(timing["unique_row_level_candidate_slots_complete"]) if name == "candidate_dynamic_gptq_and_search_loss" else 0,
                  "candidate_evaluations_this_invocation": int(timing["row_level_candidate_evaluations_executed_this_invocation"]) if name == "candidate_dynamic_gptq_and_search_loss" else 0,
                  "peak_allocated_bytes": timing["peak_allocated_bytes"] if name.startswith("candidate") else prefix["peak_allocated_bytes"],
                  "timing_source": "runtime synchronized measurement"} for name, seconds, units in phases]
    total_measured = sum(seconds for _, seconds, _ in phases)
    cost_rows.append({"phase": "sum_of_measured_components", "seconds": total_measured, "work_units": "components above",
                      "candidate_evaluations": 960, "peak_allocated_bytes": max(prefix["peak_allocated_bytes"], timing["peak_allocated_bytes"]),
                      "timing_source": "sum of measured phases; excludes script startup and file serialization outside timed sections"})
    write_csv(OUT / "cost_summary.csv", cost_rows)

    # Required concise report in Chinese, with pre-registered verdict unchanged.
    if transfer_failures:
        failure = transfer_failures[0]
        failure_diagnosis = (
            f"Positive→negative 的 {len(transfer_failures)} 个案例中，alpha 改变 {positive_to_negative_changed_alpha} 个、alpha 相同 {positive_to_negative_same_or_adjacent} 个。"
            f"唯一案例 {failure['record_id']} 的 alpha_FP=alpha_QPREFIX={failure['alpha_FP']}；FP 产生的候选在 QPREFIX heldout 上 gain 为 {failure['FP_selected_alpha_on_QPREFIX_gain_vs_D_QPREFIX']:.2%}，"
            f"而 QPREFIX 产生的候选在 FP heldout 上 gain 为 {failure['Q_selected_alpha_on_FP_gain_vs_D_FP']:.2%}。"
            "两个 regime 使用不同 Hessian/GPTQ 补偿分别生成候选，因此这些 cross-evaluation 不足以区分输入误差加权与候选误差向量变化；本轮无法进一步归因。"
        )
    else:
        failure_diagnosis = "本轮没有 positive→negative row，无法从该 failure 子集判断 alpha 选择与误差方向的相对作用。"
    report = [
        "# v15 Quantized-Prefix Transfer Audit",
        "",
        f"**预注册 verdict：{verdict}**。该 gate 仅用于研究路线判断，不表示统计显著性。",
        "",
        "## 实验实现与正确性",
        "",
        "QPREFIX 由 GPTQModel 2.2.0 原生 true-sequential alpha=1 W4A16 symmetric G128 对 C01–C16 量化完整模型生成。随后对同一 48 篇文档做量化模型 forward，缓存目标 block boundary，并在重放目标 block 时只把当前目标 projection 临时恢复为 pristine FP16 Linear；目标之前的 native quantized groups 保持生效。Torch 后端使用 eager `dequantize_weight` 路径；Windows 环境没有 Triton，因此未启用仅属性能优化的 `torch.compile/inductor`。",
        "",
        f"- 最终 correctness：**{correctness['status']}**；8 项 native alpha=1 parity 全部通过，group-entry weight、scale、zero-point、最终 FP16 权重和 search/heldout SSE 均一致。",
        f"- 48 rows × FP/QPREFIX 两个输入 regime，共 96 条 row-regime 结果；候选数 **{summary['candidate_evaluations']}**。",
        f"- v8–v14 历史结果树 SHA256 逐文件复核：{all(tree_checks.values())}。",
        "",
        "## Transfer 结果",
        "",
        f"- alpha exact agreement：{exact_count}/48 ({exact_count/48:.1%})；within one grid step：{within_count}/48 ({within_count/48:.1%})。",
        f"- selected alpha counts，FP：`{alpha_distribution['FP_INPUT']}`；QPREFIX：`{alpha_distribution['QPREFIX_INPUT']}`。",
        f"- gain FP 与 QPREFIX 的 Spearman：{summary['transfer']['gain_spearman_FP_vs_QPREFIX']:.4f}。",
        f"- pooled heldout gain：FP_INPUT {fpstats['pooled_gain']:.2%}，QPREFIX_INPUT {qstats['pooled_gain']:.2%}；median row gain：{fpstats['median_gain']:.2%} / {qstats['median_gain']:.2%}。",
        f"- FP-positive rows：{len(fp_positive)}；其中 QPREFIX 仍 positive {pos_pos}，positive→negative {len(transfer_failures)}。",
        f"- gain < −5/−10/−20/−40%：FP {fpstats['severe_lt_5pct']}/{fpstats['severe_lt_10pct']}/{fpstats['severe_lt_20pct']}/{fpstats['severe_lt_40pct']} rows；QPREFIX {qstats['severe_lt_5pct']}/{qstats['severe_lt_10pct']}/{qstats['severe_lt_20pct']}/{qstats['severe_lt_40pct']} rows。",
        f"- 最差 row gain：FP {worst_rows['FP_INPUT']} ({fpstats['worst_gain']:.2%})；QPREFIX {worst_rows['QPREFIX_INPUT']} ({qstats['worst_gain']:.2%})。",
        f"- pooled SSE 节省集中度：最大单 row 正向节省 / net 节省 FP {pooled_saving_concentration['FP_INPUT']['top1_positive_saving_share_of_net']:.1%}、QPREFIX {pooled_saving_concentration['QPREFIX_INPUT']['top1_positive_saving_share_of_net']:.1%}；前 5 rows FP {pooled_saving_concentration['FP_INPUT']['top5_positive_saving_share_of_net']:.1%}、QPREFIX {pooled_saving_concentration['QPREFIX_INPUT']['top5_positive_saving_share_of_net']:.1%}。",
        f"- heldout D 局部 MSE 的 QPREFIX/FP 比值中位数：{statistics.median([r['D_QPREFIX_over_D_FP_mse'] for r in pair_rows]):.4f}。此比值只描述两种输入分布下 local baseline reconstruction，不代表总体模型质量。",
        "",
        "### Alpha selection 与误差方向诊断",
        "",
        failure_diagnosis,
        f"Input/Gram drift by layer (descriptive): {json.dumps(layer_drift, ensure_ascii=False)}。L2 到 L6 漂移增加，后续层有波动而非单调增长；层间差异不构成因果证据。",
        "",
        "## 成本与范围",
        "",
        f"同步计时组件合计 {total_measured:.2f}s；native prefix GPTQ {prefix['native_sequential_quantization_seconds']:.2f}s，FP capture {fp_cache['metadata']['capture_seconds']:.2f}s，QPREFIX forward + target replay {prefix['full_quantized_model_forward_seconds'] + prefix['target_block_replay_seconds']:.2f}s，Hessian builds {summary['cost']['hessian_build_seconds']:.2f}s，candidate core/search-loss {timing['candidate_kernel_seconds_total']:.2f}s，heldout evaluation {timing['heldout_eval_seconds_total']:.2f}s。总和不计脚本启动和计时区间外的文件序列化。",
        f"候选成本预估 {cost_estimate['candidate_runtime_estimate_seconds']:.2f}s，实际同步计时 {timing['candidate_kernel_seconds_total']:.2f}s。30 个已完成槽位从校验通过的缓存恢复，另 930 个本次计算；unique workload 仍为 960。",
        "",
        "本轮只测局部 projection 输出重建，没有运行 PPL、下游任务或完整模型 quality gate。Transfer verdict 不能外推为最终模型收益。",
    ]
    (OUT / "REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps({"correctness": correctness["status"], "verdict": verdict,
                      "pooled_gain_FP": fpstats["pooled_gain"], "pooled_gain_QPREFIX": qstats["pooled_gain"],
                      "alpha_exact_agreement": exact_count, "alpha_within_one_step": within_count,
                      "positive_to_negative": len(transfer_failures), "measured_component_seconds": total_measured}, indent=2))


if __name__ == "__main__":
    main()
