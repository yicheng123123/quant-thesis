"""Read-only failure-mechanism audit of frozen v12 GPTQ candidate artifacts.

Run with .venv-gptq/Scripts/python.exe analyze_v13.py. No GPTQ is invoked.
All writes are confined to this script's directory.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import time
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
import torch
from scipy.stats import rankdata, spearmanr


HERE = Path(__file__).resolve().parent
V12 = HERE.parent / "range_w4_gate_v12"
RAW = V12 / "raw_validation"
QDIR = RAW / "candidate_weights"
GATE_SHA = "7943b8f26e5f94456b7b00144f6d47fc775d0224ccca4a6a1eff38e5740999c9"
GRID = [round(i / 10, 1) for i in range(1, 11)]
ABS_TOL = 1e-8
REL_TOL = 1e-5


def digest_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def tensor_hash(t: torch.Tensor) -> str:
    return hashlib.sha256(t.detach().contiguous().cpu().numpy().tobytes()).hexdigest()


def load_pt(path: Path) -> dict:
    with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
        return torch.load(path, map_location="cpu", weights_only=True)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def save_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"No rows for {path.name}")
    fields = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def check_close(actual: float, expected: float, label: str) -> dict:
    err = abs(actual - expected)
    tol = ABS_TOL + REL_TOL * abs(expected)
    if not math.isfinite(actual) or err > tol:
        raise AssertionError(f"{label}: actual={actual:.12g}, expected={expected:.12g}, abs_err={err:.5g}, tol={tol:.5g}")
    return {"label": label, "actual": actual, "expected": expected, "abs_error": err, "tolerance": tol}


def source_audit(raws: list[dict], summary: dict, gate: dict) -> tuple[dict, dict[str, bool]]:
    assert len(raws) == 480
    by_row = defaultdict(list)
    for r in raws:
        by_row[r["row_id"]].append(r)
    assert len(by_row) == 48 and all(len(v) == 10 for v in by_row.values())
    accept = {}
    margins = {}
    for row, rs in by_row.items():
        gains, ms = [], []
        for r in rs:
            ordered = sorted(r["candidates"], key=lambda c: (c["search_sse"], c["alpha"]))
            d = next(c for c in ordered if c["alpha"] == 1.0)
            gains.append(1 - ordered[0]["search_sse"] / d["search_sse"])
            ms.append((ordered[1]["search_sse"] - ordered[0]["search_sse"]) / ordered[1]["search_sse"])
        accept[row] = bool(np.median(gains) >= gate["tau_gain"] and np.median(ms) >= gate["tau_margin"])
        margins[row] = {"search_gain_median": float(np.median(gains)), "search_margin_median": float(np.median(ms))}
    dsum = sum(r["heldout_D_sse"] for r in raws)
    csum = sum(r["heldout_C_sse"] for r in raws)
    gsum = sum(r["heldout_C_sse"] if accept[r["row_id"]] else r["heldout_D_sse"] for r in raws)
    cgains = [1 - r["heldout_C_sse"] / r["heldout_D_sse"] for r in raws]
    ggains = [cgains[i] if accept[r["row_id"]] else 0.0 for i, r in enumerate(raws)]
    worst = min(raws, key=lambda r: 1 - r["heldout_C_sse"] / r["heldout_D_sse"])
    metrics = {"C_pooled_gain": 1 - csum / dsum, "Gate_pooled_gain": 1 - gsum / dsum,
               "C_severe_count": sum(x < -.1 for x in cgains),
               "Gate_severe_count": sum(x < -.1 for x in ggains),
               "C_catastrophic_count": sum(x < -.4 for x in cgains),
               "Gate_retained_gain": (dsum - gsum) / (dsum - csum),
               "accepted_rows": sum(accept.values()), "fallback_rows": sum(not x for x in accept.values()),
               "worst_row_id": worst["row_id"], "worst_split_id": worst["split_id"],
               "worst_gain": min(cgains), "D_heldout_sse": dsum, "C_heldout_sse": csum, "Gate_heldout_sse": gsum}
    saved = summary["metrics"]
    for key, val in [("C_pooled_gain", saved["C"]["pooled_gain_vs_D"]),
                     ("Gate_pooled_gain", saved["G"]["pooled_gain_vs_D"]),
                     ("Gate_retained_gain", saved["G"]["retained_C_gain"])]:
        check_close(metrics[key], val, f"source audit {key}")
    assert metrics["C_severe_count"] == saved["C"]["gain_lt_minus_10pct"] == 69
    assert metrics["Gate_severe_count"] == saved["G"]["gain_lt_minus_10pct"] == 40
    assert metrics["C_catastrophic_count"] == 20
    assert metrics["accepted_rows"] == summary["accepted_rows"] == 25
    assert metrics["worst_row_id"] == "L20_o_proj_R287" and metrics["worst_split_id"] == "S08"
    return metrics, accept


def input_tensors(cache: dict, rec: dict, split: dict, side: str) -> torch.Tensor:
    key = f'L{rec["layer"]}/{rec["projection"]}'
    docs = [cache["inputs"][key][a["token_sha256"]] for a in split[side]]
    assert len(docs) == 4 and all(d.shape[0] == 128 for d in docs)
    return torch.cat(docs, dim=0).to("cuda", dtype=torch.float32)


def gram_shift_ratio(xs: torch.Tensor, xh: torch.Tensor) -> float:
    # Frobenius identity avoids constructing a d x d Gram matrix.
    ns, nh = xs.shape[0], xh.shape[0]
    ss = xs @ xs.T
    hh = xh @ xh.T
    sh = xs @ xh.T
    norm_s2 = (ss.double().square().sum() / ns**2).item()
    norm_h2 = (hh.double().square().sum() / nh**2).item()
    inner = (sh.double().square().sum() / (ns * nh)).item()
    delta2 = norm_s2 + norm_h2 - 2 * inner
    assert delta2 >= -1e-7 * max(norm_s2, norm_h2)
    return math.sqrt(max(delta2, 0) / norm_s2)


def token_stats(vals: np.ndarray) -> dict:
    assert len(vals) == 512 and np.all(vals >= 0)
    total = float(vals.sum())
    ordered = np.sort(vals)[::-1]
    article_sse = vals.reshape(4, 128).sum(axis=1)
    return {"mean": float(vals.mean()), "std": float(vals.std()),
            "p90": float(np.percentile(vals, 90)), "p95": float(np.percentile(vals, 95)),
            "p99": float(np.percentile(vals, 99)), "max": float(vals.max()),
            "top1pct_sse_share": float(ordered[:math.ceil(.01 * len(vals))].sum() / total),
            "top5pct_sse_share": float(ordered[:math.ceil(.05 * len(vals))].sum() / total),
            "top10pct_sse_share": float(ordered[:math.ceil(.10 * len(vals))].sum() / total),
            "top1article_sse_share": float(article_sse.max() / total),
            "top2articles_sse_share": float(np.sort(article_sse)[-2:].sum() / total),
            **{f"article{i+1}_sse": float(article_sse[i]) for i in range(4)},
            **{f"article{i+1}_sse_share": float(article_sse[i] / total) for i in range(4)}}


def rank_metrics(article_losses: np.ndarray, aggregate: np.ndarray) -> dict:
    ranks = np.array([rankdata(v, method="average") for v in article_losses])
    arank = rankdata(aggregate, method="average")
    pairs = [float(spearmanr(ranks[i], ranks[j]).statistic) for i, j in combinations(range(4), 2)]
    agg_corrs = [float(spearmanr(r, arank).statistic) for r in ranks]
    bests = np.argmin(article_losses, axis=1)
    aggregate_best = int(np.argmin(aggregate))
    top3 = [set(np.argsort(v, kind="stable")[:3]) for v in article_losses]
    overlaps = [len(top3[i] & top3[j]) / len(top3[i] | top3[j]) for i, j in combinations(range(4), 2)]
    return {"pairwise_spearman_mean": float(np.mean(pairs)), "pairwise_spearman_min": float(np.min(pairs)),
            "pairwise_spearman_std": float(np.std(pairs)),
            "aggregate_article_spearman_mean": float(np.mean(agg_corrs)),
            "aggregate_article_spearman_min": float(np.min(agg_corrs)),
            "article_best_alpha_agreement_pairs": int(sum(bests[i] == bests[j] for i, j in combinations(range(4), 2))),
            "articles_best_agree_aggregate": int(sum(b == aggregate_best for b in bests)),
            "pairwise_top3_jaccard_mean": float(np.mean(overlaps)),
            "article_best_alphas": ";".join(str(GRID[i]) for i in bests),
            "article_aggregate_spearman": ";".join(f"{x:.6g}" for x in agg_corrs)}


def analyze_one(rec: dict, split: dict, cache: dict, gram_ratio: float, xs: torch.Tensor, xh: torch.Tensor,
                do_all: bool = True) -> dict:
    rid, sid = rec["row_id"], rec["split_id"]
    w = cache["weights"][rid].to("cuda", dtype=torch.float32)
    qobj = load_pt(QDIR / f"{rid}_{sid}.pt")
    assert qobj["row_manifest_sha256"] == rec["row_manifest_sha256"]
    assert qobj["split_manifest_sha256"] == rec["split_manifest_sha256"]
    assert tensor_hash(cache["weights"][rid]) == rec["weight_sha256"]
    qc = [qobj["weights"][str(a)] for a in GRID]
    assert all(tensor_hash(q) == c["weight_sha256"] for q, c in zip(qc, rec["candidates"]))
    q = torch.stack(qc, dim=1).to("cuda", dtype=torch.float32)
    err = q - w[:, None]
    pred_s = xs @ err
    pred_h = xh @ err[:, [9, GRID.index(rec["selected_alpha_on_search"])]]
    search_loss = pred_s.square().mean(dim=0).double().cpu().numpy()
    held_loss = pred_h.square().mean(dim=0).double().cpu().numpy()
    selected_index = GRID.index(rec["selected_alpha_on_search"])
    e_dc = err[:, [9, selected_index]]
    # Independent contraction order for e^T H e, without materializing H.
    hs_e = (xs.T @ (xs @ e_dc)) / xs.shape[0]
    hh_e = (xh.T @ (xh @ e_dc)) / xh.shape[0]
    quad_s = (e_dc * hs_e).sum(dim=0).double().cpu().numpy()
    quad_h = (e_dc * hh_e).sum(dim=0).double().cpu().numpy()
    direct_directional = (e_dc * (hh_e - hs_e)).sum(dim=0).double().cpu().numpy()
    d_s, c_s = map(float, quad_s)
    d_h, c_h = map(float, quad_h)
    checks = [check_close(d_s, rec["search_D_mse"], f"{rid}/{sid}/search/D"),
              check_close(c_s, rec["search_C_mse"], f"{rid}/{sid}/search/C"),
              check_close(d_h, rec["heldout_D_mse"], f"{rid}/{sid}/held/D"),
              check_close(c_h, rec["heldout_C_mse"], f"{rid}/{sid}/held/C"),
              check_close(d_s, float(search_loss[9]), f"{rid}/{sid}/search/D/direct"),
              check_close(c_s, float(search_loss[selected_index]), f"{rid}/{sid}/search/C/direct"),
              check_close(d_h, float(held_loss[0]), f"{rid}/{sid}/held/D/direct"),
              check_close(c_h, float(held_loss[1]), f"{rid}/{sid}/held/C/direct")]
    # Compare scalar differences with a separately contracted H_h-H_s expression.
    a_s, a_h = d_s - c_s, d_h - c_h
    shift_d, shift_c = d_h - d_s, c_h - c_s
    checks.extend([check_close(shift_d, float(direct_directional[0]), f"{rid}/{sid}/directional/D"),
                   check_close(shift_c, float(direct_directional[1]), f"{rid}/{sid}/directional/C"),
                   check_close(a_h - a_s, float(direct_directional[0] - direct_directional[1]),
                               f"{rid}/{sid}/shift_identity")])
    selected = rec["selected_alpha_on_search"]
    base = {"row_id": rid, "split_id": sid, "layer": rec["layer"], "projection": rec["projection"],
            "selected_alpha": selected, "search_gain": rec["search_gain"], "heldout_gain": rec["heldout_gain_C"],
            "gate_accept": None, "e_D_norm": float(torch.linalg.vector_norm(err[:, 9]).item()),
            "e_C_norm": float(torch.linalg.vector_norm(err[:, selected_index]).item()),
            "qD_search": d_s, "qC_search": c_s, "qD_heldout": d_h, "qC_heldout": c_h,
            "A_search": a_s, "A_heldout": a_h, "generalization_shift": a_h - a_s,
            "directional_shift_D": shift_d, "directional_shift_C": shift_c,
            "directional_shift_D_minus_C": shift_d - shift_c,
            "relative_global_gram_shift_F": gram_ratio}
    if not do_all:
        return {"base": base, "checks": checks}
    pred_arr = pred_s.square().cpu().numpy()
    art_loss = pred_arr.reshape(4, 128, 10).mean(axis=1).astype(np.float64)
    assert np.allclose(art_loss.mean(axis=0), search_loss, atol=ABS_TOL, rtol=REL_TOL)
    landscapes = []
    for ai, alpha in enumerate(GRID):
        losses = art_loss[:, ai]
        gains = 1 - losses / art_loss[:, 9]
        landscapes.append({"row_id": rid, "split_id": sid, "layer": rec["layer"], "projection": rec["projection"],
                           "alpha": alpha, "is_search_best": alpha == selected,
                           "aggregate_search_mse": float(search_loss[ai]),
                           **{f"article{j+1}_id": rec["search_article_ids"][j] for j in range(4)},
                           **{f"article{j+1}_mse": float(losses[j]) for j in range(4)},
                           **{f"article{j+1}_gain": float(gains[j]) for j in range(4)},
                           "article_mse_mean": float(losses.mean()), "article_mse_std": float(losses.std()),
                           "article_mse_min": float(losses.min()), "article_mse_max": float(losses.max()),
                           "article_mse_p25": float(np.percentile(losses, 25)),
                           "article_mse_p75": float(np.percentile(losses, 75)),
                           "article_gain_mean": float(gains.mean()), "article_gain_std": float(gains.std()),
                           "article_gain_min": float(gains.min()), "article_gain_max": float(gains.max()),
                           "article_gain_p25": float(np.percentile(gains, 25)),
                           "article_gain_p75": float(np.percentile(gains, 75)),
                           "articles_improved": int((gains > 0).sum()), "articles_regressed": int((gains < 0).sum())})
    rank = {**{k: base[k] for k in ["row_id", "split_id", "layer", "projection", "selected_alpha", "search_gain", "heldout_gain"]},
            **rank_metrics(art_loss, search_loss),
            "selected_articles_improved": landscapes[GRID.index(selected)]["articles_improved"],
            "selected_articles_regressed": landscapes[GRID.index(selected)]["articles_regressed"],
            "selected_article_gain_std": landscapes[GRID.index(selected)]["article_gain_std"],
            "selected_article_gain_min": landscapes[GRID.index(selected)]["article_gain_min"],
            "selected_article_gain_max": landscapes[GRID.index(selected)]["article_gain_max"]}
    tokens = []
    for side, pred in [("search", pred_s[:, [9, GRID.index(selected)]]), ("heldout", pred_h)]:
        vals = pred.square().cpu().numpy()
        for i, method in enumerate(["D", "C"]):
            tokens.append({"row_id": rid, "split_id": sid, "layer": rec["layer"],
                           "projection": rec["projection"], "side": side, "method": method,
                           "selected_alpha": selected, "heldout_gain": rec["heldout_gain_C"],
                           **token_stats(vals[:, i].astype(np.float64))})
    return {"base": base, "checks": checks, "landscapes": landscapes, "rank": rank, "tokens": tokens}


def median_iqr(x: list[float]) -> dict:
    a = np.asarray(x, dtype=np.float64)
    return {"median": float(np.median(a)), "p25": float(np.percentile(a, 25)),
            "p75": float(np.percentile(a, 75)), "iqr": float(np.percentile(a, 75) - np.percentile(a, 25))}


def features_summary(base: list[dict], ranks: list[dict], tokens: list[dict]) -> list[dict]:
    rank_by = {(r["row_id"], r["split_id"]): r for r in ranks}
    tok_by = {(r["row_id"], r["split_id"], r["side"], r["method"]): r for r in tokens}
    groups = {"robust_success_gain_ge_10pct": lambda r: r["heldout_gain"] >= .1,
              "mild_or_neutral_minus10_to_10pct": lambda r: -.1 <= r["heldout_gain"] < .1,
              "severe_lt_minus10pct": lambda r: r["heldout_gain"] < -.1,
              "catastrophic_lt_minus40pct": lambda r: r["heldout_gain"] < -.4}
    feats = ["search_gain", "e_C_norm", "e_D_norm", "relative_global_gram_shift_F",
             "directional_shift_C", "directional_shift_D", "directional_shift_D_minus_C",
             "generalization_shift", "pairwise_spearman_mean", "pairwise_top3_jaccard_mean",
             "articles_best_agree_aggregate", "selected_articles_improved", "selected_article_gain_std",
             "selected_article_gain_min", "heldout_C_top1pct_sse_share", "heldout_C_top1article_sse_share"]
    full = []
    for b in base:
        key = (b["row_id"], b["split_id"])
        full.append({**b, **rank_by[key],
                     "heldout_C_top1pct_sse_share": tok_by[key + ("heldout", "C")]["top1pct_sse_share"],
                     "heldout_C_top1article_sse_share": tok_by[key + ("heldout", "C")]["top1article_sse_share"]})
    rows = []
    for feat in feats:
        vals = [float(r[feat]) for r in full]
        correlation = spearmanr(vals, [r["heldout_gain"] for r in full])
        for group, predicate in groups.items():
            sub = [float(r[feat]) for r in full if predicate(r)]
            rows.append({"feature": feat, "outcome": "heldout_gain_C", "group": group,
                         "n": len(sub), **median_iqr(sub),
                         "all_480_spearman_vs_heldout_gain": float(correlation.statistic),
                         "all_480_spearman_pvalue_descriptive_only": float(correlation.pvalue)})
    return rows


def main() -> None:
    started = time.perf_counter()
    torch.backends.cuda.matmul.allow_tf32 = False
    assert torch.cuda.is_available(), "CUDA required to replay saved FP32 output-error arithmetic"
    source_files = ["GATE_V1.json", "VALIDATION_SUMMARY.json", "VALIDATION_CORRECTNESS.json",
                    "VALIDATION_PROTOCOL.json", "validation_row_manifest.json", "validation_split_manifest.json"]
    hashes = {n: digest_file(V12 / n) for n in source_files}
    assert hashes["GATE_V1.json"] == GATE_SHA
    gate = load_json(V12 / "GATE_V1.json")
    summary = load_json(V12 / "VALIDATION_SUMMARY.json")
    row_manifest = load_json(V12 / "validation_row_manifest.json")
    split_manifest = load_json(V12 / "validation_split_manifest.json")
    assert summary["status"] == "GATE_V1_VALIDATION_FAIL"
    assert load_json(V12 / "VALIDATION_CORRECTNESS.json")["status"] == "PASS"
    raws = [load_json(p) for p in sorted(RAW.glob("L*_R*_S??.json"))]
    assert all(r["gate_v1_sha256"] == GATE_SHA for r in raws)
    assert all(r["row_manifest_sha256"] == hashes["validation_row_manifest.json"] for r in raws)
    assert all(r["split_manifest_sha256"] == hashes["validation_split_manifest.json"] for r in raws)
    audit, accept = source_audit(raws, summary, gate)
    protocol = {"id": "range_w4_failure_mechanism_v13", "mode": "cached_data_read_only_no_GPTQ",
                "source_directory": str(V12), "source_file_sha256": hashes,
                "activation_cache_sha256": digest_file(V12 / "validation_cache" / "activation_cache.pt"),
                "source_audit_independent_from_raw": audit,
                "case_definitions": {"robust_success": "heldout_gain_C >= 0.10",
                                     "severe": "heldout_gain_C < -0.10",
                                     "catastrophic": "heldout_gain_C < -0.40"},
                "matched_case_rule": "same layer and projection; prefer heldout_gain_C>=0.10; among those choose closest absolute log D heldout MSE; tie row_id then split_id",
                "quadratic_definition": "X is 512xinput_dim FP32, e=q_FP16_as_FP32-W_FP32, q=e^T(X^TX/N)e=mean((Xe)^2)",
                "article_definition": "four contiguous 128-token full articles in frozen split manifest order",
                "rank_definition": "ascending loss, ten alpha candidates; top3 Jaccard; Spearman of average tie ranks",
                "floating_tolerance": {"absolute": ABS_TOL, "relative": REL_TOL},
                "global_gram_frobenius_method": "low-rank Gram identity using XX^T and XsXh^T; no dxd materialization",
                "torch_version": torch.__version__, "device": torch.cuda.get_device_name(0)}
    save_json(HERE / "PROTOCOL.json", protocol)
    cache = load_pt(V12 / "validation_cache" / "activation_cache.pt")
    assert cache["row_manifest_sha256"] == hashes["validation_row_manifest.json"]
    assert cache["split_manifest_sha256"] == hashes["validation_split_manifest.json"]
    split_by = {s["split_id"]: s for s in split_manifest["splits"]}
    by_key = {(r["row_id"], r["split_id"]): r for r in raws}
    smoke_keys = [("L20_o_proj_R287", "S08"), ("L4_q_proj_R60", "S00")]
    smoke = []
    for rid, sid in smoke_keys:
        rec = by_key[(rid, sid)]
        xs = input_tensors(cache, rec, split_by[sid], "search")
        xh = input_tensors(cache, rec, split_by[sid], "heldout")
        ratio = gram_shift_ratio(xs, xh)
        hs = xs.double().T @ xs.double() / xs.shape[0]
        hh = xh.double().T @ xh.double() / xh.shape[0]
        direct_ratio = float(torch.linalg.matrix_norm(hh - hs) / torch.linalg.matrix_norm(hs))
        assert abs(ratio - direct_ratio) < 1e-4, f"Gram low-rank identity {rid}/{sid}"
        smoke.append({"label": f"{rid}/{sid}/lowrank_gram_ratio", "actual": ratio,
                      "expected": direct_ratio, "abs_error": abs(ratio - direct_ratio), "tolerance": 1e-4})
        result = analyze_one(rec, split_by[sid], cache, ratio, xs, xh, do_all=False)
        smoke.extend(result["checks"])
        del xs, xh, hs, hh
    save_json(HERE / "preflight_checks.json", {"status": "PASS", "checks": smoke})
    print("PREFLIGHT_PASS", flush=True)
    bases, checks, landscapes, ranks, tokens = [], [], [], [], []
    # Group by projection/split so the 4 row cases reuse the same activation matrices and Gram ratio.
    grouped = defaultdict(list)
    for rec in raws:
        grouped[(rec["layer"], rec["projection"], rec["split_id"])].append(rec)
    for gi, (group, records) in enumerate(sorted(grouped.items())):
        xs = input_tensors(cache, records[0], split_by[group[2]], "search")
        xh = input_tensors(cache, records[0], split_by[group[2]], "heldout")
        ratio = gram_shift_ratio(xs, xh)
        for rec in records:
            result = analyze_one(rec, split_by[group[2]], cache, ratio, xs, xh)
            result["base"]["gate_accept"] = accept[rec["row_id"]]
            bases.append(result["base"])
            checks.extend(result["checks"])
            landscapes.extend(result["landscapes"])
            ranks.append({**result["rank"], "gate_accept": accept[rec["row_id"]]})
            tokens.extend(result["tokens"])
        del xs, xh
        if (gi + 1) % 20 == 0:
            print(f"analysis groups {gi+1}/120, cases {len(bases)}/480", flush=True)
    assert len(bases) == 480 and len(landscapes) == 4800 and len(tokens) == 1920
    assert max(c["abs_error"] / c["tolerance"] for c in checks) <= 1
    worst = next(b for b in bases if b["row_id"] == "L20_o_proj_R287" and b["split_id"] == "S08")
    assert abs(worst["heldout_gain"] + 4.120855466502888) < 1e-8
    catastrophic = [b for b in bases if b["heldout_gain"] < -.4]
    assert len(catastrophic) == 20
    matches = []
    for failure in catastrophic:
        candidates = [b for b in bases if b["layer"] == failure["layer"] and
                      b["projection"] == failure["projection"] and b["heldout_gain"] >= .1]
        assert candidates, f"no robust success same layer/projection for {failure['row_id']}/{failure['split_id']}"
        success = min(candidates, key=lambda b: (abs(math.log(b["qD_heldout"] / failure["qD_heldout"])), b["row_id"], b["split_id"]))
        matches.append({"failure_row_id": failure["row_id"], "failure_split_id": failure["split_id"],
                        "success_row_id": success["row_id"], "success_split_id": success["split_id"],
                        "layer": failure["layer"], "projection": failure["projection"],
                        **{f"failure_{k}": failure[k] for k in ["selected_alpha", "search_gain", "heldout_gain", "qD_heldout", "qC_heldout", "generalization_shift", "directional_shift_C", "directional_shift_D", "relative_global_gram_shift_F"]},
                        **{f"success_{k}": success[k] for k in ["selected_alpha", "search_gain", "heldout_gain", "qD_heldout", "qC_heldout", "generalization_shift", "directional_shift_C", "directional_shift_D", "relative_global_gram_shift_F"]}})
    save_csv(HERE / "quadratic_decomposition.csv", bases)
    save_csv(HERE / "catastrophic_cases.csv", catastrophic)
    save_csv(HERE / "matched_cases.csv", matches)
    save_csv(HERE / "article_level_landscapes.csv", landscapes)
    save_csv(HERE / "article_ranking_stability.csv", ranks)
    save_csv(HERE / "directional_shift_analysis.csv", [
        {k: b[k] for k in ["row_id", "split_id", "layer", "projection", "selected_alpha", "search_gain", "heldout_gain",
                          "relative_global_gram_shift_F", "directional_shift_D", "directional_shift_C", "directional_shift_D_minus_C", "generalization_shift"]}
        for b in bases])
    save_csv(HERE / "token_error_concentration.csv", tokens)
    save_csv(HERE / "feature_outcome_summary.csv", features_summary(bases, ranks, tokens))
    save_json(HERE / "quadratic_identity_checks.json", {
        "status": "PASS", "case_count": len(bases), "raw_MSE_comparisons": 4 * len(bases),
        "direct_projection_vs_quadratic_comparisons": 4 * len(bases),
        "side_difference_vs_direct_H_difference_comparisons": 2 * len(bases),
        "shift_identity_comparisons": len(bases), "absolute_tolerance": ABS_TOL,
        "relative_tolerance": REL_TOL, "max_abs_error": max(c["abs_error"] for c in checks),
        "max_error_to_tolerance": max(c["abs_error"] / c["tolerance"] for c in checks),
        "worst_comparisons": sorted(checks, key=lambda c: c["abs_error"] / c["tolerance"], reverse=True)[:10],
        "source_gate_sha256_after": digest_file(V12 / "GATE_V1.json"),
        "source_hashes_unchanged": all(digest_file(V12 / n) == h for n, h in hashes.items()),
        "wall_seconds_including_cache_load_and_analysis": time.perf_counter() - started})
    print("ANALYSIS_PASS", len(bases), "cases", flush=True)


if __name__ == "__main__":
    main()
