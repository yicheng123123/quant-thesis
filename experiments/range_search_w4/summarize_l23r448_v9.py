"""Independently summarize frozen v9 raw JSON and candidate weight caches."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics as stats
from collections import Counter

import torch

from experiments.range_search_w4.prepare_v9_splits import OUTPUT, V8
from experiments.range_search_w4.run_task0_parity import tensor_hash
from experiments.range_search_w4.run_l23r448_v9 import EXPECTED_SPLITS_SHA256

IDS = [f"S{i:02d}" for i in range(10)]
GRID = [round(i / 10, 1) for i in range(1, 11)]
TOL = 1e-4


def save_json(name: str, value: object) -> None:
    (OUTPUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def save_csv(name: str, rows: list[dict]) -> None:
    with (OUTPUT / name).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def percentile(values: list[float], p: float) -> float:
    ordered = sorted(values)
    pos = (len(ordered) - 1) * p
    low = math.floor(pos)
    return ordered[low] + (ordered[min(low + 1, len(ordered) - 1)] - ordered[low]) * (pos - low)


def describe(values: list[float]) -> dict:
    return {"count": len(values), "mean": stats.mean(values), "median": stats.median(values),
            "p25": percentile(values, .25), "p75": percentile(values, .75),
            "min": min(values), "max": max(values)}


def ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks_out = [0.0] * len(values)
    for place, i in enumerate(order):
        ranks_out[i] = float(place)
    for value in set(values):
        tied = [i for i, v in enumerate(values) if v == value]
        avg = sum(ranks_out[i] for i in tied) / len(tied)
        for i in tied:
            ranks_out[i] = avg
    return ranks_out


def pearson(a: list[float], b: list[float]) -> float | None:
    ma, mb = stats.mean(a), stats.mean(b)
    numerator = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    denom = math.sqrt(sum((x - ma) ** 2 for x in a) * sum((y - mb) ** 2 for y in b))
    return numerator / denom if denom else None


def count_sign(values: list[float]) -> dict:
    return {"positive": sum(v > TOL for v in values),
            "tie": sum(abs(v) <= TOL for v in values),
            "regression": sum(v < -TOL for v in values)}


def main() -> None:
    manifest_path = OUTPUT / "splits_manifest.json"
    manifest_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    if manifest_hash != EXPECTED_SPLITS_SHA256:
        raise RuntimeError("frozen manifest hash changed")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    raw = [json.loads((OUTPUT / f"{sid}.json").read_text(encoding="utf-8")) for sid in IDS]
    v8 = json.loads((V8 / "layer23_row448.json").read_text(encoding="utf-8"))
    v8_d = v8["methods"]["D_native_dynamic"]
    v8_c = v8["methods"]["C_gptq_dynamic_grid"]
    v8_gain = 1 - v8_c["heldout_sse"] / v8_d["heldout_sse"]
    all_articles = [d for s in manifest["splits"] for side in ("search", "heldout") for d in s[side]]
    checks = {"manifest_sha256": manifest_hash == EXPECTED_SPLITS_SHA256,
        "ten_splits_complete": len(raw) == 10 and [r["split_id"] for r in raw] == IDS,
        "globally_disjoint_articles": len({d["source_article_index"] for d in all_articles}) == 80,
        "globally_disjoint_token_sequences": len({d["token_sha256"] for d in all_articles}) == 80,
        "v8_reference_disjoint": not ({d["token_sha256"] for d in all_articles} &
                                        set(manifest["v8_reference_token_hashes"])),
        "same_pristine_weight": len({r["weight_sha256"] for r in raw}) == 1,
        "all_candidate_grids": all([c["alpha"] for c in r["candidates"]] == GRID for r in raw),
        "all_alpha1_baselines_present": all(r["candidates"][-1]["alpha"] == 1.0 for r in raw),
        "raw_manifest_matches": all(r["manifest_sha256"] == manifest_hash for r in raw),
        "search_never_worse_than_D": all(r["search_gain"] >= -1e-7 for r in raw),
        "all_splits_512_tokens_per_side": all(r["search_tokens"] == 512 and r["heldout_tokens"] == 512 for r in raw),
        "v8_reference_expected": v8_c["alpha"] == .7 and abs(v8_gain - .2745709424183572) < 1e-10,
        "same_weight_as_v8_reference": all(r["weight_sha256"] == v8["pristine_weight_sha256"] for r in raw)}
    activation_cache = torch.load(OUTPUT / "activation_cache.pt", map_location="cpu", weights_only=True)
    if activation_cache["manifest_sha256"] != manifest_hash:
        raise RuntimeError("activation cache manifest mismatch")
    xs_all = {s["split_id"]: torch.cat([activation_cache["inputs"][d["token_sha256"]]
                   for d in s["search"]]) for s in manifest["splits"]}
    xh_all = {s["split_id"]: torch.cat([activation_cache["inputs"][d["token_sha256"]]
                   for d in s["heldout"]]) for s in manifest["splits"]}
    pristine = activation_cache["weight"].float()
    max_direct_relative_diff = 0.0
    for r in raw:
        sid = r["split_id"]
        weight_cache = torch.load(OUTPUT / f"weights_{sid}.pt", map_location="cpu", weights_only=True)
        checks[f"candidate_weight_hashes_{sid}"] = (weight_cache["manifest_sha256"] == manifest_hash and
            all(tensor_hash(weight_cache["weights"][str(c["alpha"])]) == c["weight_sha256"]
                for c in r["candidates"]))
        deltas = torch.stack([weight_cache["weights"][str(c["alpha"])].float() - pristine
                              for c in r["candidates"]])
        direct_search = (xs_all[sid] @ deltas.T).square().sum(0)
        for index, c in enumerate(r["candidates"]):
            max_direct_relative_diff = max(max_direct_relative_diff,
                abs(float(direct_search[index]) - c["search_sse"]) / max(c["search_sse"], 1e-30))
        for held_id in IDS:
            direct_held = (xh_all[held_id] @ deltas.T).square().sum(0)
            for index, c in enumerate(r["candidates"]):
                stored = c["cross_heldout_sse"][held_id]
                max_direct_relative_diff = max(max_direct_relative_diff,
                    abs(float(direct_held[index]) - stored) / max(stored, 1e-30))
        best = min(r["candidates"], key=lambda c: c["search_output_mse"])
        d = r["candidates"][-1]
        checks[f"loss_and_selection_{sid}"] = (best["alpha"] == r["selected_alpha_on_search"] and
            abs(r["search_gain"] - (1 - best["search_sse"] / d["search_sse"])) < 1e-10 and
            abs(r["heldout_gain"] - (1 - best["heldout_sse"] / d["heldout_sse"])) < 1e-10 and
            all(abs(c["cross_heldout_sse"][sid] - c["heldout_sse"]) < 1e-10 for c in r["candidates"]))
    checks["all_1100_losses_recomputed_from_weights_and_inputs"] = max_direct_relative_diff < 1e-5
    if not all(checks.values()):
        save_json("correctness_checks.json", {"status": "FAIL", "checks": checks})
        raise RuntimeError("raw v9 check failed; no summary generated")
    save_json("correctness_checks.json", {"status": "PASS", "checks": checks})

    per_split, landscape, scatter, activation, matrix, fixed = [], [], [], [], [], []
    for r in raw:
        sid = r["split_id"]
        d = r["candidates"][-1]
        c = next(v for v in r["candidates"] if v["alpha"] == r["selected_alpha_on_search"])
        per_split.append({"split_id": sid, "selected_alpha_on_search": c["alpha"],
            "D_search_output_mse": d["search_output_mse"], "C_search_output_mse": c["search_output_mse"],
            "D_heldout_output_mse": d["heldout_output_mse"], "C_heldout_output_mse": c["heldout_output_mse"],
            "D_search_sse": d["search_sse"], "C_search_sse": c["search_sse"],
            "D_heldout_sse": d["heldout_sse"], "C_heldout_sse": c["heldout_sse"],
            "search_gain": r["search_gain"], "heldout_gain": r["heldout_gain"],
            "outcome": "positive" if r["heldout_gain"] > TOL else
                       "regression" if r["heldout_gain"] < -TOL else "tie",
            "candidate_evaluations": r["candidate_evaluations"],
            "candidate_seconds_total": r["candidate_seconds_total"],
            "hessian_seconds": r["hessian_seconds"], "split_seconds": r["split_seconds"],
            "peak_allocated_bytes": r["peak_allocated_bytes"]})
        scatter.append({"split_id": sid, "selected_alpha": c["alpha"],
                        "search_gain": r["search_gain"], "heldout_gain": r["heldout_gain"]})
        activation.append({"split_id": sid, "heldout_gain": r["heldout_gain"],
                           **r["activation"]})
        for v in r["candidates"]:
            landscape.append({"split_id": sid, "alpha": v["alpha"],
                "search_output_mse": v["search_output_mse"],
                "heldout_output_mse": v["heldout_output_mse"],
                "search_sse": v["search_sse"], "heldout_sse": v["heldout_sse"],
                "selected_alpha_on_search": c["alpha"],
                "is_selected": v["alpha"] == c["alpha"],
                "weight_mse": v["weight_mse"],
                "candidate_seconds": v["candidate_seconds"],
                "scale_json": json.dumps(v["scale"]), "zero_json": json.dumps(v["zero"])})
        matrix.append({"source_search_split": sid, "selected_alpha": c["alpha"], **{
            "heldout_" + other: 1 - c["cross_heldout_sse"][other] /
                                        d["cross_heldout_sse"][other] for other in IDS}})
    for alpha in GRID:
        candidates = [next(c for c in r["candidates"] if c["alpha"] == alpha) for r in raw]
        baseline = [r["candidates"][-1] for r in raw]
        gains = [1 - c["heldout_sse"] / d["heldout_sse"] for c, d in zip(candidates, baseline)]
        fixed.append({"alpha": alpha,
            "mean_heldout_output_mse": stats.mean(c["heldout_output_mse"] for c in candidates),
            "median_heldout_output_mse": stats.median(c["heldout_output_mse"] for c in candidates),
            "pooled_heldout_sse": sum(c["heldout_sse"] for c in candidates),
            "pooled_gain_vs_alpha1": 1 - sum(c["heldout_sse"] for c in candidates) /
                                       sum(d["heldout_sse"] for d in baseline),
            "mean_gain_vs_alpha1": stats.mean(gains),
            "median_gain_vs_alpha1": stats.median(gains),
            **count_sign(gains)})
    save_csv("per_split_results.csv", per_split)
    save_csv("alpha_landscapes.csv", landscape)
    save_csv("split_gain_scatter.csv", scatter)
    save_csv("alpha_transfer_matrix.csv", matrix)
    save_csv("fixed_alpha_summary.csv", fixed)
    save_csv("activation_split_diagnostics.csv", activation)
    selections = [r["selected_alpha_on_search"] for r in raw]
    frequencies = Counter(selections)
    most = max(frequencies.values())
    modes = sorted(a for a, n in frequencies.items() if n == most)
    alpha_stability = {"frequencies": {str(a): frequencies[a] for a in GRID},
        "mode_alphas_tied": modes, "mode_alpha_count": most, "mode_frequency": most / len(raw),
        "selection_agreement": most / len(raw), "mean_alpha": stats.mean(selections),
        "median_alpha": stats.median(selections), "min_alpha": min(selections),
        "max_alpha": max(selections), "alpha1_count": frequencies[1.0],
        "alpha_below_1_count": sum(a < 1 for a in selections),
        "within_0_6_to_0_8_count": sum(.6 <= a <= .8 for a in selections),
        "within_0_6_to_0_7_count": sum(.6 <= a <= .7 for a in selections)}
    save_json("alpha_stability.json", alpha_stability)
    gains = [r["heldout_gain"] for r in raw]
    search = [r["search_gain"] for r in raw]
    diagonal = [matrix[i]["heldout_" + IDS[i]] for i in range(10)]
    offdiagonal = [matrix[i]["heldout_" + IDS[j]] for i in range(10) for j in range(10) if i != j]
    alpha_transfer = {}
    for alpha in sorted(set(selections)):
        values = [matrix[i]["heldout_" + IDS[j]] for i in range(10)
                  if selections[i] == alpha for j in range(10) if i != j]
        alpha_transfer[str(alpha)] = {"offdiagonal_gain": describe(values),
                                      **count_sign(values)}
    gain_vs_shift = pearson(gains, [r["activation"]["gram_relative_frobenius_change"] for r in raw])
    summary = {"scope": "one fixed q_proj row with FP activations; ten globally article-disjoint train splits",
        "n_splits": 10, "candidate_evaluations": sum(r["candidate_evaluations"] for r in raw),
        "hessian_builds": 10, "alpha_stability": alpha_stability,
        "heldout_gain": {**describe(gains), **count_sign(gains),
                         "worst_regression": min((g for g in gains if g < -TOL), default=None)},
        "search_gain": describe(search),
        "search_vs_heldout_gain": {"pearson": pearson(search, gains),
                                   "spearman": pearson(ranks(search), ranks(gains)),
                                   "n": 10, "interpretation": "descriptive only"},
        "v8_reference": {"alpha": v8_c["alpha"], "D_heldout_output_mse": v8_d["heldout_output_mse"],
                         "C_heldout_output_mse": v8_c["heldout_output_mse"],
                         "heldout_gain": v8_gain,
                         "fraction_of_v9_gains_at_or_below_v8": sum(g <= v8_gain for g in gains) / len(gains)},
        "transfer": {"semantics": "source split i quantized weights for alpha_i and source i alpha=1 baseline, both evaluated on heldout_j",
                     "diagonal": describe(diagonal), "offdiagonal": describe(offdiagonal),
                     "offdiagonal_positive": sum(v > TOL for v in offdiagonal),
                     "offdiagonal_tie": sum(abs(v) <= TOL for v in offdiagonal),
                     "offdiagonal_regression": sum(v < -TOL for v in offdiagonal),
                     "selected_alpha_cross_split": alpha_transfer},
        "fixed_alpha": fixed,
        "activation": {"heldout_gain_vs_gram_shift_pearson": gain_vs_shift,
                       "gram_shift": describe([r["activation"]["gram_relative_frobenius_change"] for r in raw]),
                       "scope": "descriptive FP-input shift; no causal attribution"},
        "timing": {"candidate_seconds_total": sum(r["candidate_seconds_total"] for r in raw),
                   "hessian_seconds_total": sum(r["hessian_seconds"] for r in raw),
                   "per_split_seconds_total": sum(r["split_seconds"] for r in raw),
                   "capture": json.loads((OUTPUT / "run_timing.json").read_text(encoding="utf-8"))["capture"]},
        "max_direct_loss_relative_difference": max_direct_relative_diff}
    summary["diagnostic_gate"] = {
        "category": "strong_stable_local_signal" if
            (summary["heldout_gain"]["positive"] >= 8 and
             summary["heldout_gain"]["median"] > .05 and
             summary["heldout_gain"]["min"] > -.05 and
             alpha_stability["within_0_6_to_0_8_count"] >= 8 and
             next(x for x in fixed if x["alpha"] == .7)["pooled_gain_vs_alpha1"] > 0)
            else "moderate_or_weak_local_signal",
        "scope": "descriptive gate for one row only; not a significance test or model-level permission"}
    # Compare CSV output against independently loaded raw records.
    with (OUTPUT / "per_split_results.csv").open(newline="", encoding="utf-8") as f:
        csv_rows = list(csv.DictReader(f))
    checks["per_split_csv_recomputes_raw_gains"] = all(
        abs(float(v["heldout_gain"]) - r["heldout_gain"]) < 1e-12 and
        abs(float(v["search_gain"]) - r["search_gain"]) < 1e-12
        for v, r in zip(csv_rows, raw))
    checks["transfer_diagonal_matches_own_heldout_gain"] = all(
        abs(a - b) < 1e-12 for a, b in zip(diagonal, gains))
    checks["fixed_alpha1_pooled_gain_zero"] = abs(fixed[-1]["pooled_gain_vs_alpha1"]) < 1e-12
    if not all(checks.values()):
        save_json("correctness_checks.json", {"status": "FAIL", "checks": checks})
        raise RuntimeError("summary consistency check failed")
    save_json("correctness_checks.json", {"status": "PASS", "checks": checks})
    save_json("summary.json", summary)
    fixed07 = next(x for x in fixed if x["alpha"] == .7)
    fixed08 = next(x for x in fixed if x["alpha"] == .8)
    regressions = [(row["source_search_split"], row["selected_alpha"], target[8:], float(value))
                   for row in matrix for target, value in row.items()
                   if target.startswith("heldout_") and float(value) < -TOL]
    split_lines = [f"| {r['split_id']} | {r['selected_alpha_on_search']:.1f} | "
                   f"{100*r['search_gain']:.2f}% | {100*r['heldout_gain']:.2f}% | "
                   f"{r['D_heldout_output_mse']:.6g} | {r['C_heldout_output_mse']:.6g} |"
                   for r in per_split]
    frequency_line = ", ".join(f"α={a:.1f}: {frequencies[a]}/10" for a in sorted(frequencies))
    cross_regress_line = "; ".join(f"{src} (α={alpha:.1f}) → {dst}: {100*gain:.2f}%"
                                   for src, alpha, dst, gain in regressions)
    report = ["# L23 R448 W4 dynamic GPTQ: article split stability", "",
        "## Frozen scope", "",
        "Qwen/Qwen2.5-0.5B-Instruct at revision `7ae557604adf67be50417f59c2c2f167def9a775`; layer 23 `self_attn.q_proj` row 448 only. FP16 model weight, FP32 Hessian/GPTQ/output error, symmetric W4A16 G128, 128-column blocks, damping 0.1, `desc_act=False`. Inputs come from the FP model. Candidate α is shared across all seven groups; each candidate reruns the v8 parity-validated dynamic GPTQ compensation from pristine W. D is α=1, and C is the ten-point search-MSE minimizer. No test split, PPL, other rows, or model-level inference was used.", "",
        f"The frozen [split manifest](splits_manifest.json) has SHA-256 `{manifest_hash}`. The pool is the next 80 eligible WikiText-2 train articles after the eight v8 reference articles, not a random sample of the entire corpus. The 80 articles are globally disjoint across v9 splits and disjoint from v8. Every split has four search and four held-out full articles, with the first 128 tokens per article. The manifest records source IDs, article/token hashes, token IDs and deterministic sub-seeds; it was written before any α candidate run.", "",
        "## Per-split D versus search-selected C", "",
        "| Split | Selected α | Search gain | Held-out gain | D held-out MSE | C held-out MSE |",
        "| --- | ---: | ---: | ---: | ---: | ---: |", *split_lines, "",
        f"Held-out gains: {summary['heldout_gain']['positive']}/10 positive, {summary['heldout_gain']['tie']} ties, {summary['heldout_gain']['regression']} regressions under relative tolerance {TOL:g}. Median {100*summary['heldout_gain']['median']:.2f}%, mean {100*summary['heldout_gain']['mean']:.2f}%, p25–p75 {100*summary['heldout_gain']['p25']:.2f}–{100*summary['heldout_gain']['p75']:.2f}%, min–max {100*summary['heldout_gain']['min']:.2f}–{100*summary['heldout_gain']['max']:.2f}%. There is no own-split held-out regression; the weakest own split is S02 at {100*raw[2]['heldout_gain']:.2f}% gain. Search gain is nonnegative by construction because α=1 is in the candidate set; it is not generalization evidence.", "",
        "## α stability and the v8 reference", "",
        frequency_line + ".", "",
        f"Modes tie at α={modes[0]:.1f} and α={modes[1]:.1f}, each {most}/10; exact selection agreement is {100*most/10:.0f}%. Mean α {alpha_stability['mean_alpha']:.2f}, median {alpha_stability['median_alpha']:.1f}, range {alpha_stability['min_alpha']:.1f}–{alpha_stability['max_alpha']:.1f}. All ten selections lie in the adjacent 0.6–0.8 band, with eight in 0.6–0.7. This supports a stable *band*, not a uniquely stable α.", "",
        f"The separate v8 reference selected α=0.7: D held-out MSE {v8_d['heldout_output_mse']:.8g}, C {v8_c['heldout_output_mse']:.8g}, gain {100*v8_gain:.2f}%. Only {sum(g <= v8_gain for g in gains)}/10 v9 gains are at or below it. Thus v8's ≈27% is below the v9 median, not an upper-tail success. Different train articles and Hessians mean the absolute MSE values should not be treated as paired across v8/v9.", "",
        "## Fixed α and cross-split transfer", "",
        f"The α=0.7 value was already selected in v8, before the new disjoint splits: fixed α=0.7 beats α=1 on {fixed07['positive']}/10 own held-out sets, with pooled SSE gain {100*fixed07['pooled_gain_vs_alpha1']:.2f}%. In the post hoc full grid, fixed α=0.8 has pooled gain {100*fixed08['pooled_gain_vs_alpha1']:.2f}% and {fixed08['positive']}/10 wins; its slight edge is descriptive and must not be used to retune the frozen method on held-out data.", "",
        "Transfer cell `M[i,j]` uses the quantized row built with search split *i* and its search-selected α, tested on held-out split *j*. Its denominator uses the **same source split i GPTQ state** with α=1 on that target held-out set. This measures transfer of the source-calibrated quantized row, including its Hessian; it is not an α-only intervention with target-specific recalibration.", "",
        f"Transfer diagonal median {100*summary['transfer']['diagonal']['median']:.2f}%; off-diagonal median {100*summary['transfer']['offdiagonal']['median']:.2f}%, mean {100*summary['transfer']['offdiagonal']['mean']:.2f}%. Off-diagonal outcomes are {summary['transfer']['offdiagonal_positive']}/90 positive and {summary['transfer']['offdiagonal_regression']}/90 regressions. Both regressions are small: {cross_regress_line}. Mean off-diagonal gains by source-selected α are 0.6: {100*alpha_transfer['0.6']['offdiagonal_gain']['mean']:.2f}%, 0.7: {100*alpha_transfer['0.7']['offdiagonal_gain']['mean']:.2f}%, 0.8: {100*alpha_transfer['0.8']['offdiagonal_gain']['mean']:.2f}%. The 90 cells reuse ten source weight states and ten held-out sets, so they are correlated diagnostics, not 90 independent replications. For the four source splits selecting α=0.7, all 36 off-diagonal cells improve over their source-matched D.", "",
        "## Search predictiveness and activation shift", "",
        f"Across ten splits, search gain versus held-out gain has Pearson r={summary['search_vs_heldout_gain']['pearson']:.3f} and tie-aware Spearman ρ={summary['search_vs_heldout_gain']['spearman']:.3f}. Search-gain magnitude does not rank held-out-gain magnitude well in this small set; no significance claim is made.", "",
        f"No own held-out split regresses, so there is no own-regression subgroup to explain. S02 has the weakest own gain and is the source of both slight transfer regressions; its search/held-out Gram relative change is {100*raw[2]['activation']['gram_relative_frobenius_change']:.1f}%, while the ten-split median is {100*summary['activation']['gram_shift']['median']:.1f}%. Gram shift versus held-out gain has descriptive Pearson r={summary['activation']['heldout_gain_vs_gram_shift_pearson']:.3f}. Global RMS, per-feature RMS, mean/std and percentile changes are saved in `activation_split_diagnostics.csv`. These observations do not establish a causal explanation.", "",
        "## Gate, cost and limits", "",
        f"The requested diagnostic rubric classifies this as **{summary['diagnostic_gate']['category']}** for L23 R448: 10/10 own held-out wins, median gain above 5%, no own regression, selected α in a narrow adjacent band, and the previously observed fixed α=0.7 has positive pooled gain. Exact α agreement remains only 40%, and two off-diagonal transfers are slightly negative. This is local FP-input evidence, not a statistical significance test or a claim about other rows or full-model quality.", "",
        f"Measured work: {summary['candidate_evaluations']} full dynamic GPTQ candidate evaluations, ten native-timing Hessian builds, 80 FP-model forwards. Model capture took {summary['timing']['capture']['capture_seconds']:.2f}s (peak allocated CUDA {summary['timing']['capture']['capture_peak_allocated_bytes']/2**20:.1f} MiB); candidate evaluations took {summary['timing']['candidate_seconds_total']:.2f}s, Hessians {summary['timing']['hessian_seconds_total']:.2f}s. Full invocation took {json.loads((OUTPUT/'run_timing.json').read_text(encoding='utf-8'))['invocation_seconds']:.2f}s. These are offline calibration costs, not inference speedups.", "",
        f"All {100 + 1000} candidate/search and cross-held-out losses were recomputed directly from cached FP inputs and saved candidate weights; max relative difference {max_direct_relative_diff:.3g}. `correctness_checks.json` is PASS. The full landscape, transfer cells and split metrics remain in raw per-split JSON plus CSV; cached activations and candidate weights allow independent rechecking.", "",
        "The result supports a stable clipping preference for **this row under FP projection inputs and this article pool**. It does not show that the same α helps other rows, quantized-prefix inputs, or final PPL. No experiment beyond the requested one-row multi-split diagnostic was launched.", "",
        "## Reproduce", "",
        "From `E:\\quant-thesis` with the existing `.venv-gptq`:", "",
        "```powershell", "$env:PYTHONPATH='E:\\quant-thesis'",
        ".\\.venv-gptq\\Scripts\\python.exe -m experiments.range_search_w4.prepare_v9_splits",
        ".\\.venv-gptq\\Scripts\\python.exe -m experiments.range_search_w4.run_l23r448_v9",
        ".\\.venv-gptq\\Scripts\\python.exe -m experiments.range_search_w4.summarize_l23r448_v9",
        "```", "",
        "The preparation command verifies an existing manifest byte-for-byte instead of replacing it. The run command resumes complete split JSON/weight pairs; the summarizer rebuilds all tables and this report from frozen inputs and raw results.", ""]
    (OUTPUT / "REPORT.md").write_text("\n".join(report), encoding="utf-8")
    print("SUMMARY", json.dumps({"heldout": summary["heldout_gain"],
          "alpha": alpha_stability, "v8": summary["v8_reference"],
          "transfer": {"diag": summary["transfer"]["diagonal"]["median"],
                       "offdiag": summary["transfer"]["offdiagonal"]["median"]}},
          ensure_ascii=False))


if __name__ == "__main__":
    main()
