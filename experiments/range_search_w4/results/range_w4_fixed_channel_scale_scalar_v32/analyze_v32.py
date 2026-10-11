"""Artifact-only audit and analysis for the frozen v32 local 2x2 diagnostic."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
RESULTS = ROOT / "experiments/range_search_w4/results"
V24 = RESULTS / "range_w4_structured_group_e2_v24"
P = json.loads((HERE / "FROZEN_PROTOCOL.json").read_text("utf-8"))
ALPHAS = [float(x) for x in P["alpha_grid"]]
ROWS = {k: [int(x) for x in v] for k, v in P["targets"]["rows"].items()}
EPS = json.loads((V24 / "HELD_REPEAT_NOISE.json").read_text("utf-8"))["repeat_noise_mse_by_unit"]
ELEMENTS = 16 * 2048
CONTRASTS = {
    "G10_vs_G00_scaling_at_alpha1": ("G00", "G10"),
    "G11_vs_G10_search_after_scaling": ("G10", "G11"),
    "G11_vs_G01_combination_vs_original_search": ("G01", "G11"),
}


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def sha_tensor(t: torch.Tensor) -> str:
    return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def load(path: Path):
    return json.loads(path.read_text("utf-8"))


def dump(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n", "utf-8")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise RuntimeError(f"refuse empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)


def target_stem(layer: int, projection: str, split: str) -> str:
    return f"L{layer}_{projection}_{split}"


def unit_id(layer: int, projection: str, row: int, split: str) -> str:
    return f"L{layer}_{projection}_R{row}_{split}"


def outcome(delta: float, epsilon: float) -> str:
    return "win" if delta > epsilon else "loss" if delta < -epsilon else "tie"


def select(scores: dict[float, float]) -> float:
    best = min(scores.values())
    return max(a for a, value in scores.items() if value == best)


def summary_groups(rows: list[dict]):
    groupings = [
        ("overall", tuple()),
        ("split", ("split",)),
        ("target_both_splits", ("layer", "projection")),
        ("target_split", ("split", "layer", "projection")),
    ]
    for level, keys in groupings:
        buckets = defaultdict(list)
        for r in rows:
            buckets[tuple(r[k] for k in keys)].append(r)
        for key, items in sorted(buckets.items(), key=lambda kv: tuple(str(x) for x in kv[0])):
            labels = {k: v for k, v in zip(keys, key)}
            yield level, labels, items


def summarize_contrast(rows: list[dict]) -> list[dict]:
    out = []
    for level, labels, items in summary_groups(rows):
        for name, (base, candidate) in CONTRASTS.items():
            base_sse = math.fsum(float(r[f"{base}_held_mse"]) * ELEMENTS for r in items)
            candidate_sse = math.fsum(float(r[f"{candidate}_held_mse"]) * ELEMENTS for r in items)
            gains = [(float(r[f"{base}_held_mse"]) - float(r[f"{candidate}_held_mse"])) /
                     float(r[f"{base}_held_mse"]) for r in items]
            savings = sorted((max(0.0, (float(r[f"{base}_held_mse"]) - float(r[f"{candidate}_held_mse"])) * ELEMENTS)
                              for r in items), reverse=True)
            positives = math.fsum(savings)
            harms = math.fsum(max(0.0, (float(r[f"{candidate}_held_mse"]) - float(r[f"{base}_held_mse"])) * ELEMENTS)
                              for r in items)
            out.append({
                "level": level, "split": labels.get("split", "ALL"),
                "layer": labels.get("layer", "ALL"), "projection": labels.get("projection", "ALL"),
                "contrast": name, "baseline_arm": base, "candidate_arm": candidate,
                "row_split_units": len(items), "pooled_baseline_sse": base_sse,
                "pooled_candidate_sse": candidate_sse, "pooled_sse_reduction": base_sse - candidate_sse,
                "pooled_relative_gain": (base_sse - candidate_sse) / base_sse,
                "wins": sum(r[f"{name}_outcome"] == "win" for r in items),
                "ties": sum(r[f"{name}_outcome"] == "tie" for r in items),
                "losses": sum(r[f"{name}_outcome"] == "loss" for r in items),
                "median_row_relative_gain": statistics.median(gains),
                "worst_row_relative_gain": min(gains),
                "worst_unit_id": items[gains.index(min(gains))]["unit_id"],
                "severe_gt10pct_regressions": sum(g < -0.10 for g in gains),
                "gross_positive_savings_sse": positives, "gross_harm_sse": harms,
                "top1_share_of_positive_savings": savings[0] / positives if positives else 0.0,
                "top3_share_of_positive_savings": math.fsum(savings[:3]) / positives if positives else 0.0,
            })
    return out


def summarize_arms(rows: list[dict]) -> list[dict]:
    out = []
    for level, labels, items in summary_groups(rows):
        out.append({"level": level, "split": labels.get("split", "ALL"),
                    "layer": labels.get("layer", "ALL"), "projection": labels.get("projection", "ALL"),
                    "row_split_units": len(items),
                    **{f"{arm}_pooled_held_sse": math.fsum(float(r[f"{arm}_held_mse"]) * ELEMENTS for r in items)
                       for arm in ("G00", "G01", "G10", "G11")}})
    return out


def summarize_interaction(rows: list[dict]) -> list[dict]:
    out = []
    for level, labels, items in summary_groups(rows):
        i_mses = [float(r["interaction_I_mse"]) for r in items]
        i_sse = math.fsum(i * ELEMENTS for i in i_mses)
        base_sse = math.fsum(float(r["G00_held_mse"]) * ELEMENTS for r in items)
        out.append({"level": level, "split": labels.get("split", "ALL"),
                    "layer": labels.get("layer", "ALL"), "projection": labels.get("projection", "ALL"),
                    "row_split_units": len(items), "interaction_I_sse": i_sse,
                    "interaction_I_sse_over_G00_sse": i_sse / base_sse,
                    "median_interaction_I_mse": statistics.median(i_mses),
                    "positive_I_rows": sum(x > 0 for x in i_mses),
                    "zero_I_rows": sum(x == 0 for x in i_mses),
                    "negative_I_rows": sum(x < 0 for x in i_mses),
                    "meaning": "I=(L10-L11)-(L00-L01); positive means greater absolute search loss reduction after D; descriptive only"})
    return out


def stitch_local(stem: str, projection: str, chosen: dict[str, float],
                 candidates: dict[float, tuple[dict, dict]], held: dict,
                 d_meta: dict, h_meta: dict) -> dict:
    rows = ROWS[projection]
    gidx = candidates[1.0][0]["g_idx"]
    if not all(torch.equal(obj["g_idx"], gidx) for obj, _ in candidates.values()):
        raise RuntimeError(f"g_idx changed among alphas: {stem}")
    q = torch.stack([candidates[chosen[str(row)]][0]["weight_fp16"][j] for j, row in enumerate(rows)]).contiguous()
    scales = torch.stack([candidates[chosen[str(row)]][0]["scales"][j] for j, row in enumerate(rows)]).contiguous()
    zeros = torch.stack([candidates[chosen[str(row)]][0]["zeros"][j] for j, row in enumerate(rows)]).contiguous()
    if sha_tensor(q) != held["G11_stitched_weight_rows_sha256"]:
        raise RuntimeError(f"stitched selected Q disagrees with scored G11: {stem}")
    obj = {"weight_fp16": q, "scales": scales, "zeros": zeros, "g_idx": gidx.clone()}
    pt = HERE / "raw/stitched" / f"{stem}_G11.pt"
    pt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(obj, pt)
    meta = {"stem": stem, "status": "LOCAL_FIXED_ROW_DIAGNOSTIC_ONLY",
            "protocol_sha256": sha_file(HERE / "FROZEN_PROTOCOL.json"),
            "D_sha256": d_meta["D_sha256"], "H_sha256": h_meta["H_sha256"],
            "selected_alpha_by_row": chosen,
            "rows": rows, "selected_weight_sha256": sha_tensor(q),
            "selected_scales_sha256": sha_tensor(scales),
            "selected_zeros_sha256": sha_tensor(zeros),
            "g_idx_sha256": sha_tensor(gidx),
            "packability": "Qprime is valid in scaled coordinates with Xprime; Qprime D^-1 is not declared directly packable as original G128"}
    dump(pt.with_suffix(".json"), meta)
    return meta


def cost_summary(events: list[dict], budget: dict, status: dict, preflight: dict) -> dict:
    by_stage = defaultdict(list)
    for e in events:
        by_stage[e["stage"]].append(e)
    candidate_events = by_stage["scaled_candidate_saved"]
    parity_events = by_stage["native_scaled_alpha1_parity"]
    d_events = by_stage["D_statistic_saved"]
    h_events = by_stage["scaled_H_saved"]
    if len(candidate_events) != 120 or len(parity_events) != 12 or len(d_events) != 12 or len(h_events) != 12:
        raise RuntimeError("candidate/native/D/H event count mismatch")
    if len({(e["stem"], float(e["alpha"])) for e in candidate_events}) != 120:
        raise RuntimeError("candidate executions not unique")
    candidates = math.fsum(float(e["GPTQ_seconds"]) for e in candidate_events)
    search = math.fsum(float(e["search_score_seconds"]) for e in candidate_events)
    native = math.fsum(float(e["seconds"]) for e in parity_events)
    d_seconds = math.fsum(float(e["seconds"]) for e in d_events)
    h_seconds = math.fsum(float(e["seconds"]) for e in h_events)
    factor_events = by_stage["scaled_factor_prepared"] + by_stage["scaled_preflight_factor"]
    factor_seconds = math.fsum(float(e["seconds"]) for e in factor_events)
    identity_h = math.fsum(float(r["hessian_seconds"]) for r in preflight["D_identity"])
    identity_factor = math.fsum(float(r["factor_seconds"]) for r in preflight["D_identity"])
    identity_gptq = math.fsum(float(e["gptq_seconds"]) for e in by_stage["identity_preflight_pass"])
    identity_score = math.fsum(float(e["score_seconds"]) + float(e["scaled_path_score_seconds"])
                               for e in by_stage["identity_preflight_pass"])
    order_repeat = math.fsum(float(e["seconds"]) for e in by_stage["candidate_order_state_pass"])
    search_offset = math.fsum(float(e["seconds"]) for e in by_stage["search_reference_offset"])
    held_events = by_stage["held_result_saved"]
    held_offset = math.fsum(float(e["offset_seconds"]) for e in held_events)
    held_g10 = math.fsum(float(e["G10_score_seconds"]) for e in held_events)
    held_g11 = math.fsum(float(e["G11_score_seconds"]) for e in held_events)
    candidate_io = math.fsum(float(e["io_seconds"]) for e in candidate_events)
    score_formula = math.fsum(float(e["seconds"]) for e in by_stage["score_formula_audit_pass"])
    named = math.fsum([candidates, search, native, d_seconds, h_seconds, factor_seconds,
                       identity_h, identity_factor, identity_gptq, identity_score, order_repeat,
                       search_offset, held_offset, held_g10, held_g11, candidate_io, score_formula])
    total = float(budget["elapsed_seconds"])
    return {
        "status": "COMPLETE", "budget_seconds": float(budget["max_seconds"]),
        "cumulative_total_wall_seconds_including_all_three_GPU_processes": total,
        "budget_used_fraction": total / float(budget["max_seconds"]),
        "scaled_candidate_full_linear_GPTQ_executions": len(candidate_events),
        "scaled_native_alpha1_parity_full_linear_GPTQ_executions": len(parity_events),
        "D_identity_full_linear_GPTQ_executions": len(preflight["D_identity"]),
        "candidate_order_repeat_full_linear_GPTQ_executions": len(by_stage["candidate_order_state_pass"]),
        "all_full_linear_GPTQ_executions_including_parity": 120 + 12 + 2 + 1,
        "new_scaled_candidate_row_alpha_search_scores": 120 * 4,
        "new_scaled_selected_row_held_scores": 12 * 4 * 2,
        "reused_original_candidate_row_alpha_search_and_held_scores": 480 * 2,
        "D_statistic_build_seconds": d_seconds, "scaled_H_native_process_batch_seconds": h_seconds,
        "scaled_factor_prepare_seconds_including_preflight_repeat": factor_seconds,
        "scaled_candidate_GPTQ_seconds": candidates,
        "scaled_candidate_search_scoring_seconds": search,
        "native_scaled_alpha1_parity_seconds": native,
        "D_identity_H_build_seconds": identity_h,
        "D_identity_factor_seconds": identity_factor,
        "D_identity_GPTQ_seconds": identity_gptq,
        "D_identity_direct_and_scaled_score_seconds": identity_score,
        "candidate_order_repeat_GPTQ_seconds": order_repeat,
        "search_reference_offset_build_seconds": search_offset,
        "held_reference_offset_build_seconds": held_offset,
        "G10_held_score_seconds": held_g10, "G11_held_score_seconds": held_g11,
        "candidate_checkpoint_io_seconds": candidate_io,
        "post_lock_score_formula_audit_seconds": score_formula,
        "named_nonoverlapping_stage_seconds": named,
        "unattributed_wall_seconds_including_cuda_init_CPU_load_hashing_other_IO": total - named,
        "GPU_peak_allocated_bytes": int(status["peak_allocated_bytes"]),
        "GPU_peak_reserved_bytes": int(status["peak_reserved_bytes"]),
        "preflight_process_count": sum(e["stage"] == "process_start" and e["mode"] == "preflight" for e in events),
        "formal_process_count": sum(e["stage"] == "process_start" and e["mode"] == "formal" for e in events),
        "score_audit_process_count": sum(e["stage"] == "process_start" and e["mode"] == "score_formula_audit" for e in events),
        "new_model_forwards": 0, "PPL_runs": 0,
    }


def main() -> None:
    source = load(HERE / "SOURCE_AUDIT.json")
    preflight = load(HERE / "PREFLIGHT.json")
    run = load(HERE / "RUN_STATUS.json")
    score_check = load(HERE / "SCORING_FORMULA_AUDIT.json")
    budget = load(HERE / "BUDGET.json")
    protocol_sha = sha_file(HERE / "FROZEN_PROTOCOL.json")
    for name, record, required in (("source", source, "PASS"), ("preflight", preflight, "PASS"),
                                   ("formal", run, "COMPLETE"), ("scoring", score_check, "PASS")):
        if record["status"] != required:
            raise RuntimeError(f"{name} status is not {required}")
    if preflight["protocol_sha256"] != protocol_sha or run["protocol_sha256"] != protocol_sha or score_check["protocol_sha256"] != protocol_sha:
        raise RuntimeError("frozen protocol hash mismatch across stages")
    if budget["elapsed_seconds"] > budget["max_seconds"]:
        raise RuntimeError("GPU cumulative budget exceeded")

    events = [json.loads(line) for line in (HERE / "RUN_EVENTS.jsonl").read_text("utf-8").splitlines()]
    event_indices = defaultdict(dict)
    for i, event in enumerate(events):
        if "stem" in event:
            event_indices[event["stem"]].setdefault(event["stage"], []).append(i)
    baseline = {(r["bank"], int(r["layer"]), r["projection"], int(r["row"]), float(r["alpha"])): r
                for r in read_csv(HERE / "ORIGINAL_BANK_REUSED.csv")}
    if len(baseline) != 480:
        raise RuntimeError("original candidate bank incomplete")

    full_rows = []
    search_rows = []
    scale_h_audit = []
    stitched_records = []
    for layer in (0, 12, 23):
        for projection in ("up_proj", "down_proj"):
            for split in ("S0", "S1"):
                stem = target_stem(layer, projection, split)
                dpt = HERE / "raw/scales" / f"{stem}.pt"
                dmeta = load(dpt.with_suffix(".json"))
                ddata = torch.load(dpt, map_location="cpu", weights_only=True)
                d, a = ddata["D"], ddata["a_fp64"]
                if d.dtype != torch.float32 or a.dtype != torch.float64 or d.shape != a.shape:
                    raise RuntimeError(f"D/a dtype or shape mismatch: {stem}")
                raw = np.sqrt(np.maximum(a.numpy(), 1e-8))
                expected_d = torch.from_numpy((raw / math.sqrt(float(raw.min()) * float(raw.max()))).astype(np.float32))
                if not torch.equal(d, expected_d) or sha_tensor(d) != dmeta["D_sha256"] or sha_tensor(a) != dmeta["a_sha256"]:
                    raise RuntimeError(f"D formula/hash mismatch: {stem}")
                if dmeta["search_doc_indices"] != P["split_documents"][split]["search"] or dmeta["input_sha256"] != P["input_sha256"][f"L{layer}_{projection}"]:
                    raise RuntimeError(f"D source mismatch: {stem}")
                hpt = HERE / "raw/hessians" / f"{stem}.pt"
                hmeta = load(hpt.with_suffix(".json"))
                hdata = torch.load(hpt, map_location="cpu", weights_only=True)
                h = hdata["H"]
                if h.dtype != torch.float32 or h.shape != (d.numel(), d.numel()) or int(hdata["nsamples"]) != 16:
                    raise RuntimeError(f"scaled H dtype/shape/nsamples mismatch: {stem}")
                if sha_tensor(h) != hmeta["H_sha256"] or hmeta["D_sha256"] != dmeta["D_sha256"]:
                    raise RuntimeError(f"scaled H hash/source mismatch: {stem}")
                if hmeta["search_doc_indices"] != P["split_documents"][split]["search"] or hmeta["input_sha256"] != dmeta["input_sha256"]:
                    raise RuntimeError(f"scaled H article/input source mismatch: {stem}")
                old_h_meta = load(V24 / "cache/hessians" / f"{stem}.json")
                scale_h_audit.append({"stem": stem, "split": split, "layer": layer, "projection": projection,
                                      "input_width": d.numel(), "D_sha256": dmeta["D_sha256"],
                                      "a_sha256": dmeta["a_sha256"], "D_min": dmeta["D_min"], "D_max": dmeta["D_max"],
                                      "search_input_sha256": dmeta["input_sha256"], "scaled_H_sha256": hmeta["H_sha256"],
                                      "unscaled_v24_H_sha256": old_h_meta["H_sha256"],
                                      "H_changed_after_scaling": hmeta["H_sha256"] != old_h_meta["H_sha256"],
                                      "H_nsamples": int(hdata["nsamples"]),
                                      "D_statistic_seconds": dmeta["statistic_seconds"],
                                      "scaled_H_build_seconds": hmeta["hessian_build_seconds"]})

                parity = load(HERE / "raw/candidates" / f"{stem}_native_alpha1_parity.json")
                if parity["status"] != "PASS" or not all(parity["fields_exact"].values()) or parity["D_sha256"] != dmeta["D_sha256"] or parity["H_sha256"] != hmeta["H_sha256"]:
                    raise RuntimeError(f"scaled native alpha1 parity missing/fails: {stem}")
                candidates = {}
                for alpha in ALPHAS:
                    pt = HERE / "raw/candidates" / f"{stem}_a{alpha:.1f}.pt"
                    jp = pt.with_suffix(".json")
                    if not pt.is_file() or not jp.is_file():
                        raise RuntimeError(f"candidate checkpoint missing: {stem}/{alpha}")
                    obj = torch.load(pt, map_location="cpu", weights_only=True)
                    meta = load(jp)
                    if (meta["protocol_sha256"] != protocol_sha or meta["alpha"] != alpha or
                        meta["D_sha256"] != dmeta["D_sha256"] or meta["H_sha256"] != hmeta["H_sha256"] or
                        meta["search_doc_indices"] != P["split_documents"][split]["search"]):
                        raise RuntimeError(f"candidate metadata mismatch: {stem}/{alpha}")
                    for key, hash_key in (("weight_fp16", "weight_sha256"), ("scales", "scales_sha256"),
                                          ("zeros", "zeros_sha256"), ("g_idx", "g_idx_sha256")):
                        if sha_tensor(obj[key]) != meta[hash_key]:
                            raise RuntimeError(f"candidate tensor hash mismatch: {stem}/{alpha}/{key}")
                    if obj["weight_fp16"].shape != (4, d.numel()) or obj["weight_fp16"].dtype != torch.float16:
                        raise RuntimeError(f"candidate Q shape/dtype mismatch: {stem}/{alpha}")
                    candidates[alpha] = (obj, meta)
                    for row in ROWS[projection]:
                        s = float(meta["search_mse_by_row"][str(row)])
                        if not math.isfinite(s) or s < 0:
                            raise RuntimeError(f"nonfinite candidate search score: {stem}/{alpha}/{row}")
                        search_rows.append({"split": split, "layer": layer, "projection": projection,
                                            "row": row, "unit_id": unit_id(layer, projection, row, split),
                                            "alpha": alpha, "search_mse": s, "search_sse": s * ELEMENTS,
                                            "D_sha256": dmeta["D_sha256"], "H_sha256": hmeta["H_sha256"],
                                            "candidate_weight_sha256": meta["weight_sha256"],
                                            "scales_sha256": meta["scales_sha256"],
                                            "zeros_sha256": meta["zeros_sha256"],
                                            "g_idx_sha256": meta["g_idx_sha256"]})
                lock_path = HERE / "raw" / f"{stem}_SEARCH_LOCK.json"
                lock = load(lock_path)
                if lock["status"] != "LOCKED_BEFORE_HELD_SCORING" or lock["D_sha256"] != dmeta["D_sha256"] or lock["H_sha256"] != hmeta["H_sha256"]:
                    raise RuntimeError(f"search lock mismatch: {stem}")
                for row in ROWS[projection]:
                    scores = {alpha: float(candidates[alpha][1]["search_mse_by_row"][str(row)]) for alpha in ALPHAS}
                    chosen = select(scores)
                    if chosen != float(lock["selected_alpha_by_row"][str(row)]) or min(scores.values()) != float(lock["selected_search_mse_by_row"][str(row)]):
                        raise RuntimeError(f"search choice differs from locked argmin: {stem}/{row}")
                if (min(event_indices[stem]["scaled_search_lock_saved"]) >=
                    min(event_indices[stem]["held_result_saved"])):
                    raise RuntimeError(f"held result event precedes search lock: {stem}")
                held = load(HERE / "raw" / f"{stem}_HELD_RESULTS.json")
                if (held["status"] != "COMPLETE" or held["protocol_sha256"] != protocol_sha or
                    held["search_lock_sha256"] != sha_file(lock_path) or held["D_sha256"] != dmeta["D_sha256"] or
                    held["H_sha256"] != hmeta["H_sha256"] or held["held_doc_indices"] != P["split_documents"][split]["held"]):
                    raise RuntimeError(f"held result lineage mismatch: {stem}")
                stitched_records.append(stitch_local(stem, projection, lock["selected_alpha_by_row"],
                                                      candidates, held, dmeta, hmeta))
                if len(held["rows"]) != 4:
                    raise RuntimeError(f"held fixed-row count mismatch: {stem}")
                for r in held["rows"]:
                    row = int(r["row"])
                    alpha01 = float(baseline[(split, layer, projection, row, 1.0)]["original_selected_alpha"])
                    b00 = float(baseline[(split, layer, projection, row, 1.0)]["held_mse"])
                    b01 = float(baseline[(split, layer, projection, row, alpha01)]["held_mse"])
                    if (float(r["G00_held_mse"]) != b00 or float(r["G01_held_mse"]) != b01 or
                        float(r["G01_alpha"]) != alpha01 or float(r["G11_alpha"]) != float(lock["selected_alpha_by_row"][str(row)]) or
                        float(r["G10_search_mse"]) != float(candidates[1.0][1]["search_mse_by_row"][str(row)]) or
                        float(r["G11_search_mse"]) != float(lock["selected_search_mse_by_row"][str(row)])):
                        raise RuntimeError(f"four-arm baseline/selection mismatch: {stem}/{row}")
                    losses = {arm: float(r[f"{arm}_held_mse"]) for arm in ("G00", "G01", "G10", "G11")}
                    if not all(math.isfinite(x) and x > 0 for x in losses.values()):
                        raise RuntimeError(f"invalid held loss: {stem}/{row}")
                    eps = float(EPS[unit_id(layer, projection, row, split)])
                    raw = {"split": split, "layer": layer, "projection": projection, "row": row,
                           "unit_id": unit_id(layer, projection, row, split),
                           "D_sha256": dmeta["D_sha256"], "H_sha256": hmeta["H_sha256"],
                           "G00_alpha": 1.0, "G01_alpha": alpha01, "G10_alpha": 1.0,
                           "G11_alpha": float(r["G11_alpha"]),
                           "G00_search_mse": float(baseline[(split, layer, projection, row, 1.0)]["search_mse"]),
                           "G01_search_mse": float(baseline[(split, layer, projection, row, alpha01)]["search_mse"]),
                           "G10_search_mse": float(r["G10_search_mse"]), "G11_search_mse": float(r["G11_search_mse"]),
                           **{f"{arm}_held_mse": loss for arm, loss in losses.items()},
                           **{f"{arm}_held_sse": loss * ELEMENTS for arm, loss in losses.items()},
                           "epsilon_mse_v24_frozen": eps,
                           "interaction_I_mse": (losses["G10"] - losses["G11"]) - (losses["G00"] - losses["G01"]),
                           "interaction_I_sse": ((losses["G10"] - losses["G11"]) -
                                                 (losses["G00"] - losses["G01"])) * ELEMENTS,
                           "scaled_search_alpha_changed_vs_original": float(r["G11_alpha"]) != alpha01,
                           "search_input_doc_indices": ";".join(map(str, P["split_documents"][split]["search"])),
                           "held_input_doc_indices": ";".join(map(str, P["split_documents"][split]["held"])),
                           "held_coordinate_drift_mse": float(held["held_coordinate_drift"]["unquantized_fp16_coordinate_drift_mse_by_row"][str(row)])}
                    for name, (base, new) in CONTRASTS.items():
                        delta = losses[base] - losses[new]
                        raw[f"{name}_delta_mse"] = delta
                        raw[f"{name}_relative_gain"] = delta / losses[base]
                        raw[f"{name}_outcome"] = outcome(delta, eps)
                    full_rows.append(raw)

    if len(full_rows) != 48 or len(search_rows) != 480 or len(scale_h_audit) != 12 or len(stitched_records) != 12:
        raise RuntimeError("v32 expected fixed target/row/candidate counts failed")
    write_csv(HERE / "FOUR_ARM_ROW_RESULTS.csv", full_rows)
    write_csv(HERE / "SCALED_CANDIDATE_SEARCH_SCORES.csv", search_rows)
    write_csv(HERE / "SCALE_H_AUDIT.csv", scale_h_audit)
    contrast_rows = summarize_contrast(full_rows)
    arm_rows = summarize_arms(full_rows)
    interaction_rows = summarize_interaction(full_rows)
    write_csv(HERE / "CONTRAST_SUMMARY.csv", contrast_rows)
    write_csv(HERE / "ARM_SUMMARY.csv", arm_rows)
    write_csv(HERE / "INTERACTION_SUMMARY.csv", interaction_rows)
    negatives = [r for r in full_rows if any(r[f"{name}_outcome"] == "loss" for name in CONTRASTS)]
    if negatives:
        write_csv(HERE / "NEGATIVE_CASES.csv", negatives)
    else:
        with (HERE / "NEGATIVE_CASES.csv").open("w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=list(full_rows[0])).writeheader()

    alpha_counts = Counter()
    for r in full_rows:
        for method in ("G01", "G11"):
            alpha_counts[(r["split"], r["layer"], r["projection"], method, r[f"{method}_alpha"])] += 1
            alpha_counts[(r["split"], "ALL", "ALL", method, r[f"{method}_alpha"])] += 1
    distributions = []
    for split in ("S0", "S1"):
        for layer, projection, n in [("ALL", "ALL", 24)] + [(li, pr, 4) for li in (0, 12, 23) for pr in ("up_proj", "down_proj")]:
            for method in ("G01", "G11"):
                for alpha in ALPHAS:
                    count = alpha_counts[(split, layer, projection, method, alpha)]
                    distributions.append({"split": split, "layer": layer, "projection": projection,
                                          "arm": method, "alpha": alpha, "row_count": count,
                                          "fraction_of_scope_rows": count / n})
    write_csv(HERE / "ALPHA_DISTRIBUTION.csv", distributions)

    cost = cost_summary(events, budget, run, preflight)
    dump(HERE / "COST_SUMMARY.json", cost)
    checks = {"status": "PASS", "protocol_sha256": protocol_sha,
              "D_identity_preflight": 2, "unquantized_FP64_FP32_FP16_spots": 2,
              "scaled_alpha1_native_exact_parity_targets": 12,
              "scaled_H_rebuilt_from_Xprime_search_targets": 12,
              "scaled_H_hashes_distinct_from_v24_unscaled": sum(r["H_changed_after_scaling"] for r in scale_h_audit),
              "candidate_pairs_verified": 120, "candidate_search_row_alpha_scores": 480,
              "v24_original_baseline_rows_reused": 480, "search_locks_reproduced": 12,
              "held_results_after_search_lock": 12, "local_G11_stitches_verified": 12,
              "four_arm_held_rows": 48, "score_formula_audit": score_check["status"],
              "max_relative_direct_vs_decomposed_score_difference": score_check["max_relative_direct_vs_decomposed_score_difference"],
              "budget_within_20_minutes": cost["cumulative_total_wall_seconds_including_all_three_GPU_processes"] <= 1200,
              "new_model_forwards": 0, "PPL_runs": 0}
    dump(HERE / "CORRECTNESS.json", checks)
    report = {"status": "COMPLETE_EXPLORATORY", "protocol_sha256": protocol_sha,
              "correctness": checks, "arm_summaries": arm_rows,
              "contrast_summaries": contrast_rows, "interaction_summaries": interaction_rows,
              "alpha_distribution": [r for r in distributions if r["layer"] == "ALL"],
              "alpha_selection_change_rows": sum(r["scaled_search_alpha_changed_vs_original"] for r in full_rows),
              "held_fp16_coordinate_drift_mse_max": max(r["held_coordinate_drift_mse"] for r in full_rows),
              "negative_row_count_any_contrast": len(negatives),
              "scaling_D_shared_between_G10_G11": True,
              "article_seen_exploratory_only": True, "whole_model_quantization": "NOT_RUN", "PPL": "NOT_RUN",
              "cost": cost}
    dump(HERE / "RUN_REPORT.json", report)
    write_report(report)
    write_reproduce()
    write_manifest()
    overall = [r for r in contrast_rows if r["level"] == "overall"]
    split = [r for r in contrast_rows if r["level"] == "split"]
    print(json.dumps({"status": report["status"], "row_units": 48, "scaled_candidates": 120,
                      "total_wall_seconds": cost["cumulative_total_wall_seconds_including_all_three_GPU_processes"],
                      "overall": [{"contrast": r["contrast"], "gain": r["pooled_relative_gain"],
                                   "W/T/L": [r["wins"], r["ties"], r["losses"]]} for r in overall],
                      "split": [{"split": r["split"], "contrast": r["contrast"],
                                 "gain": r["pooled_relative_gain"]} for r in split]},
                     ensure_ascii=False, indent=2))


def pct(value: float) -> str:
    return f"{100 * value:+.3f}%"


def write_report(report: dict) -> None:
    contrasts = report["contrast_summaries"]
    arms = report["arm_summaries"]
    interactions = report["interaction_summaries"]
    cost = report["cost"]
    lines = ["# v32 — 固定通道缩放 × scalar10 range search：2×2 局部诊断", "",
             "> 全部 64 篇文章及历史结果此前已被查看。本轮为回顾性探索筛查；仅覆盖六个 target 的四个固定输出行、两个 split。", "",
             "## 冻结方法与数据", "",
             "G00 为原坐标 α=1；G01 为原坐标 scalar10 search 选择；G10 为固定 D 的缩放坐标 α=1；G11 使用同一个 D 做 scalar10 search 选择。正的改善表示误差下降。",
             "S0 由 A=0–15 完整文章估计 D/H 并选择，B=16–31 评价；S1 由 C=32–47 估计/选择，D=48–63 评价。每篇 2048 tokens，held 从未参与 D/H 或 alpha 选择。四臂使用固定 v18 M2 前缀缓存输入及同一原始输出参考。",
             "D 的每通道统计为 search X 的 `mean(abs(X[:,j]))`，`sqrt(max(a,1e-8))` 后除以端点几何均值。D 保存为 FP32，W′ 与 X′转换为 FP16；缩放分支由 X′用原生 `GPTQ.process_batch` 重新构建 FP32 H。动态 G128、原生 compensation 和 damp 0.1 未改变。这是固定 activation-based scaling 诊断，并非官方 AWQ 或可学习缩放。", "",
             "## 正确性", "",
             "D=I 在 896/4864 两种输入宽度上精确复现 v24 H、α=1 的 Q/scale/zero/g_idx 和逐行原评分。缩放坐标 α=1 的 12/12 个 target/split 与原生 GPTQ 张量精确一致。首个 target 的 `1.0→0.8→1.0` 复跑一致，W′与 H 未受候选污染。未量化变换的 FP64/FP32/FP16 数值差异分别保存，完整 search/held 的 FP16 坐标漂移也已记录。",
             f"原始参考输出的显式 `X′Q′ᵀ−XWᵀ` 与冻结的分解评分在两个预设 target 的全部 held 文章上比较，最大相对分数差 {report['correctness']['max_relative_direct_vs_decomposed_score_difference']:.3g}；分解评分精确复现保存的 held 分数。", "",
             "## 四臂 held 结果", "",
             "下列 SSE 仅为固定行 × 16 篇 held 文章，每行 MSE×32768 汇总；不是完整 Linear 输出或 PPL。", "",
             "| Split | G00 SSE | G01 SSE | G10 SSE | G11 SSE |", "|---|---:|---:|---:|---:|"]
    for r in arms:
        if r["level"] == "split":
            lines.append(f"| {r['split']} | {r['G00_pooled_held_sse']:.6f} | {r['G01_pooled_held_sse']:.6f} | {r['G10_pooled_held_sse']:.6f} | {r['G11_pooled_held_sse']:.6f} |")
    lines.extend(["", "| Split | Comparison | Pooled SSE gain | W/T/L | Median row gain | Worst row gain | >10% regressions | Top-3 share of positive savings |",
                  "|---|---|---:|---:|---:|---:|---:|---:|"])
    for r in contrasts:
        if r["level"] == "split":
            lines.append(f"| {r['split']} | {r['contrast']} | {pct(r['pooled_relative_gain'])} | {r['wins']}/{r['ties']}/{r['losses']} | {pct(r['median_row_relative_gain'])} | {pct(r['worst_row_relative_gain'])} ({r['worst_unit_id']}) | {r['severe_gt10pct_regressions']} | {pct(r['top3_share_of_positive_savings'])} |")
    lines.extend(["", "### Target × split 的 G11 相对 G01", "",
                  "| Split | Target | Pooled SSE gain | W/T/L | Worst row gain |",
                  "|---|---|---:|---:|---:|"])
    for r in contrasts:
        if r["level"] == "target_split" and r["contrast"] == "G11_vs_G01_combination_vs_original_search":
            lines.append(f"| {r['split']} | L{r['layer']} {r['projection']} | {pct(r['pooled_relative_gain'])} | {r['wins']}/{r['ties']}/{r['losses']} | {pct(r['worst_row_relative_gain'])} ({r['worst_unit_id']}) |")
    lines.extend(["", "### Interaction", "",
                  "`I=(L10−L11)−(L00−L01)` 与每行 held MSE 处于同一单位；正值说明缩放后搜索的绝对损失下降更大，不代表 G11 必然最好。", "",
                  "| Split | Pooled I SSE | I / G00 SSE | positive / zero / negative rows |",
                  "|---|---:|---:|---:|"])
    for r in interactions:
        if r["level"] == "split":
            lines.append(f"| {r['split']} | {r['interaction_I_sse']:.6f} | {pct(r['interaction_I_sse_over_G00_sse'])} | {r['positive_I_rows']}/{r['zero_I_rows']}/{r['negative_I_rows']} |")
    combined = next(r for r in contrasts if r["level"] == "overall" and r["contrast"] == "G11_vs_G01_combination_vs_original_search")
    l23_down = next(r for r in contrasts if r["level"] == "target_both_splits" and r["layer"] == 23 and
                    r["projection"] == "down_proj" and r["contrast"] == "G11_vs_G01_combination_vs_original_search")
    remaining_base = combined["pooled_baseline_sse"] - l23_down["pooled_baseline_sse"]
    remaining_saving = combined["pooled_sse_reduction"] - l23_down["pooled_sse_reduction"]
    lines.extend(["", "## 判读与停点", "",
                  f"G11 相对 G01 的 pooled SSE 虽下降 {pct(combined['pooled_relative_gain'])}，但仅 {combined['wins']}/{combined['ties']}/{combined['losses']} 行胜/平/负，中位逐行改善 {pct(combined['median_row_relative_gain'])}；{combined['severe_gt10pct_regressions']} 行退化超过 10%，最差 {pct(combined['worst_row_relative_gain'])}。两个 split 的 pooled 改善不能代表跨 target 稳定改善。",
                  f"L23 down_proj 两个 split 单独贡献 {l23_down['pooled_sse_reduction']:.3f} SSE 净节省，超过全体的 {combined['pooled_sse_reduction']:.3f}；去掉该 target，剩余 target 的 G11 相对 G01 为 {pct(remaining_saving / remaining_base)}。L0/L12 down_proj 在两个 split 均退化，L23 up_proj 也均退化。",
                  f"G11 与 G01 的 search 选择 alpha 在 {report['alpha_selection_change_rows']}/48 个行×split 单位不同。缩放后的搜索相对同 D 的 G10 有局部收益，但不能据此推断组合优于原搜索；总体 I={next(r for r in interactions if r['level'] == 'overall')['interaction_I_sse']:.3f} SSE，为负。",
                  f"FP16 坐标变换的 held 未量化漂移逐行 MSE 最大 {report['held_fp16_coordinate_drift_mse_max']:.3g}，已经计入 G10/G11 对原参考的损失；本实验没有将缩放效果与这项舍入误差分离。", "",
                  "结论：此固定 D 方案在预定局部单位上没有稳定优于原 scalar10 搜索，应停止当前方案的扩展。不据此否定其他预先定义的 scaling 方法；若将来重启，需先冻结合法的整模型 D 折叠和新文章验证协议。", ""])
    lines.extend(["", "全部 48 行、三种对照的数值与负例分别见 `FOUR_ARM_ROW_RESULTS.csv`、`CONTRAST_SUMMARY.csv`、`NEGATIVE_CASES.csv`。完整 target/split、pooled、alpha 分布和集中度保留在原始 CSV/JSON。", "",
                  "## 成本", "",
                  f"三次 GPU 进程累计 wall {cost['cumulative_total_wall_seconds_including_all_three_GPU_processes']:.3f}s / 1200s；其中候选 bank 为 120 次完整 Linear GPTQ，另有原生 α=1 parity 12 次、D=I 2 次、状态复跑 1 次，**实际完整 Linear GPTQ 共 135 次**。搜索逐行分数 480 个；新 held 逐行分数 96 个。", 
                  f"候选 GPTQ 累计 {cost['scaled_candidate_GPTQ_seconds']:.3f}s；D 统计 {cost['D_statistic_build_seconds']:.3f}s、缩放 H {cost['scaled_H_native_process_batch_seconds']:.3f}s、因子处理 {cost['scaled_factor_prepare_seconds_including_preflight_repeat']:.3f}s、search 评分 {cost['scaled_candidate_search_scoring_seconds']:.3f}s。峰值 allocated/reserved 为 {cost['GPU_peak_allocated_bytes']:,}/{cost['GPU_peak_reserved_bytes']:,} bytes。其他阶段和未归因时间见 `COST_SUMMARY.json`。", "",
                  "## 结论边界", "",
                  "本轮仅检验一个预固定 D 的局部机制。G10/G11 使用完全相同的 D；D 没有按 held 结果或每臂分别调节。`Q_eff=Q′D^-1` 只可辅助原坐标误差分析，不能据此把它当成仍可直接按原 G128 格式 packing 的权重。整模型部署须另行解决合法 D 折叠及下游输入传播，本轮未修改模型。", "",
                  "全部文章与历史结果已被查看，本轮不构成盲测或正式泛化证据。未运行整模型量化或 PPL，不能把局部 SSE 变化称为模型质量或在线推理收益。", "",
                  "复现命令见 `REPRODUCE.md`；逐文件 SHA256 见 `OUTPUT_MANIFEST.csv`。"])
    (HERE / "REPORT.md").write_text("\n".join(lines) + "\n", "utf-8")


def write_reproduce() -> None:
    text = """# v32 reproduction\n\nFrom `E:/quant-thesis`, using the existing `.venv-gptq` and a clean v32 output copy:\n\n```powershell\n& '.venv-gptq/Scripts/python.exe' 'experiments/range_search_w4/results/range_w4_fixed_channel_scale_scalar_v32/prepare_v32.py'\n& '.venv-gptq/Scripts/python.exe' 'experiments/range_search_w4/results/range_w4_fixed_channel_scale_scalar_v32/run_v32.py' preflight\n& '.venv-gptq/Scripts/python.exe' 'experiments/range_search_w4/results/range_w4_fixed_channel_scale_scalar_v32/run_v32.py' formal\n& '.venv-gptq/Scripts/python.exe' 'experiments/range_search_w4/results/range_w4_fixed_channel_scale_scalar_v32/check_scoring_v32.py'\n& '.venv-gptq/Scripts/python.exe' 'experiments/range_search_w4/results/range_w4_fixed_channel_scale_scalar_v32/analyze_v32.py'\n```\n\nThe protocol must be frozen before the first GPU command. Existing D/H/candidate/lock/held checkpoints are validated and reused on resume; cumulative wall is capped at 1200 seconds. Re-running completed GPU commands adds their real time to the cumulative budget, so use the saved outputs for analysis. No model forward, whole-model quantization, or PPL is invoked.\n\n`raw/stitched/` contains only the four selected rows and quantizer tensors in scaled coordinates. It is a local diagnostic artifact and not a deployable checkpoint.\n"""
    (HERE / "REPRODUCE.md").write_text(text, "utf-8")


def write_manifest() -> None:
    rows = []
    for p in sorted(HERE.rglob("*")):
        if not p.is_file() or "__pycache__" in p.parts or p.name == "OUTPUT_MANIFEST.csv" or p.name.endswith(".tmp"):
            continue
        rows.append({"relative_path": p.relative_to(HERE).as_posix(), "bytes": p.stat().st_size, "sha256": sha_file(p)})
    write_csv(HERE / "OUTPUT_MANIFEST.csv", rows)


if __name__ == "__main__":
    main()
