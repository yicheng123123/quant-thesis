"""Six preregistered q_proj rows: native dynamic G128 and shared-alpha search.

Run from repository root with .venv-gptq. Per-row JSON is an atomic resume unit.
No checkpoint, historical result, or test article is written or evaluated.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch

from experiments.range_search_w4.core import group_errors, prepare_hinv
from experiments.range_search_w4.dynamic_core import (
    dynamic_gptq, dynamic_gptq_batch, dynamic_params, fixed_minmax_gptq,
    native_hinv_from_documents, output_sse, weight_mse,
)
from experiments.range_search_w4.run import PROTOCOL as V7, capture, group_diagnostic, sync
from experiments.range_search_w4.run_task0_parity import OUTPUT, historical_fingerprint, tensor_hash

GRID = [round(i / 10, 1) for i in range(1, 11)]
LAYERS = (0, 12, 23)
ROWS = (0, 448)


def save(path: Path, value: object) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def timed(fn):
    sync()
    started = time.perf_counter()
    result = fn()
    sync()
    return result, time.perf_counter() - started


def rtn(w: torch.Tensor, alphas: list[float]) -> tuple[torch.Tensor, list[float], list[float]]:
    pieces, scales, zeros = [], [], []
    for group, start in enumerate(range(0, w.numel(), 128)):
        part = w[start:start + 128].reshape(1, -1)
        scale, zero = dynamic_params(part, alphas[group])
        q = ((part / scale).round() + zero).clamp(0, 15)
        pieces.append(((q - zero) * scale).flatten())
        scales.append(float(scale[0, 0]))
        zeros.append(float(zero[0, 0]))
    return torch.cat(pieces).half(), scales, zeros


def measure(q: torch.Tensor, w: torch.Tensor, xs: torch.Tensor, xh: torch.Tensor) -> dict:
    q = q.half()
    sse_s = output_sse(xs, w, q)
    sse_h = output_sse(xh, w, q)
    return {"search_sse": sse_s, "heldout_sse": sse_h,
            "search_output_mse": sse_s / xs.shape[0],
            "heldout_output_mse": sse_h / xh.shape[0],
            "weight_mse": weight_mse(w, q)}


def run_row(layer: int, row: int, w: torch.Tensor, xs: torch.Tensor,
            xh: torch.Tensor, hinv: torch.Tensor, h_seconds: float,
            *, diagnostics: bool = True) -> dict:
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    pristine_hash = tensor_hash(w)
    ng = (w.numel() + 127) // 128
    rtn_global = []
    (rtn_base, _, _), rtn_base_seconds = timed(lambda: rtn(w, [1.0] * ng))
    proxy_start = time.perf_counter()
    for alpha in GRID:
        q, scale, zero = rtn(w, [alpha] * ng)
        item = {"alpha": alpha, "scale": scale, "zero": zero}
        item.update(measure(q, w, xs, xh))
        rtn_global.append(item)
    sync()
    proxy_seconds = time.perf_counter() - proxy_start
    b_alpha = min(rtn_global, key=lambda v: v["search_output_mse"])["alpha"]

    global_candidates = []
    for alpha in GRID:
        result, seconds = timed(lambda a=alpha: dynamic_gptq(w[None], hinv, alpha=a))
        item = {"alpha": alpha, "seconds": seconds,
                "scale": result.scales[0].tolist(), "zero": result.zeros[0].tolist()}
        item.update(measure(result.weight_fp32[0], w, xs, xh))
        global_candidates.append(item)
    d = next(v for v in global_candidates if v["alpha"] == 1.0)
    b = next(v for v in global_candidates if v["alpha"] == b_alpha)
    c = min(global_candidates, key=lambda v: v["search_output_mse"])
    fixed, a_seconds = timed(lambda: fixed_minmax_gptq(w[None], hinv))
    a = {"alpha": 1.0, "seconds": a_seconds,
         "scale": fixed.scales[0].tolist(), "zero": fixed.zeros[0].tolist()}
    a.update(measure(fixed.weight_fp32[0], w, xs, xh))

    methods = {"D_native_dynamic": {**d, "candidate_evaluations": 1},
               "A_fixed_minmax": {**a, "candidate_evaluations": 1},
               "B_rtn_then_dynamic": {**b, "candidate_evaluations": 1,
                                       "rtn_evaluations": 10,
                                       "total_seconds": proxy_seconds + b["seconds"]},
               "C_gptq_dynamic_grid": {**c, "candidate_evaluations": 10,
                                       "total_seconds": sum(v["seconds"] for v in global_candidates)}}
    rerank = []
    proxy_order = sorted(rtn_global, key=lambda v: (v["search_output_mse"], v["alpha"]))
    for k in (1, 2, 3, 5):
        selected = sorted({v["alpha"] for v in proxy_order[:k]} | {1.0})
        results = []
        for alpha in selected:
            q, seconds = timed(lambda a=alpha: dynamic_gptq(w[None], hinv, alpha=a))
            rec = {"alpha": alpha, "seconds": seconds}
            rec.update(measure(q.weight_fp32[0], w, xs, xh))
            results.append(rec)
        best = min(results, key=lambda v: v["search_output_mse"])
        rerank.append({"k": k, "candidate_alphas": selected,
                       "gptq_candidate_evaluations": len(selected),
                       "rtn_candidate_evaluations": 10,
                       "seconds": proxy_seconds + sum(v["seconds"] for v in results),
                       "selected": best,
                       "candidate_results": results})

    # Independent candidates have separate W and compensation buffers.
    (batch_weights, _, _), batch_seconds = timed(lambda: dynamic_gptq_batch(w[None], hinv, [0.5, 1.0]))
    batch_max_fp16_difference = max(float((batch_weights[i, 0].half() -
        dynamic_gptq(w[None], hinv, alpha=alpha).weight_fp32[0].half()).abs().max())
        for i, alpha in enumerate((0.5, 1.0)))
    c_quantized = c_weight(w, hinv, c["alpha"])
    d_quantized = c_weight(w, hinv, 1.0)
    direct = (xs @ c_quantized.float() - xs @ w.float()).square().sum()
    delta = output_sse(xs, w, c_quantized)

    result = {"layer": layer, "row": row, "pristine_weight_sha256": pristine_hash,
              "search_x_sha256": tensor_hash(xs), "heldout_x_sha256": tensor_hash(xh),
              "shape": {"weight": list(w.shape), "search": list(xs.shape), "heldout": list(xh.shape)},
              "hessian_seconds_shared_layer": h_seconds, "rtn_base_seconds": rtn_base_seconds,
              "rtn_proxy_seconds": proxy_seconds, "rtn_global": rtn_global,
              "global_candidates": global_candidates, "methods": methods,
              "rerank": rerank,
              "checks": {"pristine_unchanged": tensor_hash(w) == pristine_hash,
                         "D_equals_C_alpha1": d == next(v for v in global_candidates if v["alpha"] == 1.0),
                         "A_scale_equals_initial_minmax": a["scale"] == d["scale"][:1] +
                           [float(dynamic_params(w[s:s+128][None], 1.0)[0][0, 0])
                            for s in range(128, w.numel(), 128)],
                         "batch_vs_sequential_fp16_max_abs": batch_max_fp16_difference,
                         "direct_output_vs_delta_relative": abs(float(direct) - delta) / max(delta, 1e-30),
                         "rerank_includes_minmax": all(1.0 in v["candidate_alphas"] for v in rerank)},
              "batch_seconds": batch_seconds}
    if diagnostics:
        group_candidates = []
        for group in range(ng):
            start, end = group * 128, min((group + 1) * 128, w.numel())
            for alpha in GRID:
                ratios = [1.0] * ng
                ratios[group] = alpha
                rq, scales, zeros = rtn(w, ratios)
                dynamic, seconds = timed(lambda rr=ratios: dynamic_gptq(w[None], hinv, alpha=rr))
                rec = {"group": group, "alpha": alpha, "seconds": seconds,
                       "rtn_scale": scales[group], "rtn_zero": zeros[group],
                       "gptq_scale": dynamic.scales[0].tolist(),
                       "gptq_zero": dynamic.zeros[0].tolist(),
                       "rtn_weight_mse": weight_mse(w, rq),
                       "rtn_group_weight_mse": weight_mse(w[start:end], rq[start:end]),
                       "rtn_group_search_output_mse": output_sse(xs[:, start:end], w[start:end], rq[start:end]) / xs.shape[0]}
                rec.update({"rtn_" + key: value for key, value in measure(rq, w, xs, xh).items()})
                rec.update({"gptq_" + key: value for key, value in measure(dynamic.weight_fp32[0], w, xs, xh).items()})
                group_candidates.append(rec)
        result["group_candidates"] = group_candidates
        old_hinv = prepare_hinv(xs, document_count=4, damp_percent=0.1)
        legacy, diagnostic_seconds = timed(lambda: group_diagnostic(w, xs, xh, old_hinv))
        historical = json.loads((OUTPUT.parent / "range_w4_local_v7" /
                                 f"layer{layer}_row{row}.json").read_text(encoding="utf-8"))["group_diagnostic"]
        delta_legacy = abs(legacy["coordinate_replace_one_sweep"]["heldout"]["full_mse"] -
                           historical["coordinate_replace_one_sweep"]["heldout"]["full_mse"])
        result["group_interaction"] = {"reproduction": legacy,
                                        "historical_coordinate_heldout_abs_diff": delta_legacy,
                                        "diagnostic_seconds": diagnostic_seconds,
                                        "dynamic_D_search": group_errors(xs, w, d_quantized, 128),
                                        "dynamic_D_heldout": group_errors(xh, w, d_quantized, 128),
                                        "dynamic_C_search": group_errors(xs, w, c_quantized, 128),
                                        "dynamic_C_heldout": group_errors(xh, w, c_quantized, 128)}
    sync()
    result["total_seconds_excluding_capture_and_hessian"] = time.perf_counter() - started
    result["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
    return result


def c_weight(w: torch.Tensor, hinv: torch.Tensor, alpha: float) -> torch.Tensor:
    return dynamic_gptq(w[None], hinv, alpha=alpha).weight_fp32[0].half()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="one row, global candidates only")
    args = parser.parse_args()
    parity = json.loads((OUTPUT / "dynamic_parity_report.json").read_text(encoding="utf-8"))
    if parity["status"] != "PASS":
        raise RuntimeError("Task 0 parity did not pass")
    manifest = json.loads((OUTPUT / "manifest.json").read_text(encoding="utf-8"))
    before = manifest["historical_v7_before"]
    if before != historical_fingerprint():
        raise RuntimeError("historical v7 changed after parity")
    save(OUTPUT / "protocol.json", {"id": "range_w4_dynamic_v8", "source_v7": V7,
        "main_semantics": "native dynamic G128 find_params timing, shared alpha across seven groups",
        "D": "dynamic GPTQ alpha=1", "A": "pristine-weight fixed MinMax G128 GPTQ",
        "B": "RTN whole-row search MSE selects alpha; final dynamic GPTQ",
        "C": "10 full dynamic GPTQ runs; search output MSE selects alpha",
        "group_interventions": "one group alpha changed with other groups alpha=1; full GPTQ rerun",
        "alpha_grid": GRID, "layers": list(LAYERS), "rows": list(ROWS),
        "rerank_k": [1, 2, 3, 5], "rounding": "torch.round ties to even",
        "integer_range": [0, 15], "zero": 8, "per_channel": True,
        "group_size": 128, "block_size": 128, "damp_percent": 0.1,
        "static_groups": False, "model_dtype": "float16", "compute_dtype": "float32"})
    start = time.perf_counter()
    docs, captured, weights, capture_peak = capture()
    capture_seconds = time.perf_counter() - start
    if [d["sha256"] for d in docs] != [d["sha256"] for d in manifest["documents"]]:
        raise RuntimeError("article hash mismatch with Task 0")
    targets = [(0, 0)] if args.smoke else [(i, r) for i in LAYERS for r in ROWS]
    hessian_times = {}
    for layer in sorted({i for i, _ in targets}):
        xs_docs = [x.cuda() for x in captured[str(layer)][:4]]
        xs = torch.cat(xs_docs)
        xh = torch.cat(captured[str(layer)][4:]).cuda()
        full = weights[str(layer)].cuda()
        module = torch.nn.Linear(full.shape[1], full.shape[0], bias=False,
                                 device="cuda", dtype=torch.float16)
        module.weight.data.copy_(full.half())
        (_, hinv, dead), seconds = timed(lambda: native_hinv_from_documents(xs_docs, module, 0.1))
        if dead:
            raise RuntimeError("unexpected dead columns")
        hessian_times[str(layer)] = seconds
        for _, row in [v for v in targets if v[0] == layer]:
            path = OUTPUT / (f"smoke_layer{layer}_row{row}.json" if args.smoke else f"layer{layer}_row{row}.json")
            if path.exists():
                print("RESUME", path.name, flush=True)
                continue
            rec = run_row(layer, row, full[row], xs, xh, hinv, seconds,
                          diagnostics=not args.smoke)
            save(path, rec)
            print("SAVED", path.name, "C/D held", rec["methods"]["C_gptq_dynamic_grid"]["heldout_output_mse"],
                  rec["methods"]["D_native_dynamic"]["heldout_output_mse"], flush=True)
        del module, full, xs, xh, hinv
        torch.cuda.empty_cache()
    save(OUTPUT / ("smoke_timing.json" if args.smoke else "run_timing.json"),
         {"capture_seconds": capture_seconds, "capture_peak_cuda": capture_peak,
          "hessian_seconds_by_layer": hessian_times,
          "invocation_seconds": time.perf_counter() - start,
          "historical_v7_unchanged": historical_fingerprint() == before})


if __name__ == "__main__":
    main()
