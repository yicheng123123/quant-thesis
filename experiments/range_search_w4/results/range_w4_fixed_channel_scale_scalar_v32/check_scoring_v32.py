"""Post-lock numerical check of v32's scaled score against an explicit original output."""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import torch

import run_v32 as v


HERE = Path(__file__).resolve().parent
TARGETS = [(0, "up_proj", "S0"), (0, "down_proj", "S0")]


def check_one(layer: int, projection: str, split: str) -> dict:
    stem = v.target_stem(layer, projection, split)
    x_mm = v.input_mm(layer, projection)
    w = v.pristine_weight(layer, projection, split)
    data = torch.load(HERE / "raw/scales" / f"{stem}.pt", map_location="cpu", weights_only=True)
    d = data["D"].to(v.DEVICE)
    wp = v.transformed_weight(w, d)
    held = json.loads((HERE / "raw" / f"{stem}_HELD_RESULTS.json").read_text("utf-8"))
    lock = json.loads((HERE / "raw" / f"{stem}_SEARCH_LOCK.json").read_text("utf-8"))
    rows = v.ROWS[projection]
    q10 = torch.load(v.candidate_paths(stem, 1.0)[0], map_location="cpu", weights_only=True)["weight_fp16"].to(v.DEVICE)
    q_by_alpha = {a: torch.load(v.candidate_paths(stem, a)[0], map_location="cpu", weights_only=True)["weight_fp16"]
                  for a in set(float(value) for value in lock["selected_alpha_by_row"].values())}
    q11 = torch.stack([q_by_alpha[float(lock["selected_alpha_by_row"][str(row)])][j]
                       for j, row in enumerate(rows)]).to(v.DEVICE)
    result = {"stem": stem, "held_doc_indices": v.DOCS[split]["held"], "rows": rows,
              "alpha1": {}, "selected_scalar": {}}
    for label, q in (("alpha1", q10), ("selected_scalar", q11)):
        direct_total = torch.zeros((4,), dtype=torch.float64, device=v.DEVICE)
        decomposed_total = torch.zeros((4,), dtype=torch.float64, device=v.DEVICE)
        max_abs = 0.0
        for start in range(0, 16, 2):
            x = torch.from_numpy(np.array(x_mm[v.DOCS[split]["held"][start:start + 2]], copy=True)).to(v.DEVICE)
            xf = x.float().reshape(-1, x.shape[-1])
            xp = (x.float() / d).half().float().reshape(-1, x.shape[-1])
            original_reference = xf @ w[rows].float().T
            offset = xp @ wp[rows].float().T - original_reference
            decomposed = xp @ (q.float() - wp[rows].float()).T + offset
            direct = xp @ q.float().T - original_reference
            direct_total += direct.double().square().sum(0)
            decomposed_total += decomposed.double().square().sum(0)
            max_abs = max(max_abs, float((direct - decomposed).abs().max().item()))
        direct_mse = (direct_total / (16 * v.SEQ)).cpu().tolist()
        decomp_mse = (decomposed_total / (16 * v.SEQ)).cpu().tolist()
        saved = [float(next(r for r in held["rows"] if int(r["row"]) == row)[
            "G10_held_mse" if label == "alpha1" else "G11_held_mse"]) for row in rows]
        exact = [a == b for a, b in zip(decomp_mse, saved)]
        if not all(exact):
            raise RuntimeError(f"decomposed score fails saved result exact parity: {stem}/{label}: {exact}")
        result[label] = {"decomposed_mse_by_row": {str(r): x for r, x in zip(rows, decomp_mse)},
                         "direct_original_reference_mse_by_row": {str(r): x for r, x in zip(rows, direct_mse)},
                         "decomposed_equals_saved_held_exact_by_row": {str(r): x for r, x in zip(rows, exact)},
                         "max_abs_output_error_difference_FP32": max_abs,
                         "max_abs_score_mse_difference": max(abs(a - b) for a, b in zip(decomp_mse, direct_mse)),
                         "max_relative_score_mse_difference": max(abs(a - b) / a for a, b in zip(decomp_mse, direct_mse))}
    return result


def main() -> None:
    if json.loads((HERE / "RUN_STATUS.json").read_text("utf-8"))["status"] != "COMPLETE":
        raise RuntimeError("run formal and lock selections first")
    budget = v.Budget("score_formula_audit")
    try:
        torch.cuda.init()
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        budget.check("post-lock original-reference score audit")
        started = time.perf_counter()
        records = [check_one(*args) for args in TARGETS]
        v.sync()
        wall = time.perf_counter() - started
        outcome = {"status": "PASS", "protocol_sha256": v.sha_file(v.PROTOCOL_PATH),
                   "scope": "preselected L0 up/down S0, all 16 held documents, all four fixed rows; alpha1 and search-selected scalar",
                   "meaning": "decomposed v32 score exactly reproduces saved values; explicit Yprime - original Y differs only in FP32 arithmetic order",
                   "records": records, "wall_seconds": wall,
                   "max_relative_direct_vs_decomposed_score_difference": max(
                       record[label]["max_relative_score_mse_difference"]
                       for record in records for label in ("alpha1", "selected_scalar"))}
        v.dump(HERE / "SCORING_FORMULA_AUDIT.json", outcome)
        budget.record("score_formula_audit_pass", seconds=wall,
                      max_relative_score_difference=outcome["max_relative_direct_vs_decomposed_score_difference"])
        budget.finish("COMPLETE")
        print(json.dumps({"status": "PASS", "wall_seconds": wall,
                          "cumulative_wall_seconds": budget.elapsed(),
                          "max_relative_score_difference": outcome["max_relative_direct_vs_decomposed_score_difference"]}, indent=2))
    except Exception as exc:
        budget.finish("FAIL", error=repr(exc))
        raise


if __name__ == "__main__":
    main()
