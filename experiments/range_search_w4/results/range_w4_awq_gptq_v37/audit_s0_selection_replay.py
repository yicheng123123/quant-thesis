"""CPU-only replay of frozen S0 alpha argmins from saved score arrays."""
from __future__ import annotations

import json
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
V36 = HERE.parent / "range_w4_awq_gptq_v36"


def replay(scores, recorded_alpha: torch.Tensor) -> dict:
    values = torch.stack([r["losses"].detach().cpu().double() for r in scores], dim=0)
    alphas = torch.tensor([float(r["alpha"]) for r in scores], dtype=torch.float64)
    # v36 tie rule: lower direct MSE wins; exact ties choose the larger alpha.
    order = torch.argsort(alphas, descending=True, stable=True)
    sorted_values = values[order]
    argmin = sorted_values.argmin(dim=0)
    chosen = alphas[order][argmin]
    replay_cast = chosen.to(dtype=recorded_alpha.dtype)
    return {
        "rows": int(recorded_alpha.numel()),
        "score_rows": int(values.shape[0]),
        "recorded_dtype": str(recorded_alpha.dtype),
        "row_alpha_exact_after_recorded_dtype_cast": bool(torch.equal(replay_cast, recorded_alpha.cpu())),
        "best_loss_exact": bool(torch.equal(sorted_values.min(dim=0).values, values.min(dim=0).values)),
        "max_alpha_abs_error_before_cast": float((chosen-recorded_alpha.cpu().double()).abs().max()),
        "selection_histogram": {str(a): int((replay_cast == a).sum()) for a in sorted(set(float(x) for x in alphas))},
    }


def main() -> None:
    selected = torch.load(V36 / "final/S0_selected_tensors.pt", map_location="cpu", weights_only=True)
    results = {}
    for coord in ("original", "scaled"):
        for proj in ("gate_proj", "up_proj"):
            bank = torch.load(V36 / f"raw/bank_S0_{coord}_{proj}.pt", map_location="cpu", weights_only=True)
            candidate_arm = "G01" if coord == "original" else "G11"
            key = "gate" if proj == "gate_proj" else "up"
            report = replay(bank["scores"], bank["best_alpha"])
            report["selected_tensor_alpha_exact"] = torch.equal(bank["best_alpha"], selected[candidate_arm][key]["selected_alpha"])
            results[f"{candidate_arm}.{key}"] = report
    for arm in ("G01", "G11"):
        down = torch.load(V36 / f"raw/downbank_S0_{arm}.pt", map_location="cpu", weights_only=True)["state"]
        results[f"{arm}.down"] = replay(down["scores"], down["selected_alpha"])
        results[f"{arm}.down"]["selected_tensor_alpha_exact"] = torch.equal(
            down["selected_alpha"], selected[arm]["down"]["selected_alpha"])
    out = {"status": "PASS" if all(r["row_alpha_exact_after_recorded_dtype_cast"] and
            r["best_loss_exact"] and r["selected_tensor_alpha_exact"] for r in results.values()) else "FAIL",
           "tie_rule": "exact score ties choose larger alpha, matching frozen v36 strict-comparison/update rule",
           "note": "The initial v37 checker compared FP32 down alpha to an FP64 decimal alpha without casting and therefore had two false mismatches; saved producer state and row argmin agree after casting to recorded dtype.",
           "results": results}
    (HERE / "correctness" / "S0_SELECTION_REPLAY.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
