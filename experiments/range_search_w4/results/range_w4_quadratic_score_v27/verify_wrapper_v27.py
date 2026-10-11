"""Bounded final-code spot: one existing candidate per matrix, no GPTQ rerun."""
import json
import time
from pathlib import Path

import numpy as np
import torch

import run_v27 as r

O = Path(__file__).resolve().parent


@torch.inference_mode()
def main():
    if (O / "WRAPPER_SPOT.json").exists():
        print("WRAPPER_SPOT_REUSED")
        return
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    r.verify()
    budget = r.Budget()
    result = {}
    for name in ("up_proj", "down_proj"):
        budget.check("wrapper_spot_" + name, reserve=45)
        x, w = r.input_and_weight(name)
        cache = r.CAND / f"{name}_alpha_1.0.pt"
        q = torch.load(cache, map_location=r.D, weights_only=True)["weight_fp16"]
        saved = np.load(O / f"RAW_SCORES_{name}.npz")
        direct, direct_s = r.timed(lambda: r.score_candidate(x, w, q))
        c, build_s = r.timed(lambda: r.build_C(x, w.shape[1]))
        quadratic, quadratic_s = r.timed(lambda: r.score_candidate(None, w, q, mode="quadratic", c=c))
        direct_exact = np.array_equal(direct.cpu().numpy(), saved["direct"][9])
        quadratic_exact = np.array_equal(quadratic.cpu().numpy(), saved["quadratic"][9])
        assert direct_exact and quadratic_exact
        result[name] = dict(alpha=1.0, direct_exact=direct_exact, quadratic_exact=quadratic_exact,
                            direct_wall_s=direct_s, C_build_wall_s=build_s, quadratic_wall_s=quadratic_s)
        del x, w, q, c, direct, quadratic
    r.write("WRAPPER_SPOT.json", dict(status="PASS", targets=result, gpu_budget_used_s=budget.used(),
                                      run_v27_source_sha256=r.sha(O / "run_v27.py")))
    report = json.loads((O / "RUN_REPORT.json").read_text("utf-8"))
    report["postrun_wrapper_spot"] = result
    report["gpu_wall_consumed_seconds"] = budget.used()
    r.write("RUN_REPORT.json", report)
    budget.save("WRAPPER_SPOT_COMPLETE")
    print(json.dumps(dict(status="PASS", budget_used=budget.used(), targets=result), indent=2))


if __name__ == "__main__":
    main()
