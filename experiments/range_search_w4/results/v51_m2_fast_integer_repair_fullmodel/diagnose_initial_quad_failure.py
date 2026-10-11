"""Read-only reproduction of the first uncommitted v51 initial-regression failure."""
import csv, hashlib, json, math, runpy, time
from pathlib import Path
import numpy as np
import torch
from safetensors import safe_open
from transformers import AutoModelForCausalLM

ROOT = Path(r"E:\quant-thesis")
BASE = ROOT / "experiments/range_search_w4/results"
OUT = BASE / "v51_m2_fast_integer_repair_fullmodel"
V18 = BASE / "range_w4_fullmodel_v18"

def sha(p):
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()

def th(t):
    return hashlib.sha256(t.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()

def main():
    assert torch.__version__ == "2.6.0+cu124"
    start = json.loads((OUT / "GPU_START.json").read_text(encoding="utf-8"))
    elapsed_start = time.time() - start["start_epoch"]
    assert elapsed_start < 10800 - 600
    ns = runpy.run_path(str(OUT / "run_v51.py"), run_name="v51_initial_diag")
    # runpy returns a dict copy; update the function globals used by the deadline closure.
    ns["deadline"].__globals__["GPU_START"] = float(start["start_epoch"])
    rt = ns["rt"]
    rt.OUT = OUT
    rt.WALL_START = float(start["start_epoch"])
    rt.deadline = ns["deadline"]
    rt.MODE = "diagnostic"
    rt.LAYER = -1
    rt.MODULE = ""
    model = AutoModelForCausalLM.from_pretrained(str(V18 / "M2/checkpoint/final"), torch_dtype=torch.float16, local_files_only=True, low_cpu_mem_usage=True, attn_implementation="sdpa").eval().cuda()
    ids = rt.zero_model_check(model)
    hidden = rt.embed(model)
    groups = ["qkv", "o", "up_gate", "down"]
    layer0_src = torch.load(OUT / "original_codes/layer_00.pt", map_location="cpu", weights_only=True)
    restored = []
    for group in groups:
        checkpoint = OUT / f"formal/groups/layer_00_{group}.pt"
        cp = torch.load(checkpoint, map_location="cpu", weights_only=False)
        Xg = rt.capture(model, 0, hidden, group)
        xhash = hashlib.sha256(Xg.tobytes()).hexdigest()
        assert xhash == cp["input_sha256"], f"diagnostic replay changed at layer0/{group}"
        for name, result in cp["modules"].items():
            assert torch.equal(result["scales"], layer0_src[name]["scales"])
            assert torch.equal(result["g_idx"], layer0_src[name]["g_idx"])
            with torch.no_grad():
                model.model.layers[0].get_submodule(name).weight.copy_(result["weight"].cuda())
        restored.append({"group": group, "input_sha256": xhash, "checkpoint_sha256": sha(checkpoint)})
        del Xg, cp
    hidden = rt.advance_block(model, 0, hidden)
    rt.LAYER = 1
    rt.MODULE = "qkv"
    X = rt.capture(model, 1, hidden, "qkv")
    X_hash = hashlib.sha256(X.tobytes()).hexdigest()
    C = rt.build_C(X)
    src = torch.load(OUT / "original_codes/layer_01.pt", map_location="cpu", weights_only=True)["self_attn.k_proj"]
    with safe_open(str(rt.FP / "model.safetensors"), framework="pt", device="cpu") as sf:
        W = sf.get_tensor("model.layers.1.self_attn.k_proj.weight").half()
    device = C.device
    codes = src["codes"].to(device=device, dtype=torch.long)
    scales = src["scales"].to(device=device)
    g_idx = src["g_idx"].long().to(device)
    SC = scales[:, g_idx]
    Q = (SC * (codes.float() - 8)).half().float()
    E = Q.double() - W.to(device).double()
    p = E @ C
    initial_quad = (E * p).sum(1)
    initial_fp32 = rt.direct(X, E)

    # Independent raw-X FP64 calculation. This deliberately does not use C, p, or subtraction of large losses.
    torch.cuda.synchronize(); t0 = time.perf_counter()
    raw_sse = torch.zeros(E.shape[0], device=device, dtype=torch.float64)
    for lo in range(0, len(X), 4096):
        xx = torch.from_numpy(np.array(X[lo:lo+4096], copy=True)).to(device, dtype=torch.float64)
        yy = xx @ E.T
        raw_sse += yy.square().sum(0)
    initial_raw64 = raw_sse / len(X)
    torch.cuda.synchronize(); raw64_seconds = time.perf_counter() - t0

    tol = initial_quad.abs() * 2e-5 + 1e-12
    bad_fp32 = torch.abs(initial_fp32 - initial_quad) > tol
    bad_raw64 = torch.abs(initial_raw64 - initial_quad) > tol
    finite = bool(np.isfinite(X).all() and torch.isfinite(C).all() and torch.isfinite(W).all() and torch.isfinite(E).all() and torch.isfinite(initial_fp32).all() and torch.isfinite(initial_quad).all() and torch.isfinite(initial_raw64).all())
    rows = []
    for i in range(E.shape[0]):
        d32 = float(initial_fp32[i] - initial_quad[i])
        d64 = float(initial_raw64[i] - initial_quad[i])
        if bool(bad_fp32[i]) or bool(bad_raw64[i]):
            rows.append({"row": i, "fp32_direct": float(initial_fp32[i]), "quadratic": float(initial_quad[i]), "rawX_fp64": float(initial_raw64[i]), "fp32_minus_quad": d32, "raw64_minus_quad": d64, "frozen_tolerance": float(tol[i]), "fp32_failed": bool(bad_fp32[i]), "raw64_failed": bool(bad_raw64[i])})
    if rows:
        with (OUT / "diagnostics/INITIAL_QUAD_FAILURE_ROWS.csv").open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    max_fp32 = torch.abs(initial_fp32 - initial_quad)
    max_raw64 = torch.abs(initial_raw64 - initial_quad)
    max_rel_fp32 = max_fp32 / initial_quad.abs().clamp_min(1e-30)
    max_rel_raw64 = max_raw64 / initial_quad.abs().clamp_min(1e-30)
    report = {
        "status": "DIAGNOSTIC_COMPLETE_NO_SEARCH_RESUMED",
        "gpu_elapsed_at_diagnostic_start_seconds": elapsed_start,
        "diagnostic_wall_seconds": time.time() - (start["start_epoch"] + elapsed_start),
        "diagnostic_cumulative_elapsed_seconds": time.time() - start["start_epoch"],
        "reconstructed_from_original_M2_plus_committed_v51_block0": True,
        "layer0_restored_group_input_hashes": restored,
        "failed_group": "layer 1 self_attn.k_proj, first qkv sibling; no group committed",
        "input_shape": list(X.shape), "input_dtype": "numpy float16 saved activation representation", "input_sha256": X_hash,
        "code_sha256": th(src["codes"]), "scales_sha256": th(src["scales"]), "zeros_sha256": th(src["zeros"]), "g_idx_sha256": th(src["g_idx"].int()),
        "pristine_weight_sha256": th(W), "C_sha256": th(C), "E_sha256": th(E), "p_sha256": th(p),
        "C_dtype": str(C.dtype), "p_dtype": str(p.dtype), "E_dtype": str(E.dtype), "X_storage_dtype": "float16", "finite": finite,
        "rows": int(E.shape[0]), "row_count_failing_frozen_FP32_direct_vs_quadratic": int(bad_fp32.sum()),
        "row_count_failing_independent_rawX_FP64_vs_quadratic_at_same_tolerance": int(bad_raw64.sum()),
        "max_initial_quadratic": float(initial_quad.max()), "min_initial_quadratic": float(initial_quad.min()),
        "max_abs_fp32_minus_quadratic": float(max_fp32.max()), "max_relative_fp32_minus_quadratic": float(max_rel_fp32.max()),
        "max_abs_raw64_minus_quadratic": float(max_raw64.max()), "max_relative_raw64_minus_quadratic": float(max_rel_raw64.max()),
        "max_abs_fp32_direct_minus_rawX_FP64": float(torch.abs(initial_fp32 - initial_raw64).max()),
        "rawX_fp64_seconds": raw64_seconds,
        "frozen_isclose_rtol": 2e-5, "frozen_isclose_atol": 1e-12,
        "X_not_reused_from_historical_M2_or_v49": True, "C_is_original_v49_FP64_builder": True,
        "no_search_or_model_commit_or_PPL": True, "algorithm_or_tolerance_changed": False,
        "interpretation": "This diagnostic reconstructs the deterministic v51 prefix from original M2 plus hashed committed v51 groups. It does not resume the failed search. The original failing process did not persist X/C/p; hashes describe this checked replay."
    }
    (OUT / "diagnostics/INITIAL_QUAD_FAILURE.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    print("V51_INITIAL_QUAD_DIAGNOSTIC", json.dumps(report, ensure_ascii=False), flush=True)
    del model
    torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
