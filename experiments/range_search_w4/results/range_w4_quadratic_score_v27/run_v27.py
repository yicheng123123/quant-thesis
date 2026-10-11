"""Frozen full-row M2 direct-vs-quadratic score check. Historical files are read-only."""
import csv
import hashlib
import json
import os
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

O = Path(__file__).resolve().parent
R = O.parents[3]
V = O.parent
V18 = V / "range_w4_fullmodel_v18"
V26 = V / "range_w4_factor_cache_v26"
sys.path[:0] = [str(V18), str(V / "range_w4_m2_runtime_v17_3")]
import run_precheck as rp
import optimized_native_m2 as om

P = json.loads((O / "PROTOCOL.json").read_text("utf-8"))
A = json.loads((V26 / "ACTIVATION_CACHE.json").read_text("utf-8"))
OLD = json.loads((V26 / "PROFILE_ORIGINAL.json").read_text("utf-8"))
D = torch.device("cuda:0")
GRID = [i / 10 for i in range(1, 11)]
N = 128 * 2048
BASE = R / ".hf_cache/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775/model.safetensors"
CAND = O / "candidates"
CAND.mkdir(exist_ok=True)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def thash(x):
    return hashlib.sha256(x.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def write(name, obj):
    f = O / name
    temp = f.with_suffix(f.suffix + ".tmp")
    temp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + "\n", "utf-8")
    os.replace(temp, f)


def sync():
    torch.cuda.synchronize()


def timed(fn):
    sync()
    start = time.perf_counter()
    result = fn()
    sync()
    return result, time.perf_counter() - start


class Budget:
    def __init__(self):
        old = json.loads((O / "RUN_STATUS.json").read_text("utf-8")) if (O / "RUN_STATUS.json").exists() else {}
        self.previous = float(old.get("gpu_wall_consumed_seconds", 0))
        self.start = time.perf_counter()

    def used(self):
        return self.previous + time.perf_counter() - self.start

    def check(self, label, reserve=35):
        if self.used() + reserve > 600:
            self.save("BUDGET_STOP", label=label)
            raise RuntimeError("v27 600 s GPU budget exhausted before " + label)

    def save(self, status, **extra):
        write("RUN_STATUS.json", dict(status=status, gpu_wall_consumed_seconds=self.used(), **extra))


def verify():
    for rel, expected in P["source_sha256"].items():
        assert sha(R / rel) == expected, "source changed: " + rel
    assert P["activation_sha256"] == A["input_sha256"]
    assert P["candidate_grid"] == GRID
    assert not torch.backends.cuda.matmul.allow_tf32


def input_and_weight(name):
    shape = P["targets"][name]
    path = V26 / "cache" / f"L0_{name}_128x2048.f16"
    assert path.stat().st_size == 128 * 2048 * shape[1] * 2
    assert sha(path) == A["input_sha256"][name]
    x = np.memmap(path, mode="r", dtype=np.float16, shape=(128, 2048, shape[1]))
    with safe_open(str(BASE), framework="pt", device="cpu") as f:
        w = f.get_tensor("model.layers.0.mlp." + name + ".weight").to(device=D, dtype=torch.float16).contiguous()
        assert "model.layers.0.mlp." + name + ".bias" not in f.keys()
    assert list(w.shape) == shape
    assert thash(w) == OLD["targets"][name]["pristine_weight_sha256"]
    return x, w


def raw_hessian(x, w, name, budget):
    budget.check("H_" + name)
    module = torch.nn.Linear(w.shape[1], w.shape[0], bias=False, device="meta", dtype=torch.float16)
    module.weight = torch.nn.Parameter(w.detach(), requires_grad=False)
    h, nsamples, cost = rp.build_hessian(module, x, [], "v27 " + name)
    assert nsamples == 128
    assert thash(h) == OLD["targets"][name]["H_sha256"]
    return h, cost


def make_candidates(w, h, name, budget):
    prepared, prep_s = timed(lambda: om.prepare_native_state(h, nsamples=128))
    results = []
    costs = []
    for alpha in GRID:
        budget.check(f"candidate_{name}_{alpha}")
        path = CAND / f"{name}_alpha_{alpha:.1f}.pt"
        if path.exists():
            q = torch.load(path, map_location=D, weights_only=True)
            assert q["alpha"] == alpha and q["h_sha256"] == thash(h) and q["w_sha256"] == thash(w)
            assert q["candidate_source_sha256"] == P["source_sha256"][str((V / "range_w4_m2_runtime_v17_3/optimized_native_m2.py").relative_to(R))]
            cost = dict(alpha=alpha, source="cached", gptq_wall_s=0, candidate_file_sha256=sha(path))
        else:
            native, duration = timed(lambda: om.optimized_dynamic_gptq(w, prepared, alpha=alpha, check_cholesky=(alpha == 1)))
            q = dict(alpha=alpha, h_sha256=thash(h), w_sha256=thash(w),
                     candidate_source_sha256=P["source_sha256"][str((V / "range_w4_m2_runtime_v17_3/optimized_native_m2.py").relative_to(R))],
                     weight_fp16=native.weight_fp16.detach(), scales=native.scales.detach(),
                     zeros=native.zeros.detach(), g_idx=native.g_idx.detach())
            temp = path.with_suffix(".pt.tmp")
            torch.save({k: (v.cpu() if isinstance(v, torch.Tensor) else v) for k, v in q.items()}, temp)
            os.replace(temp, path)
            cost = dict(alpha=alpha, source="generated_once", gptq_wall_s=duration, candidate_file_sha256=sha(path))
            budget.save("CANDIDATE_PROGRESS", projection=name, alpha=alpha)
        assert q["weight_fp16"].dtype == torch.float16 and torch.isfinite(q["weight_fp16"]).all()
        if q["weight_fp16"].device != D:
            q = {k: (v.to(D) if isinstance(v, torch.Tensor) else v) for k, v in q.items()}
        results.append(q)
        costs.append(cost)
    return results, prep_s, costs


def build_C(x, d):
    # Raw unweighted calibration Gram; no GPTQ damping, dead-column rewrite, inverse, or diagonal approximation.
    c = torch.zeros((d, d), dtype=torch.float32, device=D)
    for lo in range(0, 128, 4):
        xx = torch.from_numpy(np.array(x[lo:lo + 4], copy=True)).to(D).float().reshape(-1, d)
        c.addmm_(xx.T, xx)
        del xx
    c /= float(N)
    return c


def quadratic_rows(c, w, q):
    e = q.float() - w.float()
    product = e @ c
    return (e.double() * product.double()).sum(dim=1)


def score_candidate(x, w, q, *, mode="direct", c=None):
    """Optional scoring path; the frozen v18 direct scorer remains the default."""
    if mode == "direct":
        return rp.score_rows(x, w, q)
    if mode == "quadratic" and c is not None:
        return quadratic_rows(c, w, q)
    raise ValueError("mode must be direct, or quadratic with an independently built C")


def direct_grid(x, w, candidates, budget, label):
    rows = []
    for i, q in enumerate(candidates):
        budget.check(label + f"_{i}")
        score = score_candidate(x, w, q["weight_fp16"])
        rows.append(score)
    return torch.stack(rows)


def quadratic_grid(c, w, candidates, budget, label):
    rows = []
    for i, q in enumerate(candidates):
        budget.check(label + f"_{i}")
        rows.append(score_candidate(None, w, q["weight_fp16"], mode="quadratic", c=c))
    return torch.stack(rows)


def select(scores):
    best = torch.full_like(scores[0], float("inf"))
    idx = torch.full((scores.shape[1],), -1, dtype=torch.int64, device=D)
    for i, alpha in enumerate(GRID):
        better = (scores[i] < best) | ((scores[i] == best) & (i > idx))
        best = torch.where(better, scores[i], best)
        idx = torch.where(better, torch.full_like(idx, i), idx)
    sorted_scores = scores.sort(dim=0).values
    return idx, best, sorted_scores[1] - sorted_scores[0]


def stitch(candidates, idx):
    rows = torch.arange(idx.numel(), device=D)
    fields = ["weight_fp16", "scales", "zeros"]
    result = {f: torch.stack([q[f] for q in candidates])[idx, rows] for f in fields}
    result["g_idx"] = candidates[0]["g_idx"].clone()
    assert all(torch.equal(q["g_idx"], result["g_idx"]) for q in candidates)
    return result


def score_comparison(name, direct, quad, candidates, x, w, budget):
    err = (quad - direct).abs()
    rel = err / direct.abs().clamp_min(1e-30)
    di, dbest, dgap = select(direct)
    qi, qbest, qgap = select(quad)
    mismatched = (di != qi).nonzero().flatten()
    recheck = {"rows": mismatched.cpu().tolist(), "candidate_indices": [], "wall_seconds": 0.0, "exact_repeat": None}
    if mismatched.numel():
        involved = sorted(set(di[mismatched].cpu().tolist() + qi[mismatched].cpu().tolist()))
        recheck["candidate_indices"] = involved
        checks = []
        for i in involved:
            budget.check(f"direct_recheck_{name}_{i}")
            again, seconds = timed(lambda: rp.score_rows(x, w, candidates[i]["weight_fp16"]))
            recheck["wall_seconds"] += seconds
            checks.append(torch.equal(again[mismatched], direct[i, mismatched]))
        recheck["exact_repeat"] = all(checks)
    ds, qs = stitch(candidates, di), stitch(candidates, qi)
    stitch_checks = {f: torch.equal(ds[f], qs[f]) for f in ds}
    comparison = {
        "projection": name, "score_count": int(direct.numel()),
        "max_abs_error": float(err.max()), "median_abs_error": float(err.median()),
        "max_relative_error": float(rel.max()), "median_relative_error": float(rel.median()),
        "selection_difference_count": int(mismatched.numel()),
        "selection_difference_rows": mismatched.cpu().tolist(),
        "direct_tie_rows": int((dgap == 0).sum()), "quadratic_tie_rows": int((qgap == 0).sum()),
        "direct_min_best_second_gap": float(dgap.min()), "quadratic_min_best_second_gap": float(qgap.min()),
        "direct_median_best_second_gap": float(dgap.median()), "quadratic_median_best_second_gap": float(qgap.median()),
        "mismatch_direct_gaps": dgap[mismatched].cpu().tolist(),
        "mismatch_quadratic_gaps": qgap[mismatched].cpu().tolist(),
        "mismatch_direct_alpha": [GRID[i] for i in di[mismatched].cpu().tolist()],
        "mismatch_quadratic_alpha": [GRID[i] for i in qi[mismatched].cpu().tolist()],
        "selected_tensor_exact": stitch_checks,
        "selected_direct_sha256": {f: thash(ds[f]) for f in ds},
        "selected_quadratic_sha256": {f: thash(qs[f]) for f in qs},
        "direct_recheck": recheck,
    }
    return comparison, (di, qi, dgap, qgap)


def timed_score_paths(x, w, candidates, expected_direct, expected_quad, budget, repeats=2):
    records = []
    for rep in range(repeats):
        for method in ("direct", "quadratic"):
            budget.check(f"timing_{method}_{rep}", reserve=55)
            sync()
            base = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()
            if method == "direct":
                scores, score_s = timed(lambda: direct_grid(x, w, candidates, budget, f"repeat_direct_{rep}"))
                build_s = 0.0
                c_bytes = 0
            else:
                c, build_s = timed(lambda: build_C(x, w.shape[1]))
                c_bytes = c.numel() * c.element_size()
                scores, score_s = timed(lambda: quadratic_grid(c, w, candidates, budget, f"repeat_quadratic_{rep}"))
                del c
            assert torch.equal(scores, expected_direct if method == "direct" else expected_quad)
            peak = torch.cuda.max_memory_allocated()
            records.append(dict(method=method, repeat=rep + 1, C_build_wall_s=build_s,
                                score_10_wall_s=score_s, total_wall_s=build_s + score_s,
                                C_bytes=c_bytes, baseline_allocated_bytes=base,
                                peak_allocated_bytes=peak, incremental_peak_bytes=max(0, peak - base),
                                peak_reserved_bytes=torch.cuda.max_memory_reserved()))
    return records


@torch.inference_mode()
def main():
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    verify()
    budget = Budget()
    completed = json.loads((O / "RUN_REPORT.json").read_text("utf-8")) if (O / "RUN_REPORT.json").exists() else {"targets": {}}
    for name in ("up_proj", "down_proj"):
        if name in completed["targets"]:
            continue
        budget.check("projection_" + name, reserve=100)
        x, w = input_and_weight(name)
        whash = thash(w)
        h, h_cost = raw_hessian(x, w, name, budget)
        hhash = thash(h)
        candidates, prep_s, candidate_costs = make_candidates(w, h, name, budget)
        budget.check("validation_" + name, reserve=85)
        direct, direct_s = timed(lambda: direct_grid(x, w, candidates, budget, "validation_direct_" + name))
        c, c_build_s = timed(lambda: build_C(x, w.shape[1]))
        quad, quad_s = timed(lambda: quadratic_grid(c, w, candidates, budget, "validation_quad_" + name))
        c_bytes = c.numel() * c.element_size()
        del c
        assert torch.isfinite(direct).all() and torch.isfinite(quad).all()
        comparison, (di, qi, dgap, qgap) = score_comparison(name, direct, quad, candidates, x, w, budget)
        np.savez_compressed(O / f"RAW_SCORES_{name}.npz", direct=direct.cpu().numpy(), quadratic=quad.cpu().numpy(),
                            direct_selection=di.cpu().numpy(), quadratic_selection=qi.cpu().numpy(),
                            direct_best_second_gap=dgap.cpu().numpy(), quadratic_best_second_gap=qgap.cpu().numpy(),
                            alpha_grid=np.array(GRID, dtype=np.float64))
        benchmark = timed_score_paths(x, w, candidates, direct, quad, budget)
        assert thash(w) == whash and thash(h) == hhash
        completed["targets"][name] = dict(shape=P["targets"][name], input_sha256=A["input_sha256"][name],
            pristine_weight_sha256=whash, raw_H_sha256=hhash, H_build_wall_s=h_cost["seconds"],
            H_inverse_prepare_wall_s=prep_s, candidate_costs=candidate_costs,
            validation_timing=dict(direct_full10_s=direct_s, C_build_s=c_build_s,
                                   quadratic_full10_s=quad_s, C_bytes=c_bytes),
            comparison=comparison, timing_repeats=benchmark,
            raw_scores_file=f"RAW_SCORES_{name}.npz",
            raw_scores_sha256=sha(O / f"RAW_SCORES_{name}.npz"),
            original_W_and_H_unchanged=True)
        completed["status"] = "IN_PROGRESS"
        completed["gpu_wall_consumed_seconds"] = budget.used()
        write("RUN_REPORT.json", completed)
        budget.save("PROJECTION_COMPLETE", projection=name)
        print(json.dumps(dict(projection=name, mismatch=comparison["selection_difference_count"],
                              direct_s=direct_s, C_build_s=c_build_s, quadratic_s=quad_s,
                              gpu_budget_used_s=budget.used())), flush=True)
        del candidates, h, x, w, direct, quad
        torch.cuda.empty_cache()
    completed["status"] = "COMPLETE"
    completed["gpu_wall_consumed_seconds"] = budget.used()
    write("RUN_REPORT.json", completed)
    budget.save("COMPLETE")


if __name__ == "__main__":
    main()
