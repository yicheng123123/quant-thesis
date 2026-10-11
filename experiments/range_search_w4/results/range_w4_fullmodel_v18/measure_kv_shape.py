"""Matched full-shape W4 alpha timing/correctness for native GQA k/v linears."""
from __future__ import annotations

import csv
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM


OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[3]
V171 = ROOT / "experiments/range_search_w4/results/range_w4_m2_parity_v17_1"
V173 = ROOT / "experiments/range_search_w4/results/range_w4_m2_runtime_v17_3"
MODEL_DIR = ROOT / ".hf_cache/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775"
sys.path.insert(0, str(V171))
sys.path.insert(0, str(V173))
sys.path.insert(0, str(OUT))
import run_precheck as rp  # noqa: E402
from native_m2 import native_dynamic_gptq  # noqa: E402
from optimized_native_m2 import optimized_dynamic_gptq, prepare_native_state  # noqa: E402


def sync():
    torch.cuda.synchronize()


def sha(t):
    return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def record(rows, name, shape, stage, seconds, detail, alpha=None):
    alloc, reserved = rp.peak_read()
    rows.append({"module": name, "weight_shape": f"{shape[0]}x{shape[1]}", "stage": stage,
                 "replicate": len([r for r in rows if r["module"] == name and r["stage"] == stage]) + 1,
                 "alpha": "" if alpha is None else alpha, "wall_seconds": seconds,
                 "gpu_peak_allocated_bytes": alloc, "gpu_peak_reserved_bytes": reserved,
                 "detail": detail})


def main():
    torch.manual_seed(20261005)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda")
    x_path = OUT / "raw/precheck_activations/L0_qkv_input.f16"
    x = np.memmap(x_path, dtype=np.float16, mode="r", shape=(128, 2048, 896))
    x_hash = rp.sha_memmap(x)
    model = AutoModelForCausalLM.from_pretrained(
        str(MODEL_DIR), torch_dtype=torch.float16, low_cpu_mem_usage=True,
        local_files_only=True, attn_implementation="sdpa",
    ).eval().to(device)
    layer = model.model.layers[0]
    timings = []
    k_mod = layer.self_attn.k_proj
    h, ns, hstats = rp.build_hessian(k_mod, x, timings, "L0 kv_shared_input")
    if ns != 128:
        raise RuntimeError(f"expected 128 calibration documents, got {ns}")
    prepared = prepare_native_state(h, nsamples=ns)
    grid = [i / 10 for i in range(1, 11)]
    summary = {"status": "PASS", "input_sha256": x_hash, "hessian_sha256": hstats["sha256"],
               "samples": 128, "sequence_length": 2048, "modules": {}, "checks": {}}

    for short, mod in (("k", layer.self_attn.k_proj), ("v", layer.self_attn.v_proj)):
        name = f"L0 self_attn.{short}_proj"
        pristine = mod.weight.detach().clone()
        before = sha(pristine)
        # Native alpha=1 parity for the exact low-output-row Linear shape.
        ref = native_dynamic_gptq(pristine, h, nsamples=ns, alpha=1.0)
        got = optimized_dynamic_gptq(pristine, prepared, alpha=1.0, check_cholesky=True)
        parity = {"weight": torch.equal(ref.weight_fp16, got.weight_fp16),
                  "scale": torch.equal(ref.scales, got.scales),
                  "zero": torch.equal(ref.zeros, got.zeros),
                  "g_idx": torch.equal(ref.g_idx, got.g_idx)}
        if not all(parity.values()):
            raise RuntimeError(f"alpha1 parity failure for {name}: {parity}")
        del ref, got

        best_loss = torch.full((pristine.shape[0],), float("inf"), device=device, dtype=torch.float64)
        best_alpha = torch.full_like(best_loss, -1.0)
        best_q = torch.empty_like(pristine)
        best_scale = best_zero = best_g = None
        row_info = []
        for alpha in grid:
            rp.peak_reset(); sync(); start = time.perf_counter()
            result = optimized_dynamic_gptq(pristine, prepared, alpha=alpha, check_cholesky=(alpha == 1.0))
            sync(); elapsed = time.perf_counter() - start
            record(timings, name, tuple(pristine.shape), "optimized_full_linear_candidate", elapsed,
                   "full [128,896] native Linear output rows; shared qkv Hinv", alpha)
            q = result.weight_fp16
            if not (torch.isfinite(q).all() and torch.isfinite(result.scales).all() and torch.isfinite(result.zeros).all()):
                raise RuntimeError(f"nonfinite candidate {name} alpha={alpha}")
            if best_g is None:
                best_g = result.g_idx.detach().clone()
                best_scale = torch.zeros_like(result.scales)
                best_zero = torch.zeros_like(result.zeros)
            elif not torch.equal(best_g, result.g_idx):
                raise RuntimeError(f"g_idx mismatch at {name} alpha={alpha}")
            rp.peak_reset(); sync(); start = time.perf_counter()
            losses = rp.score_rows(x, pristine, q)
            sync(); elapsed = time.perf_counter() - start
            record(timings, name, tuple(pristine.shape), "all_row_direct_scoring", elapsed,
                   f"all {pristine.shape[0]} output rows, 128x2048 X", alpha)
            better = (losses < best_loss) | ((losses == best_loss) & (alpha > best_alpha))
            rp.peak_reset(); sync(); start = time.perf_counter()
            best_loss = torch.where(better, losses, best_loss)
            best_alpha = torch.where(better, torch.full_like(best_alpha, alpha), best_alpha)
            best_q[better] = q[better]
            best_scale[better] = result.scales[better]
            best_zero[better] = result.zeros[better]
            sync(); elapsed = time.perf_counter() - start
            record(timings, name, tuple(pristine.shape), "streaming_argmin_best_row_update", elapsed,
                   "per-row exact-loss tie rule prefers larger alpha", alpha)
            row_info.append(losses.detach().cpu().tolist())
            del result, q, losses, better

        alphas = [float(v) for v in best_alpha.detach().cpu().tolist()]
        rp.peak_reset(); sync(); start = time.perf_counter()
        stitched = best_q.detach().clone()
        stitched_scale = best_scale.detach().clone()
        stitched_zero = best_zero.detach().clone()
        sync(); elapsed = time.perf_counter() - start
        record(timings, name, tuple(pristine.shape), "streaming_final_row_stitch", elapsed,
               "all selected output rows; scale/zero stitched by output row")
        rp.peak_reset(); sync(); start = time.perf_counter()
        with torch.no_grad():
            mod.weight.copy_(stitched)
        sync(); elapsed = time.perf_counter() - start
        record(timings, name, tuple(pristine.shape), "full_linear_commit", elapsed,
               "one whole Linear commit after all output rows selected")
        with torch.no_grad():
            mod.weight.copy_(pristine)
        hist = {f"{a:.1f}": sum(v == a for v in alphas) for a in grid}
        summary["modules"][short] = {
            "shape": list(pristine.shape), "alpha1_native_parity": parity,
            "all_candidates_finite": True, "g_idx_constant_across_grid": True,
            "selected_alpha_histogram": hist,
            "candidate_loss_rows": len(row_info) * pristine.shape[0],
            "pristine_weight_unchanged_after_probe": sha(mod.weight) == before,
            "selected_weight_sha256": sha(stitched),
            "selected_scale_shape": list(stitched_scale.shape), "selected_zero_shape": list(stitched_zero.shape),
        }
        if not summary["modules"][short]["pristine_weight_unchanged_after_probe"]:
            raise RuntimeError(f"probe left model mutated: {name}")
        del pristine, best_q, best_scale, best_zero, best_g, stitched, stitched_scale, stitched_zero
        torch.cuda.empty_cache()

    with (OUT / "KV_SHAPE_TIMING.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(timings[0]))
        writer.writeheader(); writer.writerows(timings)
    (OUT / "KV_SHAPE_PRECHECK.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": summary["status"], "modules": summary["modules"],
                      "timing_rows": len(timings)}, indent=2))


if __name__ == "__main__":
    main()
