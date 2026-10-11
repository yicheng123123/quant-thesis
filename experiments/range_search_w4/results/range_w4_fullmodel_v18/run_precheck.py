"""Matched 2048-token M2 precheck; no full-model quantization or PPL."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM


OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[3]
RESULTS = ROOT / "experiments" / "range_search_w4" / "results"
V16 = RESULTS / "range_w4_fullmodel_baseline_v16"
V171 = RESULTS / "range_w4_m2_parity_v17_1"
V173 = RESULTS / "range_w4_m2_runtime_v17_3"
MODEL_DIR = ROOT / ".hf_cache/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775"
CAL = json.loads((OUT / "CALIBRATION_MANIFEST.json").read_text(encoding="utf-8"))
GRID = [i / 10 for i in range(1, 11)]
N_DOCS = 128
SEQ = 2048
DEVICE = torch.device("cuda")
ACT_DIR = OUT / "raw" / "precheck_activations"
ACT_DIR.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(V171))
sys.path.insert(0, str(V173))
from native_m2 import native_dynamic_gptq  # noqa: E402
from optimized_native_m2 import optimized_dynamic_gptq, prepare_native_state  # noqa: E402


def sync() -> None:
    torch.cuda.synchronize()


def sha_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def sha_tensor(t: torch.Tensor) -> str:
    return sha_bytes(t.detach().cpu().contiguous().numpy().tobytes())


def sha_memmap(mm: np.memmap, chunk_docs: int = 4) -> str:
    h = hashlib.sha256()
    for i in range(0, mm.shape[0], chunk_docs):
        h.update(np.asarray(mm[i:i + chunk_docs]).tobytes(order="C"))
    return h.hexdigest()


def memmap_finite(mm: np.memmap, chunk_docs: int = 4) -> bool:
    for i in range(0, mm.shape[0], chunk_docs):
        if not np.isfinite(np.asarray(mm[i:i + chunk_docs])).all():
            return False
    return True


def dump(name: str, payload: dict) -> None:
    (OUT / name).write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def distribution(values: list[float]) -> dict:
    vals = sorted(float(v) for v in values)
    if not vals:
        return {"n": 0, "median": None, "p95_nearest_rank": None, "min": None, "max": None}
    return {
        "n": len(vals),
        "median": statistics.median(vals),
        "p95_nearest_rank": vals[max(0, math.ceil(.95 * len(vals)) - 1)],
        "min": vals[0],
        "max": vals[-1],
    }


def peak_reset() -> None:
    sync()
    torch.cuda.reset_peak_memory_stats()


def peak_read() -> tuple[int, int]:
    sync()
    return int(torch.cuda.max_memory_allocated()), int(torch.cuda.max_memory_reserved())


def new_mm(path: Path, shape: tuple[int, ...], dtype=np.float16) -> np.memmap:
    if path.exists():
        path.unlink()
    return np.memmap(path, dtype=dtype, mode="w+", shape=shape)


def record_timing(rows: list[dict], *, module: str, shape: tuple[int, int], stage: str,
                  replicate: int, alpha, seconds: float, allocated: int, reserved: int, detail: str) -> None:
    rows.append({
        "module": module,
        "weight_shape": f"{shape[0]}x{shape[1]}",
        "stage": stage,
        "replicate": replicate,
        "alpha": "" if alpha is None else alpha,
        "wall_seconds": seconds,
        "gpu_peak_allocated_bytes": allocated,
        "gpu_peak_reserved_bytes": reserved,
        "detail": detail,
    })


def cfg():
    from gptqmodel.quantization import QuantizeConfig
    return QuantizeConfig(
        bits=4, group_size=128, sym=True, desc_act=False, static_groups=False,
        true_sequential=True, damp_percent=0.1, damp_auto_increment=0, mse=0,
    )


def direct_native(module, hessian: torch.Tensor, nsamples: int):
    from gptqmodel.quantization.gptq import GPTQ
    core = GPTQ(module, qcfg=cfg())
    core.quantizer.configure(perchannel=True, bits=4, sym=True)
    core.H = hessian.detach().clone()
    core.nsamples = int(nsamples)
    q, scale, zero, g_idx, *_ = core.quantize(blocksize=128)
    return q, scale, zero, g_idx


def build_hessian(module, x_mm: np.memmap, timing_rows: list[dict], label: str) -> tuple[torch.Tensor, int, dict]:
    from gptqmodel.quantization.gptq import GPTQ
    core = GPTQ(module, qcfg=cfg())
    core.quantizer.configure(perchannel=True, bits=4, sym=True)
    core.H = torch.zeros((module.weight.shape[1], module.weight.shape[1]), dtype=torch.float32, device=DEVICE)
    core.nsamples = 0
    print(f"[hessian:start] {label} shape={tuple(module.weight.shape)} samples={N_DOCS} seq={SEQ}", flush=True)
    peak_reset()
    start = time.perf_counter()
    for i in range(N_DOCS):
        arr = np.array(x_mm[i], copy=True)
        x = torch.from_numpy(arr).unsqueeze(0).to(DEVICE)
        core.process_batch(x)
        del arr, x
    sync()
    elapsed = time.perf_counter() - start
    allocated, reserved = peak_read()
    out = core.H.detach().clone()
    nsamples = int(core.nsamples)
    record_timing(timing_rows, module=label, shape=tuple(module.weight.shape), stage="hessian_build_128x2048", replicate=1,
                  alpha=None, seconds=elapsed, allocated=allocated, reserved=reserved,
                  detail=f"native GPTQ.process_batch over {N_DOCS} documents; nsamples={nsamples}; input_sha256={sha_memmap(x_mm)}")
    print(f"[hessian:done] {label} seconds={elapsed:.3f} nsamples={nsamples}", flush=True)
    return out, nsamples, {"seconds": elapsed, "peak_allocated": allocated, "peak_reserved": reserved,
                           "nsamples": nsamples, "sha256": sha_tensor(out)}


def run_layer_capture(model, layer, hidden_mm: np.memmap, captures: dict[str, tuple[list[str], np.memmap]],
                      output_mm: np.memmap | None, stage_name: str, timing_rows: list[dict]) -> dict:
    """Forward all 128 full-length docs once; capture shared sibling inputs."""
    handles = []
    current = {"doc": -1, "group_tensors": {}, "group_seen": {}}
    capture_modules = {}
    for group, (module_names, mm) in captures.items():
        for name in module_names:
            capture_modules[name] = (group, mm)

    def make_hook(full_name, group, mm):
        def hook(_module, args):
            inp = args[0].detach()
            seen = current["group_tensors"].get(group)
            if seen is not None and not torch.equal(seen, inp):
                raise RuntimeError(f"Sibling inputs differ in {group}: {full_name}")
            if seen is None:
                current["group_tensors"][group] = inp
                current["group_seen"][group] = 1
                mm[current["doc"]] = inp[0].to("cpu").numpy()
            else:
                current["group_seen"][group] += 1
        return hook

    for name, (group, mm) in capture_modules.items():
        handles.append(layer.get_submodule(name).register_forward_pre_hook(make_hook(name, group, mm)))

    pos = torch.arange(SEQ, device=DEVICE)[None]
    per_doc = []
    peak_reset()
    for i in range(N_DOCS):
        current["doc"] = i
        current["group_tensors"].clear()
        current["group_seen"].clear()
        hidden = torch.from_numpy(np.array(hidden_mm[i], copy=True)).unsqueeze(0).to(DEVICE)
        sync()
        start = time.perf_counter()
        with torch.inference_mode():
            result = layer(
                hidden,
                attention_mask=None,
                position_ids=pos,
                cache_position=pos[0],
                position_embeddings=model.model.rotary_emb(hidden, pos),
                use_cache=False,
            )
            out = result[0]
            if output_mm is not None:
                output_mm[i] = out[0].to("cpu").numpy()
        sync()
        per_doc.append(time.perf_counter() - start)
        if set(current["group_seen"]) != set(captures):
            raise RuntimeError(f"Missing activation capture in {stage_name}, doc={i}")
        for group, (module_names, _) in captures.items():
            if current["group_seen"][group] != len(module_names):
                raise RuntimeError(f"Capture count mismatch in {stage_name}, group={group}")
        del hidden, result, out
        if (i + 1) % 32 == 0:
            print(f"[forward:{stage_name}] {i + 1}/{N_DOCS}", flush=True)
    for handle in handles:
        handle.remove()
    for _, mm in captures.values():
        mm.flush()
    if output_mm is not None:
        output_mm.flush()
    allocated, reserved = peak_read()
    total = sum(per_doc)
    first_shape = next(iter(captures.values()))[1].shape
    record_timing(timing_rows, module=stage_name, shape=(first_shape[-1], first_shape[-1]), stage="dependency_forward_capture",
                  replicate=1, alpha=None, seconds=total, allocated=allocated, reserved=reserved,
                  detail=json.dumps({"per_doc": distribution(per_doc), "input_docs": N_DOCS, "sequence_length": SEQ}, sort_keys=True))
    return {"total_seconds": total, "per_doc": distribution(per_doc), "peak_allocated_bytes": allocated,
            "peak_reserved_bytes": reserved,
            "capture_hashes": {group: sha_memmap(mm) for group, (_, mm) in captures.items()},
            "output_hash": sha_memmap(output_mm) if output_mm is not None else None}


def capture_qkv_norm(layer, hidden_mm: np.memmap, q_mm: np.memmap, timing_rows: list[dict]) -> dict:
    per_doc = []
    peak_reset()
    for i in range(N_DOCS):
        hidden = torch.from_numpy(np.array(hidden_mm[i], copy=True)).unsqueeze(0).to(DEVICE)
        sync()
        start = time.perf_counter()
        with torch.inference_mode():
            normalized = layer.input_layernorm(hidden)
            q_mm[i] = normalized[0].to("cpu").numpy()
        sync()
        per_doc.append(time.perf_counter() - start)
        del hidden, normalized
        if (i + 1) % 32 == 0:
            print(f"[capture:qkv] {i + 1}/{N_DOCS}", flush=True)
    q_mm.flush()
    allocated, reserved = peak_read()
    total = sum(per_doc)
    record_timing(timing_rows, module="L0 q/k/v shared input", shape=(896, 896), stage="shared_qkv_activation_capture",
                  replicate=1, alpha=None, seconds=total, allocated=allocated, reserved=reserved,
                  detail=json.dumps({"per_doc": distribution(per_doc), "input_docs": N_DOCS, "sequence_length": SEQ}, sort_keys=True))
    return {"total_seconds": total, "per_doc": distribution(per_doc), "peak_allocated_bytes": allocated,
            "peak_reserved_bytes": reserved, "input_sha256": sha_memmap(q_mm)}


def score_rows(x_mm: np.memmap, w: torch.Tensor, q: torch.Tensor, docs_per_chunk: int = 4) -> torch.Tensor:
    """Direct FP32 X@delta.T; accumulate squared outputs in FP64 by row."""
    delta = q.float() - w.float()
    sums = torch.zeros((w.shape[0],), dtype=torch.float64, device=DEVICE)
    for lo in range(0, N_DOCS, docs_per_chunk):
        hi = min(lo + docs_per_chunk, N_DOCS)
        x = torch.from_numpy(np.array(x_mm[lo:hi], copy=True)).to(DEVICE).float()
        out = torch.matmul(x.reshape(-1, x.shape[-1]), delta.T)
        sums += out.to(torch.float64).square().sum(0)
        del x, out
    return sums / float(N_DOCS * SEQ)


def streaming_grid(module, x_mm: np.memmap, hessian: torch.Tensor, nsamples: int,
                   timing_rows: list[dict], label: str, *, keep_full: bool) -> dict:
    pristine = module.weight.detach().clone()
    pristine_hash = sha_tensor(pristine)
    prep_seconds = 0.0
    # prepare_native_state above is timed separately to keep H/Hinv setup out
    # of the ten per-alpha full GPTQ execution measurements.
    peak_reset()
    t0 = time.perf_counter()
    prepared = prepare_native_state(hessian, nsamples=nsamples)
    sync()
    prep_seconds = time.perf_counter() - t0
    prep_alloc, prep_reserved = peak_read()
    record_timing(timing_rows, module=label, shape=tuple(pristine.shape), stage="shared_native_hinv_prepare",
                  replicate=1, alpha=None, seconds=prep_seconds, allocated=prep_alloc, reserved=prep_reserved,
                  detail=f"one prepared state shared by ten candidates; H_sha256={prepared.h_sha256}")

    candidates = {}
    best_loss = torch.full((pristine.shape[0],), float("inf"), device=DEVICE, dtype=torch.float64)
    # Alpha is protocol metadata; keep exact Python grid values through the
    # independent streaming-vs-stored selection check.
    best_alpha = torch.full((pristine.shape[0],), -1.0, device=DEVICE, dtype=torch.float64)
    best_q = torch.empty_like(pristine)
    best_scale = None
    best_zero = None
    best_g_idx = None
    candidate_seconds = []
    score_seconds = []
    losses_payload = {}
    alpha1_native_parity = None
    all_finite = True

    # Remove one same-shape kernel/context warmup from the ten measured alpha
    # calls, and separately warm the direct scoring kernels.
    peak_reset(); sync(); warm_start = time.perf_counter()
    warm = optimized_dynamic_gptq(pristine, prepared, alpha=0.7)
    sync(); warm_seconds = time.perf_counter() - warm_start
    warm_alloc, warm_reserved = peak_read()
    record_timing(timing_rows, module=label, shape=tuple(pristine.shape), stage="candidate_warmup_excluded",
                  replicate=1, alpha=0.7, seconds=warm_seconds, allocated=warm_alloc, reserved=warm_reserved,
                  detail="one full native candidate warmup excluded from candidate distribution")
    peak_reset(); sync(); warm_score_start = time.perf_counter()
    _ = score_rows(x_mm, pristine, warm.weight_fp16)
    sync(); warm_score_seconds = time.perf_counter() - warm_score_start
    warm_alloc, warm_reserved = peak_read()
    record_timing(timing_rows, module=label, shape=tuple(pristine.shape), stage="scoring_warmup_excluded",
                  replicate=1, alpha=0.7, seconds=warm_score_seconds, allocated=warm_alloc, reserved=warm_reserved,
                  detail="one all-row direct scoring warmup excluded from score distribution")
    del warm

    for alpha in GRID:
        print(f"[candidate:{label}] alpha={alpha:.1f}", flush=True)
        peak_reset()
        sync()
        start = time.perf_counter()
        result = optimized_dynamic_gptq(pristine, prepared, alpha=alpha, check_cholesky=(alpha == 1.0))
        sync()
        elapsed = time.perf_counter() - start
        alloc, reserved = peak_read()
        candidate_seconds.append(elapsed)
        record_timing(timing_rows, module=label, shape=tuple(pristine.shape), stage="optimized_full_linear_candidate",
                      replicate=1, alpha=alpha, seconds=elapsed, allocated=alloc, reserved=reserved,
                      detail="full output-row geometry; one native GPTQ core call")
        q = result.weight_fp16
        scale = result.scales
        zero = result.zeros
        g_idx = result.g_idx
        finite = bool(torch.isfinite(q).all() and torch.isfinite(scale).all() and torch.isfinite(zero).all())
        all_finite &= finite
        if not finite:
            raise RuntimeError(f"Nonfinite candidate for {label}, alpha={alpha}")
        expected_g = torch.arange(pristine.shape[1], device=DEVICE, dtype=torch.int32) // 128
        if not torch.equal(g_idx, expected_g):
            raise RuntimeError(f"Unexpected native g_idx in {label}, alpha={alpha}")
        if best_g_idx is None:
            best_g_idx = g_idx.detach().clone()
        elif not torch.equal(best_g_idx, g_idx):
            raise RuntimeError(f"g_idx changed across alpha in {label}")

        peak_reset()
        sync()
        score_start = time.perf_counter()
        losses = score_rows(x_mm, pristine, q)
        sync()
        score_elapsed = time.perf_counter() - score_start
        alloc, reserved = peak_read()
        score_seconds.append(score_elapsed)
        record_timing(timing_rows, module=label, shape=tuple(pristine.shape), stage="all_row_direct_scoring",
                      replicate=1, alpha=alpha, seconds=score_elapsed, allocated=alloc, reserved=reserved,
                      detail=f"all {pristine.shape[0]} rows; X docs streamed in four-document blocks")
        if not bool(torch.isfinite(losses).all()):
            raise RuntimeError(f"Nonfinite row loss for {label}, alpha={alpha}")
        losses_payload[alpha] = losses.detach().cpu()
        if keep_full:
            candidates[alpha] = {
                "q": q.detach().cpu().clone(), "scale": scale.detach().cpu().clone(),
                "zero": zero.detach().cpu().clone(), "g_idx": g_idx.detach().cpu().clone(),
            }

        peak_reset(); sync(); selection_start = time.perf_counter()
        better = (losses < best_loss) | ((losses == best_loss) & (alpha > best_alpha))
        best_loss = torch.where(better, losses, best_loss)
        best_alpha = torch.where(better, torch.full_like(best_alpha, alpha), best_alpha)
        best_q[better] = q[better]
        if best_scale is None:
            best_scale = torch.zeros((pristine.shape[0], scale.shape[1]), dtype=scale.dtype, device=DEVICE)
            best_zero = torch.zeros((pristine.shape[0], zero.shape[1]), dtype=zero.dtype, device=DEVICE)
        best_scale[better] = scale[better]
        best_zero[better] = zero[better]
        sync(); selection_elapsed = time.perf_counter() - selection_start
        sel_alloc, sel_reserved = peak_read()
        record_timing(timing_rows, module=label, shape=tuple(pristine.shape), stage="streaming_argmin_best_row_update",
                      replicate=1, alpha=alpha, seconds=selection_elapsed, allocated=sel_alloc, reserved=sel_reserved,
                      detail="per-row loss compare and selected W/scale/zero update")
        del result, q, scale, zero, g_idx, losses, better

    if sha_tensor(module.weight) != pristine_hash:
        raise RuntimeError(f"Candidate loop mutated pristine module weight: {label}")

    selected_alpha = [float(a) for a in best_alpha.detach().cpu().tolist()]
    stream = {
        "alpha": selected_alpha,
        "loss": best_loss.detach().cpu(),
        "q": best_q.detach().cpu().clone(),
        "scale": best_scale.detach().cpu().clone(),
        "zero": best_zero.detach().cpu().clone(),
        "g_idx": best_g_idx.detach().cpu().clone(),
    }
    if keep_full:
        # Reconstruct the result by post-hoc per-row argmin over stored full
        # candidates; exact equality is the streaming-storage correctness test.
        peak_reset(); sync(); offline_start = time.perf_counter()
        offline_loss = torch.stack([losses_payload[a] for a in GRID])
        offline_best = torch.full((pristine.shape[0],), float("inf"), dtype=torch.float64)
        offline_alpha = torch.full((pristine.shape[0],), -1.0, dtype=torch.float64)
        offline_idx = torch.full((pristine.shape[0],), -1, dtype=torch.long)
        for ci, alpha in enumerate(GRID):
            loss = offline_loss[ci]
            take = (loss < offline_best) | ((loss == offline_best) & (alpha > offline_alpha))
            offline_best[take] = loss[take]
            offline_alpha[take] = alpha
            offline_idx[take] = ci
        offline = {
            "q": torch.stack([candidates[a]["q"] for a in GRID]),
            "scale": torch.stack([candidates[a]["scale"] for a in GRID]),
            "zero": torch.stack([candidates[a]["zero"] for a in GRID]),
        }
        rows = torch.arange(pristine.shape[0])
        offline_state = {
            "alpha": [GRID[i] for i in offline_idx.tolist()],
            "loss": offline_best,
            "q": offline["q"][offline_idx, rows],
            "scale": offline["scale"][offline_idx, rows],
            "zero": offline["zero"][offline_idx, rows],
            "g_idx": candidates[GRID[0]]["g_idx"],
        }
        sync(); offline_seconds = time.perf_counter() - offline_start
        off_alloc, off_reserved = peak_read()
        record_timing(timing_rows, module=label, shape=tuple(pristine.shape), stage="stored_candidates_argmin_and_row_stitch",
                      replicate=1, alpha=None, seconds=offline_seconds, allocated=off_alloc, reserved=off_reserved,
                      detail="post-hoc full-grid per-row argmin and W/scale/zero/g_idx gather oracle")
        offline_exact = {
            key: (stream[key] == offline_state[key] if key == "alpha" else torch.equal(stream[key], offline_state[key]))
            for key in ("alpha", "loss", "q", "scale", "zero", "g_idx")
        }
        if not all(offline_exact.values()):
            raise RuntimeError(f"Streaming and stored-candidate selection differ for {label}: {offline_exact}")
    else:
        offline_exact = None

    # Materialize the streamed best rows as the final complete Linear state.
    peak_reset(); sync(); stitch_start = time.perf_counter()
    stitched = {
        "q": stream["q"].contiguous().clone(),
        "scale": stream["scale"].contiguous().clone(),
        "zero": stream["zero"].contiguous().clone(),
        "g_idx": stream["g_idx"].contiguous().clone(),
        "alpha": list(stream["alpha"]),
        "loss": stream["loss"].contiguous().clone(),
    }
    sync(); stitch_seconds = time.perf_counter() - stitch_start
    st_alloc, st_reserved = peak_read()
    record_timing(timing_rows, module=label, shape=tuple(pristine.shape), stage="streaming_final_row_stitch",
                  replicate=1, alpha=None, seconds=stitch_seconds, allocated=st_alloc, reserved=st_reserved,
                  detail="materialize complete row-selected Q/scale/zero/g_idx and alpha/loss state")

    # Verify a complete all-row commit and restore the pristine module.
    peak_reset()
    sync()
    commit_start = time.perf_counter()
    with torch.no_grad():
        module.weight.copy_(stitched["q"].to(device=module.weight.device, dtype=module.weight.dtype))
    sync()
    commit_seconds = time.perf_counter() - commit_start
    committed = torch.equal(module.weight.detach().cpu(), stitched["q"])
    commit_hash = sha_tensor(module.weight)
    with torch.no_grad():
        module.weight.copy_(pristine)
    sync()
    alloc, reserved = peak_read()
    record_timing(timing_rows, module=label, shape=tuple(pristine.shape), stage="full_linear_commit",
                  replicate=1, alpha=None, seconds=commit_seconds, allocated=alloc, reserved=reserved,
                  detail=f"rows={pristine.shape[0]}; exact commit={committed}; copy complete stitched Linear once")

    reverse_exact = None
    if keep_full:
        reverse_exact = True
        for alpha in reversed(GRID):
            print(f"[candidate-reverse:{label}] alpha={alpha:.1f}", flush=True)
            result = optimized_dynamic_gptq(pristine, prepared, alpha=alpha, check_cholesky=(alpha == 1.0))
            ref = candidates[alpha]
            eq = all(torch.equal(got.detach().cpu(), ref[key]) for got, key in [
                (result.weight_fp16, "q"), (result.scales, "scale"), (result.zeros, "zero"), (result.g_idx, "g_idx")
            ])
            reverse_exact &= eq
            del result
        if not reverse_exact:
            raise RuntimeError(f"Candidate order invariance failed for {label}")

    alpha1 = optimized_dynamic_gptq(pristine, prepared, alpha=1.0, check_cholesky=True)
    native_alpha1 = native_dynamic_gptq(pristine, hessian, nsamples=nsamples, alpha=1.0)
    alpha1_exact = all(torch.equal(a, b) for a, b in [
        (alpha1.weight_fp16, native_alpha1.weight_fp16), (alpha1.scales, native_alpha1.scales),
        (alpha1.zeros, native_alpha1.zeros), (alpha1.g_idx, native_alpha1.g_idx),
    ])
    alpha1_result_hashes = {
        "optimized_weight": sha_tensor(alpha1.weight_fp16), "native_weight": sha_tensor(native_alpha1.weight_fp16),
        "optimized_scale": sha_tensor(alpha1.scales), "native_scale": sha_tensor(native_alpha1.scales),
        "optimized_zero": sha_tensor(alpha1.zeros), "native_zero": sha_tensor(native_alpha1.zeros),
        "optimized_g_idx": sha_tensor(alpha1.g_idx), "native_g_idx": sha_tensor(native_alpha1.g_idx),
    }
    if not alpha1_exact:
        raise RuntimeError(f"Alpha=1 native parity failed for {label}")
    del alpha1, native_alpha1

    payload = {
        "module": label,
        "shape": list(pristine.shape),
        "candidate_count": len(GRID),
        "full_linear_gptq_executions": len(GRID),
        "row_alpha_score_slots": int(pristine.shape[0] * len(GRID)),
        "input_sha256": sha_memmap(x_mm),
        "hessian_sha256": prepared.h_sha256,
        "pristine_weight_sha256": pristine_hash,
        "candidate_seconds": distribution(candidate_seconds),
        "scoring_seconds": distribution(score_seconds),
        "hinv_prepare_seconds": prep_seconds,
        "commit_seconds": commit_seconds,
        "final_stitch_seconds": stitch_seconds,
        "commit_exact": committed,
        "commit_weight_sha256": commit_hash,
        "candidate_order_invariant": reverse_exact,
        "alpha1_native_exact": alpha1_exact,
        "alpha1_hashes": alpha1_result_hashes,
        "all_finite": all_finite,
        "streaming_equals_stored_candidate_argmin": offline_exact,
        "selected_alpha": stream["alpha"],
        "selected_loss": [float(v) for v in stream["loss"].tolist()],
        "selected_weight_sha256": sha_tensor(stream["q"]),
        "selected_scale_sha256": sha_tensor(stream["scale"]),
        "selected_zero_sha256": sha_tensor(stream["zero"]),
        "selected_g_idx_sha256": sha_tensor(stream["g_idx"]),
        "streamed_state": stream if keep_full else {k: v for k, v in stream.items() if k != "q"},
    }
    # Do not serialize all candidate matrices. The representative full-grid
    # oracle is retained in memory only and summarized by exactness/hashes.
    payload.pop("streamed_state")
    if keep_full:
        payload["offline_exact"] = offline_exact
        payload["reverse_candidate_exact"] = reverse_exact
        payload["selected_weight_hash_by_row"] = [sha_bytes(row.numpy().tobytes()) for row in stream["q"]]
    del candidates, losses_payload, prepared, best_q, best_scale, best_zero
    return payload, stream


def main() -> None:
    if not torch.cuda.is_available() or torch.cuda.get_device_name(0) != "NVIDIA GeForce RTX 4060 Laptop GPU":
        raise RuntimeError("Frozen RTX 4060 CUDA environment is unavailable")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(20261005)
    np.random.seed(20261005)

    # Revalidate ordered manifest content without resampling or downloading.
    cal_path = OUT / "CALIBRATION_MANIFEST.json"
    cal_hash = sha_file(cal_path)
    if cal_hash != "664419273f63df3c0b0746b18bb66b593a5384bbe6ba8afcd6f4905c9479aae3":
        raise RuntimeError("Copied v16 calibration manifest hash mismatch")
    if len(CAL["samples"]) != N_DOCS or any(len(row["token_ids"]) != SEQ for row in CAL["samples"]):
        raise RuntimeError("Frozen calibration count/length mismatch")
    if CAL["seed"] != 20261005 or CAL["token_budget"] != N_DOCS * SEQ:
        raise RuntimeError("Frozen calibration seed or token budget mismatch")
    if len({row["source_article_index"] for row in CAL["samples"]}) != N_DOCS:
        raise RuntimeError("Calibration document index is duplicated")
    for row in CAL["samples"]:
        if sha_bytes(json.dumps(row["token_ids"]).encode()) != row["token_sha256"]:
            raise RuntimeError(f"Frozen token-id hash mismatch at article {row['source_article_index']}")

    # Manifest disjointness was frozen in v16 VERIFY_AUDIT; independently
    # recheck the article-index and full-article-hash blacklist overlap.
    v14 = json.loads((RESULTS / "range_w4_calibration_coverage_v14" / "article_manifest.json").read_text(encoding="utf-8"))
    v15 = json.loads((RESULTS / "range_w4_quantized_prefix_v15" / "article_manifest.json").read_text(encoding="utf-8"))
    old = v14["historical_blacklist"] + v14["calibration"] + v14["heldout"] + v15["calibration"] + v15["heldout"]
    old_ids = {a["source_article_index"] for a in old}
    old_hashes = {a["article_sha256"] for a in old}
    cal_ids = {a["source_article_index"] for a in CAL["samples"]}
    cal_articles = {a["article_sha256"] for a in CAL["samples"]}
    if cal_ids & old_ids or cal_articles & old_hashes or len(old_ids) != CAL["historical_articles_excluded"]:
        raise RuntimeError("Frozen calibration overlap or blacklist count mismatch")
    ordered_ids = np.asarray([tok for sample in CAL["samples"] for tok in sample["token_ids"]], dtype="<i8")
    manifest_validation = {
        "status": "PASS",
        "source": str(cal_path),
        "manifest_sha256": cal_hash,
        "ordered_token_ids_sha256_i64le": sha_bytes(ordered_ids.tobytes()),
        "ordered_sample_article_indices": [a["source_article_index"] for a in CAL["samples"]],
        "samples": N_DOCS,
        "tokens_per_sample": SEQ,
        "token_budget": int(ordered_ids.size),
        "seed": CAL["seed"],
        "sample_token_hashes_verified": N_DOCS,
        "historical_blacklist_articles": len(old_ids),
        "overlap_article_indices": 0,
        "overlap_article_hashes": 0,
        "resampled": False,
    }
    dump("CALIBRATION_VALIDATION.json", manifest_validation)

    timing_rows: list[dict] = []
    peak_reset()
    load_start = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        str(MODEL_DIR), torch_dtype=torch.float16, low_cpu_mem_usage=True,
        local_files_only=True, attn_implementation="sdpa",
    ).eval().to(DEVICE)
    sync()
    model_load_seconds = time.perf_counter() - load_start
    model_alloc, model_reserved = peak_read()
    print(f"[model] loaded in {model_load_seconds:.2f}s alloc={model_alloc} reserved={model_reserved}", flush=True)
    layer = model.model.layers[0]
    next_layer = model.model.layers[1]

    hidden_path = ACT_DIR / "L0_hidden_in.f16"
    hidden_mm = new_mm(hidden_path, (N_DOCS, SEQ, model.config.hidden_size))
    peak_reset()
    embed_times = []
    for i, sample in enumerate(CAL["samples"]):
        ids = torch.tensor(sample["token_ids"], dtype=torch.long, device=DEVICE)[None]
        sync()
        start = time.perf_counter()
        with torch.inference_mode():
            h = model.model.embed_tokens(ids)
            hidden_mm[i] = h[0].to("cpu").numpy()
        sync()
        embed_times.append(time.perf_counter() - start)
        del ids, h
    hidden_mm.flush()
    emb_alloc, emb_reserved = peak_read()
    timing_rows.append({"module": "embedding", "weight_shape": "", "stage": "calibration_input_embedding",
                        "replicate": 1, "alpha": "", "wall_seconds": sum(embed_times),
                        "gpu_peak_allocated_bytes": emb_alloc, "gpu_peak_reserved_bytes": emb_reserved,
                        "detail": json.dumps({"per_doc": distribution(embed_times), "count": N_DOCS, "seq": SEQ})})

    # Capture the shared q/k/v input from the actual input layer norm. Verify
    # the direct tensor against actual projection pre-hooks on a full block.
    q_path = ACT_DIR / "L0_qkv_input.f16"
    q_mm = new_mm(q_path, (N_DOCS, SEQ, model.config.hidden_size))
    q_capture = capture_qkv_norm(layer, hidden_mm, q_mm, timing_rows)
    hook_values = {}
    hook_handles = []
    for name in ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"):
        def hook(_module, args, key=name):
            hook_values[key] = args[0].detach().clone()
        hook_handles.append(layer.get_submodule(name).register_forward_pre_hook(hook))
    first_hidden = torch.from_numpy(np.array(hidden_mm[0], copy=True)).unsqueeze(0).to(DEVICE)
    pos = torch.arange(SEQ, device=DEVICE)[None]
    with torch.inference_mode():
        layer(first_hidden, attention_mask=None, position_ids=pos, cache_position=pos[0],
              position_embeddings=model.model.rotary_emb(first_hidden, pos), use_cache=False)
    for h in hook_handles:
        h.remove()
    hook_equal = all(torch.equal(hook_values["self_attn.q_proj"], hook_values[n]) for n in ("self_attn.k_proj", "self_attn.v_proj"))
    hook_vs_direct = torch.equal(hook_values["self_attn.q_proj"], torch.from_numpy(np.array(q_mm[0], copy=True)).unsqueeze(0).to(DEVICE))
    if not (hook_equal and hook_vs_direct):
        raise RuntimeError("Direct qkv shared-input capture differs from actual target Linear pre-hooks")
    del first_hidden, hook_values
    print(f"[correctness] qkv hook parity PASS; input_sha256={q_capture['input_sha256']}", flush=True)

    # Use a shared q/k/v input/H and evaluate a full q_proj grid before any
    # M1 state mutation. This is the complete full-row streaming equivalence
    # and candidate-order smoke required for formal storage.
    q_mod = layer.get_submodule("self_attn.q_proj")
    q_h, q_ns, q_h_stats = build_hessian(q_mod, q_mm, timing_rows, "L0 q_proj")
    q_grid, q_stream = streaming_grid(q_mod, q_mm, q_h, q_ns, timing_rows, "L0 q_proj", keep_full=True)
    print("[correctness] q full-row streaming/stitch/order/native parity PASS", flush=True)

    # Native M1 self-prefix: alpha=1 only, independent from M2. Commit k/v/q
    # from the same qkv group entry and then refresh the actual next boundary.
    for name in ("self_attn.k_proj", "self_attn.v_proj", "self_attn.q_proj"):
        mod = layer.get_submodule(name)
        start = time.perf_counter(); sync(); peak_reset()
        native = direct_native(mod, q_h, q_ns)
        sync(); elapsed = time.perf_counter() - start
        alloc, reserved = peak_read()
        record_timing(timing_rows, module=f"L0 {name}", shape=tuple(mod.weight.shape), stage="M1_native_alpha1_gptq",
                      replicate=1, alpha=1.0, seconds=elapsed, allocated=alloc, reserved=reserved,
                      detail="independent native GPTQ alpha=1 M1 prefix")
        with torch.no_grad():
            mod.weight.copy_(native[0])
        del native

    o_mm = new_mm(ACT_DIR / "L0_o_input_m1_prefix.f16", (N_DOCS, SEQ, model.config.hidden_size))
    o_cap = run_layer_capture(model, layer, hidden_mm,
                              {"o": (["self_attn.o_proj"], o_mm)}, None,
                              "qkv_commit_to_o_input", timing_rows)
    o_mod = layer.get_submodule("self_attn.o_proj")
    o_h, o_ns, o_h_stats = build_hessian(o_mod, o_mm, timing_rows, "L0 o_proj")
    start=time.perf_counter(); sync(); peak_reset(); native=direct_native(o_mod,o_h,o_ns); sync(); elapsed=time.perf_counter()-start
    alloc,reserved=peak_read(); record_timing(timing_rows,module="L0 self_attn.o_proj",shape=tuple(o_mod.weight.shape),stage="M1_native_alpha1_gptq",replicate=1,alpha=1.0,seconds=elapsed,allocated=alloc,reserved=reserved,detail="native M1 sequential prefix")
    with torch.no_grad(): o_mod.weight.copy_(native[0])
    del native

    up_mm = new_mm(ACT_DIR / "L0_upgate_input_m1_prefix.f16", (N_DOCS, SEQ, model.config.hidden_size))
    up_cap = run_layer_capture(model, layer, hidden_mm,
                               {"upgate": (["mlp.up_proj", "mlp.gate_proj"], up_mm)}, None,
                               "o_commit_to_up_gate_input", timing_rows)
    up_mod = layer.get_submodule("mlp.up_proj")
    up_h, up_ns, up_h_stats = build_hessian(up_mod, up_mm, timing_rows, "L0 up_proj")
    for name in ("mlp.up_proj", "mlp.gate_proj"):
        mod=layer.get_submodule(name); start=time.perf_counter(); sync(); peak_reset(); native=direct_native(mod,up_h,up_ns); sync(); elapsed=time.perf_counter()-start
        alloc,reserved=peak_read(); record_timing(timing_rows,module=f"L0 {name}",shape=tuple(mod.weight.shape),stage="M1_native_alpha1_gptq",replicate=1,alpha=1.0,seconds=elapsed,allocated=alloc,reserved=reserved,detail="native M1 sequential prefix; up/gate shared input and H")
        with torch.no_grad(): mod.weight.copy_(native[0])
        del native

    down_mm = new_mm(ACT_DIR / "L0_down_input_m1_prefix.f16", (N_DOCS, SEQ, layer.get_submodule("mlp.down_proj").weight.shape[1]))
    down_cap = run_layer_capture(model, layer, hidden_mm,
                                 {"down": (["mlp.down_proj"], down_mm)}, None,
                                 "up_gate_commit_to_down_input", timing_rows)
    down_mod = layer.get_submodule("mlp.down_proj")
    down_h, down_ns, down_h_stats = build_hessian(down_mod, down_mm, timing_rows, "L0 down_proj")

    # Add the up-projection shape because it differs from both q and down.
    up_grid, up_stream = streaming_grid(up_mod, up_mm, up_h, up_ns, timing_rows, "L0 up_proj", keep_full=False)
    down_grid, down_stream = streaming_grid(down_mod, down_mm, down_h, down_ns, timing_rows, "L0 down_proj", keep_full=False)
    print("[correctness] q/up/down alpha1 native parity and finite full-row grids PASS", flush=True)

    # Commit native M1 down, then measure the final layer-local refresh to the
    # actual next-layer q/k/v input, including the next input layer norm.
    peak_reset(); sync(); start=time.perf_counter()
    down_native=direct_native(down_mod,down_h,down_ns)
    sync(); elapsed=time.perf_counter()-start; alloc,reserved=peak_read()
    record_timing(timing_rows,module="L0 mlp.down_proj",shape=tuple(down_mod.weight.shape),stage="M1_native_alpha1_gptq",replicate=1,alpha=1.0,seconds=elapsed,allocated=alloc,reserved=reserved,detail="native M1 sequential prefix")
    commit_start=time.perf_counter()
    with torch.no_grad(): down_mod.weight.copy_(down_native[0])
    sync(); commit_elapsed=time.perf_counter()-commit_start
    record_timing(timing_rows,module="L0 mlp.down_proj",shape=tuple(down_mod.weight.shape),stage="M1_module_commit",replicate=1,alpha=None,seconds=commit_elapsed,allocated=alloc,reserved=reserved,detail="commit one native quantized Linear")
    del down_native
    peak_reset(); sync(); start=time.perf_counter()
    next_q_mm = new_mm(ACT_DIR / "L1_qkv_input_m1_prefix.f16", (N_DOCS, SEQ, model.config.hidden_size))
    next_norm = next_layer.input_layernorm
    pos = torch.arange(SEQ,device=DEVICE)[None]
    per_doc=[]
    for i in range(N_DOCS):
        hidden=torch.from_numpy(np.array(hidden_mm[i],copy=True)).unsqueeze(0).to(DEVICE)
        sync(); one=time.perf_counter()
        with torch.inference_mode():
            output=layer(hidden,attention_mask=None,position_ids=pos,cache_position=pos[0],position_embeddings=model.model.rotary_emb(hidden,pos),use_cache=False)[0]
            qnext=next_norm(output)
            next_q_mm[i]=qnext[0].to("cpu").numpy()
        sync(); per_doc.append(time.perf_counter()-one)
        del hidden,output,qnext
    next_q_mm.flush(); alloc,reserved=peak_read(); total=sum(per_doc)
    print(f"[forward:down-to-next] {N_DOCS}/{N_DOCS}", flush=True)
    record_timing(timing_rows,module="L0 down_proj to L1 qkv",shape=(896,896),stage="down_commit_to_next_layer_input",replicate=1,alpha=None,seconds=total,allocated=alloc,reserved=reserved,detail=json.dumps({"per_doc":distribution(per_doc),"count":N_DOCS,"seq":SEQ,"next_input_sha256":sha_memmap(next_q_mm)},sort_keys=True))
    final_forward = {"total_seconds": total, "per_doc": distribution(per_doc),
                     "peak_allocated_bytes": alloc, "peak_reserved_bytes": reserved,
                     "next_input_hash": sha_memmap(next_q_mm)}

    # Full 10-alpha full-shape cost on M1-prefix L0 up/down inputs was measured
    # above. Record peak GPU state and all correctness conditions.
    stage_refresh = {
        "qkv_shared_input_capture": q_capture,
        "qkv_commit_to_o_input": o_cap,
        "o_commit_to_up_gate_input": up_cap,
        "up_gate_commit_to_down_input": down_cap,
        "down_commit_to_next_layer": {**final_forward, "next_layer_qkv_input_sha256": sha_memmap(next_q_mm)},
    }
    matched = {
        "status": "PASS",
        "model_revision": CAL["tokenizer_revision"],
        "calibration_manifest_sha256": cal_hash,
        "calibration_ordered_token_ids_sha256_i64le": manifest_validation["ordered_token_ids_sha256_i64le"],
        "samples": N_DOCS,
        "sequence_length": SEQ,
        "total_tokens": N_DOCS * SEQ,
        "peak_gpu_allocated_bytes": max([model_alloc, emb_alloc] + [r["gpu_peak_allocated_bytes"] for r in timing_rows]),
        "peak_gpu_reserved_bytes": max([model_reserved, emb_reserved] + [r["gpu_peak_reserved_bytes"] for r in timing_rows]),
        "activation_refresh_stages": stage_refresh,
        "hessian_groups": {"qkv": q_h_stats, "o": o_h_stats, "up_gate": up_h_stats, "down": down_h_stats},
        "full_shape_grids": {"L0 q_proj": q_grid, "L0 up_proj": up_grid, "L0 down_proj": down_grid},
        "model_load_seconds": model_load_seconds,
        "model_load_peak_allocated_bytes": model_alloc,
        "model_load_peak_reserved_bytes": model_reserved,
        "checks": {
            "calibration_manifest_and_order_hash": True,
            "qkv_hook_equals_shared_norm": bool(hook_equal and hook_vs_direct),
            "alpha1_native_parity_q_up_down": all(x["alpha1_native_exact"] for x in [q_grid, up_grid, down_grid]),
            "q_fullrow_streaming_equals_stored_argmin": all(q_grid["offline_exact"].values()),
            "q_candidate_order_invariance": q_grid["candidate_order_invariant"],
            "full_row_commit_exact_q": q_grid["commit_exact"],
            "up_down_full_candidate_finite": up_grid["all_finite"] and down_grid["all_finite"],
            "qkv_and_upgate_sibling_input_equality": True,
            "all_candidate_g_idx_native": True,
            "activation_hashes_finite": all(memmap_finite(mm) for mm in [hidden_mm, q_mm, o_mm, up_mm, down_mm, next_q_mm]),
        },
        "formal_full_model_started": False,
        "ppl_started": False,
    }
    required = all(matched["checks"].values())
    matched["status"] = "PASS" if required else "FAIL"
    dump("MATCHED_SHAPE_PRECHECK.json", matched)
    with (OUT / "MATCHED_SHAPE_TIMING.csv").open("w", newline="", encoding="utf-8") as f:
        fields = ["module", "weight_shape", "stage", "replicate", "alpha", "wall_seconds", "gpu_peak_allocated_bytes", "gpu_peak_reserved_bytes", "detail"]
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader(); writer.writerows(timing_rows)
    if not required:
        dump("correctness_checks.json", {"status": "FORMAL_M2_PRECHECK_FAIL", "checks": matched["checks"]})
        raise RuntimeError("FORMAL_M2_PRECHECK_FAIL; STOP")

    # Cost model is added by cost_model_v18.py after this measured evidence is
    # reviewed. Drop large activation mappings before returning to the caller.
    del model, hidden_mm, q_mm, o_mm, up_mm, down_mm, next_q_mm
    torch.cuda.empty_cache()
    print(json.dumps({"precheck": matched["status"], "peak_allocated_bytes": matched["peak_gpu_allocated_bytes"],
                      "q_seconds": q_grid["candidate_seconds"], "up_seconds": up_grid["candidate_seconds"],
                      "down_seconds": down_grid["candidate_seconds"]}, indent=2))


if __name__ == "__main__":
    main()
