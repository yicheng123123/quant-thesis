"""Run the bounded v17.3 reuse audit, 32-row equivalence, and timing study."""
from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import statistics
import sys
import time
from pathlib import Path

import torch
from safetensors import safe_open
from transformers import AutoConfig
from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer, Qwen2RotaryEmbedding

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[3]
V16 = OUT.parent / "range_w4_fullmodel_baseline_v16"
V171 = OUT.parent / "range_w4_m2_parity_v17_1"
V172 = OUT.parent / "range_w4_fullmodel_smoke_v17_2"
MODEL_REV = "7ae557604adf67be50417f59c2c2f167def9a775"
MODEL = ROOT / ".hf_cache/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots" / MODEL_REV
GRID = [i / 10 for i in range(1, 11)]
sys.path.insert(0, str(V171))
sys.path.insert(0, str(OUT))
from native_m2 import native_dynamic_gptq  # noqa: E402
from optimized_native_m2 import optimized_dynamic_gptq, prepare_native_state  # noqa: E402


def sha(t):
    return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def sha_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(name, obj):
    (OUT / name).write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sync():
    torch.cuda.synchronize()


def stats(values):
    vals = sorted(float(x) for x in values)
    return {
        "n": len(vals),
        "mean": statistics.mean(vals),
        "median": statistics.median(vals),
        "p95_nearest_rank": vals[max(0, math.ceil(0.95 * len(vals)) - 1)],
        "min": vals[0],
        "max": vals[-1],
    }


def maxdiff(a, b):
    if a.numel() == 0:
        return 0.0
    return float((a.float() - b.float()).abs().max())


def make_layer(state):
    config = AutoConfig.from_pretrained(str(MODEL), local_files_only=True)
    config._attn_implementation = "sdpa"
    layer = Qwen2DecoderLayer(config, layer_idx=0).half().to("cuda").eval()
    layer.load_state_dict(state, strict=True)
    return layer, Qwen2RotaryEmbedding(config=config).to("cuda")


def forward_capture(layer, rope, docs, names):
    outputs = {name: [] for name in names}
    handles = []
    for name in names:
        def hook(module, args, key=name):
            outputs[key].append(args[0].detach().clone())
        handles.append(layer.get_submodule(name).register_forward_pre_hook(hook))
    with torch.inference_mode():
        for doc in docs:
            pos = torch.arange(doc.shape[1], device="cuda")[None]
            layer(
                doc,
                attention_mask=None,
                position_ids=pos,
                cache_position=pos[0],
                position_embeddings=rope(doc, pos),
                use_cache=False,
            )
    for handle in handles:
        handle.remove()
    return outputs


def build_hessian(module, batches):
    from gptqmodel.quantization import QuantizeConfig
    from gptqmodel.quantization.gptq import GPTQ

    qcfg = QuantizeConfig(
        bits=4, group_size=128, sym=True, desc_act=False,
        static_groups=False, true_sequential=True, damp_percent=0.1,
        damp_auto_increment=0, mse=0,
    )
    core = GPTQ(module, qcfg=qcfg)
    core.quantizer.configure(perchannel=True, bits=4, sym=True)
    sync()
    start = time.perf_counter()
    for batch in batches:
        core.process_batch(batch)
    sync()
    return core.H.clone(), int(core.nsamples), time.perf_counter() - start


def profile_reference(w, h, nsamples):
    """One diagnostic native call; instrumented and excluded from benchmarks."""
    import gptqmodel.quantization.gptq as gptq_module
    from gptqmodel.quantization.gptq import GPTQ
    from gptqmodel.quantization.quantizer import Quantizer

    original_find = Quantizer.find_params
    original_quantize = Quantizer.quantize
    original_inverse = GPTQ.block_cholesky_inverse
    original_sync = gptq_module.torch_sync
    sync_wall = {"seconds": 0.0, "calls": 0}

    def measured_find(self, *args, **kwargs):
        with torch.profiler.record_function("profile::range_estimation"):
            return original_find(self, *args, **kwargs)

    def measured_quantize(self, *args, **kwargs):
        with torch.profiler.record_function("profile::quant_dequant"):
            return original_quantize(self, *args, **kwargs)

    def measured_inverse(self, *args, **kwargs):
        with torch.profiler.record_function("profile::block_cholesky_inverse"):
            return original_inverse(self, *args, **kwargs)

    def measured_sync(device):
        start = time.perf_counter()
        sync_wall["calls"] += 1
        try:
            return original_sync(device)
        finally:
            sync_wall["seconds"] += time.perf_counter() - start

    Quantizer.find_params = measured_find
    Quantizer.quantize = measured_quantize
    GPTQ.block_cholesky_inverse = measured_inverse
    gptq_module.torch_sync = measured_sync
    try:
        sync()
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            record_shapes=True,
            with_stack=True,
            profile_memory=True,
        ) as prof:
            start = time.perf_counter()
            with torch.profiler.record_function("profile::candidate_reset_and_native_core"):
                result = native_dynamic_gptq(w, h, nsamples=nsamples, alpha=0.8)
            sync()
            wall = time.perf_counter() - start
    finally:
        Quantizer.find_params = original_find
        Quantizer.quantize = original_quantize
        GPTQ.block_cholesky_inverse = original_inverse
        gptq_module.torch_sync = original_sync

    def event_device_seconds(event):
        return float(getattr(event, "device_time_total", getattr(event, "cuda_time_total", 0.0))) / 1e6

    def event_cpu_seconds(event):
        return float(getattr(event, "cpu_time_total", 0.0)) / 1e6

    categories = []
    ranges = {
        "candidate_reset_and_native_core": "profile::candidate_reset_and_native_core",
        "range_estimation": "profile::range_estimation",
        "quant_dequant": "profile::quant_dequant",
        "block_cholesky_inverse": "profile::block_cholesky_inverse",
    }
    for label, key in ranges.items():
        match = next((e for e in prof.key_averages() if e.key == key), None)
        if match is not None:
            categories.append({
                "stage": label,
                "wall_seconds": wall if label == "candidate_reset_and_native_core" else "",
                "cpu_seconds": event_cpu_seconds(match),
                "cuda_seconds": event_device_seconds(match),
                "count": int(match.count),
                "scope": "one instrumented full-shape [128,896] native candidate; profiler attribution, not timing benchmark",
            })

    # Attribute native source lines from the profiler's Python stack. CUDA
    # operator time is reported by source region and is not added to wall time.
    source_cuda = {"damping": 0.0, "cholesky": 0.0, "group_setup": 0.0, "compensation": 0.0, "sync_and_loss": 0.0}
    source_cpu = {key: 0.0 for key in source_cuda}
    source_count = {key: 0 for key in source_cuda}
    for event in prof.events():
        stack = getattr(event, "stack", None) or []
        gptq_frame = next((str(frame) for frame in stack if "gptq.py" in str(frame) and "quantizer.py" not in str(frame)), None)
        if gptq_frame is None:
            continue
        import re
        match = re.search(r"gptq\.py\((\d+)\)", gptq_frame)
        if not match:
            continue
        line = int(match.group(1))
        if 346 <= line <= 348:
            group = "damping"
        elif line in (352, 362):
            group = "cholesky"
        elif 376 <= line <= 406:
            group = "group_setup"
        elif 407 <= line <= 417:
            group = "compensation"
        elif line in (426, 428):
            group = "sync_and_loss"
        else:
            continue
        source_cuda[group] += float(getattr(event, "device_time_total", getattr(event, "cuda_time_total", 0.0))) / 1e6
        source_cpu[group] += float(getattr(event, "self_cpu_time_total", 0.0)) / 1e6
        source_count[group] += int(getattr(event, "count", 1))
    for key in source_cuda:
        categories.append({
            "stage": f"native_source::{key}",
            "wall_seconds": "",
            "cpu_seconds": source_cpu[key],
            "cuda_seconds": source_cuda[key],
            "count": source_count[key],
            "scope": "profiler operator events attributed to installed gptq.py source lines; CUDA/CPU metrics are not additive",
        })
    categories.append({
        "stage": "native_torch_sync_wall",
        "wall_seconds": sync_wall["seconds"],
        "cpu_seconds": sync_wall["seconds"],
        "cuda_seconds": "",
        "count": sync_wall["calls"],
        "scope": "measured torch_sync call(s) inside native GPTQ; does not include outer benchmark boundary sync",
    })
    prof.export_chrome_trace(str(OUT / "REFERENCE_PROFILE_TRACE.json"))
    return result, categories, prof


def component_copy_probe(w, h, nsamples):
    """Profile reference wrapper's module, weight, and Hessian reset work."""
    from gptqmodel.quantization import QuantizeConfig
    from gptqmodel.quantization.gptq import GPTQ

    qcfg = QuantizeConfig(
        bits=4, group_size=128, sym=True, desc_act=False, static_groups=False,
        true_sequential=True, damp_percent=0.1, damp_auto_increment=0, mse=0,
    )
    sync()
    event_start = torch.cuda.Event(enable_timing=True)
    event_end = torch.cuda.Event(enable_timing=True)
    start = time.perf_counter()
    event_start.record()
    module = torch.nn.Linear(w.shape[1], w.shape[0], bias=False, device="meta", dtype=w.dtype)
    module.weight = torch.nn.Parameter(w.detach().clone(), requires_grad=False)
    core = GPTQ(module, qcfg=qcfg)
    core.quantizer.configure(perchannel=True, bits=4, sym=True)
    core.H = h.detach().clone()
    core.nsamples = nsamples
    event_end.record()
    sync()
    elapsed = time.perf_counter() - start
    return elapsed, float(event_start.elapsed_time(event_end)) / 1000.0


def selected_alphas(raw):
    landscape = raw["candidates"]
    selected = {}
    for row in range(32):
        selected[str(row)] = min(
            ((landscape[str(alpha)]["search_mse"][row], -alpha, alpha) for alpha in GRID)
        )[2]
    return selected


def main():
    torch.backends.cuda.matmul.allow_tf32 = False
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device unavailable; v17.3 requires the frozen GPU environment")
    device = torch.cuda.get_device_name(0)
    if device != "NVIDIA GeForce RTX 4060 Laptop GPU":
        raise RuntimeError(f"Frozen benchmark device mismatch: {device}")

    state = torch.load(V172 / "smoke_state.pt", map_location="cuda", weights_only=False)
    raw = load(V172 / "M2_SMOKE_RAW.json")
    ref_cache = torch.load(V172 / "candidate_cache.pt", map_location="cpu", weights_only=True)
    timing_old = load(V172 / "TIMING_SUMMARY.json")
    cost_old = load(V172 / "M2_COST_ESTIMATE.json")
    assert state["model_rev"] == MODEL_REV
    assert state["cal_manifest_sha256"] == sha_file(V16 / "CALIBRATION_MANIFEST.json")
    assert raw["expected_candidate_rows"] == 320 and raw["candidate_rows_completed"] == 320
    assert list(ref_cache.keys()) == [str(a) for a in GRID]

    w = state["W_k"]
    h = state["H_k"]
    nsamples = int(state["nsamples_k"])
    x = torch.cat([sample[0] for sample in state["inputs"]["self_attn.k_proj"]]).float()
    prepared = prepare_native_state(h, nsamples=nsamples)
    assert sha(prepared.hessian) == prepared.h_sha256, "Prepared immutable H hash check failed before candidate loop"

    # Correctness gate: same damped Cholesky, same cached native block inverse,
    # and native alpha=1 baseline. No timing data is taken from this call.
    optimized_alpha1 = optimized_dynamic_gptq(w, prepared, alpha=1.0, check_cholesky=True)
    native_alpha1 = native_dynamic_gptq(w, h, nsamples=nsamples, alpha=1.0)
    alpha1_parity = {
        "weight": torch.equal(optimized_alpha1.weight_fp16, native_alpha1.weight_fp16),
        "scale": torch.equal(optimized_alpha1.scales, native_alpha1.scales),
        "zero": torch.equal(optimized_alpha1.zeros, native_alpha1.zeros),
        "g_idx": torch.equal(optimized_alpha1.g_idx, native_alpha1.g_idx),
        "prepared_cholesky_exact": True,
        "prepared_hessian_sha256": prepared.h_sha256,
        "pass": False,
    }
    alpha1_parity["pass"] = all(alpha1_parity[k] for k in ("weight", "scale", "zero", "g_idx", "prepared_cholesky_exact"))
    if not alpha1_parity["pass"]:
        dump("correctness_checks.json", {"status": "FAIL", "alpha1_parity": alpha1_parity})
        raise RuntimeError("Optimized alpha=1 parity failed; stop before formal candidate experiment")

    # Full 32-row x 10-alpha candidate grid, full native [128,896] geometry.
    optimized_results = {}
    optimized_payload = {}
    output_records = []
    for alpha in GRID:
        result = optimized_dynamic_gptq(w, prepared, alpha=alpha)
        q = result.weight_fp16.detach().clone()
        scale = result.scales.detach().clone()
        zero = result.zeros.detach().clone()
        g_idx = result.g_idx.detach().clone()
        delta = q[:32].float() - w[:32].float()
        scores = (x @ delta.T).square().mean(0)
        weight_mse = delta.square().mean(1)
        optimized_results[alpha] = {
            "q": q.cpu(), "scale": scale.cpu(), "zero": zero.cpu(), "g_idx": g_idx.cpu(),
            "search_mse": [float(z) for z in scores.cpu()],
            "weight_mse": [float(z) for z in weight_mse.cpu()],
            "q_hashes": [sha(q[row]) for row in range(32)],
            "scale_hashes": [sha(scale[row]) for row in range(32)],
            "zero_hashes": [sha(zero[row]) for row in range(32)],
            "g_idx_hash": sha(g_idx),
        }
        output_records.append({
            "alpha": alpha,
            "candidate_rows": 32,
            "full_target_shape": list(q.shape),
            "search_mse_mean": float(scores.mean()),
            "search_mse_min": float(scores.min()),
            "search_mse_max": float(scores.max()),
            "weight_mse_mean": float(weight_mse.mean()),
            "weight_hash_sha256": sha(q),
            "scale_hash_sha256": sha(scale),
            "zero_hash_sha256": sha(zero),
            "g_idx_sha256": sha(g_idx),
        })
        optimized_payload[str(alpha)] = {"q": q.cpu(), "scale": scale.cpu(), "zero": zero.cpu(), "g_idx": g_idx.cpu()}
        del result, q, scale, zero, g_idx

    # Check candidate-by-candidate arrays and full native tensor outputs against
    # the frozen v17.2 artifact, which came from the certified reference path.
    candidate_rows = []
    for alpha in GRID:
        key = str(alpha)
        ref = ref_cache[key]
        got = optimized_results[alpha]
        for name in ("q", "scale", "zero", "g_idx"):
            exact = torch.equal(got[name], ref[name])
            candidate_rows.append({"alpha": alpha, "tensor": name, "exact": exact, "max_abs_diff": maxdiff(got[name], ref[name])})
        ref_candidate = raw["candidates"][key]
        mse_diff = [abs(a - b) for a, b in zip(got["search_mse"], ref_candidate["search_mse"])]
        wmse_diff = [abs(a - b) for a, b in zip(got["weight_mse"], ref_candidate["weight_mse"])]
        candidate_rows.append({
            "alpha": alpha,
            "tensor": "search_mse",
            "exact": all(diff == 0 for diff in mse_diff),
            "max_abs_diff": max(mse_diff, default=0.0),
        })
        candidate_rows.append({
            "alpha": alpha,
            "tensor": "weight_mse",
            "exact": all(diff == 0 for diff in wmse_diff),
            "max_abs_diff": max(wmse_diff, default=0.0),
        })

    # Reverse candidate order from an immutable prepared state and compare each
    # result to the already completed optimized forward-order grid.
    reverse_exact = True
    reverse_items = []
    for alpha in reversed(GRID):
        result = optimized_dynamic_gptq(w, prepared, alpha=alpha)
        tensors = {"q": result.weight_fp16.cpu(), "scale": result.scales.cpu(), "zero": result.zeros.cpu(), "g_idx": result.g_idx.cpu()}
        per_tensor = {name: torch.equal(tensors[name], optimized_results[alpha][name]) for name in tensors}
        reverse_exact = reverse_exact and all(per_tensor.values())
        reverse_items.append({"alpha": alpha, "exact": per_tensor})
        del result, tensors
    optimized_results_summary = {
        "status": "COMPLETE",
        "candidate_rows": 320,
        "full_native_calls": 10,
        "candidate_order": GRID,
        "reverse_order_exact": reverse_exact,
        "rows": output_records,
        "selected_alpha_counts": {
            str(alpha): sum(v == alpha for v in selected_alphas(raw).values()) for alpha in GRID
        },
        "selected_alpha_by_row": selected_alphas(raw),
        "selected_loss_mean": statistics.mean(min(optimized_results[a]["search_mse"][r] for a in GRID) for r in range(32)),
        "alpha1_search_mse_mean": statistics.mean(optimized_results[1.0]["search_mse"]),
        "candidate_cache_sha256": "pending",
        "debug_mode": "complete tensor outputs cached after candidate calls for exact equivalence and resume; formal mode writes score/hash per alpha and selected state only",
    }

    # Search loss direct calculation check: X @ (Q-W).T equals the equivalent
    # prediction subtraction form used by v17.2.
    direct_check = {}
    for alpha in GRID:
        q = optimized_results[alpha]["q"][:32].to("cuda").float()
        mse_delta = (x @ (q - w[:32].float()).T).square().mean(0)
        ref_scores = torch.tensor(raw["candidates"][str(alpha)]["search_mse"], device="cuda")
        direct_check[str(alpha)] = {
            "exact": torch.equal(mse_delta, ref_scores),
            "max_abs_diff": maxdiff(mse_delta, ref_scores),
        }

    # Commit selected row tensors onto a pristine FP16 model-weight state.
    selected = selected_alphas(raw)
    committed = w.detach().clone()
    selected_hashes = {}
    for row in range(32):
        alpha = selected[str(row)]
        committed[row] = optimized_results[alpha]["q"][row].to("cuda")
        selected_hashes[str(row)] = {
            "alpha": alpha,
            "weight_hash": sha(committed[row]),
            "reference_hash_matches": sha(committed[row]) == raw["candidates"][str(alpha)]["q_hashes"][row],
        }
    commit_parity = {
        "selected_weight_hashes_match_reference": all(v["reference_hash_matches"] for v in selected_hashes.values()),
        "nonselected_rows_pristine": torch.equal(committed[32:], w[32:]),
        "selected_row_count": 32,
        "selected_rows": selected_hashes,
        "pass": all(v["reference_hash_matches"] for v in selected_hashes.values()) and torch.equal(committed[32:], w[32:]),
    }

    # Restore a full native decoder layer and commit selected row 0 at its
    # chosen alpha, matching v17.2's self-prefix regression check exactly.
    # Then separately replay the complete q/k/v transition (searched k rows
    # 0:32 plus native alpha=1 v/q) before comparing the next-target Hessian.
    layer, rope = make_layer(state["layer_state"])
    baseline_layer, baseline_rope = make_layer(state["layer_state"])
    selected_row = 0
    selected_alpha = selected[str(selected_row)]
    selected_q = optimized_results[selected_alpha]["q"][selected_row].to("cuda")
    with torch.no_grad():
        layer.get_submodule("self_attn.k_proj").weight[selected_row].copy_(selected_q)
    baseline_o = forward_capture(baseline_layer, baseline_rope, state["docs"], ["self_attn.o_proj"])["self_attn.o_proj"]
    sync()
    start = time.perf_counter()
    refreshed = forward_capture(layer, rope, state["docs"], ["self_attn.o_proj"])["self_attn.o_proj"]
    sync()
    refresh_seconds = time.perf_counter() - start
    x_before = torch.cat([t[0] for t in baseline_o]).float()
    x_after = torch.cat([t[0] for t in refreshed]).float()
    ref_prefix = load(V172 / "SELF_PREFIX_PROPAGATION.json")
    prefix_same = sha(x_after) == ref_prefix["input_after"]["hash"] and sha(x_before) == ref_prefix["input_before"]["hash"]

    # The v17.2 next-target H was produced after committing the whole searched
    # k row subset and native alpha=1 v_proj/q_proj siblings. Reproduce those
    # exact dependency semantics; a row-0-only prefix is insufficient for H
    # parity even though it is sufficient for the self-prefix drift check.
    group_layer, group_rope = make_layer(state["layer_state"])
    with torch.no_grad():
        group_layer.get_submodule("self_attn.k_proj").weight[:32].copy_(committed[:32])
    sibling_commit = {}
    for name in ("self_attn.v_proj", "self_attn.q_proj"):
        mod = group_layer.get_submodule(name)
        batches = state["inputs"][name]
        sibling_h, sibling_nsamples, _ = build_hessian(mod, batches)
        sibling_result = native_dynamic_gptq(
            mod.weight.detach().clone(), sibling_h,
            nsamples=sibling_nsamples, alpha=1.0,
        )
        with torch.no_grad():
            mod.weight.copy_(sibling_result.weight_fp16)
        sibling_commit[name] = {
            "input_hash": sha(torch.cat([batch[0] for batch in batches]).float()),
            "hessian_hash": sha(sibling_h),
            "nsamples": sibling_nsamples,
            "weight_hash": sha(mod.weight),
        }
        del sibling_result, sibling_h
    refreshed_group = forward_capture(group_layer, group_rope, state["docs"], ["self_attn.o_proj"])["self_attn.o_proj"]
    hessian, h_nsamples, hessian_seconds = build_hessian(
        group_layer.get_submodule("self_attn.o_proj"), refreshed_group
    )
    h_current_sha = sha(hessian)
    self_prefix_parity = {
        "selected_row": selected_row,
        "selected_alpha": selected_alpha,
        "input_before_sha256": sha(x_before),
        "input_after_sha256": sha(x_after),
        "reference_input_before_sha256": ref_prefix["input_before"]["hash"],
        "reference_input_after_sha256": ref_prefix["input_after"]["hash"],
        "forward_activations_exact": prefix_same,
        "reference_relative_l2_drift": ref_prefix["relative_L2_drift"],
        "optimized_relative_l2_drift": float(torch.linalg.vector_norm(x_after - x_before) / torch.linalg.vector_norm(x_before)),
        "reference_hessian_sha256": ref_prefix["next_target_after_refresh"]["current_H_sha256"],
        "optimized_hessian_sha256": h_current_sha,
        "hessian_exact": h_current_sha == ref_prefix["next_target_after_refresh"]["current_H_sha256"],
        "hessian_dependency_replay": "searched k rows 0:32 plus native alpha=1 v_proj/q_proj, then one qkv-to-o_proj refresh",
        "sibling_commits": sibling_commit,
        "nsamples": h_nsamples,
        "forward_refresh_seconds": refresh_seconds,
        "next_hessian_build_seconds": hessian_seconds,
        "pass": prefix_same and h_current_sha == ref_prefix["next_target_after_refresh"]["current_H_sha256"],
    }

    # Explicit local activation and Hessian profile. Captures are diagnostic;
    # these measurements do not replace the full-model frozen cost source.
    capture_layer, capture_rope = make_layer(state["layer_state"])
    sync()
    start = time.perf_counter()
    captured_k = forward_capture(capture_layer, capture_rope, state["docs"], ["self_attn.k_proj"])["self_attn.k_proj"]
    sync()
    local_capture_seconds = time.perf_counter() - start
    capture_exact = all(torch.equal(a, b) for a, b in zip(captured_k, state["inputs"]["self_attn.k_proj"]))
    local_h, local_ns, local_h_seconds = build_hessian(capture_layer.get_submodule("self_attn.k_proj"), captured_k)
    hessian_exact = torch.equal(local_h, h) and local_ns == nsamples

    # Obtain the detailed v17.2 reference profile with a separate instrumented
    # call; then isolate reset/clone and formal output serialization.
    profiled_result, profile_rows, prof = profile_reference(w, h, nsamples)
    reference_profile_tensors = {
        "weight": torch.equal(profiled_result.weight_fp16.cpu(), ref_cache["0.8"]["q"]),
        "scale": torch.equal(profiled_result.scales.cpu(), ref_cache["0.8"]["scale"]),
        "zero": torch.equal(profiled_result.zeros.cpu(), ref_cache["0.8"]["zero"]),
        "g_idx": torch.equal(profiled_result.g_idx.cpu(), ref_cache["0.8"]["g_idx"]),
    }
    copy_wall, copy_cuda = component_copy_probe(w, h, nsamples)
    profiler_operators = []
    for event in prof.key_averages(group_by_stack_n=1):
        if "gptq.py" not in " ".join(getattr(event, "stack", []) or []) and not event.key.startswith("profile::"):
            continue
        profiler_operators.append({
            "operator_or_stack": event.key,
            "calls": int(event.count),
            "cpu_self_seconds": float(getattr(event, "self_cpu_time_total", 0.0)) / 1e6,
            "cuda_self_seconds": float(getattr(event, "self_cuda_time_total", getattr(event, "self_device_time_total", 0.0))) / 1e6,
        })
    with (OUT / "REFERENCE_PROFILE_OPERATORS.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["operator_or_stack", "calls", "cpu_self_seconds", "cuda_self_seconds"])
        writer.writeheader()
        writer.writerows(sorted(profiler_operators, key=lambda r: r["cuda_self_seconds"], reverse=True))

    old_timing = timing_old
    old_prefix_seconds = float(old_timing["prefix_forward_refresh_seconds"])
    old_commit_seconds = float(old_timing["selection_and_commit_seconds"])
    # One write mirrors the smoke/debug cache; formal efficient mode is
    # separately specified and avoids writing each candidate tensor.
    serialized = io.BytesIO()
    start = time.perf_counter()
    torch.save(optimized_payload, serialized)
    serialization_seconds = time.perf_counter() - start
    cache_bytes = len(serialized.getvalue())
    torch.save(optimized_payload, OUT / "optimized_candidate_cache.pt")
    optimized_results_summary["candidate_cache_sha256"] = sha_file(OUT / "optimized_candidate_cache.pt")
    dump("OPTIMIZED_RESULTS.json", optimized_results_summary)
    dump("OPTIMIZED_COMMIT_PARITY.json", commit_parity)
    dump("OPTIMIZED_SELF_PREFIX_PARITY.json", self_prefix_parity)

    # Correctness and order checks precede performance claims.
    candidate_equivalence_pass = all(row["exact"] for row in candidate_rows) and reverse_exact and all(v["exact"] for v in direct_check.values())
    correctness = {
        "status": "PASS" if candidate_equivalence_pass and alpha1_parity["pass"] and commit_parity["pass"] and self_prefix_parity["pass"] and capture_exact and hessian_exact and all(reference_profile_tensors.values()) else "FAIL",
        "alpha1_native_parity": alpha1_parity,
        "candidate_equivalence": candidate_equivalence_pass,
        "candidate_order_reverse_exact": reverse_exact,
        "direct_output_mse_equivalence": direct_check,
        "selected_commit": commit_parity["pass"],
        "self_prefix_and_hessian": self_prefix_parity["pass"],
        "fresh_activation_capture_matches_v17_2": capture_exact,
        "fresh_hessian_matches_v17_2": hessian_exact,
        "instrumented_reference_matches_cache": reference_profile_tensors,
        "pristine_weight_unchanged": sha(w) == raw["pristine_weight_sha256"],
        "pristine_hessian_unchanged": sha(h) == raw["H_sha256"],
    }
    dump("CANDIDATE_EQUIVALENCE.json", {
        "status": "PASS" if candidate_equivalence_pass else "FAIL",
        "reference_source": "v17.2 candidate_cache.pt and M2_SMOKE_RAW.json; certified GPTQ native path",
        "optimized_source": "optimized_native_m2.py; unchanged installed GPTQ.quantize() with cached immutable block inverse",
        "candidate_count": 320,
        "full_native_shape": [128, 896],
        "per_candidate_tensor_comparisons": candidate_rows,
        "forward_candidate_order": GRID,
        "reverse_candidate_order": list(reversed(GRID)),
        "reverse_exact": reverse_exact,
        "reverse_per_alpha": reverse_items,
        "direct_output_mse_equivalence": direct_check,
    })

    # Five full-grid repeated timings after one warmup per implementation.
    def run_ref_once(alpha):
        sync()
        start = time.perf_counter()
        out = native_dynamic_gptq(w, h, nsamples=nsamples, alpha=alpha)
        sync()
        elapsed = time.perf_counter() - start
        del out
        return elapsed

    def run_opt_once(alpha, prep):
        sync()
        start = time.perf_counter()
        out = optimized_dynamic_gptq(w, prep, alpha=alpha)
        sync()
        elapsed = time.perf_counter() - start
        del out
        return elapsed

    _ = run_ref_once(0.5)
    ref_runs = []
    for rep in range(1, 6):
        row = []
        for alpha in GRID:
            row.append({"replicate": rep, "alpha": alpha, "seconds": run_ref_once(alpha)})
        ref_runs.extend(row)

    _ = prepare_native_state(h, nsamples=nsamples)
    _ = run_opt_once(0.5, prepared)
    opt_runs = []
    prep_runs = []
    for rep in range(1, 6):
        sync()
        prep_start = time.perf_counter()
        timed_prepared = prepare_native_state(h, nsamples=nsamples)
        sync()
        prep_runs.append({"replicate": rep, "seconds": time.perf_counter() - prep_start})
        for alpha in GRID:
            opt_runs.append({"replicate": rep, "alpha": alpha, "seconds": run_opt_once(alpha, timed_prepared)})
        del timed_prepared

    with (OUT / "REFERENCE_TIMING.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["replicate", "alpha", "seconds", "warmup_excluded"])
        writer.writeheader()
        for row in ref_runs:
            writer.writerow({**row, "warmup_excluded": True})
    with (OUT / "OPTIMIZED_TIMING.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["replicate", "alpha", "seconds", "prepare_seconds", "warmup_excluded"])
        writer.writeheader()
        for row in opt_runs:
            writer.writerow({**row, "prepare_seconds": "", "warmup_excluded": True})
        for row in prep_runs:
            writer.writerow({"replicate": row["replicate"], "alpha": "PREPARE_NATIVE_STATE", "seconds": row["seconds"], "prepare_seconds": row["seconds"], "warmup_excluded": True})

    ref_grid = [sum(r["seconds"] for r in ref_runs if r["replicate"] == rep) for rep in range(1, 6)]
    opt_grid = [
        sum(r["seconds"] for r in opt_runs if r["replicate"] == rep) + prep_runs[rep - 1]["seconds"]
        for rep in range(1, 6)
    ]
    ref_candidate = [r["seconds"] for r in ref_runs]
    opt_candidate = [r["seconds"] for r in opt_runs]
    timing = {
        "warmup_runs_excluded_per_implementation": 1,
        "measured_repeats": 5,
        "candidate_count_per_full_grid": 10,
        "rows_scored_per_alpha": 32,
        "full_native_shape_per_alpha": [128, 896],
        "reference_candidate_seconds": stats(ref_candidate),
        "optimized_candidate_seconds_excluding_prepare": stats(opt_candidate),
        "optimized_prepare_seconds_per_full_grid": stats([r["seconds"] for r in prep_runs]),
        "reference_per_10alpha_row_seconds": stats([v / 32 for v in ref_grid]),
        "optimized_per_10alpha_row_seconds_including_prepare": stats([v / 32 for v in opt_grid]),
        "reference_32row_grid_seconds": stats(ref_grid),
        "optimized_32row_grid_seconds_including_prepare": stats(opt_grid),
        "speedup_reference_over_optimized_grid_median": statistics.median(ref_grid) / statistics.median(opt_grid),
        "reference_grid_replicates_seconds": ref_grid,
        "optimized_grid_replicates_seconds_including_prepare": opt_grid,
        "timing_conditions": "RTX 4060 Laptop GPU, same saved W/H/X, synchronized wall boundaries around each candidate and each prepared-state build",
    }
    dump("TIMING_SUMMARY.json", timing)

    # Assemble the required source-attributed wall/profile breakdown.
    profile_rows.extend([
        {"stage": "reference_candidate_copy_reset", "wall_seconds": copy_wall, "cpu_seconds": "", "cuda_seconds": copy_cuda, "count": 1,
         "scope": "isolated v17.2 module weight clone + GPTQ working clone + H clone probe; setup not included in candidate total separately"},
        {"stage": "activation_capture_L0_k_proj", "wall_seconds": local_capture_seconds, "cpu_seconds": "", "cuda_seconds": "", "count": 1,
         "scope": "two frozen 64-token calibration docs; verified against v17.2 captured input"},
        {"stage": "Hessian_construction_L0_k_proj", "wall_seconds": local_h_seconds, "cpu_seconds": "", "cuda_seconds": "", "count": 1,
         "scope": "GPTQ.process_batch on frozen inputs; exact H match; local smoke only"},
        {"stage": "selection_and_row_commit_v17_2", "wall_seconds": old_commit_seconds, "cpu_seconds": "", "cuda_seconds": "", "count": 1,
         "scope": "reuse verified v17.2 selection/commit measured stage"},
        {"stage": "forward_refresh_v17_2", "wall_seconds": old_prefix_seconds, "cpu_seconds": "", "cuda_seconds": "", "count": 1,
         "scope": "reuse verified v17.2 L0 selected-row prefix refresh measurement"},
        {"stage": "serialization_debug_cache", "wall_seconds": serialization_seconds, "cpu_seconds": serialization_seconds, "cuda_seconds": "", "count": 1,
         "scope": f"single complete 10-alpha debug tensor cache serialization; {cache_bytes} bytes; not formal efficient mode"},
        {"stage": "optimized_prepare_native_state", "wall_seconds": statistics.median(r["seconds"] for r in prep_runs), "cpu_seconds": "", "cuda_seconds": "",
         "count": 5, "scope": "median over five optimized full-grid preparation builds; included in optimized grid timing"},
    ])
    # Add the v17.2 uninstrumented end-to-end distribution as the main baseline.
    profile_rows.append({
        "stage": "reference_candidate_total_v17_2_median", "wall_seconds": timing_old["candidate_per_call_ms"]["median_ms"] / 1000,
        "cpu_seconds": "", "cuda_seconds": "", "count": 10,
        "scope": "v17.2 synchronized full-shape baseline; detailed operator timing comes from instrumented diagnostic call",
    })
    with (OUT / "REFERENCE_PROFILE.csv").open("w", newline="", encoding="utf-8") as f:
        fields = ["stage", "wall_seconds", "cpu_seconds", "cuda_seconds", "count", "scope"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(sorted(profile_rows, key=lambda r: float(r["wall_seconds"]) if r["wall_seconds"] not in ("", None) else -1.0, reverse=True))

    # Start from the corrected unit/refresh audit. Give cost credit only to the
    # measured 128x896 shape; retain the frozen conservative multiplier and all
    # other unmeasured shape estimates unchanged.
    corrected_cost = load(OUT / "CORRECTED_FULLMODEL_COST.json")
    updated_cost = json.loads(json.dumps(corrected_cost))
    shape_count = cost_old["shape_counts"]["128x896"]
    core_delta = shape_count * (statistics.median(opt_grid) - statistics.median(ref_grid))
    conservative_delta = 2.0 * core_delta
    updated_cost["conservative_components_seconds"]["candidate_full_linear_gptq"] += conservative_delta
    updated_cost["conservative_total_seconds"] += conservative_delta
    updated_cost["conservative_total_hours"] = updated_cost["conservative_total_seconds"] / 3600
    updated_cost["status"] = "M2_COST_GO" if updated_cost["conservative_total_hours"] <= 48 else "M2_COST_NO_GO"
    updated_cost["v17_3_timing_update"] = {
        "optimized_shape_tested": "128x896",
        "optimized_grid_median_seconds_including_one_prepare": statistics.median(opt_grid),
        "reference_grid_median_seconds": statistics.median(ref_grid),
        "updated_targets": shape_count,
        "base_core_cost_delta_seconds": core_delta,
        "conservative_core_delta_seconds": conservative_delta,
        "other_shapes_receive_speedup_credit": False,
        "capture_and_refresh_model": "corrected v17.3 activation-boundary audit, not v17.2 48.79h estimate",
        "gate_hours": 48,
    }
    dump("UPDATED_FULLMODEL_COST.json", updated_cost)

    profile_pass = all(reference_profile_tensors.values()) and capture_exact and hessian_exact
    correctness["status"] = "PASS" if all([
        candidate_equivalence_pass, alpha1_parity["pass"], commit_parity["pass"],
        self_prefix_parity["pass"], profile_pass,
    ]) else "FAIL"
    dump("correctness_checks.json", correctness)
    if correctness["status"] != "PASS":
        raise RuntimeError("v17.3 correctness gate failed; runtime cost verdict is not eligible")

    # The optimized output/checkpoint artifact is emitted only after all
    # checks; the debug candidate cache remains for this small smoke.
    patch_manifest = [
        {"path": "optimized_native_m2.py", "change": "new isolated wrapper; reuse one immutable GPTQ block inverse; remove redundant wrapper FP16 weight clone", "historical_files_modified": False},
        {"path": "run_v17_3.py", "change": "new bounded smoke/profile/benchmark runner", "historical_files_modified": False},
    ]
    with (OUT / "PATCH_MANIFEST.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(patch_manifest[0]))
        writer.writeheader()
        writer.writerows(patch_manifest)

    dump("REFERENCE_RESULTS.json", {
        "source": "v17.2 M2_SMOKE_RAW.json and candidate_cache.pt",
        "candidate_rows": 320,
        "full_native_calls": 10,
        "selected_alpha_by_row": selected,
        "selected_alpha_counts": raw["selected_count_by_alpha"],
        "alpha1_search_mse_mean": timing_old.get("alpha1_search_mse_mean", None),
        "v17_2_candidate_timing_ms": timing_old["candidate_per_call_ms"],
        "raw_artifact_sha256": sha_file(V172 / "M2_SMOKE_RAW.json"),
        "cache_artifact_sha256": sha_file(V172 / "candidate_cache.pt"),
    })
    dump("RUN_METADATA.json", {
        "device": device,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "model_revision": MODEL_REV,
        "v16_calibration_manifest_sha256": sha_file(V16 / "CALIBRATION_MANIFEST.json"),
        "v17_2_raw_sha256": sha_file(V172 / "M2_SMOKE_RAW.json"),
        "v17_2_candidate_cache_sha256": sha_file(V172 / "candidate_cache.pt"),
        "v17_2_native_source_sha256": sha_file(V171 / "native_m2.py"),
        "candidate_grid": GRID,
        "full_native_shape": list(w.shape),
        "candidate_rows_scored": 32,
        "formal_full_model_run": False,
        "ppl_run": False,
    })
    print(json.dumps({
        "correctness": correctness["status"],
        "candidate_equivalence": candidate_equivalence_pass,
        "reverse_order": reverse_exact,
        "selected_commit": commit_parity["pass"],
        "self_prefix": self_prefix_parity["pass"],
        "ref_grid_median_s": statistics.median(ref_grid),
        "opt_grid_median_s": statistics.median(opt_grid),
        "speedup": timing["speedup_reference_over_optimized_grid_median"],
        "updated_conservative_hours": updated_cost["conservative_total_hours"],
        "cost_status": updated_cost["status"],
    }, indent=2))


if __name__ == "__main__":
    main()
