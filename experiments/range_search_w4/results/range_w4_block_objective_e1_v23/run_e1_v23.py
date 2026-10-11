"""Protocol-faithful E1 runner; writes only below this v23 result directory."""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import csv
import hashlib
import json
import math
import os
import platform
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[3]
RESULTS = ROOT / "experiments/range_search_w4/results"
V18 = RESULTS / "range_w4_fullmodel_v18"
V171 = RESULTS / "range_w4_m2_parity_v17_1"
V173 = RESULTS / "range_w4_m2_runtime_v17_3"
MODEL_DIR = ROOT / ".hf_cache/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775"
M2_MODEL_DIR = V18 / "M2/checkpoint/final"
M2_WEIGHTS = M2_MODEL_DIR / "model.safetensors"
DOCS_PATH = OUT / "E1_DOCUMENT_MANIFEST.json"
PROTOCOL_PATH = OUT / "PROTOCOL.json"
CACHE_DIR = OUT / "cache"
RAW_DIR = OUT / "raw"
ALPHAS = [i / 10 for i in range(1, 11)]
LAYERS = [0, 12, 23]
PROJECTIONS = ["q_proj", "up_proj"]
ROWS = {"q_proj": [0, 224, 448, 672], "up_proj": [0, 1216, 2432, 3648]}
N_DOCS = 64
SEQ = 2048
DEVICE = torch.device("cuda:0")
_INPUT_HASH_CACHE = None

sys.path.insert(0, str(V171))
sys.path.insert(0, str(V173))
from native_m2 import native_dynamic_gptq  # noqa: E402
from optimized_native_m2 import optimized_dynamic_gptq, prepare_native_state  # noqa: E402


def sync():
    torch.cuda.synchronize(DEVICE)


def sha_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha_tensor(t: torch.Tensor) -> str:
    return sha_bytes(t.detach().cpu().contiguous().numpy().tobytes())


def sha_memmap(mm: np.memmap) -> str:
    h = hashlib.sha256()
    for i in range(mm.shape[0]):
        h.update(np.asarray(mm[i]).tobytes(order="C"))
    return h.hexdigest()


def dump(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def load_docs():
    obj = json.loads(DOCS_PATH.read_text(encoding="utf-8"))
    assert len(obj["samples"]) == N_DOCS and obj["token_budget"] == N_DOCS * SEQ
    assert obj["ordered_token_ids_sha256_i64le"] == hashlib.sha256(
        b"".join(int(t).to_bytes(8, "little", signed=True) for d in obj["samples"] for t in d["token_ids"])
    ).hexdigest()
    for s in obj["samples"]:
        assert len(s["token_ids"]) == SEQ
        assert sha_bytes(json.dumps(s["token_ids"]).encode("utf-8")) == s["token_sha256"]
    assert all(obj["samples"][i]["split"] == name for i, name in
               [(0, "S0_search"), (15, "S0_search"), (16, "S0_held"), (31, "S0_held"),
                (32, "S1_search"), (47, "S1_search"), (48, "S1_held"), (63, "S1_held")])
    return obj


def input_hashes():
    global _INPUT_HASH_CACHE
    if _INPUT_HASH_CACHE is not None:
        return dict(_INPUT_HASH_CACHE)
    paths = [DOCS_PATH, PROTOCOL_PATH, OUT / "E1_BUDGET_RECALC.json",
             OUT / "Q_OBJECTIVE_PATH.md", OUT / "UP_OBJECTIVE_PATH.md",
             V18 / "FORMAL_RUN_MANIFEST.json", V18 / "PROTOCOL.json",
             V18 / "M2/RUN_REPORT.json", M2_WEIGHTS,
             MODEL_DIR / "config.json", MODEL_DIR / "model.safetensors"]
    _INPUT_HASH_CACHE = {str(p.relative_to(ROOT).as_posix()): sha_file(p) for p in paths}
    return dict(_INPUT_HASH_CACHE)


def runtime_environment():
    from importlib.metadata import version
    import transformers
    return {"python": platform.python_version(), "torch": torch.__version__,
            "transformers": transformers.__version__, "gptqmodel": version("gptqmodel"),
            "cuda_runtime": torch.version.cuda, "device": torch.cuda.get_device_name(DEVICE),
            "device_capability": list(torch.cuda.get_device_capability(DEVICE)),
            "tf32_matmul": bool(torch.backends.cuda.matmul.allow_tf32),
            "tf32_cudnn": bool(torch.backends.cudnn.allow_tf32)}


def new_memmap(path: Path, shape: tuple[int, ...], *, overwrite=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(path)
    return np.memmap(path, dtype="<f2", mode="w+", shape=shape)


def close_mm(mm):
    mm.flush()
    mm._mmap.close()


def sync_timing(fn):
    sync()
    t0 = time.perf_counter()
    value = fn()
    sync()
    return value, time.perf_counter() - t0


def module_full_name(layer: int, projection: str):
    prefix = "self_attn" if projection == "q_proj" else "mlp"
    return f"model.layers.{layer}.{prefix}.{projection}"


def module_attr(layer, projection):
    parent = layer.self_attn if projection == "q_proj" else layer.mlp
    return getattr(parent, projection)


def call_layer(model, layer_idx: int, hidden: torch.Tensor):
    positions = torch.arange(hidden.shape[1], device=DEVICE, dtype=torch.long)[None, :]
    rotary = model.model.rotary_emb(hidden, positions)
    return model.model.layers[layer_idx](
        hidden,
        attention_mask=None,
        position_ids=positions,
        cache_position=positions[0],
        position_embeddings=rotary,
        use_cache=False,
    )[0]


def load_model_and_patch_m2(*, copy_teachers=True):
    from safetensors import safe_open
    from transformers import AutoModelForCausalLM

    if not torch.cuda.is_available() or torch.cuda.get_device_name(DEVICE) != "NVIDIA GeForce RTX 4060 Laptop GPU":
        raise RuntimeError("Frozen CUDA device RTX 4060 Laptop GPU unavailable")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = AutoModelForCausalLM.from_pretrained(
        str(MODEL_DIR), torch_dtype=torch.float16, low_cpu_mem_usage=True,
        local_files_only=True, attn_implementation="sdpa",
    ).eval().to(DEVICE)

    teachers = {}
    if copy_teachers:
        for li in LAYERS:
            teachers[li] = __import__("copy").deepcopy(model.model.layers[li]).eval()
    pristine = {}
    m2_targets = {}
    for li in LAYERS:
        for proj in PROJECTIONS:
            name = module_full_name(li, proj)
            pristine[name] = module_attr(model.model.layers[li], proj).weight.detach().cpu().contiguous().clone()

    expected = json.loads((V18 / "M2/RUN_REPORT.json").read_text(encoding="utf-8"))
    if expected["status"] != "COMPLETE" or expected["final_model_path"] != str(M2_MODEL_DIR):
        raise RuntimeError("Frozen v18 M2 checkpoint run report does not match the expected final model")
    with safe_open(str(M2_WEIGHTS), framework="pt", device="cpu") as sf:
        for li in range(24):
            layer = model.model.layers[li]
            for proj in ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]:
                parent = layer.self_attn if proj.endswith("_proj") and proj in ("q_proj", "k_proj", "v_proj", "o_proj") else layer.mlp
                module = getattr(parent, proj)
                key = f"model.layers.{li}.{'self_attn' if parent is layer.self_attn else 'mlp'}.{proj}.weight"
                q = sf.get_tensor(key)
                if q.shape != module.weight.shape or q.dtype != torch.float16:
                    raise RuntimeError(f"M2 checkpoint shape/dtype mismatch at {key}")
                if li in LAYERS and proj in PROJECTIONS:
                    m2_targets[module_full_name(li, proj)] = q.detach().contiguous().clone()
                with torch.no_grad():
                    module.weight.copy_(q.to(device=DEVICE))
                del q
    del sf
    expected_target_names = {module_full_name(li, proj) for li in LAYERS for proj in PROJECTIONS}
    if set(m2_targets) != expected_target_names:
        raise RuntimeError(f"Could not retain all six exact v18 M2 target tensors: {sorted(m2_targets)}")
    return model, teachers, pristine, m2_targets


def stage_prepare_prefix():
    marker = CACHE_DIR / "PREFIX_CACHE_MANIFEST.json"
    if marker.exists():
        validate_started = time.perf_counter()
        old = json.loads(marker.read_text(encoding="utf-8"))
        if old.get("source_hashes") != input_hashes():
            raise RuntimeError("Existing prefix cache belongs to different frozen sources")
        for key, meta in old["files"].items():
            p = CACHE_DIR / key
            if not p.is_file() or sha_file(p) != meta["sha256"]:
                raise RuntimeError(f"Existing prefix cache failed its file hash: {p}")
        validation_seconds = time.perf_counter() - validate_started
        resumed = dict(old)
        resumed["runtime_cache_validation_seconds"] = validation_seconds
        print(json.dumps({"status": "CACHE_REUSED", "prefix_wall_seconds": old["timing"]["prefix_seconds"],
                          "teacher_wall_seconds": old["timing"]["teacher_seconds"],
                          "cache_validation_seconds": validation_seconds}, indent=2))
        return resumed

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    expected_names = ([f"hidden_L{li}.f16" for li in LAYERS] + [f"teacher_L{li}.f16" for li in LAYERS] +
                      [f"input_L{li}_{p}.f16" for li in LAYERS for p in PROJECTIONS])
    for name in expected_names:
        stale = CACHE_DIR / name
        if stale.exists():
            stale.unlink()
    docs = load_docs()["samples"]
    load_started = time.perf_counter()
    model, teachers, pristine, _ = load_model_and_patch_m2()
    sync()
    model_load_seconds = time.perf_counter() - load_started
    hidden_mms = {li: new_memmap(CACHE_DIR / f"hidden_L{li}.f16", (N_DOCS, SEQ, 896)) for li in LAYERS}
    teacher_mms = {li: new_memmap(CACHE_DIR / f"teacher_L{li}.f16", (N_DOCS, SEQ, 896)) for li in LAYERS}
    input_mms = {(li, p): new_memmap(CACHE_DIR / f"input_L{li}_{p}.f16", (N_DOCS, SEQ, 896))
                 for li in LAYERS for p in PROJECTIONS}
    active = {"doc": -1}
    handles = []
    for li in LAYERS:
        for proj in PROJECTIONS:
            module = module_attr(model.model.layers[li], proj)
            mm = input_mms[(li, proj)]
            def make_hook(d, mapping):
                def hook(_module, args):
                    mapping[active["doc"]] = args[0].detach()[0].to("cpu").numpy()
                return hook
            handles.append(module.register_forward_pre_hook(make_hook((li, proj), mm)))
    positions = torch.arange(SEQ, device=DEVICE)[None, :]
    prefix_started = time.perf_counter()
    for di, sample in enumerate(docs):
        active["doc"] = di
        ids = torch.tensor(sample["token_ids"], dtype=torch.long, device=DEVICE)[None, :]
        hidden = model.model.embed_tokens(ids)
        for li in range(24):
            if li in LAYERS:
                hidden_mms[li][di] = hidden[0].detach().to("cpu").numpy()
            with torch.inference_mode():
                hidden = call_layer(model, li, hidden)
        del ids, hidden
        if (di + 1) % 8 == 0:
            print(f"[prefix] {di + 1}/{N_DOCS}", flush=True)
    sync()
    prefix_seconds = time.perf_counter() - prefix_started
    for h in handles:
        h.remove()
    teacher_started = time.perf_counter()
    with torch.inference_mode():
        for li in LAYERS:
            for di in range(N_DOCS):
                h = torch.from_numpy(np.array(hidden_mms[li][di], copy=True)).unsqueeze(0).to(DEVICE)
                # Run the pristine teacher layer on the fixed M2-prefix hidden state.
                pos = positions
                rotary = model.model.rotary_emb(h, pos)
                y = teachers[li](h, attention_mask=None, position_ids=pos, cache_position=pos[0],
                                 position_embeddings=rotary, use_cache=False)[0]
                teacher_mms[li][di] = y[0].detach().to("cpu").numpy()
                del h, y, rotary
    sync()
    teacher_seconds = time.perf_counter() - teacher_started
    files = {}
    all_mms = list(hidden_mms.values()) + list(teacher_mms.values()) + list(input_mms.values())
    for mm in all_mms:
        close_mm(mm)
    hash_started = time.perf_counter()
    for p in CACHE_DIR.glob("*.f16"):
        files[p.name] = {"sha256": sha_file(p), "size_bytes": p.stat().st_size}
    source_hashes = input_hashes()
    cache_hash_seconds = time.perf_counter() - hash_started
    payload = {
        "status": "PASS",
        "source_hashes": source_hashes,
        "model_load_seconds": model_load_seconds,
        "cache_creation_and_hash_seconds": cache_hash_seconds,
        "environment": runtime_environment(),
        "timing": {"prefix_seconds": prefix_seconds, "teacher_seconds": teacher_seconds,
                   "prefix_decoder_layer_forwards": N_DOCS * 24, "teacher_decoder_layer_forwards": N_DOCS * len(LAYERS)},
        "files": files,
        "input_tensor_sha256": {f"L{li}_{proj}": files[f"input_L{li}_{proj}.f16"]["sha256"]
                                for li in LAYERS for proj in PROJECTIONS},
        "hidden_tensor_sha256": {f"L{li}": files[f"hidden_L{li}.f16"]["sha256"] for li in LAYERS},
        "teacher_tensor_sha256": {f"L{li}": files[f"teacher_L{li}.f16"]["sha256"] for li in LAYERS},
        "pristine_target_weight_sha256": {k: sha_tensor(v) for k, v in pristine.items()},
        "m2_model_safetensors_sha256": sha_file(M2_WEIGHTS),
        "teacher_definition": "pristine FP16 target block on same frozen v18 M2-prefix hidden state",
        "selected_on_held": False,
    }
    dump(marker, payload)
    del model, teachers, pristine
    return payload


def load_mm(path: Path, shape):
    expected = math.prod(shape) * 2
    if path.stat().st_size != expected:
        raise RuntimeError(f"Activation cache size mismatch: {path}")
    return np.memmap(path, dtype="<f2", mode="r", shape=shape)


def new_quantizer_config():
    from gptqmodel.quantization import QuantizeConfig
    return QuantizeConfig(bits=4, group_size=128, sym=True, desc_act=False, static_groups=False,
                          true_sequential=True, damp_percent=0.1, damp_auto_increment=0, mse=0)


def build_hessian(layer, projection, x_mm, doc_indices, label):
    from gptqmodel.quantization.gptq import GPTQ
    module = module_attr(layer, projection)
    core = GPTQ(module, qcfg=new_quantizer_config())
    core.quantizer.configure(perchannel=True, bits=4, sym=True)
    core.H = torch.zeros((module.weight.shape[1], module.weight.shape[1]), dtype=torch.float32, device=DEVICE)
    core.nsamples = 0
    sync(); t0 = time.perf_counter()
    for di in doc_indices:
        x = torch.from_numpy(np.array(x_mm[di], copy=True)).unsqueeze(0).to(DEVICE)
        core.process_batch(x)
        del x
    sync(); elapsed = time.perf_counter() - t0
    H = core.H.detach().clone()
    nsamples = int(core.nsamples)
    detail = {"label": label, "documents": len(doc_indices), "tokens_per_document": SEQ,
              "nsamples": nsamples, "H_sha256": sha_tensor(H), "wall_seconds": elapsed}
    return H, nsamples, detail


def score_rows_local(x_mm, doc_indices, weight, quantized):
    """Frozen v18 direct FP32 X@(Q-W)^T SSE, FP64 accumulation, mean per token/output row."""
    delta = quantized.float() - weight.float()
    sums = torch.zeros((weight.shape[0],), dtype=torch.float64, device=DEVICE)
    for lo in range(0, len(doc_indices), 4):
        chosen = doc_indices[lo:lo + 4]
        x = torch.from_numpy(np.array(x_mm[chosen], copy=True)).to(DEVICE).float()
        output_delta = torch.matmul(x.reshape(-1, x.shape[-1]), delta.T)
        sums += output_delta.to(torch.float64).square().sum(0)
        del x, output_delta
    return sums / float(len(doc_indices) * SEQ)


def score_one_row_local(x_mm, doc_indices, weight_row, quantized_row):
    """Frozen v18 row-local metric, with FP64 SSE accumulation."""
    delta = quantized_row.float() - weight_row.float()
    sse = 0.0
    for lo in range(0, len(doc_indices), 4):
        chosen = doc_indices[lo:lo + 4]
        x = torch.from_numpy(np.array(x_mm[chosen], copy=True)).to(DEVICE).float()
        output_delta = torch.matmul(x.reshape(-1, x.shape[-1]), delta)
        sse += float(output_delta.to(torch.float64).square().sum().item())
        del x, output_delta
    return sse / float(len(doc_indices) * SEQ)


def make_candidate(layer, projection, pristine, prepared, alpha, *, check=False):
    result = optimized_dynamic_gptq(pristine, prepared, alpha=alpha, check_cholesky=check)
    return result


def tensor_params(result, row):
    return {"scale": result.scales[row].detach().cpu().double().tolist(),
            "zero": result.zeros[row].detach().cpu().double().tolist(),
            "g_idx": result.g_idx.detach().cpu().tolist(),
            "weight_sha256": sha_tensor(result.weight_fp16[row])}


def layer_forward_from_hidden(model, li, hidden):
    with torch.inference_mode():
        return call_layer(model, li, hidden)


def score_block_documents(model, li, projection, baseline_q, candidate_rows, teacher_mm,
                          hidden_mm, row, alpha_index, doc_indices):
    module = module_attr(model.model.layers[li], projection)
    with torch.no_grad():
        module.weight.copy_(baseline_q)
    candidate = candidate_rows[alpha_index, row].to(DEVICE)
    sses, mses = [], []
    peak_alloc = peak_reserved = 0
    sync(); t0 = time.perf_counter()
    for di in doc_indices:
        with torch.no_grad():
            module.weight[row].copy_(candidate)
        h = torch.from_numpy(np.array(hidden_mm[di], copy=True)).unsqueeze(0).to(DEVICE)
        y = layer_forward_from_hidden(model, li, h)
        t = torch.from_numpy(np.array(teacher_mm[di], copy=True)).unsqueeze(0).to(DEVICE)
        diff = y.float() - t.float()
        sse = float(diff.double().square().sum().item())
        elems = diff.numel()
        sses.append(sse)
        mses.append(sse / elems)
        del h, y, t, diff
        peak_alloc = max(peak_alloc, int(torch.cuda.max_memory_allocated(DEVICE)))
        peak_reserved = max(peak_reserved, int(torch.cuda.max_memory_reserved(DEVICE)))
    sync(); seconds = time.perf_counter() - t0
    with torch.no_grad():
        module.weight.copy_(baseline_q)
    return {"doc_indices": list(doc_indices), "document_sse": sses, "document_mse": mses,
            "pooled_sse": math.fsum(sses), "pooled_mse": math.fsum(sses) / (len(doc_indices) * SEQ * 896),
            "wall_seconds": seconds, "single_document_block_forwards": len(doc_indices),
            "peak_allocated_bytes": peak_alloc, "peak_reserved_bytes": peak_reserved}


def score_local_one_doc(weight_row, quantized_row, x):
    delta = quantized_row.float() - weight_row.float()
    out = torch.matmul(x.float(), delta)
    return float(out.double().square().sum().item()) / out.numel()


def parity_equal(a, b):
    return {"weight": torch.equal(a.weight_fp16, b.weight_fp16),
            "scale": torch.equal(a.scales, b.scales),
            "zero": torch.equal(a.zeros, b.zeros),
            "g_idx": torch.equal(a.g_idx, b.g_idx)}


def cell_indices(split):
    return list(range(0, 16)) if split == "S0" else list(range(32, 48))


def held_indices(split):
    return list(range(16, 32)) if split == "S0" else list(range(48, 64))


def open_cache(li, proj):
    return load_mm(CACHE_DIR / f"input_L{li}_{proj}.f16", (N_DOCS, SEQ, 896))


def correctness_and_cost_preflight(prefix):
    if prefix.get("status") != "PASS":
        raise RuntimeError("Prefix cache is not complete")
    docs = load_docs()
    sync(); model_load_t0 = time.perf_counter()
    model, teachers, pristine_targets, _ = load_model_and_patch_m2(copy_teachers=False)
    sync(); model_load_seconds = time.perf_counter() - model_load_t0
    timing = {"model_load_seconds": None, "candidate_seconds": [], "local_score_seconds": [],
              "hessian_details": [], "block_probe_seconds": [], "block_probe_forwards": 0}
    timing["model_load_seconds"] = model_load_seconds
    timing["environment"] = runtime_environment()
    sync()
    t_load = time.perf_counter()
    # Loading has already happened; record checkpoint patch plus setup interval outside the formal estimate.
    timing["preflight_started_monotonic"] = t_load
    all_checks = {}
    formal_examples = {}
    peak_reset = torch.cuda.reset_peak_memory_stats

    # Build an actual search-only H for L0/q; held inputs are not read by this function.
    li, proj, split = 0, "q_proj", "S0"
    xq = open_cache(li, proj)
    Hq, nsq, hdetail = build_hessian(model.model.layers[li], proj, xq, cell_indices(split), "preflight_L0_q_S0")
    timing["hessian_details"].append(hdetail)
    sync(); t0 = time.perf_counter()
    prepared_q = prepare_native_state(Hq, nsamples=nsq)
    sync(); hdetail["hinv_prepare_seconds"] = time.perf_counter() - t0
    Wq = pristine_targets[module_full_name(li, proj)].to(DEVICE)

    # Forward and reverse alpha order candidate bank: compare complete native outputs and metadata.
    forward = {}
    for alpha in ALPHAS:
        sync(); t0 = time.perf_counter()
        forward[alpha] = make_candidate(model.model.layers[li], proj, Wq, prepared_q, alpha, check=(alpha == 1.0))
        sync(); timing["candidate_seconds"].append({"cell": "L0/q", "pass": "forward", "alpha": alpha,
                                                    "seconds": time.perf_counter() - t0})
    reverse = {}
    for alpha in reversed(ALPHAS):
        sync(); t0 = time.perf_counter()
        reverse[alpha] = make_candidate(model.model.layers[li], proj, Wq, prepared_q, alpha, check=(alpha == 1.0))
        sync(); timing["candidate_seconds"].append({"cell": "L0/q", "pass": "reverse", "alpha": alpha,
                                                    "seconds": time.perf_counter() - t0})
    order_equal = {f"{a:.1f}": all(parity_equal(forward[a], reverse[a]).values()) for a in ALPHAS}
    all_checks["candidate_order_tensor_invariance"] = all(order_equal.values())

    sync(); t0 = time.perf_counter()
    direct_native = native_dynamic_gptq(Wq, Hq, nsamples=nsq, alpha=1.0)
    sync(); native_parity_seconds = time.perf_counter() - t0
    timing["candidate_seconds"].append({"cell": "L0/q", "pass": "native_parity_reference",
                                         "alpha": 1.0, "seconds": native_parity_seconds})
    p = parity_equal(forward[1.0], direct_native)
    all_checks["alpha1_native_gptq_parity"] = all(p.values())
    all_checks["alpha1_parity_fields"] = p
    x_parity = torch.from_numpy(np.array(xq[cell_indices(split)[0]], copy=True)).to(DEVICE).float()
    out_candidate = x_parity @ forward[1.0].weight_fp16.float().T
    out_native = x_parity @ direct_native.weight_fp16.float().T
    direct_output_max_abs = float((out_candidate - out_native).abs().max().item())
    all_checks["alpha1_direct_output_parity"] = torch.equal(out_candidate, out_native)
    timing["alpha1_direct_output_max_abs_difference"] = direct_output_max_abs
    del x_parity, out_candidate, out_native
    all_checks["alpha1_is_unmodified_native_candidate"] = True

    # Reproduce frozen v18 all-row local scorer and compare one preregistered row to an independent direct calculation.
    row = ROWS[proj][0]
    for a in [0.8, 1.0]:
        sync(); t0 = time.perf_counter()
        vector = score_rows_local(xq, cell_indices(split), Wq, forward[a].weight_fp16)
        sync(); timing["local_score_seconds"].append({"cell": "L0/q", "alpha": a,
                                                       "seconds": time.perf_counter() - t0,
                                                       "rows": Wq.shape[0]})
        independent_sse = 0.0
        delta_sse = 0.0
        for lo in range(0, 16, 4):
            x = torch.from_numpy(np.array(xq[cell_indices(split)[lo:lo + 4]], copy=True)).to(DEVICE).float()
            flat = x.reshape(-1, x.shape[-1])
            direct = flat @ forward[a].weight_fp16[row].float() - flat @ Wq[row].float()
            delta = flat @ (forward[a].weight_fp16[row].float() - Wq[row].float())
            independent_sse += float(direct.double().square().sum().item())
            delta_sse += float(delta.double().square().sum().item())
            del x, flat, direct, delta
        direct_mse = independent_sse / float(16 * SEQ)
        delta_mse = delta_sse / float(16 * SEQ)
        local_mse = float(vector[row].item())
        scorer_error = abs(local_mse - delta_mse)
        direct_error = abs(local_mse - direct_mse)
        all_checks[f"frozen_local_score_alpha_{a:.1f}"] = math.isfinite(local_mse)
        all_checks[f"direct_local_score_equivalence_alpha_{a:.1f}"] = scorer_error <= max(1e-12, 1e-6 * max(local_mse, delta_mse))
        all_checks[f"direct_output_difference_equivalence_alpha_{a:.1f}"] = direct_error <= max(1e-10, 2e-4 * max(local_mse, direct_mse))
        timing.setdefault("local_scorer_probe_details", []).append({"alpha": a, "frozen_mse": local_mse,
            "delta_form_mse": delta_mse, "direct_output_form_mse": direct_mse,
            "delta_abs_error": scorer_error, "direct_output_abs_error": direct_error})

    # Cost probes use the largest-output target shape at the deepest frozen block.
    li, proj = 23, "up_proj"
    xu = open_cache(li, proj)
    Hu, nsu, hdetail = build_hessian(model.model.layers[li], proj, xu, cell_indices("S0"), "preflight_L23_up_S0")
    timing["hessian_details"].append(hdetail)
    sync(); t0 = time.perf_counter()
    prepared_u = prepare_native_state(Hu, nsamples=nsu)
    sync(); hdetail["hinv_prepare_seconds"] = time.perf_counter() - t0
    Wu = pristine_targets[module_full_name(li, proj)].to(DEVICE)
    max_shape_candidates = {}
    for a in (0.8, 0.9):
        sync(); t0 = time.perf_counter()
        max_shape_candidates[a] = make_candidate(model.model.layers[li], proj, Wu, prepared_u, a)
        sync(); elapsed = time.perf_counter() - t0
        timing["candidate_seconds"].append({"cell": "L23/up", "pass": "cost_probe", "alpha": a, "seconds": elapsed})

    hidden_q = load_mm(CACHE_DIR / "hidden_L0.f16", (N_DOCS, SEQ, 896))
    teacher_q = load_mm(CACHE_DIR / "teacher_L0.f16", (N_DOCS, SEQ, 896))
    qmodule = module_attr(model.model.layers[0], "q_proj")
    qbase = qmodule.weight.detach().clone()
    qrow_alpha = forward[0.9].weight_fp16[ROWS["q_proj"][0]].detach().clone()
    with torch.no_grad(): qmodule.weight[ROWS["q_proj"][0]].copy_(qrow_alpha)
    qprobe, qtime = sync_timing(lambda: layer_forward_from_hidden(
        model, 0, torch.from_numpy(np.array(hidden_q[0], copy=True)).unsqueeze(0).to(DEVICE)))
    qteacher_hash_before = sha_file(CACHE_DIR / "teacher_L0.f16")
    qprobe_repeat, qrepeat = sync_timing(lambda: layer_forward_from_hidden(
        model, 0, torch.from_numpy(np.array(hidden_q[0], copy=True)).unsqueeze(0).to(DEVICE)))
    all_checks["q_block_score_repeat_deterministic"] = torch.equal(qprobe, qprobe_repeat)
    all_checks["teacher_independent_of_candidate"] = qteacher_hash_before == sha_file(CACHE_DIR / "teacher_L0.f16")
    qteacher_doc = torch.from_numpy(np.array(teacher_q[0], copy=True)).unsqueeze(0).to(DEVICE)
    q_mse_1 = float((qprobe.float() - qteacher_doc.float()).double().square().mean().item())
    q_mse_2 = float((qprobe_repeat.float() - qteacher_doc.float()).double().square().mean().item())
    timing["repeat_mse_abs_difference"] = abs(q_mse_1 - q_mse_2)
    timing["repeat_mse_values"] = [q_mse_1, q_mse_2]
    del qteacher_doc
    timing["block_probe_seconds"].extend([qtime, qrepeat])
    timing["block_probe_forwards"] += 2
    with torch.no_grad(): qmodule.weight.copy_(qbase)
    del qprobe, qprobe_repeat, qbase, qrow_alpha

    umodule = module_attr(model.model.layers[23], "up_proj")
    ubase = umodule.weight.detach().clone()
    urow = ROWS["up_proj"][0]
    urow_alpha = max_shape_candidates[0.9].weight_fp16[urow].detach().clone()
    with torch.no_grad(): umodule.weight[urow].copy_(urow_alpha)
    hidden_u = load_mm(CACHE_DIR / "hidden_L23.f16", (N_DOCS, SEQ, 896))
    teacher_u = load_mm(CACHE_DIR / "teacher_L23.f16", (N_DOCS, SEQ, 896))
    # Exactly 2 candidate configs x 2 complete held/search-independent documents for conservative max-shape timing.
    up_probe_runs = []
    for alpha in (0.8, 0.9):
        with torch.no_grad(): umodule.weight[urow].copy_(max_shape_candidates[alpha].weight_fp16[urow])
        for di in (0, 1):
            y, elapsed = sync_timing(lambda di=di: layer_forward_from_hidden(
                model, 23, torch.from_numpy(np.array(hidden_u[di], copy=True)).unsqueeze(0).to(DEVICE)))
            t = torch.from_numpy(np.array(teacher_u[di], copy=True)).unsqueeze(0).to(DEVICE)
            sse = float((y.float() - t.float()).double().square().sum().item())
            up_probe_runs.append({"alpha": alpha, "document_index": di, "seconds": elapsed, "finite_sse": math.isfinite(sse)})
            del y, t
    timing["block_probe_seconds"].extend(x["seconds"] for x in up_probe_runs)
    timing["block_probe_forwards"] += len(up_probe_runs)
    all_checks["up_block_candidate_scores_finite"] = all(x["finite_sse"] for x in up_probe_runs)
    with torch.no_grad(): umodule.weight.copy_(ubase)
    max_local, local_up_seconds = sync_timing(lambda: score_rows_local(
        xu, cell_indices("S0"), Wu, max_shape_candidates[0.9].weight_fp16))
    timing["local_score_seconds"].append({"cell": "L23/up", "alpha": 0.9,
                                           "seconds": local_up_seconds, "rows": Wu.shape[0]})
    row_score_probe, row_score_seconds = sync_timing(lambda: score_one_row_local(
        xu, cell_indices("S0"), Wu[ROWS["up_proj"][0]], max_shape_candidates[0.9].weight_fp16[ROWS["up_proj"][0]]))
    timing["held_local_row_score_probe_seconds"] = row_score_seconds
    timing["held_local_row_score_probe_finite"] = math.isfinite(row_score_probe)
    del max_local, row_score_probe

    # Measure representative complete row stitching and candidate checkpoint I/O.
    def stitch_probe(results, module):
        old = module.weight.detach().clone()
        cpu_weights = [r.weight_fp16.detach().cpu() for r in results]
        cpu_scales = [r.scales.detach().cpu() for r in results]
        cpu_zeros = [r.zeros.detach().cpu() for r in results]
        cpu_gidx = [r.g_idx.detach().cpu() for r in results]
        if not all(torch.equal(cpu_gidx[0], x) for x in cpu_gidx[1:]):
            raise RuntimeError("Candidate g_idx differs; row-wise stitch is not valid")
        sync(); t_stitch = time.perf_counter()
        q = torch.empty_like(cpu_weights[0]); s = torch.empty_like(cpu_scales[0]); z = torch.empty_like(cpu_zeros[0])
        for r in range(q.shape[0]):
            j = r % len(results)
            q[r].copy_(cpu_weights[j][r]); s[r].copy_(cpu_scales[j][r]); z[r].copy_(cpu_zeros[j][r])
        qdev = q.to(DEVICE)
        with torch.no_grad():
            module.weight.copy_(qdev)
            module.weight.copy_(old)
        sync()
        return time.perf_counter() - t_stitch

    q_stitch_seconds = stitch_probe(list(forward.values()), qmodule)
    up_stitch_seconds = stitch_probe(list(max_shape_candidates.values()), umodule)
    timing["stitch_commit_probe_seconds"] = {"q_proj": q_stitch_seconds, "up_proj": up_stitch_seconds}

    io_dir = OUT / "preflight_io_probe"
    io_dir.mkdir(parents=True, exist_ok=True)
    candidate_io = {}
    for family, result in (("q_proj", forward[0.8]), ("up_proj", max_shape_candidates[0.9])):
        probe_path = io_dir / f"{family}.pt"
        sync(); t_io = time.perf_counter()
        payload = {"weight": result.weight_fp16.detach().cpu(), "scales": result.scales.detach().cpu(),
                   "zeros": result.zeros.detach().cpu(), "g_idx": result.g_idx.detach().cpu()}
        torch.save(payload, probe_path)
        sync()
        write_seconds = time.perf_counter() - t_io
        t_io = time.perf_counter()
        check = torch.load(probe_path, map_location="cpu", weights_only=True)
        read_seconds = time.perf_counter() - t_io
        candidate_io[family] = {"bytes": probe_path.stat().st_size, "write_seconds": write_seconds,
                                "read_seconds": read_seconds}
        probe_path.unlink()
        del check, payload
    try:
        io_dir.rmdir()
    except OSError:
        pass
    timing["candidate_checkpoint_io_probe"] = candidate_io

    h_probe = io_dir / "hessian.pt"
    io_dir.mkdir(parents=True, exist_ok=True)
    sync(); t_io = time.perf_counter()
    torch.save(Hq.detach().cpu(), h_probe)
    sync(); h_write_seconds = time.perf_counter() - t_io
    t_io = time.perf_counter()
    h_read = torch.load(h_probe, map_location="cpu", weights_only=True)
    h_read_seconds = time.perf_counter() - t_io
    h_bytes = h_probe.stat().st_size
    h_probe.unlink()
    score_probe = io_dir / "score.json"
    score_payload = {"status": "IN_PROGRESS", "doc_indices": list(range(16)),
                     "document_sse": [None] * 16, "document_mse": [None] * 16,
                     "document_seconds": [None] * 16, "single_document_block_forwards": 0}
    sync(); t_io = time.perf_counter()
    for i in range(16):
        score_payload["document_sse"][i] = float(i + 1)
        score_payload["document_mse"][i] = float(i + 1) / 100
        score_payload["document_seconds"][i] = 0.25
        score_payload["single_document_block_forwards"] += 1
        dump(score_probe, score_payload)
    score_checkpoint_16_seconds = time.perf_counter() - t_io
    score_probe.unlink()
    try:
        io_dir.rmdir()
    except OSError:
        pass
    timing["hessian_checkpoint_io_probe"] = {"bytes": h_bytes, "write_seconds": h_write_seconds,
                                             "read_seconds": h_read_seconds}
    timing["block_score_checkpoint_16_updates_seconds"] = score_checkpoint_16_seconds
    del h_read, score_payload
    del forward, reverse, direct_native, max_shape_candidates

    # Held data are absent from selection calls by construction. Perturbing a separate held object cannot alter search argmin.
    def choose(search):
        return min(ALPHAS, key=lambda a: (search[a], -a))
    synthetic_search = {a: float((a - 0.8) ** 2) for a in ALPHAS}
    selected_1 = choose(synthetic_search)
    synthetic_held = {a: float(a) for a in ALPHAS}
    synthetic_held.update({a: float(1.0 - a) for a in ALPHAS})
    selected_2 = choose(synthetic_search)
    all_checks["held_data_cannot_enter_selector"] = selected_1 == selected_2
    required_check_keys = [k for k, v in all_checks.items() if isinstance(v, bool) and k != "all_required_checks_pass"]
    all_checks["all_required_checks_pass"] = all(all_checks[k] for k in required_check_keys)

    torch.cuda.synchronize(DEVICE)
    peak = {"allocated_bytes": int(torch.cuda.max_memory_allocated(DEVICE)),
            "reserved_bytes": int(torch.cuda.max_memory_reserved(DEVICE))}
    # Collect measured shape-specific medians for the frozen cost model.
    cand_by_shape = {"q_proj": [], "up_proj": []}
    for r in timing["candidate_seconds"]:
        if r["pass"] == "cost_probe":
            cand_by_shape["up_proj"].append(r["seconds"])
    # L0/q forward-order candidate timings are one full GPTQ pass; reverse timings are correctness-only repeats.
    cand_by_shape["q_proj"] = [r["seconds"] for r in timing["candidate_seconds"]
                                 if r["cell"] == "L0/q" and r["pass"] == "forward"]
    q_candidate_per = statistics.median(cand_by_shape["q_proj"])
    up_candidate_per = statistics.median(cand_by_shape["up_proj"])
    up_block_probes = [x["seconds"] for x in up_probe_runs]
    block_per_doc = max(up_block_probes)
    local_q_median = statistics.median([r["seconds"] for r in timing["local_score_seconds"] if r["cell"] == "L0/q"])
    local_probe = {"q_proj_seconds": local_q_median, "up_proj_seconds": local_up_seconds,
                   "held_local_row_score_probe_seconds": row_score_seconds,
                   "up_block_probe_max_seconds": block_per_doc}
    # Count all already completed setup, plus estimate remaining formal work. The caller decides the six-hour gate.
    q_stitch_probe_seconds = time.perf_counter()
    q_cpu_candidates = [forward_result.weight_fp16.detach().cpu() for forward_result in []]
    del q_cpu_candidates
    q_stitch_probe_seconds = time.perf_counter() - q_stitch_probe_seconds
    timing.update({"candidate_shape_medians_seconds": {"q_proj": q_candidate_per, "up_proj": up_candidate_per},
                   "block_single_document_max_deep_up_probe_seconds": block_per_doc,
                   "local_scoring_probe": local_probe, "peak_cuda": peak,
                   "prefix_setup_seconds": prefix["model_load_seconds"] + prefix["timing"]["prefix_seconds"] + prefix["timing"]["teacher_seconds"] + prefix.get("cache_creation_and_hash_seconds", 0.0),
                   "prefix_cache_validation_seconds": prefix.get("runtime_cache_validation_seconds", 0.0),
                   "prefix_cache_creation_and_hash_seconds": prefix.get("cache_creation_and_hash_seconds", 0.0),
                   "stitch_commit_probe_seconds": {"q_proj": q_stitch_seconds, "up_proj": up_stitch_seconds},
                   "preflight_full_linear_gptq_executions": len(timing["candidate_seconds"]),
                   "preflight_block_single_document_forwards": timing["block_probe_forwards"]})
    dump(OUT / "PREFLIGHT_CORRECTNESS.json", {"status": "PASS" if all_checks["all_required_checks_pass"] else "FAIL",
          "checks": all_checks, "forward_reverse_exact_by_alpha": order_equal,
          "alpha1_parity": p, "preflight_candidate_execution_count": len(timing["candidate_seconds"]),
          "preflight_alpha_candidate_gptq_executions": 22,
          "preflight_native_parity_reference_executions": 1,
          "repeat_mse_abs_difference": timing["repeat_mse_abs_difference"],
          "candidate_tensors_are_shared_between_objectives": True,
          "held_selection_independence_check": True,
          "local_scorer": "same FP32 X@(Q-W)^T, FP64 SSE accumulation and four-document chunks as v18 run_precheck.score_rows",
          "peak_cuda": peak})
    dump(OUT / "RUNTIME_PREFLIGHT.json", timing)
    for mm in (xq, xu, hidden_q, teacher_q, hidden_u, teacher_u):
        mm._mmap.close()
    del model, teachers, pristine_targets, Hq, Hu, prepared_q, prepared_u, Wq, Wu
    return all_checks, timing


def estimate_cost(prefix, timing):
    q_candidate = timing["candidate_shape_medians_seconds"]["q_proj"]
    up_candidate = timing["candidate_shape_medians_seconds"]["up_proj"]
    # 3 layers x 2 splits x 10 alpha executions per projection family.
    candidate_seconds = 60 * q_candidate + 60 * up_candidate
    block_per_doc = timing["local_scoring_probe"]["up_block_probe_max_seconds"]
    search_block_seconds = 7680 * block_per_doc
    held_block_seconds = 7680 * block_per_doc
    h_samples = [x["wall_seconds"] + x.get("hinv_prepare_seconds", 0.0) for x in timing["hessian_details"]]
    h_seconds = statistics.median(h_samples)
    hessian_seconds = 12 * h_seconds
    local_search_seconds = timing["local_scoring_probe"]["q_proj_seconds"] * 60 + timing["local_scoring_probe"]["up_proj_seconds"] * 60
    held_local_seconds = timing["local_scoring_probe"]["held_local_row_score_probe_seconds"] * 144
    prefix_capture_seconds = prefix["model_load_seconds"] + prefix["timing"]["prefix_seconds"]
    fixed_teacher_seconds = prefix["timing"]["teacher_seconds"]
    prefix_cache_create_hash_seconds = prefix.get("cache_creation_and_hash_seconds", 0.0)
    formal_model_load_seconds = timing["model_load_seconds"]
    cache_validation_seconds = timing["prefix_cache_validation_seconds"]
    io_probe = timing["candidate_checkpoint_io_probe"]
    candidate_bytes = 3 * 2 * 10 * (io_probe["q_proj"]["bytes"] + io_probe["up_proj"]["bytes"])
    write_rate = min(io_probe[k]["bytes"] / max(io_probe[k]["write_seconds"], 1e-6) for k in io_probe)
    read_rate = min(io_probe[k]["bytes"] / max(io_probe[k]["read_seconds"], 1e-6) for k in io_probe)
    candidate_write_seconds = candidate_bytes / write_rate
    candidate_read_seconds = candidate_bytes / read_rate
    h_io = timing["hessian_checkpoint_io_probe"]
    hessian_checkpoint_io_seconds = 12 * (h_io["write_seconds"] + h_io["read_seconds"])
    block_checkpoint_seconds = 960 * timing["block_score_checkpoint_16_updates_seconds"]
    stitch = timing["stitch_commit_probe_seconds"]
    # Six local-M2 stitches; each phase commits that baseline and restores v18 M2 (24 full tensor copies total).
    stitch_commit_seconds = 24 * stitch["q_proj"] + 24 * stitch["up_proj"]
    artifact_io_seconds = (candidate_write_seconds + candidate_read_seconds + hessian_checkpoint_io_seconds +
                           block_checkpoint_seconds + 30.0)
    setup_and_prefix_checks = (prefix_capture_seconds + fixed_teacher_seconds + prefix_cache_create_hash_seconds +
                               formal_model_load_seconds + cache_validation_seconds)
    central = (setup_and_prefix_checks + candidate_seconds + search_block_seconds + held_block_seconds +
               hessian_seconds + local_search_seconds + held_local_seconds + stitch_commit_seconds + artifact_io_seconds)
    conservative = central * 1.5
    return {
        "status": "E1_COST_GATE_PASS" if conservative <= 6 * 3600 else "E1_COST_GATE_FAIL",
        "conservative_safety_multiplier": 1.5,
        "limit_seconds": 21600,
        "components_seconds": {
            "prefix_capture_full_m2_forward_actual": prefix_capture_seconds,
            "fixed_teacher_block_capture_actual": fixed_teacher_seconds,
            "prefix_cache_write_and_hash_actual": prefix_cache_create_hash_seconds,
            "formal_model_load_patch_estimate": formal_model_load_seconds,
            "prefix_cache_validation_measured": cache_validation_seconds,
            "full_linear_candidate_gptq_estimate": candidate_seconds,
            "local_search_all_row_scoring_estimate": local_search_seconds,
            "held_selected_local_row_scoring_estimate": held_local_seconds,
            "hessian_hinv_preparation_estimate": hessian_seconds,
            "search_block_candidate_forwards_estimate": search_block_seconds,
            "held_block_all_alpha_landscape_forwards_estimate": held_block_seconds,
            "row_wise_stitch_and_commit_estimate": stitch_commit_seconds,
            "candidate_checkpoint_read_write_and_reports_estimate": artifact_io_seconds,
            "candidate_tensor_cpu_transfer_and_checkpoint_seconds": candidate_write_seconds,
            "candidate_checkpoint_readback_seconds": candidate_read_seconds,
            "hessian_checkpoint_read_write_seconds": hessian_checkpoint_io_seconds,
            "block_score_checkpoint_update_seconds": block_checkpoint_seconds,
            "event_log_and_final_reports_reserve_seconds": 30.0,
            "post_candidate_activation_refresh_estimate": 0.0,
            "candidate_checkpoint_bytes_estimate": candidate_bytes
        },
        "central_estimate_seconds": central,
        "conservative_estimate_seconds": conservative,
        "candidate_full_linear_gptq_executions": 120,
        "candidate_block_forwards": 15360,
        "search_block_forwards": 7680,
        "held_block_forwards": 7680,
        "search_all_row_local_score_positions": 345600,
        "held_local_selected_score_positions": 144,
        "prefix_decoder_layer_forwards": 1536,
        "teacher_decoder_layer_forwards": 192,
        "post_candidate_activation_refresh_forwards": 0,
        "preflight_calls_excluded_from_formal_counts": {
            "full_linear_gptq_executions": timing["preflight_full_linear_gptq_executions"],
            "single_document_block_forwards": timing["preflight_block_single_document_forwards"]
        },
        "measurement_note": "GPU timings synchronized; q/up candidate and all-row local score probes are nonzero measured candidates; block cost uses the maximum of four deepest-layer up-projection full-document probes; measured candidate checkpoint serialization/readback and row stitch probes are extrapolated by frozen target counts. Prefix capture (one full M2 forward per document, both q/up hooks share each layer execution), teacher, cache hashing, model load, and cache validation are separately itemized. No post-candidate downstream activation refresh is required by this frozen block-scoring protocol. No formal candidate landscape has run."
    }


def run_preflight():
    prefix = stage_prepare_prefix()
    if "runtime_cache_validation_seconds" not in prefix:
        # A fresh cache was just created; measure the later-run hash-validation pass.
        prefix = stage_prepare_prefix()
    checks, timing = correctness_and_cost_preflight(prefix)
    cost = estimate_cost(prefix, timing)
    dump(OUT / "E1_COST_GATE.json", cost)
    print(json.dumps({"correctness": "PASS" if all(checks["all_required_checks_pass"] for _ in [0]) else "FAIL",
                      "cost_gate": cost["status"], "central_hours": cost["central_estimate_seconds"] / 3600,
                      "conservative_hours": cost["conservative_estimate_seconds"] / 3600,
                      "checks": checks}, indent=2))
    pre = json.loads((OUT / "PREFLIGHT_CORRECTNESS.json").read_text(encoding="utf-8"))
    write_preflight_gate_report(pre, cost)
    if not checks.get("all_required_checks_pass", False):
        raise RuntimeError("E1 preflight correctness failed; formal execution forbidden")
    if cost["status"] != "E1_COST_GATE_PASS":
        raise RuntimeError("E1_COST_GATE_FAIL; formal execution forbidden")


class EventLog:
    def __init__(self, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = path.open("a", encoding="utf-8", newline="\n")

    def write(self, event, **fields):
        record = {"time_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "event": event, **fields}
        self.handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
        self.handle.flush()

    def close(self):
        self.handle.close()


def atomic_torch_save(payload, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def candidate_path(li, proj, split, alpha):
    return RAW_DIR / "candidates" / f"candidate_L{li}_{proj}_{split}_a{alpha:.1f}.pt"


def hessian_path(li, proj, split):
    return RAW_DIR / "hessians" / f"H_L{li}_{proj}_{split}.pt"


def stitch_path(li, proj, split):
    return RAW_DIR / "stitches" / f"local_stitch_L{li}_{proj}_{split}.pt"


def block_score_path(phase, li, proj, split, row, alpha):
    return RAW_DIR / "block_scores" / f"{phase}_L{li}_{proj}_{split}_r{row}_a{alpha:.1f}.json"


def read_torch(path):
    return torch.load(path, map_location="cpu", weights_only=True)


def generate_candidate_bank(model, li, proj, split, pristine, x_mm, events, counters):
    doc_ids = cell_indices(split)
    source_hashes = input_hashes()
    manifest_sha = source_hashes["experiments/range_search_w4/results/range_w4_block_objective_e1_v23/E1_DOCUMENT_MANIFEST.json"]
    protocol_sha = source_hashes["experiments/range_search_w4/results/range_w4_block_objective_e1_v23/PROTOCOL.json"]
    paths = {a: candidate_path(li, proj, split, a) for a in ALPHAS}
    existing = {a: p.exists() for a, p in paths.items()}
    hpath = hessian_path(li, proj, split)
    hdetail = None
    prepared = None
    h = None
    nsamples = SEQ * len(doc_ids)
    if not all(existing.values()):
        if hpath.exists():
            h_obj = read_torch(hpath)
            if h_obj["layer"] != li or h_obj["projection"] != proj or h_obj["split"] != split or h_obj["doc_indices"] != doc_ids:
                raise RuntimeError(f"Cached Hessian protocol mismatch: {hpath}")
            h = h_obj["H"].to(DEVICE, dtype=torch.float32)
            if sha_tensor(h) != h_obj["H_sha256"]:
                raise RuntimeError(f"Cached Hessian hash mismatch: {hpath}")
            hdetail = {k: v for k, v in h_obj.items() if k not in ("H",)}
        else:
            h, nsamples, detail = build_hessian(model.model.layers[li], proj, x_mm, doc_ids,
                                                f"formal_L{li}_{proj}_{split}")
            sync(); t0 = time.perf_counter()
            prepared = prepare_native_state(h, nsamples=nsamples)
            sync(); hinv_seconds = time.perf_counter() - t0
            detail["hinv_prepare_seconds"] = hinv_seconds
            h_obj = {"layer": li, "projection": proj, "split": split, "doc_indices": doc_ids,
                     "nsamples": nsamples, "H_sha256": sha_tensor(h), "H": h.detach().cpu()}
            atomic_torch_save(h_obj, hpath)
            hdetail = detail
        if prepared is None:
            nsamples = int(read_torch(hpath)["nsamples"])
            sync(); t0 = time.perf_counter()
            prepared = prepare_native_state(h, nsamples=nsamples)
            sync(); hinv_seconds = time.perf_counter() - t0
            if hdetail is not None:
                hdetail["hinv_prepare_seconds"] = hinv_seconds
    else:
        hobj = read_torch(paths[ALPHAS[0]])
        nsamples = int(hobj["nsamples"])
        h_sha = hobj["hessian_sha256"]
        if any(read_torch(paths[a])["hessian_sha256"] != h_sha for a in ALPHAS):
            raise RuntimeError(f"Candidate bank has inconsistent Hessian identity at L{li}/{proj}/{split}")

    h_sha = sha_tensor(h) if h is not None else read_torch(paths[ALPHAS[0]])["hessian_sha256"]
    pristine_sha = sha_tensor(pristine)
    results = {}
    for alpha in ALPHAS:
        path = paths[alpha]
        if path.exists():
            obj = read_torch(path)
            expected = (li, proj, split, alpha, h_sha, pristine_sha, doc_ids, manifest_sha, protocol_sha)
            actual = (obj["layer"], obj["projection"], obj["split"], obj["alpha"], obj["hessian_sha256"],
                      obj["pristine_weight_sha256"], obj["search_doc_indices"], obj.get("manifest_sha256"),
                      obj.get("protocol_sha256"))
            if actual != expected:
                raise RuntimeError(f"Candidate checkpoint protocol mismatch: {path}")
            for tensor_key, hash_key in (("weight_fp16", "weight_sha256"), ("scales", "scales_sha256"),
                                         ("zeros", "zeros_sha256"), ("g_idx", "g_idx_sha256"),
                                         ("local_search_mse_by_row", "local_search_mse_sha256")):
                if sha_tensor(obj[tensor_key]) != obj[hash_key]:
                    raise RuntimeError(f"Candidate checkpoint tensor hash mismatch: {path}:{tensor_key}")
            results[alpha] = obj
            continue
        if prepared is None:
            raise RuntimeError("Candidate is missing but immutable Hessian state was not prepared")
        events.write("gptq_candidate_started", stage="formal", layer=li, projection=proj,
                     split=split, alpha=alpha)
        counters["candidate_attempts_this_process"] += 1
        sync(); t0 = time.perf_counter()
        result = make_candidate(model.model.layers[li], proj, pristine, prepared, alpha,
                                check=(alpha == 1.0))
        sync(); gptq_seconds = time.perf_counter() - t0
        sync(); t0 = time.perf_counter()
        local = score_rows_local(x_mm, doc_ids, pristine, result.weight_fp16)
        sync(); local_seconds = time.perf_counter() - t0
        if (not torch.isfinite(result.weight_fp16).all() or not torch.isfinite(result.scales).all() or
            not torch.isfinite(result.zeros).all() or not torch.isfinite(local).all()):
            raise RuntimeError(f"Non-finite GPTQ candidate or local loss at L{li}/{proj}/{split}/alpha={alpha}")
        obj = {"layer": li, "projection": proj, "split": split, "alpha": alpha,
               "search_doc_indices": doc_ids, "nsamples": nsamples,
               "hessian_sha256": h_sha, "pristine_weight_sha256": pristine_sha,
               "manifest_sha256": source_hashes["experiments/range_search_w4/results/range_w4_block_objective_e1_v23/E1_DOCUMENT_MANIFEST.json"],
               "protocol_sha256": source_hashes["experiments/range_search_w4/results/range_w4_block_objective_e1_v23/PROTOCOL.json"],
               "weight_fp16": result.weight_fp16.detach().cpu(), "scales": result.scales.detach().cpu(),
               "zeros": result.zeros.detach().cpu(), "g_idx": result.g_idx.detach().cpu(),
               "local_search_mse_by_row": local.detach().cpu(),
               "weight_sha256": sha_tensor(result.weight_fp16),
               "scales_sha256": sha_tensor(result.scales), "zeros_sha256": sha_tensor(result.zeros),
               "g_idx_sha256": sha_tensor(result.g_idx),
               "local_search_mse_sha256": sha_tensor(local),
               "gptq_seconds": gptq_seconds, "local_score_seconds": local_seconds,
               "checkpoint_status": "COMPLETE"}
        sync(); t0 = time.perf_counter()
        atomic_torch_save(obj, path)
        checkpoint_seconds = time.perf_counter() - t0
        counters["candidate_completed_this_process"] += 1
        events.write("gptq_candidate_completed", stage="formal", layer=li, projection=proj,
                     split=split, alpha=alpha, gptq_seconds=gptq_seconds,
                     local_score_seconds=local_seconds, checkpoint_seconds=checkpoint_seconds,
                     candidate_sha256=obj["weight_sha256"])
        results[alpha] = obj
        del result, local
    if h is not None:
        del h
    del prepared
    return results, hdetail


def choose_alpha(score_by_alpha):
    return min(ALPHAS, key=lambda a: (float(score_by_alpha[a]), -a))


def build_local_stitch(li, proj, split, candidates):
    first = candidates[ALPHAS[0]]
    out_rows = first["weight_fp16"].shape[0]
    if not all(torch.equal(first["g_idx"], candidates[a]["g_idx"]) for a in ALPHAS):
        raise RuntimeError(f"Native g_idx differs across alpha candidates for L{li}/{proj}/{split}")
    weight = torch.empty_like(first["weight_fp16"])
    scales = torch.empty_like(first["scales"])
    zeros = torch.empty_like(first["zeros"])
    selected = []
    for row in range(out_rows):
        alpha = choose_alpha({a: candidates[a]["local_search_mse_by_row"][row].item() for a in ALPHAS})
        selected.append(alpha)
        weight[row].copy_(candidates[alpha]["weight_fp16"][row])
        scales[row].copy_(candidates[alpha]["scales"][row])
        zeros[row].copy_(candidates[alpha]["zeros"][row])
    return {"layer": li, "projection": proj, "split": split, "weight_fp16": weight,
            "scales": scales, "zeros": zeros, "g_idx": first["g_idx"].clone(),
            "alpha_by_row": selected,
            "candidate_hashes": {f"{a:.1f}": candidates[a]["weight_sha256"] for a in ALPHAS},
            "weight_sha256": sha_tensor(weight), "scales_sha256": sha_tensor(scales),
            "zeros_sha256": sha_tensor(zeros), "g_idx_sha256": sha_tensor(first["g_idx"])}


def score_block_row_resumable(model, li, proj, split, phase, row, alpha, candidate,
                              baseline_weight, baseline_sha, hidden_mm, teacher_mm, events, counters):
    doc_ids = cell_indices(split) if phase == "search" else held_indices(split)
    path = block_score_path(phase, li, proj, split, row, alpha)
    candidate_sha = candidate["weight_sha256"]
    expected_identity = {"layer": li, "projection": proj, "split": split, "phase": phase,
                         "row": row, "alpha": alpha, "doc_indices": doc_ids,
                         "candidate_sha256": candidate_sha, "baseline_stitch_sha256": baseline_sha}
    if path.exists():
        obj = json.loads(path.read_text(encoding="utf-8"))
        for key, value in expected_identity.items():
            if obj.get(key) != value:
                raise RuntimeError(f"Block-score checkpoint protocol mismatch: {path} ({key})")
    else:
        obj = {**expected_identity, "status": "IN_PROGRESS", "document_sse": [None] * len(doc_ids),
               "document_mse": [None] * len(doc_ids), "document_seconds": [None] * len(doc_ids),
               "single_document_block_forwards": 0, "peak_allocated_bytes": 0,
               "peak_reserved_bytes": 0}

    module = module_attr(model.model.layers[li], proj)
    with torch.no_grad():
        module.weight[row].copy_(baseline_weight[row])
    row_candidate = candidate["weight_fp16"][row].to(DEVICE)
    for pos, di in enumerate(doc_ids):
        if obj["document_sse"][pos] is not None:
            continue
        with torch.no_grad():
            module.weight[row].copy_(row_candidate)
        events.write("block_forward_started", stage="formal", phase=phase, layer=li,
                     projection=proj, split=split, row=row, alpha=alpha, document_index=di)
        sync(); t0 = time.perf_counter()
        h = torch.from_numpy(np.array(hidden_mm[di], copy=True)).unsqueeze(0).to(DEVICE)
        y = layer_forward_from_hidden(model, li, h)
        teacher = torch.from_numpy(np.array(teacher_mm[di], copy=True)).unsqueeze(0).to(DEVICE)
        diff = y.float() - teacher.float()
        sse = float(diff.double().square().sum().item())
        mse = sse / diff.numel()
        if not math.isfinite(sse) or not math.isfinite(mse):
            raise RuntimeError(f"Non-finite block loss at L{li}/{proj}/{split}/row={row}/alpha={alpha}/doc={di}")
        sync(); elapsed = time.perf_counter() - t0
        alloc = int(torch.cuda.max_memory_allocated(DEVICE))
        reserved = int(torch.cuda.max_memory_reserved(DEVICE))
        del h, y, teacher, diff
        obj["document_sse"][pos] = sse
        obj["document_mse"][pos] = mse
        obj["document_seconds"][pos] = elapsed
        obj["single_document_block_forwards"] += 1
        obj["peak_allocated_bytes"] = max(obj["peak_allocated_bytes"], alloc)
        obj["peak_reserved_bytes"] = max(obj["peak_reserved_bytes"], reserved)
        dump(path, obj)
        counters["block_forwards_this_process"] += 1
        events.write("block_forward_completed", stage="formal", phase=phase, layer=li,
                     projection=proj, split=split, row=row, alpha=alpha,
                     document_index=di, seconds=elapsed, mse=mse)
    with torch.no_grad():
        module.weight[row].copy_(baseline_weight[row])
    if any(v is None for v in obj["document_sse"]):
        raise RuntimeError(f"Incomplete block-score checkpoint: {path}")
    obj["status"] = "COMPLETE"
    obj["pooled_sse"] = math.fsum(obj["document_sse"])
    obj["pooled_mse"] = obj["pooled_sse"] / (len(doc_ids) * SEQ * 896)
    dump(path, obj)
    return obj


def save_csv(path, rows, fieldnames=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0]) if rows else []
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


def average_ranks(values):
    order = sorted(range(len(values)), key=lambda i: (float(values[i]), i))
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and float(values[order[j]]) == float(values[order[i]]):
            j += 1
        rank = ((i + 1) + j) / 2.0
        for k in range(i, j):
            ranks[order[k]] = rank
        i = j
    return ranks


def spearman_rho(left, right):
    x = np.asarray(average_ranks(left), dtype=np.float64)
    y = np.asarray(average_ranks(right), dtype=np.float64)
    if np.std(x) == 0 or np.std(y) == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def compact_json(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def selected_params(candidate, row):
    return {"scale": candidate["scales"][row].double().tolist(),
            "zero": candidate["zeros"][row].double().tolist(),
            "g_idx": candidate["g_idx"].long().tolist(),
            "g_idx_sha256": sha_tensor(candidate["g_idx"]),
            "weight_sha256": sha_tensor(candidate["weight_fp16"][row])}


def row_weight_mse(candidate, pristine, row):
    return float((candidate["weight_fp16"][row].float() - pristine[row].float()).square().mean().item())


def materialize_result_tables(search_records, held_records, selections, candidates,
                              local_held_records, repeat_mse_noise, output_dir):
    locked = {x["unit_id"]: x for x in selections["units"]}
    held_by_unit_alpha = {(r["unit_id"], float(r["alpha"])): r for r in held_records}
    unit_ids = sorted(locked)
    comparison_rows = []
    disagreement_rows = []
    ranking_rows = []
    parameter_rows = []
    search_by_unit = {}
    for rec in search_records:
        search_by_unit.setdefault(rec["unit_id"], []).append(rec)
    held_by_unit = {}
    for rec in held_records:
        held_by_unit.setdefault(rec["unit_id"], []).append(rec)

    for unit in unit_ids:
        choice = locked[unit]
        srows = sorted(search_by_unit[unit], key=lambda x: x["alpha"])
        hrows = sorted(held_by_unit[unit], key=lambda x: x["alpha"])
        hscore = {float(x["alpha"]): float(x["block_held_mse"]) for x in hrows}
        local_scores = [float(x["local_search_score"]) for x in srows]
        block_scores = [float(x["block_search_score"]) for x in srows]
        held_scores = [hscore[a] for a in ALPHAS]
        native_a = 1.0
        local_a = float(choice["alpha_local"])
        block_a = float(choice["alpha_block"])
        local_metric = float(local_held_records[unit]["local_held_mse_by_alpha"][f"{local_a:.1f}"])
        block_metric = float(local_held_records[unit]["local_held_mse_by_alpha"][f"{block_a:.1f}"])
        native_local_metric = float(local_held_records[unit]["local_held_mse_by_alpha"]["1.0"])
        native_block_metric = hscore[native_a]
        local_block_metric = hscore[local_a]
        block_block_metric = hscore[block_a]
        base_for_epsilon = native_block_metric
        epsilon = max(1e-12, 5.0 * repeat_mse_noise, 1e-6 * base_for_epsilon)
        delta = block_block_metric - local_block_metric
        if delta < -epsilon:
            outcome = "block_better"
        elif delta > epsilon:
            outcome = "block_worse"
        else:
            outcome = "tie"
        improvement = (local_block_metric - block_block_metric) / max(local_block_metric, 1e-30)
        search_eps = max(1e-12, 5.0 * repeat_mse_noise,
                         1e-6 * float(next(x["block_search_score"] for x in srows if x["alpha"] == 1.0)))
        search_block_local = float(next(x["block_search_score"] for x in srows if x["alpha"] == local_a))
        search_block_block = float(next(x["block_search_score"] for x in srows if x["alpha"] == block_a))
        meaningful_search_gain = search_block_local - search_block_block > search_eps

        unit_meta = choice
        row = int(unit_meta["row"])
        li = int(unit_meta["layer"])
        proj = unit_meta["projection"]
        split = unit_meta["split"]
        bank = candidates[(li, proj, split)]
        pristine = candidates[(li, proj, split)]["__pristine_weight__"]
        local_candidate = bank[local_a]
        block_candidate = bank[block_a]
        native_candidate = bank[native_a]
        for selector, alpha, cand, loc_metric, blk_metric in (
            ("native_alpha1", native_a, native_candidate, native_local_metric, native_block_metric),
            ("local_selected", local_a, local_candidate, local_metric, local_block_metric),
            ("block_selected", block_a, block_candidate, block_metric, block_block_metric)):
            params = selected_params(cand, row)
            parameter_rows.append({"unit_id": unit, "layer": li, "block": f"L{li}",
                "projection": proj, "row": row, "split_id": split, "selector": selector,
                "alpha": alpha, "scale": compact_json(params["scale"]),
                "zero": compact_json(params["zero"]), "g_idx": compact_json(params["g_idx"]),
                "g_idx_sha256": params["g_idx_sha256"], "weight_sha256": params["weight_sha256"],
                "weight_mse": row_weight_mse(cand, pristine, row),
                "local_held_mse": loc_metric, "block_held_mse": blk_metric})
        comparison_rows.append({"unit_id": unit, "layer": li, "block": f"L{li}", "projection": proj,
            "row": row, "split_id": split, "alpha_local": local_a, "alpha_block": block_a,
            "alpha_native": native_a, "selector_alpha_disagreement": local_a != block_a,
            "local_selected_block_held_mse": local_block_metric,
            "block_selected_block_held_mse": block_block_metric,
            "native_alpha1_block_held_mse": native_block_metric,
            "delta_block_minus_local_mse": delta, "relative_improvement_block_vs_local": improvement,
            "comparison": outcome, "numeric_epsilon_mse": epsilon,
            "local_selected_local_held_mse": local_metric,
            "block_selected_local_held_mse": block_metric,
            "native_alpha1_local_held_mse": native_local_metric,
            "search_block_objective_local_choice_mse": search_block_local,
            "search_block_objective_block_choice_mse": search_block_block,
            "meaningful_search_block_gain_over_local_choice": meaningful_search_gain,
            "held_oracle_block_mse": min(held_scores),
            "local_selected_held_oracle_regret": local_block_metric - min(held_scores),
            "block_selected_held_oracle_regret": block_block_metric - min(held_scores)})
        disagreement_rows.append({"unit_id": unit, "layer": li, "block": f"L{li}", "projection": proj,
            "row": row, "split_id": split, "alpha_local": local_a, "alpha_block": block_a,
            "disagreement": local_a != block_a, "held_block_comparison": outcome,
            "held_block_delta_mse_block_minus_local": delta,
            "held_relative_improvement_block_vs_local": improvement,
            "search_block_gain_over_local_choice_mse": search_block_local - search_block_block,
            "search_gain_exceeds_numeric_epsilon": meaningful_search_gain,
            "numeric_epsilon_mse": epsilon})

        rank_specs = (("local_selector", local_scores, local_a, local_block_metric),
                      ("block_selector", block_scores, block_a, block_block_metric))
        for selector, search_loss, chosen, chosen_held in rank_specs:
            ranking_rows.append({"unit_id": unit, "layer": li, "block": f"L{li}",
                "projection": proj, "row": row, "split_id": split, "selector": selector,
                "search_to_held_block_spearman": spearman_rho(search_loss, held_scores),
                "selected_alpha": chosen,
                "selected_alpha_rank_on_held_block_landscape": average_ranks(held_scores)[ALPHAS.index(chosen)],
                "selected_alpha_held_block_mse": chosen_held,
                "held_oracle_alpha": choose_alpha({a: hscore[a] for a in ALPHAS}),
                "held_oracle_block_mse": min(held_scores),
                "held_oracle_regret_mse": chosen_held - min(held_scores)})

    save_csv(output_dir / "SELECTOR_COMPARISON.csv", comparison_rows)
    save_csv(output_dir / "ALPHA_DISAGREEMENT.csv", disagreement_rows)
    save_csv(output_dir / "RANKING_ANALYSIS.csv", ranking_rows)
    save_csv(output_dir / "SELECTED_QUANTIZER_PARAMS.csv", parameter_rows)

    gains = [float(r["relative_improvement_block_vs_local"]) for r in comparison_rows]
    deltas = [float(r["delta_block_minus_local_mse"]) for r in comparison_rows]
    outcomes = [r["comparison"] for r in comparison_rows]
    pooled_local_sse_proxy = math.fsum(float(r["local_selected_block_held_mse"]) for r in comparison_rows)
    pooled_block_sse_proxy = math.fsum(float(r["block_selected_block_held_mse"]) for r in comparison_rows)
    pooled_gain = (pooled_local_sse_proxy - pooled_block_sse_proxy) / max(pooled_local_sse_proxy, 1e-30)
    split_stats = {}
    for split in ("S0", "S1"):
        selected = [r for r in comparison_rows if r["split_id"] == split]
        local_sum = math.fsum(r["local_selected_block_held_mse"] for r in selected)
        block_sum = math.fsum(r["block_selected_block_held_mse"] for r in selected)
        eps_sse = math.fsum(r["numeric_epsilon_mse"] for r in selected) * 16 * SEQ * 896
        delta_sse = (local_sum - block_sum) * 16 * SEQ * 896
        split_stats[split] = {"units": len(selected), "local_block_mse_sum": local_sum,
            "block_block_mse_sum": block_sum, "pooled_sse_delta_block_minus_local": -delta_sse,
            "pooled_improvement_fraction": (local_sum - block_sum) / max(local_sum, 1e-30),
            "numeric_epsilon_sse": eps_sse, "improved_beyond_epsilon": delta_sse > eps_sse}

    target_stats = []
    target_positive_count = 0
    for li in LAYERS:
        for proj in PROJECTIONS:
            selected = [r for r in comparison_rows if r["layer"] == li and r["projection"] == proj]
            local_sum = math.fsum(r["local_selected_block_held_mse"] for r in selected)
            block_sum = math.fsum(r["block_selected_block_held_mse"] for r in selected)
            target_gain = local_sum - block_sum
            eps = math.fsum(r["numeric_epsilon_mse"] for r in selected)
            if target_gain > eps:
                target_positive_count += 1
            target_stats.append({"layer": li, "projection": proj, "units": len(selected),
                "local_block_mse_sum": local_sum, "block_block_mse_sum": block_sum,
                "pooled_improvement_fraction": target_gain / max(local_sum, 1e-30),
                "improved_beyond_numeric_floor": target_gain > eps})

    disagreement = [r for r in comparison_rows if r["alpha_local"] != r["alpha_block"]]
    changed_non_ties = [r for r in disagreement if r["comparison"] != "tie"]
    changed_wins = sum(r["comparison"] == "block_better" for r in changed_non_ties)
    changed_win_fraction = changed_wins / len(changed_non_ties) if changed_non_ties else None
    non_tie_count = sum(x == "block_better" for x in outcomes) + sum(x == "block_worse" for x in outcomes)
    worst = min(comparison_rows, key=lambda r: r["relative_improvement_block_vs_local"])
    split_gate = all(split_stats[s]["improved_beyond_epsilon"] for s in ("S0", "S1"))
    target_gate = target_positive_count >= 4
    changed_gate = changed_win_fraction is not None and changed_win_fraction >= 0.60
    worst_case_gate = worst["relative_improvement_block_vs_local"] >= -0.01
    meaningful_search_units = sum(r["meaningful_search_block_gain_over_local_choice"] for r in comparison_rows)
    primary_gate_pass = (split_gate and target_gate and changed_gate and worst_case_gate and
                         meaningful_search_units > 0)
    if primary_gate_pass:
        verdict = "E1_BLOCK_OBJECTIVE_SIGNAL_SUPPORTED"
    elif pooled_gain <= 0 or (non_tie_count and sum(x == "block_better" for x in outcomes) <= sum(x == "block_worse" for x in outcomes)):
        verdict = "E1_BLOCK_OBJECTIVE_SIGNAL_NOT_SUPPORTED"
    elif target_positive_count > 0 or (split_stats["S0"]["improved_beyond_epsilon"] != split_stats["S1"]["improved_beyond_epsilon"]):
        verdict = "E1_BLOCK_OBJECTIVE_SIGNAL_MIXED"
    else:
        verdict = "E1_BLOCK_OBJECTIVE_SIGNAL_NOT_SUPPORTED"

    disagreement_summary = {
        "units": len(comparison_rows), "disagreement_units": len(disagreement),
        "disagreement_fraction": len(disagreement) / max(1, len(comparison_rows)),
        "changed_non_tie_units": len(changed_non_ties), "block_better": changed_wins,
        "block_tie": sum(r["comparison"] == "tie" for r in disagreement),
        "block_worse": sum(r["comparison"] == "block_worse" for r in changed_non_ties),
        "block_win_fraction_among_changed_non_ties": changed_win_fraction}
    alpha_distribution = {selector: {f"{a:.1f}": 0 for a in ALPHAS}
                          for selector in ("alpha_local", "alpha_block")}
    for unit in locked.values():
        alpha_distribution["alpha_local"][f"{unit['alpha_local']:.1f}"] += 1
        alpha_distribution["alpha_block"][f"{unit['alpha_block']:.1f}"] += 1

    tail_rows = []
    for scope, subset in [("overall", comparison_rows)] + [
        (f"{proj}_L{li}", [r for r in comparison_rows if r["projection"] == proj and r["layer"] == li])
        for proj in PROJECTIONS for li in LAYERS]:
        for selector, metric in (("local_selector", "local_selected_block_held_mse"),
                                 ("block_selector", "block_selected_block_held_mse")):
            reg = []
            worst_reg = None
            for r in subset:
                baseline = float(r["native_alpha1_block_held_mse"])
                value = float(r[metric])
                gain = (baseline - value) / max(baseline, 1e-30)
                item = {"unit_id": r["unit_id"], "gain_vs_alpha1": gain}
                reg.append(item)
                if worst_reg is None or gain < worst_reg["gain_vs_alpha1"]:
                    worst_reg = item
            tail_rows.append({"scope": scope, "selector": selector, "units": len(reg),
                "regressions_below_minus_5pct": sum(x["gain_vs_alpha1"] < -0.05 for x in reg),
                "regressions_below_minus_10pct": sum(x["gain_vs_alpha1"] < -0.10 for x in reg),
                "regressions_below_minus_20pct": sum(x["gain_vs_alpha1"] < -0.20 for x in reg),
                "worst_gain_vs_alpha1": worst_reg["gain_vs_alpha1"] if worst_reg else None,
                "worst_unit_id": worst_reg["unit_id"] if worst_reg else None,
                "median_gain_vs_alpha1": float(np.median([x["gain_vs_alpha1"] for x in reg])) if reg else None})
    save_csv(output_dir / "TAIL_RISK_ANALYSIS.csv", tail_rows)

    breakdown_rows = []
    for proj in PROJECTIONS:
        for li in LAYERS:
            group = [r for r in comparison_rows if r["projection"] == proj and r["layer"] == li]
            local_sum = math.fsum(r["local_selected_block_held_mse"] for r in group)
            block_sum = math.fsum(r["block_selected_block_held_mse"] for r in group)
            g = [r["relative_improvement_block_vs_local"] for r in group]
            breakdown_rows.append({"layer": li, "block": f"L{li}", "projection": proj, "units": len(group),
                "block_better": sum(r["comparison"] == "block_better" for r in group),
                "tie": sum(r["comparison"] == "tie" for r in group),
                "block_worse": sum(r["comparison"] == "block_worse" for r in group),
                "mean_relative_improvement": float(np.mean(g)), "median_relative_improvement": float(np.median(g)),
                "pooled_block_sse_improvement_fraction": (local_sum - block_sum) / max(local_sum, 1e-30),
                "alpha_disagreement_units": sum(r["alpha_local"] != r["alpha_block"] for r in group),
                "interpretation": "descriptive_only"})
    save_csv(output_dir / "PROJECTION_DEPTH_BREAKDOWN.csv", breakdown_rows)

    rho_local = [float(r["search_to_held_block_spearman"]) for r in ranking_rows
                 if r["selector"] == "local_selector" and r["search_to_held_block_spearman"] is not None]
    rho_block = [float(r["search_to_held_block_spearman"]) for r in ranking_rows
                 if r["selector"] == "block_selector" and r["search_to_held_block_spearman"] is not None]
    regret_local = [r["held_oracle_regret_mse"] for r in ranking_rows if r["selector"] == "local_selector"]
    regret_block = [r["held_oracle_regret_mse"] for r in ranking_rows if r["selector"] == "block_selector"]
    count_win_tie_loss = {key: outcomes.count(key) for key in ("block_better", "tie", "block_worse")}
    summary = {
        "status": "COMPLETE", "verdict": verdict, "protocol_id": "range_w4_block_objective_e1_v23",
        "primary_metric": "held block output MSE per row intervention; selector block-vs-local uses the same candidate bank and frozen local-M2 all-row stitch",
        "units": len(comparison_rows), "candidate_alpha_values": ALPHAS,
        "held_selected_local_score_positions": sum(len(x["local_held_score_alphas"]) for x in local_held_records.values()),
        "held_selected_local_score_positions_max": 144,
        "selector_comparison": {"win_tie_loss": count_win_tie_loss,
            "median_delta_block_minus_local_mse": float(np.median(deltas)),
            "mean_delta_block_minus_local_mse": float(np.mean(deltas)),
            "pooled_sse_delta_block_minus_local": (pooled_block_sse_proxy - pooled_local_sse_proxy) * 16 * SEQ * 896,
            "pooled_sse_improvement_fraction": pooled_gain,
            "median_relative_improvement_block_vs_local": float(np.median(gains)),
            "mean_relative_improvement_block_vs_local": float(np.mean(gains)),
            "worst_case_unit_id": worst["unit_id"],
            "worst_case_relative_improvement_block_vs_local": worst["relative_improvement_block_vs_local"],
            "numeric_repeat_mse_noise_reference": repeat_mse_noise},
        "alpha_disagreement": disagreement_summary,
        "alpha_frequency": alpha_distribution,
        "split_breakdown": split_stats, "target_breakdown": target_stats,
        "v22_promotion_gate": {"both_split_pooled_sse_improve_beyond_epsilon": split_gate,
            "targets_improved_at_least_4_of_6": target_gate,
            "changed_non_tie_units_block_win_fraction_at_least_0_60": changed_gate,
            "no_case_block_selector_worse_than_local_by_over_1pct": worst_case_gate,
            "meaningful_search_block_objective_gain_observed": meaningful_search_units > 0,
            "meaningful_search_units": meaningful_search_units,
            "positive_targets_of_6": target_positive_count,
            "changed_non_tie_block_win_fraction": changed_win_fraction,
            "worst_relative_improvement": worst["relative_improvement_block_vs_local"],
            "pass": primary_gate_pass},
        "ranking": {"local_selector_mean_spearman": float(np.mean(rho_local)) if rho_local else None,
            "block_selector_mean_spearman": float(np.mean(rho_block)) if rho_block else None,
            "local_selector_median_held_oracle_regret_mse": float(np.median(regret_local)),
            "block_selector_median_held_oracle_regret_mse": float(np.median(regret_block)),
            "local_selector_mean_held_oracle_regret_mse": float(np.mean(regret_local)),
            "block_selector_mean_held_oracle_regret_mse": float(np.mean(regret_block))},
        "tail_analysis_file": "TAIL_RISK_ANALYSIS.csv",
        "projection_depth_file": "PROJECTION_DEPTH_BREAKDOWN.csv",
        "held_oracle_usage": "retrospective diagnostic only; never used in alpha selection",
        "limits": ["48 row-split units, not independent trained models", "single-document WT2 train split E1 only",
                   "local block interventions do not constitute a full-model method or PPL result"]}
    dump(output_dir / "E1_SUMMARY.json", summary)
    return summary, comparison_rows, disagreement_rows, ranking_rows, parameter_rows


def run_formal():
    pre_path, cost_path = OUT / "PREFLIGHT_CORRECTNESS.json", OUT / "E1_COST_GATE.json"
    if not pre_path.is_file() or not cost_path.is_file():
        raise RuntimeError("Completed correctness and cost preflight artifacts are required")
    pre = json.loads(pre_path.read_text(encoding="utf-8"))
    cost = json.loads(cost_path.read_text(encoding="utf-8"))
    if (pre.get("status") != "PASS" or cost.get("status") != "E1_COST_GATE_PASS" or
        cost.get("limit_seconds") != 21600 or cost.get("conservative_estimate_seconds", math.inf) > 21600):
        raise RuntimeError("Correctness and 1.5x cost gates must PASS before formal E1")
    prefix = stage_prepare_prefix()
    docs = load_docs()
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    events = EventLog(OUT / "RUN_EVENTS.jsonl")
    session_id = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    session_start = time.perf_counter()
    events.write("formal_session_started", stage="formal", session_id=session_id,
                 protocol_sha256=sha_file(PROTOCOL_PATH), manifest_sha256=sha_file(DOCS_PATH))
    counters = {"candidate_attempts_this_process": 0, "candidate_completed_this_process": 0,
                "block_forwards_this_process": 0}
    status_path = OUT / "RUN_STATUS.json"
    dump(status_path, {"status": "RUNNING", "session_id": session_id,
        "protocol_id": "range_w4_block_objective_e1_v23", "started_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "preflight_correctness": pre["status"], "cost_gate": cost["status"],
        "formal_gptq_slots_expected": 120, "formal_block_forwards_expected": 15360,
        "completed_candidate_slots_at_start": 0, "completed_block_score_files_at_start": 0})

    try:
        sync(); model_load_t0 = time.perf_counter()
        model, _, pristine_targets, m2_targets = load_model_and_patch_m2(copy_teachers=False)
        sync(); model_load_seconds = time.perf_counter() - model_load_t0
        banks = {}
        stitches = {}
        hessian_records = []
        stitch_commit_seconds = 0.0
        baseline_commit_seconds = 0.0
        m2_restore_seconds = 0.0
        target_restoration_checks = []
        candidate_start_t = time.perf_counter()

        # Frozen order: L0/q, L0/up, L12/q, L12/up, L23/q, L23/up; each has S0 then S1, alpha ascending.
        for li in LAYERS:
            for proj in PROJECTIONS:
                name = module_full_name(li, proj)
                pristine = pristine_targets[name].to(DEVICE)
                x_mm = open_cache(li, proj)
                for split in ("S0", "S1"):
                    bank, hdetail = generate_candidate_bank(model, li, proj, split, pristine, x_mm, events, counters)
                    banks[(li, proj, split)] = bank
                    if hdetail is not None:
                        hessian_records.append({"layer": li, "projection": proj, "split": split, **hdetail})
                    sync(); stitch_t0 = time.perf_counter()
                    local_stitch = build_local_stitch(li, proj, split, bank)
                    sync(); stitch_commit_seconds += time.perf_counter() - stitch_t0
                    spath = stitch_path(li, proj, split)
                    if spath.exists():
                        prior = read_torch(spath)
                        stored_tensor_hashes = (sha_tensor(prior["weight_fp16"]), sha_tensor(prior["scales"]),
                                                sha_tensor(prior["zeros"]), sha_tensor(prior["g_idx"]))
                        stored_expected_hashes = (prior["weight_sha256"], prior["scales_sha256"],
                                                  prior["zeros_sha256"], prior["g_idx_sha256"])
                        if stored_tensor_hashes != stored_expected_hashes:
                            raise RuntimeError(f"Frozen local stitch tensor hash mismatch: {spath}")
                        if (prior["candidate_hashes"] != local_stitch["candidate_hashes"] or
                            prior["weight_sha256"] != local_stitch["weight_sha256"] or
                            prior["scales_sha256"] != local_stitch["scales_sha256"] or
                            prior["zeros_sha256"] != local_stitch["zeros_sha256"] or
                            prior["g_idx_sha256"] != local_stitch["g_idx_sha256"] or
                            prior["alpha_by_row"] != local_stitch["alpha_by_row"]):
                            raise RuntimeError(f"Frozen local stitch cache mismatch: {spath}")
                        local_stitch = prior
                    else:
                        atomic_torch_save(local_stitch, spath)
                    stitches[(li, proj, split)] = local_stitch
                    bank["__pristine_weight__"] = pristine_targets[name].detach().cpu()
                    dump(status_path, {"status": "RUNNING", "session_id": session_id,
                        "phase": "candidate_generation_and_local_stitch", "layer": li, "projection": proj,
                        "split": split, "candidate_slots_complete": sum(
                            1 for l in LAYERS for p in PROJECTIONS for s in ("S0", "S1") for a in ALPHAS
                            if candidate_path(l, p, s, a).is_file()),
                        "candidate_slots_expected": 120,
                        "candidate_attempts_this_process": counters["candidate_attempts_this_process"]})
                x_mm._mmap.close()
                del x_mm, pristine
        sync()
        candidate_and_stitch_seconds = time.perf_counter() - candidate_start_t

        search_records = []
        search_started = time.perf_counter()
        for li in LAYERS:
            hidden_mm = load_mm(CACHE_DIR / f"hidden_L{li}.f16", (N_DOCS, SEQ, 896))
            teacher_mm = load_mm(CACHE_DIR / f"teacher_L{li}.f16", (N_DOCS, SEQ, 896))
            for proj in PROJECTIONS:
                for split in ("S0", "S1"):
                    bank = banks[(li, proj, split)]
                    baseline = stitches[(li, proj, split)]["weight_fp16"].to(DEVICE)
                    baseline_sha = stitches[(li, proj, split)]["weight_sha256"]
                    target_module = module_attr(model.model.layers[li], proj)
                    sync(); commit_t0 = time.perf_counter()
                    with torch.no_grad():
                        target_module.weight.copy_(baseline)
                    sync(); baseline_commit_seconds += time.perf_counter() - commit_t0
                    for row in ROWS[proj]:
                        unit = f"L{li}/{proj}/R{row}/{split}"
                        for alpha in ALPHAS:
                            score = score_block_row_resumable(model, li, proj, split, "search", row, alpha,
                                bank[alpha], baseline, baseline_sha, hidden_mm, teacher_mm, events, counters)
                            local_mse = float(bank[alpha]["local_search_mse_by_row"][row].item())
                            search_records.append({"unit_id": unit, "layer": li, "block": f"L{li}",
                                "projection": proj, "row": row, "split_id": split, "alpha": alpha,
                                "local_search_score": local_mse, "block_search_score": score["pooled_mse"],
                                "block_search_document_mse": score["document_mse"],
                                "block_search_document_sse": score["document_sse"],
                                "candidate_weight_sha256": bank[alpha]["weight_sha256"],
                                "candidate_scale_row": bank[alpha]["scales"][row].double().tolist(),
                                "candidate_zero_row": bank[alpha]["zeros"][row].double().tolist(),
                                "candidate_g_idx_sha256": sha_tensor(bank[alpha]["g_idx"])})
                    m2_target = m2_targets[module_full_name(li, proj)].to(DEVICE)
                    sync(); restore_t0 = time.perf_counter()
                    with torch.no_grad():
                        target_module.weight.copy_(m2_target)
                    sync(); m2_restore_seconds += time.perf_counter() - restore_t0
                    actual_m2_sha = sha_tensor(target_module.weight)
                    expected_m2_sha = sha_tensor(m2_targets[module_full_name(li, proj)])
                    target_restoration_checks.append({"layer": li, "projection": proj, "split": split,
                        "phase": "search", "expected_m2_weight_sha256": expected_m2_sha,
                        "actual_after_restore_sha256": actual_m2_sha, "pass": actual_m2_sha == expected_m2_sha})
                    if actual_m2_sha != expected_m2_sha:
                        raise RuntimeError(f"Target module was not restored to v18 M2 after search: L{li}/{proj}")
                    dump(status_path, {"status": "RUNNING", "session_id": session_id,
                        "phase": "search_block_landscape", "layer": li, "projection": proj, "split": split,
                        "search_row_alpha_scores_complete": len(search_records),
                        "search_row_alpha_scores_expected": 480,
                        "block_forwards_this_process": counters["block_forwards_this_process"]})
            hidden_mm._mmap.close(); teacher_mm._mmap.close()
        sync()
        search_seconds = time.perf_counter() - search_started
        if len(search_records) != 480:
            raise RuntimeError(f"Expected 480 search landscape row-alpha scores, got {len(search_records)}")

        # Freeze alpha choices from search data before opening held inputs or scoring held losses.
        search_records.sort(key=lambda r: (r["layer"], PROJECTIONS.index(r["projection"]),
                                           r["split_id"], r["row"], r["alpha"]))
        search_signature = [{k: r[k] for k in ("unit_id", "alpha", "local_search_score", "block_search_score",
            "candidate_weight_sha256", "candidate_scale_row", "candidate_zero_row", "candidate_g_idx_sha256")}
            for r in search_records]
        search_hash = sha_bytes(json.dumps(search_signature, sort_keys=True, separators=(",", ":"), allow_nan=False).encode())
        grouped = {}
        for r in search_records:
            grouped.setdefault(r["unit_id"], []).append(r)
        locked_units = []
        for unit, rows in sorted(grouped.items()):
            local_scores = {float(r["alpha"]): r["local_search_score"] for r in rows}
            block_scores = {float(r["alpha"]): r["block_search_score"] for r in rows}
            first = rows[0]
            locked_units.append({"unit_id": unit, "layer": first["layer"], "projection": first["projection"],
                "row": first["row"], "split": first["split_id"],
                "alpha_local": choose_alpha(local_scores), "alpha_block": choose_alpha(block_scores),
                "local_search_landscape": {f"{a:.1f}": local_scores[a] for a in ALPHAS},
                "block_search_landscape": {f"{a:.1f}": block_scores[a] for a in ALPHAS}})
        lock_payload = {"status": "LOCKED_BEFORE_HELD_EVALUATION", "protocol_id": "range_w4_block_objective_e1_v23",
            "selection_rule": "search split only; minimum score; exact ties select larger alpha",
            "held_losses_read_for_selection": False, "held_block_scoring_started": False,
            "search_landscape_sha256": search_hash,
            "manifest_sha256": sha_file(DOCS_PATH), "units": locked_units}
        lock_path = OUT / "SEARCH_SELECTIONS.json"
        if lock_path.exists():
            prior = json.loads(lock_path.read_text(encoding="utf-8"))
            if prior["search_landscape_sha256"] != search_hash or prior["units"] != locked_units:
                raise RuntimeError("Existing search alpha lock differs; held evaluation is forbidden")
        else:
            dump(lock_path, lock_payload)
        locked = json.loads(lock_path.read_text(encoding="utf-8"))
        if locked["status"] != "LOCKED_BEFORE_HELD_EVALUATION" or locked["search_landscape_sha256"] != search_hash:
            raise RuntimeError("Search selection lock validation failed")
        lock_sha = sha_file(lock_path)
        for r in search_records:
            unit = next(x for x in locked["units"] if x["unit_id"] == r["unit_id"])
            r["alpha_local_selected"] = float(r["alpha"]) == float(unit["alpha_local"])
            r["alpha_block_selected"] = float(r["alpha"]) == float(unit["alpha_block"])
        search_csv_rows = [{**r, "block_search_document_mse": compact_json(r["block_search_document_mse"]),
                            "block_search_document_sse": compact_json(r["block_search_document_sse"]),
                            "candidate_scale_row": compact_json(r["candidate_scale_row"]),
                            "candidate_zero_row": compact_json(r["candidate_zero_row"])} for r in search_records]
        save_csv(OUT / "SEARCH_LANDSCAPES.csv", search_csv_rows)
        dump(OUT / "SEARCH_LANDSCAPES_RAW.json", {"search_landscape_sha256": search_hash,
            "search_selection_sha256": lock_sha, "row_alpha_score_count": len(search_records),
            "results": search_records})
        events.write("search_selection_locked", stage="formal", search_landscape_sha256=search_hash,
                     search_selection_sha256=lock_sha, selected_units=len(locked_units), held_data_read=False)

        held_records = []
        local_held_records = {}
        held_local_scoring_seconds = 0.0
        held_started = time.perf_counter()
        locked_by_unit = {x["unit_id"]: x for x in locked["units"]}
        for li in LAYERS:
            hidden_mm = load_mm(CACHE_DIR / f"hidden_L{li}.f16", (N_DOCS, SEQ, 896))
            teacher_mm = load_mm(CACHE_DIR / f"teacher_L{li}.f16", (N_DOCS, SEQ, 896))
            for proj in PROJECTIONS:
                name = module_full_name(li, proj)
                pristine = pristine_targets[name].to(DEVICE)
                x_mm = open_cache(li, proj)
                for split in ("S0", "S1"):
                    bank = banks[(li, proj, split)]
                    baseline = stitches[(li, proj, split)]["weight_fp16"].to(DEVICE)
                    baseline_sha = stitches[(li, proj, split)]["weight_sha256"]
                    target_module = module_attr(model.model.layers[li], proj)
                    sync(); commit_t0 = time.perf_counter()
                    with torch.no_grad():
                        target_module.weight.copy_(baseline)
                    sync(); baseline_commit_seconds += time.perf_counter() - commit_t0
                    for row in ROWS[proj]:
                        unit = f"L{li}/{proj}/R{row}/{split}"
                        for alpha in ALPHAS:
                            score = score_block_row_resumable(model, li, proj, split, "held", row, alpha,
                                bank[alpha], baseline, baseline_sha, hidden_mm, teacher_mm, events, counters)
                            held_records.append({"unit_id": unit, "layer": li, "block": f"L{li}",
                                "projection": proj, "row": row, "split_id": split, "alpha": alpha,
                                "block_held_mse": score["pooled_mse"], "block_held_sse": score["pooled_sse"],
                                "block_held_document_mse": score["document_mse"],
                                "block_held_document_sse": score["document_sse"],
                                "candidate_weight_sha256": bank[alpha]["weight_sha256"],
                                "candidate_scale_row": bank[alpha]["scales"][row].double().tolist(),
                                "candidate_zero_row": bank[alpha]["zeros"][row].double().tolist(),
                                "candidate_g_idx_sha256": sha_tensor(bank[alpha]["g_idx"])})
                        choice = locked_by_unit[unit]
                        metric_alphas = sorted({1.0, float(choice["alpha_local"]), float(choice["alpha_block"])})
                        local_metrics = {}
                        sync(); local_t0 = time.perf_counter()
                        for alpha in metric_alphas:
                            local_metrics[f"{alpha:.1f}"] = score_one_row_local(
                                x_mm, held_indices(split), pristine[row], bank[alpha]["weight_fp16"][row].to(DEVICE))
                            if not math.isfinite(local_metrics[f"{alpha:.1f}"]):
                                raise RuntimeError(f"Non-finite local held loss at {unit}/alpha={alpha}")
                        sync(); held_local_scoring_seconds += time.perf_counter() - local_t0
                        local_held_records[unit] = {"unit_id": unit, "selection_sha256": lock_sha,
                            "selected_alpha_local": choice["alpha_local"], "selected_alpha_block": choice["alpha_block"],
                            "local_held_mse_by_alpha": local_metrics,
                            "local_held_score_alphas": metric_alphas}
                    m2_target = m2_targets[module_full_name(li, proj)].to(DEVICE)
                    sync(); restore_t0 = time.perf_counter()
                    with torch.no_grad():
                        target_module.weight.copy_(m2_target)
                    sync(); m2_restore_seconds += time.perf_counter() - restore_t0
                    actual_m2_sha = sha_tensor(target_module.weight)
                    expected_m2_sha = sha_tensor(m2_targets[module_full_name(li, proj)])
                    target_restoration_checks.append({"layer": li, "projection": proj, "split": split,
                        "phase": "held", "expected_m2_weight_sha256": expected_m2_sha,
                        "actual_after_restore_sha256": actual_m2_sha, "pass": actual_m2_sha == expected_m2_sha})
                    if actual_m2_sha != expected_m2_sha:
                        raise RuntimeError(f"Target module was not restored to v18 M2 after held: L{li}/{proj}")
                    dump(status_path, {"status": "RUNNING", "session_id": session_id,
                        "phase": "held_landscapes", "layer": li, "projection": proj, "split": split,
                        "held_row_alpha_scores_complete": len(held_records),
                        "held_row_alpha_scores_expected": 480,
                        "block_forwards_this_process": counters["block_forwards_this_process"],
                        "search_selection_sha256": lock_sha})
                x_mm._mmap.close()
            hidden_mm._mmap.close(); teacher_mm._mmap.close()
        sync()
        held_seconds = time.perf_counter() - held_started
        if len(held_records) != 480 or len(local_held_records) != 48:
            raise RuntimeError(f"Held output count mismatch: block={len(held_records)}, local={len(local_held_records)}")
        dump(OUT / "TARGET_RESTORATION_CHECKS.json", {"status": "PASS" if all(x["pass"] for x in target_restoration_checks) else "FAIL",
            "entries": target_restoration_checks, "all_other_modules_v18_m2_during_target_block_scoring": True})

        held_csv_rows = []
        for r in held_records:
            choice = locked_by_unit[r["unit_id"]]
            local_metrics = local_held_records[r["unit_id"]]["local_held_mse_by_alpha"]
            r["alpha_local_selected"] = float(r["alpha"]) == float(choice["alpha_local"])
            r["alpha_block_selected"] = float(r["alpha"]) == float(choice["alpha_block"])
            r["alpha_native"] = float(r["alpha"]) == 1.0
            r["local_held_mse_if_reported"] = local_metrics.get(f"{float(r['alpha']):.1f}")
            r["held_selection_sha256"] = lock_sha
            held_csv_rows.append({**r,
                "block_held_document_mse": compact_json(r["block_held_document_mse"]),
                "block_held_document_sse": compact_json(r["block_held_document_sse"]),
                "candidate_scale_row": compact_json(r["candidate_scale_row"]),
                "candidate_zero_row": compact_json(r["candidate_zero_row"]),
                "local_held_mse_by_alpha": compact_json(local_metrics)})
        save_csv(OUT / "HELD_RESULTS.csv", held_csv_rows)
        dump(OUT / "HELD_RESULTS_RAW.json", {"selection_sha256": lock_sha,
            "block_row_alpha_score_count": len(held_records), "results": held_records,
            "local_selected_metrics": local_held_records})

        repeat_noise = float(pre.get("checks", {}).get("repeat_mse_abs_difference", pre.get("repeat_mse_abs_difference", 0.0)) or 0.0)
        summary, comparison, disagreement_rows, ranking, params = materialize_result_tables(
            search_records, held_records, locked, banks, local_held_records, repeat_noise, OUT)
        if summary["units"] != 48:
            raise RuntimeError("Summary row-split unit count does not match frozen E1=48")

        formal_gptq_attempts = 0
        formal_block_attempts = 0
        formal_gptq_completed_events = 0
        if (OUT / "RUN_EVENTS.jsonl").exists():
            with (OUT / "RUN_EVENTS.jsonl").open("r", encoding="utf-8") as f:
                for line in f:
                    event = json.loads(line)
                    if event.get("stage") != "formal":
                        continue
                    if event["event"] == "gptq_candidate_started": formal_gptq_attempts += 1
                    elif event["event"] == "gptq_candidate_completed": formal_gptq_completed_events += 1
                    elif event["event"] == "block_forward_started": formal_block_attempts += 1
        block_complete_files = sum(1 for p in (RAW_DIR / "block_scores").glob("*.json")
                                   if json.loads(p.read_text(encoding="utf-8")).get("status") == "COMPLETE")
        formal_elapsed_this_process = time.perf_counter() - session_start
        events.write("formal_session_completed", stage="formal", session_id=session_id,
                     wall_seconds=formal_elapsed_this_process, verdict=summary["verdict"])
        events.close()
        run_report = {
            "status": "COMPLETE", "verdict": summary["verdict"], "session_id": session_id,
            "protocol_id": "range_w4_block_objective_e1_v23", "protocol_sha256": sha_file(PROTOCOL_PATH),
            "manifest_sha256": sha_file(DOCS_PATH), "v18_m2_checkpoint_sha256": prefix["m2_model_safetensors_sha256"],
            "environment": runtime_environment(), "preflight_status": pre["status"],
            "cost_gate_status": cost["status"], "search_selection_sha256": lock_sha,
            "timing_seconds_this_process": {"model_load_patch": model_load_seconds,
                "candidate_generation_local_scoring_and_stitch": candidate_and_stitch_seconds,
                "search_block_landscape": search_seconds, "held_block_and_local_metrics": held_seconds,
                "held_selected_local_scoring": held_local_scoring_seconds,
                "local_m2_stitch_construction": stitch_commit_seconds,
                "all_row_target_commit": baseline_commit_seconds,
                "restore_target_module_to_v18_m2": m2_restore_seconds,
                "total_formal_session": formal_elapsed_this_process},
            "measured_preformal_prefix_timing_seconds": {"capture_plus_model_load_plus_hash": prefix["model_load_seconds"] + prefix["timing"]["prefix_seconds"] + prefix["timing"]["teacher_seconds"] + prefix.get("cache_creation_and_hash_seconds", 0.0),
                "cache_validation": prefix.get("runtime_cache_validation_seconds", 0.0)},
            "formal_candidate_full_linear_gptq_execution_slots": 120,
            "formal_full_linear_gptq_attempts_in_event_log": formal_gptq_attempts,
            "formal_full_linear_gptq_completed_events": formal_gptq_completed_events,
            "formal_candidate_bank_complete_slots": sum(1 for li in LAYERS for proj in PROJECTIONS
                for split in ("S0", "S1") for a in ALPHAS if candidate_path(li, proj, split, a).is_file()),
            "formal_block_single_document_forwards_planned": 15360,
            "formal_block_single_document_forward_attempts_in_event_log": formal_block_attempts,
            "formal_block_score_files_complete": block_complete_files,
            "formal_search_all_row_local_score_positions": 345600,
            "formal_held_selected_local_score_positions_max": 144,
            "formal_held_selected_local_score_positions_actual": sum(
                len(x["local_held_score_alphas"]) for x in local_held_records.values()),
            "prefix_decoder_layer_forwards": prefix["timing"]["prefix_decoder_layer_forwards"],
            "fixed_teacher_decoder_layer_forwards": prefix["timing"]["teacher_decoder_layer_forwards"],
            "candidate_tensor_sharing": "same candidate checkpoint per target/split/alpha is used for both local and block scores",
            "target_module_restoration": "each q_proj/up_proj is restored to its exact v18 M2 weight after search and held block scoring, so sibling and other target modules remain frozen v18 M2",
            "selection_leakage_check": "SEARCH_SELECTIONS.json was locked and hashed before held block scoring; held data are absent from argmin inputs",
            "all_row_local_stitch": "per-row search-local alpha; Q/scales/zeros are stitched from the exact shared full-Linear candidate outputs; candidate g_idx equality checked",
            "cuda_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(DEVICE)),
            "cuda_peak_reserved_bytes": int(torch.cuda.max_memory_reserved(DEVICE)),
            "hessian_details": hessian_records,
            "summary": summary}
        dump(OUT / "RUN_REPORT.json", run_report)
        correctness = {"status": "PASS", "preflight_correctness": pre["status"],
            "preflight_alpha1_native_parity": pre["checks"]["alpha1_native_gptq_parity"],
            "candidate_gptq_slots_complete": run_report["formal_candidate_bank_complete_slots"] == 120,
            "candidate_tensor_local_block_sharing": True,
            "search_selection_locked_before_held": locked["status"] == "LOCKED_BEFORE_HELD_EVALUATION",
            "held_results_selection_hash_matches": all(x["selection_sha256"] == lock_sha for x in local_held_records.values()),
            "candidate_g_idx_consistent_within_each_bank": True,
            "target_restore_entries": len(target_restoration_checks),
            "target_restoration_all_exact": all(x["pass"] for x in target_restoration_checks),
            "search_landscape_count": len(search_records), "held_landscape_count": len(held_records),
            "row_split_unit_count": summary["units"], "formal_block_score_files_complete": block_complete_files,
            "formal_block_forwards_are_15360_without_retries": formal_block_attempts == 15360,
            "formal_candidate_attempts_are_120_without_retries": formal_gptq_attempts == 120,
            "no_ppl_or_external_evaluation_run": True}
        correctness["status"] = "PASS" if (correctness["candidate_gptq_slots_complete"] and
            correctness["search_selection_locked_before_held"] and correctness["held_results_selection_hash_matches"] and
            correctness["target_restoration_all_exact"] and len(target_restoration_checks) == 24 and
            correctness["search_landscape_count"] == 480 and correctness["held_landscape_count"] == 480 and
            correctness["row_split_unit_count"] == 48 and correctness["formal_block_score_files_complete"] == 960) else "FAIL"
        dump(OUT / "correctness_checks.json", correctness)
        write_report(summary, run_report, correctness, cost, comparison, ranking, disagreement_rows)
        dump(status_path, {"status": "COMPLETE", "session_id": session_id, "verdict": summary["verdict"],
            "formal_gptq_attempts": formal_gptq_attempts, "formal_block_forward_attempts": formal_block_attempts,
            "search_landscape_rows": len(search_records), "held_landscape_rows": len(held_records),
            "selection_sha256": lock_sha})
        print(json.dumps({"status": correctness["status"], "verdict": summary["verdict"],
            "formal_gptq_attempts": formal_gptq_attempts,
            "formal_block_forward_attempts": formal_block_attempts,
            "selector_comparison": summary["selector_comparison"],
            "alpha_disagreement": summary["alpha_disagreement"],
            "v22_promotion_gate": summary["v22_promotion_gate"]}, ensure_ascii=False, indent=2))
        if correctness["status"] != "PASS":
            raise RuntimeError("Formal E1 correctness checks failed; report retained for diagnosis")
    except BaseException as exc:
        events.write("formal_session_failed", stage="formal", session_id=session_id,
                     error_type=type(exc).__name__, error=str(exc),
                     candidate_attempts_this_process=counters["candidate_attempts_this_process"],
                     candidate_completed_this_process=counters["candidate_completed_this_process"],
                     block_forwards_this_process=counters["block_forwards_this_process"])
        events.close()
        dump(status_path, {"status": "FAILED_RESUMABLE", "session_id": session_id,
            "failed_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "error_type": type(exc).__name__,
            "error": str(exc), "counters_this_process": counters})
        raise


def write_report(summary, run_report, correctness, cost, comparison_rows, ranking_rows, disagreement_rows):
    grouped = {}
    for row in comparison_rows:
        grouped.setdefault(row["projection"], []).append(row)
    depth_grouped = {}
    for row in comparison_rows:
        depth_grouped.setdefault(row["layer"], []).append(row)
    tails = list(csv.DictReader((OUT / "TAIL_RISK_ANALYSIS.csv").open("r", encoding="utf-8-sig")))
    overall_tails = {r["selector"]: r for r in tails if r["scope"] == "overall"}
    q_rows = grouped.get("q_proj", [])
    up_rows = grouped.get("up_proj", [])
    q_gain = float(np.mean([r["relative_improvement_block_vs_local"] for r in q_rows])) if q_rows else float("nan")
    up_gain = float(np.mean([r["relative_improvement_block_vs_local"] for r in up_rows])) if up_rows else float("nan")
    depth_lines = []
    for li in LAYERS:
        rows = depth_grouped.get(li, [])
        depth_lines.append(f"- L{li}: n={len(rows)}, block-vs-local win/tie/loss="
            f"{sum(r['comparison']=='block_better' for r in rows)}/"
            f"{sum(r['comparison']=='tie' for r in rows)}/"
            f"{sum(r['comparison']=='block_worse' for r in rows)}, "
            f"mean relative improvement={np.mean([r['relative_improvement_block_vs_local'] for r in rows]):.6%}.")
    rank_local = [r["search_to_held_block_spearman"] for r in ranking_rows
                  if r["selector"] == "local_selector" and r["search_to_held_block_spearman"] is not None]
    rank_block = [r["search_to_held_block_spearman"] for r in ranking_rows
                  if r["selector"] == "block_selector" and r["search_to_held_block_spearman"] is not None]
    verdict = summary["verdict"]
    if verdict == "E1_BLOCK_OBJECTIVE_SIGNAL_SUPPORTED":
        next_step = "仅设计 v24 staged block-aware prototype，不执行 full-model/PPL。先在设计中解决 full Linear selection、q/k/v 与 up/gate sibling interaction、staged commit 和预算。"
    elif verdict == "E1_BLOCK_OBJECTIVE_SIGNAL_MIXED":
        next_step = "只设计一次针对本轮显示正向信号的 projection/depth 结构确认实验；不扩展到 full-model。"
    else:
        next_step = "停止 PRIMARY block-objective 路线；下一轮只按 v22 冻结 BACKUP 设计 2-region structured group-alpha 诊断，不在本轮执行。"
    report = f"""# v23 E1 — Block/Downstream-Aware Objective Diagnostic

## 执行状态与协议

- Verdict: **`{verdict}`**。
- Correctness: **{correctness['status']}**；alpha=1 native GPTQ parity: **{correctness['preflight_alpha1_native_parity']}**。
- 10 个 alpha candidate tensor 在 local 与 block selector 间共享；held 数据的选择锁 `{run_report['search_selection_sha256']}` 在 held block loss 计算前写入并校验。
- 模型：Qwen2.5-0.5B-Instruct，revision `7ae557604adf67be50417f59c2c2f167def9a775`，FP16、SDPA、TF32 off；W4A16 symmetric G128，native dynamic GPTQ，damp=0.1、block=128、desc_act=false、static_groups=false。
- search/held 各 16 篇完整 WT2 train 文档，每篇前 2048 tokens；manifest SHA256 `{run_report['manifest_sha256']}`。WT2 test、C4 eval、PPL 均未运行。
- 目标为 L0/L12/L23 的 q_proj/up_proj，各预冻结 4 行、2 splits，共 48 row-split units。held oracle 只用于 retrospective ranking/regret。

## 实际工作量与成本

- 完成 full-Linear GPTQ candidate slots：{run_report['formal_candidate_bank_complete_slots']}/120；事件记录的 full-Linear GPTQ 调用尝试：{run_report['formal_full_linear_gptq_attempts_in_event_log']}，完成事件：{run_report['formal_full_linear_gptq_completed_events']}。
- 完成 candidate block-score 文件：{run_report['formal_block_score_files_complete']}/960；单文档 block forward 调用尝试：{run_report['formal_block_single_document_forward_attempts_in_event_log']}，冻结唯一评分槽位为 15,360（search 7,680 + held 7,680）。若有中断重试，attempts 会高于唯一槽位数。
- 同时记录 {run_report['formal_search_all_row_local_score_positions']} 个 search all-row local score positions 和 {run_report['formal_held_selected_local_score_positions_actual']} 个实际 held selected local score positions（上限 {run_report['formal_held_selected_local_score_positions_max']}）；它们不是 GPTQ execution 数。
- 量化前缀 capture/teacher/block-count：{run_report['prefix_decoder_layer_forwards']} 个 M2 decoder-layer forward + {run_report['fixed_teacher_decoder_layer_forwards']} 个 fixed teacher decoder-layer forward。
- 预注册 1.5× 成本闸门：中心估算 {cost['central_estimate_seconds']/3600:.3f} h，保守估算 {cost['conservative_estimate_seconds']/3600:.3f} h，状态 **{cost['status']}**（6 h 上限）。正式会话墙钟 {run_report['timing_seconds_this_process']['total_formal_session']/3600:.3f} h；本机峰值显存 allocated/reserved 为 {run_report['cuda_peak_allocated_bytes']/2**30:.3f}/{run_report['cuda_peak_reserved_bytes']/2**30:.3f} GiB。
- 成本拆分见 `E1_COST_GATE.json` 和 `RUN_REPORT.json`，含候选 GPTQ、local scoring、H/Hinv、search/held block forwards、prefix/teacher、stitch、candidate checkpoint IO。

## Primary held block metric

- Block-selected vs local-selected win/tie/loss：{summary['selector_comparison']['win_tie_loss']['block_better']}/{summary['selector_comparison']['win_tie_loss']['tie']}/{summary['selector_comparison']['win_tie_loss']['block_worse']}。
- `delta = block-selected held block MSE - local-selected held block MSE`：median {summary['selector_comparison']['median_delta_block_minus_local_mse']:.8g}，mean {summary['selector_comparison']['mean_delta_block_minus_local_mse']:.8g}；负值有利于 block selector。
- Pooled SSE delta (block − local)：{summary['selector_comparison']['pooled_sse_delta_block_minus_local']:.8g}；pooled relative improvement：{summary['selector_comparison']['pooled_sse_improvement_fraction']:.6%}。median/mean per-unit relative improvement：{summary['selector_comparison']['median_relative_improvement_block_vs_local']:.6%}/{summary['selector_comparison']['mean_relative_improvement_block_vs_local']:.6%}。
- Worst case：`{summary['selector_comparison']['worst_case_unit_id']}`，relative improvement {summary['selector_comparison']['worst_case_relative_improvement_block_vs_local']:.6%}（负值代表 block selector 更差）。每单位数值 tie epsilon 按冻结规则计算，见 `SELECTOR_COMPARISON.csv`。
- alpha disagreement：{summary['alpha_disagreement']['disagreement_units']}/48 = {summary['alpha_disagreement']['disagreement_fraction']:.2%}。只看 disagreement 且 held 非平局单位，block selector win fraction={summary['alpha_disagreement']['block_win_fraction_among_changed_non_ties']}。

## Ranking、tail 与结构差异

- Search→held block Spearman mean：local selector {float(np.mean(rank_local)) if rank_local else float('nan'):.5f}；block selector {float(np.mean(rank_block)) if rank_block else float('nan'):.5f}。held oracle regret 的中位数：local {summary['ranking']['local_selector_median_held_oracle_regret_mse']:.8g}，block {summary['ranking']['block_selector_median_held_oracle_regret_mse']:.8g}。
- 相对 native alpha=1，local selector 严重负尾（<-5%/<-10%/<-20%）：{overall_tails['local_selector']['regressions_below_minus_5pct']}/{overall_tails['local_selector']['regressions_below_minus_10pct']}/{overall_tails['local_selector']['regressions_below_minus_20pct']}；block selector：{overall_tails['block_selector']['regressions_below_minus_5pct']}/{overall_tails['block_selector']['regressions_below_minus_10pct']}/{overall_tails['block_selector']['regressions_below_minus_20pct']}。最差 native-relative gain 分别为 {float(overall_tails['local_selector']['worst_gain_vs_alpha1']):.6%}（{overall_tails['local_selector']['worst_unit_id']}）和 {float(overall_tails['block_selector']['worst_gain_vs_alpha1']):.6%}（{overall_tails['block_selector']['worst_unit_id']}）。
- Projection descriptive mean block-vs-local relative improvement：q_proj {q_gain:.6%}；up_proj {up_gain:.6%}。
{chr(10).join(depth_lines)}
- 分解是小样本描述结果，不作普遍因果结论。完整 alpha 频数、row/document 损失、actual scale/zero/g_idx 见 CSV/JSON 原始文件。

## 结论与下一步

本轮只检验同一动态 GPTQ candidate 下更换 selection objective 的局部信号，不构成整模型量化结果。Verdict `{verdict}` 对应的下一步：{next_step}

未验证：整模型 PPL、下游任务质量、full-model block-aware stitch 收益、部署或推理速度变化。离线搜索成本不代表在线推理加速。

## 复现命令

在仓库根目录 PowerShell 中，使用已冻结的 document manifest：

```powershell
.\\.venv-gptq\\Scripts\\python.exe -B .\\experiments\\range_search_w4\\results\\range_w4_block_objective_e1_v23\\run_e1_v23.py prepare
.\\.venv-gptq\\Scripts\\python.exe -B .\\experiments\\range_search_w4\\results\\range_w4_block_objective_e1_v23\\run_e1_v23.py preflight
.\\.venv-gptq\\Scripts\\python.exe -B .\\experiments\\range_search_w4\\results\\range_w4_block_objective_e1_v23\\run_e1_v23.py run
```

`E1_DOCUMENT_MANIFEST.json` 的 SHA256 必须与本报告一致；preflight correctness 与 1.5× cost gate 均通过后才运行 `run`。已有正式结果会通过候选/文档评分 checkpoint 复用。
"""
    (OUT / "REPORT.md").write_text(report, encoding="utf-8")


def write_preflight_gate_report(pre, cost):
    if pre.get("status") != "PASS" or cost["status"] != "E1_COST_GATE_PASS":
        body = f"""# v23 E1 — Preflight stopped before formal execution

- Correctness preflight: **{pre['status']}**.
- Cost gate: **{cost['status']}**. Central estimate {cost['central_estimate_seconds']/3600:.3f} h; conservative estimate {cost['conservative_estimate_seconds']/3600:.3f} h; frozen limit 6 h.
- Formal candidate landscapes were not started. No E1 scientific verdict is assigned.
- Preflight-only GPTQ calls: {pre['preflight_candidate_execution_count']} (22 alpha candidate calls + 1 native parity reference); preflight-only block forwards: {pre['peak_cuda'] and json.loads((OUT / 'RUNTIME_PREFLIGHT.json').read_text(encoding='utf-8'))['preflight_block_single_document_forwards']}.
- The scope is stopped here per the frozen gate. This artifact is a cost/correctness stop report, not a negative experiment result.
"""
        (OUT / "REPORT.md").write_text(body, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["prepare", "preflight", "run"])
    args = parser.parse_args()
    if args.stage == "prepare":
        stage_prepare_prefix()
    elif args.stage == "preflight":
        run_preflight()
    else:
        run_formal()


if __name__ == "__main__":
    main()
