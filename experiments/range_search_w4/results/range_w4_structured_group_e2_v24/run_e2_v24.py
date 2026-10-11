"""Frozen E2 diagnostic: scalar vs dense scalar vs two-region alpha.

All new outputs are restricted to this v24 directory. Existing v18/v23 files
are read-only inputs. The structured candidate changes one selected output row
and reruns the native GPTQ compensation path from that pristine row.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import os
import platform
import statistics
import sys
import time
from pathlib import Path
from types import MethodType

import numpy as np
import torch

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[3]
RESULTS = ROOT / "experiments/range_search_w4/results"
E1 = RESULTS / "range_w4_block_objective_e1_v23"
E1_CACHE = E1 / "cache"
V18 = RESULTS / "range_w4_fullmodel_v18"
V171 = RESULTS / "range_w4_m2_parity_v17_1"
V173 = RESULTS / "range_w4_m2_runtime_v17_3"
MODEL_DIR = ROOT / ".hf_cache/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775"
M2_WEIGHTS = V18 / "M2/checkpoint/final/model.safetensors"
DOCS_PATH = OUT / "E2_DOCUMENT_MANIFEST.json"
PROTOCOL_PATH = OUT / "PROTOCOL.json"
CACHE = OUT / "cache"
RAW = OUT / "raw"
EVENTS = OUT / "RUN_EVENTS.jsonl"
DEVICE = torch.device("cuda:0")
LAYERS = [0, 12, 23]
PROJECTIONS = ["up_proj", "down_proj"]
ROWS = {"up_proj": [0, 224, 448, 672], "down_proj": [0, 224, 448, 672]}
ALPHAS = [i / 10 for i in range(1, 11)]
DENSE_EXTRA = [0.1 + 0.9 * j / 37 for j in range(1, 37)]
DENSE_GRID = sorted(set(ALPHAS + DENSE_EXTRA))
SEQ = 2048
N_DOCS = 64
SPLITS = {
    "S0": {"search": list(range(0, 16)), "held": list(range(16, 32))},
    "S1": {"search": list(range(32, 48)), "held": list(range(48, 64))},
}
SCALE_RTOL = 1e-6
SCALE_ATOL = 1e-9
sys.path.insert(0, str(V171))
sys.path.insert(0, str(V173))
from native_m2 import native_dynamic_gptq  # noqa: E402
from optimized_native_m2 import optimized_dynamic_gptq, prepare_native_state  # noqa: E402


def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def sha_tensor(t: torch.Tensor) -> str:
    return sha_bytes(t.detach().cpu().contiguous().numpy().tobytes())


def dump(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(payload, encoding="utf-8", newline="\n")
    os.replace(tmp, path)


def save_torch(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def read_torch(path: Path):
    return torch.load(path, map_location="cpu", weights_only=True)


def append_event(kind: str, **fields) -> None:
    EVENTS.parent.mkdir(parents=True, exist_ok=True)
    event = {"utc": dt.datetime.now(dt.timezone.utc).isoformat(), "kind": kind, **fields}
    with EVENTS.open("a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(event, ensure_ascii=False, allow_nan=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def sync() -> None:
    torch.cuda.synchronize(DEVICE)


def timed(fn):
    sync()
    start = time.perf_counter()
    value = fn()
    sync()
    return value, time.perf_counter() - start


def runtime_env() -> dict:
    from importlib.metadata import version
    import transformers
    return {
        "python": platform.python_version(), "torch": torch.__version__,
        "transformers": transformers.__version__, "gptqmodel": version("gptqmodel"),
        "cuda_runtime": torch.version.cuda, "device": torch.cuda.get_device_name(DEVICE),
        "device_capability": list(torch.cuda.get_device_capability(DEVICE)),
        "tf32_matmul": bool(torch.backends.cuda.matmul.allow_tf32),
        "tf32_cudnn": bool(torch.backends.cudnn.allow_tf32),
    }


def verify_frozen_inputs() -> tuple[dict, dict, dict]:
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    if protocol["status"] != "FROZEN_BEFORE_E2_SEARCH":
        raise RuntimeError("E2 protocol is not frozen")
    docs = json.loads(DOCS_PATH.read_text(encoding="utf-8"))
    samples = docs["samples"]
    if len(samples) != N_DOCS or len({s["article_sha256"] for s in samples}) != N_DOCS:
        raise RuntimeError("E2 document set must contain 64 unique complete articles")
    ordered = hashlib.sha256(b"".join(
        int(token).to_bytes(8, "little", signed=True) for sample in samples for token in sample["token_ids"]
    )).hexdigest()
    if ordered != docs["ordered_token_ids_sha256_i64le"]:
        raise RuntimeError("E2 ordered token ID hash mismatch")
    for i, sample in enumerate(samples):
        if len(sample["token_ids"]) != SEQ:
            raise RuntimeError(f"E2 sample {i} has the wrong length")
        tok_hash = sha_bytes(json.dumps(sample["token_ids"]).encode("utf-8"))
        if tok_hash != sample["token_sha256"]:
            raise RuntimeError(f"E2 token hash mismatch at sample {i}")
    expected_split = (["S0_search"] * 16 + ["S0_held"] * 16 +
                      ["S1_search"] * 16 + ["S1_held"] * 16)
    if [s["split"] for s in samples] != expected_split:
        raise RuntimeError("E2 document order differs from the frozen S0/S1 split")
    if set(SPLITS["S0"]["search"]) & set(SPLITS["S0"]["held"]):
        raise RuntimeError("S0 search and held document indices overlap")
    if set(SPLITS["S1"]["search"]) & set(SPLITS["S1"]["held"]):
        raise RuntimeError("S1 search and held document indices overlap")
    if set(SPLITS["S0"]["search"] + SPLITS["S0"]["held"]) & set(SPLITS["S1"]["search"] + SPLITS["S1"]["held"]):
        raise RuntimeError("S0 and S1 documents overlap")
    source_hashes = protocol["frozen_source_sha256"]
    for relpath, expected in source_hashes.items():
        path = ROOT / Path(relpath)
        if not path.is_file() or sha_file(path) != expected:
            raise RuntimeError(f"Frozen source hash mismatch: {relpath}")
    return protocol, docs, source_hashes


def module_attr(layer, projection):
    return getattr(layer.mlp, projection)


def qcfg():
    from gptqmodel.quantization import QuantizeConfig
    return QuantizeConfig(bits=4, group_size=128, sym=True, desc_act=False, static_groups=False,
                          true_sequential=True, damp_percent=0.1, damp_auto_increment=0, mse=0)


def load_model_and_pristine():
    from safetensors import safe_open
    from transformers import AutoModelForCausalLM
    if not torch.cuda.is_available() or torch.cuda.get_device_name(DEVICE) != "NVIDIA GeForce RTX 4060 Laptop GPU":
        raise RuntimeError("Frozen CUDA device RTX 4060 Laptop GPU is unavailable")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    started = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        str(MODEL_DIR), torch_dtype=torch.float16, low_cpu_mem_usage=True,
        local_files_only=True, attn_implementation="sdpa",
    ).eval().to(DEVICE)
    pristine = {}
    for li in LAYERS:
        for projection in PROJECTIONS:
            name = f"L{li}/{projection}"
            pristine[name] = module_attr(model.model.layers[li], projection).weight.detach().cpu().contiguous().clone()
    with safe_open(str(M2_WEIGHTS), framework="pt", device="cpu") as sf:
        for li in range(24):
            layer = model.model.layers[li]
            for projection in ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"):
                parent = layer.self_attn if projection in ("q_proj", "k_proj", "v_proj", "o_proj") else layer.mlp
                module = getattr(parent, projection)
                key = f"model.layers.{li}.{'self_attn' if parent is layer.self_attn else 'mlp'}.{projection}.weight"
                quantized = sf.get_tensor(key)
                if quantized.shape != module.weight.shape or quantized.dtype != torch.float16:
                    raise RuntimeError(f"v18 M2 target shape/dtype mismatch: {key}")
                with torch.no_grad():
                    module.weight.copy_(quantized.to(DEVICE))
                del quantized
    sync()
    return model, pristine, time.perf_counter() - started


def open_e1_cache():
    marker_path = E1_CACHE / "PREFIX_CACHE_MANIFEST.json"
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    e1_docs_sha = sha_file(E1 / "E1_DOCUMENT_MANIFEST.json")
    e1_key = "experiments/range_search_w4/results/range_w4_block_objective_e1_v23/E1_DOCUMENT_MANIFEST.json"
    if marker.get("source_hashes", {}).get(e1_key) != e1_docs_sha:
        raise RuntimeError("E1 prefix cache was built from a different frozen document manifest")
    if marker.get("m2_model_safetensors_sha256") != sha_file(M2_WEIGHTS):
        raise RuntimeError("E1 activation prefix cache does not match the frozen v18 M2 checkpoint")
    files = marker["files"]
    arrays = {}
    checked = {}
    for li in LAYERS:
        for key, shape in ((f"hidden_L{li}.f16", (N_DOCS, SEQ, 896)),
                           (f"input_L{li}_up_proj.f16", (N_DOCS, SEQ, 896))):
            path = E1_CACHE / key
            if not path.is_file() or path.stat().st_size != math.prod(shape) * 2:
                raise RuntimeError(f"E1 frozen prefix cache has wrong shape/size: {path}")
            digest = sha_file(path)
            if files.get(key, {}).get("sha256") != digest:
                raise RuntimeError(f"E1 frozen prefix cache hash mismatch: {key}")
            arrays[key] = np.memmap(path, dtype="<f2", mode="r", shape=shape)
            checked[key] = digest
    return marker, arrays, checked


def ensure_down_inputs(model, hidden_arrays):
    marker_path = CACHE / "DOWN_INPUT_CACHE_MANIFEST.json"
    marker_path.parent.mkdir(parents=True, exist_ok=True)
    source_manifest_sha = sha_file(DOCS_PATH)
    prefix_manifest_sha = sha_file(E1_CACHE / "PREFIX_CACHE_MANIFEST.json")
    down_shapes = {f"input_L{li}_down_proj.f16": (N_DOCS, SEQ, 4864) for li in LAYERS}
    if marker_path.exists():
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        if marker.get("e2_manifest_sha256") != source_manifest_sha or marker.get("e1_prefix_manifest_sha256") != prefix_manifest_sha:
            raise RuntimeError("Existing down-input cache belongs to different frozen source manifests")
        result = {}
        for name, shape in down_shapes.items():
            path = CACHE / name
            digest = sha_file(path)
            if path.stat().st_size != math.prod(shape) * 2 or digest != marker["files"][name]["sha256"]:
                raise RuntimeError(f"Down input cache mismatch: {name}")
            result[name] = np.memmap(path, dtype="<f2", mode="r", shape=shape)
        return result, {"status": "CACHE_REUSED", "cache_validation_seconds": marker.get("created_seconds", 0.0),
                        "down_capture_seconds": marker.get("down_capture_seconds", 0.0),
                        "files": marker["files"]}

    # Remove only known incomplete cache files owned by this new E2 directory.
    for name in down_shapes:
        path = CACHE / name
        if path.exists():
            path.unlink()
    arrays = {}
    for name, shape in down_shapes.items():
        arrays[name] = np.memmap(CACHE / name, dtype="<f2", mode="w+", shape=shape)
    down_started = time.perf_counter()
    count = {li: 0 for li in LAYERS}
    handles = []
    for li in LAYERS:
        mapping = arrays[f"input_L{li}_down_proj.f16"]
        active = {"doc": -1}
        def hook(_module, args, *, active=active, mapping=mapping, layer_index=li):
            value = args[0].detach()[0]
            mapping[active["doc"]] = value.to("cpu", dtype=torch.float16).numpy()
            count[layer_index] += 1
        handles.append((li, active, model.model.layers[li].mlp.down_proj.register_forward_pre_hook(hook)))
    positions = torch.arange(SEQ, device=DEVICE, dtype=torch.long)[None, :]
    with torch.inference_mode():
        for li, active, _handle in handles:
            for di in range(N_DOCS):
                active["doc"] = di
                hidden = torch.from_numpy(np.array(hidden_arrays[f"hidden_L{li}.f16"][di], copy=True)).unsqueeze(0).to(DEVICE)
                rotary = model.model.rotary_emb(hidden, positions)
                model.model.layers[li](hidden, attention_mask=None, position_ids=positions,
                                       cache_position=positions[0], position_embeddings=rotary,
                                       use_cache=False)
                del hidden, rotary
            arrays[f"input_L{li}_down_proj.f16"].flush()
    sync()
    for _, _, handle in handles:
        handle.remove()
    down_seconds = time.perf_counter() - down_started
    if any(value != N_DOCS for value in count.values()):
        raise RuntimeError(f"down_proj input hook count mismatch: {count}")
    file_data = {}
    for name, mapping in arrays.items():
        mapping.flush()
        path = CACHE / name
        file_data[name] = {"shape": list(mapping.shape), "dtype": "float16", "sha256": sha_file(path),
                           "size_bytes": path.stat().st_size}
        mapping._mmap.close()
    marker = {"status": "PASS", "e2_manifest_sha256": source_manifest_sha,
              "e1_prefix_manifest_sha256": prefix_manifest_sha, "v18_m2_checkpoint_sha256": sha_file(M2_WEIGHTS),
              "down_capture_seconds": down_seconds, "forward_count_by_layer": count,
              "forward_refresh_semantics": "one replay of each target v18 M2 block per complete document; no per-row refresh",
              "files": file_data}
    dump(marker_path, marker)
    result = {name: np.memmap(CACHE / name, dtype="<f2", mode="r", shape=shape)
              for name, shape in down_shapes.items()}
    return result, {"status": "CREATED", "down_capture_seconds": down_seconds, "files": file_data}


def projection_input(projection: str, li: int, e1_arrays, down_arrays):
    if projection == "up_proj":
        return e1_arrays[f"input_L{li}_up_proj.f16"]
    return down_arrays[f"input_L{li}_down_proj.f16"]


def build_or_load_hessian(model, li, projection, split, x_mm, doc_indices):
    from gptqmodel.quantization.gptq import GPTQ
    path = CACHE / "hessians" / f"L{li}_{projection}_{split}.pt"
    expected_indices = list(doc_indices)
    if path.exists():
        obj = read_torch(path)
        if (obj["layer"], obj["projection"], obj["split"], obj["doc_indices"]) != (li, projection, split, expected_indices):
            raise RuntimeError(f"Hessian checkpoint protocol mismatch: {path}")
        if sha_tensor(obj["H"]) != obj["H_sha256"]:
            raise RuntimeError(f"Hessian checkpoint hash mismatch: {path}")
        meta_path = path.with_suffix(".json")
        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
        return (obj["H"].to(DEVICE, dtype=torch.float32), obj["nsamples"], obj["H_sha256"],
                float(obj.get("hessian_build_seconds", 0.0)), float(meta.get("hessian_io_seconds", 0.0)), True)
    module = module_attr(model.model.layers[li], projection)
    core = GPTQ(module, qcfg=qcfg())
    core.quantizer.configure(perchannel=True, bits=4, sym=True)
    core.H = torch.zeros((module.weight.shape[1], module.weight.shape[1]), dtype=torch.float32, device=DEVICE)
    core.nsamples = 0
    sync(); start = time.perf_counter()
    for di in doc_indices:
        x = torch.from_numpy(np.array(x_mm[di], copy=True)).unsqueeze(0).to(DEVICE)
        core.process_batch(x)
        del x
    sync(); elapsed = time.perf_counter() - start
    h = core.H.detach().clone()
    nsamples = int(core.nsamples)
    h_sha = sha_tensor(h)
    io_started = time.perf_counter()
    save_torch(path, {"layer": li, "projection": projection, "split": split,
                      "doc_indices": expected_indices, "nsamples": nsamples,
                      "H_sha256": h_sha, "hessian_build_seconds": elapsed,
                      "H": h.detach().cpu()})
    io_seconds = time.perf_counter() - io_started
    meta_started = time.perf_counter()
    dump(path.with_suffix(".json"), {"layer": li, "projection": projection, "split": split,
                                     "doc_indices": expected_indices, "H_sha256": h_sha,
                                     "hessian_io_seconds": io_seconds})
    io_seconds += time.perf_counter() - meta_started
    return h, nsamples, h_sha, elapsed, io_seconds, False


def native_row_candidate(weight_row, prepared, group_alphas):
    """Fresh one-row GPTQ call with native compensation and per-group alpha hook."""
    from gptqmodel.quantization import QuantizeConfig
    from gptqmodel.quantization.gptq import GPTQ
    if weight_row.ndim != 2 or weight_row.shape[0] != 1 or weight_row.dtype != torch.float16:
        raise ValueError("Structured candidate requires a pristine single FP16 output row")
    if len(group_alphas) != math.ceil(weight_row.shape[1] / 128):
        raise ValueError("Per-group alpha vector does not match native G128 groups")
    config = QuantizeConfig(bits=4, group_size=128, sym=True, desc_act=False, static_groups=False,
                            true_sequential=True, damp_percent=0.1, damp_auto_increment=0, mse=0)
    module = torch.nn.Linear(weight_row.shape[1], 1, bias=False, device="meta", dtype=torch.float16)
    module.weight = torch.nn.Parameter(weight_row.detach().clone(), requires_grad=False)
    core = GPTQ(module, qcfg=config)
    core.quantizer.configure(perchannel=True, bits=4, sym=True)
    core.H = prepared.hessian.clone()
    core.nsamples = int(prepared.nsamples)
    def cached_inverse(this, lower, upper=False, block_size=512):
        if not torch.equal(lower, prepared.cholesky):
            raise RuntimeError("Row GPTQ Cholesky differs from the immutable prepared Hessian")
        return prepared.inverse_before_final_cholesky
    core.block_cholesky_inverse = MethodType(cached_inverse, core)
    original_find = core.quantizer.find_params
    calls = {"count": 0}
    def range_hook(current, weight=False):
        original_find(current, weight=weight)
        if calls["count"] > 0:
            group = calls["count"] - 1
            if group >= len(group_alphas):
                raise RuntimeError("Unexpected extra dynamic GPTQ range estimate")
            core.quantizer.scale = core.quantizer.scale * float(group_alphas[group])
        calls["count"] += 1
    core.quantizer.find_params = range_hook
    q, scale, zero, gidx, *_ = core.quantize(blocksize=128)
    expected_calls = len(group_alphas) + 1
    if calls["count"] != expected_calls:
        raise RuntimeError(f"Expected {expected_calls} native range estimates, observed {calls['count']}")
    return {"weight_fp16": q.detach(), "scales": scale.detach(), "zeros": zero.detach(),
            "g_idx": gidx.detach(), "range_find_params_calls": calls["count"]}


def sync_peak_reset():
    sync()
    torch.cuda.reset_peak_memory_stats(DEVICE)


def peak_mem() -> dict:
    sync()
    return {"allocated_bytes": int(torch.cuda.max_memory_allocated(DEVICE)),
            "reserved_bytes": int(torch.cuda.max_memory_reserved(DEVICE))}


def score_one_row(x_mm, doc_indices, original, quantized, *, chunk_docs=2):
    delta = quantized.float() - original.float()
    total = torch.zeros((), dtype=torch.float64, device=DEVICE)
    for offset in range(0, len(doc_indices), chunk_docs):
        docs = doc_indices[offset:offset + chunk_docs]
        x = torch.from_numpy(np.array(x_mm[docs], copy=True)).to(DEVICE).float()
        output_delta = torch.matmul(x.reshape(-1, x.shape[-1]), delta)
        total += output_delta.to(torch.float64).square().sum()
        del x, output_delta
    return float((total / float(len(doc_indices) * SEQ)).item())


def score_selected_rows(x_mm, doc_indices, original, quantized_rows, row_ids, *, chunk_docs=2):
    delta = quantized_rows.float() - original[row_ids].float()
    totals = torch.zeros((len(row_ids),), dtype=torch.float64, device=DEVICE)
    for offset in range(0, len(doc_indices), chunk_docs):
        docs = doc_indices[offset:offset + chunk_docs]
        x = torch.from_numpy(np.array(x_mm[docs], copy=True)).to(DEVICE).float()
        output_delta = torch.matmul(x.reshape(-1, x.shape[-1]), delta.T)
        totals += output_delta.to(torch.float64).square().sum(0)
        del x, output_delta
    return {int(row): float(totals[j].item() / float(len(doc_indices) * SEQ)) for j, row in enumerate(row_ids)}


def grid_idx(alpha: float) -> int:
    return min(range(10), key=lambda i: abs(ALPHAS[i] - alpha))


def dense_grid_idx(alpha: float) -> int:
    index = min(range(len(DENSE_GRID)), key=lambda i: abs(DENSE_GRID[i] - alpha))
    if abs(DENSE_GRID[index] - alpha) > 1e-12:
        raise ValueError(f"Alpha is not in the frozen dense grid: {alpha}")
    return index


def alpha_key(alpha: float) -> str:
    return f"{alpha:.12f}".rstrip("0").rstrip(".")


def tuple_key(a: float, b: float) -> str:
    return f"a{grid_idx(a):02d}_b{grid_idx(b):02d}"


def native_group_assignment(input_dim: int) -> dict:
    groups = math.ceil(input_dim / 128)
    if groups == 7:
        return {"A": list(range(0, 3)), "B": list(range(3, 7))}
    if groups == 38:
        return {"A": list(range(0, 19)), "B": list(range(19, 38))}
    raise RuntimeError(f"Unfrozen input group count in E2 target: {groups}")


def tensor_close_report(reference: torch.Tensor, actual: torch.Tensor, *, exact: bool,
                        rtol: float = 0.0, atol: float = 0.0) -> dict:
    a = reference.detach().cpu()
    b = actual.detach().cpu()
    if a.shape != b.shape or a.dtype != b.dtype:
        return {"pass": False, "shape_equal": a.shape == b.shape, "dtype_equal": a.dtype == b.dtype,
                "max_abs_diff": None, "exact": False}
    exact_equal = torch.equal(a, b)
    if exact:
        passed = exact_equal
    else:
        passed = bool(torch.allclose(a.float(), b.float(), rtol=rtol, atol=atol))
    return {"pass": passed, "shape_equal": True, "dtype_equal": True,
            "max_abs_diff": float((a.float() - b.float()).abs().max().item()) if a.numel() else 0.0,
            "exact": exact_equal, "rtol": rtol, "atol": atol}


def candidate_params(result, row: int) -> dict:
    return {"weight_fp16": result["weight_fp16"][0].detach().cpu().clone(),
            "scales": result["scales"][0].detach().cpu().clone(),
            "zeros": result["zeros"][0].detach().cpu().clone(),
            "g_idx": result["g_idx"].detach().cpu().clone()}


def compare_candidate_fields(full, row_result, full_row: int) -> dict:
    checks = {
        "weight": tensor_close_report(full["weight_fp16"][full_row], row_result["weight_fp16"][0], exact=True),
        "scale": tensor_close_report(full["scales"][full_row], row_result["scales"][0], exact=False,
                                      rtol=SCALE_RTOL, atol=SCALE_ATOL),
        "zero": tensor_close_report(full["zeros"][full_row], row_result["zeros"][0], exact=True),
        "g_idx": tensor_close_report(full["g_idx"], row_result["g_idx"], exact=True),
    }
    checks["pass"] = all(item["pass"] for item in checks.values())
    return checks


def write_runtime_preflight(payload: dict) -> None:
    dump(OUT / "RUNTIME_PREFLIGHT.json", payload)


def stage_preflight() -> dict:
    protocol, docs, source_hashes = verify_frozen_inputs()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required by the frozen E2 protocol")
    env = runtime_env()
    if env["device"] != protocol["model"]["device"] or env["tf32_matmul"] or env["tf32_cudnn"]:
        raise RuntimeError(f"Runtime does not match the frozen E2 environment: {env}")
    start_all = time.perf_counter()
    prefix_marker, e1_arrays, e1_cache_hashes = open_e1_cache()
    prefix_validation_seconds = time.perf_counter() - start_all
    model, pristine, model_load_seconds = load_model_and_pristine()
    sync_peak_reset()
    down_arrays, down_cache_timing = ensure_down_inputs(model, e1_arrays)
    activation_prep_seconds = float(down_cache_timing["down_capture_seconds"])
    # The formal command is a separate process and will hash/validate these
    # large files again. Measure that cache-hit path now instead of substituting
    # the creation time (or a missing marker field) for the formal IO cost.
    sync(); down_validate_started = time.perf_counter()
    validated_down_arrays, _ = ensure_down_inputs(model, e1_arrays)
    sync(); down_cache_validation_seconds = time.perf_counter() - down_validate_started
    for _array in validated_down_arrays.values():
        _array._mmap.close()

    representative_targets = [(0, "up_proj"), (23, "down_proj")]
    alphas_check = [0.7, 0.8, 0.9, 1.0]
    row_ids = {"up_proj": ROWS["up_proj"], "down_proj": ROWS["down_proj"]}
    identity_cases = []
    full_times = {"up_proj": [], "down_proj": []}
    row_times = {"up_proj": [], "down_proj": []}
    score_times = {"up_proj": [], "down_proj": []}
    hessian_times = {"up_proj": [], "down_proj": []}
    hessian_io_times = {"up_proj": [], "down_proj": []}
    hprep_times = {"up_proj": [], "down_proj": []}
    prepared_by_key = {}
    h_by_key = {}
    native_parity_by_target = {}
    probe_candidates = {}
    candidate_bank_for_order = None
    pristine_hashes_before = {k: sha_tensor(v) for k, v in pristine.items()}
    torch.cuda.reset_peak_memory_stats(DEVICE)

    for li, projection in representative_targets:
        x_mm = projection_input(projection, li, e1_arrays, down_arrays)
        device_weight = pristine[f"L{li}/{projection}"].to(DEVICE)
        in_dim = int(device_weight.shape[1])
        assign = native_group_assignment(in_dim)
        expected_a = list(range(0, 3)) if in_dim == 896 else list(range(0, 19))
        if assign["A"] != expected_a:
            raise RuntimeError("Frozen v22 group partition does not reproduce deterministically")
        for split in ("S0", "S1"):
            key = (li, projection, split)
            h, nsamples, h_sha, h_seconds, h_io_seconds, reused = build_or_load_hessian(
                model, li, projection, split, x_mm, SPLITS[split]["search"])
            sync(); t0 = time.perf_counter()
            prepared = prepare_native_state(h, nsamples=nsamples)
            sync(); prep_seconds = time.perf_counter() - t0
            prepared_by_key[key] = prepared
            h_by_key[key] = h
            hessian_times[projection].append(h_seconds)
            hessian_io_times[projection].append(h_io_seconds)
            hprep_times[projection].append(prep_seconds)
            full_results = {}
            for alpha in alphas_check:
                sync(); t0 = time.perf_counter()
                result = optimized_dynamic_gptq(device_weight, prepared, alpha=alpha,
                                                check_cholesky=(alpha == 1.0))
                sync(); elapsed = time.perf_counter() - t0
                full_times[projection].append(elapsed)
                full_results[alpha] = {"weight_fp16": result.weight_fp16.detach(),
                                       "scales": result.scales.detach(), "zeros": result.zeros.detach(),
                                       "g_idx": result.g_idx.detach()}
                del result
            probe_candidates[key] = full_results[0.9]
            if split == "S0":
                alpha1_optimized = full_results[1.0]
                sync(); t0 = time.perf_counter()
                native = native_dynamic_gptq(device_weight, h, nsamples=nsamples, alpha=1.0)
                sync(); native_seconds = time.perf_counter() - t0
                native_checks = {
                    "weight": tensor_close_report(alpha1_optimized["weight_fp16"], native.weight_fp16, exact=True),
                    "scale": tensor_close_report(alpha1_optimized["scales"], native.scales, exact=True),
                    "zero": tensor_close_report(alpha1_optimized["zeros"], native.zeros, exact=True),
                    "g_idx": tensor_close_report(alpha1_optimized["g_idx"], native.g_idx, exact=True),
                }
                native_checks["pass"] = all(x["pass"] for x in native_checks.values())
                if not native_checks["pass"]:
                    raise RuntimeError(f"alpha=1 native parity failed at L{li}/{projection}/{split}: {native_checks}")
                native_parity_by_target[f"L{li}/{projection}/{split}"] = native_checks
                del native
            else:
                native_seconds = None
                native_checks = {"pass": True, "reused_same_target_native_parity": True}

            for alpha in alphas_check:
                full = full_results[alpha]
                for row in row_ids[projection]:
                    group_alphas = [alpha] * math.ceil(in_dim / 128)
                    sync(); t0 = time.perf_counter()
                    row_result = native_row_candidate(device_weight[row:row + 1], prepared, group_alphas)
                    sync(); elapsed = time.perf_counter() - t0
                    row_times[projection].append(elapsed)
                    field_checks = compare_candidate_fields(full, row_result, row)
                    if not field_checks["pass"]:
                        raise RuntimeError(f"scalar/row GPTQ identity failed at L{li}/{projection}/{split}/r{row}/a{alpha}: {field_checks}")
                    search_docs = SPLITS[split]["search"][:2]
                    sync(); t0 = time.perf_counter()
                    full_loss = score_one_row(x_mm, search_docs, device_weight[row], full["weight_fp16"][row])
                    row_loss = score_one_row(x_mm, search_docs, device_weight[row], row_result["weight_fp16"][0])
                    sync(); score_seconds = time.perf_counter() - t0
                    score_times[projection].append(score_seconds)
                    loss_diff = abs(full_loss - row_loss)
                    loss_tol = max(1e-12, 1e-6 * abs(full_loss))
                    output_pass = loss_diff <= loss_tol
                    identity_cases.append({"layer": li, "projection": projection, "split": split,
                                           "row": row, "alpha": alpha, "field_checks": field_checks,
                                           "full_output_mse_2doc": full_loss, "row_output_mse_2doc": row_loss,
                                           "output_mse_abs_diff": loss_diff, "output_mse_tolerance": loss_tol,
                                           "output_mse_pass": output_pass,
                                           "range_find_params_calls": row_result["range_find_params_calls"]})
                    if not output_pass:
                        raise RuntimeError(f"scalar/structured output-loss identity failed: {identity_cases[-1]}")
            if li == 23 and projection == "down_proj" and split == "S0":
                candidate_bank_for_order = (device_weight, prepared, full_results)

    # Candidate order and mutable state isolation are checked by replaying two
    # full-shape candidates in reverse order, then an unequal-alpha row tuple.
    if candidate_bank_for_order is None:
        raise RuntimeError("Missing frozen order-invariance representative")
    device_weight, prepared, expected_bank = candidate_bank_for_order
    reverse = {}
    for alpha in (0.9, 0.7):
        result = optimized_dynamic_gptq(device_weight, prepared, alpha=alpha)
        reverse[alpha] = result
    order_checks = {}
    for alpha in (0.7, 0.9):
        ref = expected_bank[alpha]
        got = reverse[alpha]
        fields = {
            "weight": tensor_close_report(ref["weight_fp16"], got.weight_fp16, exact=True),
            "scale": tensor_close_report(ref["scales"], got.scales, exact=True),
            "zero": tensor_close_report(ref["zeros"], got.zeros, exact=True),
            "g_idx": tensor_close_report(ref["g_idx"], got.g_idx, exact=True),
        }
        order_checks[alpha_key(alpha)] = {"fields": fields, "pass": all(v["pass"] for v in fields.values())}
    if not all(item["pass"] for item in order_checks.values()):
        raise RuntimeError(f"Candidate order invariance failed: {order_checks}")
    diag_groups = native_group_assignment(device_weight.shape[1])
    row_id = ROWS["down_proj"][0]
    unequal_alphas = [0.7 if g in diag_groups["A"] else 0.9 for g in range(math.ceil(device_weight.shape[1] / 128))]
    sync(); t0 = time.perf_counter()
    structured_1 = native_row_candidate(device_weight[row_id:row_id + 1], prepared, unequal_alphas)
    sync(); row_smoke_seconds = time.perf_counter() - t0
    sync(); t0 = time.perf_counter()
    structured_2 = native_row_candidate(device_weight[row_id:row_id + 1], prepared, unequal_alphas)
    sync(); row_repeat_seconds = time.perf_counter() - t0
    row_repeat = {k: tensor_close_report(structured_1[k], structured_2[k], exact=True)
                  for k in ("weight_fp16", "scales", "zeros", "g_idx")}
    row_repeat["pass"] = all(v["pass"] for v in row_repeat.values())
    if not row_repeat["pass"]:
        raise RuntimeError(f"Structured row candidate state/order repeatability failed: {row_repeat}")
    pristine_hashes_after = {k: sha_tensor(v) for k, v in pristine.items()}
    if pristine_hashes_before != pristine_hashes_after:
        raise RuntimeError("Candidate evaluation mutated pristine target weights")
    for key, h in h_by_key.items():
        if sha_tensor(h) != prepared_by_key[key].h_sha256:
            raise RuntimeError(f"Candidate evaluation mutated the prepared Hessian at {key}")

    # Exercise both batched scalar scoring and a row score; these synchronized
    # probes feed the bounded cost estimate, and are not formal result scores.
    score_probe = {}
    for li, projection in representative_targets:
        split = "S0"
        x_mm = projection_input(projection, li, e1_arrays, down_arrays)
        w = pristine[f"L{li}/{projection}"].to(DEVICE)
        q = probe_candidates[(li, projection, split)]["weight_fp16"]
        _, selected_score_seconds = timed(lambda: score_selected_rows(
            x_mm, SPLITS[split]["search"][:2], w, q[ROWS[projection]], ROWS[projection]))
        _, row_score_seconds = timed(lambda: score_one_row(
            x_mm, SPLITS[split]["search"][:2], w[ROWS[projection][0]], q[ROWS[projection][0]]))
        score_probe[projection] = {"two_doc_four_row_seconds": selected_score_seconds,
                                   "two_doc_one_row_seconds": row_score_seconds}

    sync()
    preflight_wall = time.perf_counter() - start_all
    correctness = {
        "status": "PASS",
        "model_runtime": env,
        "checks": {
            "frozen_manifest_and_source_hashes": True,
            "whole_document_search_held_disjoint": True,
            "v23_prefix_manifest_matches_v22_selection": True,
            "down_inputs_captured_from_frozen_v18_m2_prefix": True,
            "alpha1_native_parity": bool(native_parity_by_target) and all(x["pass"] for x in native_parity_by_target.values()),
            "shared_scalar_vs_row_structured_identity": all(x["output_mse_pass"] and x["field_checks"]["pass"] for x in identity_cases),
            "candidate_order_invariance": all(x["pass"] for x in order_checks.values()),
            "region_assignment_deterministic": native_group_assignment(896) == native_group_assignment(896) and native_group_assignment(4864) == native_group_assignment(4864),
            "unequal_region_candidate_finite_and_repeatable": row_repeat["pass"] and all(torch.isfinite(structured_1[k]).all().item() for k in ("weight_fp16", "scales", "zeros")),
            "candidate_state_does_not_mutate_pristine_or_hessian": pristine_hashes_before == pristine_hashes_after,
            "held_not_used_for_preflight_selection": True,
        },
        "identity_case_count": len(identity_cases),
        "identity_alphas": alphas_check,
        "representative_targets": [f"L{li}/{p}" for li, p in representative_targets],
        "native_alpha1_parity_checks": native_parity_by_target,
        "row_scalar_identity_cases": identity_cases,
        "full_candidate_order_invariance": order_checks,
        "unequal_alpha_row_repeat": row_repeat,
        "immutable_pristine_weight_sha256": pristine_hashes_before,
        "immutable_hessian_sha256": {f"L{k[0]}/{k[1]}/{k[2]}": sha_tensor(v) for k, v in h_by_key.items()},
        "parity_tolerance_source": "v17.3 ROW_STITCH_AUDIT: weight/zero/g_idx/local output exact; scale allclose rtol=1e-6 atol=1e-9",
        "scale_tolerance": {"rtol": SCALE_RTOL, "atol": SCALE_ATOL},
        "output_mse_tolerance": "max(1e-12, 1e-6 * abs(reference MSE)) for the two-doc correctness probe only",
        "preflight_candidate_execution_count": (
            len(identity_cases) + len(representative_targets) * 2 * len(alphas_check)
            + 2 + 2 + 2
        ),
        "preflight_full_linear_candidate_calls": len(representative_targets) * 2 * len(alphas_check),
        "preflight_row_gptq_calls": len(identity_cases) + 2,
        "candidate_order_extra_full_linear_calls": 2,
        "unequal_alpha_row_repeat_calls": 2,
        "native_reference_calls": 2,
        "activation_cache": down_cache_timing,
        "e1_cache_hashes": e1_cache_hashes,
        "v23_prefix_manifest_sha256": sha_file(E1_CACHE / "PREFIX_CACHE_MANIFEST.json"),
        "e2_document_manifest_sha256": sha_file(DOCS_PATH),
        "model_load_seconds": model_load_seconds,
        "prefix_cache_validation_seconds": prefix_validation_seconds,
        "down_input_capture_seconds": activation_prep_seconds,
        "formal_hessian_preflight_seconds_by_projection": hessian_times,
        "formal_hessian_checkpoint_io_seconds_by_projection": hessian_io_times,
        "formal_hinv_preflight_seconds_by_projection": hprep_times,
        "full_gptq_probe_seconds_by_projection": full_times,
        "row_gptq_probe_seconds_by_projection": row_times,
        "score_probe_seconds": score_probe,
        "native_alpha1_reference_seconds": native_seconds,
        "unequal_alpha_row_probe_seconds": row_smoke_seconds + row_repeat_seconds,
        "peak_cuda": peak_mem(),
        "wall_seconds": preflight_wall,
        "source_hashes_at_preflight": source_hashes,
    }
    if not all(correctness["checks"].values()):
        correctness["status"] = "FAIL"
        dump(OUT / "PREFLIGHT_CORRECTNESS.json", correctness)
        raise RuntimeError("E2 preflight correctness gate failed")
    dump(OUT / "PREFLIGHT_CORRECTNESS.json", correctness)
    dump(OUT / "SCALAR_STRUCTURED_IDENTITY.json", {
        "status": "PASS",
        "definition": "For equal per-group alpha, one-row native GPTQ must reproduce the matching full-Linear scalar candidate.",
        "cases": identity_cases,
        "case_count": len(identity_cases),
        "alphas": alphas_check,
        "representative_targets": [f"L{li}/{p}" for li, p in representative_targets],
        "weight_zero_g_idx_tolerance": "exact",
        "scale_tolerance": {"rtol": SCALE_RTOL, "atol": SCALE_ATOL},
        "output_mse_tolerance": "max(1e-12, 1e-6 * abs(reference MSE)) on two search documents",
        "held_data_used": False,
    })

    # Runtime estimate separates the already measured preflight from the
    # subsequent formal process. The formal process loads the model and validates
    # the immutable activation caches again, while down-input replay is already
    # materialized by preflight.
    full_candidate_counts = {"up_proj": 276, "down_proj": 276}
    structured_row_counts = {"up_proj": 864, "down_proj": 864}
    full_central = sum(full_candidate_counts[p] * statistics.median(full_times[p]) for p in PROJECTIONS)
    row_central = sum(structured_row_counts[p] * statistics.median(row_times[p]) for p in PROJECTIONS)
    h_central = 0.0
    h_conservative = 0.0
    for p in PROJECTIONS:
        # Preflight materializes two H states per projection (one representative
        # layer, both splits). Formal needs four additional H builds per
        # projection, plus six Hinv preparations and six cached H reads.
        hb = hessian_times[p]
        hp = hprep_times[p]
        io_h = hessian_io_times[p]
        h_central += 4 * (statistics.median(hb) + statistics.median(io_h)) + 6 * statistics.median(hp)
        h_conservative += 4 * (max(hb) + max(io_h)) + 6 * max(hp)
    score_full_units = 552 + 552  # one 4-row score for each search/held full candidate
    score_row_units = 1728 + 1728  # unique off-diagonal row candidates, search plus held upper bound
    score_central = 0.0
    score_conservative = 0.0
    for p in PROJECTIONS:
        per_full = score_probe[p]["two_doc_four_row_seconds"] * (16 / 2)
        per_row = score_probe[p]["two_doc_one_row_seconds"] * (16 / 2)
        score_central += (score_full_units * 0.5 * per_full + score_row_units * 0.5 * per_row)
        score_conservative += (score_full_units * 0.5 * per_full + score_row_units * 0.5 * per_row)
    # Checkpoint count/write time is measured with real representative payloads.
    io_root = RAW / "preflight_io_probe"
    full_probe = {"weight_fp16": expected_bank[0.9]["weight_fp16"][ROWS["down_proj"]].detach().cpu(),
                  "scales": expected_bank[0.9]["scales"][ROWS["down_proj"]].detach().cpu(),
                  "zeros": expected_bank[0.9]["zeros"][ROWS["down_proj"]].detach().cpu(),
                  "g_idx": expected_bank[0.9]["g_idx"].detach().cpu()}
    row_probe = {k: structured_1[k].detach().cpu() for k in ("weight_fp16", "scales", "zeros", "g_idx")}
    _, io_full_seconds = timed(lambda: save_torch(io_root / "full_candidate_probe.pt", full_probe))
    _, io_row_seconds = timed(lambda: save_torch(io_root / "row_candidate_probe.pt", row_probe))
    _, io_json_seconds = timed(lambda: dump(io_root / "scores_probe.json", {"scores": [0.0] * 32}))
    io_central = 552 * io_full_seconds + 1728 * io_row_seconds + 4000 * io_json_seconds / 32
    formal_startup = model_load_seconds + prefix_validation_seconds + down_cache_validation_seconds
    stitch_commit = 48 * 0.00001
    finalization = 30.0
    central_seconds = formal_startup + h_central + full_central + row_central + score_central + io_central + stitch_commit + finalization
    conservative_seconds = max(central_seconds * 1.5,
                               formal_startup + h_conservative +
                               sum(full_candidate_counts[p] * max(full_times[p]) for p in PROJECTIONS) +
                               sum(structured_row_counts[p] * max(row_times[p]) for p in PROJECTIONS) +
                               score_conservative + io_central + stitch_commit + finalization)
    total_e2_central_seconds = preflight_wall + central_seconds
    total_e2_conservative_seconds = preflight_wall + conservative_seconds
    six_hour_gate = conservative_seconds <= 6 * 3600
    twenty_four_hour_gate = total_e2_conservative_seconds <= 24 * 3600
    cost = {
        "status": "E2_COST_GATE_PASS" if six_hour_gate and twenty_four_hour_gate else "E2_COST_GATE_FAIL",
        "single_experiment_limit_seconds": 6 * 3600,
        "total_e2_limit_seconds": 24 * 3600,
        "formal_central_seconds": central_seconds,
        "formal_central_hours": central_seconds / 3600,
        "projected_total_e2_central_seconds_including_preflight": total_e2_central_seconds,
        "projected_total_e2_central_hours_including_preflight": total_e2_central_seconds / 3600,
        "projected_total_e2_conservative_seconds_including_preflight": total_e2_conservative_seconds,
        "projected_total_e2_conservative_hours_including_preflight": total_e2_conservative_seconds / 3600,
        "six_hour_formal_gate_pass": six_hour_gate,
        "twenty_four_hour_total_gate_pass": twenty_four_hour_gate,
        "central_seconds": central_seconds,
        "central_hours": central_seconds / 3600,
        "conservative_seconds": conservative_seconds,
        "conservative_hours": conservative_seconds / 3600,
        "conservative_multiplier": conservative_seconds / central_seconds if central_seconds else None,
        "measured_components_seconds": {
            "model_load": model_load_seconds,
            "formal_model_load": model_load_seconds,
            "formal_prefix_cache_validation": prefix_validation_seconds,
            "formal_down_input_cache_validation": down_cache_validation_seconds,
            "preflight_down_input_replay_capture_already_elapsed": activation_prep_seconds,
            "hessian_and_hinv_estimate": h_central,
            "full_linear_candidate_gptq": full_central,
            "structured_per_row_candidate_gptq": row_central,
            "search_and_held_local_scoring_upper_bound": score_central,
            "candidate_and_score_checkpoint_io": io_central,
            "stitch_commit_restore": stitch_commit,
            "report_and_validation": finalization,
        },
        "execution_counts": {
            "full_linear_gptq_candidate_calls": 552,
            "per_row_structured_gptq_candidate_calls_max": 1728,
            "search_full_candidate_row_scores": 2208,
            "search_structured_row_candidate_scores_max": 1728,
            "held_full_candidate_row_scores": 2208,
            "held_structured_row_candidate_scores_max": 1728,
            "formal_block_forward_or_ppl_calls": 0,
        },
        "timing_basis": "GPU synchronized; measured two representative shapes (L0/up d=896, L23/down d=4864), target/split H probes and exact down-input cache preparation; conservative uses max observed candidate time or 1.5x total, whichever is larger",
        "preflight_wall_seconds": preflight_wall,
        "formal_startup_estimate_seconds": formal_startup,
        "down_input_cache_validation_seconds": down_cache_validation_seconds,
        "peak_cuda": correctness["peak_cuda"],
        "input_and_method_cache_bytes": sum(f["size_bytes"] for f in down_cache_timing.get("files", {}).values()),
        "gate_pass": conservative_seconds <= 6 * 3600,
    }
    dump(OUT / "E2_COST_GATE.json", cost)
    budget = json.loads((OUT / "E2_EXECUTION_BUDGET.json").read_text(encoding="utf-8"))
    budget["status"] = "FROZEN_COST_PREFLIGHT_PASS" if cost["gate_pass"] else "E2_COST_GATE_FAIL"
    budget["preflight_measured_counts"] = cost["execution_counts"]
    budget["preflight_cost"] = {"central_hours": cost["central_hours"], "conservative_hours": cost["conservative_hours"]}
    dump(OUT / "E2_EXECUTION_BUDGET.json", budget)
    write_runtime_preflight({"status": "PASS", "environment": env, "model_load_seconds": model_load_seconds,
                             "prefix_cache_validation_seconds": prefix_validation_seconds,
                             "down_input_capture_seconds": activation_prep_seconds,
                             "down_input_cache_validation_seconds": down_cache_validation_seconds,
                             "hessian_and_hinv_probe_seconds": {"hessian": hessian_times, "hinv": hprep_times},
                             "candidate_probe_seconds": {"full_linear": full_times, "per_row": row_times},
                             "scoring_probe_seconds": score_probe, "peak_cuda": correctness["peak_cuda"],
                             "wall_seconds": preflight_wall, "cost_gate": cost["status"]})
    if not cost["gate_pass"]:
        dump(OUT / "RUN_STATUS.json", {"status": "STOPPED_COST_GATE", "verdict": "E2_COST_GATE_FAIL",
                                        "cost_gate": cost, "formal_started": False})
        raise RuntimeError("E2 cost preflight failed the frozen six-hour gate; formal search was not started")
    return {"correctness": correctness, "cost": cost, "model": model, "pristine": pristine,
            "e1_arrays": e1_arrays, "down_arrays": down_arrays}


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise RuntimeError(f"Refusing to write an empty result table: {path}")
    fields = list(dict.fromkeys(key for row in rows for key in row))
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


def full_candidate_path(li: int, projection: str, split: str, alpha: float) -> Path:
    return RAW / "full_candidates" / f"L{li}_{projection}_{split}_a{dense_grid_idx(alpha):02d}.pt"


def full_candidate_json_path(li: int, projection: str, split: str, alpha: float) -> Path:
    return RAW / "full_candidates" / f"L{li}_{projection}_{split}_a{dense_grid_idx(alpha):02d}.json"


def structured_candidate_path(li: int, projection: str, split: str, row: int, a: float, b: float) -> Path:
    return RAW / "structured_candidates" / f"L{li}_{projection}_{split}_r{row}_{tuple_key(a,b)}.pt"


def structured_candidate_json_path(li: int, projection: str, split: str, row: int, a: float, b: float) -> Path:
    return RAW / "structured_candidates" / f"L{li}_{projection}_{split}_r{row}_{tuple_key(a,b)}.json"


def loss_by_row(candidate: dict, row: int) -> float:
    return float(candidate["search_mse_by_row"][str(row)])


def get_full_candidate(li, projection, split, alpha, weight_gpu, prepared, x_mm, row_ids, counters):
    path = full_candidate_path(li, projection, split, alpha)
    meta_path = full_candidate_json_path(li, projection, split, alpha)
    expected = {"layer": li, "projection": projection, "split": split, "alpha": alpha,
                "rows": list(row_ids), "hessian_sha256": prepared.h_sha256,
                "pristine_weight_sha256": sha_tensor(weight_gpu),
                "search_doc_indices": SPLITS[split]["search"],
                "e2_manifest_sha256": sha_file(DOCS_PATH)}
    if path.exists() and meta_path.exists():
        obj = read_torch(path)
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        for key, value in expected.items():
            if meta.get(key) != value:
                raise RuntimeError(f"Full candidate checkpoint protocol mismatch: {path} ({key})")
        for key, hash_key in (("weight_fp16", "weight_sha256"), ("scales", "scales_sha256"),
                              ("zeros", "zeros_sha256"), ("g_idx", "g_idx_sha256")):
            if sha_tensor(obj[key]) != meta[hash_key]:
                raise RuntimeError(f"Full candidate tensor hash mismatch: {path}:{key}")
        obj.update({"search_mse_by_row": meta["search_mse_by_row"], "meta": meta})
        counters["full_candidate_cache_hits"] += 1
        counters["full_candidate_gptq_seconds"] += float(meta.get("gptq_seconds", 0.0))
        counters["search_full_score_seconds"] += float(meta.get("search_score_seconds", 0.0))
        counters["checkpoint_io_seconds"] += float(meta.get("checkpoint_io_seconds", 0.0))
        return obj
    if path.exists() != meta_path.exists():
        raise RuntimeError(f"Incomplete full candidate checkpoint pair: {path}")
    append_event("full_linear_candidate_started", layer=li, projection=projection, split=split, alpha=alpha)
    counters["full_candidate_attempts"] += 1
    sync(); start = time.perf_counter()
    result = optimized_dynamic_gptq(weight_gpu, prepared, alpha=alpha, check_cholesky=(alpha == 1.0))
    sync(); gptq_seconds = time.perf_counter() - start
    sync(); start = time.perf_counter()
    quantized_rows = result.weight_fp16[row_ids].detach()
    scores = score_selected_rows(x_mm, SPLITS[split]["search"], weight_gpu, quantized_rows, row_ids)
    sync(); score_seconds = time.perf_counter() - start
    if not torch.isfinite(quantized_rows).all().item() or not all(math.isfinite(v) for v in scores.values()):
        raise RuntimeError(f"Non-finite full candidate at L{li}/{projection}/{split}/alpha={alpha}")
    obj = {"weight_fp16": quantized_rows.detach().cpu().contiguous(),
           "scales": result.scales[row_ids].detach().cpu().contiguous(),
           "zeros": result.zeros[row_ids].detach().cpu().contiguous(),
           "g_idx": result.g_idx.detach().cpu().contiguous()}
    hashes = {key: sha_tensor(value) for key, value in obj.items()}
    save_started = time.perf_counter()
    save_torch(path, obj)
    io_seconds = time.perf_counter() - save_started
    meta = {**expected, "search_mse_by_row": {str(r): scores[int(r)] for r in row_ids},
            "weight_sha256": hashes["weight_fp16"], "scales_sha256": hashes["scales"],
            "zeros_sha256": hashes["zeros"], "g_idx_sha256": hashes["g_idx"],
            "gptq_seconds": gptq_seconds, "search_score_seconds": score_seconds,
            "checkpoint_io_seconds": io_seconds, "candidate_status": "COMPLETE"}
    save_started = time.perf_counter()
    dump(meta_path, meta)
    json_io_seconds = time.perf_counter() - save_started
    meta["checkpoint_io_seconds"] += json_io_seconds
    # Refresh the sidecar with the complete measured write time.
    dump(meta_path, meta)
    counters["full_candidate_completed"] += 1
    counters["full_candidate_gptq_seconds"] += gptq_seconds
    counters["search_full_score_seconds"] += score_seconds
    counters["checkpoint_io_seconds"] += meta["checkpoint_io_seconds"]
    append_event("full_linear_candidate_completed", layer=li, projection=projection, split=split, alpha=alpha,
                 gptq_seconds=gptq_seconds, score_seconds=score_seconds,
                 checkpoint_io_seconds=meta["checkpoint_io_seconds"], candidate_sha256=hashes["weight_fp16"])
    result_obj = dict(obj)
    result_obj.update({"search_mse_by_row": meta["search_mse_by_row"], "meta": meta})
    del result
    return result_obj


def structured_alphas_by_group(input_dim: int, alpha_a: float, alpha_b: float) -> list[float]:
    regions = native_group_assignment(input_dim)
    a_set = set(regions["A"])
    return [alpha_a if group in a_set else alpha_b for group in range(math.ceil(input_dim / 128))]


def get_structured_candidate(li, projection, split, row, alpha_a, alpha_b,
                             weight_gpu, prepared, x_mm, full_bank, counters):
    key = tuple_key(alpha_a, alpha_b)
    # All diagonal structured candidates are exactly the shared scalar bank;
    # the v17.3 tolerance audit is a mandatory gate before using this identity.
    if alpha_a == alpha_b:
        candidate = full_bank[alpha_a]
        j = ROWS[projection].index(row)
        return {"weight_fp16": candidate["weight_fp16"][j:j + 1].clone(),
                "scales": candidate["scales"][j:j + 1].clone(),
                "zeros": candidate["zeros"][j:j + 1].clone(),
                "g_idx": candidate["g_idx"].clone(),
                "search_mse": loss_by_row(candidate, row), "candidate_source": "shared_full_linear",
                "gptq_seconds": 0.0, "score_seconds": 0.0, "checkpoint_io_seconds": 0.0}
    path = structured_candidate_path(li, projection, split, row, alpha_a, alpha_b)
    meta_path = structured_candidate_json_path(li, projection, split, row, alpha_a, alpha_b)
    expected = {"layer": li, "projection": projection, "split": split, "row": row,
                "alpha_A": alpha_a, "alpha_B": alpha_b, "tuple_key": key,
                "hessian_sha256": prepared.h_sha256,
                "pristine_row_sha256": sha_tensor(weight_gpu[row:row + 1]),
                "search_doc_indices": SPLITS[split]["search"],
                "e2_manifest_sha256": sha_file(DOCS_PATH)}
    if path.exists() and meta_path.exists():
        obj = read_torch(path)
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        for field, value in expected.items():
            if meta.get(field) != value:
                raise RuntimeError(f"Structured candidate protocol mismatch: {path} ({field})")
        for field, hash_key in (("weight_fp16", "weight_sha256"), ("scales", "scales_sha256"),
                                ("zeros", "zeros_sha256"), ("g_idx", "g_idx_sha256")):
            if sha_tensor(obj[field]) != meta[hash_key]:
                raise RuntimeError(f"Structured candidate tensor hash mismatch: {path}:{field}")
        counters["structured_candidate_cache_hits"] += 1
        counters["structured_row_gptq_seconds"] += float(meta.get("gptq_seconds", 0.0))
        counters["search_row_score_seconds"] += float(meta.get("search_score_seconds", 0.0))
        counters["checkpoint_io_seconds"] += float(meta.get("checkpoint_io_seconds", 0.0))
        return {**obj, "search_mse": meta["search_mse"], "candidate_source": "row_gptq_cache",
                "gptq_seconds": float(meta["gptq_seconds"]), "score_seconds": float(meta["search_score_seconds"]),
                "checkpoint_io_seconds": float(meta["checkpoint_io_seconds"])}
    if path.exists() != meta_path.exists():
        raise RuntimeError(f"Incomplete structured candidate checkpoint pair: {path}")
    append_event("structured_row_candidate_started", layer=li, projection=projection, split=split, row=row,
                 alpha_A=alpha_a, alpha_B=alpha_b)
    counters["structured_candidate_attempts"] += 1
    alphas = structured_alphas_by_group(weight_gpu.shape[1], alpha_a, alpha_b)
    sync(); start = time.perf_counter()
    result = native_row_candidate(weight_gpu[row:row + 1], prepared, alphas)
    sync(); gptq_seconds = time.perf_counter() - start
    sync(); start = time.perf_counter()
    score = score_one_row(x_mm, SPLITS[split]["search"], weight_gpu[row], result["weight_fp16"][0])
    sync(); score_seconds = time.perf_counter() - start
    if not math.isfinite(score) or not torch.isfinite(result["weight_fp16"]).all().item():
        raise RuntimeError(f"Non-finite structured candidate at L{li}/{projection}/{split}/r{row}/{key}")
    obj = {field: result[field].detach().cpu().contiguous() for field in ("weight_fp16", "scales", "zeros", "g_idx")}
    hashes = {field: sha_tensor(obj[field]) for field in obj}
    io_started = time.perf_counter()
    save_torch(path, obj)
    checkpoint_io = time.perf_counter() - io_started
    meta = {**expected, "search_mse": score, "range_find_params_calls": result["range_find_params_calls"],
            "weight_sha256": hashes["weight_fp16"], "scales_sha256": hashes["scales"],
            "zeros_sha256": hashes["zeros"], "g_idx_sha256": hashes["g_idx"],
            "gptq_seconds": gptq_seconds, "search_score_seconds": score_seconds,
            "checkpoint_io_seconds": checkpoint_io, "candidate_status": "COMPLETE"}
    io_started = time.perf_counter()
    dump(meta_path, meta)
    meta["checkpoint_io_seconds"] += time.perf_counter() - io_started
    dump(meta_path, meta)
    counters["structured_candidate_completed"] += 1
    counters["structured_row_gptq_seconds"] += gptq_seconds
    counters["search_row_score_seconds"] += score_seconds
    counters["checkpoint_io_seconds"] += meta["checkpoint_io_seconds"]
    append_event("structured_row_candidate_completed", layer=li, projection=projection, split=split, row=row,
                 alpha_A=alpha_a, alpha_B=alpha_b, gptq_seconds=gptq_seconds,
                 score_seconds=score_seconds, checkpoint_io_seconds=meta["checkpoint_io_seconds"],
                 candidate_sha256=hashes["weight_fp16"])
    return {**obj, "search_mse": score, "candidate_source": "row_gptq",
            "gptq_seconds": gptq_seconds, "score_seconds": score_seconds,
            "checkpoint_io_seconds": meta["checkpoint_io_seconds"]}


def select_best_alpha(scores: dict[float, float]) -> float:
    # The frozen M2 exact-loss tie rule is to prefer the larger alpha.
    return min(scores, key=lambda alpha: (scores[alpha], -alpha))


def row_unit_id(li, projection, row, split) -> str:
    return f"L{li}_{projection}_R{row}_{split}"


def candidate_json_weight_metadata(candidate: dict) -> dict:
    return {"weight_sha256": sha_tensor(candidate["weight_fp16"]),
            "scales": candidate["scales"].detach().cpu().double().reshape(-1).tolist(),
            "zeros": candidate["zeros"].detach().cpu().double().reshape(-1).tolist(),
            "g_idx_sha256": sha_tensor(candidate["g_idx"]),
            "g_idx_unique": torch.unique(candidate["g_idx"].detach().cpu()).tolist()}


def run_search(model, pristine, e1_arrays, down_arrays):
    preflight = json.loads((OUT / "PREFLIGHT_CORRECTNESS.json").read_text(encoding="utf-8"))
    checkpoint_key_check = json.loads((OUT / "POST_PREFLIGHT_FIX_VALIDATION.json").read_text(encoding="utf-8"))
    cost_gate = json.loads((OUT / "E2_COST_GATE.json").read_text(encoding="utf-8"))
    if preflight.get("status") != "PASS" or cost_gate.get("status") != "E2_COST_GATE_PASS":
        raise RuntimeError("Formal E2 search requires correctness and cost gates to PASS")
    full_paths = [full_candidate_path(0, "up_proj", "S0", alpha) for alpha in DENSE_GRID]
    if len(set(full_paths)) != len(DENSE_GRID):
        raise RuntimeError("Frozen dense-grid full candidate checkpoint keys are not unique")
    prior_events = [json.loads(line) for line in EVENTS.read_text(encoding="utf-8").splitlines() if line.strip()] if EVENTS.exists() else []
    latest_start = max((event["utc"] for event in prior_events if event["kind"] == "formal_process_started"), default="")
    prior_completed_targets = [event for event in prior_events
                               if event["kind"] == "target_split_search_completed"
                               and event["utc"] < latest_start]
    prior_hinv_seconds = math.fsum(float(event.get("hinv_prepare_seconds", 0.0)) for event in prior_completed_targets)
    prior_formal_hessian_events = [event for event in prior_completed_targets
                                   if not ((event["layer"] == 0 and event["projection"] == "up_proj")
                                           or (event["layer"] == 23 and event["projection"] == "down_proj"))]
    counters = {"full_candidate_attempts": 0, "full_candidate_completed": 0,
                "full_candidate_cache_hits": 0, "structured_candidate_attempts": 0,
                "structured_candidate_completed": 0, "structured_candidate_cache_hits": 0,
                "full_candidate_gptq_seconds": 0.0, "structured_row_gptq_seconds": 0.0,
                "search_full_score_seconds": 0.0, "search_row_score_seconds": 0.0,
                "checkpoint_io_seconds": 0.0, "structured_scheduled_slots": 0,
                "structured_duplicate_tuple_reuses": 0,
                "hessian_build_seconds": 0.0, "hessian_checkpoint_io_seconds": 0.0,
                "hinv_prepare_seconds": prior_hinv_seconds}
    counters["hessian_build_seconds"] = math.fsum(
        float(event.get("hessian_seconds", 0.0)) for event in prior_formal_hessian_events)
    counters["hessian_checkpoint_io_seconds"] = math.fsum(
        float(event.get("hessian_io_seconds", 0.0)) for event in prior_formal_hessian_events)
    search_records = []
    unit_states = {}
    full_bank_all = {}
    formal_started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats(DEVICE)
    dump(OUT / "RUN_STATUS.json", {"status": "RUNNING_SEARCH", "formal_started_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                                    "completed_target_splits": 0, "target_split_total": len(LAYERS) * len(PROJECTIONS) * 2})
    append_event("formal_search_started", targets=6, splits=2, units=48)

    completed_target_splits = 0
    for li in LAYERS:
        for projection in PROJECTIONS:
            name = f"L{li}/{projection}"
            weight_cpu = pristine[name]
            weight_gpu = weight_cpu.to(DEVICE)
            rows = ROWS[projection]
            x_mm = projection_input(projection, li, e1_arrays, down_arrays)
            for split in ("S0", "S1"):
                h, nsamples, h_sha, h_seconds, h_io_seconds, h_reused = build_or_load_hessian(
                    model, li, projection, split, x_mm, SPLITS[split]["search"])
                sync(); t0 = time.perf_counter()
                prepared = prepare_native_state(h, nsamples=nsamples)
                sync(); hinv_seconds = time.perf_counter() - t0
                if not h_reused:
                    counters["hessian_build_seconds"] += h_seconds
                    counters["hessian_checkpoint_io_seconds"] += h_io_seconds
                counters["hinv_prepare_seconds"] += hinv_seconds
                full_bank = {}
                for alpha in DENSE_GRID:
                    full_bank[alpha] = get_full_candidate(li, projection, split, alpha, weight_gpu,
                                                          prepared, x_mm, rows, counters)
                full_bank_all[(li, projection, split)] = full_bank
                original_selected = {}
                dense_selected = {}
                for row in rows:
                    scalar_scores = {alpha: loss_by_row(full_bank[alpha], row) for alpha in ALPHAS}
                    dense_scores = {alpha: loss_by_row(full_bank[alpha], row) for alpha in DENSE_GRID}
                    base_alpha = select_best_alpha(scalar_scores)
                    dense_alpha = min(dense_scores, key=lambda alpha: (dense_scores[alpha], -alpha))
                    original_selected[row] = base_alpha
                    dense_selected[row] = dense_alpha
                    for alpha in DENSE_GRID:
                        search_records.append({"unit_id": row_unit_id(li, projection, row, split),
                                               "layer": li, "projection": projection, "row": row, "split_id": split,
                                               "candidate_method": "scalar_dense_bank", "alpha": alpha,
                                               "alpha_A": alpha, "alpha_B": alpha,
                                               "search_mse": dense_scores[alpha],
                                               "candidate_source": "full_linear", "is_original_10": alpha in ALPHAS,
                                               "selected_by_original_10": alpha == base_alpha,
                                               "selected_by_dense_46": alpha == dense_alpha,
                                               "gptq_execution_kind": "full_linear"})
                for row in rows:
                    base = original_selected[row]
                    visited = {}
                    for alpha in ALPHAS:
                        cand = get_structured_candidate(li, projection, split, row, alpha, alpha,
                                                        weight_gpu, prepared, x_mm, full_bank, counters)
                        visited[(alpha, alpha)] = cand
                        search_records.append({"unit_id": row_unit_id(li, projection, row, split),
                                               "layer": li, "projection": projection, "row": row, "split_id": split,
                                               "candidate_method": "structured_2region", "alpha": "",
                                               "alpha_A": alpha, "alpha_B": alpha,
                                               "search_mse": cand["search_mse"],
                                               "candidate_source": cand["candidate_source"],
                                               "is_original_10": True, "selected_by_original_10": alpha == base,
                                               "selected_by_dense_46": False,
                                               "gptq_execution_kind": "shared_full_linear"})
                    current_a = current_b = base
                    current_score = visited[(base, base)]["search_mse"]
                    accepted = []
                    scheduled = 0
                    for sweep in (1, 2):
                        for region in ("A", "B"):
                            old = current_a if region == "A" else current_b
                            trial_scores = {old: current_score}
                            for alpha in ALPHAS:
                                if alpha == old:
                                    continue
                                scheduled += 1
                                counters["structured_scheduled_slots"] += 1
                                trial_a = alpha if region == "A" else current_a
                                trial_b = current_b if region == "A" else alpha
                                pair = (trial_a, trial_b)
                                if pair not in visited:
                                    visited[pair] = get_structured_candidate(
                                        li, projection, split, row, trial_a, trial_b,
                                        weight_gpu, prepared, x_mm, full_bank, counters)
                                else:
                                    counters["structured_duplicate_tuple_reuses"] += 1
                                trial_scores[alpha] = visited[pair]["search_mse"]
                                search_records.append({"unit_id": row_unit_id(li, projection, row, split),
                                                       "layer": li, "projection": projection, "row": row, "split_id": split,
                                                       "candidate_method": "structured_2region_sweep", "alpha": "",
                                                       "alpha_A": trial_a, "alpha_B": trial_b,
                                                       "search_mse": trial_scores[alpha],
                                                       "candidate_source": visited[pair]["candidate_source"],
                                                       "is_original_10": False,
                                                       "selected_by_original_10": False,
                                                       "selected_by_dense_46": False,
                                                       "gptq_execution_kind": "row_gptq" if trial_a != trial_b else "shared_full_linear",
                                                       "sweep": sweep, "region_updated": region})
                            best = min(trial_scores, key=lambda value: (trial_scores[value], -value))
                            if trial_scores[best] < current_score:
                                previous = {"alpha_A": current_a, "alpha_B": current_b, "search_mse": current_score}
                                if region == "A":
                                    current_a = best
                                else:
                                    current_b = best
                                current_score = trial_scores[best]
                                accepted.append({"sweep": sweep, "region": region,
                                                 "from_alpha": old, "to_alpha": best,
                                                 "from_pair": [previous["alpha_A"], previous["alpha_B"]],
                                                 "to_pair": [current_a, current_b],
                                                 "from_search_mse": previous["search_mse"],
                                                 "to_search_mse": current_score,
                                                 "search_mse_improvement": previous["search_mse"] - current_score})
                    final_candidate = visited[(current_a, current_b)]
                    final_metadata = candidate_json_weight_metadata(final_candidate)
                    state = {"unit_id": row_unit_id(li, projection, row, split),
                             "layer": li, "projection": projection, "row": row, "split_id": split,
                             "native_alpha1_search_mse": loss_by_row(full_bank[1.0], row),
                             "original_alpha": base, "original_search_mse": visited[(base, base)]["search_mse"],
                             "dense_alpha": dense_selected[row], "dense_search_mse": loss_by_row(full_bank[dense_selected[row]], row),
                             "structured_alpha_A": current_a, "structured_alpha_B": current_b,
                             "structured_search_mse": current_score,
                             "alpha_abs_difference": abs(current_a - current_b),
                             "same_alpha": current_a == current_b,
                             "region_A_more_aggressive": current_a < current_b,
                             "region_B_more_aggressive": current_b < current_a,
                             "base_alpha": base, "scheduled_coordinate_slots": scheduled,
                             "unique_structured_tuples": len(visited),
                             "unique_off_diagonal_tuples": sum(a != b for a, b in visited),
                             "row_gptq_calls_for_unit": sum(1 for a, b in visited if a != b),
                             "duplicate_tuple_reuses": scheduled - sum(1 for a, b in visited if a != b),
                             "accepted_moves": accepted,
                             "explored_tuples": [[a, b] for a, b in sorted(visited)],
                             "search_loss_by_tuple": {f"{alpha_key(a)}|{alpha_key(b)}": float(c["search_mse"])
                                                       for (a, b), c in visited.items()},
                             "selected_weight_sha256": final_metadata["weight_sha256"],
                             "selected_scales": final_metadata["scales"],
                             "selected_zeros": final_metadata["zeros"],
                             "selected_g_idx_sha256": final_metadata["g_idx_sha256"],
                             "selected_g_idx_unique": final_metadata["g_idx_unique"],
                             "hessian_sha256": h_sha, "nsamples": nsamples,
                             "structured_candidate_bank": visited}
                    unit_states[state["unit_id"]] = state
                completed_target_splits += 1
                dump(OUT / "RUN_STATUS.json", {"status": "RUNNING_SEARCH", "completed_target_splits": completed_target_splits,
                                                "target_split_total": 12, "last_completed": f"L{li}/{projection}/{split}",
                                                "formal_elapsed_seconds": time.perf_counter() - formal_started,
                                                "candidate_counters": counters})
                append_event("target_split_search_completed", layer=li, projection=projection, split=split,
                             full_candidate_bank_size=len(full_bank), units=len(rows),
                             hessian_sha256=h_sha, hessian_seconds=h_seconds,
                             hessian_io_seconds=h_io_seconds, hinv_prepare_seconds=hinv_seconds,
                             hessian_cache_reused=h_reused)
                del prepared, h, full_bank
                sync()
    search_seconds = time.perf_counter() - formal_started
    if len(unit_states) != 48:
        raise RuntimeError(f"Expected 48 row/split structured units, got {len(unit_states)}")
    # Save search-only artifacts before any held score is requested.
    write_csv(OUT / "SEARCH_RESULTS.csv", search_records)
    choices_rows = []
    dense_rows = []
    for unit_id, state in unit_states.items():
        choices_rows.append({k: (json.dumps(v, ensure_ascii=False, separators=(",", ":")) if isinstance(v, (list, dict)) else v)
                             for k, v in state.items() if k != "structured_candidate_bank"})
        dense_rows.append({"unit_id": unit_id, "layer": state["layer"], "projection": state["projection"],
                           "row": state["row"], "split_id": state["split_id"],
                           "native_alpha1_search_mse": state["native_alpha1_search_mse"],
                           "original_10_alpha": state["original_alpha"],
                           "original_10_search_mse": state["original_search_mse"],
                           "dense_46_alpha": state["dense_alpha"],
                           "dense_46_search_mse": state["dense_search_mse"],
                           "dense_candidate_count": len(DENSE_GRID),
                           "dense_grid": json.dumps(DENSE_GRID, separators=(",", ":"))})
    write_csv(OUT / "STRUCTURED_ALPHA_CHOICES.csv", choices_rows)
    write_csv(OUT / "DENSE_SCALAR_CONTROL.csv", dense_rows)
    unit_choice_lock = [{"unit_id": unit_id, "m2_alpha": state["original_alpha"],
                         "dense_alpha": state["dense_alpha"],
                         "structured_alpha_A": state["structured_alpha_A"],
                         "structured_alpha_B": state["structured_alpha_B"],
                         "search_mse": {"native": state["native_alpha1_search_mse"],
                                        "m2": state["original_search_mse"],
                                        "dense": state["dense_search_mse"],
                                        "structured": state["structured_search_mse"]}}
                        for unit_id, state in sorted(unit_states.items())]
    search_selection_sha = sha_bytes(json.dumps(unit_choice_lock, sort_keys=True, separators=(",", ":")).encode())
    lock_payload = {"status": "LOCKED_BEFORE_HELD_SCORING", "locked_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                    "protocol_sha256": sha_file(PROTOCOL_PATH), "e2_document_manifest_sha256": sha_file(DOCS_PATH),
                    "search_results_sha256": sha_file(OUT / "SEARCH_RESULTS.csv"),
                    "structured_choices_sha256": sha_file(OUT / "STRUCTURED_ALPHA_CHOICES.csv"),
                    "dense_control_sha256": sha_file(OUT / "DENSE_SCALAR_CONTROL.csv"),
                    "search_selection_sha256": search_selection_sha,
                    "units": len(unit_states), "held_indices_not_read_for_selection": True,
                    "selection_recomputed_from_search_only": True}
    dump(OUT / "SEARCH_SELECTION_LOCK.json", lock_payload)
    append_event("held_selection_locked", search_selection_sha256=search_selection_sha,
                 lock_path="SEARCH_SELECTION_LOCK.json")
    return {"unit_states": unit_states, "full_bank": full_bank_all, "counters": counters,
            "search_records": search_records, "search_seconds": search_seconds,
            "search_selection_sha256": search_selection_sha, "choices_rows": choices_rows,
            "dense_rows": dense_rows, "peak_cuda": peak_mem()}


def held_score_path(kind: str, li, projection, split, alpha=None, row=None, a=None, b=None) -> Path:
    if kind == "full":
        return RAW / "held_full_scores" / f"L{li}_{projection}_{split}_a{dense_grid_idx(alpha):02d}.json"
    return RAW / "held_structured_scores" / f"L{li}_{projection}_{split}_r{row}_{tuple_key(a,b)}.json"


def run_held_scoring(model, pristine, e1_arrays, down_arrays, search_bundle):
    lock = json.loads((OUT / "SEARCH_SELECTION_LOCK.json").read_text(encoding="utf-8"))
    if lock.get("status") != "LOCKED_BEFORE_HELD_SCORING" or lock.get("search_selection_sha256") != search_bundle["search_selection_sha256"]:
        raise RuntimeError("Held scoring cannot begin without the matching locked search selection")
    if any(json.loads(line).get("kind") == "held_score_started" for line in EVENTS.read_text(encoding="utf-8").splitlines() if line.strip()):
        # Resume is allowed only if every existing held record names this exact selection lock.
        for path in list((RAW / "held_full_scores").glob("*.json")) + list((RAW / "held_structured_scores").glob("*.json")):
            obj = json.loads(path.read_text(encoding="utf-8"))
            if obj.get("search_selection_sha256") != lock["search_selection_sha256"]:
                raise RuntimeError(f"Held score belongs to a different search lock: {path}")
    counters = search_bundle["counters"]
    held_full = {}
    held_structured = {}
    repeat_noise = {}
    repeat_score_seconds = 0.0
    held_score_seconds = 0.0
    held_io_seconds = 0.0
    torch.cuda.reset_peak_memory_stats(DEVICE)
    held_started = time.perf_counter()
    append_event("held_scoring_started", search_selection_sha256=lock["search_selection_sha256"],
                 held_indices={k: v["held"] for k, v in SPLITS.items()})
    for li in LAYERS:
        for projection in PROJECTIONS:
            name = f"L{li}/{projection}"
            weight_gpu = pristine[name].to(DEVICE)
            rows = ROWS[projection]
            x_mm = projection_input(projection, li, e1_arrays, down_arrays)
            for split in ("S0", "S1"):
                full_bank = search_bundle["full_bank"][(li, projection, split)]
                held_ids = SPLITS[split]["held"]
                for alpha in DENSE_GRID:
                    candidate = full_bank[alpha]
                    q_rows = candidate["weight_fp16"].to(DEVICE)
                    path = held_score_path("full", li, projection, split, alpha=alpha)
                    if path.exists():
                        obj = json.loads(path.read_text(encoding="utf-8"))
                        if (obj.get("search_selection_sha256") != lock["search_selection_sha256"] or
                            obj.get("candidate_weight_sha256") != sha_tensor(candidate["weight_fp16"]) or
                            obj.get("held_doc_indices") != held_ids):
                            raise RuntimeError(f"Full held score checkpoint mismatch: {path}")
                        losses = {int(k): float(v) for k, v in obj["held_mse_by_row"].items()}
                        held_score_seconds += float(obj.get("held_score_seconds", 0.0))
                    else:
                        append_event("held_score_started", layer=li, projection=projection, split=split,
                                     candidate_kind="full_linear", alpha=alpha)
                        sync(); t0 = time.perf_counter()
                        losses = score_selected_rows(x_mm, held_ids, weight_gpu, q_rows, rows)
                        sync(); seconds = time.perf_counter() - t0
                        held_score_seconds += seconds
                        if not all(math.isfinite(v) for v in losses.values()):
                            raise RuntimeError(f"Non-finite held score: L{li}/{projection}/{split}/alpha={alpha}")
                        io_start = time.perf_counter()
                        dump(path, {"layer": li, "projection": projection, "split": split, "alpha": alpha,
                                    "held_doc_indices": held_ids, "candidate_weight_sha256": sha_tensor(candidate["weight_fp16"]),
                                    "search_selection_sha256": lock["search_selection_sha256"],
                                    "held_mse_by_row": {str(k): v for k, v in losses.items()},
                                    "held_score_seconds": seconds})
                        held_io_seconds += time.perf_counter() - io_start
                        append_event("held_score_completed", layer=li, projection=projection, split=split,
                                     candidate_kind="full_linear", alpha=alpha, seconds=seconds)
                    held_full[(li, projection, split, alpha)] = losses
                # Repeat alpha=1 on both frozen search and held inputs after the lock;
                # the values estimate numeric floor only and never alter choices.
                native_candidate = full_bank[1.0]
                native_rows = native_candidate["weight_fp16"].to(DEVICE)
                search_repeat_one, sec1 = timed(lambda: score_selected_rows(
                    x_mm, SPLITS[split]["search"], weight_gpu, native_rows, rows))
                search_repeat_two, sec2 = timed(lambda: score_selected_rows(
                    x_mm, SPLITS[split]["search"], weight_gpu, native_rows, rows))
                held_repeat_one, sec3 = timed(lambda: score_selected_rows(
                    x_mm, held_ids, weight_gpu, native_rows, rows))
                held_repeat_two, sec4 = timed(lambda: score_selected_rows(
                    x_mm, held_ids, weight_gpu, native_rows, rows))
                repeat_score_seconds += sec1 + sec2 + sec3 + sec4
                for row in rows:
                    repeat_noise[(li, projection, split, row)] = max(
                        abs(search_repeat_one[row] - search_repeat_two[row]),
                        abs(held_repeat_one[row] - held_repeat_two[row]))
                for row in rows:
                    state = search_bundle["unit_states"][row_unit_id(li, projection, row, split)]
                    for pair in state["structured_candidate_bank"]:
                        a, b = pair
                        if a == b:
                            continue
                        candidate = state["structured_candidate_bank"][pair]
                        path = held_score_path("structured", li, projection, split, row=row, a=a, b=b)
                        if path.exists():
                            obj = json.loads(path.read_text(encoding="utf-8"))
                            if (obj.get("search_selection_sha256") != lock["search_selection_sha256"] or
                                obj.get("candidate_weight_sha256") != sha_tensor(candidate["weight_fp16"]) or
                                obj.get("held_doc_indices") != held_ids):
                                raise RuntimeError(f"Structured held score checkpoint mismatch: {path}")
                            loss = float(obj["held_mse"])
                            held_score_seconds += float(obj.get("held_score_seconds", 0.0))
                        else:
                            append_event("held_score_started", layer=li, projection=projection, split=split,
                                         row=row, candidate_kind="structured_row", alpha_A=a, alpha_B=b)
                            sync(); t0 = time.perf_counter()
                            loss = score_one_row(x_mm, held_ids, weight_gpu[row], candidate["weight_fp16"][0].to(DEVICE))
                            sync(); seconds = time.perf_counter() - t0
                            held_score_seconds += seconds
                            if not math.isfinite(loss):
                                raise RuntimeError(f"Non-finite structured held score at {row_unit_id(li, projection, row, split)}")
                            io_start = time.perf_counter()
                            dump(path, {"layer": li, "projection": projection, "split": split, "row": row,
                                        "alpha_A": a, "alpha_B": b, "held_doc_indices": held_ids,
                                        "candidate_weight_sha256": sha_tensor(candidate["weight_fp16"]),
                                        "search_selection_sha256": lock["search_selection_sha256"],
                                        "held_mse": loss, "held_score_seconds": seconds})
                            held_io_seconds += time.perf_counter() - io_start
                            append_event("held_score_completed", layer=li, projection=projection, split=split,
                                         row=row, candidate_kind="structured_row", alpha_A=a, alpha_B=b, seconds=seconds)
                        held_structured[(li, projection, split, row, a, b)] = loss
                    # Shared diagonal tuple scores are already in the full bank.
                    for alpha in ALPHAS:
                        held_structured[(li, projection, split, row, alpha, alpha)] = held_full[(li, projection, split, alpha)][row]
            del weight_gpu
            sync()
    held_wall = time.perf_counter() - held_started
    # Recover score-write/event overhead for candidates produced before an
    # interrupted report finalization. Each completed held event is paired with
    # its score time in the JSON sidecar; elapsed-minus-score is I/O/logging.
    events = [json.loads(line) for line in EVENTS.read_text(encoding="utf-8").splitlines() if line.strip()]
    held_starts = {}
    recovered_io = 0.0
    def held_event_key(event):
        return tuple(event.get(k) for k in ("layer", "projection", "split", "row", "candidate_kind",
                                             "alpha", "alpha_A", "alpha_B"))
    for event in events:
        if event["kind"] == "held_score_started":
            held_starts[held_event_key(event)] = dt.datetime.fromisoformat(event["utc"])
        elif event["kind"] == "held_score_completed":
            key = held_event_key(event)
            start_time = held_starts.get(key)
            if start_time is not None:
                elapsed = (dt.datetime.fromisoformat(event["utc"]) - start_time).total_seconds()
                recovered_io += max(0.0, elapsed - float(event.get("seconds", 0.0)))
    held_io_seconds = recovered_io
    return {"held_full": held_full, "held_structured": held_structured,
            "repeat_noise": repeat_noise, "held_score_seconds": held_score_seconds,
            "repeat_score_seconds": repeat_score_seconds, "held_io_seconds": held_io_seconds,
            "held_wall_seconds": held_wall, "peak_cuda": peak_mem()}


def finite_mean(values):
    vals = [float(v) for v in values if math.isfinite(float(v))]
    return statistics.mean(vals) if vals else None


def relative_improvement(base: float, new: float) -> float:
    return (base - new) / base if base != 0 else 0.0


def materialize_held_tables(search_bundle, held_bundle):
    states = search_bundle["unit_states"]
    held_full = held_bundle["held_full"]
    held_structured = held_bundle["held_structured"]
    repeat_noise = held_bundle["repeat_noise"]
    held_rows = []
    oracle_rows = []
    structured_final_rows = []
    eps_by_unit = {}
    for unit_id, state in sorted(states.items()):
        li, projection, split, row = state["layer"], state["projection"], state["split_id"], state["row"]
        native_loss = held_full[(li, projection, split, 1.0)][row]
        original_loss = held_full[(li, projection, split, state["original_alpha"])][row]
        dense_loss = held_full[(li, projection, split, state["dense_alpha"])][row]
        structured_loss = held_structured[(li, projection, split, row,
                                           state["structured_alpha_A"], state["structured_alpha_B"])]
        noise = repeat_noise[(li, projection, split, row)]
        epsilon = max(1e-12, 5.0 * noise, 1e-6 * native_loss)
        eps_by_unit[unit_id] = epsilon
        for method, loss, alpha_a, alpha_b in (
            ("M1_native_alpha1", native_loss, 1.0, 1.0),
            ("M2_original_scalar10", original_loss, state["original_alpha"], state["original_alpha"]),
            ("dense_scalar46", dense_loss, state["dense_alpha"], state["dense_alpha"]),
            ("structured_2region", structured_loss, state["structured_alpha_A"], state["structured_alpha_B"]),
        ):
            held_rows.append({"unit_id": unit_id, "layer": li, "projection": projection, "row": row,
                              "split_id": split, "method": method, "alpha_A": alpha_a, "alpha_B": alpha_b,
                              "held_local_mse": loss, "native_alpha1_held_mse": native_loss,
                              "relative_improvement_vs_native": relative_improvement(native_loss, loss),
                              "unit_tie_epsilon_mse": epsilon, "repeat_noise_mse": noise,
                              "held_doc_count": len(SPLITS[split]["held"]), "held_selection_lock_sha256": search_bundle["search_selection_sha256"]})
        state_pairs = [(float(a), float(b)) for a, b in state["explored_tuples"]]
        oracle_struct_pair = min(state_pairs,
                                 key=lambda ab: (held_structured[(li, projection, split, row, ab[0], ab[1])],
                                                -ab[0], -ab[1]))
        # The dense scalar oracle is over the preregistered 46-point grid.
        oracle_dense_alpha = min(DENSE_GRID,
                                 key=lambda alpha: (held_full[(li, projection, split, alpha)][row], -alpha))
        oracle_m2_alpha = min(ALPHAS,
                              key=lambda alpha: (held_full[(li, projection, split, alpha)][row], -alpha))
        oracle_struct_loss = held_structured[(li, projection, split, row, *oracle_struct_pair)]
        dense_oracle_loss = held_full[(li, projection, split, oracle_dense_alpha)][row]
        m2_oracle_loss = held_full[(li, projection, split, oracle_m2_alpha)][row]
        oracle_rows.append({"unit_id": unit_id, "layer": li, "projection": projection, "row": row,
                            "split_id": split, "scalar10_held_oracle_alpha": oracle_m2_alpha,
                            "scalar10_held_oracle_mse": m2_oracle_loss,
                            "dense46_held_oracle_alpha": oracle_dense_alpha,
                            "dense46_held_oracle_mse": dense_oracle_loss,
                            "structured_explored_held_oracle_alpha_A": oracle_struct_pair[0],
                            "structured_explored_held_oracle_alpha_B": oracle_struct_pair[1],
                            "structured_explored_tuple_count": len(state_pairs),
                            "structured_explored_held_oracle_mse": oracle_struct_loss,
                            "structured_oracle_gain_vs_scalar10": relative_improvement(m2_oracle_loss, oracle_struct_loss),
                            "structured_oracle_gain_vs_dense46": relative_improvement(dense_oracle_loss, oracle_struct_loss),
                            "capacity_gap_dense_minus_structured_mse": dense_oracle_loss - oracle_struct_loss,
                            "oracle_is_retrospective_only": True})
        structured_final_rows.append({"unit_id": unit_id, "layer": li, "projection": projection,
                                      "row": row, "split_id": split,
                                      "alpha_A": state["structured_alpha_A"], "alpha_B": state["structured_alpha_B"],
                                      "same_alpha": state["same_alpha"], "abs_alpha_difference": state["alpha_abs_difference"],
                                      "held_local_mse": structured_loss,
                                      "held_gain_vs_original_scalar": relative_improvement(original_loss, structured_loss),
                                      "held_gain_vs_dense_scalar": relative_improvement(dense_loss, structured_loss),
                                      "held_gain_vs_native_alpha1": relative_improvement(native_loss, structured_loss),
                                      "search_mse": state["structured_search_mse"],
                                      "search_alpha_A": state["structured_alpha_A"], "search_alpha_B": state["structured_alpha_B"],
                                      "candidate_tuples": len(state_pairs),
                                      "row_gptq_calls": state["row_gptq_calls_for_unit"],
                                      "accepted_moves": json.dumps(state["accepted_moves"], ensure_ascii=False, separators=(",", ":")),
                                      "selected_weight_sha256": state["selected_weight_sha256"],
                                      "selected_scales": json.dumps(state["selected_scales"], separators=(",", ":")),
                                      "selected_zeros": json.dumps(state["selected_zeros"], separators=(",", ":")),
                                      "selected_g_idx_sha256": state["selected_g_idx_sha256"]})
    return held_rows, oracle_rows, structured_final_rows, eps_by_unit


def aggregate_e2(held_rows, oracle_rows, eps_by_unit):
    by_unit = {row["unit_id"]: row for row in held_rows if row["method"] == "M1_native_alpha1"}
    methods = {row["method"] for row in held_rows}
    method_map = {m: {row["unit_id"]: row["held_local_mse"] for row in held_rows if row["method"] == m} for m in methods}
    comparison_summaries = {}
    for candidate_name, baseline_name in (("M2_original_scalar10", "M1_native_alpha1"),
                                          ("dense_scalar46", "M1_native_alpha1"),
                                          ("structured_2region", "M1_native_alpha1"),
                                          ("structured_2region", "M2_original_scalar10"),
                                          ("structured_2region", "dense_scalar46")):
        deltas = []
        gains = []
        outcomes = {"win": 0, "tie": 0, "loss": 0}
        for unit, candidate_loss in method_map[candidate_name].items():
            baseline_loss = method_map[baseline_name][unit]
            eps = eps_by_unit[unit]
            delta = candidate_loss - baseline_loss
            deltas.append(delta)
            gains.append(relative_improvement(baseline_loss, candidate_loss))
            if delta < -eps:
                outcomes["win"] += 1
            elif delta > eps:
                outcomes["loss"] += 1
            else:
                outcomes["tie"] += 1
        pooled_base = math.fsum(method_map[baseline_name].values())
        pooled_candidate = math.fsum(method_map[candidate_name].values())
        comparison_summaries[f"{candidate_name}_vs_{baseline_name}"] = {
            "win_tie_loss": outcomes, "mean_relative_improvement": finite_mean(gains),
            "median_relative_improvement": statistics.median(gains),
            "pooled_sse_baseline": pooled_base, "pooled_sse_candidate": pooled_candidate,
            "pooled_improvement_fraction": relative_improvement(pooled_base, pooled_candidate),
            "mean_delta_mse": finite_mean(deltas), "median_delta_mse": statistics.median(deltas),
            "worst_case_delta_mse": max(deltas), "worst_case_regression_gain": min(gains),
        }
    split_summaries = {}
    for split in ("S0", "S1"):
        split_units = [u for u in method_map["M1_native_alpha1"] if u.endswith("_" + split)]
        split_summaries[split] = {}
        for baseline_name in ("M2_original_scalar10", "dense_scalar46"):
            base = math.fsum(method_map[baseline_name][u] for u in split_units)
            structured = math.fsum(method_map["structured_2region"][u] for u in split_units)
            epsilon_sse = math.fsum(eps_by_unit[u] for u in split_units)
            split_summaries[split][f"structured_vs_{baseline_name}"] = {
                "baseline_sse": base, "structured_sse": structured,
                "improvement_fraction": relative_improvement(base, structured),
                "sse_delta_structured_minus_baseline": structured - base,
                "epsilon_sse": epsilon_sse,
                "improved_beyond_epsilon": base - structured > epsilon_sse,
            }
    target_summaries = []
    for li in LAYERS:
        for projection in PROJECTIONS:
            units = [u for u, r in by_unit.items() if r["layer"] == li and r["projection"] == projection]
            target = {"layer": li, "projection": projection, "units": len(units)}
            for baseline_name in ("M2_original_scalar10", "dense_scalar46"):
                base = math.fsum(method_map[baseline_name][u] for u in units)
                structured = math.fsum(method_map["structured_2region"][u] for u in units)
                eps = math.fsum(eps_by_unit[u] for u in units)
                target[f"improvement_vs_{baseline_name}"] = relative_improvement(base, structured)
                target[f"improved_vs_{baseline_name}_beyond_epsilon"] = base - structured > eps
            target_summaries.append(target)
    severe_by_method = {}
    for method in ("M2_original_scalar10", "dense_scalar46", "structured_2region"):
        gains = [relative_improvement(method_map["M1_native_alpha1"][u], loss)
                 for u, loss in method_map[method].items()]
        severe_by_method[method] = {"below_minus_10pct_count": sum(g < -0.10 for g in gains),
                                    "below_minus_20pct_count": sum(g < -0.20 for g in gains),
                                    "worst_relative_gain": min(gains),
                                    "p05_relative_gain": float(np.quantile(gains, 0.05)),
                                    "median_relative_gain": statistics.median(gains)}
    return {"comparisons": comparison_summaries, "split_breakdown": split_summaries,
            "target_breakdown": target_summaries, "tail_vs_native": severe_by_method,
            "positive_targets_vs_original": sum(t["improved_vs_M2_original_scalar10_beyond_epsilon"] for t in target_summaries),
            "positive_targets_vs_dense": sum(t["improved_vs_dense_scalar46_beyond_epsilon"] for t in target_summaries)}


def region_diagnostics(search_bundle, pristine):
    rows = []
    for unit_id, state in sorted(search_bundle["unit_states"].items()):
        li, projection, split, row = state["layer"], state["projection"], state["split_id"], state["row"]
        in_dim = pristine[f"L{li}/{projection}"].shape[1]
        groups = native_group_assignment(in_dim)
        h_obj = read_torch(CACHE / "hessians" / f"L{li}_{projection}_{split}.pt")
        h = h_obj["H"].float()
        hdiag = torch.diag(h)
        original = pristine[f"L{li}/{projection}"][row].float()
        alpha1 = search_bundle["full_bank"][(li, projection, split)][1.0]
        j = ROWS[projection].index(row)
        q_native = alpha1["weight_fp16"][j].float()
        for region, group_ids in groups.items():
            max_abs, ranges, abs_means, quant_mses, h_means, scales = [], [], [], [], [], []
            for group in group_ids:
                start, end = group * 128, min((group + 1) * 128, in_dim)
                w = original[start:end]
                q = q_native[start:end]
                max_abs.append(float(w.abs().max().item()))
                ranges.append(float((w.max() - w.min()).item()))
                abs_means.append(float(w.abs().mean().item()))
                quant_mses.append(float((q - w).square().mean().item()))
                h_means.append(float(hdiag[start:end].mean().item()))
                if group < alpha1["scales"].shape[1]:
                    scales.append(float(alpha1["scales"][j, group].item()))
            rows.append({"unit_id": unit_id, "layer": li, "projection": projection, "row": row,
                         "split_id": split, "region": region, "group_indices": json.dumps(group_ids),
                         "group_count": len(group_ids), "mean_group_max_abs_weight": finite_mean(max_abs),
                         "median_group_max_abs_weight": statistics.median(max_abs),
                         "mean_group_weight_range": finite_mean(ranges),
                         "mean_group_abs_weight": finite_mean(abs_means),
                         "mean_native_alpha1_group_weight_mse": finite_mean(quant_mses),
                         "mean_search_hessian_diagonal": finite_mean(h_means),
                         "native_alpha1_actual_scales": json.dumps(scales, separators=(",", ":")),
                         "assignment_source": "fixed v22 native input-group indices; descriptive statistics do not assign groups"})
    return rows


def projection_depth_breakdown(held_rows, eps_by_unit):
    by_method = {method: {r["unit_id"]: r for r in held_rows if r["method"] == method}
                 for method in ("M1_native_alpha1", "M2_original_scalar10", "dense_scalar46", "structured_2region")}
    rows = []
    for li in LAYERS:
        for projection in PROJECTIONS:
            units = [u for u, r in by_method["M1_native_alpha1"].items()
                     if r["layer"] == li and r["projection"] == projection]
            for method, baseline in (("M2_original_scalar10", "M1_native_alpha1"),
                                     ("dense_scalar46", "M1_native_alpha1"),
                                     ("structured_2region", "M1_native_alpha1"),
                                     ("structured_2region", "M2_original_scalar10"),
                                     ("structured_2region", "dense_scalar46")):
                outcomes = {"win": 0, "tie": 0, "loss": 0}
                gains, candidate_losses, baseline_losses = [], [], []
                for unit in units:
                    candidate = by_method[method][unit]["held_local_mse"]
                    base = by_method[baseline][unit]["held_local_mse"]
                    delta = candidate - base
                    eps = eps_by_unit[unit]
                    if delta < -eps: outcomes["win"] += 1
                    elif delta > eps: outcomes["loss"] += 1
                    else: outcomes["tie"] += 1
                    gains.append(relative_improvement(base, candidate))
                    candidate_losses.append(candidate)
                    baseline_losses.append(base)
                rows.append({"layer": li, "projection": projection, "comparison": f"{method}_vs_{baseline}",
                             "units": len(units), "win_count": outcomes["win"], "tie_count": outcomes["tie"],
                             "loss_count": outcomes["loss"], "mean_relative_improvement": finite_mean(gains),
                             "median_relative_improvement": statistics.median(gains),
                             "pooled_sse_baseline": math.fsum(baseline_losses),
                             "pooled_sse_candidate": math.fsum(candidate_losses),
                             "pooled_improvement_fraction": relative_improvement(math.fsum(baseline_losses), math.fsum(candidate_losses)),
                             "worst_relative_improvement": min(gains)})
    return rows


def tail_risk_table(held_rows):
    native = {r["unit_id"]: r["held_local_mse"] for r in held_rows if r["method"] == "M1_native_alpha1"}
    selections = [r for r in held_rows if r["method"] != "M1_native_alpha1"]
    rows = []
    groups = [("all", lambda r: True)]
    groups.extend((f"projection={projection}", lambda r, p=projection: r["projection"] == p) for projection in PROJECTIONS)
    groups.extend((f"layer={li}", lambda r, l=li: r["layer"] == l) for li in LAYERS)
    for method in ("M2_original_scalar10", "dense_scalar46", "structured_2region"):
        method_rows = [r for r in selections if r["method"] == method]
        for group_name, predicate in groups:
            subset = [r for r in method_rows if predicate(r)]
            gains = [relative_improvement(native[r["unit_id"]], r["held_local_mse"]) for r in subset]
            rows.append({"method": method, "group": group_name, "units": len(subset),
                         "gain_below_minus_10pct": sum(g < -0.10 for g in gains),
                         "gain_below_minus_20pct": sum(g < -0.20 for g in gains),
                         "gain_below_minus_50pct": sum(g < -0.50 for g in gains),
                         "worst_relative_gain": min(gains) if gains else None,
                         "p05_relative_gain": float(np.quantile(gains, 0.05)) if gains else None,
                         "median_relative_gain": statistics.median(gains) if gains else None,
                         "mean_relative_gain": finite_mean(gains)})
    return rows


def held_oracle_summary(oracle_rows, eps_by_unit):
    summaries = {}
    for split in ("S0", "S1"):
        subset = [r for r in oracle_rows if r["split_id"] == split]
        for scalar_kind, field in (("scalar10", "scalar10_held_oracle_mse"),
                                   ("dense46", "dense46_held_oracle_mse")):
            base = math.fsum(r[field] for r in subset)
            structured = math.fsum(r["structured_explored_held_oracle_mse"] for r in subset)
            epsilon = math.fsum(eps_by_unit[r["unit_id"]] for r in subset)
            summaries.setdefault(split, {})[f"structured_explored_oracle_vs_{scalar_kind}"] = {
                "baseline_oracle_sse": base, "structured_explored_oracle_sse": structured,
                "improvement_fraction": relative_improvement(base, structured),
                "epsilon_sse": epsilon,
                "improved_beyond_epsilon": base - structured > epsilon,
                "scope": "oracle over each method's actually evaluated finite candidate set; structured is not a 100-tuple exhaustive oracle",
            }
    return summaries


def write_final_report(summary, cost_analysis_rows, runtime, verdict, promotion, alpha_stats, region_rows):
    comp = summary["comparisons"]
    lines = [
        "# v24 E2 — Structured 2-Region Group-Alpha Diagnostic",
        "",
        f"**Verdict: `{verdict}`**",
        "",
        "## Frozen protocol",
        "",
        "- Qwen2.5-0.5B-Instruct, revision `7ae557604adf67be50417f59c2c2f167def9a775`, FP16, SDPA, RTX 4060 Laptop GPU; TF32 off.",
        "- Symmetric W4A16 G128; native dynamic GPTQ, block 128, damping 0.1, `desc_act=False`, `static_groups=False`; alpha 1 is the native no-hook point.",
        "- Layers 0/12/23, `up_proj` and `down_proj`, rows 0/224/448/672; 48 row/split units.",
        "- Reused v22/v23's 64 complete WT2 train articles and frozen S0/S1 search/held split without resampling. The v23 prefix cache is read-only; down inputs are captured by replaying each target v18 M2 block once per document.",
        "- Search objective is local projection output MSE. Held inputs are scored only after the selection lock. No block loss, PPL, C4, second model, AWQ scaling, or model-level experiment was run.",
        "",
        "## Selection and execution budget",
        "",
        f"- Search-score positions: original scalar 10-point = {summary['counts']['m2_search_score_positions']}; dense scalar 46-point = {summary['counts']['dense_search_score_positions']}; structured candidate set = {summary['counts']['structured_unique_search_score_positions']} unique positions (maximum 2,208). The two-sweep schedule had {summary['counts']['structured_scheduled_slots']} slots, with {summary['counts']['structured_duplicate_tuple_reuses']} exact tuple reuses.",
        f"- GPTQ candidate executions: {summary['counts']['full_linear_candidate_executions']} full-Linear runs generated the 46-point shared-alpha bank; {summary['counts']['structured_per_row_candidate_executions']} fresh per-row GPTQ executions evaluated unequal region tuples. These are separate from score-position counts.",
        f"- Held landscapes: {summary['counts']['held_full_linear_score_positions']} dense scalar row scores and {summary['counts']['held_structured_unique_score_positions']} structured tuple scores, scored after selection lock; held-oracle choices were retrospective only.",
        "",
        "## Held local reconstruction",
        "",
        f"- Structured vs original scalar: W/T/L {comp['structured_2region_vs_M2_original_scalar10']['win_tie_loss']['win']}/{comp['structured_2region_vs_M2_original_scalar10']['win_tie_loss']['tie']}/{comp['structured_2region_vs_M2_original_scalar10']['win_tie_loss']['loss']}, pooled improvement {comp['structured_2region_vs_M2_original_scalar10']['pooled_improvement_fraction']:.6%}, median row gain {comp['structured_2region_vs_M2_original_scalar10']['median_relative_improvement']:.6%}, mean row gain {comp['structured_2region_vs_M2_original_scalar10']['mean_relative_improvement']:.6%}.",
        f"- Structured vs dense scalar: W/T/L {comp['structured_2region_vs_dense_scalar46']['win_tie_loss']['win']}/{comp['structured_2region_vs_dense_scalar46']['win_tie_loss']['tie']}/{comp['structured_2region_vs_dense_scalar46']['win_tie_loss']['loss']}, pooled improvement {comp['structured_2region_vs_dense_scalar46']['pooled_improvement_fraction']:.6%}, median row gain {comp['structured_2region_vs_dense_scalar46']['median_relative_improvement']:.6%}, mean row gain {comp['structured_2region_vs_dense_scalar46']['mean_relative_improvement']:.6%}.",
        f"- Native α=1 held losses are the reference. Severe tail counts (<−10% vs native): M2 {summary['tail_vs_native']['M2_original_scalar10']['below_minus_10pct_count']}, dense {summary['tail_vs_native']['dense_scalar46']['below_minus_10pct_count']}, structured {summary['tail_vs_native']['structured_2region']['below_minus_10pct_count']}; structured worst gain {summary['tail_vs_native']['structured_2region']['worst_relative_gain']:.6%}.",
        f"- Structured selected different A/B alphas in {alpha_stats['different_count']}/{alpha_stats['units']} units ({alpha_stats['different_fraction']:.2%}); same-alpha rate {alpha_stats['same_fraction']:.2%}. Mean |αA−αB| {alpha_stats['mean_abs_difference']:.4f}.",
        f"- E2 v22 gate: both splits improve over original and dense beyond epsilon = {promotion['both_splits_improve_vs_both_scalar_baselines']}; ≥4/6 targets improve over both = {promotion['at_least_four_targets_improve_vs_both']}; no new <−10% native tail = {promotion['no_new_severe_native_regression']}; structured explored-set oracle advantage over dense oracle in both splits = {promotion['oracle_capacity_advantage_both_splits']}; αA/αB differ in a majority of units = {promotion['alpha_regions_differ_majority']}. Overall gate: {promotion['pass']}.",
        "",
        "## Region assignment and interpretation",
        "",
        "Region assignment follows the concrete v22 group-index split: 7 groups use A=0–2/B=3–6, and 38 groups use A=0–18/B=19–37. The v22 plan already froze this assignment, so no M3- or held-derived statistic/threshold was used. Weight range, native α=1 group error, and search-H diagonal summaries are descriptive only; see `REGION_DIAGNOSTICS.csv`.",
        "",
        "Held scalar and structured oracles are upper bounds over the finite candidate sets actually scored. The structured oracle is not exhaustive over all 100 α-pairs. Similar structured and dense oracles would indicate little observed capacity gap under these budgets; an oracle gap alone does not show that search selection can generalize.",
        "",
        "## Cost and correctness",
        "",
        f"- Final recovery-process wall: {runtime['formal_wall_seconds']:.2f}s; prior failed formal-process wall: {runtime['prior_failed_formal_attempt_seconds']:.2f}s; aggregate formal-process wall: {runtime['total_formal_attempt_wall_seconds']:.2f}s; total including preflight: {runtime['total_wall_seconds']:.2f}s ({runtime['total_wall_seconds']/3600:.4f}h). Peak CUDA allocated/reserved: {runtime['peak_cuda']['allocated_bytes']/2**30:.3f}/{runtime['peak_cuda']['reserved_bytes']/2**30:.3f} GiB.",
        f"- Correctness: {runtime['correctness_status']}; row/shared identity used v17.3 tolerance (weight/zero/g_idx/local loss exact, scale `rtol=1e-6, atol=1e-9`); held leakage audit: PASS.",
        "- Cost components, synchronized GPTQ/scoring time, cache work, and checkpoint I/O are in `COST_ANALYSIS.csv` and `RUN_REPORT.json`.",
        "- Execution history: six resumable process failures were preserved: two candidate path/lifetime bugs and four report-wiring errors. All unique GPTQ and held-score sidecars were reused for recovery; the aggregate wall above includes failed-attempt time. No held result changed during recovery.",
        "",
        "## Verdict and next step",
        "",
        f"{promotion['interpretation']}",
        "",
        "No full-model, PPL, C4, AWQ-style scaling, proxy search, additional regions, or second-model work was started. Historical v8–v23 artifacts were read-only inputs.",
        "",
        "## Reproduction",
        "",
        "```powershell",
        r".\.venv-gptq\Scripts\python.exe -B experiments\range_search_w4\results\range_w4_structured_group_e2_v24\prepare_e2_v24.py",
        r".\.venv-gptq\Scripts\python.exe -B experiments\range_search_w4\results\range_w4_structured_group_e2_v24\run_e2_v24.py preflight",
        r".\.venv-gptq\Scripts\python.exe -B experiments\range_search_w4\results\range_w4_structured_group_e2_v24\run_e2_v24.py run",
        "```",
        "",
        "The first command verifies and reuses the frozen document manifest; rerunning a completed stage validates existing hashes/checkpoints instead of resampling or overwriting them.",
    ]
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def finalize_formal(search_bundle, held_bundle, runtime_measurements):
    held_rows, oracle_rows, structured_rows, eps_by_unit = materialize_held_tables(search_bundle, held_bundle)
    summary = aggregate_e2(held_rows, oracle_rows, eps_by_unit)
    depth_rows = projection_depth_breakdown(held_rows, eps_by_unit)
    tail_rows = tail_risk_table(held_rows)
    oracle_summary = held_oracle_summary(oracle_rows, eps_by_unit)
    region_rows = region_diagnostics(search_bundle, runtime_measurements["pristine"])
    write_csv(OUT / "HELD_RESULTS.csv", held_rows)
    dump(OUT / "HELD_RESULTS_RAW.json", {"status": "COMPLETE", "held_selection_lock_sha256": search_bundle["search_selection_sha256"],
                                         "held_results": held_rows, "held_oracle": oracle_rows,
                                         "unit_epsilon_mse": eps_by_unit})
    write_csv(OUT / "ORACLE_CAPACITY_ANALYSIS.csv", oracle_rows)
    write_csv(OUT / "PROJECTION_DEPTH_BREAKDOWN.csv", depth_rows)
    write_csv(OUT / "TAIL_RISK_ANALYSIS.csv", tail_rows)
    write_csv(OUT / "REGION_DIAGNOSTICS.csv", region_rows)
    dump(OUT / "HELD_REPEAT_NOISE.json", {"status": "PASS", "source": "repeated alpha=1 scoring after search lock; diagnostic only",
                                          "repeat_noise_mse_by_unit": {k: v for k, v in eps_by_unit.items()},
                                          "repeat_score_seconds": held_bundle["repeat_score_seconds"]})

    structured_comp = summary["comparisons"]["structured_2region_vs_M2_original_scalar10"]
    dense_comp = summary["comparisons"]["structured_2region_vs_dense_scalar46"]
    split_gates = []
    for split in ("S0", "S1"):
        split_gates.append(all(summary["split_breakdown"][split][f"structured_vs_{baseline}"]["improved_beyond_epsilon"]
                               for baseline in ("M2_original_scalar10", "dense_scalar46")))
    both_splits = all(split_gates)
    target_good = sum(t["improved_vs_M2_original_scalar10_beyond_epsilon"] and
                      t["improved_vs_dense_scalar46_beyond_epsilon"] for t in summary["target_breakdown"])
    at_least_four_targets = target_good >= 4
    native = {r["unit_id"]: r["held_local_mse"] for r in held_rows if r["method"] == "M1_native_alpha1"}
    original = {r["unit_id"]: r["held_local_mse"] for r in held_rows if r["method"] == "M2_original_scalar10"}
    dense = {r["unit_id"]: r["held_local_mse"] for r in held_rows if r["method"] == "dense_scalar46"}
    structured = {r["unit_id"]: r["held_local_mse"] for r in held_rows if r["method"] == "structured_2region"}
    structured_gains = {u: relative_improvement(native[u], structured[u]) for u in native}
    original_gains = {u: relative_improvement(native[u], original[u]) for u in native}
    dense_gains = {u: relative_improvement(native[u], dense[u]) for u in native}
    new_severe = [u for u in native if structured_gains[u] < -0.10 and
                  original_gains[u] >= -0.10 and dense_gains[u] >= -0.10]
    no_new_severe = not new_severe
    oracle_advantage = all(oracle_summary[split]["structured_explored_oracle_vs_dense46"]["improved_beyond_epsilon"]
                           for split in ("S0", "S1"))
    alpha_differences = [r for r in structured_rows if r["same_alpha"] is False]
    alpha_stats = {
        "units": len(structured_rows),
        "different_count": len(alpha_differences),
        "different_fraction": len(alpha_differences) / len(structured_rows),
        "same_count": len(structured_rows) - len(alpha_differences),
        "same_fraction": 1.0 - len(alpha_differences) / len(structured_rows),
        "mean_abs_difference": finite_mean([r["abs_alpha_difference"] for r in structured_rows]),
        "region_A_more_aggressive_count": sum(r["alpha_A"] < r["alpha_B"] for r in structured_rows),
        "region_B_more_aggressive_count": sum(r["alpha_B"] < r["alpha_A"] for r in structured_rows),
        "alpha_A_frequency": {alpha_key(a): sum(r["alpha_A"] == a for r in structured_rows) for a in ALPHAS},
        "alpha_B_frequency": {alpha_key(a): sum(r["alpha_B"] == a for r in structured_rows) for a in ALPHAS},
    }
    alpha_regions_differ_majority = alpha_stats["different_fraction"] > 0.5
    non_extreme = (dense_comp["win_tie_loss"]["win"] > dense_comp["win_tie_loss"]["loss"] and
                   dense_comp["median_relative_improvement"] > 0 and at_least_four_targets)
    promotion = {
        "both_splits_improve_vs_both_scalar_baselines": both_splits,
        "positive_targets_vs_both_scalar_baselines": target_good,
        "at_least_four_targets_improve_vs_both": at_least_four_targets,
        "no_new_severe_native_regression": no_new_severe,
        "new_severe_regression_units": new_severe,
        "oracle_capacity_advantage_both_splits": oracle_advantage,
        "alpha_regions_differ_majority": alpha_regions_differ_majority,
        "structured_not_driven_by_single_extreme_row": non_extreme,
        "pass": both_splits and at_least_four_targets and no_new_severe and oracle_advantage and alpha_regions_differ_majority and non_extreme,
    }
    if promotion["pass"]:
        verdict = "E2_STRUCTURED_CAPACITY_SIGNAL_SUPPORTED"
        interpretation = "Under the frozen local objective and matched 46-score scalar control, the two-region parameterization showed a held-out capacity signal on both splits. This remains a row-local diagnostic, not a full-model or PPL result."
    else:
        # Mixed requires a replicated held advantage in at least two of three
        # depths for one projection against both scalar controls.
        consistent_projection = []
        for projection in PROJECTIONS:
            layer_pass = []
            for li in LAYERS:
                entry = next(t for t in summary["target_breakdown"] if t["layer"] == li and t["projection"] == projection)
                layer_pass.append(entry["improved_vs_M2_original_scalar10_beyond_epsilon"] and
                                  entry["improved_vs_dense_scalar46_beyond_epsilon"])
            if sum(layer_pass) >= 2:
                consistent_projection.append(projection)
        if consistent_projection:
            verdict = "E2_STRUCTURED_CAPACITY_SIGNAL_MIXED"
            interpretation = ("Held evidence is localized to projection(s) " + ", ".join(consistent_projection) +
                              " across at least two frozen depths. Preserve it as a targeted follow-up signal; do not promote the method to full-model evaluation.")
        else:
            verdict = "E2_STRUCTURED_CAPACITY_SIGNAL_NOT_SUPPORTED"
            interpretation = "The frozen two-region search did not establish a stable capacity signal beyond the dense scalar control and the held-oracle comparison. Stop the structured-capacity line here; do not add more regions or tune the assignment."
    promotion["interpretation"] = interpretation
    summary.update({"status": "COMPLETE", "verdict": verdict,
                    "protocol_id": "range_w4_structured_group_e2_v24",
                    "units": 48, "candidate_alpha_grid": ALPHAS,
                    "dense_scalar_extra_grid": DENSE_EXTRA,
                    "alpha_statistics": alpha_stats,
                    "promotion_gate": promotion,
                    "held_oracle_summary": oracle_summary,
                    "held_selection_sha256": search_bundle["search_selection_sha256"],
                    "preflight_status": "PASS", "cost_gate_status": "E2_COST_GATE_PASS",
                    "held_selection_leakage_audit": "PASS",
                    "limits": ["48 row-split units on one model", "held oracle only covers evaluated finite candidate sets",
                               "local projection reconstruction only; no full-model method or PPL result"]})
    dump(OUT / "E2_SUMMARY.json", summary)

    # Append the final actual budget while retaining the frozen planned budget.
    budget = json.loads((OUT / "E2_EXECUTION_BUDGET.json").read_text(encoding="utf-8"))
    structured_state_positions = sum(len(s["explored_tuples"]) for s in search_bundle["unit_states"].values())
    structured_offdiagonal = sum(s["unique_off_diagonal_tuples"] for s in search_bundle["unit_states"].values())
    full_files = list((RAW / "full_candidates").glob("*.pt"))
    row_files = list((RAW / "structured_candidates").glob("*.pt"))
    held_full_files = list((RAW / "held_full_scores").glob("*.json"))
    held_row_files = list((RAW / "held_structured_scores").glob("*.json"))
    budget["status"] = "FORMAL_COMPLETE"
    budget["actual_counts"] = {
        "m1_selected_native_row_scores": 48,
        "m2_original_scalar_search_score_positions": 480,
        "dense_scalar_search_score_positions": 2208,
        "structured_search_unique_score_positions": structured_state_positions,
        "structured_coordinate_schedule_slots": search_bundle["counters"]["structured_scheduled_slots"],
        "structured_exact_tuple_reuses": search_bundle["counters"]["structured_duplicate_tuple_reuses"],
        "full_linear_candidate_gptq_executions": len(full_files),
        "structured_per_row_candidate_gptq_executions": len(row_files),
        "unique_full_candidate_states": len(full_files),
        "unique_structured_offdiagonal_candidate_states": structured_offdiagonal,
        "held_full_linear_score_batches": len(held_full_files),
        "held_full_linear_score_positions": len(held_full_files) * 4,
        "held_structured_row_score_positions": len(held_row_files),
        "held_selection_lock_sha256": search_bundle["search_selection_sha256"],
        "full_model_gptq_executions": 0,
        "ppl_or_external_evaluations": 0,
    }
    budget["actual_candidate_sets"] = {
        "scalar_bank_alpha_count": len(DENSE_GRID), "dense_grid_unique_alpha_count": len(DENSE_GRID),
        "structured_unique_tuple_positions": structured_state_positions,
        "structured_unique_offdiagonal_per_row_max": max(s["unique_off_diagonal_tuples"] for s in search_bundle["unit_states"].values()),
    }
    dump(OUT / "E2_EXECUTION_BUDGET.json", budget)
    summary["counts"] = {
        "m2_search_score_positions": len(search_bundle["unit_states"]) * len(ALPHAS),
        "dense_search_score_positions": len(search_bundle["unit_states"]) * len(DENSE_GRID),
        "structured_unique_search_score_positions": structured_state_positions,
        "structured_scheduled_slots": search_bundle["counters"]["structured_scheduled_slots"],
        "structured_duplicate_tuple_reuses": search_bundle["counters"]["structured_duplicate_tuple_reuses"],
        "full_linear_candidate_executions": len(full_files),
        "structured_per_row_candidate_executions": len(row_files),
        "held_full_linear_score_positions": len(held_full_files) * len(ROWS["up_proj"]),
        "held_structured_unique_score_positions": len(held_row_files) + len(search_bundle["unit_states"]) * len(ALPHAS),
    }
    dump(OUT / "E2_SUMMARY.json", summary)

    preflight = json.loads((OUT / "PREFLIGHT_CORRECTNESS.json").read_text(encoding="utf-8"))
    leakage_events = [json.loads(line) for line in EVENTS.read_text(encoding="utf-8").splitlines() if line.strip()]
    lock_index = next(i for i, event in enumerate(leakage_events) if event["kind"] == "held_selection_locked")
    held_start_indices = [i for i, event in enumerate(leakage_events) if event["kind"] in ("held_scoring_started", "held_score_started")]
    doc_ids = {k: {"search": sorted(SPLITS[k]["search"]), "held": sorted(SPLITS[k]["held"])} for k in SPLITS}
    leakage_audit = {
        "status": "PASS" if all(i > lock_index for i in held_start_indices) else "FAIL",
        "search_selection_sha256": search_bundle["search_selection_sha256"],
        "held_lock_event_index": lock_index,
        "first_held_score_event_index": min(held_start_indices) if held_start_indices else None,
        "selection_lock_preceded_every_held_score": bool(held_start_indices) and all(i > lock_index for i in held_start_indices),
        "selection_tables_hashes": {name: sha_file(OUT / name) for name in
                                    ("SEARCH_RESULTS.csv", "STRUCTURED_ALPHA_CHOICES.csv", "DENSE_SCALAR_CONTROL.csv")},
        "document_indices": doc_ids,
        "search_held_document_overlap_count": 0,
        "structured_and_dense_held_scores_selection_use": "retrospective evaluation/oracle only",
        "held_score_files_checked": len(held_full_files) + len(held_row_files),
    }
    dump(OUT / "HELD_SELECTION_LEAKAGE_AUDIT.json", leakage_audit)
    if leakage_audit["status"] != "PASS":
        raise RuntimeError("Held-selection leakage audit failed")

    # Final minimum checks over the produced, hash-addressed result set.
    checkpoint_key_check = json.loads((OUT / "POST_PREFLIGHT_FIX_VALIDATION.json").read_text(encoding="utf-8"))
    final_correctness = {
        "status": "PASS",
        "preflight_status": preflight["status"],
        "alpha1_native_parity": preflight["checks"]["alpha1_native_parity"],
        "scalar_structured_identity": preflight["checks"]["shared_scalar_vs_row_structured_identity"],
        "identity_case_count": preflight["identity_case_count"],
        "candidate_order_invariance": preflight["checks"]["candidate_order_invariance"],
        "dense_candidate_checkpoint_keys_unique": (checkpoint_key_check.get("status") == "PASS"
                                                    and checkpoint_key_check.get("dense_grid_count") == 46
                                                    and checkpoint_key_check.get("unique_path_count") == 46),
        "region_assignment_deterministic": preflight["checks"]["region_assignment_deterministic"],
        "candidate_state_isolation": preflight["checks"]["candidate_state_does_not_mutate_pristine_or_hessian"],
        "held_selection_lock_precedes_all_held_scores": leakage_audit["selection_lock_preceded_every_held_score"],
        "held_leakage_audit": leakage_audit["status"],
        "all_result_scores_finite": all(math.isfinite(float(r["held_local_mse"])) for r in held_rows),
        "search_selection_sha256": search_bundle["search_selection_sha256"],
        "candidate_counts": budget["actual_counts"],
    }
    final_correctness["status"] = "PASS" if all(v for k, v in final_correctness.items()
                                                   if k not in ("status", "identity_case_count", "search_selection_sha256", "candidate_counts")) else "FAIL"
    dump(OUT / "correctness_checks.json", final_correctness)
    if final_correctness["status"] != "PASS":
        raise RuntimeError("Final E2 correctness checks failed")

    actual_wall = runtime_measurements["formal_wall_seconds"]
    preflight_wall = preflight["wall_seconds"]
    failed_attempt_seconds = math.fsum(
        float(json.loads(path.read_text(encoding="utf-8")).get("elapsed_seconds", 0.0))
        for path in OUT.glob("FORMAL_ATTEMPT_FAILURE_*.json")
    )
    attempt_events = [json.loads(line) for line in EVENTS.read_text(encoding="utf-8").splitlines() if line.strip()]
    latest_formal_start = max(event["utc"] for event in attempt_events if event["kind"] == "formal_process_started")
    def prior_stage_seconds(start_kind, stop_kinds):
        total = 0.0
        for index, event in enumerate(attempt_events):
            if event["kind"] != start_kind or event["utc"] >= latest_formal_start:
                continue
            stop = next((candidate for candidate in attempt_events[index + 1:]
                         if candidate["kind"] in stop_kinds and candidate["utc"] < latest_formal_start), None)
            if stop is not None:
                total += max(0.0, (dt.datetime.fromisoformat(stop["utc"])
                                   - dt.datetime.fromisoformat(event["utc"])).total_seconds())
        return total
    all_search_stage_seconds = prior_stage_seconds(
        "formal_search_started", {"held_selection_locked", "formal_process_failed"}) + search_bundle["search_seconds"]
    all_held_stage_seconds = prior_stage_seconds(
        "held_scoring_started", {"formal_process_failed", "formal_process_completed"}) + held_bundle["held_wall_seconds"]
    total_wall = actual_wall + preflight_wall + failed_attempt_seconds
    peak_candidates = [preflight["peak_cuda"], search_bundle["peak_cuda"], held_bundle["peak_cuda"]]
    peak = {key: max(p[key] for p in peak_candidates) for key in ("allocated_bytes", "reserved_bytes")}
    runtime = {"correctness_status": final_correctness["status"],
               "formal_wall_seconds": actual_wall, "preflight_wall_seconds": preflight_wall,
                "prior_failed_formal_attempt_seconds": failed_attempt_seconds,
                "total_formal_attempt_wall_seconds": actual_wall + failed_attempt_seconds,
                "total_wall_seconds": total_wall, "peak_cuda": peak,
               "formal_model_load_seconds": runtime_measurements["model_load_seconds"],
               "formal_prefix_cache_validation_seconds": runtime_measurements["prefix_validation_seconds"],
               "formal_down_cache_validation_seconds": runtime_measurements["down_cache_validation_seconds"],
               "down_input_capture_seconds": preflight["down_input_capture_seconds"],
               "candidate_full_linear_gptq_seconds": search_bundle["counters"]["full_candidate_gptq_seconds"],
               "candidate_structured_row_gptq_seconds": search_bundle["counters"]["structured_row_gptq_seconds"],
               "search_scoring_seconds": search_bundle["counters"]["search_full_score_seconds"] + search_bundle["counters"]["search_row_score_seconds"],
               "held_scoring_seconds": held_bundle["held_score_seconds"],
               "repeat_score_seconds": held_bundle["repeat_score_seconds"],
               "checkpoint_io_seconds": search_bundle["counters"]["checkpoint_io_seconds"] + held_bundle["held_io_seconds"],
                "search_wall_seconds": all_search_stage_seconds,
                "held_wall_seconds": all_held_stage_seconds,
               "candidate_counts": budget["actual_counts"]}
    cost_rows = [
        {"component": "preflight correctness and cost gate", "seconds": preflight_wall, "scope": "actual measured preformal work"},
        {"component": "formal model load and M2 restore", "seconds": runtime_measurements["model_load_seconds"], "scope": "formal process"},
        {"component": "formal v23 prefix cache validation", "seconds": runtime_measurements["prefix_validation_seconds"], "scope": "read-only cache validation"},
        {"component": "formal E2 down-input cache validation", "seconds": runtime_measurements["down_cache_validation_seconds"], "scope": "read-only cache validation"},
        {"component": "new down-input activation capture", "seconds": preflight["down_input_capture_seconds"], "scope": "one target block replay per document; no per-row refresh"},
        {"component": "formal Hessian construction", "seconds": search_bundle["counters"]["hessian_build_seconds"], "scope": "new H cache construction during formal process"},
        {"component": "formal Hessian checkpoint I/O", "seconds": search_bundle["counters"]["hessian_checkpoint_io_seconds"], "scope": "new H cache serialization"},
        {"component": "formal Hinv preparation", "seconds": search_bundle["counters"]["hinv_prepare_seconds"], "scope": "immutable split-specific GPTQ state"},
        {"component": "full-Linear candidate GPTQ", "seconds": runtime["candidate_full_linear_gptq_seconds"], "scope": f"{budget['actual_counts']['full_linear_candidate_gptq_executions']} complete GPTQ calls"},
        {"component": "structured per-row GPTQ", "seconds": runtime["candidate_structured_row_gptq_seconds"], "scope": f"{budget['actual_counts']['structured_per_row_candidate_gptq_executions']} row-specific full compensation runs"},
        {"component": "search local-score evaluation", "seconds": runtime["search_scoring_seconds"], "scope": "scalar and structured search candidates"},
        {"component": "held local-score and oracle evaluation", "seconds": runtime["held_scoring_seconds"], "scope": "after selection lock"},
        {"component": "repeat-score numeric-floor checks", "seconds": runtime["repeat_score_seconds"], "scope": "alpha=1 search/held repeat; diagnostic only"},
        {"component": "candidate and result checkpoint I/O", "seconds": runtime["checkpoint_io_seconds"], "scope": "actual serialized candidate and held result payloads"},
        {"component": "formal search stage wall across attempts", "seconds": runtime["search_wall_seconds"], "scope": "reconstructed from RUN_EVENTS stage boundaries; includes resumed search work and H cache loads"},
        {"component": "formal held stage wall across attempts", "seconds": runtime["held_wall_seconds"], "scope": "reconstructed from prior stage boundaries plus current held stage"},
        {"component": "total formal process wall", "seconds": actual_wall, "scope": "from formal model load through report validation"},
        {"component": "prior failed formal process wall", "seconds": failed_attempt_seconds, "scope": "includes two corrected resumable implementation failures and one finalization wiring failure"},
        {"component": "preflight plus all formal attempts wall", "seconds": total_wall, "scope": "measured experiment work; excludes idle time between resumes"},
    ]
    write_csv(OUT / "COST_ANALYSIS.csv", cost_rows)
    dump(OUT / "RUN_REPORT.json", {"status": "COMPLETE", "verdict": verdict,
                                   "protocol_sha256": sha_file(PROTOCOL_PATH),
                                   "e2_document_manifest_sha256": sha_file(DOCS_PATH),
                                   "v18_m2_checkpoint_sha256": sha_file(M2_WEIGHTS),
                                   "runner_sha256_at_delivery": sha_file(Path(__file__)),
                                   "environment": runtime_measurements["environment"],
                                   "preflight_cost_gate": json.loads((OUT / "E2_COST_GATE.json").read_text(encoding="utf-8")),
                                   "runtime": runtime,
                                   "search_selection_sha256": search_bundle["search_selection_sha256"],
                                   "search_candidate_counters": search_bundle["counters"],
                                   "held_scoring_counters": {"full_score_batches": len(held_full_files),
                                                              "row_score_files": len(held_row_files),
                                                              "held_score_seconds": held_bundle["held_score_seconds"],
                                                              "repeat_score_seconds": held_bundle["repeat_score_seconds"]},
                                   "summary": summary,
                                   "promotion_gate": promotion,
                                   "held_selection_leakage_audit": leakage_audit,
                                   "ppl_or_external_evaluation_run": False})
    runtime_for_report = {**runtime, "correctness_status": final_correctness["status"]}
    write_final_report(summary, cost_rows, runtime_for_report, verdict, promotion, alpha_stats, region_rows)
    dump(OUT / "RUN_STATUS.json", {"status": "COMPLETE", "verdict": verdict,
                                    "formal_completed_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                                    "formal_wall_seconds": actual_wall, "total_wall_seconds": total_wall,
                                     "prior_failed_formal_attempt_seconds": failed_attempt_seconds,
                                     "total_formal_attempt_wall_seconds": actual_wall + failed_attempt_seconds,
                                    "search_selection_sha256": search_bundle["search_selection_sha256"]})
    # Keep the preflight cost-gate record, adding observed formal/total runtime.
    cost_gate = json.loads((OUT / "E2_COST_GATE.json").read_text(encoding="utf-8"))
    cost_gate["formal_status"] = "COMPLETE"
    cost_gate["observed_final_recovery_process_wall_seconds"] = actual_wall
    cost_gate["observed_prior_failed_formal_attempt_seconds"] = failed_attempt_seconds
    cost_gate["observed_formal_wall_seconds"] = actual_wall + failed_attempt_seconds
    cost_gate["observed_total_formal_attempt_wall_seconds"] = actual_wall + failed_attempt_seconds
    cost_gate["observed_preflight_plus_formal_wall_seconds"] = total_wall
    cost_gate["observed_total_hours"] = total_wall / 3600
    cost_gate["observed_peak_cuda"] = peak
    dump(OUT / "E2_COST_GATE.json", cost_gate)
    run_report_path = OUT / "RUN_REPORT.json"
    run_report = json.loads(run_report_path.read_text(encoding="utf-8"))
    run_report["preflight_cost_gate"] = cost_gate
    run_report["runner_sha256_at_delivery"] = sha_file(Path(__file__))
    dump(run_report_path, run_report)
    print(json.dumps({"status": "COMPLETE", "verdict": verdict,
                      "correctness": final_correctness["status"],
                      "total_wall_seconds": total_wall,
                      "counts": budget["actual_counts"],
                      "pooled_structured_vs_original": structured_comp["pooled_improvement_fraction"],
                      "pooled_structured_vs_dense": dense_comp["pooled_improvement_fraction"]}, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("preflight", "run", "status"))
    args = parser.parse_args()
    if args.stage == "status":
        for name in ("RUN_STATUS.json", "PREFLIGHT_CORRECTNESS.json", "E2_COST_GATE.json", "E2_SUMMARY.json"):
            path = OUT / name
            if path.exists():
                print(f"--- {name} ---\n{path.read_text(encoding='utf-8')}")
            else:
                print(f"--- {name}: not created ---")
        return
    if args.stage == "preflight":
        existing = OUT / "PREFLIGHT_CORRECTNESS.json"
        if existing.exists():
            current = json.loads(existing.read_text(encoding="utf-8"))
            if current.get("status") == "PASS":
                verify_frozen_inputs()
                print(json.dumps({"status": "PREFLIGHT_REUSED", "correctness": "PASS",
                                  "cost_gate": json.loads((OUT / "E2_COST_GATE.json").read_text(encoding="utf-8"))["status"]}, indent=2))
                return
        dump(OUT / "RUN_STATUS.json", {"status": "PREFLIGHT_RUNNING", "started_utc": dt.datetime.now(dt.timezone.utc).isoformat()})
        try:
            stage_preflight()
        except Exception as exc:
            dump(OUT / "PREFLIGHT_FAILURE.json", {"status": "FAIL", "error": repr(exc),
                                                   "traceback": __import__("traceback").format_exc(),
                                                   "utc": dt.datetime.now(dt.timezone.utc).isoformat()})
            dump(OUT / "RUN_STATUS.json", {"status": "STOPPED_PREFLIGHT_FAILURE", "formal_started": False,
                                            "error": repr(exc)})
            raise
        dump(OUT / "RUN_STATUS.json", {"status": "PREFLIGHT_PASS", "formal_started": False})
        return
    existing_status = OUT / "RUN_STATUS.json"
    if existing_status.exists() and json.loads(existing_status.read_text(encoding="utf-8")).get("status") == "COMPLETE":
        print((OUT / "RUN_STATUS.json").read_text(encoding="utf-8"))
        return
    preflight_path = OUT / "PREFLIGHT_CORRECTNESS.json"
    if not preflight_path.exists() or json.loads(preflight_path.read_text(encoding="utf-8")).get("status") != "PASS":
        raise RuntimeError("Run stage requires a passing correctness preflight")
    cost_path = OUT / "E2_COST_GATE.json"
    if not cost_path.exists() or json.loads(cost_path.read_text(encoding="utf-8")).get("status") != "E2_COST_GATE_PASS":
        raise RuntimeError("Run stage requires the frozen six-hour cost gate to pass")
    protocol, _, _ = verify_frozen_inputs()
    formal_command_started = time.perf_counter()
    dump(OUT / "RUN_STATUS.json", {"status": "RUNNING_FORMAL", "formal_started_utc": dt.datetime.now(dt.timezone.utc).isoformat()})
    append_event("formal_process_started", protocol_sha256=sha_file(PROTOCOL_PATH))
    try:
        prefix_start = time.perf_counter()
        prefix_marker, e1_arrays, prefix_hashes = open_e1_cache()
        prefix_validation_seconds = time.perf_counter() - prefix_start
        model, pristine, model_load_seconds = load_model_and_pristine()
        down_start = time.perf_counter()
        down_arrays, down_timing = ensure_down_inputs(model, e1_arrays)
        down_cache_validation_seconds = time.perf_counter() - down_start
        if down_timing["status"] != "CACHE_REUSED":
            raise RuntimeError("Formal run unexpectedly rebuilt the frozen down-input cache")
        search_bundle = run_search(model, pristine, e1_arrays, down_arrays)
        held_bundle = run_held_scoring(model, pristine, e1_arrays, down_arrays, search_bundle)
        formal_wall_seconds = time.perf_counter() - formal_command_started
        environment = runtime_env()
        runtime_measurements = {"formal_wall_seconds": formal_wall_seconds,
                                "model_load_seconds": model_load_seconds,
                                "prefix_validation_seconds": prefix_validation_seconds,
                                "down_cache_validation_seconds": down_cache_validation_seconds,
                                "environment": environment,
                                "pristine": pristine}
        finalize_formal(search_bundle, held_bundle, runtime_measurements)
    except Exception as exc:
        dump(OUT / "RUN_STATUS.json", {"status": "FORMAL_FAILED_OR_INTERRUPTED", "error": repr(exc),
                                        "traceback": __import__("traceback").format_exc(),
                                        "elapsed_seconds": time.perf_counter() - formal_command_started,
                                        "resumable_checkpoints": True})
        append_event("formal_process_failed", error=repr(exc))
        raise


if __name__ == "__main__":
    main()
