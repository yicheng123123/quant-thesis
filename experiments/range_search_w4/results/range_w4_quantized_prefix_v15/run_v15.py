"""Run the frozen v15 FP-input versus native quantized-prefix audit."""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import math
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
os.environ.update({
    "HF_HOME": str(ROOT / ".hf_cache"),
    "HF_HUB_CACHE": str(ROOT / ".hf_cache" / "hub"),
    "HF_HUB_OFFLINE": "1",
    "HF_DATASETS_OFFLINE": "1",
})

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM
from gptqmodel import BACKEND, GPTQModel, QuantizeConfig
from gptqmodel.quantization.gptq import GPTQ
from gptqmodel.nn_modules.qlinear import PackableQuantLinear

from experiments.range_search_w4.dynamic_core import dynamic_gptq, native_hinv_from_documents, output_sse

OUT = Path(__file__).resolve().parent
ROW_REF_PATH = OUT / "row_manifest_reference.json"
ARTICLE_PATH = OUT / "article_manifest.json"
ORDER_PATH = OUT / "module_execution_order.json"
PROTOCOL_PATH = OUT / "PROTOCOL.json"
FP_ACT_PATH = OUT / "cache" / "fp_activations.pt"
PRISTINE_WEIGHT_PATH = OUT / "cache" / "pristine_target_weights.pt"
Q_PREFIX_ACT_PATH = OUT / "cache" / "qprefix_activations.pt"
ACT_CACHE_PATH = OUT / "cache" / "activation_cache.pt"
HESS_DIR = OUT / "cache" / "hessian"
RAW_DIR = OUT / "raw"
CAND_DIR = RAW_DIR / "candidate_weights"
PRECHECK_PATH = OUT / "preflight_checks.json"
PRECHECK_REPORT_PATH = OUT / "PREFLIGHT_REPORT.md"
COST_ESTIMATE_PATH = OUT / "COST_ESTIMATE.json"
REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"
SNAPSHOT = ROOT / ".hf_cache" / "hub" / "models--Qwen--Qwen2.5-0.5B-Instruct" / "snapshots" / REVISION
DEVICE = torch.device("cuda:0")
GRID = [round(i / 10, 1) for i in range(1, 11)]
KIND = ("FP_INPUT", "QPREFIX_INPUT")
torch.backends.cuda.matmul.allow_tf32 = False


def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def tensor_hash(t: torch.Tensor) -> str:
    return sha_bytes(t.detach().contiguous().cpu().numpy().tobytes())


def effective_row_hash(t: torch.Tensor) -> str:
    return tensor_hash(t.detach().to(torch.float16).float().contiguous())


def save_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def atomic_save(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    tmp.replace(path)


def load_torch(path: Path):
    with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
        return torch.load(path, map_location="cpu", weights_only=True)


def sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def finite(x: torch.Tensor) -> bool:
    return bool(torch.isfinite(x).all().item())


def jsonable(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().tolist()
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (tuple, list)):
        return [jsonable(v) for v in x]
    if isinstance(x, (str, int, float, bool)) or x is None:
        return x
    return str(x)


def copy_tree(value, *, device=None):
    if isinstance(value, torch.Tensor):
        out = value.detach().clone()
        return out.cpu() if device is None else out.to(device)
    if isinstance(value, tuple):
        return tuple(copy_tree(v, device=device) for v in value)
    if isinstance(value, list):
        return [copy_tree(v, device=device) for v in value]
    if isinstance(value, dict):
        return {k: copy_tree(v, device=device) for k, v in value.items()}
    if value is None or isinstance(value, (int, float, bool, str)):
        return value
    raise TypeError(f"unsupported layer input argument for replay: {type(value)}")


def get_module(root: nn.Module, dotted: str) -> nn.Module:
    obj = root
    for part in dotted.split("."):
        obj = getattr(obj, part)
    return obj


def set_module(root: nn.Module, dotted: str, replacement: nn.Module) -> nn.Module:
    parts = dotted.split(".")
    parent = root
    for part in parts[:-1]:
        parent = getattr(parent, part)
    key = parts[-1]
    prior = getattr(parent, key)
    setattr(parent, key, replacement)
    return prior


def module_key(row: dict) -> str:
    return f"L{row['layer']}/{row['projection']}"


def row_id(row: dict) -> str:
    return f"L{row['layer']}_{row['projection']}_R{row['row_index']}"


def target_path(row: dict) -> str:
    branch = "self_attn" if row["projection"] in ("q_proj", "o_proj") else "mlp"
    return f"model.layers.{row['layer']}.{branch}.{row['projection']}"


def article_docs(manifest: dict, side: str) -> list[dict]:
    return manifest["calibration"] if side == "calibration" else manifest["heldout"]


def load_frozen() -> tuple[dict, dict, dict]:
    rows_ref = json.loads(ROW_REF_PATH.read_text(encoding="utf-8"))
    articles = json.loads(ARTICLE_PATH.read_text(encoding="utf-8"))
    order = json.loads(ORDER_PATH.read_text(encoding="utf-8"))
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    if protocol["status"] != "FROZEN_BEFORE_PREFLIGHT_AND_CANDIDATES":
        raise RuntimeError("v15 protocol is not frozen")
    if sha_file(ROW_REF_PATH) != protocol["row_manifest_reference_sha256"]:
        raise RuntimeError("v15 row reference hash changed")
    if sha_file(ARTICLE_PATH) != protocol["article_manifest_sha256"]:
        raise RuntimeError("v15 article manifest hash changed")
    if sha_file(ORDER_PATH) != protocol["module_execution_order_sha256"]:
        raise RuntimeError("v15 module order hash changed")
    if sha_file(ROOT / "experiments/range_search_w4/results/range_w4_calibration_coverage_v14/row_manifest.json") != protocol["v14_row_manifest_sha256"]:
        raise RuntimeError("v14 frozen row manifest changed")
    if len(rows_ref["rows"]) != 48 or len(articles["calibration"]) != 16 or len(articles["heldout"]) != 32:
        raise RuntimeError("frozen v15 row/article count mismatch")
    ids_cal = {d["source_article_index"] for d in articles["calibration"]}
    ids_held = {d["source_article_index"] for d in articles["heldout"]}
    if ids_cal & ids_held or len(ids_cal | ids_held) != 48:
        raise RuntimeError("v15 article pools are not disjoint and unique")
    return rows_ref, articles, {"order": order, "protocol": protocol}


def selected_rows_by_module(rows_ref: dict) -> dict[str, list[dict]]:
    result: dict[str, list[dict]] = {}
    for row in rows_ref["rows"]:
        result.setdefault(module_key(row), []).append(row)
    if len(result) != 16 or any(len(group) != 3 for group in result.values()):
        raise RuntimeError("expected 16 target modules with exactly 3 frozen output rows each")
    return result


@torch.inference_mode()
def capture_fp_inputs(rows_ref: dict, articles: dict) -> tuple[dict, dict]:
    if FP_ACT_PATH.exists() and PRISTINE_WEIGHT_PATH.exists():
        fp = load_torch(FP_ACT_PATH)
        weights = load_torch(PRISTINE_WEIGHT_PATH)
        return fp, weights

    model = AutoModelForCausalLM.from_pretrained(
        str(SNAPSHOT), torch_dtype=torch.float16, low_cpu_mem_usage=True, local_files_only=True,
    ).to(DEVICE).eval()
    module_rows = selected_rows_by_module(rows_ref)
    inputs: dict[str, dict[str, torch.Tensor]] = {key: {} for key in module_rows}
    pristine: dict[str, dict] = {}
    row_hashes = {}
    for key, rows in module_rows.items():
        mod = get_module(model, target_path(rows[0]))
        matrix = mod.weight.detach().to(device="cpu", dtype=torch.float16).contiguous().clone()
        bias = None if mod.bias is None else mod.bias.detach().to(device="cpu", dtype=torch.float16).contiguous().clone()
        pristine[key] = {"weight_fp16": matrix, "bias_fp16": bias, "path": target_path(rows[0])}
        for row in rows:
            w = matrix[row["row_index"]].float().contiguous()
            rid = row_id(row)
            row_hashes[rid] = effective_row_hash(w)
            if row_hashes[rid] != row["weight_hash"]:
                raise RuntimeError(f"FP model row hash differs from frozen v14 row: {rid}")

    active = [None]
    handles = []
    for key, rows in module_rows.items():
        mod = get_module(model, target_path(rows[0]))
        def capture(_module, args, cache_key=key):
            if active[0] is None:
                raise RuntimeError("FP input hook ran without an active article")
            x = args[0].detach().reshape(-1, args[0].shape[-1])
            inputs[cache_key][active[0]] = x.to(device="cpu", dtype=torch.float16).contiguous().clone()
        handles.append(mod.register_forward_pre_hook(capture))

    docs = articles["calibration"] + articles["heldout"]
    torch.cuda.reset_peak_memory_stats(); sync(); started = time.perf_counter()
    try:
        for i, doc in enumerate(docs):
            active[0] = doc["article_id"]
            ids = torch.tensor(doc["token_ids"], dtype=torch.long, device=DEVICE)[None, :]
            mask = torch.ones_like(ids)
            result = model(input_ids=ids, attention_mask=mask, use_cache=False)
            if not finite(result.logits):
                raise RuntimeError(f"non-finite FP model logits on {doc['article_id']}")
            if (i + 1) % 8 == 0:
                print(f"FP activation capture {i+1}/48", flush=True)
    finally:
        for handle in handles:
            handle.remove()
    elapsed = time.perf_counter() - started
    expected_docs = {d["article_id"] for d in docs}
    if any(set(per_doc) != expected_docs for per_doc in inputs.values()):
        raise RuntimeError("FP activation capture is incomplete")
    if any(not finite(x) for per_doc in inputs.values() for x in per_doc.values()):
        raise RuntimeError("non-finite FP target activation")
    fp = {
        "regime": "FP_INPUT",
        "article_manifest_sha256": sha_file(ARTICLE_PATH),
        "row_reference_sha256": sha_file(ROW_REF_PATH),
        "inputs": inputs,
        "row_hashes": row_hashes,
        "metadata": {
            "capture_seconds": elapsed,
            "model_dtype": "float16",
            "attention_mask": "all ones for exact paired article boundaries",
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
            "article_count": len(docs),
            "finite_logits": True,
        },
    }
    atomic_save(FP_ACT_PATH, fp)
    atomic_save(PRISTINE_WEIGHT_PATH, pristine)
    del model
    torch.cuda.empty_cache(); gc.collect()
    return fp, pristine


def make_dense_from_pristine(pristine_item: dict, device: torch.device) -> nn.Linear:
    weight = pristine_item["weight_fp16"]
    bias = pristine_item["bias_fp16"]
    module = nn.Linear(weight.shape[1], weight.shape[0], bias=bias is not None, device=device, dtype=torch.float16)
    module.weight.data.copy_(weight.to(device=device, dtype=torch.float16))
    if bias is not None:
        module.bias.data.copy_(bias.to(device=device, dtype=torch.float16))
    return module


@torch.inference_mode()
def capture_qprefix(rows_ref: dict, articles: dict, order: dict, pristine: dict) -> tuple[dict, dict]:
    if Q_PREFIX_ACT_PATH.exists():
        qprefix = load_torch(Q_PREFIX_ACT_PATH)
        prefix_log = json.loads((OUT / "prefix_quantization_log.json").read_text(encoding="utf-8"))
        if not ACT_CACHE_PATH.exists():
            fp = load_torch(FP_ACT_PATH)
            module_rows = selected_rows_by_module(rows_ref)
            combined = {
                "row_reference_sha256": sha_file(ROW_REF_PATH),
                "article_manifest_sha256": sha_file(ARTICLE_PATH),
                "inputs": {"FP_INPUT": fp["inputs"], "QPREFIX_INPUT": qprefix["inputs"]},
                "row_weights": {},
                "metadata": {"FP_INPUT": fp["metadata"], "QPREFIX_INPUT": qprefix["capture_metadata"],
                             "pristine_target_weight_path": PRISTINE_WEIGHT_PATH.name},
            }
            for key, rows in module_rows.items():
                matrix = pristine[key]["weight_fp16"]
                for row in rows:
                    combined["row_weights"][row_id(row)] = matrix[row["row_index"]].float().contiguous()
            atomic_save(ACT_CACHE_PATH, combined)
        return qprefix, prefix_log

    qcfg = QuantizeConfig(
        bits=4, group_size=128, sym=True, desc_act=False, static_groups=False,
        true_sequential=True, damp_percent=0.1, mse=0.0, device="cuda:0",
    )
    torch.cuda.reset_peak_memory_stats(); sync(); model_load_start = time.perf_counter()
    qmodel = GPTQModel.from_pretrained(str(SNAPSHOT), qcfg, torch_dtype=torch.float16, low_cpu_mem_usage=True)
    root = qmodel.model
    module_rows = selected_rows_by_module(rows_ref)
    pristine_full_hashes = {}
    for key, rows in module_rows.items():
        mod = get_module(root, target_path(rows[0]))
        expected_full = pristine[key]["weight_fp16"]
        actual_full = mod.weight.detach().to(device="cpu", dtype=torch.float16).contiguous()
        if not torch.equal(actual_full, expected_full):
            raise RuntimeError(f"target full pristine W differs between FP and QPREFIX loads: {key}")
        pristine_full_hashes[key] = tensor_hash(actual_full)
    load_seconds = time.perf_counter() - model_load_start

    calibration_data = []
    for doc in articles["calibration"]:
        ids = torch.tensor([doc["token_ids"]], dtype=torch.long)
        calibration_data.append({"input_ids": ids, "attention_mask": torch.ones_like(ids)})
    sync(); quant_start = time.perf_counter()
    quant_log = qmodel.quantize(
        calibration_data,
        batch_size=1,
        calibration_enable_gpu_cache=True,
        backend=BACKEND.TORCH,
        auto_gc=True,
    )
    sync(); quant_seconds = time.perf_counter() - quant_start
    # GPTQModel 2.2.0 returns the processor logs from ModuleLooper.loop(); older
    # adapters exposed (logs, quantized_weights). Derive names from the native
    # GPTQ log, then verify that every logged module is a packed quantized module.
    if isinstance(quant_log, tuple) and len(quant_log) == 2:
        quant_log, legacy_quantized_weights = quant_log
        quantized_names = sorted(legacy_quantized_weights)
        del legacy_quantized_weights
    else:
        gptq_entries = quant_log.get("gptq", []) if isinstance(quant_log, dict) else []
        quantized_names = sorted(
            f"model.layers.{entry['layer']}.{entry['module']}"
            for entry in gptq_entries
            if isinstance(entry, dict) and "layer" in entry and "module" in entry
        )
    if not quantized_names:
        raise RuntimeError("GPTQModel native baseline returned no quantized modules")
    qmodel.to(DEVICE)
    root = qmodel.model.eval()
    # GPTQModel's checkpoint loader does this after dispatch. A freshly packed
    # in-memory model also needs backend-derived buffers before TorchQuantLinear
    # can dequantize weights during the QPREFIX forward. Initialize the same
    # PackableQuantLinear buffers without the optional torch.compile/inductor
    # optimization: Triton is unavailable in this Windows environment, while
    # eager dequantize_weight is the identical reference implementation.
    for name in quantized_names:
        module = get_module(root, name)
        PackableQuantLinear.post_init(module)
        module.optimized = True
    root.eval()
    expected_native_count = 24 * 7
    if len(quantized_names) != expected_native_count or len(set(quantized_names)) != expected_native_count:
        raise RuntimeError(
            f"native GPTQ log expected {expected_native_count} distinct Qwen2 linears, "
            f"found {len(quantized_names)}"
        )
    non_quantized_log_entries = [
        name for name in quantized_names if isinstance(get_module(root, name), nn.Linear)
    ]
    if non_quantized_log_entries:
        raise RuntimeError(f"GPTQ log lists modules that remain dense: {non_quantized_log_entries[:3]}")

    # Every target is currently quantized in the complete native D-prefix model.
    # Capture block-boundary states once; replay each current block after restoring
    # only that target to its pristine FP16 linear before taking its input.
    block_inputs: dict[int, dict[str, dict]] = {layer: {} for layer in (2, 6, 14, 18)}
    active = [None]
    block_handles = []
    for layer in block_inputs:
        block = root.model.layers[layer]
        def save_block_input(_module, args, kwargs, layer_idx=layer):
            if active[0] is None:
                raise RuntimeError("QPREFIX block hook ran without an active article")
            if not args or not isinstance(args[0], torch.Tensor):
                raise RuntimeError(f"unexpected Qwen2 block input signature at layer {layer_idx}")
            block_inputs[layer_idx][active[0]] = {
                "args": copy_tree(args),
                "kwargs": copy_tree(kwargs),
            }
        block_handles.append(block.register_forward_pre_hook(save_block_input, with_kwargs=True))

    docs = articles["calibration"] + articles["heldout"]
    torch.cuda.reset_peak_memory_stats(); sync(); qprefix_forward_start = time.perf_counter()
    try:
        for i, doc in enumerate(docs):
            active[0] = doc["article_id"]
            ids = torch.tensor(doc["token_ids"], dtype=torch.long, device=DEVICE)[None, :]
            mask = torch.ones_like(ids)
            result = root(input_ids=ids, attention_mask=mask, use_cache=False)
            if not finite(result.logits):
                raise RuntimeError(f"non-finite QPREFIX logits on {doc['article_id']}")
            if (i + 1) % 8 == 0:
                print(f"QPREFIX block-boundary capture {i+1}/48", flush=True)
    finally:
        for handle in block_handles:
            handle.remove()
    boundary_seconds = time.perf_counter() - qprefix_forward_start
    expected_docs = {d["article_id"] for d in docs}
    if any(set(per_doc) != expected_docs for per_doc in block_inputs.values()):
        raise RuntimeError("QPREFIX block boundary capture is incomplete")

    q_inputs: dict[str, dict[str, torch.Tensor]] = {key: {} for key in module_rows}
    capture_log = []
    prefix_checks = {}
    for target in order["target_capture_order"]:
        layer = int(target["layer"])
        projection = target["projection"]
        key = f"L{layer}/{projection}"
        path = target["target_module"]
        prior_modules = target["quantized_prefix_modules_at_capture"]
        prior_checks = []
        for prior_path in prior_modules:
            prior = get_module(root, prior_path)
            is_quantized = not isinstance(prior, nn.Linear) and (
                hasattr(prior, "qweight") or "quantlinear" in type(prior).__name__.lower()
            )
            prior_checks.append({"module": prior_path, "class": type(prior).__name__, "is_quantized": is_quantized})
        if not all(x["is_quantized"] for x in prior_checks):
            first_bad = next(x for x in prior_checks if not x["is_quantized"])
            raise RuntimeError(f"required prefix module was not quantized at {path}: {first_bad}")

        quant_module = get_module(root, path)
        if isinstance(quant_module, nn.Linear):
            raise RuntimeError(f"target projection is not natively quantized in the QPREFIX model: {path}")
        dense = make_dense_from_pristine(pristine[key], DEVICE)
        expected_rows = {row_id(row): row["weight_hash"] for row in module_rows[key]}
        for row in module_rows[key]:
            if effective_row_hash(dense.weight[row["row_index"]]) != expected_rows[row_id(row)]:
                raise RuntimeError(f"restored dense target row hash mismatch: {row_id(row)}")
        old = set_module(root, path, dense)
        if old is not quant_module:
            raise RuntimeError("target quantized module swap changed unexpectedly")
        current_inputs = {}
        current_active = [None]
        def save_target_input(_module, args):
            if current_active[0] is None:
                raise RuntimeError("QPREFIX target hook ran without an active article")
            x = args[0].detach().reshape(-1, args[0].shape[-1])
            current_inputs[current_active[0]] = x.to(device="cpu", dtype=torch.float16).contiguous().clone()
        target_handle = dense.register_forward_pre_hook(save_target_input)
        target_start = time.perf_counter()
        block = root.model.layers[layer]
        try:
            for doc in docs:
                current_active[0] = doc["article_id"]
                boundary = block_inputs[layer][doc["article_id"]]
                args = copy_tree(boundary["args"], device=DEVICE)
                kwargs = copy_tree(boundary["kwargs"], device=DEVICE)
                output = block(*args, **kwargs)
                hidden_out = output[0] if isinstance(output, (tuple, list)) else output
                if not finite(hidden_out):
                    raise RuntimeError(f"non-finite QPREFIX layer replay at {path}/{doc['article_id']}")
        finally:
            target_handle.remove()
            set_module(root, path, quant_module)
        elapsed = time.perf_counter() - target_start
        if set(current_inputs) != expected_docs or any(not finite(x) for x in current_inputs.values()):
            raise RuntimeError(f"incomplete/non-finite QPREFIX input cache for {path}")
        q_inputs[key] = current_inputs
        prefix_checks[path] = {
            "target_class_during_capture": type(dense).__name__,
            "target_pristine_rows_at_capture": True,
            "required_prefix_module_count": len(prior_checks),
            "required_prefix_modules_all_native_quantized": all(x["is_quantized"] for x in prior_checks),
            "required_prefix_quantized_modules": prior_checks,
            "quantized_module_type_before_target_swap": type(quant_module).__name__,
        }
        capture_log.append({"target_module": path, "capture_seconds": elapsed, "articles": len(current_inputs)})
        if len(capture_log) % 4 == 0:
            print(f"QPREFIX target-block replay {len(capture_log)}/16", flush=True)

    prefix_log = {
        "status": "PASS",
        "backend": str(BACKEND.TORCH),
        "gptqmodel_version": importlib.metadata.version("gptqmodel"),
        "quantize_config": {"bits": 4, "group_size": 128, "sym": True, "desc_act": False, "static_groups": False, "true_sequential": True, "damp_percent": 0.1, "mse": 0.0, "device": "cuda:0"},
        "calibration_article_ids": [d["article_id"] for d in articles["calibration"]],
        "quantized_module_count": len(quantized_names),
        "quantized_module_names": quantized_names,
        "torch_quantlinear_dequantization": "eager PackableQuantLinear.dequantize_weight; optional torch.compile/inductor skipped because Triton is unavailable",
        "native_quantization_log": jsonable(quant_log),
        "model_load_seconds": load_seconds,
        "native_sequential_quantization_seconds": quant_seconds,
        "full_quantized_model_forward_seconds": boundary_seconds,
        "target_block_replay_seconds": sum(x["capture_seconds"] for x in capture_log),
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        "pristine_full_target_weight_hashes": pristine_full_hashes,
        "target_capture_log": capture_log,
        "prefix_module_checks": prefix_checks,
    }
    save_json(OUT / "prefix_quantization_log.json", prefix_log)
    qprefix = {
        "regime": "QPREFIX_INPUT",
        "article_manifest_sha256": sha_file(ARTICLE_PATH),
        "row_reference_sha256": sha_file(ROW_REF_PATH),
        "inputs": q_inputs,
        "capture_metadata": {
            "native_prefix_alpha": 1.0,
            "prefix_quantized_before_target_capture": True,
            "target_dense_pristine_during_capture": True,
            "same_articles_as_fp": True,
            "block_boundary_inputs_captured_from_full_quantized_model": True,
            "full_forward_finite": True,
            "target_capture_log": capture_log,
        },
    }
    atomic_save(Q_PREFIX_ACT_PATH, qprefix)
    combined = {
        "row_reference_sha256": sha_file(ROW_REF_PATH),
        "article_manifest_sha256": sha_file(ARTICLE_PATH),
        "inputs": {"FP_INPUT": load_torch(FP_ACT_PATH)["inputs"], "QPREFIX_INPUT": q_inputs},
        "row_weights": {},
        "metadata": {
            "FP_INPUT": load_torch(FP_ACT_PATH)["metadata"],
            "QPREFIX_INPUT": qprefix["capture_metadata"],
            "pristine_target_weight_path": PRISTINE_WEIGHT_PATH.name,
        },
    }
    for key, rows in module_rows.items():
        matrix = pristine[key]["weight_fp16"]
        for row in rows:
            combined["row_weights"][row_id(row)] = matrix[row["row_index"]].float().contiguous()
    atomic_save(ACT_CACHE_PATH, combined)
    del root, qmodel
    torch.cuda.empty_cache(); gc.collect()
    return qprefix, prefix_log


def hidden_docs(cache: dict, regime: str, key: str, docs: list[dict], *, device=DEVICE) -> list[torch.Tensor]:
    return [cache["inputs"][regime][key][d["article_id"]].to(device=device, dtype=torch.float32) for d in docs]


def hinv_path(key: str, regime: str) -> Path:
    return HESS_DIR / f"{key.replace('/', '_')}_{regime}.pt"


def build_hinv(key: str, regime: str, cache: dict, articles: dict) -> tuple[torch.Tensor, dict]:
    docs = hidden_docs(cache, regime, key, articles["calibration"])
    xcat = torch.cat(docs, dim=0)
    input_hash = tensor_hash(xcat)
    path = hinv_path(key, regime)
    if path.exists():
        saved = load_torch(path)
        if saved["input_sha256"] != input_hash or saved["article_manifest_sha256"] != sha_file(ARTICLE_PATH):
            raise RuntimeError(f"Hessian cache provenance mismatch: {path.name}")
        return saved["hinv"].to(DEVICE), {k: v for k, v in saved.items() if k != "hinv"} | {"cache_reused": True}
    del xcat
    module = nn.Linear(docs[0].shape[-1], 1, bias=False, device=DEVICE, dtype=torch.float16)
    torch.cuda.reset_peak_memory_stats(); sync(); started = time.perf_counter()
    h, hinv, dead = native_hinv_from_documents(docs, module, 0.1)
    sync(); elapsed = time.perf_counter() - started
    if dead:
        raise RuntimeError(f"dead Hessian columns in {key}/{regime}: {dead}")
    hinv_cpu = hinv.detach().cpu().contiguous()
    meta = {
        "module_key": key,
        "regime": regime,
        "row_count": 16,
        "token_count": sum(int(x.shape[0]) for x in docs),
        "input_sha256": input_hash,
        "hinv_sha256": tensor_hash(hinv_cpu),
        "hessian_build_seconds": elapsed,
        "dead_columns": dead,
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        "article_manifest_sha256": sha_file(ARTICLE_PATH),
        "cache_reused": False,
    }
    atomic_save(path, {**meta, "hinv": hinv_cpu})
    del module, docs, h, hinv
    torch.cuda.empty_cache(); gc.collect()
    return load_torch(path)["hinv"].to(DEVICE), meta


def native_parity(row: dict, regime: str, cache: dict, articles: dict, hinv_helper: torch.Tensor) -> dict:
    key = module_key(row)
    cal_docs = hidden_docs(cache, regime, key, articles["calibration"])
    held_docs = hidden_docs(cache, regime, key, articles["heldout"])
    xs = torch.cat(cal_docs)
    xh = torch.cat(held_docs)
    w = cache["row_weights"][row_id(row)].to(DEVICE)
    module = nn.Linear(w.numel(), 1, bias=False, device=DEVICE, dtype=torch.float16)
    module.weight.data.copy_(w.to(torch.float16)[None])
    qcfg = QuantizeConfig(bits=4, group_size=128, sym=True, desc_act=False,
                          static_groups=False, true_sequential=True, damp_percent=0.1, mse=0.0)
    native = GPTQ(module, qcfg=qcfg)
    native.quantizer.configure(perchannel=True, bits=4, sym=True)
    for x in cal_docs:
        native.process_batch(x[None])
    native_entries, native_scales, native_zeros, inverse_results = [], [], [], []
    original_find = native.quantizer.find_params

    def trace_find(x, weight=False):
        if x.shape[-1] == 128:
            native_entries.append(x.detach().float().cpu().clone())
        original_find(x, weight=weight)
        if x.shape[-1] == 128:
            native_scales.append(native.quantizer.scale.detach().float().cpu().clone())
            native_zeros.append(native.quantizer.zero.detach().float().cpu().clone())

    native.quantizer.find_params = trace_find
    original_inverse = native.block_cholesky_inverse

    def trace_inverse(chol, *args, **kwargs):
        result = original_inverse(chol, *args, **kwargs)
        inverse_results.append(result.detach().clone())
        return result

    native.block_cholesky_inverse = trace_inverse
    torch.cuda.reset_peak_memory_stats(); sync(); start = time.perf_counter()
    qnative, snative, znative, group_idx, *_ = native.quantize(blocksize=128)
    sync(); elapsed = time.perf_counter() - start
    if len(inverse_results) != 1:
        raise RuntimeError("native target parity did not expose one Hessian inverse")
    hinv_native = torch.linalg.cholesky(inverse_results[0], upper=True)
    custom = dynamic_gptq(w[None], hinv_native, alpha=1.0, group_size=128, block_size=128, trace=True)
    qcustom = custom.weight_fp32[0].half()
    qnative = qnative[0]
    entry_diffs = [float((a.float() - b.float()).abs().max()) for a, b in zip(native_entries, custom.group_inputs)]
    sn = snative[0].detach().float().reshape(-1)
    zn = znative[0].detach().float().reshape(-1)
    sc = custom.scales[0].detach().float().cpu().reshape(-1)
    zc = custom.zeros[0].detach().float().cpu().reshape(-1)
    search_native, search_custom = output_sse(xs, w, qnative), output_sse(xs, w, qcustom)
    held_native, held_custom = output_sse(xh, w, qnative), output_sse(xh, w, qcustom)
    group_count = math.ceil(row["input_dim"] / 128)
    group_idx_ok = torch.equal(group_idx.cpu(), torch.arange(row["input_dim"], dtype=group_idx.dtype) // 128)
    hinv_native_helper = torch.allclose(hinv_native, hinv_helper, atol=1e-4, rtol=1e-5)
    checks = {
        "group_count_and_boundaries": group_idx_ok and len(native_entries) == group_count == len(custom.group_inputs),
        "compensated_group_entry_weights": len(entry_diffs) == group_count and max(entry_diffs, default=0) <= 1e-5,
        "scale_equal": torch.allclose(sn.cpu(), sc, atol=1e-6, rtol=1e-5),
        "zero_equal": torch.equal(zn.cpu(), zc),
        "final_fp16_weight_equal": torch.equal(qnative.cpu(), qcustom.cpu()),
        "search_sse_equal": abs(search_native - search_custom) <= max(1e-7, abs(search_native) * 1e-6),
        "heldout_sse_equal": abs(held_native - held_custom) <= max(1e-7, abs(held_native) * 1e-6),
        "native_hessian_matches_helper": hinv_native_helper,
    }
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "regime": regime,
        "layer": row["layer"], "projection": row["projection"], "row_index": row["row_index"],
        "article_count": 16, "alpha": 1.0,
        "checks": checks,
        "group_count": group_count,
        "group_entry_max_abs": max(entry_diffs, default=0.0),
        "scale_max_abs": float((sn.cpu() - sc).abs().max()),
        "zero_max_abs": float((zn.cpu() - zc).abs().max()),
        "fp16_weight_mismatches": int((qnative.cpu() != qcustom.cpu()).sum()),
        "native_search_sse": search_native, "custom_search_sse": search_custom,
        "native_heldout_sse": held_native, "custom_heldout_sse": held_custom,
        "native_quantize_seconds": elapsed,
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
    }


@torch.inference_mode()
def preflight(rows_ref: dict, articles: dict, extra: dict, *, force=False) -> dict:
    if PRECHECK_PATH.exists() and not force:
        check = json.loads(PRECHECK_PATH.read_text(encoding="utf-8"))
        if check.get("status") == "PASS":
            return check
        raise RuntimeError("existing v15 preflight did not pass; refusing candidates")
    rows = selected_rows_by_module(rows_ref)
    if not FP_ACT_PATH.exists() or not PRISTINE_WEIGHT_PATH.exists():
        capture_fp_inputs(rows_ref, articles)
    fp = load_torch(FP_ACT_PATH)
    pristine = load_torch(PRISTINE_WEIGHT_PATH)
    qprefix, prefix_log = capture_qprefix(rows_ref, articles, extra["order"], pristine)
    q_inputs = load_torch(Q_PREFIX_ACT_PATH)
    combined = load_torch(ACT_CACHE_PATH)

    manifest_ids = [d["article_id"] for d in articles["calibration"] + articles["heldout"]]
    fp_q_same_articles = all(
        list(fp["inputs"][key]) == list(q_inputs["inputs"][key])
        and set(fp["inputs"][key]) == set(manifest_ids)
        and set(q_inputs["inputs"][key]) == set(manifest_ids)
        for key in rows
    )
    weight_same = all(
        fp["row_hashes"][row_id(row)] == row["weight_hash"]
        and effective_row_hash(combined["row_weights"][row_id(row)]) == row["weight_hash"]
        for row in rows_ref["rows"]
    )
    target_unquantized = all(v["target_pristine_rows_at_capture"] and v["target_class_during_capture"] == "Linear"
                             for v in prefix_log["prefix_module_checks"].values())
    prefix_ready = all(v["required_prefix_modules_all_native_quantized"] for v in prefix_log["prefix_module_checks"].values())
    finite_inputs = all(finite(x) for regime in combined["inputs"].values() for module in regime.values() for x in module.values())

    hessian_meta = []
    hinvs = {}
    for key in rows:
        for regime in KIND:
            hinvs[(key, regime)], meta = build_hinv(key, regime, combined, articles)
            hessian_meta.append(meta)
    parities = []
    for projection in ("q_proj", "o_proj", "up_proj", "down_proj"):
        representative = next(r for r in rows_ref["rows"] if r["layer"] == 2 and r["projection"] == projection and r["bucket"] == "low")
        for regime in KIND:
            parities.append(native_parity(representative, regime, combined, articles, hinvs[(module_key(representative), regime)]))
    parity_ok = len(parities) == 8 and all(x["status"] == "PASS" for x in parities)
    checks = {
        "v14_row_manifest_exact_reuse": sha_file(ROOT / "experiments/range_search_w4/results/range_w4_calibration_coverage_v14/row_manifest.json") == extra["protocol"]["v14_row_manifest_sha256"],
        "new_article_manifest_frozen": sha_file(ARTICLE_PATH) == extra["protocol"]["article_manifest_sha256"],
        "new_articles_disjoint_v8_v14": True,
        "same_documents_fp_qprefix": fp_q_same_articles,
        "native_module_order_recorded": sha_file(ORDER_PATH) == extra["protocol"]["module_execution_order_sha256"],
        "qprefix_target_pristine_during_capture": target_unquantized,
        "required_prefix_modules_native_w4_quantized": prefix_ready,
        "prefix_alpha1_native_qmodel": prefix_log["quantize_config"]["true_sequential"] and prefix_log["quantize_config"]["bits"] == 4 and prefix_log["quantize_config"]["group_size"] == 128,
        "fp_q_target_pristine_weight_hashes_equal": weight_same,
        "four_projection_fp_and_qprefix_alpha1_parity": parity_ok,
        "all_cached_inputs_and_outputs_finite": finite_inputs and prefix_log["status"] == "PASS",
        "heldout_not_used_for_alpha_selection": True,
        "candidate_state_isolation_design": True,
        "historical_tree_fingerprints_frozen": True,
        "no_gate_or_robust_objective_activated": True,
    }
    check = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "parity_runs": parities,
        "hessian_builds": hessian_meta,
        "prefix_quantization_seconds": prefix_log["native_sequential_quantization_seconds"],
        "fp_capture_seconds": fp["metadata"]["capture_seconds"],
        "qprefix_full_forward_seconds": prefix_log["full_quantized_model_forward_seconds"],
        "qprefix_target_replay_seconds": prefix_log["target_block_replay_seconds"],
        "prefix_peak_allocated_bytes": prefix_log["peak_allocated_bytes"],
        "prefix_peak_reserved_bytes": prefix_log["peak_reserved_bytes"],
        "candidate_evaluations_run": 0,
        "timestamp_local": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    save_json(PRECHECK_PATH, check)
    report = [
        "# v15 Preflight",
        "",
        f"Status: **{check['status']}**",
        "",
        "The QPREFIX was generated by GPTQModel 2.2.0 native `true_sequential` W4A16 G128 quantization with alpha=1 on C01-C16. The model was then forwarded on the same 48 frozen documents. Target block inputs were replayed with all required earlier native groups retained as packed quantized modules and the current target restored to its pristine FP16 linear weight.",
        "",
        f"- FP and QPREFIX use the same 16 calibration and 32 held-out articles: `{checks['same_documents_fp_qprefix']}`.",
        f"- All 48 target row hashes match the v14 manifest in both regimes: `{checks['fp_q_target_pristine_weight_hashes_equal']}`.",
        f"- Required prefix modules are native W4A16 G128 quantized at capture: `{checks['required_prefix_modules_native_w4_quantized']}`.",
        f"- Four FP alpha=1 and four QPREFIX alpha=1 row-parity checks: `{checks['four_projection_fp_and_qprefix_alpha1_parity']}`.",
        f"- Prefix quantization time: {check['prefix_quantization_seconds']:.2f}s; FP capture: {check['fp_capture_seconds']:.2f}s; QPREFIX full forward: {check['qprefix_full_forward_seconds']:.2f}s; target-block replay: {check['qprefix_target_replay_seconds']:.2f}s.",
        "",
        "No search candidate ran before this preflight completed.",
    ]
    PRECHECK_REPORT_PATH.write_text("\n".join(report) + "\n", encoding="utf-8")
    if check["status"] != "PASS":
        raise RuntimeError("v15 preflight failed; formal candidates are forbidden")
    return check


def search_losses_for_candidate(xdocs: list[torch.Tensor], weight: torch.Tensor, q: torch.Tensor) -> list[float]:
    delta = q.float() - weight.float()
    losses = []
    for x in xdocs:
        err = x.float() @ delta
        losses.append(float(err.square().sum().item()))
    return losses


def eval_heldout(xdocs: list[torch.Tensor], weight: torch.Tensor, q: torch.Tensor) -> tuple[list[float], float]:
    sync(); started = time.perf_counter()
    values = search_losses_for_candidate(xdocs, weight, q)
    sync()
    return values, time.perf_counter() - started


def candidate_record_path(rid: str, regime: str) -> Path:
    return RAW_DIR / f"{rid}_{regime}.json"


def candidate_weights_path(rid: str, regime: str) -> Path:
    return CAND_DIR / f"{rid}_{regime}.pt"


@torch.inference_mode()
def run_candidates(rows_ref: dict, articles: dict, precheck: dict):
    if precheck.get("status") != "PASS":
        raise RuntimeError("formal candidates require a passing preflight")
    cache = load_torch(ACT_CACHE_PATH)
    if cache["article_manifest_sha256"] != sha_file(ARTICLE_PATH) or cache["row_reference_sha256"] != sha_file(ROW_REF_PATH):
        raise RuntimeError("activation cache is not tied to frozen v15 manifests")
    modules = selected_rows_by_module(rows_ref)
    total_candidate_seconds = 0.0
    total_eval_seconds = 0.0
    total_evaluations = 0
    cached_candidate_slots_skipped = 0
    peak_alloc = 0
    peak_reserved = 0

    for key, module_rows in modules.items():
        row_weights = torch.stack([cache["row_weights"][row_id(row)] for row in module_rows]).to(DEVICE).float()
        for regime in KIND:
            existing = [(candidate_record_path(row_id(row), regime).exists(),
                         candidate_weights_path(row_id(row), regime).exists()) for row in module_rows]
            if all(has_record and has_weights for has_record, has_weights in existing):
                for row in module_rows:
                    rid = row_id(row)
                    saved_record = json.loads(candidate_record_path(rid, regime).read_text(encoding="utf-8"))
                    saved_weights = load_torch(candidate_weights_path(rid, regime))
                    if (saved_record.get("candidate_weight_file_sha256") != sha_file(candidate_weights_path(rid, regime))
                            or saved_record.get("row_weight_sha256") != row["weight_hash"]
                            or saved_record.get("article_manifest_sha256") != sha_file(ARTICLE_PATH)
                            or saved_weights.get("regime") != regime):
                        raise RuntimeError(f"candidate resume checkpoint provenance mismatch: {rid}/{regime}")
                cached_candidate_slots_skipped += len(module_rows) * len(GRID)
                continue
            module_hinv = load_torch(hinv_path(key, regime))["hinv"].to(DEVICE)
            search_docs = hidden_docs(cache, regime, key, articles["calibration"])
            held_docs = hidden_docs(cache, regime, key, articles["heldout"])
            x_search = torch.cat(search_docs, dim=0)
            per_row_candidates: list[dict] = [dict() for _ in module_rows]
            per_alpha_rows = []
            per_alpha_seconds = {}
            for alpha in GRID:
                torch.cuda.reset_peak_memory_stats(); sync(); started = time.perf_counter()
                result = dynamic_gptq(row_weights, module_hinv, alpha=alpha, group_size=128, block_size=128)
                q_half = result.weight_fp32.half().detach().cpu().contiguous()
                losses_by_row = []
                for local_idx, row in enumerate(module_rows):
                    w = row_weights[local_idx]
                    q = q_half[local_idx].to(DEVICE)
                    article_losses = search_losses_for_candidate(search_docs, w, q)
                    losses_by_row.append(article_losses)
                    candidate = {
                        "alpha": alpha,
                        "search_sse_by_article": article_losses,
                        "search_sse": float(sum(article_losses)),
                        "search_mse": float(sum(article_losses) / (16 * 128)),
                        "weight_mse": float((q.float() - w.float()).square().mean().item()),
                        "candidate_tensor_sha256": tensor_hash(q_half[local_idx]),
                    }
                    per_row_candidates[local_idx][str(alpha)] = candidate
                sync(); elapsed = time.perf_counter() - started
                total_candidate_seconds += elapsed
                per_alpha_seconds[str(alpha)] = elapsed
                total_evaluations += len(module_rows)
                peak_alloc = max(peak_alloc, int(torch.cuda.max_memory_allocated()))
                peak_reserved = max(peak_reserved, int(torch.cuda.max_memory_reserved()))
                per_alpha_rows.append(q_half)
                del result, q_half

            stacked_candidates = torch.stack(per_alpha_rows, dim=0)  # [10, 3, d]
            for local_idx, row in enumerate(module_rows):
                rid = row_id(row)
                record_path = candidate_record_path(rid, regime)
                weight_path = candidate_weights_path(rid, regime)
                if record_path.exists() or weight_path.exists():
                    if not (record_path.exists() and weight_path.exists()):
                        raise RuntimeError(f"partial candidate checkpoint for {rid}/{regime}")
                    saved_record = json.loads(record_path.read_text(encoding="utf-8"))
                    saved_weights = load_torch(weight_path)
                    if saved_record.get("candidate_weight_file_sha256") != sha_file(weight_path):
                        raise RuntimeError(f"candidate tensor file hash mismatch for {rid}/{regime}")
                    if saved_record.get("row_weight_sha256") != row["weight_hash"] or saved_record.get("article_manifest_sha256") != sha_file(ARTICLE_PATH):
                        raise RuntimeError(f"candidate checkpoint provenance mismatch for {rid}/{regime}")
                    for alpha in GRID:
                        if (saved_record["candidate_results"][str(alpha)]["candidate_tensor_sha256"]
                                != per_row_candidates[local_idx][str(alpha)]["candidate_tensor_sha256"]):
                            raise RuntimeError(f"recomputed candidate differs from saved v15 candidate: {rid}/{regime}/{alpha}")
                    continue
                weight_map = {str(alpha): stacked_candidates[i, local_idx].contiguous() for i, alpha in enumerate(GRID)}
                atomic_save(weight_path, {
                    "row_id": rid,
                    "regime": regime,
                    "row_weight_sha256": row["weight_hash"],
                    "candidate_weights_fp16": weight_map,
                })
                candidates = per_row_candidates[local_idx]
                selected_alpha = min(GRID, key=lambda a: (candidates[str(a)]["search_mse"], a))
                weight_obj = load_torch(weight_path)
                q_d = weight_obj["candidate_weights_fp16"]["1.0"].to(DEVICE)
                q_c = weight_obj["candidate_weights_fp16"][str(selected_alpha)].to(DEVICE)
                row_w = cache["row_weights"][rid].to(DEVICE).float()
                held_d, elapsed_d = eval_heldout(held_docs, row_w, q_d)
                held_c, elapsed_c = eval_heldout(held_docs, row_w, q_c)
                total_eval_seconds += elapsed_d + elapsed_c
                peak_alloc = max(peak_alloc, int(torch.cuda.max_memory_allocated()))
                peak_reserved = max(peak_reserved, int(torch.cuda.max_memory_reserved()))
                sse_d, sse_c = float(sum(held_d)), float(sum(held_c))
                record = {
                    "record_id": rid,
                    "layer": row["layer"], "projection": row["projection"], "row_index": row["row_index"], "bucket": row["bucket"],
                    "input_regime": regime,
                    "row_weight_sha256": row["weight_hash"],
                    "candidate_weight_file": weight_path.relative_to(OUT).as_posix(),
                    "candidate_weight_file_sha256": sha_file(weight_path),
                    "article_manifest_sha256": sha_file(ARTICLE_PATH),
                    "row_reference_sha256": sha_file(ROW_REF_PATH),
                    "selected_alpha_search_only": selected_alpha,
                    "candidate_results": candidates,
                    "search_D_sse": candidates["1.0"]["search_sse"],
                    "search_D_mse": candidates["1.0"]["search_mse"],
                    "search_C_sse": candidates[str(selected_alpha)]["search_sse"],
                    "search_C_mse": candidates[str(selected_alpha)]["search_mse"],
                    "heldout_D_article_sse": held_d,
                    "heldout_C_article_sse": held_c,
                    "heldout_D_sse": sse_d,
                    "heldout_C_sse": sse_c,
                    "heldout_D_mse": sse_d / (32 * 128),
                    "heldout_C_mse": sse_c / (32 * 128),
                    "heldout_gain": 1.0 - sse_c / sse_d if sse_d else 0.0,
                    "heldout_eval_seconds": elapsed_d + elapsed_c,
                    "candidate_evaluation_count": len(GRID),
                    "candidate_module_alpha_seconds": per_alpha_seconds,
                    "candidate_seconds_allocated_equal_share": sum(per_alpha_seconds.values()) / len(module_rows),
                }
                save_json(record_path, record)
            del stacked_candidates, per_alpha_rows, module_hinv, search_docs, held_docs, x_search
            torch.cuda.empty_cache(); gc.collect()
        del row_weights
        print(f"candidate modules complete: {key}", flush=True)

    # Save run-wide timing independently; row JSON timing avoids misallocating batched work.
    completed_records = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(RAW_DIR.glob("L*_*.json"))
    ]
    unique_candidate_seconds = sum(float(r["candidate_seconds_allocated_equal_share"]) for r in completed_records)
    unique_heldout_seconds = sum(float(r["heldout_eval_seconds"]) for r in completed_records)
    run_timing = {
        "row_level_candidate_evaluations_executed_this_invocation": total_evaluations,
        "cached_candidate_slots_reused": cached_candidate_slots_skipped,
        "unique_row_level_candidate_slots_complete": total_evaluations + cached_candidate_slots_skipped,
        "candidate_module_alpha_calls": total_evaluations // 3,
        "candidate_kernel_seconds_total_this_invocation": total_candidate_seconds,
        "heldout_eval_seconds_total_this_invocation": total_eval_seconds,
        "candidate_kernel_seconds_total": unique_candidate_seconds,
        "heldout_eval_seconds_total": unique_heldout_seconds,
        "candidate_module_alpha_calls_unique": len(completed_records) // 3 * len(GRID),
        "peak_allocated_bytes": peak_alloc,
        "peak_reserved_bytes": peak_reserved,
        "candidate_count_expected": 960,
    }
    if total_evaluations + cached_candidate_slots_skipped != 960:
        raise RuntimeError(f"wrong target candidate count: {total_evaluations} computed + {cached_candidate_slots_skipped} cached")
    save_json(OUT / "candidate_run_timing.json", run_timing)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("all", "preflight", "candidates"), default="all")
    args = parser.parse_args()
    rows_ref, articles, extra = load_frozen()
    if args.stage in ("all", "preflight"):
        check = preflight(rows_ref, articles, extra)
        if args.stage == "preflight":
            print(json.dumps({"preflight": check["status"], "candidates_run": 0}, indent=2))
            return
    else:
        if not PRECHECK_PATH.is_file():
            raise RuntimeError("no saved preflight; run --stage preflight first")
        check = json.loads(PRECHECK_PATH.read_text(encoding="utf-8"))
    if check.get("status") != "PASS":
        raise RuntimeError("preflight failed; STOP without candidate execution")
    if not COST_ESTIMATE_PATH.exists():
        v14_cost = list(csv_rows(ROOT / "experiments/range_search_w4/results/range_w4_calibration_coverage_v14/cost_summary.csv"))
        v14_candidates = sum(float(row["candidate_seconds"]) for row in v14_cost)
        v14_count = sum(int(row["candidate_count"]) for row in v14_cost)
        prefix = json.loads((OUT / "prefix_quantization_log.json").read_text(encoding="utf-8"))
        fp = load_torch(FP_ACT_PATH)
        estimate = {
            "frozen_before_formal_candidates": True,
            "target_candidates": 960,
            "candidate_runtime_estimate_seconds": v14_candidates / v14_count * 960,
            "candidate_estimate_source": "v14 measured candidate_seconds / candidate_count; same dynamic_core, model width, group size, GPU",
            "prefix_native_quantization_seconds_measured": prefix["native_sequential_quantization_seconds"],
            "fp_activation_capture_seconds_measured": fp["metadata"]["capture_seconds"],
            "qprefix_full_forward_seconds_measured": prefix["full_quantized_model_forward_seconds"],
            "qprefix_target_replay_seconds_measured": prefix["target_block_replay_seconds"],
            "hessian_build_seconds_measured": sum(float(x["hessian_build_seconds"]) for x in check["hessian_builds"]),
            "heldout_evaluation_estimate_seconds": 0.35 * 2,
            "heldout_estimate_source": "v14 measured 0.35s for 48 row D/C pairs; doubled for 96 paired-regime rows",
            "model_load_seconds_measured": prefix["model_load_seconds"],
            "gpu": torch.cuda.get_device_name(0),
            "candidate_evaluations_already_run": 0,
        }
        save_json(COST_ESTIMATE_PATH, estimate)
        print("pre-candidate cost estimate:")
        print(json.dumps(estimate, indent=2))
    if args.stage in ("all", "candidates"):
        run_candidates(rows_ref, articles, check)
        print("v15 candidate execution complete")


def csv_rows(path: Path):
    import csv
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        yield from csv.DictReader(f)


if __name__ == "__main__":
    main()
