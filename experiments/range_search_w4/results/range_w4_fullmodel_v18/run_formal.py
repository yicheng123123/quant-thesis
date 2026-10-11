"""Frozen v18 sequential M1/M2 full-model quantization runner.

Only writes below this v18 directory. Layer checkpoints are atomic and contain
all row selections/quant metadata needed to resume without revisiting a
completed layer.
"""
from __future__ import annotations

import csv
import gc
import hashlib
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM


OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[3]
RESULTS = ROOT / "experiments/range_search_w4/results"
V171 = RESULTS / "range_w4_m2_parity_v17_1"
V173 = RESULTS / "range_w4_m2_runtime_v17_3"
MODEL_DIR = ROOT / ".hf_cache/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775"
CAL_PATH = OUT / "CALIBRATION_MANIFEST.json"
PROTOCOL_PATH = OUT / "PROTOCOL.json"
CAL = json.loads(CAL_PATH.read_text(encoding="utf-8"))
PROTOCOL = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
GRID = [float(x) for x in PROTOCOL["m2"]["alpha_grid"]]
N_DOCS = int(PROTOCOL["calibration"]["documents"])
SEQ = int(PROTOCOL["calibration"]["tokens_per_document"])
LAYERS = 24
DEVICE = torch.device("cuda")
ACT_DIR = OUT / "raw" / "activations"
SPOT_LAYERS = {0, 6, 12, 18, 23}
sys.path.insert(0, str(V171))
sys.path.insert(0, str(V173))
sys.path.insert(0, str(OUT))
import run_precheck as rp  # noqa: E402
from native_m2 import native_dynamic_gptq  # noqa: E402
from optimized_native_m2 import optimized_dynamic_gptq, prepare_native_state  # noqa: E402


MODULE_ORDERS = {
    "qkv": ["self_attn.k_proj", "self_attn.v_proj", "self_attn.q_proj"],
    "o": ["self_attn.o_proj"],
    "up_gate": ["mlp.up_proj", "mlp.gate_proj"],
    "down": ["mlp.down_proj"],
}
MODULE_GROUP = {name: group for group, names in MODULE_ORDERS.items() for name in names}
TIMING_FIELDS = ["method", "stage", "layer", "module", "alpha", "wall_seconds",
                 "gpu_peak_allocated_bytes", "gpu_peak_reserved_bytes", "detail"]
FORMAL_TIMING = OUT / "FORMAL_TIMING.csv"
FORMAL_RUNTIME = OUT / "FORMAL_RUNTIME.json"
CHECKS_PATH = OUT / "correctness_checks.json"
MANIFEST_PATH = OUT / "FORMAL_RUN_MANIFEST.json"
CODE_BUNDLE = [Path(__file__), OUT / "run_precheck.py", V171 / "native_m2.py", V173 / "optimized_native_m2.py"]


def sync():
    torch.cuda.synchronize()


def sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def sha_tensor(t: torch.Tensor) -> str:
    return sha_bytes(t.detach().cpu().contiguous().numpy().tobytes())


def hash_code_bundle() -> str:
    h = hashlib.sha256()
    for path in CODE_BUNDLE:
        h.update(path.name.encode("utf-8"))
        h.update(bytes.fromhex(sha_file(path)))
    return h.hexdigest()


def peak_reset():
    sync()
    torch.cuda.reset_peak_memory_stats()


def peak_read():
    sync()
    return int(torch.cuda.max_memory_allocated()), int(torch.cuda.max_memory_reserved())


def close_mm(mm):
    if mm is not None:
        mm.flush()
        mm._mmap.close()


def new_mm(path: Path, shape: tuple[int, ...]) -> np.memmap:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    return np.memmap(path, dtype=np.float16, mode="w+", shape=shape)


def write_json_atomic(path: Path, payload: dict):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def update_manifest(**updates):
    data = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    data.update(updates)
    write_json_atomic(MANIFEST_PATH, data)


class Recorder:
    def __init__(self):
        self.path = FORMAL_TIMING
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.file = self.path.open("a", newline="", encoding="utf-8")
        self.writer = csv.DictWriter(self.file, fieldnames=TIMING_FIELDS)
        if self.path.stat().st_size == 0:
            self.writer.writeheader()
            self.file.flush()
        self.peak_allocated = 0
        self.peak_reserved = 0
        self.stage_totals: dict[str, float] = {}
        self.counts: dict[str, int] = {}

    def record(self, method, stage, layer, module, seconds, *, alpha="", detail="", peak=None):
        if peak is None:
            alloc, reserved = peak_read()
        else:
            alloc, reserved = peak
        self.peak_allocated = max(self.peak_allocated, int(alloc))
        self.peak_reserved = max(self.peak_reserved, int(reserved))
        self.stage_totals[stage] = self.stage_totals.get(stage, 0.0) + float(seconds)
        self.counts[stage] = self.counts.get(stage, 0) + 1
        self.writer.writerow({"method": method, "stage": stage, "layer": layer,
                              "module": module, "alpha": alpha,
                              "wall_seconds": float(seconds),
                              "gpu_peak_allocated_bytes": int(alloc),
                              "gpu_peak_reserved_bytes": int(reserved), "detail": detail})
        self.file.flush()

    def close(self):
        self.file.flush()
        self.file.close()


def load_model(source: Path, rec: Recorder, method: str):
    peak_reset()
    started = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        str(source), torch_dtype=torch.float16, low_cpu_mem_usage=True,
        local_files_only=True, attn_implementation="sdpa",
    ).eval().to(DEVICE)
    sync()
    elapsed = time.perf_counter() - started
    rec.record(method, "model_load", -1, str(source), elapsed,
               detail="local-only FP16 model load; RTX 4060 Laptop GPU")
    return model


def token_rows():
    rows = CAL["samples"]
    if len(rows) != N_DOCS or any(len(x["token_ids"]) != SEQ for x in rows):
        raise RuntimeError("frozen calibration shape mismatch")
    if CAL.get("seed") != 20261005 or CAL.get("token_budget") != N_DOCS * SEQ:
        raise RuntimeError("frozen calibration seed/budget mismatch")
    for entry in rows:
        if sha_bytes(json.dumps(entry["token_ids"]).encode()) != entry["token_sha256"]:
            raise RuntimeError(f"frozen token hash mismatch at article {entry['source_article_index']}")
    return rows


def embed_documents(model, token_samples, path: Path, method: str, rec: Recorder):
    hidden = new_mm(path, (N_DOCS, SEQ, model.config.hidden_size))
    peak_reset()
    started = time.perf_counter()
    for i, sample in enumerate(token_samples):
        ids = torch.tensor(sample["token_ids"], dtype=torch.long, device=DEVICE)[None]
        with torch.inference_mode():
            out = model.model.embed_tokens(ids)
            hidden[i] = out[0].to("cpu").numpy()
        del ids, out
    hidden.flush()
    sync()
    elapsed = time.perf_counter() - started
    rec.record(method, "initial_embedding_capture", -1, "model.embed_tokens", elapsed,
               detail=f"{N_DOCS} frozen documents x {SEQ} tokens; ordered input IDs hash verified")
    return hidden


def capture_norm(layer, hidden_mm, path: Path, method: str, layer_index: int, rec: Recorder):
    x_mm = new_mm(path, (N_DOCS, SEQ, layer.input_layernorm.weight.numel()))
    peak_reset()
    per_doc = []
    for i in range(N_DOCS):
        hidden = torch.from_numpy(np.array(hidden_mm[i], copy=True)).unsqueeze(0).to(DEVICE)
        sync()
        started = time.perf_counter()
        with torch.inference_mode():
            normalized = layer.input_layernorm(hidden)
            x_mm[i] = normalized[0].to("cpu").numpy()
        sync()
        per_doc.append(time.perf_counter() - started)
        del hidden, normalized
    x_mm.flush()
    elapsed = sum(per_doc)
    rec.record(method, "qkv_shared_input_capture", layer_index, "input_layernorm", elapsed,
               detail=json.dumps({"input_sha256": rp.sha_memmap(x_mm), "per_doc_median": float(np.median(per_doc)),
                                  "per_doc_p95": float(np.quantile(per_doc, .95, method="higher"))}))
    return x_mm


def run_block_capture(model, layer, hidden_mm, module_names: list[str], x_path: Path,
                      method: str, layer_index: int, stage: str, rec: Recorder):
    d_in = int(layer.get_submodule(module_names[0]).weight.shape[1])
    x_mm = new_mm(x_path, (N_DOCS, SEQ, d_in))
    active = {"doc": -1, "input": None, "count": 0}
    handles = []

    def make_hook(full_name):
        def hook(_module, args):
            value = args[0].detach()
            if active["input"] is None:
                active["input"] = value
                x_mm[active["doc"]] = value[0].to("cpu").numpy()
            elif not torch.equal(active["input"], value):
                raise RuntimeError(f"sibling input mismatch at layer {layer_index}: {full_name}")
            active["count"] += 1
        return hook

    try:
        for name in module_names:
            handles.append(layer.get_submodule(name).register_forward_pre_hook(make_hook(name)))
        positions = torch.arange(SEQ, device=DEVICE)[None]
        peak_reset()
        per_doc = []
        for i in range(N_DOCS):
            active.update(doc=i, input=None, count=0)
            hidden = torch.from_numpy(np.array(hidden_mm[i], copy=True)).unsqueeze(0).to(DEVICE)
            sync()
            started = time.perf_counter()
            with torch.inference_mode():
                result = layer(hidden, attention_mask=None, position_ids=positions,
                               cache_position=positions[0],
                               position_embeddings=model.model.rotary_emb(hidden, positions),
                               use_cache=False)
            sync()
            per_doc.append(time.perf_counter() - started)
            if active["count"] != len(module_names):
                raise RuntimeError(f"capture hook count mismatch {method} L{layer_index} {stage} doc={i}")
            del hidden, result
        for handle in handles:
            handle.remove()
        handles.clear()
        x_mm.flush()
    finally:
        for handle in handles:
            handle.remove()
    elapsed = sum(per_doc)
    x_hash = rp.sha_memmap(x_mm)
    rec.record(method, stage, layer_index, ",".join(module_names), elapsed,
               detail=json.dumps({"input_sha256": x_hash, "sibling_inputs_equal": True,
                                  "documents": N_DOCS, "sequence_length": SEQ,
                                  "per_doc_median": float(np.median(per_doc)),
                                  "per_doc_p95": float(np.quantile(per_doc, .95, method="higher"))}))
    return x_mm, x_hash


def run_block_to_next(model, layer_index: int, layer, next_layer, hidden_mm,
                      hidden_path: Path, qkv_path: Path, method: str, rec: Recorder):
    next_hidden = new_mm(hidden_path, (N_DOCS, SEQ, model.config.hidden_size))
    next_qkv = new_mm(qkv_path, (N_DOCS, SEQ, next_layer.input_layernorm.weight.numel()))
    positions = torch.arange(SEQ, device=DEVICE)[None]
    peak_reset()
    per_doc = []
    for i in range(N_DOCS):
        hidden = torch.from_numpy(np.array(hidden_mm[i], copy=True)).unsqueeze(0).to(DEVICE)
        sync()
        started = time.perf_counter()
        with torch.inference_mode():
            result = layer(hidden, attention_mask=None, position_ids=positions,
                           cache_position=positions[0],
                           position_embeddings=model.model.rotary_emb(hidden, positions),
                           use_cache=False)
            output = result[0]
            normalized = next_layer.input_layernorm(output)
            next_hidden[i] = output[0].to("cpu").numpy()
            next_qkv[i] = normalized[0].to("cpu").numpy()
        sync()
        per_doc.append(time.perf_counter() - started)
        del hidden, result, output, normalized
    next_hidden.flush(); next_qkv.flush()
    elapsed = sum(per_doc)
    next_hash = rp.sha_memmap(next_qkv)
    rec.record(method, "down_commit_to_next_layer_input", layer_index,
               f"L{layer_index} block + L{layer_index+1} input_layernorm", elapsed,
               detail=json.dumps({"next_qkv_input_sha256": next_hash,
                                  "documents": N_DOCS, "sequence_length": SEQ,
                                  "per_doc_median": float(np.median(per_doc)),
                                  "per_doc_p95": float(np.quantile(per_doc, .95, method="higher"))}))
    return next_hidden, next_qkv, next_hash


def replay_completed_prefix(model, method: str, token_samples, completed: int, rec: Recorder):
    hidden_a_path = ACT_DIR / f"{method}_hidden_a.f16"
    hidden_b_path = ACT_DIR / f"{method}_hidden_b.f16"
    q_a_path = ACT_DIR / f"{method}_q_a.f16"
    q_b_path = ACT_DIR / f"{method}_q_b.f16"
    if completed >= LAYERS:
        return None, None, None, None
    hidden = embed_documents(model, token_samples, hidden_a_path, method, rec)
    hidden_path = hidden_a_path
    if completed:
        for index in range(completed):
            target_path = hidden_b_path if hidden_path == hidden_a_path else hidden_a_path
            target = new_mm(target_path, (N_DOCS, SEQ, model.config.hidden_size))
            layer = model.model.layers[index]
            pos = torch.arange(SEQ, device=DEVICE)[None]
            peak_reset(); started = time.perf_counter()
            for i in range(N_DOCS):
                current = torch.from_numpy(np.array(hidden[i], copy=True)).unsqueeze(0).to(DEVICE)
                with torch.inference_mode():
                    out = layer(current, attention_mask=None, position_ids=pos,
                                cache_position=pos[0],
                                position_embeddings=model.model.rotary_emb(current, pos),
                                use_cache=False)[0]
                    target[i] = out[0].to("cpu").numpy()
                del current, out
            target.flush(); sync()
            rec.record(method, "resume_completed_layer_forward", index, "decoder_layer", time.perf_counter()-started,
                       detail="reconstructed self-prefix activations; completed quantization layer was not rerun")
            close_mm(hidden)
            hidden = target; hidden_path = target_path
    current_layer = model.model.layers[completed]
    q_path = q_a_path if completed % 2 == 0 else q_b_path
    q = capture_norm(current_layer, hidden, q_path, method, completed, rec)
    return hidden, q, hidden_path, q_path


def layer_state_path(method: str, layer_index: int) -> Path:
    path = OUT / "checkpoints" / method / f"layer_{layer_index:02d}.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def list_completed_layers(method: str, code_hash: str, cal_hash: str, protocol_hash: str):
    paths = [layer_state_path(method, i) for i in range(LAYERS)]
    completed = 0
    loaded = []
    missing_seen = False
    previous_state_hash = "0" * 64
    for index, path in enumerate(paths):
        if not path.exists():
            missing_seen = True
            continue
        if missing_seen:
            raise RuntimeError(f"non-contiguous checkpoint sequence for {method}: {path.name}")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        meta = payload["metadata"]
        expected = {"method": method, "layer": index, "code_bundle_sha256": code_hash,
                    "calibration_manifest_sha256": cal_hash, "protocol_sha256": protocol_hash}
        for key, value in expected.items():
            if meta.get(key) != value:
                raise RuntimeError(f"checkpoint {path.name} metadata mismatch: {key}")
        for module_name, state in payload["modules"].items():
            if module_state_hash(state) != state.get("state_sha256"):
                raise RuntimeError(f"checkpoint {path.name} module state hash mismatch: {module_name}")
        layer_hash = quantized_layer_hash(payload["modules"])
        if meta.get("previous_model_state_sha256") != previous_state_hash:
            raise RuntimeError(f"checkpoint chain mismatch before {path.name}")
        if meta.get("layer_quant_state_sha256") != layer_hash:
            raise RuntimeError(f"checkpoint layer hash mismatch: {path.name}")
        if meta.get("selected_alpha_table_sha256") != alpha_table_hash(payload["modules"]):
            raise RuntimeError(f"checkpoint alpha table hash mismatch: {path.name}")
        state_hash = sha_bytes((previous_state_hash + layer_hash).encode("ascii"))
        if meta.get("model_state_sha256") != state_hash:
            raise RuntimeError(f"checkpoint model-state chain hash mismatch: {path.name}")
        previous_state_hash = state_hash
        loaded.append(payload)
        completed += 1
    return completed, loaded


def restore_layer(model, index: int, payload: dict):
    layer = model.model.layers[index]
    for name, state in payload["modules"].items():
        target = layer.get_submodule(name).weight
        value = state["weight"].to(device=target.device, dtype=target.dtype)
        if value.shape != target.shape:
            raise RuntimeError(f"checkpoint weight shape mismatch L{index} {name}")
        with torch.no_grad():
            target.copy_(value)


def save_layer_checkpoint(method: str, index: int, payload: dict):
    path = layer_state_path(method, index)
    tmp = path.with_suffix(".pt.tmp")
    started = time.perf_counter()
    torch.save(payload, tmp)
    with tmp.open("rb+") as f:
        os.fsync(f.fileno())
    os.replace(tmp, path)
    elapsed = time.perf_counter() - started
    return path, elapsed


def module_state_hash(state: dict) -> str:
    h = hashlib.sha256()
    for key in ("weight", "scales", "zeros", "g_idx", "selected_alpha"):
        tensor = state.get(key)
        if tensor is not None:
            h.update(key.encode())
            h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def quantize_native_module(method: str, layer_index: int, name: str, module,
                           hessian, nsamples, x_mm, input_hash, h_hash, rec: Recorder):
    pristine = module.weight.detach().clone()
    start_hash = sha_tensor(pristine)
    start = time.perf_counter(); peak_reset()
    result = native_dynamic_gptq(pristine, hessian, nsamples=nsamples, alpha=1.0)
    sync(); elapsed = time.perf_counter() - start
    alloc, reserved = peak_read()
    rec.record(method, "native_alpha1_full_linear_gptq", layer_index, name, elapsed,
               detail=json.dumps({"input_sha256": input_hash, "H_sha256": h_hash,
                                  "pristine_weight_sha256": start_hash}), peak=(alloc, reserved))
    q = result.weight_fp16
    score_start = time.perf_counter(); peak_reset()
    losses = rp.score_rows(x_mm, pristine, q)
    sync(); score_elapsed = time.perf_counter() - score_start
    alloc, reserved = peak_read()
    rec.record(method, "native_m1_all_row_output_scoring", layer_index, name, score_elapsed,
               detail=f"direct FP32 X@delta.T; all {q.shape[0]} output rows", peak=(alloc, reserved))
    w_mse = (q.float() - pristine.float()).square().mean(dim=1)
    if not (torch.isfinite(q).all() and torch.isfinite(result.scales).all()
            and torch.isfinite(result.zeros).all() and torch.isfinite(losses).all()):
        raise RuntimeError(f"nonfinite native GPTQ output: L{layer_index} {name}")
    expected_g = torch.arange(pristine.shape[1], device=DEVICE, dtype=result.g_idx.dtype) // 128
    if not torch.equal(result.g_idx, expected_g):
        raise RuntimeError(f"unexpected g_idx: L{layer_index} {name}")
    commit_start = time.perf_counter(); peak_reset()
    with torch.no_grad():
        module.weight.copy_(q)
    sync(); commit_elapsed = time.perf_counter() - commit_start
    alloc, reserved = peak_read()
    if sha_tensor(module.weight) != sha_tensor(q):
        raise RuntimeError(f"native M1 commit mismatch L{layer_index} {name}")
    rec.record(method, "whole_linear_commit", layer_index, name, commit_elapsed,
               detail="single full-Linear native GPTQ commit", peak=(alloc, reserved))
    state = {
        "weight": q.detach().cpu().contiguous(),
        "scales": result.scales.detach().cpu().contiguous(),
        "zeros": result.zeros.detach().cpu().contiguous(),
        "g_idx": result.g_idx.detach().cpu().contiguous(),
        "selected_alpha": torch.ones(q.shape[0], dtype=torch.float64),
        "best_loss": losses.detach().cpu().contiguous(),
        "alpha1_loss": losses.detach().cpu().contiguous(),
        "weight_mse_selected": w_mse.detach().cpu().contiguous(),
        "weight_mse_alpha1": w_mse.detach().cpu().contiguous(),
        "candidate_losses": losses.detach().cpu().reshape(1, -1).contiguous(),
        "input_sha256": input_hash,
        "hessian_sha256": h_hash,
        "candidate_seconds": [elapsed],
        "score_seconds": [score_elapsed],
        "row_update_seconds": [],
    }
    state["state_sha256"] = module_state_hash(state)
    del pristine, result, losses, q, w_mse
    return state


def quantize_search_module(method: str, layer_index: int, name: str, module,
                           hessian, prepared, nsamples, x_mm, input_hash, h_hash,
                           rec: Recorder, formal_counts: dict):
    pristine = module.weight.detach().clone()
    pristine_hash = sha_tensor(pristine)
    rows = int(pristine.shape[0])
    best_loss = torch.full((rows,), float("inf"), device=DEVICE, dtype=torch.float64)
    best_alpha = torch.full((rows,), -1.0, device=DEVICE, dtype=torch.float64)
    best_q = torch.empty_like(pristine)
    best_scale = None
    best_zero = None
    best_g_idx = None
    alpha1_loss = None
    alpha1_weight_mse = None
    candidate_loss_rows = []
    candidate_seconds = []
    score_seconds = []
    row_update_seconds = []

    for alpha in GRID:
        peak_reset(); start = time.perf_counter()
        result = optimized_dynamic_gptq(pristine, prepared, alpha=alpha, check_cholesky=(alpha == 1.0))
        sync(); elapsed = time.perf_counter() - start
        alloc, reserved = peak_read()
        candidate_seconds.append(elapsed)
        formal_counts["full_linear_alpha_candidate_executions"] += 1
        formal_counts["row_alpha_score_slots"] += rows
        rec.record(method, "full_linear_alpha_candidate_gptq", layer_index, name, elapsed,
                   alpha=f"{alpha:.1f}", detail=json.dumps({"shape": list(pristine.shape),
                   "pristine_weight_sha256": pristine_hash, "input_sha256": input_hash,
                   "H_sha256": h_hash, "candidate_local_state": True}), peak=(alloc, reserved))
        q = result.weight_fp16
        scale, zero, g_idx = result.scales, result.zeros, result.g_idx
        if not (torch.isfinite(q).all() and torch.isfinite(scale).all() and
                torch.isfinite(zero).all()):
            raise RuntimeError(f"nonfinite M2 candidate L{layer_index} {name} alpha={alpha}")
        expected_g = torch.arange(pristine.shape[1], device=DEVICE, dtype=g_idx.dtype) // 128
        if not torch.equal(g_idx, expected_g):
            raise RuntimeError(f"unexpected g_idx L{layer_index} {name} alpha={alpha}")
        if best_g_idx is None:
            best_g_idx = g_idx.detach().clone()
            best_scale = torch.zeros_like(scale)
            best_zero = torch.zeros_like(zero)
        elif not torch.equal(best_g_idx, g_idx):
            raise RuntimeError(f"candidate g_idx differs L{layer_index} {name} alpha={alpha}")

        peak_reset(); score_start = time.perf_counter()
        losses = rp.score_rows(x_mm, pristine, q)
        sync(); score_elapsed = time.perf_counter() - score_start
        alloc, reserved = peak_read()
        score_seconds.append(score_elapsed)
        if not torch.isfinite(losses).all():
            raise RuntimeError(f"nonfinite row output loss L{layer_index} {name} alpha={alpha}")
        candidate_loss_rows.append(losses.detach().cpu().numpy().astype(np.float64, copy=True))
        rec.record(method, "all_row_direct_output_scoring", layer_index, name, score_elapsed,
                   alpha=f"{alpha:.1f}", detail=f"all {rows} rows; FP32 X@delta.T; X streamed by document chunks",
                   peak=(alloc, reserved))

        if alpha == 1.0:
            alpha1_loss = losses.detach().clone()
            alpha1_weight_mse = (q.float() - pristine.float()).square().mean(dim=1).detach().clone()
            if layer_index in SPOT_LAYERS:
                parity_start = time.perf_counter(); peak_reset()
                reference = native_dynamic_gptq(pristine, hessian, nsamples=nsamples, alpha=1.0)
                sync(); parity_seconds = time.perf_counter() - parity_start
                alloc, reserved = peak_read()
                parity = {
                    "weight": torch.equal(reference.weight_fp16, q),
                    "scale": torch.equal(reference.scales, scale),
                    "zero": torch.equal(reference.zeros, zero),
                    "g_idx": torch.equal(reference.g_idx, g_idx),
                }
                rec.record(method, "alpha1_native_parity_reference", layer_index, name, parity_seconds,
                           alpha="1.0", detail=json.dumps(parity, sort_keys=True), peak=(alloc, reserved))
                formal_counts["alpha1_native_parity_reference_executions"] += 1
                if not all(parity.values()):
                    raise RuntimeError(f"alpha=1 native parity failed L{layer_index} {name}: {parity}")
                del reference

        peak_reset(); sync(); update_start = time.perf_counter()
        better = (losses < best_loss) | ((losses == best_loss) & (alpha > best_alpha))
        best_loss = torch.where(better, losses, best_loss)
        best_alpha = torch.where(better, torch.full_like(best_alpha, alpha), best_alpha)
        best_q[better] = q[better]
        best_scale[better] = scale[better]
        best_zero[better] = zero[better]
        sync(); update_elapsed = time.perf_counter() - update_start
        alloc, reserved = peak_read()
        row_update_seconds.append(update_elapsed)
        rec.record(method, "streaming_per_row_argmin_update", layer_index, name, update_elapsed,
                   alpha=f"{alpha:.1f}", detail="exact loss ties choose larger alpha; only better rows copied",
                   peak=(alloc, reserved))
        del result, q, scale, zero, g_idx, losses, better
        if (formal_counts["full_linear_alpha_candidate_executions"] % 20) == 0:
            write_json_atomic(OUT / "raw" / "formal_progress.json", {
                "method": method, "layer": layer_index, "module": name, "last_alpha": alpha,
                **formal_counts, "updated_utc": datetime.now(timezone.utc).isoformat(),
            })
            print(f"[{method}] L{layer_index:02d} {name} alpha={alpha:.1f} "
                  f"candidates={formal_counts['full_linear_alpha_candidate_executions']}/1680", flush=True)

    if alpha1_loss is None or alpha1_weight_mse is None:
        raise RuntimeError("alpha=1 baseline missing from frozen grid")
    if not torch.all(best_loss <= alpha1_loss):
        raise RuntimeError(f"selected search loss exceeds alpha1 on search input L{layer_index} {name}")
    if sha_tensor(module.weight) != pristine_hash:
        raise RuntimeError(f"candidate evaluation changed target weight before commit L{layer_index} {name}")

    peak_reset(); sync(); start = time.perf_counter()
    stitched_q = best_q.detach().clone()
    stitched_scale = best_scale.detach().clone()
    stitched_zero = best_zero.detach().clone()
    selected_alpha = best_alpha.detach().cpu().numpy().astype(np.float64, copy=True)
    selected_loss = best_loss.detach().clone()
    weight_mse_selected = (stitched_q.float() - pristine.float()).square().mean(dim=1)
    candidate_losses = np.stack(candidate_loss_rows, axis=0)
    sync(); stitch_elapsed = time.perf_counter() - start
    alloc, reserved = peak_read()
    rec.record(method, "full_linear_row_stitch", layer_index, name, stitch_elapsed,
               detail=f"all {rows} output rows; W/scale/zero; shared g_idx", peak=(alloc, reserved))

    peak_reset(); commit_start = time.perf_counter()
    with torch.no_grad():
        module.weight.copy_(stitched_q)
    sync(); commit_elapsed = time.perf_counter() - commit_start
    alloc, reserved = peak_read()
    committed_hash = sha_tensor(module.weight)
    if committed_hash != sha_tensor(stitched_q):
        raise RuntimeError(f"stitched Linear commit mismatch L{layer_index} {name}")
    rec.record(method, "whole_linear_commit", layer_index, name, commit_elapsed,
               detail=json.dumps({"committed_weight_sha256": committed_hash,
                                  "rows": rows, "refresh_after_whole_linear_only": True}), peak=(alloc, reserved))
    state = {
        "weight": stitched_q.detach().cpu().contiguous(),
        "scales": stitched_scale.detach().cpu().contiguous(),
        "zeros": stitched_zero.detach().cpu().contiguous(),
        "g_idx": best_g_idx.detach().cpu().contiguous(),
        "selected_alpha": torch.from_numpy(selected_alpha),
        "best_loss": selected_loss.detach().cpu().contiguous(),
        "alpha1_loss": alpha1_loss.detach().cpu().contiguous(),
        "weight_mse_selected": weight_mse_selected.detach().cpu().contiguous(),
        "weight_mse_alpha1": alpha1_weight_mse.detach().cpu().contiguous(),
        "candidate_losses": torch.from_numpy(candidate_losses),
        "input_sha256": input_hash,
        "hessian_sha256": h_hash,
        "candidate_seconds": candidate_seconds,
        "score_seconds": score_seconds,
        "row_update_seconds": row_update_seconds,
        "stitch_seconds": stitch_elapsed,
        "commit_seconds": commit_elapsed,
        "committed_weight_sha256": committed_hash,
    }
    state["state_sha256"] = module_state_hash(state)
    del pristine, best_loss, best_alpha, best_q, best_scale, best_zero, stitched_q, stitched_scale, stitched_zero
    del selected_loss, weight_mse_selected, alpha1_loss, alpha1_weight_mse, candidate_loss_rows, candidate_losses
    return state


def alpha_table_hash(modules: dict) -> str:
    h = hashlib.sha256()
    for name in sorted(modules):
        h.update(name.encode("utf-8"))
        h.update(modules[name]["selected_alpha"].contiguous().numpy().astype("<f8", copy=False).tobytes())
    return h.hexdigest()


def quantized_layer_hash(modules: dict) -> str:
    h = hashlib.sha256()
    for name in sorted(modules):
        h.update(name.encode("utf-8"))
        h.update(modules[name]["state_sha256"].encode("ascii"))
    return h.hexdigest()


def save_final_model(model, method: str, quant_meta: dict, rec: Recorder):
    final_dir = OUT / method / "checkpoint" / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    peak_reset(); start = time.perf_counter()
    model.save_pretrained(str(final_dir), safe_serialization=True, max_shard_size="5GB")
    sync(); elapsed = time.perf_counter() - start
    alloc, reserved = peak_read()
    rec.record(method, "final_dense_model_save", LAYERS - 1, str(final_dir), elapsed,
               detail="dense-dequant FP16 safetensors checkpoint", peak=(alloc, reserved))
    write_json_atomic(OUT / method / "QUANTIZATION_METADATA.json", quant_meta)
    return final_dir, elapsed


def validate_layer_checkpoints(method: str, code_hash: str, cal_hash: str, protocol_hash: str):
    completed, payloads = list_completed_layers(method, code_hash, cal_hash, protocol_hash)
    if completed != LAYERS:
        raise RuntimeError(f"{method} has {completed}/{LAYERS} complete layer checkpoints")
    expected_names = {name for names in MODULE_ORDERS.values() for name in names}
    total_rows = 0
    module_count = 0
    for layer_index, payload in enumerate(payloads):
        if set(payload["modules"]) != expected_names:
            raise RuntimeError(f"{method} L{layer_index} module inventory mismatch")
        if payload["metadata"]["row_count"] != 12672:
            raise RuntimeError(f"{method} L{layer_index} output row count mismatch")
        for name, state in payload["modules"].items():
            weight = state["weight"]
            scale, zero, g_idx = state["scales"], state["zeros"], state["g_idx"]
            rows, columns = weight.shape
            groups = columns // 128
            if weight.dtype != torch.float16 or scale.shape != (rows, groups) or zero.shape != (rows, groups):
                raise RuntimeError(f"{method} L{layer_index} {name} W/scale/zero shape or dtype mismatch")
            expected_g = torch.arange(columns, dtype=g_idx.dtype) // 128
            if g_idx.shape != (columns,) or not torch.equal(g_idx, expected_g):
                raise RuntimeError(f"{method} L{layer_index} {name} g_idx mismatch")
            for key, tensor in (("weight", weight), ("scales", scale), ("zeros", zero),
                                ("best_loss", state["best_loss"]),
                                ("alpha1_loss", state["alpha1_loss"])):
                if not torch.isfinite(tensor).all():
                    raise RuntimeError(f"{method} L{layer_index} {name} has nonfinite {key}")
            alpha = state["selected_alpha"].numpy()
            if alpha.shape != (rows,):
                raise RuntimeError(f"{method} L{layer_index} {name} selected alpha shape mismatch")
            if method == "M2" and not np.isin(alpha, np.asarray(GRID, dtype=np.float64)).all():
                raise RuntimeError(f"{method} L{layer_index} {name} selected alpha outside frozen grid")
            expected_candidates = len(GRID) if method == "M2" else 1
            if state["candidate_losses"].shape != (expected_candidates, rows):
                raise RuntimeError(f"{method} L{layer_index} {name} candidate loss shape mismatch")
            if not torch.isfinite(state["candidate_losses"]).all():
                raise RuntimeError(f"{method} L{layer_index} {name} candidate losses nonfinite")
            if len(state["input_sha256"]) != 64 or len(state["hessian_sha256"]) != 64:
                raise RuntimeError(f"{method} L{layer_index} {name} missing frozen input/H hash")
            if method == "M2" and torch.any(state["best_loss"] > state["alpha1_loss"]):
                raise RuntimeError(f"M2 selected search loss exceeded alpha=1 L{layer_index} {name}")
            total_rows += rows
            module_count += 1
        del payload
    if module_count != 168 or total_rows != 304128:
        raise RuntimeError(f"{method} module/row inventory mismatch: {module_count}/{total_rows}")
    result = {"pass": True, "method": method, "layers": completed,
              "linear_modules": module_count, "output_rows": total_rows,
              "selected_alpha_grid": GRID if method == "M2" else [1.0],
              "finite_weights_scales_zeros_losses": True,
              "scale_zero_shapes_correct": True, "g_idx_native": True,
              "per_row_search_not_worse_than_alpha1_on_search": True if method == "M2" else None}
    write_json_atomic(OUT / method / "QUANTIZATION_CORRECTNESS.json", result)
    return result


def reload_correctness(final_dir: Path, model, token_samples, method: str, rec: Recorder):
    sample_ids = torch.tensor(token_samples[0]["token_ids"][:64], dtype=torch.long, device=DEVICE)[None]
    with torch.inference_mode():
        before = model(input_ids=sample_ids, use_cache=False).logits.detach().cpu()
    model.to("cpu")
    del model
    gc.collect(); torch.cuda.empty_cache()
    reloaded = load_model(final_dir, rec, method + "_reload")
    with torch.inference_mode():
        after = reloaded(input_ids=sample_ids, use_cache=False).logits.detach().cpu()
    diff = (before.float() - after.float()).abs()
    max_abs = float(diff.max().item())
    mean_abs = float(diff.mean().item())
    close = bool(torch.isfinite(before).all() and torch.isfinite(after).all()
                 and torch.allclose(before, after, rtol=1e-5, atol=1e-6))
    del sample_ids, before, after, diff, reloaded
    gc.collect(); torch.cuda.empty_cache()
    result = {"pass": close, "input_tokens": 64, "rtol": 1e-5, "atol": 1e-6,
              "max_abs_logit_difference": max_abs, "mean_abs_logit_difference": mean_abs}
    write_json_atomic(OUT / method / "RELOAD_CORRECTNESS.json", result)
    if not close:
        raise RuntimeError(f"save/reload output mismatch for {method}: {result}")
    return result


def run_method(method: str, token_samples: list[dict], recorder: Recorder,
               code_hash: str, cal_hash: str, protocol_hash: str):
    method_start = time.perf_counter()
    final_dir = OUT / method / "checkpoint" / "final"
    final_weight = final_dir / "model.safetensors"
    if final_weight.exists():
        completed, payloads = list_completed_layers(method, code_hash, cal_hash, protocol_hash)
        if completed != LAYERS:
            raise RuntimeError(f"final {method} model exists but layer checkpoint set is incomplete")
        del payloads
        quant_check = validate_layer_checkpoints(method, code_hash, cal_hash, protocol_hash)
        model = load_model(final_dir, recorder, method + "_final_resume")
        reload_result = reload_correctness(final_dir, model, token_samples, method, recorder)
        old_report_path = OUT / method / "RUN_REPORT.json"
        old_report = json.loads(old_report_path.read_text(encoding="utf-8")) if old_report_path.exists() else {}
        formal_counts = old_report.get("formal_counts") or {
            "full_linear_alpha_candidate_executions": 1680 if method == "M2" else 0,
            "row_alpha_score_slots": 3041280 if method == "M2" else 0,
            "alpha1_native_parity_reference_executions": 35 if method == "M2" else 0,
            "layers_completed_before_this_process": LAYERS,
        }
        state = {
            **old_report, "method": method, "status": "COMPLETE", "layers": LAYERS,
            "linear_modules": 168, "output_rows": 304128,
            "formal_counts": formal_counts, "final_model_path": str(final_dir),
            "final_model_sha256": sha_file(final_weight),
            "reload_correctness": reload_result, "quantization_correctness": quant_check,
            "resumed_from_final_model": True,
            "wall_seconds_including_load_replay_save": time.perf_counter() - method_start,
            "completed_utc": datetime.now(timezone.utc).isoformat(),
        }
        write_json_atomic(old_report_path, state)
        del model
        gc.collect(); torch.cuda.empty_cache()
        return state
    model = load_model(MODEL_DIR, recorder, method)
    completed, old_payloads = list_completed_layers(method, code_hash, cal_hash, protocol_hash)
    for i, payload in enumerate(old_payloads):
        restore_layer(model, i, payload)
    previous_state_hash = old_payloads[-1]["metadata"]["model_state_sha256"] if old_payloads else "0" * 64
    hidden, qkv_x, hidden_path, qkv_path = replay_completed_prefix(model, method, token_samples, completed, recorder)
    formal_counts = {"full_linear_alpha_candidate_executions": 0,
                     "row_alpha_score_slots": 0,
                     "alpha1_native_parity_reference_executions": 0,
                     "layers_completed_before_this_process": completed}
    if method == "M2":
        # Count completed prior layer candidates for an exact run-wide ledger.
        for payload in old_payloads:
            for state in payload["modules"].values():
                formal_counts["full_linear_alpha_candidate_executions"] += len(state["candidate_seconds"])
                formal_counts["row_alpha_score_slots"] += int(state["weight"].shape[0]) * len(state["candidate_seconds"])
                if payload["metadata"]["layer"] in SPOT_LAYERS:
                    formal_counts["alpha1_native_parity_reference_executions"] += 1

    modules_seen = 0
    for layer_index in range(completed, LAYERS):
        layer = model.model.layers[layer_index]
        layer_started = time.perf_counter()
        if qkv_x is None:
            q_path_current = ACT_DIR / f"{method}_q_{layer_index % 2}.f16"
            qkv_x = capture_norm(layer, hidden, q_path_current, method, layer_index, recorder)
            qkv_path = q_path_current

        layer_modules = {}
        boundary_hashes = {"qkv_input_sha256": rp.sha_memmap(qkv_x)}

        for group in ("qkv", "o", "up_gate", "down"):
            names = MODULE_ORDERS[group]
            if group != "qkv":
                x_path = ACT_DIR / f"{method}_group_input.f16"
                qkv_x, input_hash = run_block_capture(
                    model, layer, hidden, names, x_path, method, layer_index,
                    {"o": "qkv_commit_to_o_input", "up_gate": "o_commit_to_up_gate_input",
                     "down": "up_gate_commit_to_down_input"}[group], recorder)
                if group == "o":
                    stage_key = "o_input_sha256"
                elif group == "up_gate":
                    stage_key = "up_gate_input_sha256"
                else:
                    stage_key = "down_input_sha256"
                boundary_hashes[stage_key] = input_hash
                qkv_path = x_path
            else:
                input_hash = rp.sha_memmap(qkv_x)

            representative = layer.get_submodule(names[0])
            h_log = []
            hessian, nsamples, hstats = rp.build_hessian(
                representative, qkv_x, h_log, f"{method} L{layer_index} {group} shared input")
            if nsamples != N_DOCS:
                raise RuntimeError(f"Hessian nsamples mismatch L{layer_index} {group}: {nsamples}")
            hinv_seconds = 0.0
            prepared = None
            if method == "M2":
                peak_reset(); hinv_start = time.perf_counter()
                prepared = prepare_native_state(hessian, nsamples=nsamples)
                sync(); hinv_seconds = time.perf_counter() - hinv_start
                if prepared.h_sha256 != hstats["sha256"]:
                    raise RuntimeError(f"prepared H hash mismatch L{layer_index} {group}")
                recorder.record(method, "shared_native_hinv_prepare", layer_index, group, hinv_seconds,
                                detail=json.dumps({"H_sha256": hstats["sha256"],
                                                   "input_sha256": input_hash, "nsamples": nsamples}))
            recorder.record(method, "unique_group_hessian_build", layer_index, group, hstats["seconds"],
                            detail=json.dumps({"H_sha256": hstats["sha256"],
                                               "input_sha256": input_hash, "nsamples": nsamples,
                                               "group_input_shared": group in ("qkv", "up_gate")}))

            for name in names:
                module = layer.get_submodule(name)
                full_name = f"model.layers.{layer_index}.{name}"
                modules_seen += 1
                if method == "M1":
                    state = quantize_native_module(method, layer_index, full_name, module,
                                                   hessian, nsamples, qkv_x, input_hash,
                                                   hstats["sha256"], recorder)
                else:
                    state = quantize_search_module(method, layer_index, full_name, module,
                                                   hessian, prepared, nsamples, qkv_x,
                                                   input_hash, hstats["sha256"], recorder,
                                                   formal_counts)
                layer_modules[name] = state
                print(f"[{method}] L{layer_index:02d} committed {name}; rows={module.weight.shape[0]} "
                      f"input={input_hash[:12]} H={hstats['sha256'][:12]}", flush=True)
            if prepared is not None:
                del prepared
            del hessian
            close_mm(qkv_x)
            qkv_x = None

        if layer_index < LAYERS - 1:
            next_hidden_path = ACT_DIR / f"{method}_hidden_{(layer_index + 1) % 2}.f16"
            next_q_path = ACT_DIR / f"{method}_q_{(layer_index + 1) % 2}.f16"
            next_hidden, next_qkv, next_hash = run_block_to_next(
                model, layer_index, layer, model.model.layers[layer_index + 1], hidden,
                next_hidden_path, next_q_path, method, recorder)
            boundary_hashes["next_layer_qkv_input_sha256"] = next_hash
        else:
            next_hidden = None
            next_qkv = None
            next_hash = None

        selected_table_hash = alpha_table_hash(layer_modules)
        current_layer_hash = quantized_layer_hash(layer_modules)
        model_state_hash = sha_bytes((previous_state_hash + current_layer_hash).encode("ascii"))
        layer_payload = {
            "metadata": {
                "method": method, "layer": layer_index,
                "code_bundle_sha256": code_hash,
                "calibration_manifest_sha256": cal_hash,
                "calibration_ordered_token_ids_sha256": json.loads((OUT / "CALIBRATION_VALIDATION.json").read_text(encoding="utf-8"))["ordered_token_ids_sha256_i64le"],
                "protocol_sha256": protocol_hash,
                "previous_model_state_sha256": previous_state_hash,
                "layer_quant_state_sha256": current_layer_hash,
                "model_state_sha256": model_state_hash,
                "selected_alpha_table_sha256": selected_table_hash,
                "boundary_activation_hashes": boundary_hashes,
                "module_count": len(layer_modules),
                "row_count": sum(int(s["weight"].shape[0]) for s in layer_modules.values()),
                "candidate_count": sum(len(s["candidate_seconds"]) for s in layer_modules.values()),
                "rng_state_torch_cpu": torch.get_rng_state(),
                "rng_state_torch_cuda": torch.cuda.get_rng_state_all(),
                "rng_state_numpy": np.random.get_state(),
                "completed_utc": datetime.now(timezone.utc).isoformat(),
            },
            "modules": layer_modules,
        }
        checkpoint_path, checkpoint_seconds = save_layer_checkpoint(method, layer_index, layer_payload)
        recorder.record(method, "atomic_layer_checkpoint_io", layer_index, checkpoint_path.name,
                        checkpoint_seconds, detail=json.dumps({"layer_state_sha256": current_layer_hash,
                                                               "rows": layer_payload["metadata"]["row_count"]}))
        previous_state_hash = model_state_hash
        print(f"[{method}] checkpointed complete layer {layer_index:02d}/23 "
              f"elapsed={time.perf_counter()-layer_started:.1f}s", flush=True)
        write_json_atomic(OUT / "raw" / f"{method}_progress.json", {
            "method": method, "last_completed_layer": layer_index,
            "last_model_state_sha256": model_state_hash,
            **formal_counts, "updated_utc": datetime.now(timezone.utc).isoformat(),
        })
        # Release the just-saved CPU copy before processing the next layer.
        del layer_payload, layer_modules
        if next_hidden is not None:
            close_mm(hidden)
            hidden = next_hidden
            qkv_x = next_qkv
            hidden_path = next_hidden_path
            qkv_path = next_q_path
        else:
            close_mm(hidden)
            hidden = None

    if qkv_x is not None:
        close_mm(qkv_x)
    if hidden is not None:
        close_mm(hidden)

    if modules_seen not in (0, 168):
        raise RuntimeError(f"unexpected modules processed during this process: {modules_seen}")
    quant_check = validate_layer_checkpoints(method, code_hash, cal_hash, protocol_hash)
    quant_meta = {
        "method": method, "representation": "dense-dequant FP16",
        "model_revision": PROTOCOL["model"]["revision"],
        "format": PROTOCOL["quantization"],
        "calibration_manifest_sha256": cal_hash,
        "protocol_sha256": protocol_hash,
        "code_bundle_sha256": code_hash,
        "layer_checkpoints": [str(layer_state_path(method, i).relative_to(OUT)) for i in range(LAYERS)],
        "metadata_location": "Each layer checkpoint stores each Linear's scales, zeros, g_idx, selected alpha, row losses, input/H hashes, and dense FP16 weight.",
        "all_expected_modules": 168,
        "all_expected_output_rows": 304128,
        "quantization_correctness": str((OUT / method / "QUANTIZATION_CORRECTNESS.json").relative_to(OUT)),
    }
    final_dir, save_seconds = save_final_model(model, method, quant_meta, recorder)
    reload_result = reload_correctness(final_dir, model, token_samples, method, recorder)
    elapsed = time.perf_counter() - method_start
    state = {
        "method": method, "status": "COMPLETE", "layers": LAYERS, "linear_modules": 168,
        "output_rows": 304128, "resume_start_layer": completed,
        "formal_counts": formal_counts, "final_model_path": str(final_dir),
        "final_model_sha256": sha_file(final_dir / "model.safetensors"),
        "save_seconds": save_seconds, "reload_correctness": reload_result,
        "quantization_correctness": quant_check,
        "wall_seconds_including_load_replay_save": elapsed,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_json_atomic(OUT / method / "RUN_REPORT.json", state)
    del model
    gc.collect(); torch.cuda.empty_cache()
    return state


def finalize_row_artifacts():
    for method in ("M1", "M2"):
        row_path = OUT / f"{method}_ROW_RESULTS.csv"
        module_path = OUT / f"{method}_MODULE_SUMMARY.csv"
        with row_path.open("w", newline="", encoding="utf-8") as row_f, \
             module_path.open("w", newline="", encoding="utf-8") as module_f:
            row_writer = csv.DictWriter(row_f, fieldnames=["module", "row", "selected_alpha",
                "search_output_mse", "alpha1_output_mse", "relative_output_gain",
                "selected_weight_mse", "alpha1_weight_mse"])
            module_writer = csv.DictWriter(module_f, fieldnames=["module", "layer", "rows",
                "input_sha256", "hessian_sha256", "mean_search_output_mse", "mean_alpha1_output_mse",
                "mean_relative_output_gain", "median_relative_output_gain", "p95_relative_output_gain",
                "rows_improved", "rows_tied", "selected_alpha_histogram", "state_sha256"])
            row_writer.writeheader(); module_writer.writeheader()
            alpha_hist_total = {f"{a:.1f}": 0 for a in GRID}
            layer_loss_file = None
            for layer_index in range(LAYERS):
                payload = torch.load(layer_state_path(method, layer_index), map_location="cpu", weights_only=False)
                loss_arrays = {}
                for name in sorted(payload["modules"]):
                    state = payload["modules"][name]
                    selected = state["best_loss"].numpy().astype(np.float64, copy=False)
                    base = state["alpha1_loss"].numpy().astype(np.float64, copy=False)
                    alpha = state["selected_alpha"].numpy().astype(np.float64, copy=False)
                    wsel = state["weight_mse_selected"].numpy().astype(np.float64, copy=False)
                    wbase = state["weight_mse_alpha1"].numpy().astype(np.float64, copy=False)
                    gain = np.divide(base - selected, base, out=np.zeros_like(base), where=base != 0)
                    for idx, (a, s, b, g, ws, wb) in enumerate(zip(alpha, selected, base, gain, wsel, wbase)):
                        row_writer.writerow({"module": f"model.layers.{layer_index}.{name}", "row": idx,
                            "selected_alpha": f"{a:.1f}", "search_output_mse": f"{s:.12g}",
                            "alpha1_output_mse": f"{b:.12g}", "relative_output_gain": f"{g:.12g}",
                            "selected_weight_mse": f"{ws:.12g}", "alpha1_weight_mse": f"{wb:.12g}"})
                    hist = {f"{a:.1f}": int(np.count_nonzero(alpha == a)) for a in GRID}
                    for a, count in hist.items():
                        alpha_hist_total[a] += count
                    if method == "M2":
                        loss_arrays[name.replace(".", "_")] = state["candidate_losses"].numpy()
                    module_writer.writerow({"module": f"model.layers.{layer_index}.{name}",
                        "layer": layer_index, "rows": len(alpha),
                        "input_sha256": state["input_sha256"], "hessian_sha256": state["hessian_sha256"],
                        "mean_search_output_mse": float(selected.mean()),
                        "mean_alpha1_output_mse": float(base.mean()),
                        "mean_relative_output_gain": float(gain.mean()),
                        "median_relative_output_gain": float(np.median(gain)),
                        "p95_relative_output_gain": float(np.quantile(gain, .95, method="higher")),
                        "rows_improved": int(np.count_nonzero(selected < base)),
                        "rows_tied": int(np.count_nonzero(selected == base)),
                        "selected_alpha_histogram": json.dumps(hist, sort_keys=True),
                        "state_sha256": state["state_sha256"]})
                if method == "M2":
                    raw_dir = OUT / "raw" / "M2_candidate_losses"
                    raw_dir.mkdir(parents=True, exist_ok=True)
                    np.savez(raw_dir / f"layer_{layer_index:02d}.npz", **loss_arrays)
                row_f.flush(); module_f.flush()
                del payload
        if method == "M2":
            dist_path = OUT / "M2_ALPHA_DISTRIBUTION.csv"
            with dist_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=["scope", "alpha", "rows", "fraction"])
                writer.writeheader()
                total = sum(alpha_hist_total.values())
                for alpha, count in alpha_hist_total.items():
                    writer.writerow({"scope": "full_model", "alpha": alpha, "rows": count,
                                     "fraction": count / total})
            write_json_atomic(OUT / "raw" / "M2_alpha_histogram.json", alpha_hist_total)

    return {
        "M1_row_results": "M1_ROW_RESULTS.csv", "M2_row_results": "M2_ROW_RESULTS.csv",
        "M1_module_summary": "M1_MODULE_SUMMARY.csv", "M2_module_summary": "M2_MODULE_SUMMARY.csv",
        "M2_alpha_distribution": "M2_ALPHA_DISTRIBUTION.csv",
        "M2_candidate_loss_arrays": "raw/M2_candidate_losses/layer_XX.npz",
    }


def main():
    if not torch.cuda.is_available() or torch.cuda.get_device_name(0) != "NVIDIA GeForce RTX 4060 Laptop GPU":
        raise RuntimeError("Frozen v18 CUDA device is unavailable")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(20261005)
    np.random.seed(20261005)
    token_samples = token_rows()
    cal_hash = sha_file(CAL_PATH)
    protocol_hash = sha_file(PROTOCOL_PATH)
    expected_cal_hash = "664419273f63df3c0b0746b18bb66b593a5384bbe6ba8afcd6f4905c9479aae3"
    if cal_hash != expected_cal_hash:
        raise RuntimeError("copied v16 frozen calibration manifest hash changed")
    audit = json.loads((OUT / "UPSTREAM_HASH_AUDIT.json").read_text(encoding="utf-8"))
    if audit.get("status") != "PASS":
        raise RuntimeError("upstream frozen source/hash audit is not PASS")
    if json.loads((OUT / "MATCHED_SHAPE_PRECHECK.json").read_text(encoding="utf-8")).get("status") != "PASS":
        raise RuntimeError("matched-shape precheck did not pass")
    if json.loads((OUT / "KV_SHAPE_PRECHECK.json").read_text(encoding="utf-8")).get("status") != "PASS":
        raise RuntimeError("k/v shape addendum did not pass")
    cost = json.loads((OUT / "FINAL_COST_MODEL.json").read_text(encoding="utf-8"))
    if cost.get("status") != "M2_COST_GO" or cost["scenario_hours"]["conservative_M2"] > 48:
        raise RuntimeError("frozen 48-hour M2 cost gate is not GO")
    code_hash = hash_code_bundle()
    rec = Recorder()
    overall_start = time.perf_counter()
    overall_start_utc = datetime.now(timezone.utc).isoformat()
    update_manifest(status="FORMAL_RUNNING", code_bundle_sha256=code_hash,
                    formal_started_utc=datetime.now(timezone.utc).isoformat())
    try:
        m1_state = run_method("M1", token_samples, rec, code_hash, cal_hash, protocol_hash)
        update_manifest(status="M1_COMPLETE_M2_RUNNING", results={
            "matched_shape_precheck": "PASS", "cost_gate": "M2_COST_GO", "m1": "COMPLETE",
            "m2": "RUNNING", "wikitext_evaluation": "NOT_STARTED", "c4_evaluation": "NOT_STARTED"})
        m2_state = run_method("M2", token_samples, rec, code_hash, cal_hash, protocol_hash)
        artifacts = finalize_row_artifacts()
        precheck = json.loads((OUT / "MATCHED_SHAPE_PRECHECK.json").read_text(encoding="utf-8"))
        kv_precheck = json.loads((OUT / "KV_SHAPE_PRECHECK.json").read_text(encoding="utf-8"))
        smoke = json.loads((OUT / "FORMAL_RUNNER_SMOKE.json").read_text(encoding="utf-8"))
        checks = {
            "matched_shape_precheck": precheck["status"] == "PASS" and all(precheck["checks"].values()),
            "kv_full_shape_precheck": kv_precheck["status"] == "PASS",
            "formal_runner_smoke": smoke["status"] == "PASS",
            "m1_168_linears_304128_rows": m1_state["quantization_correctness"]["linear_modules"] == 168 and m1_state["quantization_correctness"]["output_rows"] == 304128,
            "m2_168_linears_304128_rows": m2_state["quantization_correctness"]["linear_modules"] == 168 and m2_state["quantization_correctness"]["output_rows"] == 304128,
            "m2_1680_full_linear_alpha_candidates": m2_state["formal_counts"]["full_linear_alpha_candidate_executions"] == 1680,
            "m2_3041280_row_alpha_score_slots": m2_state["formal_counts"]["row_alpha_score_slots"] == 3041280,
            "m2_35_alpha1_native_spot_references": m2_state["formal_counts"]["alpha1_native_parity_reference_executions"] == 35,
            "m1_reload_output_match": m1_state["reload_correctness"]["pass"],
            "m2_reload_output_match": m2_state["reload_correctness"]["pass"],
        }
        if not all(checks.values()):
            raise RuntimeError(f"formal correctness matrix failed: {checks}")
        write_json_atomic(CHECKS_PATH, {"status": "PASS", "checks": checks,
            "m2_candidate_executions": m2_state["formal_counts"]["full_linear_alpha_candidate_executions"],
            "m2_row_alpha_score_slots": m2_state["formal_counts"]["row_alpha_score_slots"],
            "m2_spot_parity_reference_executions": m2_state["formal_counts"]["alpha1_native_parity_reference_executions"]})
        runtime = {
            "status": "QUANTIZATION_COMPLETE",
            "started_utc": overall_start_utc,
            "m1": m1_state, "m2": m2_state,
            "formal_work_units": {"m2_full_linear_alpha_candidate_executions": 1680,
                "m2_row_alpha_score_slots": 3041280,
                "m2_auxiliary_alpha1_native_parity_references": 35,
                "m1_native_full_linear_gptq_executions": 168,
                "m2_output_rows": 304128},
            "overall_quantization_wall_seconds": time.perf_counter() - overall_start,
            "cuda_peak_allocated_bytes": rec.peak_allocated,
            "cuda_peak_reserved_bytes": rec.peak_reserved,
            "timing_csv": "FORMAL_TIMING.csv",
            "row_and_candidate_artifacts": artifacts,
            "device": torch.cuda.get_device_name(0),
            "torch": torch.__version__,
            "transformers": __import__("transformers").__version__,
            "python": sys.version,
            "platform": platform.platform(),
            "completed_utc": datetime.now(timezone.utc).isoformat(),
        }
        write_json_atomic(FORMAL_RUNTIME, runtime)
        update_manifest(status="QUANTIZATION_COMPLETE_EVALUATION_READY", results={
            "matched_shape_precheck": "PASS", "cost_gate": "M2_COST_GO", "m1": "COMPLETE_RELOAD_PASS",
            "m2": "COMPLETE_RELOAD_PASS", "wikitext_evaluation": "READY", "c4_evaluation": "READY"})
        print(json.dumps({"status": runtime["status"], "m1_seconds": m1_state["wall_seconds_including_load_replay_save"],
                          "m2_seconds": m2_state["wall_seconds_including_load_replay_save"],
                          "total_seconds": runtime["overall_quantization_wall_seconds"],
                          "candidate_executions": 1680, "row_alpha_scores": 3041280,
                          "peak_allocated_bytes": rec.peak_allocated,
                          "peak_reserved_bytes": rec.peak_reserved}, indent=2))
    except Exception as exc:
        update_manifest(status="FORMAL_FAILED_RESUMABLE", formal_failure={
            "type": type(exc).__name__, "message": str(exc),
            "failed_utc": datetime.now(timezone.utc).isoformat()})
        raise
    finally:
        rec.close()


if __name__ == "__main__":
    main()
