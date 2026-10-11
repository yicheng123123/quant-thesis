#!/usr/bin/env python3
"""Kaggle-only 2048-token OmniQuant preflight; never starts formal M3."""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import importlib
import importlib.util
import json
import math
import os
try:
    import resource
except ImportError:  # Windows local preflight has no resource module.
    resource = None
import statistics
import subprocess
import sys
import tempfile
import time
import types
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import torch

from run_layer_timing import TARGETS, move_tree, run_layer_sequence


HERE = Path(__file__).resolve().parent
DEFAULT_CONFIG = HERE / "configs/m3_preflight.json"


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def tree_to_cpu(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().contiguous()
    if isinstance(value, tuple):
        return tuple(tree_to_cpu(item) for item in value)
    if isinstance(value, list):
        return [tree_to_cpu(item) for item in value]
    if isinstance(value, dict):
        return {key: tree_to_cpu(item) for key, item in value.items()}
    return value


def tree_to_device(value: Any, device: torch.device) -> Any:
    return move_tree(value, device)


def validate_calibration(bundle_root: Path, out: Path, config: dict[str, Any]) -> dict[str, Any]:
    path = bundle_root / config["formal_training"]["calibration_manifest"]
    raw_hash = sha_file(path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    problems = []
    if raw_hash != config["formal_training"]["calibration_manifest_sha256"]:
        problems.append("serialized manifest SHA256 mismatch")
    for field, expected in (("nsamples", 128), ("seqlen", 2048), ("seed", 20261005), ("split", "train")):
        if manifest.get(field) != expected:
            problems.append(f"manifest {field} mismatch: {manifest.get(field)!r}")
    if len(manifest.get("samples", [])) != 128:
        problems.append(f"expected 128 samples, got {len(manifest.get('samples', []))}")
    token_records = []
    for index, sample in enumerate(manifest.get("samples", [])):
        ids = sample.get("token_ids", [])
        token_hash = sha_bytes(json.dumps(ids).encode("utf-8"))
        if len(ids) != 2048:
            problems.append(f"sample {index} has {len(ids)} tokens")
        if token_hash != sample.get("token_sha256"):
            problems.append(f"sample {index} token hash mismatch")
        token_records.append({"sample_index": index,
                              "source_article_index": sample.get("source_article_index"),
                              "article_sha256": sample.get("article_sha256"),
                              "token_count": len(ids), "token_sha256": token_hash})
    identity = {
        "status": "PASS" if not problems else "FAIL",
        "manifest_sha256": raw_hash,
        "samples": len(manifest.get("samples", [])),
        "tokens_per_sample": sorted({len(row.get("token_ids", [])) for row in manifest.get("samples", [])}),
        "seed": manifest.get("seed"),
        "split": manifest.get("split"),
        "ordered_sample_metadata_sha256": sha_bytes(json.dumps(token_records, separators=(",", ":"), sort_keys=True).encode()),
        "sample_hashes": token_records,
        "problems": problems,
        "resampling": False,
        "tokenizer_invoked": False,
    }
    write_json(out / "CALIBRATION_CHECK.json", identity)
    if problems:
        raise RuntimeError("frozen calibration identity failed; stopping before model use")
    return manifest


def verify_step_count(bundle_root: Path, config: dict[str, Any], out: Path) -> dict[str, Any]:
    source = bundle_root / "omniquant_port/upstream/quantize/omniquant.py"
    lines = source.read_text(encoding="utf-8").splitlines()
    snippets = {
        "blocks": "smoke_layer_limit = getattr(args, \"smoke_layer_limit\", len(layers))",
        "layer_loop": "for i in range(smoke_layer_limit):",
        "epochs": "for epochs in range(args.epochs):",
        "sample_batches": "for j in range(args.nsamples//args.batch_size):",
        "update": "norm = loss_scaler(loss, optimizer,parameters= get_omni_parameters(qlayer, use_shift)).cpu()",
        "batch_guard": "if args.batch_size != 1:",
    }
    locations = {}
    for key, snippet in snippets.items():
        found = [i + 1 for i, line in enumerate(lines) if snippet in line]
        if len(found) != 1:
            raise RuntimeError(f"cannot uniquely verify formal loop snippet {key}: {found}")
        locations[key] = {"line": found[0], "snippet": snippet}
    formal = config["formal_training"]
    module_map = json.loads((bundle_root / "manifests/v20_evidence/QWEN25_MODULE_MAP.json").read_text(encoding="utf-8"))
    if module_map.get("status") != "PASS" or module_map.get("layers") != 24 or module_map.get("target_projection_count") != 168:
        raise RuntimeError("frozen v20 module map does not prove the expected 24-layer/168-projection model")
    layers = int(module_map["layers"])
    nsamples = int(formal["samples"])
    batch = int(formal["batch_size"])
    epochs = int(formal["epochs"])
    if nsamples % batch:
        raise RuntimeError("formal sample count is not divisible by the frozen batch size")
    steps_per_epoch_per_block = nsamples // batch
    total = layers * epochs * steps_per_epoch_per_block
    record = {
        "status": "PASS",
        "source_path": "omniquant_port/upstream/quantize/omniquant.py",
        "source_sha256": sha_file(source),
        "decoder_layers": layers,
        "target_projections": int(module_map["target_projection_count"]),
        "samples_per_epoch": nsamples,
        "batch_size": batch,
        "steps_per_sample_per_epoch": 1,
        "optimizer_updates_per_block_per_epoch": steps_per_epoch_per_block,
        "optimizer_updates_per_block": epochs * steps_per_epoch_per_block,
        "optimizer_updates_per_epoch_all_blocks": layers * steps_per_epoch_per_block,
        "total_optimizer_updates": total,
        "formula": "24 decoder blocks * 20 epochs * (128 calibration samples / batch size 1) = 61,440 optimizer updates",
        "source_locations": locations,
        "formal_epochs_unchanged": True,
        "preflight_smoke_step_limits_are_not_included": True,
    }
    write_json(out / "FORMAL_STEP_COUNT.json", record)
    return record


def load_exact_model(model_id: str, revision: str, model_path: str | None,
                     device: torch.device, out: Path):
    from huggingface_hub import snapshot_download
    from transformers import AutoModelForCausalLM

    download_seconds = 0.0
    if model_path:
        snapshot = Path(model_path).expanduser().resolve()
        sidecar = snapshot / "MODEL_REVISION.json"
        sidecar_revision = None
        if sidecar.is_file():
            sidecar_revision = json.loads(sidecar.read_text(encoding="utf-8")).get("revision")
        if snapshot.name != revision and sidecar_revision != revision:
            raise RuntimeError("local model path must be a snapshot named by the frozen revision or contain MODEL_REVISION.json")
        actual_revision = sidecar_revision or snapshot.name
    else:
        started = time.perf_counter()
        snapshot = Path(snapshot_download(repo_id=model_id, revision=revision)).resolve()
        download_seconds = time.perf_counter() - started
        actual_revision = snapshot.name
    if actual_revision != revision:
        raise RuntimeError(f"model snapshot revision mismatch: {actual_revision}")
    model_lock = json.loads((HERE / "manifests/MODEL_FILE_HASHES.json").read_text(encoding="utf-8"))
    if model_lock.get("model_id") != model_id or model_lock.get("revision") != revision:
        raise RuntimeError("bundled model file hash lock does not match the frozen model identity")
    model_file_checks = []
    for relative, expected in model_lock["files"].items():
        path = snapshot / relative
        actual_hash = sha_file(path) if path.is_file() else None
        actual_bytes = path.stat().st_size if path.is_file() else None
        passed = actual_hash == expected["sha256"] and actual_bytes == expected["bytes"]
        model_file_checks.append({"path": relative, "bytes": actual_bytes,
                                  "sha256": actual_hash, "pass": passed})
    if not all(row["pass"] for row in model_file_checks):
        raise RuntimeError("downloaded/local model file SHA256 differs from the v20 frozen snapshot")

    started = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        str(snapshot), torch_dtype=torch.float16, low_cpu_mem_usage=True,
        attn_implementation="sdpa", local_files_only=True,
    ).eval()
    model_load_seconds = time.perf_counter() - started
    commit_hash = getattr(model.config, "_commit_hash", None)
    if commit_hash not in (None, revision):
        raise RuntimeError(f"loaded model config commit hash mismatch: {commit_hash}")
    if commit_hash is None and snapshot.name != revision:
        raise RuntimeError("loaded local model lacks a verifiable frozen commit hash")
    torch.cuda.synchronize(device)
    transfer_started = time.perf_counter()
    model = model.to(device).eval()
    torch.cuda.synchronize(device)
    transfer_seconds = time.perf_counter() - transfer_started
    identity_files = []
    for path in sorted(snapshot.rglob("*")):
        if path.is_file() and (path.suffix == ".safetensors" or path.name.startswith("pytorch_model")
                               or path.name in {"config.json", "model.safetensors.index.json"}):
            identity_files.append({"path": path.relative_to(snapshot).as_posix(),
                                   "bytes": path.stat().st_size, "sha256": sha_file(path)})
    identity = {
        "status": "PASS", "model_id": model_id, "requested_revision": revision,
        "actual_snapshot_revision": actual_revision, "config_commit_hash": commit_hash,
        "snapshot_path": str(snapshot), "dtype": str(next(model.parameters()).dtype),
        "model_type": model.config.model_type, "decoder_layers": len(model.model.layers),
        "model_file_hashes": identity_files,
        "frozen_model_file_hash_checks": model_file_checks,
        "download_seconds": download_seconds, "from_pretrained_seconds": model_load_seconds,
        "weights_to_cuda0_seconds": transfer_seconds,
    }
    write_json(out / "MODEL_IDENTITY.json", identity)
    if model.config.model_type != "qwen2" or len(model.model.layers) != 24:
        raise RuntimeError("model architecture does not match frozen Qwen2.5-0.5B-Instruct")
    return model, identity


def measure_dense_export_proxy(model, out: Path, device: torch.device) -> dict[str, Any]:
    """Measure FP16 state serialization/reload as an export IO proxy, without evaluation."""
    with tempfile.TemporaryDirectory(prefix=".dense_export_proxy_", dir=out) as temp_name:
        target = Path(temp_name) / "dense_fp16_state.pt"
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
        torch.cuda.synchronize(device)
        cpu_copy_s = time.perf_counter() - started
        started = time.perf_counter()
        torch.save(state, target)
        torch.cuda.synchronize(device)
        save_s = time.perf_counter() - started
        file_bytes = target.stat().st_size
        del state
        gc.collect()
        started = time.perf_counter()
        loaded = torch.load(target, map_location="cpu", weights_only=True)
        reload_s = time.perf_counter() - started
        loaded_keys = len(loaded)
        del loaded
        gc.collect()
    result = {
        "status": "PASS", "proxy": "FP16 base-model state CPU copy, serialization, and reload",
        "cpu_copy_seconds": cpu_copy_s, "serialization_seconds": save_s,
        "reload_seconds": reload_s, "file_bytes": file_bytes, "state_tensor_count": loaded_keys,
        "not_equivalent_to": "OmniQuant per-layer W4 dequantization/export; this is a measured dense-state IO proxy only",
    }
    write_json(out / "DENSE_EXPORT_IO_PROXY.json", result)
    return result


def capture_formal_l0_inputs(model, input_ids: list[list[int]], device: torch.device,
                             out: Path) -> list[float]:
    """Measure the real token embedding+RoPE input-capture boundary used before block 0."""
    class StopAtLayer0(Exception):
        pass

    captured: list[torch.Tensor] = []

    def stop_hook(_module, args, kwargs):
        captured.append(args[0].detach().cpu())
        raise StopAtLayer0()

    handle = model.model.layers[0].register_forward_pre_hook(stop_hook, with_kwargs=True)
    times = []
    try:
        for ids in input_ids:
            torch.cuda.synchronize(device)
            started = time.perf_counter()
            try:
                with torch.inference_mode():
                    model(torch.tensor(ids, dtype=torch.long, device=device).view(1, -1), use_cache=False)
            except StopAtLayer0:
                pass
            torch.cuda.synchronize(device)
            times.append(time.perf_counter() - started)
    finally:
        handle.remove()
    if len(captured) != len(input_ids):
        raise RuntimeError("did not capture each real sample at the L0 dependency boundary")
    return times


def capture_representative_layers(model, sample_ids: list[list[int]], layers: list[int],
                                  device: torch.device):
    captured: dict[int, dict[str, list[Any]]] = {
        index: {"inputs": [], "kwargs": [], "outputs": []} for index in layers
    }
    handles = []
    for index in layers:
        def make_pre_hook(layer_index):
            def hook(_module, args, kwargs):
                captured[layer_index]["inputs"].append(args[0].detach().cpu().contiguous())
                captured[layer_index]["kwargs"].append(tree_to_cpu(dict(kwargs)))
            return hook
        def make_post_hook(layer_index):
            def hook(_module, _args, output):
                hidden = output[0] if isinstance(output, (tuple, list)) else output
                captured[layer_index]["outputs"].append(hidden.detach().cpu().contiguous())
            return hook
        handles.append(model.model.layers[index].register_forward_pre_hook(make_pre_hook(index), with_kwargs=True))
        handles.append(model.model.layers[index].register_forward_hook(make_post_hook(index)))

    per_sample_seconds = []
    try:
        for ids in sample_ids:
            torch.cuda.synchronize(device)
            started = time.perf_counter()
            with torch.inference_mode():
                model(torch.tensor(ids, dtype=torch.long, device=device).view(1, -1), use_cache=False)
            torch.cuda.synchronize(device)
            per_sample_seconds.append(time.perf_counter() - started)
    finally:
        for handle in handles:
            handle.remove()
    expected = len(sample_ids)
    for index in layers:
        for name in ("inputs", "kwargs", "outputs"):
            if len(captured[index][name]) != expected:
                raise RuntimeError(f"L{index} captured {len(captured[index][name])} {name}; expected {expected}")
        if any(item.shape[1] != 2048 for item in captured[index]["inputs"]):
            raise RuntimeError(f"L{index} capture did not retain sequence length 2048")
    return captured, per_sample_seconds


def parity_checks(model, captured, layers: list[int], device: torch.device, upstream: Path, out: Path) -> dict[str, Any]:
    sys.path.insert(0, str(upstream))
    from models.int_qwen2_layer import QuantQwen2DecoderLayer
    from quantize.utils import set_quant_state
    quant_params = {"n_bits": 4, "per_channel_axes": [0], "symmetric": False,
                    "dynamic_method": "per_channel", "group_size": 128, "lwc": True,
                    "disable_zero_point": False}
    act_params = {"n_bits": 16, "per_channel_axes": [], "symmetric": False, "dynamic_method": "per_token"}
    rows = []
    for index in layers:
        native = model.model.layers[index]
        port = QuantQwen2DecoderLayer(model.config, native,
              type("AdapterArgs", (), {"weight_quant_params": quant_params, "act_quant_params": act_params})()).to(device).eval()
        set_quant_state(port, weight_quant=False, act_quant=False)
        hidden = captured[index]["inputs"][0].to(device)
        kwargs = tree_to_device(captured[index]["kwargs"][0], device)
        reference = captured[index]["outputs"][0].to(device)
        with torch.inference_mode():
            candidate = port(hidden, **kwargs)[0]
        difference = (candidate.float() - reference.float()).abs()
        row = {"layer": index, "sequence_length": int(hidden.shape[1]),
               "max_abs_diff": float(difference.max().item()),
               "mean_abs_diff": float(difference.mean().item()),
               "allclose_rtol_1e-5_atol_1e-6": bool(torch.allclose(candidate, reference, rtol=1e-5, atol=1e-6)),
               "bitwise_equal": bool(torch.equal(candidate, reference))}
        rows.append(row)
        del port
        torch.cuda.empty_cache()
    result = {"status": "PASS" if all(row["allclose_rtol_1e-5_atol_1e-6"] for row in rows) else "FAIL",
              "source": "v20 QuantQwen2DecoderLayer; native frozen model outputs from same 2048-token samples",
              "layers": rows}
    write_json(out / "QUANT_OFF_PARITY.json", result)
    return result


def percentile(values: list[float], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def summarize_runtime(out: Path, formal_steps: dict[str, Any], layer_results: dict[int, dict[str, Any]],
                      capture_l0: list[float], model_identity: dict[str, Any],
                      full_capture_times: list[float], dense_export_proxy: dict[str, Any],
                      env: dict[str, Any], config: dict[str, Any],
                      session_budget_hours: float | None = None) -> dict[str, Any]:
    rep_weights = {int(k): int(v) for k, v in config["preflight"]["representative_layer_cost_weights"].items()}
    layer_medians = {li: float(layer_results[li]["measured_summary"]["total_s"]["median"]) for li in rep_weights}
    layer_means = {li: float(layer_results[li]["measured_summary"]["total_s"]["mean"]) for li in rep_weights}
    measured_total = [float(row["total_s"]) for result in layer_results.values()
                      for row in result["rows"] if not row["warmup_excluded"]]
    if not measured_total:
        raise RuntimeError("no measured optimizer step rows")
    total_updates = int(formal_steps["total_optimizer_updates"])
    layers_count = int(formal_steps["decoder_layers"])
    samples = int(config["formal_training"]["samples"])
    n_rep = sum(rep_weights.values())
    central_train = 0.0
    fixed_components = {"teacher_cache": 0.0, "transition": 0.0, "final_quantization": 0.0,
                        "scale_registration": 0.0, "checkpoint_io": 0.0, "block_overhead": 0.0}
    for li, weight in rep_weights.items():
        result = layer_results[li]
        per_block_updates = int(config["formal_training"]["epochs"]) * samples
        central_train += weight * per_block_updates * layer_means[li]
        fixed_components["teacher_cache"] += weight * samples * float(result["teacher_forward_per_sample_seconds"])
        fixed_components["transition"] += weight * samples * float(result["layer_transition_per_sample_seconds"])
        fixed_components["final_quantization"] += weight * float(result["final_quantization_seconds"])
        fixed_components["scale_registration"] += weight * float(result["scale_registration_seconds"])
        n_observed = int(result["updates_observed"])
        observed_ckpt = float(result["checkpoint_io_seconds"])
        # OmniQuant overwrites a cumulative parameter state after every block.
        # A triangular serialization count approximates the resulting write volume.
        fixed_components["checkpoint_io"] += weight * observed_ckpt * (layers_count + 1) / 2
        fixed_components["block_overhead"] += weight * max(0.0, float(result["other_block_overhead_seconds"])) * samples / max(1, n_observed)

    train_opt = min(layer_medians.values()) * total_updates
    train_central = central_train
    pooled_p95 = percentile(measured_total, 95)
    train_cons = pooled_p95 * total_updates * float(config["preflight"]["conservative_safety_multiplier"])

    # Initial official block-0 capture is separately measured at the true catch boundary.
    capture_per_sample = statistics.mean(capture_l0)
    formal_capture_s = capture_per_sample * samples
    # Full native prefix capture is diagnostic input preparation for these three rows,
    # not part of formal sequential calibration and therefore excluded from totals.
    load_s = float(model_identity["download_seconds"] + model_identity["from_pretrained_seconds"]
                   + model_identity["weights_to_cuda0_seconds"])
    fixed = {
        "initial_model_load": load_s,
        "calibration_capture": formal_capture_s,
        "teacher_cache_prep": fixed_components["teacher_cache"],
        "optimizer_training": train_central,
        "layer_transition": fixed_components["transition"],
        "final_quantization": fixed_components["final_quantization"],
        "scale_registration": fixed_components["scale_registration"],
        "checkpoint_io": fixed_components["checkpoint_io"],
        "block_overhead": fixed_components["block_overhead"],
        "dense_export": float(dense_export_proxy["cpu_copy_seconds"] + dense_export_proxy["serialization_seconds"]),
        "dense_export_reload_proxy": float(dense_export_proxy["reload_seconds"]),
        "diagnostic_full_native_capture_not_charged": sum(full_capture_times),
    }
    known_extras = (load_s + formal_capture_s + fixed_components["teacher_cache"]
                    + fixed_components["transition"] + fixed_components["final_quantization"]
                    + fixed_components["scale_registration"] + fixed_components["checkpoint_io"]
                    + fixed_components["block_overhead"] + fixed["dense_export"]
                    + fixed["dense_export_reload_proxy"])
    optimistic_s = train_opt + known_extras
    central_s = train_central + known_extras
    conservative_s = train_cons + known_extras
    breakdown = []
    for component, seconds in fixed.items():
        if component == "diagnostic_full_native_capture_not_charged":
            breakdown.append({"component": component, "estimated_wall_h": seconds / 3600,
                              "evidence_source": "measured 8 frozen samples through native full FP16 model to create diagnostic L12/L23 inputs",
                              "confidence": "measured, diagnostic only; excluded from formal totals"})
        elif component == "dense_export":
            breakdown.append({"component": component, "estimated_wall_h": seconds / 3600,
                              "evidence_source": "measured full FP16 state CPU copy and serialization proxy; W4-to-dense conversion itself not run",
                              "confidence": "low; actual dequantization is covered by the safety margin"})
        elif component == "dense_export_reload_proxy":
            breakdown.append({"component": component, "estimated_wall_h": seconds / 3600,
                              "evidence_source": "measured CPU reload of the FP16 export proxy",
                              "confidence": "low; IO proxy"})
        else:
            breakdown.append({"component": component, "estimated_wall_h": seconds / 3600,
                              "evidence_source": "Kaggle v20.1 measured component, extrapolated only where noted",
                              "confidence": "medium" if component in {"optimizer_training", "teacher_cache_prep", "layer_transition"} else "low"})
    with (out / "FORMAL_M3_COST_BREAKDOWN.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["status", "component", "estimated_wall_h", "evidence_source", "confidence"])
        writer.writeheader()
        writer.writerows({"status": "MEASURED_DIAGNOSTIC" if row["component"] == "diagnostic_full_native_capture_not_charged" else "ESTIMATED",
                          **row} for row in breakdown)
    memory_peaks = {li: layer_results[li]["memory"]["peak_reserved_bytes"] / (1024 ** 3) for li in rep_weights}
    max_reserved = max(memory_peaks.values())
    correctness = bool(all(result["all_losses_finite"] and result["all_lwc_gradients_finite"]
                           and result["all_steps_have_nonzero_lwc_gradient"] and result["any_lwc_parameter_changed"]
                           and result["lwc_parameters_present"] for result in layer_results.values()))
    memory_pass = max_reserved <= float(config["preflight"]["frozen_v20_reserved_vram_gate_gib"])
    runtime_gate = conservative_s <= float(config["preflight"]["frozen_formal_runtime_gate_hours"]) * 3600
    ready = correctness and memory_pass and runtime_gate
    # Session budgets are account/runtime settings and are not inferable from GPU VRAM.
    session_feasibility = ("KAGGLE_SINGLE_SESSION_FEASIBLE" if session_budget_hours is not None
                           and conservative_s <= session_budget_hours * 3600
                           else "KAGGLE_SINGLE_SESSION_REQUIRES_RESUME")
    recalc = {
        "status": "PASS",
        "formal_optimizer_updates": total_updates,
        "representative_layers": [0, 12, 23],
        "representative_layer_weights": rep_weights,
        "training_step_mean_seconds_by_layer": layer_means,
        "training_step_median_seconds_by_layer": layer_medians,
        "pooled_measured_step_p95_seconds": pooled_p95,
        "optimistic_training_seconds": train_opt,
        "central_training_seconds": train_central,
        "conservative_training_seconds_before_margin": pooled_p95 * total_updates,
        "conservative_safety_multiplier": config["preflight"]["conservative_safety_multiplier"],
        "cost_components_seconds": fixed,
        "optimistic_total_hours": optimistic_s / 3600,
        "central_total_hours": central_s / 3600,
        "conservative_total_hours": conservative_s / 3600,
        "cost_formula_note": "Training uses measured 2048-token step durations and three equal 8-layer strata; teacher, transition and checkpoint components are added separately. Native full-model target-input capture is diagnostic and excluded. Dense-export time was not measured; the conservative safety multiplier covers this uncertainty but is not a measured export cost.",
        "memory_peak_reserved_gib_by_layer": memory_peaks,
        "maximum_peak_reserved_gib": max_reserved,
        "memory_gate_pass": memory_pass,
        "correctness_gate_pass": correctness,
        "runtime_gate_pass_le_24h": runtime_gate,
        "kaggle_single_session_status": session_feasibility,
        "single_session_budget_hours": session_budget_hours,
        "single_session_reason": ("Measured conservative runtime compared with the explicitly supplied budget."
                                  if session_budget_hours is not None else
                                  "No Kaggle runtime-budget API was available; retain block-level checkpoint/resume for a formal run."),
        "verdict": "LOCAL_M3_FORMAL_GO" if ready else "LOCAL_M3_FORMAL_NO_GO",
    }
    write_json(out / "LOCAL_COST_RECALC.json", recalc)
    write_json(out / "MEMORY_FEASIBILITY.json", {
        "status": "PASS" if memory_pass else "FAIL",
        "gate_reserved_vram_gib": config["preflight"]["frozen_v20_reserved_vram_gate_gib"],
        "peak_reserved_gib_by_layer": memory_peaks,
        "maximum_peak_reserved_gib": max_reserved,
        "gpu_count": env.get("gpu_count"),
        "selected_device": "cuda:0",
        "kaggle_single_session_status": session_feasibility,
        "full_model_dense_export_memory": "not measured; excluded from the 2048 optimizer memory gate",
    })
    return recalc


def write_report(out: Path, env: dict[str, Any], model_identity: dict[str, Any],
                 formal_steps: dict[str, Any], calibration: dict[str, Any],
                 parity: dict[str, Any], lwc: dict[str, Any], layer_results: dict[int, dict[str, Any]],
                 sequence: dict[str, Any], recalc: dict[str, Any], verdict: str) -> None:
    gpu = env["gpus"][0] if env.get("gpus") else {}
    with (out / "CALIBRATION_CAPTURE_TIMING.csv").open(newline="", encoding="utf-8") as stream:
        capture_rows = list(csv.DictReader(stream))
    official_capture = next(row for row in capture_rows if row["stage"] == "official_l0_input_capture_boundary")
    lines = [
        "# v20.1-Kaggle 2048-token matched-shape preflight",
        "",
        f"**Verdict: `{verdict}`**",
        "",
        "No formal 20-epoch training, WT2 PPL, or C4 PPL was run.",
        "",
        "## Measured environment and identity",
        "",
        f"1. GPU: `{gpu.get('name', 'unknown')}`; GPU count `{env.get('gpu_count')}`; selected device `cuda:0`.",
        f"2. GPU total VRAM: `{gpu.get('total_vram_gib', 'unknown')}` GiB.",
        f"3. OmniQuant commit: `{model_identity.get('actual_upstream_commit', 'feffe8ea87d80f7bb57b6e25e7cff9dc950fcc14')}`; v20 patch files verified by bundle hashes.",
        f"4. Model: `{model_identity['model_id']}` at `{model_identity['actual_snapshot_revision']}`; config commit `{model_identity.get('config_commit_hash')}`.",
        f"5. Calibration hash: `{calibration['manifest_sha256']}`; {calibration['samples']} samples × {calibration['tokens_per_sample'][0]} tokens, in frozen order.",
        f"6. Formal update count: `{formal_steps['total_optimizer_updates']:,}` = 24 × 20 × (128 / 1).",
        "",
        "## 2048-token layer timings",
        "",
        "| Layer | Measured updates | Median total (s) | P95 total (s) | Median student (s) | Median backward (s) | Peak allocated (GiB) | Peak reserved (GiB) |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for layer in (0, 12, 23):
        result = layer_results[layer]
        summary = result["measured_summary"]
        lines.append(f"| L{layer} | {result['measured_steps']} | {summary['total_s']['median']:.4f} | {summary['total_s']['p95']:.4f} | {summary['student_s']['median']:.4f} | {summary['backward_s']['median']:.4f} | {result['memory']['peak_allocated_bytes']/(1024**3):.3f} | {result['memory']['peak_reserved_bytes']/(1024**3):.3f} |")
    slowest = max(layer_results, key=lambda index: layer_results[index]["measured_summary"]["total_s"]["mean"])
    sequence_growth = sequence.get("memory_growth_diagnostic", {})
    lines += [
        "",
        f"- Slowest representative layer by measured mean total step: **L{slowest}**.",
        f"- LWC correctness: **{lwc['status']}**; finite nonzero gradients, an observed parameter update, and finite reconstruction losses are checked on real 2048-token inputs.",
        f"- Quant-off native/port 2048 parity: **{parity['status']}**.",
        f"- OOM: **no** for completed target runs.",
        f"- L0 eight-step allocated growth: `{sequence_growth.get('allocated_growth_gib', 'unknown')}` GiB; monotonic non-decreasing `{sequence_growth.get('allocated_monotonic_non_decreasing', 'unknown')}`; unexplained-growth warning `{sequence_growth.get('unexplained_growth_warning', 'unknown')}`.",
        "",
        "## Cost and limits",
        "",
        f"- L0 official capture boundary: {float(official_capture['mean_s']):.4f} s/sample; full 128-sample extrapolation is in `FORMAL_M3_COST_BREAKDOWN.csv`.",
        f"- Central total estimate: **{recalc['central_total_hours']:.3f} h**.",
        f"- Conservative total estimate: **{recalc['conservative_total_hours']:.3f} h** with ×{recalc['conservative_safety_multiplier']:.2f} measured-step safety multiplier.",
        f"- 24-hour gate: **{'PASS' if recalc['runtime_gate_pass_le_24h'] else 'FAIL'}**.",
        f"- Kaggle single-session status: **{recalc['kaggle_single_session_status']}**. {recalc['single_session_reason']}",
        "- Layer weights in the cost projection are three equal eight-layer strata (L0, L12, L23); unsampled within-stratum variation is not measured.",
        "- L12/L23 inputs use native FP16 full-model prefix activations. The preflight does not claim they match a formal run's sequentially quantized prefixes.",
        "- Dense export cost uses a measured FP16 state serialization/reload proxy; per-layer W4-to-dense conversion itself was not run. The stated safety margin covers this unmeasured operation but does not turn it into a direct measurement.",
        "",
        "## Scope decision",
        "",
        "This preflight supports a runtime/memory readiness decision only. It contains no formal M3 checkpoint and no M3 quality result. A READY verdict authorizes a later formal calibration request; it does not establish quality relative to M2.",
    ]
    (out / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle-root", type=Path, default=HERE)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=Path("range_w4_omniquant_kaggle_v20_1"))
    parser.add_argument("--model-path", default=None,
                        help="Optional local snapshot; must be named by the exact revision or carry MODEL_REVISION.json")
    parser.add_argument("--session-budget-hours", type=float, default=None,
                        help="Observed Kaggle session budget; omitted means single-session feasibility is not assumed")
    args = parser.parse_args()
    bundle_root = args.bundle_root.resolve()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    write_json(out / "PROTOCOL.json", {
        "config": config,
        "config_sha256": sha_file(args.config.resolve()),
        "bundle_lock_sha256": sha_file(bundle_root / "hashes/EXPECTED_FILES.json"),
        "formal_steps": "verified in FORMAL_STEP_COUNT.json from the shipped v20 loop source",
        "preflight_only": True,
        "formal_training_executed": False,
        "formal_ppl_executed": False,
    })
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["OMP_NUM_THREADS"] = os.environ.get("OMP_NUM_THREADS", "4")
    torch.manual_seed(int(config["formal_training"]["seed"]))
    np.random.seed(int(config["formal_training"]["seed"]))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    # Local package and input checks are safe before any model work.
    verification = subprocess.run(
        [sys.executable, str(bundle_root / "verify_bundle.py"), "--bundle-root", str(bundle_root),
         "--output", str(out / "BUNDLE_VERIFY.json")], capture_output=True, text=True,
    )
    if verification.returncode != 0:
        write_json(out / "PREFLIGHT_VERDICT.json", {"verdict": "LOCAL_M3_FORMAL_NO_GO",
                                                     "failure": "bundle verification failed",
                                                     "formal_training_executed": False, "formal_ppl_executed": False})
        raise RuntimeError("bundle verification failed; inspect BUNDLE_VERIFY.json")
    formal_steps = verify_step_count(bundle_root, config, out)
    shutil.copy2(bundle_root / "M3_RESUME_PLAN.md", out / "M3_RESUME_PLAN.md")
    shutil.copy2(bundle_root / "KAGGLE_COMMANDS.md", out / "KAGGLE_COMMANDS.md")
    try:
        manifest = validate_calibration(bundle_root, out, config)
    except Exception as exc:
        write_json(out / "PREFLIGHT_VERDICT.json", {"verdict": "LOCAL_M3_FORMAL_NO_GO",
                                                     "failure": f"{type(exc).__name__}: {exc}",
                                                     "formal_training_executed": False, "formal_ppl_executed": False})
        raise

    environment_run = subprocess.run(
        [sys.executable, str(bundle_root / "environment_check.py"), "--output",
         str(out / "LOCAL_ENVIRONMENT.json"), "--disk-path", str(out)], capture_output=True, text=True,
    )
    if environment_run.returncode != 0:
        write_json(out / "PREFLIGHT_VERDICT.json", {"verdict": "LOCAL_M3_FORMAL_NO_GO",
                                                     "failure": "Kaggle CUDA environment gate failed",
                                                     "formal_training_executed": False, "formal_ppl_executed": False})
        raise RuntimeError("Kaggle CUDA environment gate failed; no CPU fallback was attempted")
    env = json.loads((out / "LOCAL_ENVIRONMENT.json").read_text(encoding="utf-8"))
    if env["transformers_version"] != "4.49.0" or env["tokenizers_version"] != "0.21.4":
        write_json(out / "PREFLIGHT_VERDICT.json", {"verdict": "LOCAL_M3_FORMAL_NO_GO",
                                                     "failure": "transformers/tokenizers do not match v20 port runtime",
                                                     "formal_training_executed": False, "formal_ppl_executed": False})
        raise RuntimeError("environment differs from v20 Qwen2 architecture validation versions")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    try:
        model, model_identity = load_exact_model(config["model_id"], config["model_revision"],
                                                  args.model_path, device, out)
        dense_export_proxy = measure_dense_export_proxy(model, out, device)
        sample_ids = [sample["token_ids"] for sample in manifest["samples"][:8]]
        capture_l0_times = capture_formal_l0_inputs(model, sample_ids, device, out)
        selected_layers = list(config["preflight"]["representative_layers"])
        captured, full_capture_times = capture_representative_layers(model, sample_ids, selected_layers, device)

        with (out / "CALIBRATION_CAPTURE_TIMING.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=["status", "stage", "samples", "sequence_length", "total_s", "mean_s", "extrapolated_128_sample_s", "charged_to_formal_cost"])
            writer.writeheader()
            writer.writerow({"status": "MEASURED", "stage": "official_l0_input_capture_boundary", "samples": len(capture_l0_times),
                             "sequence_length": 2048, "total_s": sum(capture_l0_times),
                             "mean_s": statistics.mean(capture_l0_times),
                             "extrapolated_128_sample_s": statistics.mean(capture_l0_times) * 128,
                             "charged_to_formal_cost": True})
            writer.writerow({"status": "MEASURED", "stage": "native_full_model_representative_input_capture", "samples": len(full_capture_times),
                             "sequence_length": 2048, "total_s": sum(full_capture_times),
                             "mean_s": statistics.mean(full_capture_times),
                             "extrapolated_128_sample_s": "", "charged_to_formal_cost": False})

        sys.path.insert(0, str(bundle_root / "omniquant_port/upstream"))
        if importlib.util.find_spec("termcolor") is None:
            termcolor = types.ModuleType("termcolor")
            termcolor.colored = lambda text, *args, **kwargs: text
            sys.modules["termcolor"] = termcolor
        from models.int_qwen2_layer import QuantQwen2DecoderLayer
        from quantize.utils import set_quant_state
        sys.path.insert(0, str(bundle_root / "omniquant_port/upstream"))
        import quantize.omniquant as omni_module
        # The import form above can resolve the module function on some Python versions;
        # normalize to the module object without changing its source.
        if not hasattr(omni_module, "omniquant"):
            omni_module = importlib.import_module("quantize.omniquant")

        parity = parity_checks(model, captured, selected_layers, device,
                               bundle_root / "omniquant_port/upstream", out)
        if parity["status"] != "PASS":
            write_json(out / "correctness_checks.json", {"status": "FAIL", "reason": "Kaggle quant-off parity failed",
                                                           "quant_off_parity": parity})
            raise RuntimeError("Kaggle port parity failed; stopping before optimizer timing")

        timing_specs = {
            0: (int(config["preflight"]["l0_warmup_steps"]), int(config["preflight"]["l0_measured_steps"])),
            12: (int(config["preflight"]["l12_warmup_steps"]), int(config["preflight"]["l12_measured_steps"])),
            23: (int(config["preflight"]["l23_warmup_steps"]), int(config["preflight"]["l23_measured_steps"])),
        }
        layer_results: dict[int, dict[str, Any]] = {}
        for layer, (warmup, measured) in timing_specs.items():
            if not captured[layer]["inputs"]:
                raise RuntimeError(f"missing real frozen activation input for L{layer}")
            result = run_layer_sequence(
                omni_module=omni_module, config=config, full_model=model, layer_index=layer,
                activations=captured[layer]["inputs"], layer_kwargs=captured[layer]["kwargs"],
                token_ids=sample_ids, warmup_steps=warmup, measured_steps=measured,
                output_csv=out / f"L{layer}_2048_TIMING.csv",
                sequence_json=out / f"L{layer}_2048_RUN.json",
            )
            layer_results[layer] = result
            write_json(out / f"L{layer}_2048_MEMORY.json", {
                "status": "PASS" if result["memory"]["peak_reserved_bytes"] <= 8 * (1024 ** 3) else "FAIL",
                "layer": layer, "sequence_length": 2048, "selected_device": "cuda:0",
                "gpu_total_vram_bytes": result["memory"]["total_vram_bytes"],
                "free_vram_before_bytes": result["memory"]["free_vram_before_bytes"],
                "free_vram_after_bytes": result["memory"]["free_vram_after_bytes"],
                "baseline_allocated_bytes": result["memory"]["baseline_allocated_bytes"],
                "baseline_reserved_bytes": result["memory"]["baseline_reserved_bytes"],
                "peak_allocated_bytes": result["memory"]["peak_allocated_bytes"],
                "peak_reserved_bytes": result["memory"]["peak_reserved_bytes"],
                "peak_allocated_gib": result["memory"]["peak_allocated_bytes"] / (1024 ** 3),
                "peak_reserved_gib": result["memory"]["peak_reserved_bytes"] / (1024 ** 3),
                "cpu_peak_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) if resource is not None else None,
                "oom": False,
            })

        # Eight consecutive samples use one fresh L0 block, optimizer and LWC state.
        l0_sequence = run_layer_sequence(
            omni_module=omni_module, config=config, full_model=model, layer_index=0,
            activations=captured[0]["inputs"], layer_kwargs=captured[0]["kwargs"],
            token_ids=sample_ids, warmup_steps=0,
            measured_steps=int(config["preflight"]["l0_continuous_sequence_steps"]),
            output_csv=out / "L0_8STEP_SEQUENCE_INTERNAL.csv",
            sequence_json=out / "L0_8STEP_SEQUENCE.json",
        )
        allocated = [row["allocated_after_step_gb"] for row in l0_sequence["rows"]]
        monotonic = all(a <= b for a, b in zip(allocated, allocated[1:]))
        allocated_growth = max(allocated) - min(allocated) if allocated else 0.0
        growth_warning = (len(allocated) >= int(config["preflight"]["memory_growth_min_steps"])
                          and monotonic and allocated_growth > float(config["preflight"]["memory_allocated_growth_warning_gib"]))
        l0_sequence["memory_growth_diagnostic"] = {
            "allocated_after_step_gib": allocated,
            "reserved_after_step_gib": [row["reserved_after_step_gb"] for row in l0_sequence["rows"]],
            "allocated_monotonic_non_decreasing": monotonic,
            "allocated_growth_gib": allocated_growth,
            "warning_threshold_gib": config["preflight"]["memory_allocated_growth_warning_gib"],
            "unexplained_growth_warning": growth_warning,
            "reserved_is_allocator_cache_and_is_not_by_itself_a_leak": True,
        }
        write_json(out / "L0_8STEP_SEQUENCE.json", l0_sequence)

        l0_rows = layer_results[0]["rows"]
        lwc = {
            "status": "PASS" if (layer_results[0]["all_lwc_gradients_finite"]
                                   and layer_results[0]["all_steps_have_nonzero_lwc_gradient"]
                                   and layer_results[0]["any_lwc_parameter_changed"]
                                   and layer_results[0]["lwc_parameters_present"]
                                   and layer_results[0]["all_losses_finite"]) else "FAIL",
            "sequence_length": 2048,
            "optimizer_updates": len(l0_rows),
            "lwc_parameter_exists": layer_results[0]["lwc_parameters_present"],
            "gradient_finite_all_steps": layer_results[0]["all_lwc_gradients_finite"],
            "nonzero_lwc_gradient_all_steps": layer_results[0]["all_steps_have_nonzero_lwc_gradient"],
            "parameter_update_observed": layer_results[0]["any_lwc_parameter_changed"],
            "reconstruction_loss_finite_all_steps": layer_results[0]["all_losses_finite"],
            "step_checks": [{"step": i, "loss": row["loss"], "gradient_finite": row["lwc_gradient_finite"],
                             "nonzero_grad_tensor_count": row["lwc_nonzero_grad_tensor_count"],
                             "lwc_update_l1": row["lwc_update_l1"], "lwc_changed": row["lwc_changed"]}
                            for i, row in enumerate(l0_rows)],
            "algorithm": "frozen v20 official optimizer, LWC, reconstruction loss and QuantLinear STE path",
        }
        write_json(out / "LWC_CHECK.json", lwc)

        recalc = summarize_runtime(out, formal_steps, layer_results, capture_l0_times,
                                   model_identity, full_capture_times, dense_export_proxy,
                                   env, config, args.session_budget_hours)
        checks = {
            "bundle_verify": json.loads((out / "BUNDLE_VERIFY.json").read_text(encoding="utf-8"))["status"] == "PASS",
            "model_revision": model_identity["actual_snapshot_revision"] == config["model_revision"],
            "calibration_hash": json.loads((out / "CALIBRATION_CHECK.json").read_text(encoding="utf-8"))["status"] == "PASS",
            "quant_off_parity": parity["status"] == "PASS",
            "lwc_gradient_update_reconstruction": lwc["status"] == "PASS",
            "all_representative_runs": len(layer_results) == 3,
            "no_unexplained_8step_growth": not growth_warning,
            "all_memory_gates": all(r["memory"]["peak_reserved_bytes"] <= 8 * (1024 ** 3) for r in layer_results.values()),
        }
        write_json(out / "correctness_checks.json", {
            "status": "PASS" if all(checks.values()) else "FAIL",
            "checks": checks,
            "model_identity": {k: model_identity[k] for k in ("model_id", "actual_snapshot_revision", "config_commit_hash", "dtype")},
            "calibration_manifest_sha256": config["formal_training"]["calibration_manifest_sha256"],
            "formal_optimizer_updates": formal_steps["total_optimizer_updates"],
        })
        readiness = "LOCAL_M3_FORMAL_GO" if (recalc["verdict"] == "LOCAL_M3_FORMAL_GO"
                    and all(checks.values())) else "LOCAL_M3_FORMAL_NO_GO"
        write_json(out / "PREFLIGHT_VERDICT.json", {"verdict": readiness,
                                                    "runtime_gate": recalc["runtime_gate_pass_le_24h"],
                                                    "memory_gate": recalc["memory_gate_pass"],
                                                    "correctness_gate": all(checks.values()),
                                                    "formal_training_executed": False,
                                                    "formal_ppl_executed": False})
        write_report(out, env, model_identity, formal_steps,
                     json.loads((out / "CALIBRATION_CHECK.json").read_text(encoding="utf-8")),
                     parity, lwc, layer_results, l0_sequence, recalc, readiness)
        return 0 if readiness == "LOCAL_M3_FORMAL_GO" else 1
    except torch.cuda.OutOfMemoryError as exc:
        torch.cuda.empty_cache()
        write_json(out / "MEMORY_FEASIBILITY.json", {
            "status": "LOCAL_M3_MEMORY_NO_GO", "oom": True, "selected_device": "cuda:0",
            "error": f"{type(exc).__name__}: {exc}",
            "automatic_fallbacks": [], "sequence_length": 2048,
        })
        write_json(out / "PREFLIGHT_VERDICT.json", {"verdict": "LOCAL_M3_MEMORY_NO_GO",
                                                    "formal_training_executed": False,
                                                    "formal_ppl_executed": False})
        raise
    except Exception as exc:
        write_json(out / "PREFLIGHT_VERDICT.json", {
            "verdict": "LOCAL_M3_FORMAL_NO_GO",
            "failure": f"{type(exc).__name__}: {exc}",
            "formal_training_executed": False, "formal_ppl_executed": False,
        })
        raise
    finally:
        if "model" in locals():
            del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    raise SystemExit(main())
