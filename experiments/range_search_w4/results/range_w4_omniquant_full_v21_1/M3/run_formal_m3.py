from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import importlib
import json
import os
import platform
import random
import shutil
import sys
import time
import traceback
import types
import uuid
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PREFLIGHT = ROOT / "LOCAL_PREFLIGHT"
BUNDLE = ROOT.parent / "range_w4_omniquant_kaggle_v20_1" / "kaggle_bundle"
FROZEN_MODEL = Path(r"E:\quant-thesis\artifacts\models\qwen2.5-0.5b-instruct-7ae5576\7ae557604adf67be50417f59c2c2f167def9a775")
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
MODEL_REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"
DEVICE = torch.device("cuda:0")
EPOCHS = 20
SAMPLES = 128
SEQUENCE_LENGTH = 2048
EXPECTED_UPDATES = 61440
SEED = 20261005
CHECKPOINTS = HERE / "checkpoints"
EXPORT_DIR = HERE / "dense_export"
LOG_DIR = HERE / "logs"
STEP_LOG = LOG_DIR / "training_steps.csv"
EPOCH_LOG = LOG_DIR / "training_epochs.csv"
BLOCK_LOG = LOG_DIR / "block_health.csv"
STATUS_PATH = HERE / "RUN_STATUS.json"


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".tmp.{os.getpid()}.{uuid.uuid4().hex[:8]}")
    with temp.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, ensure_ascii=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def atomic_torch_save(value: Any, path: Path, save_fn) -> None:
    temp = path.with_name(path.name + f".tmp.{os.getpid()}.{uuid.uuid4().hex[:8]}")
    save_fn(value, temp)
    with temp.open("rb+") as stream:
        os.fsync(stream.fileno())
    os.replace(temp, path)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def frozen_context() -> tuple[dict[str, Any], dict[str, Any], list[list[int]], dict[str, Any]]:
    config_path = BUNDLE / "configs" / "m3_preflight.json"
    config = load_json(config_path)
    if config["model_id"] != MODEL_ID or config["model_revision"] != MODEL_REVISION:
        raise RuntimeError("frozen config model identity mismatch")
    if config["formal_training"]["epochs"] != EPOCHS or config["formal_training"]["samples"] != SAMPLES:
        raise RuntimeError("frozen formal training workload mismatch")
    if config["formal_training"]["tokens_per_sample"] != SEQUENCE_LENGTH:
        raise RuntimeError("frozen sequence length mismatch")
    if config["quantization"] != {
        "weight_bits": 4, "activation_bits": 16, "group_size": 128,
        "weight_group_axis": "per output row, consecutive input features",
        "symmetric": False, "integer_range": [0, 15],
        "zero_point": "official OmniQuant UniformAffineQuantizer; asymmetric affine",
        "lwc": True, "let": False,
    }:
        raise RuntimeError("frozen W4A16/G128 OmniQuant quantization configuration mismatch")

    preflight_correctness = load_json(PREFLIGHT / "correctness_checks.json")
    preflight_verdict = load_json(PREFLIGHT / "PREFLIGHT_VERDICT.json")
    cost = load_json(PREFLIGHT / "LOCAL_COST_RECALC.json")
    parity = load_json(PREFLIGHT / "QUANT_OFF_PARITY.json")
    identity = load_json(PREFLIGHT / "MODEL_IDENTITY.json")
    tokenizer_check = load_json(ROOT / "TOKENIZER_CHECK.json")
    if preflight_correctness.get("status") != "PASS" or preflight_verdict.get("verdict") != "LOCAL_M3_FORMAL_GO":
        raise RuntimeError("local formal gate does not have an auditable PASS verdict")
    if not all(cost.get(key) is True for key in ("memory_gate_pass", "correctness_gate_pass", "runtime_gate_pass_le_24h")):
        raise RuntimeError("local memory/correctness/24h cost gate failed")
    if float(cost["conservative_total_hours"]) > 24.0:
        raise RuntimeError("conservative local formal cost exceeds the frozen 24h gate")
    if parity.get("status") != "PASS" or identity.get("status") != "PASS" or tokenizer_check.get("status") != "PASS":
        raise RuntimeError("model/tokenizer/quant-off parity gate is not PASS")
    if identity.get("requested_revision") != MODEL_REVISION or identity.get("actual_snapshot_revision") != MODEL_REVISION:
        raise RuntimeError("loaded local model identity is not the frozen exact revision")
    model_manifest_path = ROOT / "LOCAL_MODEL_MANIFEST.json"
    model_manifest = load_json(model_manifest_path)
    if model_manifest.get("status") != "PASS":
        raise RuntimeError("frozen model snapshot manifest is not PASS")
    if FROZEN_MODEL.name != MODEL_REVISION:
        raise RuntimeError("isolated model directory is not named by the exact revision")
    for record in model_manifest["files"]:
        p = FROZEN_MODEL / record["relative_path"]
        if not p.is_file() or p.stat().st_size != record["bytes"] or sha_file(p) != record["sha256"]:
            raise RuntimeError(f"frozen model file changed after identity gate: {record['relative_path']}")

    calibration_path = BUNDLE / "calibration" / "CALIBRATION_MANIFEST.json"
    calibration = load_json(calibration_path)
    calibration_sha = sha_file(calibration_path)
    if calibration_sha != config["formal_training"]["calibration_manifest_sha256"]:
        raise RuntimeError("frozen calibration manifest hash mismatch")
    if calibration.get("nsamples") != SAMPLES or calibration.get("seqlen") != SEQUENCE_LENGTH or calibration.get("seed") != SEED or calibration.get("split") != "train":
        raise RuntimeError("frozen calibration metadata mismatch")
    token_ids = []
    token_hashes = []
    for index, sample in enumerate(calibration["samples"]):
        ids = sample["token_ids"]
        token_sha = sha_bytes(json.dumps(ids).encode("utf-8"))
        if len(ids) != SEQUENCE_LENGTH or sample.get("token_sha256") != token_sha:
            raise RuntimeError(f"frozen token sample {index} failed length/hash validation")
        token_ids.append(ids)
        token_hashes.append(token_sha)
    if len(token_ids) != SAMPLES:
        raise RuntimeError("frozen calibration sample count mismatch")

    port_dir = HERE / "official_port"
    source_manifest = load_json(HERE / "OMNIQUANT_SOURCE_AUDIT.json")
    for relative, info in source_manifest["files"].items():
        actual = sha_file(port_dir / relative)
        expected = info["patched_sha256"] if info.get("patched_sha256") else info["official_sha256"]
        if actual != expected:
            raise RuntimeError(f"isolated OmniQuant source hash mismatch: {relative}")

    device_props = torch.cuda.get_device_properties(DEVICE)
    runtime_env = {
        "python": platform.python_version(), "torch": torch.__version__,
        "transformers": __import__("transformers").__version__,
        "tokenizers": __import__("tokenizers").__version__,
        "cuda_runtime": torch.version.cuda, "gpu_name": device_props.name,
        "gpu_capability": list(torch.cuda.get_device_capability(DEVICE)),
        "device_count": torch.cuda.device_count(), "device": str(DEVICE),
        "tf32_matmul": bool(torch.backends.cuda.matmul.allow_tf32),
        "tf32_cudnn": bool(torch.backends.cudnn.allow_tf32),
    }
    expected_env = load_json(PREFLIGHT / "LOCAL_ENVIRONMENT.json")
    expected_gpu = (expected_env.get("gpus") or [{}])[0].get("name")
    if runtime_env["gpu_name"] != expected_gpu or runtime_env["device_count"] != 1:
        raise RuntimeError("current CUDA device differs from the passing local preflight hardware")
    if runtime_env["tf32_matmul"] or runtime_env["tf32_cudnn"]:
        raise RuntimeError("TF32 must remain disabled under the frozen protocol")

    lock = {
        "protocol_id": "range_w4_omniquant_full_v21_1_M3",
        "parent_protocol_sha256": sha_file(ROOT / "PROTOCOL.json"),
        "model_id": MODEL_ID, "model_revision": MODEL_REVISION,
        "model_manifest_sha256": sha_file(model_manifest_path),
        "model_weight_sha256": sha_file(FROZEN_MODEL / "model.safetensors"),
        "config_path": str(config_path.relative_to(ROOT.parent.parent.parent)),
        "config_sha256": sha_file(config_path),
        "calibration_manifest_sha256": calibration_sha,
        "ordered_token_hashes_sha256": sha_bytes(json.dumps(token_hashes, separators=(",", ":")).encode("utf-8")),
        "ordered_token_hashes": token_hashes,
        "preflight_correctness_sha256": sha_file(PREFLIGHT / "correctness_checks.json"),
        "preflight_cost_sha256": sha_file(PREFLIGHT / "LOCAL_COST_RECALC.json"),
        "preflight_parity_sha256": sha_file(PREFLIGHT / "QUANT_OFF_PARITY.json"),
        "formal_runner_sha256": sha_file(Path(__file__).resolve()),
        "omniquant_source_audit_sha256": sha_file(HERE / "OMNIQUANT_SOURCE_AUDIT.json"),
        "official_commit": "feffe8ea87d80f7bb57b6e25e7cff9dc950fcc14",
        "quantization": config["quantization"],
        "training": config["formal_training"],
        "decoder_blocks": 24,
        "updates_per_block": EPOCHS * SAMPLES,
        "total_optimizer_updates": EXPECTED_UPDATES,
        "runtime_environment": runtime_env,
        "resume_semantics": "block-boundary only; replay completed official quantized prefix; incomplete block restarts from pristine weights",
        "algorithm_changes": [],
    }
    lock_path = HERE / "FORMAL_PROTOCOL_LOCK.json"
    if lock_path.exists():
        previous_lock = load_json(lock_path)
        if previous_lock != lock:
            previous_status = load_json(STATUS_PATH) if STATUS_PATH.exists() else {}
            recoverable_checkpoint_io_failure = (
                previous_status.get("status") == "FAILED_OR_INTERRUPTED"
                and previous_status.get("exception_type") == "OSError"
                and "Bad file descriptor" in previous_status.get("message", "")
                and not (CHECKPOINTS / "LATEST.json").exists()
                and not (HERE / "M3_TRAINING_SUMMARY.json").exists()
            )
            if not recoverable_checkpoint_io_failure:
                raise RuntimeError("existing immutable formal protocol lock differs after formal execution began")
            prior_lock_path = HERE / "FORMAL_PROTOCOL_LOCK_PRE_RECOVERY.json"
            if prior_lock_path.exists():
                raise RuntimeError("pre-recovery protocol lock already exists; refusing another implicit protocol change")
            shutil.copy2(lock_path, prior_lock_path)
            prior_step_rows = 0
            if STEP_LOG.exists():
                with STEP_LOG.open(newline="", encoding="utf-8") as stream:
                    prior_step_rows = sum(1 for _ in csv.DictReader(stream))
            write_json(HERE / "RUNNER_RECOVERY_NOTE.json", {
                "status": "CHECKPOINT_IO_FIX_BEFORE_RESTART",
                "failure": previous_status.get("message"),
                "failure_type": previous_status.get("exception_type"),
                "pre_recovery_protocol_lock_sha256": sha_file(lock_path),
                "pre_recovery_runner_sha256": previous_lock.get("formal_runner_sha256"),
                "corrected_runner_sha256": lock.get("formal_runner_sha256"),
                "optimizer_updates_attempted_in_uncommitted_run": prior_step_rows,
                "committed_completed_blocks_before_recovery": 0,
                "checkpoint_latest_existed": False,
                "quantization_algorithm_changed": False,
                "recovery_action": "preserve failure artifacts and rerun the incomplete first block from pristine frozen weights",
            })
            write_json(lock_path, lock)
    else:
        write_json(lock_path, lock)
    return config, calibration, token_ids, lock


def current_rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all(),
    }


def restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    torch.cuda.set_rng_state_all(state["torch_cuda"])


def to_cpu_state(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().contiguous()
    if isinstance(value, dict):
        return type(value)((key, to_cpu_state(item)) for key, item in value.items())
    if isinstance(value, list):
        return [to_cpu_state(item) for item in value]
    if isinstance(value, tuple):
        return tuple(to_cpu_state(item) for item in value)
    return value


def checkpoint_context(lock: dict[str, Any]) -> dict[str, Any]:
    return {
        "formal_protocol_lock_sha256": sha_file(HERE / "FORMAL_PROTOCOL_LOCK.json"),
        "model_manifest_sha256": lock["model_manifest_sha256"],
        "model_weight_sha256": lock["model_weight_sha256"],
        "config_sha256": lock["config_sha256"],
        "calibration_manifest_sha256": lock["calibration_manifest_sha256"],
        "ordered_token_hashes_sha256": lock["ordered_token_hashes_sha256"],
        "omniquant_source_audit_sha256": lock["omniquant_source_audit_sha256"],
        "official_commit": lock["official_commit"],
        "runtime_environment": lock["runtime_environment"],
        "total_optimizer_updates": EXPECTED_UPDATES,
    }


def save_block_checkpoint(state: dict[Any, Any], lock: dict[str, Any], save_fn) -> dict[str, Any]:
    keys = sorted(int(key) for key in state)
    if keys != list(range(len(keys))) or not keys:
        raise RuntimeError(f"checkpoint state does not contain a contiguous completed prefix: {keys[:4]}..{keys[-4:] if keys else []}")
    block = keys[-1]
    completed_layers = block + 1
    if completed_layers > 24:
        raise RuntimeError("checkpoint has more than 24 completed decoder blocks")
    CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    final_dir = CHECKPOINTS / f"block_{block:02d}"
    if final_dir.exists():
        raise RuntimeError(f"refusing to overwrite existing checkpoint {final_dir}")
    temp_dir = CHECKPOINTS / f".block_{block:02d}.tmp.{os.getpid()}.{uuid.uuid4().hex[:8]}"
    temp_dir.mkdir(parents=False, exist_ok=False)
    state_path = temp_dir / "omni_parameters.pth"
    rng_path = temp_dir / "rng_state.pth"
    atomic_torch_save(to_cpu_state(state), state_path, save_fn)
    atomic_torch_save(current_rng_state(), rng_path, save_fn)
    manifest = {
        "status": "COMPLETE_BLOCK_CHECKPOINT",
        **checkpoint_context(lock),
        "completed_block_id": block,
        "completed_layer_count": completed_layers,
        "next_block_id": completed_layers if completed_layers < 24 else None,
        "effective_optimizer_updates_in_checkpoint": completed_layers * EPOCHS * SAMPLES,
        "state_file": "omni_parameters.pth",
        "state_file_bytes": state_path.stat().st_size,
        "state_file_sha256": sha_file(state_path),
        "rng_file": "rng_state.pth",
        "rng_file_bytes": rng_path.stat().st_size,
        "rng_file_sha256": sha_file(rng_path),
        "committed_at_local": time.strftime("%Y-%m-%d %H:%M:%S%z"),
    }
    write_json(temp_dir / "CHECKPOINT_MANIFEST.json", manifest)
    os.replace(temp_dir, final_dir)
    pointer = {"checkpoint_dir": final_dir.name, "manifest_sha256": sha_file(final_dir / "CHECKPOINT_MANIFEST.json"),
               "completed_block_id": block, "completed_layer_count": completed_layers}
    write_json(CHECKPOINTS / "LATEST.json", pointer)
    return {**pointer, "state_path": str(final_dir / "omni_parameters.pth"), "manifest": manifest}


def verify_latest_checkpoint(lock: dict[str, Any]) -> tuple[Path | None, int, dict[str, Any] | None]:
    latest = CHECKPOINTS / "LATEST.json"
    if not latest.is_file():
        return None, 0, None
    pointer = load_json(latest)
    checkpoint_dir = CHECKPOINTS / pointer["checkpoint_dir"]
    manifest_path = checkpoint_dir / "CHECKPOINT_MANIFEST.json"
    if not manifest_path.is_file() or sha_file(manifest_path) != pointer["manifest_sha256"]:
        raise RuntimeError("latest checkpoint manifest missing or hash mismatch")
    manifest = load_json(manifest_path)
    expected = checkpoint_context(lock)
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise RuntimeError(f"checkpoint frozen-context mismatch: {key}")
    completed = int(manifest["completed_layer_count"])
    block = int(manifest["completed_block_id"])
    if completed != block + 1 or completed != int(pointer["completed_layer_count"]) or block != int(pointer["completed_block_id"]):
        raise RuntimeError("checkpoint block boundary metadata is inconsistent")
    for key in ("state", "rng"):
        path = checkpoint_dir / manifest[f"{key}_file"]
        if not path.is_file() or path.stat().st_size != manifest[f"{key}_file_bytes"] or sha_file(path) != manifest[f"{key}_file_sha256"]:
            raise RuntimeError(f"checkpoint {key} file failed integrity validation")
    if manifest["effective_optimizer_updates_in_checkpoint"] != completed * EPOCHS * SAMPLES:
        raise RuntimeError("checkpoint optimizer update count is inconsistent")
    state = torch.load(checkpoint_dir / manifest["state_file"], map_location="cpu", weights_only=False)
    state_keys = sorted(int(key) for key in state)
    if state_keys != list(range(completed)):
        raise RuntimeError("checkpoint state is not a contiguous prefix from block zero")
    rng_state = torch.load(checkpoint_dir / manifest["rng_file"], map_location="cpu", weights_only=False)
    return checkpoint_dir / manifest["state_file"], completed, {"rng_state": rng_state, "state": state}


class CsvAppend:
    def __init__(self, path: Path, fields: list[str]):
        self.path = path
        self.fields = fields
        path.parent.mkdir(parents=True, exist_ok=True)
        new = not path.exists() or path.stat().st_size == 0
        self.stream = path.open("a", newline="", encoding="utf-8")
        self.writer = csv.DictWriter(self.stream, fieldnames=fields, extrasaction="ignore")
        if new:
            self.writer.writeheader()
            self.stream.flush()

    def write(self, row: dict[str, Any], flush: bool = False) -> None:
        self.writer.writerow(row)
        if flush:
            self.stream.flush()
            os.fsync(self.stream.fileno())

    def close(self) -> None:
        self.stream.flush()
        self.stream.close()


def make_args(config: dict[str, Any], resume_path: Path | None, completed_layers: int, output_dir: Path):
    q = config["quantization"]
    t = config["formal_training"]
    return SimpleNamespace(
        net="qwen2", model=MODEL_ID, nsamples=SAMPLES, seqlen=SEQUENCE_LENGTH,
        batch_size=1, deactive_amp=False, epochs=EPOCHS, smoke_layer_limit=24,
        smoke_step_limit=None, aug_loss=False, let=False, lwc=True,
        wbits=4, abits=16, group_size=128, symmetric=False,
        disable_zero_point=False, w_dynamic_method="per_channel", a_dynamic_method="per_token",
        alpha=0.5, let_lr=float(t["let_learning_rate_recorded_but_inactive"]),
        lwc_lr=float(t["lwc_learning_rate"]), wd=float(t["weight_decay"]),
        resume=str(resume_path) if resume_path else None,
        output_dir=str(output_dir), real_quant=False,
        weight_quant_params={
            "n_bits": int(q["weight_bits"]), "per_channel_axes": [0],
            "symmetric": bool(q["symmetric"]), "dynamic_method": "per_channel",
            "group_size": int(q["group_size"]), "lwc": bool(q["lwc"]),
            "disable_zero_point": False,
        },
        act_quant_params={"n_bits": 16, "per_channel_axes": [], "symmetric": False,
                          "dynamic_method": "per_token"},
        v21_resume_completed_layers=completed_layers,
    )


def install_source_callbacks(args, lock, resume_state, attempt_id):
    from quantize.utils import lwc_parameters

    epoch_csv = CsvAppend(EPOCH_LOG, ["attempt_id", "layer", "epoch_index", "loss_mean", "norm_mean", "lwc_param_count", "lwc_finite"])
    step_csv = CsvAppend(STEP_LOG, ["attempt_id", "layer", "epoch_index", "sample_index", "loss", "lwc_grad_param_count", "lwc_grad_finite", "lwc_nonzero"])
    block_csv = CsvAppend(BLOCK_LOG, ["attempt_id", "layer", "event", "lwc_parameter_count", "lwc_changed_count", "lwc_max_abs_change", "float_tensor_count", "all_float_finite", "quantlinear_count", "scales_and_zeros_finite"])
    baselines: dict[int, dict[str, torch.Tensor]] = {}
    counts = {"steps": 0, "finite_losses": 0, "finite_gradients": 0, "nonzero_gradients": 0,
              "optimized_blocks": 0, "quantized_blocks": 0, "prefix_replayed_blocks": 0}

    def epoch_cb(layer: int, epoch: int, loss: torch.Tensor, norm: torch.Tensor, module):
        params = [(name, p) for name, p in module.named_parameters() if "bound_factor" in name]
        finite = all(bool(torch.isfinite(p.detach()).all().item()) for _, p in params)
        row = {"attempt_id": attempt_id, "layer": layer, "epoch_index": epoch,
               "loss_mean": float(loss.detach().float().item()), "norm_mean": float(norm.detach().float().item()),
               "lwc_param_count": len(params), "lwc_finite": finite}
        epoch_csv.write(row, flush=True)
        if not finite or not np.isfinite(row["loss_mean"]) or not np.isfinite(row["norm_mean"]):
            raise FloatingPointError(f"nonfinite epoch diagnostics at layer={layer}, epoch={epoch}")

    def step_check(module, layer: int, epoch: int, sample: int, loss: torch.Tensor):
        params = list(lwc_parameters(module))
        grads = [param.grad.detach().reshape(-1) for param in params if param.grad is not None]
        if len(grads) != len(params) or not grads:
            raise FloatingPointError(f"missing LWC gradients at layer={layer}, epoch={epoch}, sample={sample}")
        flat = torch.cat([grad.float() for grad in grads])
        flags = torch.stack((torch.isfinite(flat).all(), torch.any(flat != 0))).detach().cpu().tolist()
        finite_grad, nonzero_grad = bool(flags[0]), bool(flags[1])
        loss_value = float(loss.detach().float().item())
        step_csv.write({"attempt_id": attempt_id, "layer": layer, "epoch_index": epoch,
                        "sample_index": sample, "loss": loss_value,
                        "lwc_grad_param_count": len(params), "lwc_grad_finite": finite_grad,
                        "lwc_nonzero": nonzero_grad}, flush=(sample == SAMPLES - 1))
        counts["steps"] += 1
        counts["finite_losses"] += int(np.isfinite(loss_value))
        counts["finite_gradients"] += int(finite_grad)
        counts["nonzero_gradients"] += int(nonzero_grad)
        if not np.isfinite(loss_value) or not finite_grad:
            raise FloatingPointError(f"LWC loss/gradient check failed at layer={layer}, epoch={epoch}, sample={sample}")

    def layer_cb(event: str, layer: int, module):
        if event == "optimizer_init":
            baselines[layer] = {name: param.detach().cpu().clone()
                                for name, param in module.named_parameters() if "bound_factor" in name}
            return
        if event == "optimizer_complete":
            final = {name: param.detach().cpu() for name, param in module.named_parameters() if "bound_factor" in name}
            changed = [name for name, value in final.items()
                       if name in baselines.get(layer, {}) and not torch.equal(value, baselines[layer][name])]
            finite = all(bool(torch.isfinite(value).all()) for value in final.values())
            if not finite or not changed:
                raise FloatingPointError(f"LWC update/finite check failed for layer={layer}")
            max_delta = max(float((final[name].float() - baselines[layer][name].float()).abs().max()) for name in changed)
            block_csv.write({"attempt_id": attempt_id, "layer": layer, "event": event,
                             "lwc_parameter_count": len(final), "lwc_changed_count": len(changed),
                             "lwc_max_abs_change": max_delta, "float_tensor_count": len(final),
                             "all_float_finite": finite, "quantlinear_count": "", "scales_and_zeros_finite": ""}, flush=True)
            counts["optimized_blocks"] += 1
            return
        if event not in ("quantized_block", "replayed_prefix"):
            return
        float_tensors = [(name, tensor) for name, tensor in list(module.named_parameters()) + list(module.named_buffers())
                         if tensor is not None and tensor.is_floating_point()]
        flags = [torch.isfinite(tensor).all() for _, tensor in float_tensors]
        all_finite = bool(torch.stack(flags).all().item()) if flags else True
        from quantize.int_linear import QuantLinear
        qlinears = [item for item in module.modules() if isinstance(item, QuantLinear)]
        q_flags = []
        for qlinear in qlinears:
            q_flags.append(qlinear.weight is not None and torch.isfinite(qlinear.weight).all())
            q_flags.append(qlinear.weight_quantizer.scales is not None and torch.isfinite(qlinear.weight_quantizer.scales).all())
            zeros = qlinear.weight_quantizer.zeros
            q_flags.append(zeros is not None and torch.isfinite(zeros).all())
        q_ok = bool(torch.stack([flag if isinstance(flag, torch.Tensor) else torch.tensor(bool(flag)) for flag in q_flags]).all().item()) if q_flags else False
        if not all_finite or not q_ok or len(qlinears) != 7:
            raise FloatingPointError(f"quantized layer finite/scale/zero check failed at layer={layer}")
        block_csv.write({"attempt_id": attempt_id, "layer": layer, "event": event,
                         "lwc_parameter_count": sum("bound_factor" in name for name, _ in module.named_parameters()),
                         "lwc_changed_count": "", "lwc_max_abs_change": "",
                         "float_tensor_count": len(float_tensors), "all_float_finite": all_finite,
                         "quantlinear_count": len(qlinears), "scales_and_zeros_finite": q_ok}, flush=True)
        if event == "quantized_block":
            counts["quantized_blocks"] += 1
        else:
            counts["prefix_replayed_blocks"] += 1

    args.v21_epoch_callback = epoch_cb
    args.v21_step_check_callback = step_check
    args.v21_layer_callback = layer_cb
    if resume_state is not None:
        rng_state = resume_state["rng_state"]
        args.v21_restore_rng = lambda: restore_rng_state(rng_state)
    return counts, (step_csv, epoch_csv, block_csv)


def write_runtime_environment() -> None:
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["OMP_NUM_THREADS"] = os.environ.get("OMP_NUM_THREADS", "4")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.cuda.set_device(DEVICE)


def run_calibration(config: dict[str, Any], token_ids: list[list[int]], lock: dict[str, Any]) -> dict[str, Any]:
    from transformers import AutoTokenizer

    original_torch_save = torch.save
    resume_path, completed_layers, resume_state = verify_latest_checkpoint(lock)
    attempt_id = f"{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    write_runtime_environment()
    sys.path.insert(0, str(HERE / "official_port"))
    if importlib.util.find_spec("termcolor") is None:
        termcolor = types.ModuleType("termcolor")
        termcolor.colored = lambda text, *args, **kwargs: text
        sys.modules["termcolor"] = termcolor
    args = make_args(config, resume_path, completed_layers, HERE / "upstream_output")
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    counts, csv_logs = install_source_callbacks(args, lock, resume_state, attempt_id)

    def checkpointing_torch_save(value, f, *save_args, **save_kwargs):
        try:
            target = Path(os.fspath(f))
        except TypeError:
            target = None
        if target is not None and target.name == "omni_parameters.pth":
            record = save_block_checkpoint(value, lock, original_torch_save)
            print(f"CHECKPOINT_COMMITTED block={record['completed_block_id']} updates={record['manifest']['effective_optimizer_updates_in_checkpoint']}", flush=True)
            return None
        return original_torch_save(value, f, *save_args, **save_kwargs)

    RUN_STATUS = {"status": "RUNNING", "attempt_id": attempt_id,
                  "formal_protocol_lock_sha256": sha_file(HERE / "FORMAL_PROTOCOL_LOCK.json"),
                  "resume_completed_layers": completed_layers,
                  "resumed_from": str(resume_path) if resume_path else None,
                  "optimizer_updates_expected_effective": EXPECTED_UPDATES,
                  "updated_at": time.strftime("%Y-%m-%d %H:%M:%S%z")}
    write_json(STATUS_PATH, RUN_STATUS)
    omni_module = importlib.import_module("quantize.omniquant")
    from run_local_preflight import load_exact_model
    model, identity = load_exact_model(MODEL_ID, MODEL_REVISION, str(FROZEN_MODEL), DEVICE, HERE)
    tokenizer = AutoTokenizer.from_pretrained(str(FROZEN_MODEL), local_files_only=True, use_fast=True)
    if tokenizer.__class__.__name__ != "Qwen2TokenizerFast":
        raise RuntimeError(f"unexpected frozen tokenizer class {tokenizer.__class__.__name__}")
    dataloader = [(torch.tensor(ids, dtype=torch.long).view(1, -1), index)
                  for index, ids in enumerate(token_ids)]
    lm = SimpleNamespace(model=model, device=DEVICE, _device=DEVICE, seqlen=SEQUENCE_LENGTH)
    calibration_wall_started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats(DEVICE)
    torch.cuda.synchronize(DEVICE)
    torch.save = checkpointing_torch_save
    try:
        result_model = omni_module.omniquant(lm, args, dataloader, None, None, NullLogger())
        torch.cuda.synchronize(DEVICE)
    finally:
        torch.save = original_torch_save
        calibration_wall = time.perf_counter() - calibration_wall_started
        for handle in csv_logs:
            handle.close()
    peak_allocated = int(torch.cuda.max_memory_allocated(DEVICE))
    peak_reserved = int(torch.cuda.max_memory_reserved(DEVICE))
    latest_path, completed_after, latest_record = verify_latest_checkpoint(lock)
    if completed_after != 24 or latest_record is None:
        raise RuntimeError(f"formal M3 did not commit all 24 decoder blocks; completed={completed_after}")
    if latest_record["state"] and sorted(int(k) for k in latest_record["state"]) != list(range(24)):
        raise RuntimeError("formal final checkpoint state is missing decoder blocks")
    if counts["steps"] != (24 - completed_layers) * EPOCHS * SAMPLES:
        raise RuntimeError(f"this attempt update count mismatch: observed={counts['steps']} expected={(24-completed_layers)*EPOCHS*SAMPLES}")
    if counts["finite_losses"] != counts["steps"] or counts["finite_gradients"] != counts["steps"] or counts["nonzero_gradients"] == 0:
        raise RuntimeError("formal step-level finite LWC gradient or aggregate nonzero-gradient check failed")
    if counts["optimized_blocks"] != 24 - completed_layers or counts["quantized_blocks"] != 24 - completed_layers:
        raise RuntimeError("formal optimizer or quantized block count mismatch")
    if latest_record["manifest"]["effective_optimizer_updates_in_checkpoint"] != EXPECTED_UPDATES:
        raise RuntimeError("final effective optimizer update count differs from the frozen 61,440")

    summary = {
        "status": "PASS", "attempt_id": attempt_id, "formal_protocol_lock_sha256": sha_file(HERE / "FORMAL_PROTOCOL_LOCK.json"),
        "model_revision": MODEL_REVISION, "model_weight_sha256": lock["model_weight_sha256"],
        "calibration_manifest_sha256": lock["calibration_manifest_sha256"],
        "ordered_token_hashes_sha256": lock["ordered_token_hashes_sha256"],
        "completed_blocks": completed_after, "effective_optimizer_updates": EXPECTED_UPDATES,
        "optimizer_updates_in_this_attempt": counts["steps"],
        "resume_completed_layers": completed_layers,
        "prefix_blocks_replayed": counts["prefix_replayed_blocks"],
        "training_step_check_counts": counts,
        "calibration_wall_seconds": calibration_wall,
        "calibration_wall_hours": calibration_wall / 3600.0,
        "peak_allocated_bytes": peak_allocated, "peak_reserved_bytes": peak_reserved,
        "peak_allocated_gib": peak_allocated / (1024 ** 3), "peak_reserved_gib": peak_reserved / (1024 ** 3),
        "checkpoint_latest": str(latest_path),
        "checkpoint_manifest_sha256": sha_file(CHECKPOINTS / "block_23" / "CHECKPOINT_MANIFEST.json"),
        "model_load_seconds": identity["from_pretrained_seconds"] + identity["weights_to_cuda0_seconds"],
        "result_model": result_model,
        "model": model,
        "tokenizer": tokenizer,
    }
    return summary


class NullLogger:
    def info(self, message: str) -> None:
        print(message, flush=True)


def replace_module(root: torch.nn.Module, dotted: str, replacement: torch.nn.Module) -> None:
    parts = dotted.split(".")
    parent = root
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], replacement)


def export_dense_and_check(model, tokenizer, token_ids: list[list[int]]) -> dict[str, Any]:
    from transformers import AutoModelForCausalLM
    from quantize.int_linear import QuantLinear

    if EXPORT_DIR.exists():
        raise RuntimeError(f"dense export directory already exists; refusing to overwrite: {EXPORT_DIR}")
    EXPORT_DIR.mkdir(parents=True, exist_ok=False)
    model.eval()
    model.to(DEVICE)
    probe_ids = token_ids[0][:64]
    probe_hash = sha_bytes(json.dumps(probe_ids).encode("utf-8"))
    with torch.inference_mode():
        reference = model(torch.tensor(probe_ids, dtype=torch.long, device=DEVICE).view(1, -1), use_cache=False).logits.float().cpu()
    torch.cuda.synchronize(DEVICE)

    quantized_linears = 0
    for index, wrapper in enumerate(list(model.model.layers)):
        if not hasattr(wrapper, "native_layer"):
            raise RuntimeError(f"expected OmniQuant wrapper at decoder block {index}")
        native = wrapper.native_layer
        for name, module in list(native.named_modules()):
            if not isinstance(module, QuantLinear):
                continue
            quantized_linears += 1
            if module.weight.device.type != "cpu":
                module.cpu()
            replacement = torch.nn.Linear(module.in_features, module.out_features,
                                          bias=module.bias is not None,
                                          device="cpu", dtype=module.weight.dtype)
            with torch.no_grad():
                replacement.weight.copy_(module.weight)
                if module.bias is not None:
                    replacement.bias.copy_(module.bias)
            replace_module(native, name, replacement)
        model.model.layers[index] = native
    if quantized_linears != 168:
        raise RuntimeError(f"expected 168 quantized projections before dense export, found {quantized_linears}")
    model.cpu().eval()
    torch.cuda.empty_cache()
    model.save_pretrained(EXPORT_DIR, safe_serialization=True, max_shard_size="2GB")
    tokenizer.save_pretrained(EXPORT_DIR)
    export_files = [{"path": p.relative_to(EXPORT_DIR).as_posix(), "bytes": p.stat().st_size, "sha256": sha_file(p)}
                    for p in sorted(EXPORT_DIR.rglob("*")) if p.is_file()]
    write_json(HERE / "DENSE_EXPORT_MANIFEST.json", {"status": "PASS", "representation": "dense-dequant FP16",
                                                       "model_revision": MODEL_REVISION, "probe_token_ids_sha256": probe_hash,
                                                       "linear_modules_replaced": quantized_linears, "files": export_files})
    del model
    gc.collect()
    torch.cuda.empty_cache()

    reloaded = AutoModelForCausalLM.from_pretrained(str(EXPORT_DIR), torch_dtype=torch.float16,
                                                    low_cpu_mem_usage=True, attn_implementation="sdpa",
                                                    local_files_only=True).eval().to(DEVICE)
    with torch.inference_mode():
        actual = reloaded(torch.tensor(probe_ids, dtype=torch.long, device=DEVICE).view(1, -1), use_cache=False).logits.float().cpu()
    torch.cuda.synchronize(DEVICE)
    delta = (actual - reference).abs()
    result = {
        "status": "PASS" if torch.equal(actual, reference) or torch.allclose(actual, reference, rtol=1e-3, atol=1e-3) else "FAIL",
        "probe_token_ids_sha256": probe_hash,
        "reference_logits_shape": list(reference.shape), "reloaded_logits_shape": list(actual.shape),
        "bitwise_equal": bool(torch.equal(actual, reference)),
        "allclose_rtol_1e-3_atol_1e-3": bool(torch.allclose(actual, reference, rtol=1e-3, atol=1e-3)),
        "max_abs_diff": float(delta.max().item()), "mean_abs_diff": float(delta.mean().item()),
        "dense_export_path": str(EXPORT_DIR), "linear_modules_replaced": quantized_linears,
        "export_file_count": len(export_files),
    }
    write_json(HERE / "DENSE_EXPORT_RELOAD_CHECK.json", result)
    if result["status"] != "PASS":
        raise RuntimeError("dense export/reload logits parity failed")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true", help="resume only from a verified completed decoder-block checkpoint")
    args_cli = parser.parse_args()
    HERE.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("frozen formal protocol requires exactly one CUDA device on cuda:0")
    write_runtime_environment()
    config, calibration, token_ids, lock = frozen_context()
    if not args_cli.resume and (CHECKPOINTS / "LATEST.json").exists():
        raise RuntimeError("checkpoint exists; use --resume to continue from the verified latest completed block")
    if args_cli.resume and not (CHECKPOINTS / "LATEST.json").exists():
        raise RuntimeError("--resume requested but no completed-block checkpoint exists")
    write_json(HERE / "RUN_STATUS.json", {"status": "FORMAL_M3_STARTING", "protocol_lock_sha256": sha_file(HERE / "FORMAL_PROTOCOL_LOCK.json"),
                                          "started_at_local": time.strftime("%Y-%m-%d %H:%M:%S%z"),
                                          "formal_optimizer_updates_expected": EXPECTED_UPDATES})
    try:
        summary = run_calibration(config, token_ids, lock)
        model = summary.pop("model")
        tokenizer = summary.pop("tokenizer")
        summary.pop("result_model")
        summary["status"] = "PASS"
        write_json(HERE / "M3_TRAINING_SUMMARY.json", summary)
        write_json(STATUS_PATH, {"status": "CALIBRATION_PASS", "attempt_id": summary["attempt_id"],
                                 "effective_optimizer_updates": summary["effective_optimizer_updates"],
                                 "calibration_wall_seconds": summary["calibration_wall_seconds"],
                                 "updated_at": time.strftime("%Y-%m-%d %H:%M:%S%z")})
        reload_check = export_dense_and_check(model, tokenizer, token_ids)
        status = {"status": "DENSE_EXPORT_RELOAD_PASS", "calibration": "PASS", "dense_export_reload": reload_check["status"],
                  "effective_optimizer_updates": EXPECTED_UPDATES, "updated_at": time.strftime("%Y-%m-%d %H:%M:%S%z")}
        write_json(STATUS_PATH, status)
        print(json.dumps(status, indent=2), flush=True)
        return 0
    except BaseException as exc:
        write_json(STATUS_PATH, {"status": "FAILED_OR_INTERRUPTED", "exception_type": type(exc).__name__,
                                 "message": str(exc), "traceback": traceback.format_exc(),
                                 "latest_checkpoint": str(CHECKPOINTS / "LATEST.json") if (CHECKPOINTS / "LATEST.json").exists() else None,
                                 "updated_at": time.strftime("%Y-%m-%d %H:%M:%S%z")})
        raise


if __name__ == "__main__":
    raise SystemExit(main())
