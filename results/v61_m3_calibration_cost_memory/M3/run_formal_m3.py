from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import importlib
import math
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
PREFLIGHT = Path('E:\\quant-thesis\\experiments\\range_search_w4\\results\\range_w4_omniquant_resume_v21_4\\LOCAL_PREFLIGHT')
BUNDLE = Path('E:\\quant-thesis\\experiments\\range_search_w4\\results\\range_w4_omniquant_kaggle_v20_1\\kaggle_bundle')
FROZEN_MODEL = Path(r"E:\quant-thesis\artifacts\models\qwen2.5-0.5b-instruct-7ae5576\7ae557604adf67be50417f59c2c2f167def9a775")
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
MODEL_REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"
DEVICE = torch.device("cuda:0")
EPOCHS = 20
SAMPLES = 128
SEQUENCE_LENGTH = 2048
EXPECTED_SCHEDULED_ITERATIONS = 61440
SEED = 20261005
CHECKPOINTS = HERE / "checkpoints"
EXPORT_DIR = HERE / "dense_export"
LOG_DIR = HERE / "logs"
STEP_LOG = LOG_DIR / "training_steps.csv"
EPOCH_LOG = LOG_DIR / "training_epochs.csv"
BLOCK_LOG = LOG_DIR / "block_health.csv"
AMP_EVENT_LOG = HERE / "AMP_EVENT_LOG.csv"
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


def csv_has_data_rows(path: Path) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return False
    with path.open(newline="", encoding="utf-8") as stream:
        return next(csv.DictReader(stream), None) is not None


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
    expected_runtime = {
        "python": str(expected_env.get("python_version", "")).split(" ")[0],
        "torch": expected_env.get("torch_version"),
        "transformers": expected_env.get("transformers_version"),
        "tokenizers": expected_env.get("tokenizers_version"),
        "cuda_runtime": expected_env.get("cuda_runtime"),
    }
    mismatches = {key: {"expected": value, "actual": runtime_env[key]}
                  for key, value in expected_runtime.items() if value and runtime_env[key] != value}
    if mismatches:
        raise RuntimeError(f"current Python/ML runtime differs from the passing frozen preflight: {mismatches}")
    if runtime_env["tf32_matmul"] or runtime_env["tf32_cudnn"]:
        raise RuntimeError("TF32 must remain disabled under the frozen protocol")

    lock = {
        "protocol_id": "range_w4_omniquant_resume_v21_4_M3",
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
        "protocol_scheduled_iteration_total": EXPECTED_SCHEDULED_ITERATIONS,
        "training_update_semantics": "fixed epoch/sample schedule; dynamic GradScaler may skip optimizer steps",
        "runtime_environment": runtime_env,
        "resume_semantics": "block-boundary only; replay completed official quantized prefix; incomplete block restarts from pristine weights",
        "algorithm_changes": [],
    }
    lock_path = HERE / "FORMAL_PROTOCOL_LOCK.json"
    if lock_path.exists():
        previous_lock = load_json(lock_path)
        if previous_lock != lock:
            previous_status = load_json(STATUS_PATH) if STATUS_PATH.exists() else {}
            latest_is_only_migrated_prefix = False
            latest_path = CHECKPOINTS / "LATEST.json"
            if latest_path.exists():
                try:
                    latest_record = load_json(latest_path)
                    latest_manifest_path = CHECKPOINTS / latest_record["checkpoint_dir"] / "CHECKPOINT_MANIFEST.json"
                    latest_manifest = load_json(latest_manifest_path)
                    latest_is_only_migrated_prefix = (
                        latest_record.get("checkpoint_dir") == "block_20"
                        and latest_record.get("completed_block_id") == 20
                        and latest_manifest.get("completed_block_id") == 20
                        and latest_manifest.get("scheduled_training_iterations") == 53760
                    )
                except Exception:
                    latest_is_only_migrated_prefix = False
            safe_bootstrap_failure = (
                previous_status.get("status") == "FAILED_OR_INTERRUPTED"
                and previous_status.get("exception_type") == "ModuleNotFoundError"
                and previous_status.get("message") == "No module named 'run_local_preflight'"
            )
            pristine_v21_4_setup = (
                previous_lock.get("protocol_id") == "range_w4_omniquant_full_v21_1_M3"
                or previous_lock.get("protocol_id") == "range_w4_omniquant_resume_v21_4_M3"
            ) and (
                (not STATUS_PATH.exists() or safe_bootstrap_failure)
                and (not latest_path.exists() or latest_is_only_migrated_prefix)
                and not (HERE / "M3_TRAINING_SUMMARY.json").exists()
                and not csv_has_data_rows(STEP_LOG)
                and not csv_has_data_rows(EPOCH_LOG)
                and not csv_has_data_rows(BLOCK_LOG)
                and not csv_has_data_rows(AMP_EVENT_LOG)
            )
            if pristine_v21_4_setup:
                preserved_path = HERE / f"FORMAL_PROTOCOL_LOCK_PRE_FORMAL_{sha_file(lock_path)[:12]}.json"
                if preserved_path.exists():
                    raise RuntimeError("pre-formal protocol lock preservation file already exists")
                shutil.copy2(lock_path, preserved_path)
                note_path = HERE / f"PROTOCOL_LOCK_MIGRATION_NOTE_{sha_file(preserved_path)[:12]}.json"
                write_json(note_path, {
                    "status": "SAFE_PRE_RUN_PROTOCOL_LOCK_MIGRATION",
                    "preserved_lock": preserved_path.name,
                    "preserved_lock_sha256": sha_file(preserved_path),
                    "prior_protocol_id": previous_lock.get("protocol_id"),
                    "new_protocol_id": lock["protocol_id"],
                    "formal_training_started": False,
                    "latest_checkpoint_pointer_present": latest_path.exists(),
                    "pretraining_bootstrap_failure": previous_status if safe_bootstrap_failure else None,
                    "training_logs_present": False,
                    "reason": "A correctness metadata fix was made before formal training; only the untouched imported block_20 prefix is present and was retained for validation.",
                    "algorithm_semantics_changed": False,
                })
                write_json(lock_path, lock)
            else:
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


def optimizer_state_signature(optimizer: torch.optim.Optimizer) -> dict[str, Any]:
    """Read-only metadata signature used to prove scaler skips leave AdamW state untouched."""
    entries = []
    tensor_count = 0
    for group_index, group in enumerate(optimizer.param_groups):
        for parameter_index, parameter in enumerate(group["params"]):
            state = optimizer.state.get(parameter, {})
            for key, value in sorted(state.items(), key=lambda item: str(item[0])):
                if isinstance(value, torch.Tensor):
                    tensor_count += 1
                    descriptor = {
                        "kind": "tensor", "identity": id(value), "version": int(value._version),
                        "shape": list(value.shape), "dtype": str(value.dtype), "device": str(value.device),
                    }
                else:
                    descriptor = {"kind": type(value).__name__, "value": repr(value)}
                entries.append([group_index, parameter_index, str(key), descriptor])
    encoded = json.dumps(entries, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {"sha256": sha_bytes(encoded), "entry_count": len(entries), "tensor_count": tensor_count}


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
        "protocol_scheduled_iteration_total": EXPECTED_SCHEDULED_ITERATIONS,
    }


def save_block_checkpoint(state: dict[Any, Any], lock: dict[str, Any], save_fn, progress: dict[str, Any]) -> dict[str, Any]:
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
        "scheduled_training_iterations": completed_layers * EPOCHS * SAMPLES,
        "successful_optimizer_steps": progress["prefix_successful_optimizer_steps"] + progress["counts"]["successful_optimizer_steps"],
        "gradscaler_skipped_steps": progress["prefix_gradscaler_skipped_steps"] + progress["counts"]["gradscaler_skipped_steps"],
        "update_count_provenance": "historical prefix successful steps inferred from finite owned gradients and native scaler source; resumed steps counted by optimizer hooks",
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
    if manifest["scheduled_training_iterations"] != completed * EPOCHS * SAMPLES:
        raise RuntimeError("checkpoint scheduled iteration count is inconsistent")
    if manifest["successful_optimizer_steps"] + manifest["gradscaler_skipped_steps"] != manifest["scheduled_training_iterations"]:
        raise RuntimeError("checkpoint successful/skipped update counts do not sum to scheduled iterations")
    state = torch.load(checkpoint_dir / manifest["state_file"], map_location="cpu", weights_only=False)
    state_keys = sorted(int(key) for key in state)
    if state_keys != list(range(completed)):
        raise RuntimeError("checkpoint state is not a contiguous prefix from block zero")
    rng_state = torch.load(checkpoint_dir / manifest["rng_file"], map_location="cpu", weights_only=False)
    return checkpoint_dir / manifest["state_file"], completed, {"rng_state": rng_state, "state": state, "manifest": manifest}


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

    epoch_csv = CsvAppend(EPOCH_LOG, ["attempt_id", "layer", "epoch_index", "loss_mean", "norm_mean", "norm_finite", "lwc_param_count", "lwc_finite"])
    step_csv = CsvAppend(STEP_LOG, ["attempt_id", "scheduled_iteration_count", "layer", "epoch_index", "sample_index", "loss", "scale_before", "found_inf", "optimizer_step_skipped", "scale_after", "lwc_grad_param_count", "lwc_grad_finite", "optimizer_step_executed", "lwc_parameters_finite", "lwc_parameters_unchanged_on_skip", "optimizer_state_finite", "optimizer_state_unchanged_on_skip", "scaler_state_finite", "grad_norm_finite"])
    amp_csv = CsvAppend(AMP_EVENT_LOG, ["attempt_id", "scheduled_iteration_count", "layer", "epoch_index", "sample_index", "loss", "scale_before", "found_inf", "optimizer_step_skipped", "scale_after", "parameters_finite", "optimizer_state_unchanged"])
    block_csv = CsvAppend(BLOCK_LOG, ["attempt_id", "layer", "event", "lwc_parameter_count", "lwc_changed_count", "lwc_max_abs_change", "float_tensor_count", "all_float_finite", "quantlinear_count", "scales_and_zeros_finite"])
    baselines: dict[int, dict[str, torch.Tensor]] = {}
    prefix_iterations = args.v21_resume_completed_layers * EPOCHS * SAMPLES
    counts = {"scheduled_iterations": 0, "successful_optimizer_steps": 0, "gradscaler_skipped_steps": 0,
              "finite_losses": 0, "finite_gradients": 0, "nonzero_finite_gradients": 0,
              "finite_norm_iterations": 0, "nonfinite_norm_iterations": 0,
              "nonfinite_epoch_norms": 0,
              "optimized_blocks": 0, "quantized_blocks": 0, "prefix_replayed_blocks": 0}
    args.v21_count_state = {
        "prefix_successful_optimizer_steps": args.v21_prefix_successful_optimizer_steps,
        "prefix_gradscaler_skipped_steps": args.v21_prefix_gradscaler_skipped_steps,
        "counts": counts,
    }

    def epoch_cb(layer: int, epoch: int, loss: torch.Tensor, norm: torch.Tensor, module):
        params = [(name, p) for name, p in module.named_parameters() if "bound_factor" in name]
        finite = all(bool(torch.isfinite(p.detach()).all().item()) for _, p in params)
        norm_finite = bool(torch.isfinite(norm.detach()).all().item())
        row = {"attempt_id": attempt_id, "layer": layer, "epoch_index": epoch,
               "loss_mean": float(loss.detach().float().item()), "norm_mean": float(norm.detach().float().item()),
               "norm_finite": norm_finite, "lwc_param_count": len(params), "lwc_finite": finite}
        epoch_csv.write(row, flush=True)
        counts["nonfinite_epoch_norms"] += int(not norm_finite)
        if not finite or not np.isfinite(row["loss_mean"]):
            raise FloatingPointError(f"nonfinite epoch diagnostics at layer={layer}, epoch={epoch}")

    def step_check(module, layer: int, epoch: int, sample: int, loss: torch.Tensor, norm: torch.Tensor,
                   optimizer: torch.optim.Optimizer, amp: dict[str, Any], hook_counts: dict[str, int],
                   parameter_snapshot: list[torch.Tensor], optimizer_state_before: dict[str, Any]):
        params = list(lwc_parameters(module))
        grads = [param.grad.detach().reshape(-1) for param in params if param.grad is not None]
        if len(grads) != len(params) or not grads:
            raise FloatingPointError(f"missing LWC gradients at layer={layer}, epoch={epoch}, sample={sample}")
        flat = torch.cat([grad.float() for grad in grads])
        flags = torch.stack((torch.isfinite(flat).all(), torch.any(flat != 0))).detach().cpu().tolist()
        finite_grad, nonzero_grad = bool(flags[0]), bool(flags[1])
        loss_value = float(loss.detach().float().item())
        if amp is None:
            raise FloatingPointError("native GradScaler did not expose post-update diagnostics")
        if optimizer_state_before is None or "sha256" not in optimizer_state_before:
            raise FloatingPointError("optimizer state pre-step signature is missing")
        found_inf = float(amp["found_inf"])
        if not math.isfinite(found_inf) or found_inf < 0:
            raise FloatingPointError("GradScaler found_inf diagnostic is invalid")
        if not math.isfinite(float(amp["scale_before"])) or not math.isfinite(float(amp["scale_after"])) or not amp["scaler_state_finite"]:
            raise FloatingPointError("GradScaler state became nonfinite")
        if hook_counts["pre"] != hook_counts["post"] or hook_counts["pre"] not in (0, 1):
            raise FloatingPointError(f"unexpected AdamW step hook counts: {hook_counts}")
        step_executed = hook_counts["pre"] == 1
        skipped = found_inf > 0
        if skipped == step_executed:
            raise FloatingPointError("GradScaler found_inf and actual optimizer.step hook disagree")
        current_values = [p.detach() for p in params]
        params_finite = all(bool(torch.isfinite(p).all().item()) for p in current_values)
        state_tensors = [value for state in optimizer.state.values() for value in state.values() if isinstance(value, torch.Tensor)]
        state_finite = all(bool(torch.isfinite(value).all().item()) for value in state_tensors)
        state_after_signature = optimizer_state_signature(optimizer)
        state_unchanged = state_after_signature["sha256"] == optimizer_state_before["sha256"]
        # The native wrapper returns the unscaled norm to the caller; it is intentionally diagnostic.
        norm_finite = bool(torch.isfinite(norm).all().item())
        if skipped:
            if found_inf <= 0 or not amp["scale_after"] < amp["scale_before"]:
                raise FloatingPointError("detected overflow was not followed by a dynamic scale backoff")
            if any(not torch.equal(before, after) for before, after in zip(parameter_snapshot, current_values)):
                raise FloatingPointError("LWC parameter changed despite a skipped GradScaler step")
            if not state_unchanged:
                raise FloatingPointError("optimizer state changed despite a skipped GradScaler step")
        elif not finite_grad:
            raise FloatingPointError("successful GradScaler step had nonfinite optimizer-owned LWC gradients")
        if not params_finite or not state_finite:
            raise FloatingPointError("LWC parameters or optimizer state became nonfinite")
        iteration = prefix_iterations + counts["scheduled_iterations"] + 1
        row = {"attempt_id": attempt_id, "scheduled_iteration_count": iteration,
               "layer": layer, "epoch_index": epoch, "sample_index": sample, "loss": loss_value,
               "scale_before": amp["scale_before"], "found_inf": found_inf,
               "optimizer_step_skipped": skipped, "scale_after": amp["scale_after"],
               "lwc_grad_param_count": len(params), "lwc_grad_finite": finite_grad,
               "optimizer_step_executed": step_executed, "lwc_parameters_finite": params_finite,
               "lwc_parameters_unchanged_on_skip": skipped and all(torch.equal(before, after) for before, after in zip(parameter_snapshot, current_values)),
               "optimizer_state_finite": state_finite, "optimizer_state_unchanged_on_skip": skipped and state_unchanged,
               "scaler_state_finite": amp["scaler_state_finite"],
               "grad_norm_finite": norm_finite}
        step_csv.write(row, flush=(sample == SAMPLES - 1))
        if skipped:
            amp_csv.write({"attempt_id": attempt_id, "scheduled_iteration_count": iteration,
                           "layer": layer, "epoch_index": epoch, "sample_index": sample,
                           "loss": loss_value, "scale_before": amp["scale_before"],
                           "found_inf": found_inf, "optimizer_step_skipped": True,
                           "scale_after": amp["scale_after"], "parameters_finite": params_finite,
                           "optimizer_state_unchanged": state_unchanged}, flush=True)
        counts["scheduled_iterations"] += 1
        counts["successful_optimizer_steps"] += int(step_executed)
        counts["gradscaler_skipped_steps"] += int(skipped)
        counts["finite_losses"] += int(np.isfinite(loss_value))
        counts["finite_gradients"] += int(finite_grad)
        counts["nonzero_finite_gradients"] += int(finite_grad and nonzero_grad)
        counts["finite_norm_iterations"] += int(norm_finite)
        counts["nonfinite_norm_iterations"] += int(not norm_finite)
        if not np.isfinite(loss_value):
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
    args.v21_optimizer_state_snapshot = optimizer_state_signature
    if resume_state is not None:
        rng_state = resume_state["rng_state"]
        args.v21_restore_rng = lambda: restore_rng_state(rng_state)
    return counts, (step_csv, epoch_csv, block_csv, amp_csv)


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
    if completed_layers:
        old_manifest = load_json(resume_path.parent / "CHECKPOINT_MANIFEST.json")
        args = make_args(config, resume_path, completed_layers, HERE / "upstream_output")
        args.v21_prefix_successful_optimizer_steps = int(old_manifest["successful_optimizer_steps"])
        args.v21_prefix_gradscaler_skipped_steps = int(old_manifest["gradscaler_skipped_steps"])
    else:
        args = make_args(config, resume_path, completed_layers, HERE / "upstream_output")
        args.v21_prefix_successful_optimizer_steps = 0
        args.v21_prefix_gradscaler_skipped_steps = 0
    attempt_id = f"{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    write_runtime_environment()
    sys.path.insert(0, str(HERE / "official_port"))
    if importlib.util.find_spec("termcolor") is None:
        termcolor = types.ModuleType("termcolor")
        termcolor.colored = lambda text, *args, **kwargs: text
        sys.modules["termcolor"] = termcolor
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    counts, csv_logs = install_source_callbacks(args, lock, resume_state, attempt_id)
    args.v61_phase_callback = METER.switch

    def checkpointing_torch_save(value, f, *save_args, **save_kwargs):
        try:
            target = Path(os.fspath(f))
        except TypeError:
            target = None
        if target is not None and target.name == "omni_parameters.pth":
            METER.switch("checkpoint_save", len(value)-1)
            record = save_block_checkpoint(value, lock, original_torch_save, args.v21_count_state)
            METER.switch("block_cleanup", len(value)-1)
            print(f"CHECKPOINT_COMMITTED block={record['completed_block_id']} scheduled={record['manifest']['scheduled_training_iterations']} successful={record['manifest']['successful_optimizer_steps']} skipped={record['manifest']['gradscaler_skipped_steps']}", flush=True)
            return None
        return original_torch_save(value, f, *save_args, **save_kwargs)

    RUN_STATUS = {"status": "RUNNING", "attempt_id": attempt_id,
                  "formal_protocol_lock_sha256": sha_file(HERE / "FORMAL_PROTOCOL_LOCK.json"),
                  "resume_completed_layers": completed_layers,
                  "resumed_from": str(resume_path) if resume_path else None,
                  "scheduled_iterations_expected": EXPECTED_SCHEDULED_ITERATIONS,
                  "updated_at": time.strftime("%Y-%m-%d %H:%M:%S%z")}
    write_json(STATUS_PATH, RUN_STATUS)
    omni_module = importlib.import_module("quantize.omniquant")
    sys.path.insert(0, str(PREFLIGHT))
    from run_local_preflight import load_exact_model
    METER.switch("model_load")
    model, identity = load_exact_model(MODEL_ID, MODEL_REVISION, str(FROZEN_MODEL), DEVICE, HERE)
    METER.record_runtime(model, args)
    METER.switch("data_prepare")
    tokenizer = AutoTokenizer.from_pretrained(str(FROZEN_MODEL), local_files_only=True, use_fast=True)
    if tokenizer.__class__.__name__ != "Qwen2TokenizerFast":
        raise RuntimeError(f"unexpected frozen tokenizer class {tokenizer.__class__.__name__}")
    dataloader = [(torch.tensor(ids, dtype=torch.long).view(1, -1), index)
                  for index, ids in enumerate(token_ids)]
    lm = SimpleNamespace(model=model, device=DEVICE, _device=DEVICE, seqlen=SEQUENCE_LENGTH)
    calibration_wall_started = time.perf_counter()
    # stage/global peak bookkeeping is owned by METER
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
    expected_attempt_iterations = (24 - completed_layers) * EPOCHS * SAMPLES
    if counts["scheduled_iterations"] != expected_attempt_iterations:
        raise RuntimeError(f"this attempt scheduled iteration count mismatch: observed={counts['scheduled_iterations']} expected={expected_attempt_iterations}")
    if counts["successful_optimizer_steps"] + counts["gradscaler_skipped_steps"] != counts["scheduled_iterations"]:
        raise RuntimeError("attempted iterations do not equal successful steps plus GradScaler skips")
    if counts["finite_losses"] != counts["scheduled_iterations"] or counts["successful_optimizer_steps"] == 0:
        raise RuntimeError("formal finite-loss or successful-step gate failed")
    if counts["optimized_blocks"] != 24 - completed_layers or counts["quantized_blocks"] != 24 - completed_layers:
        raise RuntimeError("formal optimizer or quantized block count mismatch")
    if latest_record["manifest"]["scheduled_training_iterations"] != EXPECTED_SCHEDULED_ITERATIONS:
        raise RuntimeError("final scheduled iteration count differs from the frozen 61,440")
    if latest_record["manifest"]["successful_optimizer_steps"] + latest_record["manifest"]["gradscaler_skipped_steps"] != EXPECTED_SCHEDULED_ITERATIONS:
        raise RuntimeError("final successful/skipped counts do not equal scheduled iterations")

    summary = {
        "status": "PASS", "attempt_id": attempt_id, "formal_protocol_lock_sha256": sha_file(HERE / "FORMAL_PROTOCOL_LOCK.json"),
        "model_revision": MODEL_REVISION, "model_weight_sha256": lock["model_weight_sha256"],
        "calibration_manifest_sha256": lock["calibration_manifest_sha256"],
        "ordered_token_hashes_sha256": lock["ordered_token_hashes_sha256"],
        "completed_blocks": completed_after,
        "scheduled_iterations": latest_record["manifest"]["scheduled_training_iterations"],
        "successful_optimizer_steps": latest_record["manifest"]["successful_optimizer_steps"],
        "gradscaler_skipped_steps": latest_record["manifest"]["gradscaler_skipped_steps"],
        "scheduled_iterations_in_this_attempt": counts["scheduled_iterations"],
        "successful_optimizer_steps_in_this_attempt": counts["successful_optimizer_steps"],
        "gradscaler_skipped_steps_in_this_attempt": counts["gradscaler_skipped_steps"],
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
    torch.cuda.synchronize(DEVICE)
    METER.switch("export_finalize")
    export_total_started = time.perf_counter()
    # stage/global peak bookkeeping is owned by METER
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
    torch.cuda.synchronize(DEVICE)
    METER.switch("model_save")
    export_started = time.perf_counter()
    model.save_pretrained(EXPORT_DIR, safe_serialization=True, max_shard_size="2GB")
    tokenizer.save_pretrained(EXPORT_DIR)
    export_files = [{"path": p.relative_to(EXPORT_DIR).as_posix(), "bytes": p.stat().st_size, "sha256": sha_file(p)}
                    for p in sorted(EXPORT_DIR.rglob("*")) if p.is_file()]
    dense_save_seconds = time.perf_counter() - export_started
    write_json(HERE / "DENSE_EXPORT_MANIFEST.json", {"status": "PASS", "representation": "dense-dequant FP16",
                                                       "model_revision": MODEL_REVISION, "probe_token_ids_sha256": probe_hash,
                                                       "linear_modules_replaced": quantized_linears, "files": export_files})
    del model
    gc.collect()
    torch.cuda.empty_cache()

    torch.cuda.synchronize(DEVICE)
    METER.switch("reload_check")
    reload_started = time.perf_counter()
    reloaded = AutoModelForCausalLM.from_pretrained(str(EXPORT_DIR), torch_dtype=torch.float16,
                                                    low_cpu_mem_usage=True, attn_implementation="sdpa",
                                                    local_files_only=True).eval().to(DEVICE)
    with torch.inference_mode():
        actual = reloaded(torch.tensor(probe_ids, dtype=torch.long, device=DEVICE).view(1, -1), use_cache=False).logits.float().cpu()
    torch.cuda.synchronize(DEVICE)
    dense_reload_seconds = time.perf_counter() - reload_started
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
        "dense_save_seconds": dense_save_seconds,
        "dense_reload_and_probe_seconds": dense_reload_seconds,
        "dense_export_reload_total_seconds": time.perf_counter() - export_total_started,
        "export_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(DEVICE)),
        "export_peak_reserved_bytes": int(torch.cuda.max_memory_reserved(DEVICE)),
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
    METER.switch("source_validation")
    config, calibration, token_ids, lock = frozen_context()
    if not args_cli.resume and (CHECKPOINTS / "LATEST.json").exists():
        raise RuntimeError("checkpoint exists; use --resume to continue from the verified latest completed block")
    if args_cli.resume and not (CHECKPOINTS / "LATEST.json").exists():
        raise RuntimeError("--resume requested but no completed-block checkpoint exists")
    write_json(HERE / "RUN_STATUS.json", {"status": "FORMAL_M3_STARTING", "protocol_lock_sha256": sha_file(HERE / "FORMAL_PROTOCOL_LOCK.json"),
                                          "started_at_local": time.strftime("%Y-%m-%d %H:%M:%S%z"),
                                          "scheduled_iterations_expected": EXPECTED_SCHEDULED_ITERATIONS})
    try:
        summary = run_calibration(config, token_ids, lock)
        model = summary.pop("model")
        tokenizer = summary.pop("tokenizer")
        summary.pop("result_model")
        summary["status"] = "PASS"
        write_json(HERE / "M3_TRAINING_SUMMARY.json", summary)
        write_json(STATUS_PATH, {"status": "CALIBRATION_PASS", "attempt_id": summary["attempt_id"],
                                 "scheduled_iterations": summary["scheduled_iterations"],
                                 "successful_optimizer_steps": summary["successful_optimizer_steps"],
                                 "gradscaler_skipped_steps": summary["gradscaler_skipped_steps"],
                                 "calibration_wall_seconds": summary["calibration_wall_seconds"],
                                 "updated_at": time.strftime("%Y-%m-%d %H:%M:%S%z")})
        reload_check = export_dense_and_check(model, tokenizer, token_ids)
        status = {"status": "DENSE_EXPORT_RELOAD_PASS", "calibration": "PASS", "dense_export_reload": reload_check["status"],
                  "scheduled_iterations": EXPECTED_SCHEDULED_ITERATIONS,
                  "successful_optimizer_steps": summary["successful_optimizer_steps"],
                  "gradscaler_skipped_steps": summary["gradscaler_skipped_steps"],
                  "updated_at": time.strftime("%Y-%m-%d %H:%M:%S%z")}
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
    from measurement import StageMeter
    METER = StageMeter(ROOT)
    try:
        METER.switch("runtime_setup")
        result = main()
    except BaseException as exc:
        METER.finish("FAILED", repr(exc))
        raise
    else:
        METER.finish("COMPLETE")
        raise SystemExit(result)
