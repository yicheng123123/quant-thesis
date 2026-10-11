from __future__ import annotations

import csv
import hashlib
import importlib.machinery
import importlib.util
import json
import math
import sys
import time
import types
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RESULTS = ROOT.parent
PREFLIGHT = ROOT / "LOCAL_PREFLIGHT"
OFFICIAL_PORT = HERE / "official_port"
CHECKPOINTS = HERE / "checkpoints"
BUNDLE = RESULTS / "range_w4_omniquant_kaggle_v20_1" / "kaggle_bundle"
CALIBRATION_PATH = BUNDLE / "calibration" / "CALIBRATION_MANIFEST.json"
SUMMARY_PATH = HERE / "M3_TRAINING_SUMMARY.json"


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def bool_value(value: str) -> bool:
    return str(value).strip().lower() == "true"


def parse_local_time(value: str) -> datetime:
    return datetime.strptime(value, "%Y-%m-%d %H:%M:%S%z")


def check_checkpoints(lock: dict[str, Any], runner) -> tuple[dict[str, Any], dict[str, Any]]:
    latest_path = CHECKPOINTS / "LATEST.json"
    latest = read_json(latest_path)
    if latest.get("checkpoint_dir") != "block_23" or latest.get("completed_block_id") != 23:
        raise RuntimeError("latest v21.4 checkpoint is not block_23")
    _, completed, _ = runner.verify_latest_checkpoint(lock)
    if completed != 24:
        raise RuntimeError(f"final checkpoint completed layer count is {completed}, expected 24")
    manifest = read_json(CHECKPOINTS / "block_23" / "CHECKPOINT_MANIFEST.json")
    if manifest.get("scheduled_training_iterations") != 61440:
        raise RuntimeError("final checkpoint scheduled iteration total is not 61,440")
    if manifest.get("successful_optimizer_steps") != 61438 or manifest.get("gradscaler_skipped_steps") != 2:
        raise RuntimeError("final checkpoint counts differ from observed step logs")
    if manifest["successful_optimizer_steps"] + manifest["gradscaler_skipped_steps"] != manifest["scheduled_training_iterations"]:
        raise RuntimeError("final checkpoint scheduled/success/skip totals do not balance")

    for block in range(21, 24):
        block_dir = CHECKPOINTS / f"block_{block:02d}"
        block_manifest_path = block_dir / "CHECKPOINT_MANIFEST.json"
        block_manifest = read_json(block_manifest_path)
        if block_manifest.get("completed_block_id") != block or block_manifest.get("completed_layer_count") != block + 1:
            raise RuntimeError(f"block {block} checkpoint boundary mismatch")
        expected_scheduled = (block + 1) * 20 * 128
        if block_manifest.get("scheduled_training_iterations") != expected_scheduled:
            raise RuntimeError(f"block {block} scheduled count mismatch")
        if block_manifest.get("successful_optimizer_steps", -1) + block_manifest.get("gradscaler_skipped_steps", -1) != expected_scheduled:
            raise RuntimeError(f"block {block} success/skip count mismatch")
        if block_manifest.get("formal_protocol_lock_sha256") != sha_file(HERE / "FORMAL_PROTOCOL_LOCK.json"):
            raise RuntimeError(f"block {block} protocol lock mismatch")
        for key in ("state", "rng"):
            artifact = block_dir / block_manifest[f"{key}_file"]
            if artifact.stat().st_size != block_manifest[f"{key}_file_bytes"] or sha_file(artifact) != block_manifest[f"{key}_file_sha256"]:
                raise RuntimeError(f"block {block} {key} hash mismatch")

    # v21.4 imports the last v21.1 prefix checkpoint; earlier prefix blocks stay
    # in the immutable source run and are checked there rather than copied here.
    historical_root = RESULTS / "range_w4_omniquant_full_v21_1" / "M3" / "checkpoints"
    for block in range(21):
        path = historical_root / f"block_{block:02d}" / "CHECKPOINT_MANIFEST.json"
        if not path.is_file():
            raise RuntimeError(f"historical v21.1 checkpoint block_{block:02d} is missing")
    return latest, manifest


def check_training_logs(attempt_id: str) -> dict[str, Any]:
    step_path = HERE / "logs" / "training_steps.csv"
    amp_path = HERE / "AMP_EVENT_LOG.csv"
    epoch_path = HERE / "logs" / "training_epochs.csv"
    block_path = HERE / "logs" / "block_health.csv"
    with step_path.open(newline="", encoding="utf-8") as stream:
        all_step_rows = list(csv.DictReader(stream))
    rows = [row for row in all_step_rows if row["attempt_id"] == attempt_id]
    expected_locations = {(layer, epoch, sample) for layer in (21, 22, 23) for epoch in range(20) for sample in range(128)}
    locations = {(int(row["layer"]), int(row["epoch_index"]), int(row["sample_index"])) for row in rows}
    if len(rows) != 7680 or locations != expected_locations:
        raise RuntimeError(f"formal step log coverage mismatch: rows={len(rows)} unique={len(locations)}")
    if any(not math.isfinite(float(row["loss"])) for row in rows):
        raise RuntimeError("formal step log contains a nonfinite reconstruction loss")
    if any(not bool_value(row["lwc_parameters_finite"]) or not bool_value(row["optimizer_state_finite"]) or not bool_value(row["scaler_state_finite"]) for row in rows):
        raise RuntimeError("formal step log contains nonfinite parameter, optimizer, or scaler state")
    skipped = [row for row in rows if bool_value(row["optimizer_step_skipped"])]
    successful = [row for row in rows if bool_value(row["optimizer_step_executed"])]
    if len(skipped) != 2 or len(successful) != 7678:
        raise RuntimeError("formal step log successful/skip totals mismatch")
    if any(bool_value(row["optimizer_step_executed"]) for row in skipped):
        raise RuntimeError("a GradScaler skip row executed optimizer.step")
    if any(not bool_value(row["lwc_parameters_unchanged_on_skip"]) or not bool_value(row["optimizer_state_unchanged_on_skip"]) for row in skipped):
        raise RuntimeError("skipped GradScaler step changed LWC parameters or optimizer state")
    if any(not bool_value(row["lwc_grad_finite"]) and not bool_value(row["optimizer_step_skipped"]) for row in rows):
        raise RuntimeError("nonfinite owned LWC gradients occurred without a native scaler skip")
    if any(not bool_value(row["optimizer_step_skipped"]) and not bool_value(row["optimizer_step_executed"]) for row in rows):
        raise RuntimeError("a scheduled non-skip iteration did not execute AdamW")
    if any(float(row["found_inf"]) > 0 and not bool_value(row["optimizer_step_skipped"]) for row in rows):
        raise RuntimeError("GradScaler found_inf did not match optimizer skip")

    with amp_path.open(newline="", encoding="utf-8") as stream:
        all_amp_rows = list(csv.DictReader(stream))
    amp_rows = [row for row in all_amp_rows if row["attempt_id"] == attempt_id]
    if len(amp_rows) != 2:
        raise RuntimeError(f"AMP event log contains {len(amp_rows)} overflow rows, expected 2")
    if {(int(row["layer"]), int(row["epoch_index"]), int(row["sample_index"])) for row in amp_rows} != {
        (21, 0, 0), (21, 0, 74)
    }:
        raise RuntimeError("AMP event locations differ from the formal step log")
    if any(not bool_value(row["optimizer_step_skipped"]) or not bool_value(row["parameters_finite"]) or not bool_value(row["optimizer_state_unchanged"]) for row in amp_rows):
        raise RuntimeError("one or more AMP overflow events were not correctly skipped with unchanged state")
    for row in amp_rows:
        if not float(row["scale_after"]) < float(row["scale_before"]):
            raise RuntimeError("a GradScaler overflow did not back off the scale")

    with epoch_path.open(newline="", encoding="utf-8") as stream:
        epochs = [row for row in csv.DictReader(stream) if row["attempt_id"] == attempt_id]
    if len(epochs) != 60 or any(not bool_value(row["lwc_finite"]) or not math.isfinite(float(row["loss_mean"])) for row in epochs):
        raise RuntimeError("epoch log is incomplete or contains nonfinite loss/LWC state")
    with block_path.open(newline="", encoding="utf-8") as stream:
        blocks = [row for row in csv.DictReader(stream) if row["attempt_id"] == attempt_id]
    return {
        "rows": rows,
        "step_row_count": len(rows),
        "successful_optimizer_steps": len(successful),
        "gradscaler_skipped_steps": len(skipped),
        "finite_losses": len(rows),
        "finite_gradients": sum(bool_value(row["lwc_grad_finite"]) for row in rows),
        "nonfinite_gradient_iterations_all_skipped": sum(not bool_value(row["lwc_grad_finite"]) and bool_value(row["optimizer_step_skipped"]) for row in rows),
        "finite_norm_iterations": sum(bool_value(row["grad_norm_finite"]) for row in rows),
        "nonfinite_norm_iterations": sum(not bool_value(row["grad_norm_finite"]) for row in rows),
        "epoch_rows": epochs,
        "nonfinite_epoch_norms": sum(not bool_value(row["norm_finite"]) for row in epochs),
        "block_rows": blocks,
        "amp_rows": amp_rows,
        "all_attempt_step_log_rows": len(all_step_rows),
    }


def build_training_summary(run_status: dict[str, Any], manifest: dict[str, Any], logs: dict[str, Any]) -> dict[str, Any]:
    attempt_id = run_status["attempt_id"]
    start_text = attempt_id[:15]
    start_naive = datetime.strptime(start_text, "%Y%m%d_%H%M%S")
    commit_time = parse_local_time(manifest["committed_at_local"])
    run_start = start_naive.replace(tzinfo=timezone(commit_time.utcoffset()))
    wall_span = (commit_time - run_start).total_seconds()
    if wall_span <= 0:
        raise RuntimeError("formal attempt start and final checkpoint timestamps are inconsistent")
    counts = {
        "scheduled_iterations": 7680,
        "successful_optimizer_steps": 7678,
        "gradscaler_skipped_steps": 2,
        "finite_losses": logs["finite_losses"],
        "finite_gradients": logs["finite_gradients"],
        "nonzero_finite_gradients": None,
        "finite_norm_iterations": logs["finite_norm_iterations"],
        "nonfinite_norm_iterations": logs["nonfinite_norm_iterations"],
        "nonfinite_epoch_norms": logs["nonfinite_epoch_norms"],
        "optimized_blocks": 3,
        "quantized_blocks": 3,
        "prefix_replayed_blocks": 21,
        "historical_nonzero_gradient_count_recorded": False,
    }
    return {
        "status": "PASS",
        "attempt_id": attempt_id,
        "formal_protocol_lock_sha256": sha_file(HERE / "FORMAL_PROTOCOL_LOCK.json"),
        "model_revision": lock["model_revision"],
        "model_weight_sha256": lock["model_weight_sha256"],
        "calibration_manifest_sha256": lock["calibration_manifest_sha256"],
        "ordered_token_hashes_sha256": lock["ordered_token_hashes_sha256"],
        "completed_blocks": 24,
        "scheduled_iterations": manifest["scheduled_training_iterations"],
        "successful_optimizer_steps": manifest["successful_optimizer_steps"],
        "gradscaler_skipped_steps": manifest["gradscaler_skipped_steps"],
        "scheduled_iterations_in_this_attempt": logs["step_row_count"],
        "successful_optimizer_steps_in_this_attempt": logs["successful_optimizer_steps"],
        "gradscaler_skipped_steps_in_this_attempt": logs["gradscaler_skipped_steps"],
        "resume_completed_layers": 21,
        "prefix_blocks_replayed": 21,
        "training_step_check_counts": counts,
        "calibration_wall_seconds": wall_span,
        "calibration_wall_hours": wall_span / 3600,
        "calibration_wall_measurement": "wall-clock span from formal attempt_id timestamp to block_23 committed_at_local; includes model/tokenizer load, frozen prefix capture/replay, blocks21-23 and checkpoint commit. The runner's internal calibration timer value was not persisted because post-training summary assembly raised KeyError.",
        "calibration_wall_inner_timer_seconds": None,
        "training_start_local": run_start.isoformat(),
        "final_checkpoint_committed_at_local": manifest["committed_at_local"],
        "peak_allocated_bytes": int(1756.9970703125 * (1024 ** 2)),
        "peak_reserved_bytes": None,
        "peak_allocated_gib": (1756.9970703125 * 1024 ** 2) / (1024 ** 3),
        "peak_reserved_gib": None,
        "peak_memory_provenance": "formal console epoch reports repeatedly showed torch.cuda.max_memory_allocated=1756.9970703125 MiB for blocks21-23; final max_memory_reserved summary was not written after the post-training KeyError",
        "checkpoint_latest": str(CHECKPOINTS / "LATEST.json"),
        "checkpoint_manifest_sha256": sha_file(CHECKPOINTS / "block_23" / "CHECKPOINT_MANIFEST.json"),
        "formal_steps_completed_before_runner_postprocessing_error": True,
        "training_summary_reconstructed_from_checkpoint_and_step_logs": True,
        "nonzero_finite_gradients": None,
        "model_load_seconds": None,
    }


def main() -> None:
    global lock
    if SUMMARY_PATH.exists() or (HERE / "DENSE_EXPORT_RELOAD_CHECK.json").exists():
        raise RuntimeError("post-training artifacts already exist; refusing to overwrite")
    run_status = read_json(HERE / "RUN_STATUS.json")
    if run_status.get("status") != "FAILED_OR_INTERRUPTED" or run_status.get("exception_type") != "KeyError" or "manifest" not in run_status.get("message", ""):
        raise RuntimeError("recovery script only handles the recorded post-training manifest assembly error")
    executed_runner_sha = sha_file(HERE / "run_formal_m3.py")
    current_lock = read_json(HERE / "FORMAL_PROTOCOL_LOCK.json")
    if executed_runner_sha != current_lock.get("formal_runner_sha256"):
        raise RuntimeError("formal runner source no longer matches its pre-run protocol lock")

    sys.path.insert(0, str(OFFICIAL_PORT))
    sys.path.insert(0, str(PREFLIGHT))
    if importlib.util.find_spec("termcolor") is None:
        termcolor = types.ModuleType("termcolor")
        termcolor.colored = lambda text, *args, **kwargs: text
        termcolor.__spec__ = importlib.machinery.ModuleSpec("termcolor", loader=None)
        sys.modules["termcolor"] = termcolor
    spec = importlib.util.spec_from_file_location("v21_4_runner_rebuild", HERE / "run_formal_m3.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load frozen v21.4 formal runner")
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    runner.write_runtime_environment()
    config, calibration, token_ids, lock = runner.frozen_context()

    latest, checkpoint_manifest = check_checkpoints(lock, runner)
    with (HERE / "logs" / "training_steps.csv").open(newline="", encoding="utf-8") as stream:
        attempt_ids = {row["attempt_id"] for row in csv.DictReader(stream) if row.get("attempt_id")}
    if len(attempt_ids) != 1:
        raise RuntimeError(f"expected one formal v21.4 training attempt in step log, found {sorted(attempt_ids)}")
    attempt_id = next(iter(attempt_ids))
    run_status["attempt_id"] = attempt_id
    logs = check_training_logs(attempt_id)
    training_summary = build_training_summary(run_status, checkpoint_manifest, logs)
    if training_summary["scheduled_iterations"] != 61440 or training_summary["successful_optimizer_steps"] + training_summary["gradscaler_skipped_steps"] != 61440:
        raise RuntimeError("reconstructed training summary does not meet the frozen iteration accounting")

    # Save the original process failure separately; it occurred after the final
    # checkpoint commit and did not undo or alter any training state.
    failure_copy = HERE / "RUN_STATUS_TRAINING_FINALIZATION_FAILURE.json"
    if not failure_copy.exists():
        failure_copy.write_bytes((HERE / "RUN_STATUS.json").read_bytes())

    from transformers import AutoTokenizer
    from run_local_preflight import load_exact_model

    model, identity = load_exact_model(runner.MODEL_ID, runner.MODEL_REVISION,
                                       str(runner.FROZEN_MODEL), runner.DEVICE, HERE)
    tokenizer = AutoTokenizer.from_pretrained(str(runner.FROZEN_MODEL), local_files_only=True, use_fast=True)
    if tokenizer.__class__.__name__ != "Qwen2TokenizerFast":
        raise RuntimeError(f"unexpected frozen tokenizer class {tokenizer.__class__.__name__}")
    if identity.get("actual_snapshot_revision") != runner.MODEL_REVISION:
        raise RuntimeError("post-training reconstruction loaded a different model revision")

    resume_path, completed, _ = runner.verify_latest_checkpoint(lock)
    if completed != 24 or resume_path is None:
        raise RuntimeError("cannot reconstruct dense output without the verified 24-block checkpoint")
    old_manifest = read_json(resume_path.parent / "CHECKPOINT_MANIFEST.json")
    args = runner.make_args(config, resume_path, completed, HERE / "postprocess_output")
    args.v21_prefix_successful_optimizer_steps = int(old_manifest["successful_optimizer_steps"])
    args.v21_prefix_gradscaler_skipped_steps = int(old_manifest["gradscaler_skipped_steps"])
    args.v21_resume_completed_layers = completed
    args.smoke_layer_limit = 24
    args.smoke_step_limit = None
    (HERE / "postprocess_output").mkdir(parents=True, exist_ok=True)
    dataloader = [(torch.tensor(ids, dtype=torch.long).view(1, -1), index)
                  for index, ids in enumerate(token_ids)]
    lm = SimpleNamespace(model=model, device=runner.DEVICE, _device=runner.DEVICE, seqlen=runner.SEQUENCE_LENGTH)
    sys.path.insert(0, str(OFFICIAL_PORT))
    omni_module = __import__("quantize.omniquant", fromlist=["omniquant"])
    torch.cuda.reset_peak_memory_stats(runner.DEVICE)
    torch.cuda.synchronize(runner.DEVICE)
    reconstruction_started = time.perf_counter()
    reconstructed_model = omni_module.omniquant(lm, args, dataloader, None, None, runner.NullLogger())
    torch.cuda.synchronize(runner.DEVICE)
    reconstruction_seconds = time.perf_counter() - reconstruction_started
    reconstruction_peak_allocated = int(torch.cuda.max_memory_allocated(runner.DEVICE))
    reconstruction_peak_reserved = int(torch.cuda.max_memory_reserved(runner.DEVICE))
    export_reload = runner.export_dense_and_check(reconstructed_model, tokenizer, token_ids)

    training_summary.update({
        "reconstruction_after_checkpoint_error_seconds": reconstruction_seconds,
        "reconstruction_after_checkpoint_peak_allocated_bytes": reconstruction_peak_allocated,
        "reconstruction_after_checkpoint_peak_reserved_bytes": reconstruction_peak_reserved,
        "post_training_export_reload_seconds": export_reload["dense_export_reload_total_seconds"],
        "post_training_export_reload_peak_allocated_bytes": export_reload["export_peak_allocated_bytes"],
        "post_training_export_reload_peak_reserved_bytes": export_reload["export_peak_reserved_bytes"],
        "post_training_finalization_error": run_status.get("message"),
        "recovery_did_not_repeat_optimizer_steps": True,
    })
    write_json(SUMMARY_PATH, training_summary)
    write_json(HERE / "POST_TRAINING_RECONSTRUCTION.json", {
        "status": "PASS",
        "purpose": "rebuild in-memory dense model for export/evaluation from completed block_23 checkpoint after the original runner failed while assembling its summary",
        "formal_checkpoint_state_loaded": str(resume_path),
        "latest_checkpoint_pointer_sha256": sha_file(CHECKPOINTS / "LATEST.json"),
        "checkpoint_manifest_sha256": sha_file(resume_path.parent / "CHECKPOINT_MANIFEST.json"),
        "completed_blocks": completed,
        "optimizer_steps_executed_during_reconstruction": 0,
        "training_reconstruction_is_replay_only": True,
        "prefix_replay_seconds": reconstruction_seconds,
        "prefix_replay_peak_allocated_bytes": reconstruction_peak_allocated,
        "prefix_replay_peak_reserved_bytes": reconstruction_peak_reserved,
        "model_identity_check": identity,
        "dense_export_reload_status": export_reload["status"],
        "formal_training_runner_failure_preserved": failure_copy.name,
    })
    write_json(HERE / "RUN_STATUS.json", {
        "status": "FORMAL_TRAINING_AND_DENSE_EXPORT_RELOAD_PASS",
        "completed_blocks": 24,
        "scheduled_iterations": 61440,
        "successful_optimizer_steps": 61438,
        "gradscaler_skipped_steps": 2,
        "original_post_training_summary_error": run_status.get("message"),
        "dense_export_reload": export_reload["status"],
        "updated_at_local": time.strftime("%Y-%m-%d %H:%M:%S%z"),
    })
    print(json.dumps({"training_summary_status": "PASS", "completed_blocks": 24,
                      "scheduled_iterations": 61440, "successful_optimizer_steps": 61438,
                      "gradscaler_skipped_steps": 2, "dense_export_reload": export_reload["status"],
                      "formal_execution_wall_span_seconds": training_summary["calibration_wall_seconds"],
                      "post_training_reconstruction_seconds": reconstruction_seconds,
                      "dense_export_reload_seconds": export_reload["dense_export_reload_total_seconds"]}, indent=2))


if __name__ == "__main__":
    main()
