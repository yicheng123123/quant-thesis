from __future__ import annotations

import csv
import difflib
import hashlib
import importlib.util
import json
import math
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parent
M3 = ROOT / "M3"
CHECKPOINTS = M3 / "checkpoints"
REPO = ROOT.parents[3]
RESULTS = ROOT.parent
V21_1 = RESULTS / "range_w4_omniquant_full_v21_1"
V21_1_M3 = V21_1 / "M3"
V21_3 = RESULTS / "range_w4_omniquant_amp_audit_v21_3"
BUNDLE = RESULTS / "range_w4_omniquant_kaggle_v20_1" / "kaggle_bundle"
CALIBRATION = BUNDLE / "calibration" / "CALIBRATION_MANIFEST.json"
CONFIG = BUNDLE / "configs" / "m3_preflight.json"
PREFLIGHT = ROOT / "LOCAL_PREFLIGHT"
FROZEN_MODEL_MANIFEST = ROOT / "LOCAL_MODEL_MANIFEST.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def csv_has_data_rows(path: Path) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return False
    with path.open(newline="", encoding="utf-8") as stream:
        return next(csv.DictReader(stream), None) is not None


def walk_tensors(value: Any):
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from walk_tensors(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from walk_tensors(item)


def check_history() -> dict[str, Any]:
    source_checkpoint_root = V21_1_M3 / "checkpoints"
    source_pointer_path = source_checkpoint_root / "LATEST.json"
    source_manifest_path = source_checkpoint_root / "block_20" / "CHECKPOINT_MANIFEST.json"
    source_pointer = load_json(source_pointer_path)
    source_manifest = load_json(source_manifest_path)
    source_manifest_sha = sha256_file(source_manifest_path)
    if source_pointer.get("checkpoint_dir") != "block_20" or source_pointer.get("completed_block_id") != 20:
        raise RuntimeError("v21.1 LATEST pointer does not identify block_20")
    if source_pointer.get("manifest_sha256") != source_manifest_sha:
        raise RuntimeError("v21.1 block_20 manifest hash does not match LATEST")
    if source_manifest.get("completed_layer_count") != 21 or source_manifest.get("completed_block_id") != 20:
        raise RuntimeError("v21.1 block_20 boundary metadata mismatch")

    source_dir = source_checkpoint_root / "block_20"
    copied_dir = CHECKPOINTS / "block_20"
    copied_manifest = copied_dir / "CHECKPOINT_MANIFEST.json"
    historical_copy = copied_dir / "CHECKPOINT_MANIFEST_V21_1.json"
    manifest_for_source_comparison = historical_copy if historical_copy.exists() else copied_manifest
    if manifest_for_source_comparison.read_bytes() != source_manifest_path.read_bytes():
        raise RuntimeError("copied v21.1 checkpoint manifest differs from historical source")
    if source_manifest["formal_protocol_lock_sha256"] != sha256_file(V21_1_M3 / "FORMAL_PROTOCOL_LOCK.json"):
        raise RuntimeError("block_20 does not reference the v21.1 formal lock")
    if source_manifest["config_sha256"] != sha256_file(CONFIG):
        raise RuntimeError("block_20 config hash differs from frozen v20.1 config")
    calibration_sha = sha256_file(CALIBRATION)
    if source_manifest["calibration_manifest_sha256"] != calibration_sha:
        raise RuntimeError("block_20 calibration hash differs from frozen calibration manifest")

    files = {}
    for key in ("state", "rng"):
        name = source_manifest[f"{key}_file"]
        source_path = source_dir / name
        copied_path = copied_dir / name
        actual_sha = sha256_file(source_path)
        if source_path.stat().st_size != source_manifest[f"{key}_file_bytes"]:
            raise RuntimeError(f"historical checkpoint {key} size mismatch")
        if actual_sha != source_manifest[f"{key}_file_sha256"]:
            raise RuntimeError(f"historical checkpoint {key} SHA256 mismatch")
        if copied_path.stat().st_size != source_path.stat().st_size or sha256_file(copied_path) != actual_sha:
            raise RuntimeError(f"isolated block_20 copy failed {key} byte identity check")
        files[key] = {"path": str(copied_path.relative_to(ROOT)), "bytes": copied_path.stat().st_size, "sha256": actual_sha}

    state = torch.load(copied_dir / source_manifest["state_file"], map_location="cpu", weights_only=False)
    keys = sorted(int(key) for key in state)
    if keys != list(range(21)):
        raise RuntimeError(f"block_20 state is not a contiguous 0..20 prefix: {keys}")
    state_tensors = list(walk_tensors(state))
    state_float_tensors = [tensor for tensor in state_tensors if tensor.is_floating_point()]
    if not state_tensors or not all(bool(torch.isfinite(tensor).all()) for tensor in state_float_tensors):
        raise RuntimeError("block_20 checkpoint contains nonfinite or missing parameter tensors")

    rng = torch.load(copied_dir / source_manifest["rng_file"], map_location="cpu", weights_only=False)
    required_rng = {"python", "numpy", "torch_cpu", "torch_cuda"}
    if set(rng) != required_rng or not isinstance(rng["torch_cpu"], torch.Tensor) or not rng["torch_cuda"]:
        raise RuntimeError("block_20 checkpoint RNG state has unexpected structure")
    if not all(isinstance(item, torch.Tensor) for item in rng["torch_cuda"]):
        raise RuntimeError("block_20 CUDA RNG state entries are not tensors")

    attempt_id = "20261001_183359_6c00bf87"
    steps_path = V21_1_M3 / "logs" / "training_steps.csv"
    rows = []
    with steps_path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if row.get("attempt_id") == attempt_id:
                rows.append(row)
    committed_prefix_rows = [row for row in rows if int(row["layer"]) <= 20]
    expected_steps = 21 * 20 * 128
    if len(committed_prefix_rows) != expected_steps:
        raise RuntimeError(f"historical prefix row count is {len(committed_prefix_rows)}, expected {expected_steps}")
    if any(row.get("lwc_grad_finite", "").strip().lower() != "true" for row in committed_prefix_rows):
        raise RuntimeError("historical completed prefix has a nonfinite owned LWC gradient row")
    if any(not math.isfinite(float(row["loss"])) for row in committed_prefix_rows):
        raise RuntimeError("historical completed prefix contains a nonfinite loss")
    uniqueness = {(int(row["layer"]), int(row["epoch_index"]), int(row["sample_index"])) for row in committed_prefix_rows}
    expected_locations = {(layer, epoch, sample) for layer in range(21) for epoch in range(20) for sample in range(128)}
    if uniqueness != expected_locations:
        raise RuntimeError("historical committed-prefix training rows are incomplete or duplicated")

    local_model_manifest = load_json(FROZEN_MODEL_MANIFEST)
    preflight_model_identity = load_json(PREFLIGHT / "MODEL_IDENTITY.json")
    if local_model_manifest.get("status") != "PASS" or preflight_model_identity.get("actual_snapshot_revision") != "7ae557604adf67be50417f59c2c2f167def9a775":
        raise RuntimeError("frozen model identity evidence is not PASS")
    source_lock = load_json(V21_1_M3 / "FORMAL_PROTOCOL_LOCK.json")
    if source_lock.get("model_manifest_sha256") != sha256_file(FROZEN_MODEL_MANIFEST):
        raise RuntimeError("isolated model manifest hash differs from v21.1 checkpoint lock")
    if source_lock.get("ordered_token_hashes_sha256") != load_json(M3 / "FORMAL_PROTOCOL_LOCK.json").get("ordered_token_hashes_sha256"):
        raise RuntimeError("copied calibration token identity differs from the v21.1 frozen lock")

    return {
        "historical_checkpoint_pointer_sha256": sha256_file(source_pointer_path),
        "historical_checkpoint_manifest_sha256": source_manifest_sha,
        "historical_checkpoint_lock_sha256": source_manifest["formal_protocol_lock_sha256"],
        "historical_source_manifest": source_manifest,
        "files": files,
        "state_keys": keys,
        "state_tensor_count": len(state_tensors),
        "state_float_tensor_count": len(state_float_tensors),
        "rng_keys": sorted(rng),
        "historical_training_attempt_id": attempt_id,
        "historical_completed_prefix_step_rows": len(committed_prefix_rows),
        "historical_completed_prefix_locations_unique": len(uniqueness),
        "historical_successful_step_count": expected_steps,
        "historical_success_count_provenance": "inferred from complete v21.1 owned-LWC finite-gradient rows and exact v21.3 native GradScaler semantics; no historical optimizer step hooks or found_inf snapshots were recorded",
        "historical_skipped_step_count": 0,
        "model_revision": preflight_model_identity["actual_snapshot_revision"],
        "config_sha256": sha256_file(CONFIG),
        "calibration_manifest_sha256": calibration_sha,
    }


def update_protocol_and_source_audit() -> None:
    protocol_path = ROOT / "PROTOCOL.json"
    protocol = load_json(protocol_path)
    parent_sha = protocol.get("parent_v21_1_protocol_sha256") or sha256_file(protocol_path)
    protocol.update({
        "protocol_id": "range_w4_omniquant_resume_v21_4",
        "status": "FROZEN_BEFORE_FORMAL_RESUME",
        "parent_v21_1_protocol_sha256": parent_sha,
        "resume_authority": "v21.3 FORMAL_RESUME_PLAN.md; completed-block-only resume from block_20",
        "scope": ["correctness-gate repair", "block21-23 formal completion", "dense dequant export and reload", "frozen WT2 and fixed-budget C4 evaluation", "final M0-M3 comparison"],
        "frozen_training_semantics": {
            "model_revision": "7ae557604adf67be50417f59c2c2f167def9a775",
            "official_omniquant_commit": "feffe8ea87d80f7bb57b6e25e7cff9dc950fcc14",
            "weight_activation": "W4A16",
            "group_size": 128,
            "asymmetric_affine_range": [0, 15],
            "lwc": True,
            "let": False,
            "epochs": 20,
            "samples": 128,
            "tokens_per_sample": 2048,
            "seed": 20261005,
            "scheduled_iterations": 61440,
            "iteration_semantics": "each scheduled sample is consumed once; native GradScaler may skip optimizer steps; no retry or compensating sample",
            "optimizer_scaler_resume": "fresh native AdamW and dynamic GradScaler for each new decoder block, consistent with upstream block boundary semantics",
        },
        "historical_artifacts_mutated": False,
        "outputs_root": str(ROOT),
    })
    write_json(protocol_path, protocol)

    audit_path = M3 / "OMNIQUANT_SOURCE_AUDIT.json"
    audit = load_json(audit_path)
    base = BUNDLE / "omniquant_port" / "upstream"
    port = M3 / "official_port"
    for relative, info in audit["files"].items():
        if sha256_file(base / relative) != info["official_sha256"]:
            raise RuntimeError(f"frozen official source baseline hash mismatch: {relative}")
        current_sha = sha256_file(port / relative)
        if relative in ("quantize/omniquant.py", "utils.py"):
            info["patched_sha256"] = current_sha
            info["byte_identical_to_v20_1"] = False
        elif current_sha != info["official_sha256"]:
            raise RuntimeError(f"unexpected unreviewed upstream source modification: {relative}")
    audit["change_scope"] = [
        "block-boundary prefix replay for resume",
        "resume-aware checkpoint parameter lookup",
        "optional finite-gradient/epoch/block diagnostics callbacks",
        "post-native-GradScaler step diagnostics and AdamW pre/post hooks",
        "forward output finite gate before reconstruction loss",
        "nonfinite loss raises instead of opening pdb",
    ]
    audit["only_algorithm_source_modified"] = "quantize/omniquant.py and utils.py contain resume, observer, and correctness-gate code; quantizer, optimizer update math, loss math, and AMP operation sequence unchanged"
    audit["other_upstream_python_files_byte_identical"] = True
    audit["optimizer_or_loss_math_modified"] = False
    audit["algorithm_semantics_changed"] = False
    audit["formal_gate_semantics"] = "known found_inf is accepted only after native scaler step/update, observed optimizer skip, scale backoff, finite parameters/state/scaler, and unchanged LWC parameters; nonfinite forward output or loss remains fatal"
    write_json(audit_path, audit)

    diffs = []
    for relative in ("quantize/omniquant.py", "utils.py"):
        before = (base / relative).read_text(encoding="utf-8").splitlines(keepends=True)
        after = (port / relative).read_text(encoding="utf-8").splitlines(keepends=True)
        diffs.extend(difflib.unified_diff(before, after,
                                          fromfile=f"official_feffe8ea/{relative}",
                                          tofile=f"v21_4/{relative}"))
        diffs.append("\n")
    (M3 / "OMNIQUANT_PATCH.diff").write_text("".join(diffs), encoding="utf-8", newline="\n")


def write_patch_manifest() -> None:
    path = ROOT / "RUNNER_GATE_PATCH_MANIFEST.csv"
    rows = [
        ["file", "function", "old_gate", "new_gate", "reason", "algorithm_semantics_changed"],
        ["M3/run_formal_m3.py", "step_check", "post-scaler any nonfinite owned gradient is fatal; step success inferred from loop progress", "read native found_inf and actual AdamW pre/post hooks after scaler update; accept only native skip plus backoff, unchanged finite parameters/state/scaler; compare read-only AdamW tensor identity/version signature", "align diagnostics with already-completed official dynamic GradScaler handling and verify skipped steps do not mutate optimizer state", "no"],
        ["M3/run_formal_m3.py", "run_calibration model-loader import", "frozen preflight module was not importable from the isolated runner", "add the isolated LOCAL_PREFLIGHT directory to sys.path before importing the existing exact-revision model loader", "fix pretraining bootstrap import without changing model or training settings", "no"],
        ["M3/run_formal_m3.py", "epoch_cb and final count gate", "nonfinite norm and finite-gradient-count != scheduled-count are fatal", "record norm finiteness; require scheduled = successful steps + scaler skips", "scheduled samples and successful optimizer updates are distinct under native GradScaler", "no"],
        ["M3/official_port/utils.py", "NativeScalerWithGradNormCount.__call__", "native scale/backward/unscale/step/update without exported post-update diagnostics", "same native call sequence plus read-only scale/found_inf/state observation", "prove skip/backoff and state integrity without changing scaler decisions", "no"],
        ["M3/official_port/quantize/omniquant.py", "omniquant", "no explicit forward output finite check; hook count tracked across epoch", "finite forward-output fatal gate; per-sample optimizer hook counter reset; capture optimizer-state signature immediately before native scaler call", "preserve numerical integrity checks and inspect the exact current sample's step/state", "no"],
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        csv.writer(stream).writerows(rows)


def freeze_new_lock_and_migrate_checkpoint(history: dict[str, Any]) -> dict[str, Any]:
    sys.path.insert(0, str(M3 / "official_port"))
    sys.path.insert(0, str(PREFLIGHT))
    spec = importlib.util.spec_from_file_location("v21_4_formal_runner", M3 / "run_formal_m3.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot import v21.4 formal runner")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.write_runtime_environment()
    _, _, _, lock = module.frozen_context()

    checkpoint_dir = CHECKPOINTS / "block_20"
    current_manifest_path = checkpoint_dir / "CHECKPOINT_MANIFEST.json"
    source_manifest = history["historical_source_manifest"]
    original_manifest_path = checkpoint_dir / "CHECKPOINT_MANIFEST_V21_1.json"
    if not original_manifest_path.exists():
        shutil.copy2(current_manifest_path, original_manifest_path)
    elif original_manifest_path.read_bytes() != (V21_1_M3 / "checkpoints" / "block_20" / "CHECKPOINT_MANIFEST.json").read_bytes():
        raise RuntimeError("preserved v21.1 checkpoint manifest no longer matches historical source")
    if current_manifest_path.exists() and current_manifest_path.read_bytes() != original_manifest_path.read_bytes():
        pre_fix_manifest = checkpoint_dir / "CHECKPOINT_MANIFEST_V21_4_PRECOUNT_FIX.json"
        if not pre_fix_manifest.exists():
            shutil.copy2(current_manifest_path, pre_fix_manifest)
    migrated = {
        "status": "COMPLETE_BLOCK_CHECKPOINT",
        **module.checkpoint_context(lock),
        "completed_block_id": 20,
        "completed_layer_count": 21,
        "next_block_id": 21,
        "scheduled_training_iterations": 21 * 20 * 128,
        "successful_optimizer_steps": history["historical_successful_step_count"],
        "gradscaler_skipped_steps": history["historical_skipped_step_count"],
        "update_count_provenance": history["historical_success_count_provenance"],
        "state_file": source_manifest["state_file"],
        "state_file_bytes": history["files"]["state"]["bytes"],
        "state_file_sha256": history["files"]["state"]["sha256"],
        "rng_file": source_manifest["rng_file"],
        "rng_file_bytes": history["files"]["rng"]["bytes"],
        "rng_file_sha256": history["files"]["rng"]["sha256"],
        "committed_at_local": source_manifest["committed_at_local"],
        "source_v21_1_checkpoint_manifest_sha256": history["historical_checkpoint_manifest_sha256"],
        "manifest_migrated_at_local": time.strftime("%Y-%m-%d %H:%M:%S%z"),
    }
    current_before_migration = load_json(current_manifest_path) if current_manifest_path.exists() else {}
    prior_runtime = current_before_migration.get("runtime_environment", {})
    target_runtime = lock["runtime_environment"]
    if any(prior_runtime.get(key) != target_runtime.get(key)
           for key in ("python", "torch", "transformers", "tokenizers", "cuda_runtime")):
        pre_runtime_manifest = checkpoint_dir / "CHECKPOINT_MANIFEST_V21_4_PRE_RUNTIME_FIX.json"
        if current_manifest_path.exists() and not pre_runtime_manifest.exists():
            shutil.copy2(current_manifest_path, pre_runtime_manifest)
    write_json(current_manifest_path, migrated)
    pointer = {
        "checkpoint_dir": "block_20",
        "manifest_sha256": sha256_file(current_manifest_path),
        "completed_block_id": 20,
        "completed_layer_count": 21,
    }
    write_json(CHECKPOINTS / "LATEST.json", pointer)
    return {"lock": lock, "migrated_manifest": migrated, "latest_pointer": pointer, "runner_module": module}


def main() -> None:
    run_status_path = M3 / "RUN_STATUS.json"
    bootstrap_failure = None
    if run_status_path.exists():
        current_status = load_json(run_status_path)
        if (current_status.get("status") == "FAILED_OR_INTERRUPTED"
                and current_status.get("exception_type") == "ModuleNotFoundError"
                and current_status.get("message") == "No module named 'run_local_preflight'"):
            bootstrap_failure = current_status
            preserved_status = M3 / "RUN_STATUS_PREFLIGHT_FAILURE.json"
            if not preserved_status.exists():
                shutil.copy2(run_status_path, preserved_status)
        else:
            raise RuntimeError("v21.4 has a non-bootstrap run status; prepare script will not alter run state")
    if (M3 / "M3_TRAINING_SUMMARY.json").exists():
        raise RuntimeError("v21.4 formal training has started; prepare script will not alter run state")
    if (any(csv_has_data_rows(M3 / "logs" / name) for name in ("training_steps.csv", "training_epochs.csv", "block_health.csv"))
            or csv_has_data_rows(M3 / "AMP_EVENT_LOG.csv")):
        raise RuntimeError("v21.4 training logs exist; prepare script will not alter run state")
    latest_path = CHECKPOINTS / "LATEST.json"
    if latest_path.exists():
        latest = load_json(latest_path)
        manifest = load_json(CHECKPOINTS / latest["checkpoint_dir"] / "CHECKPOINT_MANIFEST.json")
        if latest.get("checkpoint_dir") != "block_20" or manifest.get("completed_block_id") != 20 or manifest.get("scheduled_training_iterations") != 53760:
            raise RuntimeError("v21.4 has progressed beyond the imported block_20 prefix")
    history = check_history()
    update_protocol_and_source_audit()
    write_patch_manifest()
    prepared = freeze_new_lock_and_migrate_checkpoint(history)

    v21_3_evidence = {}
    for name in ("FORMAL_RESUME_PLAN.md", "STOP_GATE_ASSESSMENT.md", "FORMAL_STEP_SEMANTICS.md"):
        path = V21_3 / name
        v21_3_evidence[name] = {"path": str(path.relative_to(REPO)), "sha256": sha256_file(path)}
    validation = {
        "status": "PASS",
        "validation_scope": "isolated v21.4 block_20 resume preconditions; no training step executed",
        "pretraining_bootstrap_failure": bootstrap_failure,
        "historical_source_checkpoint": str((V21_1_M3 / "checkpoints" / "block_20").relative_to(REPO)),
        "historical_source_latest_pointer_sha256": history["historical_checkpoint_pointer_sha256"],
        "historical_source_manifest_sha256": history["historical_checkpoint_manifest_sha256"],
        "historical_source_manifest_preserved_in_v21_4": "M3/checkpoints/block_20/CHECKPOINT_MANIFEST_V21_1.json",
        "isolated_checkpoint_files": history["files"],
        "historical_model_state_keys": history["state_keys"],
        "historical_model_state_tensor_count": history["state_tensor_count"],
        "historical_float_tensor_count": history["state_float_tensor_count"],
        "historical_model_state_all_finite": True,
        "historical_rng_state_keys": history["rng_keys"],
        "rng_state_hash_and_structure_valid": True,
        "model_revision": history["model_revision"],
        "config_sha256": history["config_sha256"],
        "calibration_manifest_sha256": history["calibration_manifest_sha256"],
        "ordered_token_hashes_sha256": prepared["lock"]["ordered_token_hashes_sha256"],
        "source_audit_sha256": sha256_file(M3 / "OMNIQUANT_SOURCE_AUDIT.json"),
        "protocol_lock_sha256": sha256_file(M3 / "FORMAL_PROTOCOL_LOCK.json"),
        "migrated_checkpoint_manifest_sha256": sha256_file(CHECKPOINTS / "block_20" / "CHECKPOINT_MANIFEST.json"),
        "latest_completed_block_id": 20,
        "completed_layer_count": 21,
        "scheduled_iterations": 53760,
        "successful_optimizer_steps": history["historical_successful_step_count"],
        "successful_step_count_is_inferred": True,
        "successful_step_count_provenance": history["historical_success_count_provenance"],
        "gradscaler_skipped_steps": 0,
        "resume_boundary_semantics": {
            "completed_quantized_prefix_replayed": True,
            "saved_rng_restored_at_first_incomplete_block": True,
            "fresh_adamw_for_layer21": True,
            "fresh_native_dynamic_gradscaler_for_layer21": True,
            "optimizer_or_scaler_state_loaded_from_block20": False,
            "reason": "Upstream source creates AdamW and NativeScaler inside each decoder-block iteration; a completed-block checkpoint does not contain optimizer/scaler moments. v21.3 explicitly directs a fresh layer-21 optimizer/scaler and RNG restoration at the boundary.",
        },
        "historical_step_log_audit": {
            "attempt_id": history["historical_training_attempt_id"],
            "complete_prefix_rows": history["historical_completed_prefix_step_rows"],
            "unique_layer_epoch_sample_locations": history["historical_completed_prefix_locations_unique"],
            "all_owned_lwc_gradients_finite": True,
            "all_losses_finite": True,
            "note": "Historical source did not record AdamW hook counts or GradScaler found_inf snapshots; successful-step total remains an evidence-based inference, not a directly observed counter.",
        },
        "v21_3_resume_evidence": v21_3_evidence,
        "migrated_checkpoint_manifest_sha256": sha256_file(CHECKPOINTS / "block_20" / "CHECKPOINT_MANIFEST.json"),
        "latest_pointer_sha256": sha256_file(CHECKPOINTS / "LATEST.json"),
        "validation_time_local": time.strftime("%Y-%m-%d %H:%M:%S%z"),
    }
    write_json(M3 / "RESUME_STATE_VALIDATION.json", validation)
    _, completed, record = prepared["runner_module"].verify_latest_checkpoint(prepared["lock"])
    if completed != 21 or record is None:
        raise RuntimeError("migrated block_20 failed v21.4 native checkpoint validation")
    latest_manifest = load_json(CHECKPOINTS / "block_20" / "CHECKPOINT_MANIFEST.json")
    print(json.dumps({"status": "PASS", "completed_layers": completed,
                      "scheduled_iterations": latest_manifest["scheduled_training_iterations"],
                      "successful_optimizer_steps_inferred": latest_manifest["successful_optimizer_steps"],
                      "skipped_steps": latest_manifest["gradscaler_skipped_steps"]}, indent=2))


if __name__ == "__main__":
    main()
