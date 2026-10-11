from __future__ import annotations

import importlib.util
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import torch


BASE = Path(__file__).resolve().parent
os.environ["V28_RUN_LABEL"] = "validation"
os.environ["V28_SCORE_MODE"] = "dual"
spec = importlib.util.spec_from_file_location("v28_runner_recovery", BASE / "run_v28.py")
runner = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(runner)

manifest_path = BASE / "validation" / "RUN_MANIFEST.json"
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
old_code_hash = manifest["code_bundle_sha256"]
expected_old_hash = "9553008ccc94438083ce218d8648a98ed088cdc018be43971bfec8f485f1135b"
if old_code_hash != expected_old_hash:
    raise RuntimeError(f"unexpected checkpoint code hash {old_code_hash}")
new_code_hash = runner.hash_code_bundle()
if new_code_hash == old_code_hash:
    raise RuntimeError("expected the post-startup budget-ledger source revision")

# The only source delta after the formal process loaded was the cumulative
# budget ledger. Retain the current source hash in the recovered RUN_REPORT,
# while validating immutable quantization checkpoints against their exact
# recorded source bundle and all existing chain hashes.
def validate_legacy_completed_layers(method: str, code_hash: str, cal_hash: str, protocol_hash: str):
    paths = [runner.layer_state_path(method, i) for i in range(runner.LAYERS)]
    completed, payloads, missing_seen = 0, [], False
    previous_state_hash = "0" * 64
    for index, path in enumerate(paths):
        if not path.exists():
            missing_seen = True
            continue
        if missing_seen:
            raise RuntimeError(f"non-contiguous checkpoint sequence: {path.name}")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        meta = payload["metadata"]
        expected = {"method": method, "layer": index,
                    "calibration_manifest_sha256": cal_hash,
                    "protocol_sha256": protocol_hash}
        for key, value in expected.items():
            if meta.get(key) != value:
                raise RuntimeError(f"checkpoint metadata mismatch {path.name}: {key}")
        if meta.get("code_bundle_sha256") != old_code_hash:
            raise RuntimeError(f"unexpected historical code bundle in {path.name}")
        for module_name, state in payload["modules"].items():
            if runner.module_state_hash(state) != state.get("state_sha256"):
                raise RuntimeError(f"module hash mismatch {path.name}: {module_name}")
        layer_hash = runner.quantized_layer_hash(payload["modules"])
        if meta.get("previous_model_state_sha256") != previous_state_hash:
            raise RuntimeError(f"previous-state chain mismatch {path.name}")
        if meta.get("layer_quant_state_sha256") != layer_hash:
            raise RuntimeError(f"layer hash mismatch {path.name}")
        if meta.get("selected_alpha_table_sha256") != runner.alpha_table_hash(payload["modules"]):
            raise RuntimeError(f"alpha table hash mismatch {path.name}")
        state_hash = runner.sha_bytes((previous_state_hash + layer_hash).encode("ascii"))
        if meta.get("model_state_sha256") != state_hash:
            raise RuntimeError(f"model-state chain mismatch {path.name}")
        previous_state_hash = state_hash
        payloads.append(payload)
        completed += 1
    return completed, payloads


runner.list_completed_layers = validate_legacy_completed_layers
original_replay = runner.replay_completed_prefix


def skip_replay_only_when_fully_complete(model, method, token_samples, completed, rec):
    if completed == runner.LAYERS:
        # All quantized layer weights are restored from the complete verified
        # chain below; layer 24 is not a valid replay target.
        return None, None, None, None
    return original_replay(model, method, token_samples, completed, rec)


runner.replay_completed_prefix = skip_replay_only_when_fully_complete
original_budget_check = runner.check_gpu_budget


def guarded_budget_check(stage, reserve=None):
    # Preserve a 120s finalization reserve under the same hard 5,400s cap.
    return original_budget_check(stage, reserve=120.0 if reserve is None else reserve)


runner.check_gpu_budget = guarded_budget_check

run_dir = BASE / "validation"
(run_dir / "M2").mkdir(parents=True, exist_ok=True)
budget_path = BASE / "BUDGET.json"
if not budget_path.exists():
    elapsed = float(manifest.get("gpu_wall_seconds", 0.0))
    budget_path.write_text(json.dumps({
        "max_seconds": runner.GPU_BUDGET_LIMIT,
        "gpu_wall_elapsed_seconds": elapsed,
        "last_recorded_stage": "dual_validation_complete_before_finalization_recovery",
        "run_label": "validation", "score_mode": "dual",
        "updated_utc": datetime.now(timezone.utc).isoformat(),
    }, indent=2) + "\n", encoding="utf-8")

audit = {
    "event": "resume_finalization_without_candidate_rerun",
    "reason": "first complete pass wrote QUANTIZATION_CORRECTNESS.json before creating validation/M2",
    "failed_formal_status": manifest.get("status"),
    "all_layers_checkpointed": manifest.get("last_complete_group", {}).get("layer") == 23
        and manifest.get("last_complete_group", {}).get("group") == "down",
    "checkpoint_code_bundle_sha256": old_code_hash,
    "current_recovery_code_bundle_sha256": new_code_hash,
    "recovery_compatibility": "accept only the exact old hash stored in every checkpoint, while verifying calibration/protocol, per-module hashes, alpha table, and full sequential model-state chain",
    "candidate_rerun": False,
    "activation_replay": "not repeated during output finalization; all 24 layers had completed during the original pass, and their per-boundary hashes are audited against v18",
    "budget_start_seconds": float(manifest.get("gpu_wall_seconds", 0.0)),
    "budget_limit_seconds": runner.GPU_BUDGET_LIMIT,
    "recovery_finalization_reserve_seconds": 120,
    "updated_utc": datetime.now(timezone.utc).isoformat(),
}
(BASE / "validation" / "RECOVERY_AUDIT.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
runner.main()
