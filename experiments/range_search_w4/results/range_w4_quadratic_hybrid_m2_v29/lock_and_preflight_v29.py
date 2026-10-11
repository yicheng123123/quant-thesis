"""Lock v29 implementation after passing the frozen L7 full-Linear spot gate."""
from __future__ import annotations

import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
V28 = ROOT / "experiments/range_search_w4/results/range_w4_quadratic_full_m2_v28"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    protocol = json.loads((HERE / "PROTOCOL.json").read_text("utf-8"))
    manifest_path = HERE / "RUN_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text("utf-8"))
    if manifest.get("protocol_sha256") != sha(HERE / "PROTOCOL.json"):
        raise RuntimeError("frozen protocol hash changed")
    spot = json.loads((HERE / "spot" / "SPOT_RECHECK.json").read_text("utf-8"))
    if spot.get("status") != "PASS" or not all(spot.get("checks", {}).values()):
        raise RuntimeError("known-difference full-Linear spot gate has not passed")
    audit = json.loads((HERE / "FROZEN_TRIGGER_AUDIT.json").read_text("utf-8"))
    if not audit.get("row_884_covered") or audit.get("v28_sensitive_rows") != 201 or audit.get("v28_triggered_full_linears") != 82:
        raise RuntimeError("frozen trigger audit does not match the recorded v28 result")
    if (HERE / "validation" / "FORMAL_RUNTIME.json").exists():
        raise RuntimeError("formal v29 run already has runtime artifacts; do not relock")
    expected_v28 = protocol["source_sha256"]
    v18 = ROOT / "experiments/range_search_w4/results/range_w4_fullmodel_v18"
    audit28 = V28 / "validation" / "audit"
    evidence_paths = {
        "v18_PROTOCOL.json": v18 / "PROTOCOL.json",
        "v18_run_precheck.py": v18 / "run_precheck.py",
        "v18_run_formal.py": v18 / "run_formal.py",
        "v28_PROTOCOL.json": V28 / "PROTOCOL.json",
        "v28_run_v28.py": V28 / "run_v28.py",
        "v28_sensitive_rows.csv": audit28 / "SENSITIVE_ROWS.csv",
        "v28_known_difference.json": audit28 / "NUMERICAL_SELECTION_DIFFERENCE.json",
    }
    for rel, expected in expected_v28.items():
        if rel == "v28_raw_scores_layer_00_23":
            for i in range(24):
                path = audit28 / "raw_scores" / f"layer_{i:02d}_scores.npz"
                if not path.is_file() or sha(path) != expected[f"layer_{i:02d}"]:
                    raise RuntimeError(f"frozen v28 evidence changed: {path.name}")
            continue
        path = evidence_paths[rel]
        if not path.is_file() or sha(path) != expected:
            raise RuntimeError(f"frozen v28 evidence changed: {rel}")
    code = HERE / "run_v29.py"
    spot_code = HERE / "spot_recheck_v29.py"
    freeze_code = HERE / "freeze_v29.py"
    runner_paths = [code, ROOT / "experiments/range_search_w4/results/range_w4_fullmodel_v18/run_precheck.py",
                    ROOT / "experiments/range_search_w4/results/range_w4_m2_parity_v17_1/native_m2.py",
                    ROOT / "experiments/range_search_w4/results/range_w4_m2_runtime_v17_3/optimized_native_m2.py"]
    code_hash = hashlib.sha256()
    for path in runner_paths:
        code_hash.update(path.name.encode("utf-8"))
        code_hash.update(bytes.fromhex(sha(path)))
    device = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    if device != "NVIDIA GeForce RTX 4060 Laptop GPU":
        raise RuntimeError(f"frozen CUDA device unavailable: {device}")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if torch.backends.cuda.matmul.allow_tf32 or torch.backends.cudnn.allow_tf32:
        raise RuntimeError("TF32 must remain disabled")
    budget = json.loads((HERE / "BUDGET.json").read_text("utf-8"))
    if budget["gpu_wall_elapsed_seconds"] >= protocol["validation"]["gpu_wall_budget_seconds"]:
        raise RuntimeError("v29 cumulative budget is already exhausted")
    lock = {
        "status": "IMPLEMENTATION_LOCKED_BEFORE_FULL_VALIDATION",
        "protocol_sha256": sha(HERE / "PROTOCOL.json"),
        "v29_runner_code_bundle_sha256": code_hash.hexdigest(),
        "implementation_sha256": {
            "run_v29.py": sha(code), "spot_recheck_v29.py": sha(spot_code),
            "freeze_v29.py": sha(freeze_code), "lock_and_preflight_v29.py": sha(Path(__file__)),
        },
        "spot_recheck_sha256": sha(HERE / "spot" / "SPOT_RECHECK.json"),
        "frozen_v28_source_hashes_reverified": True,
        "frozen_trigger_audit_sha256": sha(HERE / "FROZEN_TRIGGER_AUDIT.json"),
        "preflight": {
            "cuda_available": bool(torch.cuda.is_available()), "device": device,
            "tf32_matmul": bool(torch.backends.cuda.matmul.allow_tf32),
            "tf32_cudnn": bool(torch.backends.cudnn.allow_tf32),
            "frozen_rule_covers_row_884": True,
            "spot_checks_all_pass": True,
            "formal_validation_output_absent": True,
        },
        "cumulative_budget_before_full_validation_seconds": budget["gpu_wall_elapsed_seconds"],
        "locked_utc": datetime.now(timezone.utc).isoformat(),
    }
    (HERE / "IMPLEMENTATION_LOCK.json").write_text(json.dumps(lock, ensure_ascii=False, indent=2) + "\n", "utf-8")
    manifest.update({"status": "IMPLEMENTATION_LOCKED_READY_FOR_FULL_VALIDATION",
                     "implementation_lock_sha256": sha(HERE / "IMPLEMENTATION_LOCK.json"),
                     "v29_runner_code_bundle_sha256": code_hash.hexdigest(),
                     "spot_recheck_sha256": lock["spot_recheck_sha256"],
                     "updated_utc": lock["locked_utc"]})
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", "utf-8")
    print(json.dumps(lock, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
