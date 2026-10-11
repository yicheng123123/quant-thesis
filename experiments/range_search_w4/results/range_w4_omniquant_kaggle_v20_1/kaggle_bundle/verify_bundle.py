#!/usr/bin/env python3
"""Verify the frozen source, v20 port, protocol, calibration, and bundle files."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle-root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--output", type=Path, default=Path("BUNDLE_VERIFY.json"))
    args = parser.parse_args()
    root = args.bundle_root.resolve()
    checks: dict[str, object] = {}
    failures: list[str] = []

    def check(name: str, passed: bool, details: object) -> None:
        checks[name] = {"pass": bool(passed), "details": details}
        if not passed:
            failures.append(name)

    required = [
        "README_KAGGLE.md", "requirements_kaggle.txt", "environment_check.py",
        "run_preflight.py", "run_layer_timing.py", "audit_step_count.py", "audit_calibration.py",
        "M3_RESUME_PLAN.md", "KAGGLE_COMMANDS.md", "configs/m3_preflight.json",
        "calibration/CALIBRATION_MANIFEST.json", "manifests/OFFICIAL_SOURCE_SHA256.json",
        "manifests/V20_UPSTREAM_MANIFEST.json", "manifests/OMNIQUANT_PATCH_MANIFEST.csv",
        "manifests/MODEL_FILE_HASHES.json", "manifests/pristine_quantize_omniquant.py",
        "omniquant_port/upstream/quantize/omniquant.py",
        "omniquant_port/upstream/models/int_qwen2_layer.py",
        "omniquant_port/upstream/quantize/quantizer.py",
        "omniquant_port/upstream/quantize/int_linear.py",
        "omniquant_port/upstream/quantize/utils.py", "omniquant_port/upstream/utils.py",
        "evaluator/ppl.py", "evaluator/ppl_datasets.py", "hashes/EXPECTED_FILES.json",
    ]
    missing = [rel for rel in required if not (root / rel).is_file()]
    check("required_files", not missing, {"missing": missing, "count": len(required)})
    if missing:
        result = {"status": "FAIL", "checks": checks, "failures": failures}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(result, indent=2))
        return 2

    protocol = json.loads((root / "configs/m3_preflight.json").read_text(encoding="utf-8"))
    lock = json.loads((root / "hashes/EXPECTED_FILES.json").read_text(encoding="utf-8"))
    v20 = json.loads((root / "manifests/V20_UPSTREAM_MANIFEST.json").read_text(encoding="utf-8"))
    official = json.loads((root / "manifests/OFFICIAL_SOURCE_SHA256.json").read_text(encoding="utf-8"))
    model_files = json.loads((root / "manifests/MODEL_FILE_HASHES.json").read_text(encoding="utf-8"))
    calibration_path = root / "calibration/CALIBRATION_MANIFEST.json"
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    check("upstream_commit", protocol["upstream_commit"] == "feffe8ea87d80f7bb57b6e25e7cff9dc950fcc14"
          and v20.get("commit") == protocol["upstream_commit"] and v20.get("commit_pass") is True,
          {"config": protocol["upstream_commit"], "v20": v20.get("commit")})
    check("model_revision", protocol["model_revision"] == "7ae557604adf67be50417f59c2c2f167def9a775"
          and lock.get("model_revision") == protocol["model_revision"],
          {"model_id": protocol["model_id"], "revision": protocol["model_revision"]})
    check("model_file_lock", model_files.get("model_id") == protocol["model_id"]
          and model_files.get("revision") == protocol["model_revision"]
          and "model.safetensors" in model_files.get("files", {}),
          {"model_id": model_files.get("model_id"), "revision": model_files.get("revision"),
           "locked_files": len(model_files.get("files", {}))})
    check("calibration_manifest", sha(calibration_path) == protocol["formal_training"]["calibration_manifest_sha256"]
          and lock.get("calibration_manifest_sha256") == sha(calibration_path)
          and calibration.get("nsamples") == 128 and calibration.get("seqlen") == 2048
          and calibration.get("seed") == 20261005 and len(calibration.get("samples", [])) == 128,
          {"sha256": sha(calibration_path), "samples": len(calibration.get("samples", [])),
           "seqlen": calibration.get("seqlen"), "seed": calibration.get("seed")})

    # Validate every pristine upstream source file. The patched v20 omniquant.py is
    # checked against its v20 hash; its pristine original is included separately.
    pristine_override = {"quantize/omniquant.py": "manifests/pristine_quantize_omniquant.py"}
    source_rows = []
    source_tree = hashlib.sha256()
    for rel, expected_hash in sorted(official.items()):
        bundled_rel = pristine_override.get(rel, f"omniquant_port/upstream/{rel}")
        file_path = root / bundled_rel
        actual = sha(file_path) if file_path.is_file() else None
        source_rows.append({"path": rel, "bundled_path": bundled_rel, "pass": actual == expected_hash})
        if actual == expected_hash:
            source_tree.update(rel.replace("\\", "/").encode("utf-8") + b"\0" + bytes.fromhex(actual))
    source_tree_hash = source_tree.hexdigest()
    check("official_upstream_files", len(official) == 257 and all(row["pass"] for row in source_rows)
          and source_tree_hash == v20.get("source_tree_sha256")
          and v20.get("frozen_source_manifest_sha256") == sha(root / "manifests/OFFICIAL_SOURCE_SHA256.json"),
          {"files": len(source_rows), "mismatches": [r["path"] for r in source_rows if not r["pass"]],
           "tree_sha256": source_tree_hash, "expected_tree_sha256": v20.get("source_tree_sha256"),
           "manifest_sha256": sha(root / "manifests/OFFICIAL_SOURCE_SHA256.json"),
           "manifest_expected_sha256": v20.get("frozen_source_manifest_sha256")})

    patch_files = {
        "omniquant_port/upstream/quantize/omniquant.py": "v20_original_hash/quantize/omniquant.py",
        "omniquant_port/upstream/models/int_qwen2_layer.py": "v20_original_hash/models/int_qwen2_layer.py",
    }
    patch_check = {}
    patch_lock = json.loads((root / "hashes/V20_PATCH_HASHES.json").read_text(encoding="utf-8"))
    for actual_rel, expected_key in patch_files.items():
        actual = sha(root / actual_rel)
        expected_hash = patch_lock[expected_key]
        patch_check[actual_rel] = {"actual_sha256": actual, "expected_sha256": expected_hash, "pass": actual == expected_hash}
    check("v20_patch_hashes", all(row["pass"] for row in patch_check.values()), patch_check)

    protected = v20.get("protected_algorithm_files", {})
    protected_check = {}
    for rel, metadata in protected.items():
        path = root / "omniquant_port/upstream" / rel
        actual = sha(path)
        protected_check[rel] = {"actual_sha256": actual, "expected_sha256": metadata["official"],
                                "pass": actual == metadata["official"] and metadata.get("identical") is True}
    check("protected_algorithm_sources", all(x["pass"] for x in protected_check.values()), protected_check)

    patch_manifest_path = root / "manifests/OMNIQUANT_PATCH_MANIFEST.csv"
    with patch_manifest_path.open(newline="", encoding="utf-8") as stream:
        patch_rows = list(csv.DictReader(stream))
    check("no_algorithm_semantic_patch", bool(patch_rows)
          and all(row.get("algorithm_semantic_change", "").strip().lower() == "no" for row in patch_rows),
          {"rows": len(patch_rows), "semantic_changes": [r["file"] for r in patch_rows
            if r.get("algorithm_semantic_change", "").strip().lower() != "no"]})

    locked_failures = []
    for rel, expected_hash in lock.get("files", {}).items():
        path = root / rel
        if not path.is_file() or sha(path) != expected_hash:
            locked_failures.append(rel)
    check("bundle_file_hashes", not locked_failures,
          {"file_count": len(lock.get("files", {})), "mismatches": locked_failures,
           "bundle_tree_sha256": lock.get("bundle_tree_sha256")})
    actual_files = {path.relative_to(root).as_posix() for path in root.rglob("*")
                    if path.is_file() and "__pycache__" not in path.parts}
    unexpected = sorted(actual_files - set(lock.get("files", {})) - {"hashes/EXPECTED_FILES.json"})
    check("no_unfrozen_bundle_members", not unexpected, {"unexpected_files": unexpected})
    evaluator_hashes = {rel: sha(root / rel) for rel in ("evaluator/ppl.py", "evaluator/ppl_datasets.py")}
    check("evaluator_hashes_recorded", all(rel in lock.get("files", {}) and lock["files"][rel] == digest
                                             for rel, digest in evaluator_hashes.items()), evaluator_hashes)

    result = {"status": "PASS" if not failures else "FAIL", "bundle_root": str(root),
              "checks": checks, "failures": failures}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
