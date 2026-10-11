#!/usr/bin/env python3
"""Freeze every bundle input by relative path and SHA-256."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
LOCK = ROOT / "hashes" / "EXPECTED_FILES.json"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    protocol = json.loads((ROOT / "configs/m3_preflight.json").read_text(encoding="utf-8"))
    source_manifest = json.loads((ROOT / "manifests/OFFICIAL_SOURCE_SHA256.json").read_text(encoding="utf-8"))
    expected_files = {}
    for path in sorted(ROOT.rglob("*")):
        if not path.is_file() or path == LOCK or "__pycache__" in path.parts:
            continue
        expected_files[path.relative_to(ROOT).as_posix()] = sha(path)
    tree = hashlib.sha256()
    for relative, digest in expected_files.items():
        tree.update(relative.encode("utf-8") + b"\0" + bytes.fromhex(digest))
    frozen = {
        "format": "v20.1-kaggle-bundle-lock-v1",
        "bundle_tree_sha256": tree.hexdigest(),
        "file_count_excluding_lock": len(expected_files),
        "upstream_commit": protocol["upstream_commit"],
        "official_upstream_manifest_file_count": len(source_manifest),
        "model_id": protocol["model_id"],
        "model_revision": protocol["model_revision"],
        "calibration_manifest_sha256": protocol["formal_training"]["calibration_manifest_sha256"],
        "files": expected_files,
    }
    LOCK.write_text(json.dumps(frozen, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in frozen.items() if k != "files"}, indent=2))


if __name__ == "__main__":
    main()
