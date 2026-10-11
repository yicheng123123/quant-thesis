#!/usr/bin/env python3
"""Verify the frozen WikiText-2 calibration bytes and per-document token hashes."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


EXPECTED = "664419273f63df3c0b0746b18bb66b593a5384bbe6ba8afcd6f4905c9479aae3"


def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path,
                        default=Path(__file__).resolve().parent / "calibration/CALIBRATION_MANIFEST.json")
    parser.add_argument("--output", type=Path, default=Path("KAGGLE_CALIBRATION_CHECK.json"))
    args = parser.parse_args()
    path = args.manifest.resolve()
    manifest = json.loads(path.read_text(encoding="utf-8"))
    raw_sha = hashlib.sha256(path.read_bytes()).hexdigest()
    errors = []
    expected_fields = {"nsamples": 128, "seqlen": 2048, "seed": 20261005, "split": "train"}
    for field, expected in expected_fields.items():
        if manifest.get(field) != expected:
            errors.append(f"{field} mismatch")
    if raw_sha != EXPECTED:
        errors.append("manifest SHA256 mismatch")
    samples = manifest.get("samples", [])
    if len(samples) != 128:
        errors.append(f"sample count mismatch: {len(samples)}")
    rows = []
    for index, sample in enumerate(samples):
        ids = sample.get("token_ids", [])
        token_sha = sha_bytes(json.dumps(ids).encode("utf-8"))
        if len(ids) != 2048:
            errors.append(f"sample {index} sequence length mismatch")
        if token_sha != sample.get("token_sha256"):
            errors.append(f"sample {index} token hash mismatch")
        rows.append({"sample_index": index, "source_article_index": sample.get("source_article_index"),
                     "article_sha256": sample.get("article_sha256"), "token_count": len(ids),
                     "token_sha256": token_sha})
    result = {
        "status": "PASS" if not errors else "FAIL", "manifest_sha256": raw_sha,
        "sample_count": len(samples), "tokens_per_sample": sorted({row["token_count"] for row in rows}),
        "seed": manifest.get("seed"), "split": manifest.get("split"),
        "ordered_sample_metadata_sha256": sha_bytes(json.dumps(rows, separators=(",", ":"), sort_keys=True).encode()),
        "per_sample_hashes_valid": not errors, "errors": errors,
        "runtime_context": "Kaggle" if "/kaggle/" in str(args.output.resolve()).lower() else "local package verification",
        "resampling": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
