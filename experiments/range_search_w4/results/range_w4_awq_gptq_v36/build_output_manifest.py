"""Write a SHA-256 inventory for all v36 deliverables and resumable artifacts."""
from __future__ import annotations

import csv
import hashlib
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "OUTPUT_MANIFEST.csv"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


files = sorted(p for p in HERE.rglob("*") if p.is_file() and "__pycache__" not in p.parts
               and p.suffix != ".tmp" and p != OUT)
with OUT.open("w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["relative_path", "size_bytes", "sha256"])
    for path in files:
        writer.writerow([path.relative_to(HERE).as_posix(), path.stat().st_size, digest(path)])
print(f"listed={len(files)} manifest={OUT}")
