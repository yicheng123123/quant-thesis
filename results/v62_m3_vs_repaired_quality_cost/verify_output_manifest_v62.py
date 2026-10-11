import csv, hashlib
from pathlib import Path

root = Path(__file__).resolve().parent
rows = list(csv.DictReader((root / "OUTPUT_MANIFEST.csv").open(encoding="utf-8-sig", newline="")))
bad = []
checked = 0
for row in rows:
    if row["self_hash"] == "True":
        continue
    path = root / row["path"]
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    checked += 1
    if h.hexdigest() != row["sha256"] or path.stat().st_size != int(row["bytes"]):
        bad.append(row["path"])
print({"files_checked": checked, "bad_count": len(bad), "bad_paths": bad[:10], "manifest_rows": len(rows)})
if bad:
    raise SystemExit(1)
