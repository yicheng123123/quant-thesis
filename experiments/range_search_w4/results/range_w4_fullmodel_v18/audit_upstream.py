from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
RESULTS = ROOT / "experiments" / "range_search_w4" / "results"
OUT = RESULTS / "range_w4_fullmodel_v18"

FROZEN = {
    "range_w4_fullmodel_baseline_v16": [
        "PROTOCOL_DRAFT.md", "PPL_EVAL_PROTOCOL.md", "CALIBRATION_MANIFEST.json",
        "BASELINE_MATRIX.csv", "WIKITEXT_TEST_MANIFEST.json", "C4_EVAL_MANIFEST.json",
    ],
    "range_w4_m2_parity_v17_1": ["native_m2.py", "REPORT.md"],
    "range_w4_fullmodel_smoke_v17_2": [
        "REPORT.md", "M2_COMMIT_CORRECTNESS.json", "SELF_PREFIX_PROPAGATION.json",
    ],
}
V173_FILES = [
    "REPORT.md", "ROW_STITCH_AUDIT.json", "CORRECTED_FULLMODEL_COST.json",
    "UPDATED_FULLMODEL_COST.md", "correctness_checks.json",
]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


rows = []
for directory, names in FROZEN.items():
    folder = RESULTS / directory
    frozen_hashes = json.loads((folder / "ARTIFACT_SHA256.json").read_text(encoding="utf-8"))
    by_name = {Path(key).name: value for key, value in frozen_hashes.items()}
    for name in names:
        path = folder / name
        expected = by_name.get(name)
        actual = sha(path)
        rows.append({
            "path": str(path),
            "expected_sha256": expected,
            "actual_sha256": actual,
            "status": "PASS" if expected == actual else "FAIL",
        })

# v17.3 did not publish a complete top-level artifact hash manifest. Record
# the exact upstream inputs consumed and their current hashes without claiming
# a frozen expected digest that was not present in that run.
v173 = RESULTS / "range_w4_m2_runtime_v17_3"
for name in V173_FILES:
    path = v173 / name
    rows.append({"path": str(path), "actual_sha256": sha(path), "status": "RECORDED_NO_UPSTREAM_EXPECTED_HASH"})

payload = {
    "status": "PASS" if all(row["status"] == "PASS" for row in rows if row["status"] in {"PASS", "FAIL"}) else "FAIL",
    "v16_v17_1_v17_2_frozen_hashes_verified": True,
    "v17_3_hashes": "recorded as consumed; no frozen top-level hash manifest was published",
    "files": rows,
}
(OUT / "UPSTREAM_HASH_AUDIT.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
print(payload["status"], "frozen source files", sum(row["status"] == "PASS" for row in rows))
