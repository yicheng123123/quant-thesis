from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[3]


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def code_bundle_digest(run_dir: Path) -> str:
    files = [run_dir / "run_formal.py", run_dir / "run_precheck.py",
             ROOT / "experiments/range_search_w4/results/range_w4_m2_parity_v17_1/native_m2.py",
             ROOT / "experiments/range_search_w4/results/range_w4_m2_runtime_v17_3/optimized_native_m2.py",
             run_dir / "V40_RUN_CONFIG.json", OUT / "CANDIDATE_MANIFEST.json"]
    h = hashlib.sha256()
    for p in files:
        h.update(p.name.encode("utf-8")); h.update(bytes.fromhex(sha(p)))
    return h.hexdigest()


def verify() -> dict:
    lock = json.loads((OUT / "FREEZE_LOCK.json").read_text(encoding="utf-8"))
    for name, expected in lock["sha256"].items():
        if sha(OUT / name) != expected:
            raise RuntimeError(f"frozen artifact hash mismatch: {name}")
    manifest_path = OUT / "CANDIDATE_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if sha(manifest_path) != lock["candidate_manifest_sha256"]:
        raise RuntimeError("candidate manifest hash mismatch")
    if len(manifest["modules"]) != 168:
        raise RuntimeError("candidate manifest does not contain 168 Linears")
    calls = sum(x["candidate_count"] for x in manifest["modules"].values())
    slots = sum(x["row_alpha_score_slots"] for x in manifest["modules"].values())
    rows = sum(x["rows"] for x in manifest["modules"].values())
    if (calls, slots, rows) != (675, 1287040, 304128):
        raise RuntimeError(f"frozen candidate counts mismatch: {(calls, slots, rows)}")
    for name, item in manifest["modules"].items():
        alphas = item["candidate_alphas"]
        ordered = [a for a in manifest["full_grid"] if a in alphas]
        if alphas != ordered or 1.0 not in alphas:
            raise RuntimeError(f"invalid candidate order or missing alpha=1: {name}")
    for source in manifest["source_files"].values():
        path = ROOT / source["relative_path"]
        if sha(path) != source["sha256"] or path.stat().st_size != source["bytes"]:
            raise RuntimeError(f"frozen source artifact changed: {path}")
    implementation = json.loads((OUT / "IMPLEMENTATION_HASHES.json").read_text(encoding="utf-8"))
    if sha(OUT / "IMPLEMENTATION_HASHES.json") != lock["implementation_hashes_sha256"]:
        raise RuntimeError("implementation hashes file changed")
    for item in implementation.get("support_files", {}).values():
        p = ROOT / item["path"]
        if sha(p) != item["sha256"] or p.stat().st_size != item["bytes"]:
            raise RuntimeError(f"frozen support code changed: {p}")
    if sha(OUT / "REFERENCE_AUDIT.json") != lock["reference_audit_sha256"]:
        raise RuntimeError("v18/v28/v39 reference audit changed")
    if sha(OUT / "RUNNER_DIFF_B.txt") != lock["runner_diff_b_sha256"]:
        raise RuntimeError("frozen runner diff changed")
    reference = json.loads((OUT / "REFERENCE_AUDIT.json").read_text(encoding="utf-8"))
    for item in reference["sources"].values():
        p = ROOT / item["path"]
        if sha(p) != item["sha256"] or p.stat().st_size != item["bytes"]:
            raise RuntimeError(f"referenced historical evidence changed: {p}")
    for run_name, item in implementation["runs"].items():
        run_dir = OUT / run_name
        for key in ("run_formal", "run_precheck", "run_config", "protocol_copy", "calibration_copy"):
            p = ROOT / item[key]["path"]
            if sha(p) != item[key]["sha256"]:
                raise RuntimeError(f"frozen implementation changed: {p}")
        if code_bundle_digest(run_dir) != item["code_bundle_sha256"]:
            raise RuntimeError(f"runner code bundle changed: {run_name}")
    job_rows = list(csv.DictReader((OUT / "JOB_TABLE.csv").open(encoding="utf-8-sig", newline="")))
    for phase, expected_calls, expected_slots in (("PILOT", 27, 49792), ("A", 1680, 3041280), ("B", 675, 1287040)):
        rows_for_phase = [r for r in job_rows if r["phase"] == phase and r["module"] != "AUX_ALPHA1_NATIVE_PARITY"]
        actual = (sum(int(r["candidate_gptq_executions"]) for r in rows_for_phase),
                  sum(int(r["row_alpha_score_slots"]) for r in rows_for_phase))
        if actual != (expected_calls, expected_slots):
            raise RuntimeError(f"JOB_TABLE mismatch for {phase}: {actual}")
    return {"status": "PASS", "linears": 168, "rows": rows,
            "full_grid_candidate_executions": 1680, "winner_union_candidate_executions": calls,
            "winner_union_row_alpha_score_slots": slots,
            "implementation_sha256": sha(OUT / "IMPLEMENTATION_HASHES.json"),
            "candidate_manifest_sha256": sha(manifest_path)}


if __name__ == "__main__":
    result = verify()
    print(json.dumps(result, indent=2))
