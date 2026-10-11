from __future__ import annotations

import difflib
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[3]
RESULTS = ROOT / "experiments/range_search_w4/results"
V18 = RESULTS / "range_w4_fullmodel_v18"
V28 = RESULTS / "range_w4_quadratic_full_m2_v28"
V39 = RESULTS / "range_w4_robust_hessian_precheck_v39"
PRECHECK = RESULTS / "range_w4_candidate_budget_precheck"
V171 = RESULTS / "range_w4_m2_parity_v17_1"
V173 = RESULTS / "range_w4_m2_runtime_v17_3"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def json_read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def source_record(path: Path) -> dict:
    return {"path": str(path.relative_to(ROOT)), "sha256": sha(path), "bytes": path.stat().st_size}


def code_bundle_digest(run_dir: Path) -> str:
    files = [run_dir / "run_formal.py", run_dir / "run_precheck.py", V171 / "native_m2.py",
             V173 / "optimized_native_m2.py", run_dir / "V40_RUN_CONFIG.json", OUT / "CANDIDATE_MANIFEST.json"]
    h = hashlib.sha256()
    for path in files:
        h.update(path.name.encode("utf-8"))
        h.update(bytes.fromhex(sha(path)))
    return h.hexdigest()


def main() -> None:
    lock_path = OUT / "FREEZE_LOCK.json"
    lock = json_read(lock_path)
    if lock.get("status") != "FROZEN" or lock.get("gpu_run_started") is True:
        raise RuntimeError("v40 protocol lock is not in the pre-execution state")
    for name, digest in lock["sha256"].items():
        if sha(OUT / name) != digest:
            raise RuntimeError(f"frozen file changed before code seal: {name}")
    source_files = [V28 / "validation/correctness_checks.json",
                    V28 / "validation/audit/CONSISTENCY_SUMMARY.json",
                    V28 / "validation/audit/FINAL_SUMMARY.json",
                    V39 / "REPORT.md", V39 / "DECISION.md", V39 / "SUMMARY.json",
                    V18 / "FORMAL_RUNTIME.json", PRECHECK / "RUN_SUMMARY.json"]
    source_records = {str(p.relative_to(ROOT)): source_record(p) for p in source_files}
    v28_checks = json_read(V28 / "validation/correctness_checks.json")
    v28_consistency = json_read(V28 / "validation/audit/CONSISTENCY_SUMMARY.json")
    v28_final = json_read(V28 / "validation/audit/FINAL_SUMMARY.json")
    v39_summary = json_read(V39 / "SUMMARY.json")
    precheck_summary = json_read(PRECHECK / "RUN_SUMMARY.json")
    reference = {
        "schema": "v40-reference-audit-v1",
        "recorded_utc": datetime.now(timezone.utc).isoformat(),
        "v18": {"m2_status": json_read(V18 / "FORMAL_RUNTIME.json")["m2"]["status"],
                "m2_wall_seconds": json_read(V18 / "FORMAL_RUNTIME.json")["m2"]["wall_seconds_including_load_replay_save"],
                "candidate_gptq_executions": 1680, "row_alpha_score_slots": 3041280,
                "row_results_sha256": sha(V18 / "M2_ROW_RESULTS.csv")},
        "v28": {"direct_path_against_v18": v28_final["direct_path_vs_v18"],
                "direct_runtime_correctness": v28_checks,
                "quadratic_scorer": v28_consistency,
                "interpretation": "The original direct path matched v18; the quadratic scorer had one row selection difference. v40 uses the original direct scorer."},
        "v39": {"summary": v39_summary,
                "report_sha256": sha(V39 / "REPORT.md"), "decision_sha256": sha(V39 / "DECISION.md"),
                "p3_read_by_v39": False,
                "scope": "Robust-Hessian mathematical precheck only; no new candidates or GPU quality experiment."},
        "candidate_budget_precheck": {"summary": precheck_summary,
                "gpu_initializations": 0, "model_forwards": 0, "gptq_runs": 0,
                "new_held_scoring": 0, "p3_read": False},
        "sources": source_records,
    }
    ref_path = OUT / "REFERENCE_AUDIT.json"
    write(ref_path, reference)

    source_code = (V18 / "run_formal.py").read_text(encoding="utf-8").splitlines(keepends=True)
    run_code = (OUT / "B/run_formal.py").read_text(encoding="utf-8").splitlines(keepends=True)
    diff = list(difflib.unified_diff(source_code, run_code, fromfile="v18/run_formal.py", tofile="v40/B/run_formal.py", n=2))
    (OUT / "RUNNER_DIFF_B.txt").write_text("".join(diff), encoding="utf-8")
    direct_score = 'losses = rp.score_rows(x_mm, pristine, q)' in (OUT / "B/run_formal.py").read_text(encoding="utf-8")
    native_candidate = 'optimized_dynamic_gptq(pristine, prepared, alpha=alpha, check_cholesky=(alpha == 1.0))' in (OUT / "B/run_formal.py").read_text(encoding="utf-8")
    if not direct_score or not native_candidate:
        raise RuntimeError("patched runner changed the frozen candidate GPTQ/direct scoring calls")

    impl = {}
    for run_name in ("PILOT", "A", "B"):
        run_dir = OUT / run_name
        impl[run_name] = {
            "run_formal": source_record(run_dir / "run_formal.py"),
            "run_precheck": source_record(run_dir / "run_precheck.py"),
            "run_config": source_record(run_dir / "V40_RUN_CONFIG.json"),
            "protocol_copy": source_record(run_dir / "PROTOCOL.json"),
            "calibration_copy": source_record(run_dir / "CALIBRATION_MANIFEST.json"),
            "code_bundle_sha256": code_bundle_digest(run_dir),
        }
    implementation = {
        "schema": "v40-implementation-freeze-v1",
        "sealed_utc": datetime.now(timezone.utc).isoformat(),
        "builder": source_record(OUT / "build_v40.py"),
        "sealer": source_record(Path(__file__)),
        "candidate_generation_call_unchanged": native_candidate,
        "direct_row_scoring_call_unchanged": direct_score,
        "candidate_specific_runtime_changes": [
            "A iterates the unchanged v18 scalar10 grid; B and pilot iterate each frozen per-Linear winner union plus alpha=1 in original order.",
            "Validation and progress counters use the per-module candidate list.",
            "The run entry executes M2 only; model quantization, native GPTQ compensation, Hessian preparation, direct scoring, row selection, stitch, commit and refresh remain v18 calls.",
            "The pilot stops after the complete layer-0 checkpoint and next-layer input boundary.",
            "Checkpoint fallback counts are derived from saved per-module candidate states for safe resume.",
        ],
        "runs": impl,
        "support_files": {name: source_record(OUT / name) for name in (
            "seal_v40_freeze.py", "verify_v40_freeze.py", "run_v40_stage.py", "analyze_v40.py")},
        "reference_audit_sha256": sha(ref_path),
        "runner_diff_b_sha256": sha(OUT / "RUNNER_DIFF_B.txt"),
    }
    impl_path = OUT / "IMPLEMENTATION_HASHES.json"
    write(impl_path, implementation)
    lock["implementation_hashes_sha256"] = sha(impl_path)
    lock["reference_audit_sha256"] = sha(ref_path)
    lock["runner_diff_b_sha256"] = sha(OUT / "RUNNER_DIFF_B.txt")
    lock["run_code_bundle_sha256"] = {name: item["code_bundle_sha256"] for name, item in impl.items()}
    lock["gpu_run_started"] = False
    lock["execution_code_freeze_completed_utc"] = datetime.now(timezone.utc).isoformat()
    write(lock_path, lock)
    print(json.dumps({"status": "CODE_AND_REFERENCE_FREEZE_COMPLETE", "reference_audit_sha256": sha(ref_path),
                      "code_bundle_sha256": lock["run_code_bundle_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
