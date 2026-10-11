from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from verify_v40_freeze import OUT, ROOT, sha, verify


BUDGET_PATH = OUT / "BUDGET.json"
EXECUTION_CUTOFF = 4700.0
HARD_CAP = 5400.0
RESERVE = 700.0


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def add_phase_estimates(budget: dict) -> dict:
    ref = budget["reference"]
    full = float(ref["v18_m2_wall_seconds"])
    reduced = float(ref["projected_B_total_seconds_with_v18_non_candidate_residual"])
    # Frozen scheduling margins only. They affect launch decisions, never candidate scoring.
    pilot = ref.get("pilot_projected_seconds")
    if pilot is None:
        pilot = estimate_pilot_seconds()
    return {"pilot": float(pilot),
            "A": full * 1.20, "B": reduced * 1.25,
            "margin_factors": {"A": 1.20, "B": 1.25}}


def update_budget(budget: dict, path=BUDGET_PATH):
    write_json(path, budget)


def prelaunch_gate(phase: str, budget: dict) -> None:
    completed = set(budget.get("completed_phases", []))
    if phase == "PILOT":
        if completed or "PILOT" in completed:
            raise RuntimeError(f"unexpected pre-pilot completed phase state: {completed}")
    elif phase == "A":
        pilot = OUT / "PILOT_PARITY.json"
        if "PILOT" not in completed or not pilot.exists() or read_json(pilot).get("status") != "PASS":
            raise RuntimeError("A is gated on PASS from the complete block0 pilot parity check")
        if "A" in completed:
            raise RuntimeError("A full run is already complete; no repeated full-model timing runs are allowed")
    elif phase == "B":
        validity = OUT / "A_RUN_VALIDITY.json"
        if "A" not in completed or not validity.exists() or read_json(validity).get("status") != "PASS":
            raise RuntimeError("B is gated on a complete A run matching the historical M2 reference")
        if "B" in completed:
            raise RuntimeError("B full run is already complete; no repeated full-model timing runs are allowed")


def estimate_pilot_seconds() -> float:
    # The pilot estimate reuses v18 layer-0 stage timings and the frozen B candidate list.
    v18 = ROOT / "experiments/range_search_w4/results/range_w4_fullmodel_v18"
    manifest = read_json(OUT / "CANDIDATE_MANIFEST.json")
    timing = list(csv.DictReader((v18 / "FORMAL_TIMING.csv").open(encoding="utf-8-sig", newline="")))
    candidate_stages = {"full_linear_alpha_candidate_gptq", "all_row_direct_output_scoring"}
    projected_variable = 0.0
    for stage in candidate_stages:
        by_module = {}
        for r in timing:
            if r["method"] == "M2" and int(r["layer"]) == 0 and r["stage"] == stage:
                by_module.setdefault(r["module"], []).append(float(r["wall_seconds"]))
        for module, item in manifest["modules"].items():
            if item["layer"] == 0:
                times = by_module[module]
                projected_variable += sum(times) / len(times) * item["candidate_count"]
    fixed_layer0 = sum(float(r["wall_seconds"]) for r in timing
                       if r["method"] == "M2" and (int(r["layer"]) == 0 or int(r["layer"]) == -1)
                       and r["stage"] not in candidate_stages)
    return (projected_variable + fixed_layer0) * 1.5


def run(phase: str) -> int:
    verify()
    budget = read_json(BUDGET_PATH)
    if budget.get("hard_cap_seconds") != HARD_CAP or budget.get("execution_cutoff_seconds") != EXECUTION_CUTOFF:
        raise RuntimeError("frozen v40 budget constants changed")
    prelaunch_gate(phase, budget)
    elapsed = float(budget.get("gpu_experiment_elapsed_seconds", 0.0))
    estimates = add_phase_estimates(budget)
    pending = []
    if phase == "PILOT":
        pending = [estimates["pilot"], estimates["A"], estimates["B"]]
    elif phase == "A":
        pending = [estimates["A"], estimates["B"]]
    else:
        pending = [estimates["B"]]
    required = sum(pending)
    if elapsed + required > EXECUTION_CUTOFF:
        budget["status"] = "BUDGET_GATE_STOP"
        budget["launch_denial"] = {"phase": phase, "elapsed_seconds": elapsed,
                                   "remaining_execution_window_seconds": EXECUTION_CUTOFF - elapsed,
                                   "estimated_remaining_seconds_including_next_required_runs": required,
                                   "reserve_seconds": RESERVE}
        update_budget(budget)
        print(json.dumps(budget["launch_denial"], indent=2))
        return 3

    now = datetime.now(timezone.utc)
    if not budget.get("started_utc"):
        budget["started_utc"] = now.isoformat()
        budget["cutoff_utc"] = datetime.fromtimestamp(now.timestamp() + EXECUTION_CUTOFF, timezone.utc).isoformat()
        budget["status"] = "RUNNING"
        budget["reference"]["pilot_projected_seconds"] = estimate_pilot_seconds()
        estimates["pilot"] = float(budget["reference"]["pilot_projected_seconds"])
    budget["launch_estimates"] = estimates
    budget["active_phase"] = phase
    budget["active_phase_started_utc"] = now.isoformat()
    budget["active_phase_wall_seconds"] = 0.0
    update_budget(budget)
    lock_path = OUT / "FREEZE_LOCK.json"
    lock = read_json(lock_path)
    lock["gpu_run_started"] = True
    lock.setdefault("gpu_started_utc", now.isoformat())
    write_json(lock_path, lock)

    run_dir = OUT / phase
    log_dir = OUT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{phase}.log"
    cmd = [sys.executable, "-B", "-X", "utf8", str(run_dir / "run_formal.py")]
    started = time.monotonic()
    last_budget_write = started
    hard_stopped = False
    with log_path.open("ab", buffering=0) as log:
        log.write((f"\n--- phase {phase} start {now.isoformat()} ---\n").encode("utf-8"))
        proc = subprocess.Popen(cmd, cwd=run_dir, stdout=log, stderr=subprocess.STDOUT,
                                env=os.environ.copy(), creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0))
        try:
            while proc.poll() is None:
                stage_seconds = time.monotonic() - started
                cumulative = elapsed + stage_seconds
                if cumulative >= EXECUTION_CUTOFF:
                    hard_stopped = True
                    proc.terminate()
                    try:
                        proc.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        proc.kill(); proc.wait(timeout=15)
                    break
                if time.monotonic() - last_budget_write >= 15:
                    budget = read_json(BUDGET_PATH)
                    budget["active_phase_wall_seconds"] = stage_seconds
                    budget["gpu_experiment_elapsed_seconds"] = cumulative
                    update_budget(budget)
                    last_budget_write = time.monotonic()
                time.sleep(2)
            return_code = proc.wait()
        except KeyboardInterrupt:
            proc.terminate()
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                proc.kill(); proc.wait(timeout=15)
            return_code = proc.returncode if proc.returncode is not None else 130
            hard_stopped = True
        finally:
            stage_seconds = time.monotonic() - started
            budget = read_json(BUDGET_PATH)
            record = {"phase": phase, "started_utc": now.isoformat(),
                      "finished_utc": datetime.now(timezone.utc).isoformat(),
                      "wall_seconds": stage_seconds, "return_code": proc.returncode,
                      "budget_stopped": hard_stopped, "log": str(log_path.relative_to(OUT))}
            budget.setdefault("phase_ledger", []).append(record)
            budget["gpu_experiment_elapsed_seconds"] = sum(x["wall_seconds"] for x in budget["phase_ledger"])
            budget["active_phase"] = None
            budget["active_phase_wall_seconds"] = 0.0
            if hard_stopped or proc.returncode != 0:
                budget["status"] = "STOPPED_AT_BUDGET" if hard_stopped else "PHASE_FAILED_RESUMABLE"
                budget.setdefault("failed_attempts", []).append(record)
            else:
                if phase not in budget.setdefault("completed_phases", []):
                    budget["completed_phases"].append(phase)
                budget["status"] = {"PILOT": "PILOT_COMPLETE_NEEDS_PARITY",
                                    "A": "A_COMPLETE_NEEDS_REFERENCE_PARITY",
                                    "B": "RUNS_COMPLETE_NEEDS_FINAL_AUDIT"}[phase]
            update_budget(budget)
            log.write((f"--- phase {phase} end {record['finished_utc']} wall={stage_seconds:.3f}s code={proc.returncode} ---\n").encode("utf-8"))
    print(json.dumps(record, indent=2))
    return 124 if hard_stopped else int(return_code)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=["PILOT", "A", "B"])
    args = parser.parse_args()
    raise SystemExit(run(args.phase))
