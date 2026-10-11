"""CPU-only integrity gate before resuming the interrupted v51 runner."""
import csv, hashlib, json, runpy, sys, time
from pathlib import Path
import torch

ROOT = Path(r"E:\quant-thesis")
BASE = ROOT / "experiments/range_search_w4/results"
OUT = BASE / "v51_m2_fast_integer_repair_fullmodel"

def sha(p):
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()

def main():
    assert torch.__version__ == "2.6.0+cu124"
    start = json.loads((OUT / "GPU_START.json").read_text(encoding="utf-8"))
    assert start["protocol_sha256"] == sha(OUT / "PROTOCOL.json")
    assert not (OUT / "CHECKPOINT_LOCK.json").exists() and not (OUT / "COMPLETE.json").exists()
    status = json.loads((OUT / "RUN_STATUS.json").read_text(encoding="utf-8"))
    assert status["status"] == "FAILED_STOP" and "FileNotFoundError" in status["error"]
    err = list((OUT / "FAILED_ATTEMPTS.jsonl").read_text(encoding="utf-8").splitlines())[-1]
    assert "COST_PROJECTION.json" in err
    groups = ["qkv", "o", "up_gate", "down"]
    files = sorted((OUT / "formal/groups").glob("layer_*.pt"))
    assert len(files) == 4 and {p.name for p in files} == {f"layer_00_{g}.pt" for g in groups}
    src = torch.load(OUT / "original_codes/layer_00.pt", map_location="cpu", weights_only=True)
    ns = runpy.run_path(str(OUT / "run_v51.py"), run_name="v51_resume_cpu_preflight")
    assert Path(ns["rt"].OUT).resolve() == OUT.resolve()
    for g in groups:
        cp = torch.load(OUT / f"formal/groups/layer_00_{g}.pt", map_location="cpu", weights_only=False)
        ns["assert_group"](cp, 0, g, src)
    expected = json.loads((OUT / "PROTOCOL.json").read_text(encoding="utf-8"))["budget"]
    elapsed = time.time() - float(start["start_epoch"])
    block_rows = list(csv.DictReader((OUT / "BLOCK_TIMINGS.csv").open(encoding="utf-8", newline="")))
    formal0 = [float(x["seconds"]) for x in block_rows if int(x["layer"]) == 0 and int(x["new_groups"]) > 0]
    assert formal0
    v46 = list(csv.DictReader((BASE / "v46_m1_single_integer_repair_fullmodel/BLOCK_TIMINGS.csv").open(encoding="utf-8", newline="")))
    v46max = max(float(x["seconds"]) for x in v46)
    remaining_projection = 23 * 1.35 * max(sum(formal0), v46max) + expected["finish_reserve_seconds"] + expected["setup_reserve_seconds"]
    remaining_budget = 10800 - elapsed
    assert remaining_projection < remaining_budget
    record = {
        "pass": True, "gpu_started_epoch": start["start_epoch"], "elapsed_including_failed_attempt": elapsed,
        "gpu_cap_seconds": 10800, "remaining_budget_seconds": remaining_budget,
        "remaining_projection_seconds_not_measurement": remaining_projection,
        "block0_formal_repair_seconds": sum(formal0), "v46_max_block_seconds": v46max,
        "committed_groups_restored_without_rerun": files and [p.name for p in files],
        "group_checkpoints_pass_code_grid_budget_and_protocol_checks": True,
        "no_layer1_or_later_group_exists": True, "no_ppl_or_final_lock_exists": True,
        "runner_sha256_current": sha(OUT / "run_v51.py"), "fast_accept_sha256": sha(BASE / "v48_fast_numerical_acceptance/fast_accept.py"),
        "resume_uses_original_gpu_start_and_cumulative_cap": True,
    }
    (OUT / "PRE_RESUME_AUDIT.json").write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("V51_RESUME_CPU_PREFLIGHT_PASS", json.dumps(record, ensure_ascii=False), flush=True)

if __name__ == "__main__":
    main()
