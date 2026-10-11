"""Complete the two frozen v56 evaluations after the original run was interrupted.

This script does not reconstruct or modify the model. It loads the locked v56
checkpoint and invokes the exact evaluator function used by run_v56.py.
"""
import csv
import hashlib
import importlib.util
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(r"E:\quant-thesis")
OUT = ROOT / "experiments/range_search_w4/results/v56_m1_fast_integer_repair"
MODEL = OUT / "M1_REPAIRED/checkpoint/final"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


lock = json.loads((OUT / "CHECKPOINT_LOCK.json").read_text(encoding="utf8"))
protocol_sha = sha(OUT / "PROTOCOL.json")
model_sha = sha(MODEL / "model.safetensors")
assert lock["final_model_sha256"] == model_sha
assert lock["protocol_sha256"] == protocol_sha
assert lock["linear_modules"] == 168 and lock["rows"] == 304128
assert not (OUT / "evaluation/v56_wikitext.json").exists()
assert not (OUT / "evaluation/v56_c4_fixed_budget.json").exists()
assert not (OUT / "M1_REPAIRED_wikitext.json").exists()
assert not (OUT / "M1_REPAIRED_c4_fixed_budget.json").exists()

spec = importlib.util.spec_from_file_location("v56_frozen_runner", OUT / "run_v56.py")
runner = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = runner
spec.loader.exec_module(runner)
runner.in_memory_eval_protocol()
assert runner.sha(OUT / "fast_accept_v52.py") == runner.FAST_SHA
assert runner.sha(OUT / "PROTOCOL.json") == protocol_sha

import torch
from transformers import AutoModelForCausalLM

runner.budget.start("v56 evaluation-only continuation after interrupted WT2")
eval_started = time.time()
model = None
try:
    # Preserve a 360-second close/report reserve throughout evaluation.
    def eval_deadline(reserve=0):
        runner.deadline(max(360.0, float(reserve)))

    runner.rt.deadline = eval_deadline
    model = AutoModelForCausalLM.from_pretrained(
        str(MODEL),
        torch_dtype=torch.float16,
        local_files_only=True,
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
    ).eval().cuda()
    runner.rt.MODE = "evaluation"
    runner.rt.evaluate(model)

    expected = {
        "wikitext": (299077, 582, "wikitext"),
        "c4_fixed_budget": (131071, 253, "c4_fixed_budget"),
    }
    copied = []
    for dataset, (tokens_expected, windows_expected, manifest_key) in expected.items():
        src_json = OUT / f"M1_REPAIRED_{dataset}.json"
        src_csv = OUT / f"M1_REPAIRED_{dataset}_WINDOWS.csv"
        result = json.loads(src_json.read_text(encoding="utf8"))
        rows = list(csv.DictReader(src_csv.open(encoding="utf8", newline="")))
        assert result["status"] == "PASS"
        assert result["model_sha256"] == model_sha
        assert result["manifest_sha256"] == runner.EVAL_KEY_MAP[manifest_key]
        assert int(result["scored_tokens"]) == tokens_expected
        assert int(result["windows"]) == windows_expected == len(rows)
        assert math.isclose(
            float(result["nll"]),
            sum(float(row["window_nll"]) for row in rows) / tokens_expected,
            rel_tol=0.0,
            abs_tol=1e-10,
        )
        target_json = OUT / f"evaluation/v56_{dataset}.json"
        target_csv = OUT / f"evaluation/v56_{dataset}_WINDOWS.csv"
        assert not target_json.exists() and not target_csv.exists()
        src_json.replace(target_json)
        src_csv.replace(target_csv)
        copied.append({
            "dataset": dataset,
            "result_sha256": sha(target_json),
            "windows_sha256": sha(target_csv),
            "ppl": float(result["ppl"]),
            "nll": float(result["nll"]),
            "valid_tokens": tokens_expected,
            "windows": windows_expected,
            "fresh_frozen_protocol_evaluation": True,
        })
    runner.rt.dump("QUALITY_EVALUATION_SOURCE.json", {
        "pass": len(copied) == 2,
        "new_PPL_evaluations": 2,
        "reuse_history": False,
        "P3_read": False,
        "evaluator_code_sha256": sha(OUT / "runtime_v54.py"),
        "source_manifests": runner.EVAL_KEY_MAP,
        "new_results": copied,
        "continuation_after_interrupted_partial_WT2": True,
        "partial_attempt_excluded_from_quality": True,
        "eval_only_wall_seconds": time.time() - eval_started,
    })
finally:
    if model is not None:
        del model
    runner.budget.finish()

total, _ = runner.budget.seconds()
runner.dump("EVALUATION_RESUME_COMPLETE.json", {
    "pass": True,
    "checkpoint_sha256": model_sha,
    "results": [f"evaluation/v56_{name}.json" for name in expected],
    "partial_prior_attempt_used_as_result": False,
    "v56_total_budget_seconds_including_recovery": total,
    "v56_budget_cap_seconds": 9000.0,
    "historical_shared_v55_seconds_separate": 11037.312983,
    "evaluation_implementation": "runtime_v54.py frozen v55 evaluator called via run_v56.py; no algorithm changes",
})
