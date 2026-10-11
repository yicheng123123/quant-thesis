"""One-time continuation for v46 PPL after its frozen launcher field lookup bug.

Does not touch the repair implementation or frozen protocol. Verifies the
original locked R1 artifact and the already-audited v18 manifest hashes, then
calls runtime_v46.evaluate unchanged exactly once.
"""
import csv
import hashlib
import importlib.util
import json
import sys
import threading
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[3]
sys.path.insert(0, str(OUT))
spec = importlib.util.spec_from_file_location("runtime_v46_eval_continuation", OUT / "runtime_v46.py")
rt = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = rt
spec.loader.exec_module(rt)

CAP = 10800
RESERVE = 360


def read(name):
    return json.loads((OUT / name).read_text(encoding="utf8"))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    status = read("RUN_STATUS.json")
    assert status["status"] == "FAILED_STOP" and status["error"] == "KeyError('v18')"
    assert "evaluate(model)" in status["traceback"] and "~PPL~" not in status["traceback"]

    repair = read("FORMAL_REPAIR_COMPLETE.json")
    assert repair["modules"] == 168 and repair["rows"] == 304128
    lock = read("CHECKPOINT_LOCK.json")
    deployment = read("DEPLOYMENT_LOCK.json")
    assert deployment["pass"] is True
    assert lock["final_model_sha256"] == deployment["final_model_sha256"]
    assert lock["config_sha256"] == deployment["config_sha256"]
    assert lock["protocol_sha256"] == sha(OUT / "PROTOCOL.json")
    model_path = OUT / "R1/checkpoint/final/model.safetensors"
    config_path = OUT / "R1/checkpoint/final/config.json"
    assert sha(model_path) == lock["final_model_sha256"]
    assert sha(config_path) == lock["config_sha256"]
    for item in lock["group_checkpoints"]:
        assert sha(item["path"]) == item["sha256"]
    assert len(lock["group_checkpoints"]) == 96

    # The manifest SHA values were frozen and checked in BASELINE_AUDIT.json.
    baseline = read("BASELINE_AUDIT.json")
    assert baseline["pass"] and baseline["comparison_protocol_shared"]
    expected = baseline["evaluation_manifests"]
    manifest_names = {
        "wikitext": ("WIKITEXT_TEST_MANIFEST.json", "v16_wikitext_manifest_sha256"),
        "c4_fixed_budget": ("C4_EVAL_MANIFEST.json", "v16_c4_manifest_sha256"),
    }
    source_protocol = {}
    for dataset, (manifest, key) in manifest_names.items():
        path = rt.V18 / "evaluation/manifests" / manifest
        got = sha(path)
        assert got == expected[dataset], f"audited manifest changed: {dataset}"
        payload = json.loads(path.read_text(encoding="utf8"))
        assert len(payload["token_ids"]) == baseline["token_counts"][dataset][0] + 1
        source_protocol[key] = got
        assert not (OUT / f"R1_{dataset}.json").exists()
        assert not (OUT / f"R1_{dataset}_WINDOWS.csv").exists()

    # runtime_v46.evaluate is unchanged. Supply its missing hash lookup in
    # memory only, from the independently frozen baseline audit above.
    rt.P.setdefault("v18", {})["source_protocol"] = source_protocol
    start = read("GPU_START.json")["unix"]
    assert read("GPU_START.json")["hard_cap_seconds"] == CAP
    rt.WALL_START = start
    rt.START = time.perf_counter()
    rt.MODE = "evaluation"

    def deadline(reserve=0):
        if rt.elapsed() > CAP - RESERVE - reserve:
            raise TimeoutError(f"v46 original GPU cap/reserve reached: elapsed={rt.elapsed():.2f}s")

    rt.deadline = deadline

    def hard_stop():
        rt.dump("EVALUATION_CONTINUATION_STATUS.json", {
            "status": "INCOMPLETE_HARD_CAP", "elapsed_seconds": rt.elapsed(),
            "model_sha256": lock["final_model_sha256"], "PPL_run": "partial_or_unknown",
        })
        import os
        os._exit(124)

    watchdog = threading.Timer(max(1, CAP - RESERVE - rt.elapsed()), hard_stop)
    watchdog.daemon = True
    watchdog.start()
    try:
        with rt.timer("evaluation_checkpoint_reload"):
            model = AutoModelForCausalLM.from_pretrained(
                str(model_path.parent), torch_dtype=torch.float16,
                local_files_only=True, low_cpu_mem_usage=True,
                attn_implementation="sdpa",
            ).eval().cuda()
        rt.evaluate(model)
        summary = {}
        for dataset in ("wikitext", "c4_fixed_budget"):
            result = read(f"R1_{dataset}.json")
            assert result["status"] == "PASS"
            assert result["model_sha256"] == lock["final_model_sha256"]
            summary[dataset] = {
                "ppl": result["ppl"], "nll": result["nll"],
                "valid_tokens": result["scored_tokens"],
                "windows": result["windows"],
                "result_sha256": sha(OUT / f"R1_{dataset}.json"),
                "windows_sha256": sha(OUT / f"R1_{dataset}_WINDOWS.csv"),
            }
        final = {
            "status": "COMPLETE", "category": "quality_pending_aggregation",
            "repair_complete": True, "modules": 168, "rows": 304128,
            "PPL_run": True, "model_sha256": lock["final_model_sha256"],
            "elapsed_seconds_from_original_gpu_start": rt.elapsed(),
            "hard_cap_seconds": CAP, "reserved_seconds": RESERVE,
            "continuation_reason": "evaluation lookup KeyError('v18'); no repair rerun",
            "manifest_hashes_verified_from_baseline_audit": expected,
            "evaluation": summary,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            "gpu_name": torch.cuda.get_device_name(), "torch": torch.__version__,
        }
        rt.dump("FINAL_STATUS.json", final)
        rt.dump("EVALUATION_CONTINUATION_STATUS.json", final)
    finally:
        watchdog.cancel()


if __name__ == "__main__":
    main()
