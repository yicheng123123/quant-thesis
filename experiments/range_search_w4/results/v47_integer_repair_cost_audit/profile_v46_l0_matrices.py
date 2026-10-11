"""Bounded synchronized stage profiling on the two complete L0 M1 matrices.

Reconstructs X from the locked R1 model, verifies exact v46 X/C hashes, and
replays the byte-identical v45 repair with one additional p-update timer.
No model weights are committed and no model checkpoint is written.
"""
from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import os
import sys
import threading
import time
import traceback
from pathlib import Path

import torch
from safetensors import safe_open
from transformers import AutoModelForCausalLM


ROOT = Path(r"E:\quant-thesis")
V46 = ROOT / "experiments/range_search_w4/results/v46_m1_single_integer_repair_fullmodel"
OUT = ROOT / "experiments/range_search_w4/results/v47_integer_repair_cost_audit"
CAP = 600.0
START = time.perf_counter()
DEADLINE = START + CAP
WATCHDOG = None


def dump(name, obj):
    p = OUT / name
    p.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def tensor_sha(t):
    return hashlib.sha256(t.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def elapsed():
    return time.perf_counter() - START


def hard_deadline(reserve=0):
    if time.perf_counter() >= DEADLINE - reserve:
        raise TimeoutError(f"v47 GPU profiling cap: elapsed={elapsed():.3f}s, cap={CAP:.1f}s, reserve={reserve:.1f}s")


def make_instrumented_repair(rt):
    src_path = V46 / "repair_v45.py"
    original = src_path.read_text(encoding="utf-8")
    needle = "        st['p'][ar]+=C[ac,:]*dv[accepted,None];st['loss'][ar]=newloss[ar];st['updates'][ar]+=1\n"
    replacement = (
        "        with rt.timer('profile_p_vector_update'):\n"
        "            st['p'][ar]+=C[ac,:]*dv[accepted,None]\n"
        "        st['loss'][ar]=newloss[ar];st['updates'][ar]+=1\n"
    )
    if original.count(needle) != 1:
        raise RuntimeError("Frozen v45 p-update statement no longer matches the expected source")
    instrumented = original.replace(needle, replacement)
    out_path = OUT / "instrumented_repair_v45.py"
    out_path.write_text(instrumented, encoding="utf-8")
    # Persist a narrow diff to show this copy adds only one timer around p.
    diff = (
        "--- v46/repair_v45.py\n+++ v47/instrumented_repair_v45.py\n"
        "@@ p update in settle() @@\n"
        "-        st['p'][ar]+=C[ac,:]*dv[accepted,None];st['loss'][ar]=newloss[ar];st['updates'][ar]+=1\n"
        "+        with rt.timer('profile_p_vector_update'):\n"
        "+            st['p'][ar]+=C[ac,:]*dv[accepted,None]\n"
        "+        st['loss'][ar]=newloss[ar];st['updates'][ar]+=1\n"
    )
    (OUT / "INSTRUMENTATION_DIFF.txt").write_text(diff, encoding="utf-8")
    return out_path


def import_runtime_and_repair():
    sys.path.insert(0, str(V46))
    sys.path.insert(0, str(ROOT))
    spec = importlib.util.spec_from_file_location("run_v42", V46 / "runtime_v46.py")
    rt = importlib.util.module_from_spec(spec)
    sys.modules["run_v42"] = rt
    spec.loader.exec_module(rt)
    rt.OUT = OUT
    rt.MODE = "profile"
    rt.LAYER = 0
    rt.WALL_START = time.time()
    rt.START = time.perf_counter()
    rt.deadline = hard_deadline
    rt.TIMINGS = []
    rt.STATS = []
    instrumented_path = make_instrumented_repair(rt)
    rspec = importlib.util.spec_from_file_location("repair_v45_profile", instrumented_path)
    repair = importlib.util.module_from_spec(rspec)
    sys.modules[rspec.name] = repair
    rspec.loader.exec_module(repair)
    return rt, repair


def load_expected(path, module):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if not ck.get("committed"):
        raise RuntimeError(f"Historical source group is not committed: {path}")
    result = ck["modules"][module]
    keep = {k: result[k] for k in ("codes", "scales", "g_idx", "zeros", "updates", "evaluations", "initial_mse", "final_mse")}
    return ck["input_sha256"], ck["C_sha256"], keep


def main():
    global WATCHDOG
    OUT.mkdir(parents=True, exist_ok=True)
    start_json = {"status": "RUNNING", "cap_seconds": CAP, "start_utc_epoch": time.time(),
                  "gpu_used": True, "targets": ["L0 mlp.up_proj full (262144x896 -> 4864)",
                                                  "L0 mlp.down_proj full (262144x4864 -> 896)"],
                  "source_model_sha256": file_sha(V46 / "R1/checkpoint/final/model.safetensors"),
                  "v46_protocol_sha256": file_sha(V46 / "PROTOCOL.json"),
                  "v46_repair_sha256": file_sha(V46 / "repair_v45.py"),
                  "v46_runtime_sha256": file_sha(V46 / "runtime_v46.py")}
    dump("PROFILE_START.json", start_json)

    def hard_stop():
        try:
            dump("PROFILE_HARD_CAP.json", {"status": "HARD_CAP_STOP", "elapsed_seconds": elapsed(), "cap_seconds": CAP})
        finally:
            os._exit(124)

    WATCHDOG = threading.Timer(CAP, hard_stop)
    WATCHDOG.daemon = True
    WATCHDOG.start()
    rt = None
    results = []
    try:
        rt, repair = import_runtime_and_repair()
        torch.set_num_threads(4)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable")
        dump("PROFILE_STATUS.json", {"status": "PREFLIGHT_PASS", "elapsed_seconds": elapsed(),
                                     "gpu": torch.cuda.get_device_name(), "torch": torch.__version__,
                                     "cap_seconds": CAP})
        with rt.timer("profile_warmup"):
            z = torch.zeros((64, 64), device="cuda", dtype=torch.float32)
            _ = z @ z

        with rt.timer("profile_R1_model_load"):
            model = AutoModelForCausalLM.from_pretrained(
                str(V46 / "R1/checkpoint/final"), torch_dtype=torch.float16,
                local_files_only=True, low_cpu_mem_usage=True, attn_implementation="sdpa").eval().cuda()
        initial_model_sha = file_sha(V46 / "R1/checkpoint/final/model.safetensors")
        with torch.inference_mode():
            hidden = rt.embed(model)

        state = torch.load(V46 / "original_codes/layer_00.pt", map_location="cpu", weights_only=True)
        targets = [
            ("up_gate", "mlp.up_proj", V46 / "formal/groups/layer_00_up_gate.pt"),
            ("down", "mlp.down_proj", V46 / "formal/groups/layer_00_down.pt"),
        ]
        with safe_open(str(rt.FP / "model.safetensors"), framework="pt", device="cpu") as pristine:
            for group, module, ckpt_path in targets:
                hard_deadline()
                expected_input_sha, expected_C_sha, expected = load_expected(ckpt_path, module)
                X = rt.capture(model, 0, hidden, group)
                input_sha = hashlib.sha256(X.tobytes()).hexdigest()
                if input_sha != expected_input_sha:
                    raise RuntimeError(f"R1 replay input hash mismatch for {group}: {input_sha} != {expected_input_sha}")
                with rt.timer("profile_input_hash"):
                    _ = input_sha
                C = rt.build_C(X)
                c_sha = rt.th(C.detach().cpu())
                if c_sha != expected_C_sha:
                    raise RuntimeError(f"C hash mismatch for {group}: {c_sha} != {expected_C_sha}")
                s = state[module]
                W = pristine.get_tensor(f"model.layers.0.{module}.weight").half()
                model_weight_before_sha = tensor_sha(model.model.layers[0].get_submodule(module).weight)
                with torch.inference_mode():
                    result = repair.batch_repair(s["codes"], s["scales"], s["g_idx"], W, X, C, module,
                                                 save_rows=False)
                exact = {
                    "codes": torch.equal(result["codes"], expected["codes"]),
                    "scales": torch.equal(result["scales"], expected["scales"]),
                    "g_idx": torch.equal(result["g_idx"], expected["g_idx"]),
                    "zeros": torch.equal(result["zeros"], expected["zeros"]),
                    "updates": torch.equal(result["updates"], expected["updates"]),
                    "evaluations": torch.equal(result["evaluations"], expected["evaluations"]),
                    "initial_mse": torch.equal(result["initial_mse"], expected["initial_mse"]),
                    "final_mse": torch.equal(result["final_mse"], expected["final_mse"]),
                }
                if not all(exact.values()):
                    raise RuntimeError(f"v45 replay differs from committed v46 result {group}: {exact}")
                model_weight_after_sha = tensor_sha(model.model.layers[0].get_submodule(module).weight)
                if model_weight_after_sha != model_weight_before_sha:
                    raise RuntimeError(f"R1 model tensor changed during non-committing profiling: {module}")
                results.append({"layer": 0, "group": group, "module": module,
                                "rows": int(result["codes"].shape[0]), "input_dim": int(result["codes"].shape[1]),
                                "input_sha256": input_sha, "C_sha256": c_sha,
                                "expected_input_sha256": expected_input_sha, "expected_C_sha256": expected_C_sha,
                                "exact_match_to_v46_terminal_state": exact,
                                "accepted_updates": int(result["updates"].sum()),
                                "candidate_evaluations": int(result["evaluations"].sum()),
                                "direct_FP64_rechecks": len(result["recheck_events"]),
                                "rejected_candidates": int(result["rejections"].sum()),
                                "initial_search_mse_sum": float(result["initial_mse"].sum()),
                                "final_search_mse_sum": float(result["final_mse"].sum()),
                                "model_weight_sha256_before_after_equal": model_weight_before_sha == model_weight_after_sha,
                                "checkpoint_written": False})
                del result, expected, X, C
                torch.cuda.empty_cache()
        final_model_sha = file_sha(V46 / "R1/checkpoint/final/model.safetensors")
        if final_model_sha != initial_model_sha:
            raise RuntimeError("R1 checkpoint file changed during profile")
        torch.cuda.synchronize()
        status = {"status": "COMPLETE", "elapsed_seconds": elapsed(), "cap_seconds": CAP,
                  "model_file_unchanged": True, "checkpoint_written": False,
                  "held_or_PPL_scored": False, "committed_to_model": False,
                  "targets": results, "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                  "peak_reserved_bytes": torch.cuda.max_memory_reserved(), "gpu": torch.cuda.get_device_name(),
                  "torch": torch.__version__, "v46_source_model_sha256": initial_model_sha,
                  "candidate_formula_and_order_unchanged": True,
                  "only_v45_source_addition": "one synchronized timer around st['p'][accepted_rows] += C[columns,:] * actual_delta"}
        dump("PROFILE_RESULTS.json", status)
        dump("PROFILE_STATUS.json", status)
    except BaseException as exc:
        dump("PROFILE_STATUS.json", {"status": "FAILED", "elapsed_seconds": elapsed(),
                                     "cap_seconds": CAP, "error": repr(exc),
                                     "traceback": traceback.format_exc(), "completed_targets": results})
        raise
    finally:
        if WATCHDOG is not None:
            WATCHDOG.cancel()


if __name__ == "__main__":
    main()
