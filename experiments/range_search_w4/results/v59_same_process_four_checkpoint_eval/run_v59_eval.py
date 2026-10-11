"""Same-process, same-runtime evaluation of the four frozen Qwen checkpoints.

Only model loading and PPL evaluation are performed. This runner reuses the
unchanged benchmark.eval.ppl.evaluate_ppl function and journals each complete
window for safe continuation. No checkpoint is modified.
"""
from __future__ import annotations

import csv
import gc
import hashlib
import json
import math
import os
import platform
import sys
import time
import inspect
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import tokenizers
import transformers
from safetensors import safe_open
from transformers import AutoConfig, AutoModelForCausalLM
from transformers.models.qwen2 import modeling_qwen2

ROOT = Path(r"E:\quant-thesis")
RESULTS = ROOT / "experiments/range_search_w4/results"
OUT = RESULTS / "v59_same_process_four_checkpoint_eval"
V18 = RESULTS / "range_w4_fullmodel_v18"
V56 = RESULTS / "v56_m1_fast_integer_repair"
V57 = RESULTS / "v57_uniform_runtime_four_arm_eval"
V58 = RESULTS / "v58_m2_repaired_unified_eval"
MAX_LENGTH = 2048
STRIDE = 512
CAP_SECONDS = 3600.0
FINISH_RESERVE_SECONDS = 180.0
POST_REPORT_RESERVE_SECONDS = 20.0
MODEL_REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"
EXPECTED_EFFECTIVE_CONFIG = "a33901fd787a1385efbdd5574d88012774c97110064a3f6a2cb05d070cba9dd2"
EVALUATOR_WRAPPER_SHA = "bbe1e4c22200f1ccb3d85f35a274c44cf5d46cd606160b59b0cd9afc346bae11"
V58_RUNNER_EXPECTED_SHA = "c27fb3a5dbf2f940a7ed0fd465ac02a07da8de94c9dc2a060adea962a63ed2df"
EXPECTED_CORE_EVALUATOR_SHA = "5062225fc58d6d1272ba7c828f697ff9540407db5409340d2a94662ff32841a1"
MODEL_ORDER = ["M1", "M2", "M1_plus_repair", "M2_plus_repair"]
DATASET_ORDER = ["wikitext", "c4_fixed_budget"]
ARM_SOURCES = {
    "M1": (V18 / "M1/checkpoint/final", "M1"),
    "M2": (V18 / "M2/checkpoint/final", "M2"),
    "M1_plus_repair": (V56 / "M1_REPAIRED/checkpoint/final", "M1_plus_repair"),
    "M2_plus_repair": (RESULTS / "v55_m2_decoder_fixed_integer_repair/R1/checkpoint/final", "M2_plus_repair"),
}
MANIFESTS = {
    "wikitext": (V18 / "evaluation/manifests/WIKITEXT_TEST_MANIFEST.json",
                 "acd46b20a6b84b534d5625bfab505d599001679a775d619960a40e8da4f62f46",
                 "8ffcee84dd4b684ec631cb52845e18c9c181ce6fdb8b26d65a4ffdc3d35fbda3",
                 299078, 299077, 582),
    "c4_fixed_budget": (V18 / "evaluation/manifests/C4_EVAL_MANIFEST.json",
                        "0b0d70d70d3c1e0ba62126c81a3f3e97531e58d6701236fb8b3315450b7188d4",
                        "bc9a7575527abbc41352352a0439c6bee062ea265b7bd03120704e83f4af7053",
                        131072, 131071, 253),
}
V58_PAIR_WALL_SECONDS = 445.34196853637695
RUN_STATE_PATH = OUT / "BUDGET_STATE.json"
RUN_STATE = None

def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()

def atomic_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf8")
    tmp.replace(path)

def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf8"))

def now_utc():
    return datetime.now(timezone.utc).isoformat()

def digest_tensor(t: torch.Tensor) -> str:
    b = t.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    return sha_bytes(b)

def canonical_config(config: dict) -> dict:
    volatile = {"_name_or_path", "name_or_path", "transformers_version", "_commit_hash"}
    return {k: v for k, v in config.items() if k not in volatile}

def stable_runtime_snapshot() -> dict:
    def sdp_flag(name):
        fn = getattr(torch.backends.cuda, name, None)
        try:
            return bool(fn()) if callable(fn) else "NOT_AVAILABLE"
        except Exception as e:
            return f"UNAVAILABLE:{type(e).__name__}"
    prop = torch.cuda.get_device_properties(0) if torch.cuda.is_available() else None
    free_bytes, total_bytes = torch.cuda.mem_get_info(0) if prop else (None, None)
    model_code_path = Path(inspect.getsourcefile(modeling_qwen2)).resolve()
    return {
        "python": platform.python_version(), "python_executable": sys.executable,
        "pytorch": torch.__version__, "cuda_runtime": torch.version.cuda,
        "transformers": transformers.__version__, "tokenizers": tokenizers.__version__,
        "gpu": torch.cuda.get_device_name(0) if prop else "CUDA_UNAVAILABLE",
        "gpu_total_memory_bytes": int(prop.total_memory) if prop else None,
        "gpu_free_memory_at_process_start_bytes": int(free_bytes) if free_bytes is not None else None,
        "gpu_total_memory_driver_bytes": int(total_bytes) if total_bytes is not None else None,
        "device_index": torch.cuda.current_device() if prop else None,
        "default_dtype": str(torch.get_default_dtype()), "threads": torch.get_num_threads(),
        "tf32_matmul": bool(torch.backends.cuda.matmul.allow_tf32),
        "tf32_cudnn": bool(torch.backends.cudnn.allow_tf32),
        "fp32_matmul_precision": torch.get_float32_matmul_precision(),
        "fp16_reduced_precision_reduction": bool(torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
        "autocast_cuda_enabled": bool(torch.is_autocast_enabled("cuda")),
        "autocast_cpu_enabled": bool(torch.is_autocast_enabled("cpu")),
        "sdpa_backend_policy": {
            "flash_enabled": sdp_flag("flash_sdp_enabled"),
            "memory_efficient_enabled": sdp_flag("mem_efficient_sdp_enabled"),
            "math_enabled": sdp_flag("math_sdp_enabled"),
            "cudnn_enabled": sdp_flag("cudnn_sdp_enabled"),
            "forced_context": "none; PyTorch default backend dispatch",
            "actual_kernel": "not instrumented; backend policy is recorded",
        },
        "environment_variables": {
            k: os.environ.get(k) for k in ["CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER", "CUDA_MODULE_LOADING", "CUBLAS_WORKSPACE_CONFIG", "PYTORCH_CUDA_ALLOC_CONF"]
        },
        "model_code_source": str(model_code_path),
        "model_code_sha256": sha_file(model_code_path),
        "runner_source": str(Path(__file__).resolve()),
        "runner_source_sha256": sha_file(Path(__file__).resolve()),
        "requested_model_setting": "Luna Max requested by user; model-selection API unavailable to runner",
    }

def begin_budget() -> None:
    global RUN_STATE
    now = time.time()
    if RUN_STATE_PATH.exists():
        prior = read_json(RUN_STATE_PATH)
        completed = float(prior.get("completed_active_seconds", 0.0))
        if prior.get("status") == "RUNNING" and prior.get("active_start_unix"):
            completed += max(0.0, now - float(prior["active_start_unix"]))
        first_start = prior.get("first_start_unix", now)
    else:
        marker = OUT / "START_EPOCH.json"
        marker_doc = read_json(marker) if marker.exists() else {"start_unix": now, "pre_start_audit_charge_seconds": 0.0}
        first_start = float(marker_doc["start_unix"])
        # Charge all elapsed wall since the v59 output was first created, including
        # CPU-only source/preflight work before this process starts.
        completed = max(float(marker_doc.get("pre_start_audit_charge_seconds", 0.0)), max(0.0, now - first_start))
    RUN_STATE = {
        "version": "v59", "cap_seconds": CAP_SECONDS,
        "completed_active_seconds": completed,
        "first_start_unix": first_start, "active_start_unix": now,
        "active_start_utc": now_utc(), "status": "RUNNING",
        "history_budget_reset": False,
    }
    atomic_json(RUN_STATE_PATH, RUN_STATE)

def elapsed() -> float:
    if RUN_STATE is None:
        return 0.0
    active = RUN_STATE.get("active_start_unix")
    return float(RUN_STATE["completed_active_seconds"]) + (max(0.0, time.time() - float(active)) if active is not None else 0.0)

def deadline(reserve: float = FINISH_RESERVE_SECONDS) -> None:
    if elapsed() + reserve >= CAP_SECONDS:
        raise TimeoutError(f"v59 budget stop: elapsed={elapsed():.1f}s reserve={reserve:.1f}s cap={CAP_SECONDS:.1f}s")

def end_budget(status: str, error: str | None = None) -> None:
    global RUN_STATE
    if RUN_STATE is None:
        return
    end = time.time()
    completed = elapsed()
    RUN_STATE.update({
        "completed_active_seconds": completed, "active_start_unix": None,
        "active_start_utc": None, "status": status, "ended_utc": now_utc(),
        "remaining_seconds": max(0.0, CAP_SECONDS - completed), "error": error,
    })
    atomic_json(RUN_STATE_PATH, RUN_STATE)

def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with tmp.open("w", encoding="utf8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows)
    tmp.replace(path)

def load_eval_function():
    sys.path.insert(0, str(ROOT))
    from benchmark.eval.ppl import evaluate_ppl
    return evaluate_ppl

class WindowJournal(torch.nn.Module):
    def __init__(self, inner, ids, arm, dataset, lock, journal_dir):
        super().__init__()
        self.inner = inner
        self.ids = ids
        self.arm = arm
        self.dataset = dataset
        self.lock = lock
        self.journal_dir = journal_dir
        self.index = 0
        self.sequence = int(ids.shape[1])
        self.expected_windows = max(1, math.ceil((self.sequence - MAX_LENGTH) / STRIDE) + 1)
        self.rows = []
        self.new_forwards = 0
        self.cache_hits = 0

    def forward(self, *, input_ids, labels, use_cache=False, **kwargs):
        deadline()
        idx = self.index
        if idx >= self.expected_windows:
            raise RuntimeError(f"unexpected evaluator window {self.arm}/{self.dataset}/{idx}")
        begin = idx * STRIDE
        end = min(begin + MAX_LENGTH, self.sequence)
        previous_end = 0 if idx == 0 else min((idx - 1) * STRIDE + MAX_LENGTH, self.sequence)
        target_length = end - previous_end
        expected_ids = self.ids[:, begin:end].to(input_ids.device)
        if tuple(input_ids.shape) != tuple(expected_ids.shape) or not torch.equal(input_ids, expected_ids):
            raise RuntimeError(f"token window mismatch at {self.arm}/{self.dataset}/{idx}")
        expected_labels = expected_ids.clone()
        expected_labels[:, :-target_length] = -100
        if tuple(labels.shape) != tuple(expected_labels.shape) or not torch.equal(labels, expected_labels):
            raise RuntimeError(f"score mask mismatch at {self.arm}/{self.dataset}/{idx}")
        valid = int((labels[:, 1:] != -100).sum().item())
        input_hash = digest_tensor(expected_ids)
        labels_hash = digest_tensor(expected_labels)
        path = self.journal_dir / f"window_{idx:04d}.json"
        self.index += 1
        if path.exists():
            row = read_json(path)
            if not (
                row.get("run_lock_sha256") == self.lock["run_lock_sha256"]
                and int(row.get("window_index", -1)) == idx
                and int(row.get("begin", -1)) == begin and int(row.get("end", -1)) == end
                and int(row.get("valid_tokens", -1)) == valid
                and row.get("input_ids_sha256") == input_hash and row.get("labels_sha256") == labels_hash
            ):
                raise RuntimeError(f"resume journal source/lock mismatch: {path}")
            self.cache_hits += 1
            self.rows.append(row)
            return SimpleNamespace(loss=torch.tensor(float(row["window_loss"]), dtype=torch.float32, device=input_ids.device))
        deadline()
        outputs = self.inner(input_ids=input_ids, labels=labels, use_cache=use_cache, **kwargs)
        loss = float(outputs.loss.float().item())
        if not math.isfinite(loss):
            raise ArithmeticError(f"non-finite loss at {self.arm}/{self.dataset}/{idx}")
        row = {
            "arm": self.arm, "dataset": self.dataset, "window_index": idx,
            "begin": begin, "end": end, "target_length": target_length,
            "valid_tokens": valid, "window_loss": loss, "window_nll": loss * valid,
            "input_ids_sha256": input_hash, "labels_sha256": labels_hash,
            "model_sha256": self.lock["model_sha256"],
            "manifest_sha256": self.lock["manifest_sha256"],
            "token_ids_sha256_i64le": self.lock["token_ids_sha256_i64le"],
            "config_effective_sha256": self.lock["effective_config_sha256"],
            "evaluator_core_sha256": self.lock["evaluator_core_sha256"],
            "run_lock_sha256": self.lock["run_lock_sha256"],
            "process_id": os.getpid(), "scored_new_in_this_process": True,
            "written_utc": now_utc(),
        }
        atomic_json(path, row)
        self.new_forwards += 1
        self.rows.append(row)
        progress_path = self.journal_dir.parent / "PROGRESS.json"
        if idx % 20 == 0 or idx + 1 == self.expected_windows:
            atomic_json(progress_path, {
                "arm": self.arm, "dataset": self.dataset,
                "completed_windows": idx + 1, "total_windows": self.expected_windows,
                "valid_tokens_so_far": sum(int(r["valid_tokens"]) for r in self.rows),
                "process_id": os.getpid(), "elapsed_seconds": elapsed(), "updated_utc": now_utc(),
            })
        return outputs

def verify_source_records():
    checkpoint_rows = list(csv.DictReader((V57 / "CHECKPOINT_SOURCE_AUDIT.csv").open(encoding="utf-8-sig", newline="")))
    named_rows = list(csv.DictReader((V57 / "NAMED_TENSOR_HASHES.csv").open(encoding="utf-8-sig", newline="")))
    config_comp = read_json(V57 / "CONFIG_RUNTIME_COMPARISON.json")
    expected_config_hashes = config_comp["effective_config_sha256_by_arm"]
    records = {}
    for arm in MODEL_ORDER:
        model_dir, v57_arm = ARM_SOURCES[arm]
        model_file = model_dir / "model.safetensors"
        config_file = model_dir / "config.json"
        row = next(x for x in checkpoint_rows if x["arm"] == v57_arm)
        model_sha = sha_file(model_file)
        config_sha = sha_file(config_file)
        if model_sha != row["expected_checkpoint_sha256"] or model_sha != row["actual_checkpoint_sha256"]:
            raise RuntimeError(f"checkpoint SHA mismatch for {arm}")
        if row["checkpoint_sha_match"] != "True" or row["named_tensor_count_recomputed"] != "290" or row["named_tensor_manifest_matches"] not in ("True", "new_v57_cpu_recomputation"):
            raise RuntimeError(f"v57 source audit did not pass for {arm}")
        if row["effective_config_sha256"] != EXPECTED_EFFECTIVE_CONFIG or expected_config_hashes[v57_arm] != EXPECTED_EFFECTIVE_CONFIG:
            raise RuntimeError(f"effective config differs for {arm}")
        config = AutoConfig.from_pretrained(str(model_dir), local_files_only=True)
        actual_effective_hash = sha_bytes(json.dumps(canonical_config(config.to_dict()), sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf8"))
        if actual_effective_hash != EXPECTED_EFFECTIVE_CONFIG:
            raise RuntimeError(f"current loader effective config mismatch for {arm}: {actual_effective_hash}")
        named_expected = {r["tensor"]: r for r in named_rows if r["arm"] == v57_arm}
        if len(named_expected) != 290:
            raise RuntimeError(f"named tensor manifest count mismatch for {arm}: {len(named_expected)}")
        records[arm] = {
            "checkpoint": str(model_file), "checkpoint_sha256": model_sha,
            "config_path": str(config_file), "config_sha256": config_sha,
            "effective_config_sha256": actual_effective_hash,
            "named_tensor_count": len(named_expected), "named_tensor_reference": "v57 NAMED_TENSOR_HASHES.csv",
            "v57_checkpoint_source_audit_pass": True,
            "historical_ppl_not_used": True,
            "named_expected_map": named_expected,
        }
    manifests = {}
    for ds, (path, manifest_expected, token_expected, n_input, n_valid, n_windows) in MANIFESTS.items():
        manifest_sha = sha_file(path)
        if manifest_sha != manifest_expected:
            raise RuntimeError(f"frozen manifest hash mismatch: {ds}")
        doc = read_json(path)
        tokens = doc["token_ids"]
        token_hash = sha_bytes(np.asarray(tokens, dtype="<i8").tobytes())
        if len(tokens) != n_input or len(tokens) - 1 != n_valid or token_hash != token_expected:
            raise RuntimeError(f"frozen token source mismatch: {ds}")
        window_count = max(1, math.ceil((len(tokens) - MAX_LENGTH) / STRIDE) + 1)
        if window_count != n_windows:
            raise RuntimeError(f"frozen window count mismatch: {ds}")
        manifests[ds] = {
            "path": str(path), "manifest_sha256": manifest_sha,
            "token_ids_sha256_i64le": token_hash, "input_tokens": n_input,
            "valid_tokens": n_valid, "windows": n_windows,
            "max_length": MAX_LENGTH, "stride": STRIDE,
            "add_special_tokens": False, "chat_template": False,
        }
    return records, manifests

def verify_loaded_named_tensors(model, arm: str, expected_map: dict) -> dict:
    actual = dict(model.named_parameters())
    if len(actual) != len(expected_map):
        raise RuntimeError(f"loaded named parameter count mismatch {arm}: {len(actual)} vs {len(expected_map)}")
    mismatches = []
    max_check_seconds = time.perf_counter()
    for name, p in actual.items():
        ref = expected_map.get(name)
        if ref is None:
            mismatches.append({"name": name, "reason": "missing_from_v57_manifest"})
            continue
        digest = digest_tensor(p)
        if digest != ref["sha256"] or str(p.dtype).replace("torch.", "") != ref["dtype"] or list(p.shape) != json.loads(ref["shape"]):
            mismatches.append({"name": name, "sha256_match": digest == ref["sha256"], "dtype_match": str(p.dtype).replace("torch.", "") == ref["dtype"], "shape_match": list(p.shape) == json.loads(ref["shape"])})
    if mismatches:
        atomic_json(OUT / "NAMED_TENSOR_MISMATCH.json", {"arm": arm, "mismatches": mismatches})
        raise RuntimeError(f"loaded named tensor source mismatch for {arm}")
    return {"arm": arm, "count": len(actual), "all_hashes_match": True, "seconds": time.perf_counter() - max_check_seconds}

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    begin_budget()
    run_start = time.perf_counter()
    try:
        torch.set_num_threads(4)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable")
        runtime = stable_runtime_snapshot()
        expected_v58_env = read_json(V58 / "ENVIRONMENT_LOCK.json")
        for k, old_key in [("python", "python"), ("pytorch", "pytorch"), ("cuda_runtime", "cuda_build"), ("transformers", "transformers"), ("tokenizers", "tokenizers"), ("gpu", "device")]:
            if runtime[k] != expected_v58_env[old_key]:
                raise RuntimeError(f"runtime differs from v58 for {k}: {runtime[k]} vs {expected_v58_env[old_key]}")
        runtime["attention_loader_request"] = "attn_implementation=sdpa"
        runtime["eval_mode"] = True
        runtime["use_cache"] = False
        runtime["dtype"] = "float16"
        runtime["evaluator_core_path"] = str(ROOT / "benchmark/eval/ppl.py")
        runtime["evaluator_core_sha256"] = sha_file(ROOT / "benchmark/eval/ppl.py")
        wrapper_sha = sha_file(V58 / "run_v58_eval.py")
        if wrapper_sha != V58_RUNNER_EXPECTED_SHA:
            raise RuntimeError(f"v58 frozen evaluator wrapper changed: {wrapper_sha}")
        runtime["evaluator_wrapper_sha256"] = wrapper_sha
        runtime["v56_evaluator_helper_sha256"] = EVALUATOR_WRAPPER_SHA
        runtime["model_code_revision"] = MODEL_REVISION
        runtime["tokenizer_revision"] = MODEL_REVISION
        runtime["process_id"] = os.getpid()
        if runtime["evaluator_core_sha256"] != EXPECTED_CORE_EVALUATOR_SHA:
            raise RuntimeError("current evaluator source differs from v58 evaluator source")
        atomic_json(OUT / "PROCESS_ENVIRONMENT.json", runtime)
        eval_ppl = load_eval_function()
        if eval_ppl.__module__ != "benchmark.eval.ppl":
            raise RuntimeError("unexpected evaluator module")

        source_records, manifest_records = verify_source_records()
        source_rows = []
        for arm in MODEL_ORDER:
            r = dict(source_records[arm]); r.pop("named_expected_map")
            r["arm"] = arm
            source_rows.append(r)
        write_csv(OUT / "CHECKPOINT_SOURCE_AUDIT.csv", source_rows)
        write_csv(OUT / "INPUT_SOURCE_AUDIT.csv", [{"dataset": k, **v} for k, v in manifest_records.items()])

        runtime_signature = {k: v for k, v in runtime.items() if k not in {"process_id", "gpu_free_memory_at_process_start_bytes"}}
        runtime_hash = sha_bytes(json.dumps(runtime_signature, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf8"))
        evaluator_sha = runtime["evaluator_core_sha256"]
        source_hash = sha_file(V57 / "NAMED_TENSOR_HASHES.csv")
        protocol = {
            "version": "v59", "status": "FROZEN_BEFORE_FIRST_MODEL_LOAD",
            "requested_model_setting": "Luna Max requested; execution interface cannot verify model selection",
            "objective": "Actual same-process, same-runtime PPL comparison of M1, M2, v56 M1+repair and v55 M2+repair",
            "prohibited": ["quantization", "integer repair", "search", "new calibration", "P3", "PPL tuning"],
            "model_order": MODEL_ORDER, "dataset_order_per_model": DATASET_ORDER,
            "model_revision": MODEL_REVISION, "tokenizer_revision": MODEL_REVISION,
            "loader": {"dtype": "FP16", "device": "CUDA", "attn_implementation": "sdpa", "local_files_only": True, "low_cpu_mem_usage": True, "eval": True, "use_cache": False},
            "runtime_signature_sha256": runtime_hash,
            "evaluator_core_sha256": evaluator_sha,
            "evaluator_wrapper_source": str(V58 / "run_v58_eval.py"),
            "evaluator_wrapper_source_sha256": runtime["evaluator_wrapper_sha256"],
            "source_manifest_sha256": source_hash,
            "evaluation": {k: {x: manifest_records[k][x] for x in ["manifest_sha256", "token_ids_sha256_i64le", "input_tokens", "valid_tokens", "windows", "max_length", "stride", "add_special_tokens", "chat_template"]} for k in DATASET_ORDER},
            "four_arms_main_table_only_from_this_v59_run": True,
            "historical_calibration_propagation_validity": "UNKNOWN and not re-evaluated here",
            "budget_seconds": CAP_SECONDS,
            "resume": "window journal rows are reused only after runtime/run lock, model, manifest, input and label-mask validation; resumed rows retain source process id, and same-process eligibility is checked explicitly",
        }
        protocol_sha = sha_bytes(json.dumps(protocol, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf8"))
        atomic_json(OUT / "PROTOCOL.json", protocol)
        frozen = {
            "runtime_signature_sha256": runtime_hash,
            "evaluator_core_sha256": evaluator_sha,
            "v57_named_tensor_source_sha256": source_hash,
            "protocol_sha256": protocol_sha,
            "model_order": MODEL_ORDER, "dataset_order": DATASET_ORDER,
            "manifests": manifest_records,
        }
        frozen_sha = sha_bytes(json.dumps(frozen, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf8"))
        frozen["run_lock_sha256"] = frozen_sha
        lock_path = OUT / "RUN_LOCK.json"
        if lock_path.exists():
            old = read_json(lock_path)
            if old != frozen:
                raise RuntimeError("resume runtime/protocol/source lock differs")
        else:
            atomic_json(lock_path, frozen)

        # Cost forecast uses v58's measured current-runtime process wall, not v56's slower record.
        pair_wall = float(read_json(V58 / "COST_ACCOUNTING.json")["process_wall_including_loader_and_evaluation"])
        named_verify_allowance = 4 * 15.0
        source_precheck_allowance = 90.0
        report_reserve = 60.0
        charged_before_load = elapsed()
        projection = charged_before_load + 4 * pair_wall + named_verify_allowance + source_precheck_allowance + report_reserve + FINISH_RESERVE_SECONDS
        cost_precheck = {
            "cap_seconds": CAP_SECONDS,
            "charged_before_first_model_load_seconds": charged_before_load,
            "v58_measured_one_model_pair_process_wall_seconds": pair_wall,
            "four_pair_extrapolation_seconds": 4 * pair_wall,
            "loaded_named_tensor_verification_allowance_seconds_estimate": named_verify_allowance,
            "startup_source_and_runtime_checks_seconds_estimate": source_precheck_allowance,
            "report_hash_closeout_seconds_estimate": report_reserve,
            "finish_reserve_seconds": FINISH_RESERVE_SECONDS,
            "projected_total_seconds": projection,
            "remaining_margin_seconds": CAP_SECONDS - projection,
            "prediction_only": True,
            "basis": "v58 measured current-runtime runner process wall per one checkpoint plus 90s source/setup, 60s closeout and explicit estimates; no use of v56 slow wall",
        }
        if projection >= CAP_SECONDS:
            raise TimeoutError(f"v59 forecast exceeds cap: {projection:.1f}s")
        atomic_json(OUT / "COST_PRECHECK.json", cost_precheck)

        v18_proto = read_json(V18 / "PROTOCOL.json")
        hist_validity = {
            "M1_M2_v18_calibration_propagation": "UNKNOWN; this PPL comparison does not repair that evidence",
            "M1_repair_v56_calibration_propagation": "UNKNOWN; retained from v53/v56 history",
            "M2_repair_v55_calibration_propagation": "UNKNOWN; retained from v53-v55 history",
            "model_revision": v18_proto["model"]["revision"],
        }
        atomic_json(OUT / "HISTORICAL_VALIDITY_BOUNDARY.json", hist_validity)

        load_rows = []
        stage_rows = []
        all_results = []
        for arm in MODEL_ORDER:
            deadline(FINISH_RESERVE_SECONDS)
            model_dir, source_arm = ARM_SOURCES[arm]
            source = source_records[arm]
            if sha_file(model_dir / "model.safetensors") != source["checkpoint_sha256"]:
                raise RuntimeError(f"checkpoint changed before load: {arm}")
            torch.cuda.reset_peak_memory_stats()
            load_start = time.perf_counter()
            model = AutoModelForCausalLM.from_pretrained(
                str(model_dir), torch_dtype=torch.float16, local_files_only=True,
                low_cpu_mem_usage=True, attn_implementation="sdpa",
            ).eval().cuda()
            torch.cuda.synchronize()
            load_seconds = time.perf_counter() - load_start
            first_param = next(model.parameters())
            actual_effective_hash = sha_bytes(json.dumps(canonical_config(model.config.to_dict()), sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf8"))
            loaded_attn = getattr(model.config, "_attn_implementation", None)
            if first_param.dtype != torch.float16 or first_param.device.type != "cuda" or model.training or actual_effective_hash != EXPECTED_EFFECTIVE_CONFIG or loaded_attn != "sdpa":
                raise RuntimeError(f"loaded model runtime/config mismatch for {arm}")
            named_start = time.perf_counter()
            named_check = verify_loaded_named_tensors(model, arm, source["named_expected_map"])
            named_seconds = time.perf_counter() - named_start
            load_rows.append({
                "arm": arm, "checkpoint_path": str(model_dir / "model.safetensors"),
                "checkpoint_sha256": source["checkpoint_sha256"], "named_tensor_count": named_check["count"],
                "named_hashes_match_v57": named_check["all_hashes_match"],
                "config_effective_sha256": actual_effective_hash,
                "dtype": str(first_param.dtype), "device": str(first_param.device),
                "attention_requested": "sdpa", "attention_loaded": loaded_attn,
                "eval_mode": not model.training, "use_cache_eval_argument": False,
                "load_seconds": load_seconds, "loaded_parameter_hash_seconds": named_seconds,
                "process_id": os.getpid(),
                "peak_cuda_allocated_bytes_after_load": torch.cuda.max_memory_allocated(),
                "peak_cuda_reserved_bytes_after_load": torch.cuda.max_memory_reserved(),
            })
            write_csv(OUT / "MODEL_LOAD_AUDIT.csv", load_rows)

            for ds in DATASET_ORDER:
                deadline(FINISH_RESERVE_SECONDS)
                manifest = manifest_records[ds]
                ids_list = read_json(Path(manifest["path"]))["token_ids"]
                ids = torch.tensor(ids_list, dtype=torch.long).unsqueeze(0)
                journal_dir = OUT / "evaluation" / arm / ds / "windows"
                journal_dir.mkdir(parents=True, exist_ok=True)
                run_obj = {
                    "run_lock_sha256": frozen_sha, "arm": arm,
                    "model_sha256": source["checkpoint_sha256"],
                    "effective_config_sha256": EXPECTED_EFFECTIVE_CONFIG,
                    "manifest_sha256": manifest["manifest_sha256"],
                    "token_ids_sha256_i64le": manifest["token_ids_sha256_i64le"],
                    "evaluator_core_sha256": evaluator_sha,
                    "runtime_signature_sha256": runtime_hash,
                    "dataset": ds, "max_length": MAX_LENGTH, "stride": STRIDE,
                    "dtype": "float16", "attention": "sdpa", "tf32": False,
                    "use_cache": False, "score": "valid-token weighted NLL then exp",
                }
                window_lock_hash = sha_bytes(json.dumps(run_obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf8"))
                window_lock = {**run_obj, "run_lock_sha256": window_lock_hash}
                window_lock_path = journal_dir.parent / "WINDOW_LOCK.json"
                if window_lock_path.exists() and read_json(window_lock_path) != window_lock:
                    raise RuntimeError(f"window resume lock mismatch {arm}/{ds}")
                if not window_lock_path.exists():
                    atomic_json(window_lock_path, window_lock)
                wrapper = WindowJournal(model, ids, arm, ds, window_lock, journal_dir)
                wrapper.eval()
                eval_start = time.perf_counter()
                ppl, mean_nll, valid_total = eval_ppl(wrapper, ids, max_length=MAX_LENGTH, stride=STRIDE, progress_every=25)
                torch.cuda.synchronize()
                eval_seconds = time.perf_counter() - eval_start
                if len(wrapper.rows) != manifest["windows"] or int(valid_total) != manifest["valid_tokens"]:
                    raise RuntimeError(f"incomplete evaluator output {arm}/{ds}")
                nll_sum = sum(float(r["window_nll"]) for r in wrapper.rows)
                valid_sum = sum(int(r["valid_tokens"]) for r in wrapper.rows)
                recalc_nll = nll_sum / valid_sum
                recalc_ppl = math.exp(recalc_nll)
                if valid_sum != manifest["valid_tokens"] or not math.isclose(recalc_nll, float(mean_nll), rel_tol=0, abs_tol=1e-10) or not math.isclose(recalc_ppl, float(ppl), rel_tol=1e-12, abs_tol=1e-12):
                    raise ArithmeticError(f"window aggregation mismatch {arm}/{ds}")
                # Read durable files again, so a reported result always has every frozen window.
                durable = [read_json(journal_dir / f"window_{i:04d}.json") for i in range(manifest["windows"])]
                if [int(r["window_index"]) for r in durable] != list(range(manifest["windows"])):
                    raise RuntimeError(f"non-contiguous windows {arm}/{ds}")
                durable_sum = sum(float(r["window_nll"]) for r in durable)
                durable_valid = sum(int(r["valid_tokens"]) for r in durable)
                if durable_valid != manifest["valid_tokens"]:
                    raise RuntimeError(f"durable valid token mismatch {arm}/{ds}")
                csv_rows = [{k: r[k] for k in ["arm", "dataset", "window_index", "begin", "end", "target_length", "valid_tokens", "window_loss", "window_nll", "input_ids_sha256", "labels_sha256", "model_sha256", "manifest_sha256", "token_ids_sha256_i64le", "config_effective_sha256", "evaluator_core_sha256", "process_id"]} for r in durable]
                write_csv(OUT / "evaluation" / f"{arm}_{ds}_WINDOWS.csv", csv_rows)
                result = {
                    "status": "PASS", "arm": arm, "dataset": ds,
                    "checkpoint_sha256": source["checkpoint_sha256"],
                    "effective_config_sha256": EXPECTED_EFFECTIVE_CONFIG,
                    "manifest_sha256": manifest["manifest_sha256"],
                    "token_ids_sha256_i64le": manifest["token_ids_sha256_i64le"],
                    "input_tokens": manifest["input_tokens"], "valid_tokens": durable_valid,
                    "windows": len(durable), "nll_sum": durable_sum,
                    "weighted_nll": durable_sum / durable_valid,
                    "ppl": math.exp(durable_sum / durable_valid),
                    "max_length": MAX_LENGTH, "stride": STRIDE,
                    "evaluator_core_sha256": evaluator_sha,
                    "runtime_signature_sha256": runtime_hash,
                    "eval_seconds_this_invocation": eval_seconds,
                    "new_forwards_this_process": wrapper.new_forwards,
                    "resumed_window_cache_hits": wrapper.cache_hits,
                    "window_process_ids": sorted({int(r["process_id"]) for r in durable}),
                    "independent_window_aggregation_pass": True,
                }
                atomic_json(OUT / "evaluation" / f"{arm}_{ds}.json", result)
                all_results.append(result)
                stage_rows.append({"arm": arm, "dataset": ds, "stage": "evaluation", "seconds": eval_seconds, "windows": len(durable), "new_forwards": wrapper.new_forwards, "resumed_hits": wrapper.cache_hits, "process_id": os.getpid()})
                write_csv(OUT / "STAGE_TIMINGS.csv", stage_rows)
                atomic_json(OUT / "RUN_PROGRESS.json", {
                    "status": "IN_PROGRESS", "last_completed_arm": arm,
                    "last_completed_dataset": ds, "completed_evaluations": len(all_results),
                    "target_evaluations": 8, "completed_windows": len(durable),
                    "elapsed_seconds": elapsed(), "remaining_seconds": max(0.0, CAP_SECONDS - elapsed()),
                    "process_id": os.getpid(), "updated_utc": now_utc(),
                })
                del wrapper, ids, ids_list
            load_rows[-1]["peak_cuda_allocated_bytes_after_both_evaluations"] = torch.cuda.max_memory_allocated()
            load_rows[-1]["peak_cuda_reserved_bytes_after_both_evaluations"] = torch.cuda.max_memory_reserved()
            write_csv(OUT / "MODEL_LOAD_AUDIT.csv", load_rows)
            unload_start = time.perf_counter()
            del model
            model = None
            gc.collect()
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            stage_rows.append({"arm": arm, "dataset": "", "stage": "unload_gc_empty_cache", "seconds": time.perf_counter() - unload_start, "windows": 0, "new_forwards": 0, "resumed_hits": 0, "process_id": os.getpid()})
            write_csv(OUT / "STAGE_TIMINGS.csv", stage_rows)

        all_window_process_ids = set()
        for arm in MODEL_ORDER:
            for ds in DATASET_ORDER:
                r = read_json(OUT / "evaluation" / f"{arm}_{ds}.json")
                all_window_process_ids.update(r["window_process_ids"])
        if len(all_results) != 8:
            raise RuntimeError(f"only {len(all_results)} of 8 evaluations completed")
        single_process = len(all_window_process_ids) == 1 and next(iter(all_window_process_ids)) == os.getpid()
        atomic_json(OUT / "RUN_SUMMARY.json", {
            "status": "COMPLETE", "complete_evaluations": 8,
            "same_process_all_scored_windows": single_process,
            "process_ids_across_windows": sorted(all_window_process_ids),
            "same_runtime_signature_all_windows": True,
            "historical_calibration_propagation_validity": "UNKNOWN and unchanged",
            "P3_read": False, "models_modified": False,
            "elapsed_seconds": elapsed(), "budget_cap_seconds": CAP_SECONDS,
        })
        atomic_json(OUT / "RUN_PROGRESS.json", {
            "status": "ALL_EVALUATIONS_COMPLETE", "completed_evaluations": 8,
            "target_evaluations": 8, "elapsed_seconds": elapsed(),
            "remaining_seconds": max(0.0, CAP_SECONDS - elapsed()), "process_id": os.getpid(),
        })
        atomic_json(OUT / "COST_ACCOUNTING.json", {
            "cap_seconds": CAP_SECONDS,
            "runner_active_seconds": elapsed(),
            "process_id": os.getpid(),
            "model_load_seconds": {r["arm"]: float(r["load_seconds"]) for r in load_rows},
            "loaded_named_tensor_hash_seconds": {r["arm"]: float(r["loaded_parameter_hash_seconds"]) for r in load_rows},
            "evaluation_seconds_by_arm_dataset": {f"{r['arm']}/{r['dataset']}": r["seconds"] for r in stage_rows if r["stage"] == "evaluation"},
            "prediction_from_v58_not_actual_speed_claim": {"source_pair_wall_seconds": pair_wall, "four_pair_projection_seconds": 4 * pair_wall},
            "source_and_setup_seconds_not_separately_synchronized": "included in runner_active_seconds",
            "post_report_reserve_seconds": POST_REPORT_RESERVE_SECONDS,
            "history_ledgers_reset": False,
        })
        end_budget("COMPLETE")
        atomic_json(OUT / "RUN_STATUS.json", {
            "status": "COMPLETE", "complete_evaluations": 8,
            "same_process_all_scored_windows": single_process,
            "process_ids_across_windows": sorted(all_window_process_ids),
            "charged_active_seconds": elapsed(), "budget_cap_seconds": CAP_SECONDS,
            "remaining_seconds": max(0.0, CAP_SECONDS - elapsed()),
            "historical_ledgers_reset": False,
        })
        print("V59_COMPLETE", json.dumps(read_json(OUT / "RUN_STATUS.json"), ensure_ascii=False), flush=True)
    except Exception as e:
        try:
            end_budget("FAILED_OR_PARTIAL", f"{type(e).__name__}: {e}")
        finally:
            atomic_json(OUT / "RUN_STATUS.json", {
                "status": "FAILED_OR_PARTIAL", "error": f"{type(e).__name__}: {e}",
                "completed_evaluations": sum(1 for a in MODEL_ORDER for d in DATASET_ORDER if (OUT / "evaluation" / f"{a}_{d}.json").exists()),
                "charged_active_seconds": elapsed(), "budget_cap_seconds": CAP_SECONDS,
                "history_ledgers_reset": False,
            })
        raise

if __name__ == "__main__":
    main()
