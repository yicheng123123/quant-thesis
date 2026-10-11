"""Re-evaluate the locked v55 checkpoint with the v56 frozen PPL evaluator.

This script loads an existing model only. It does not quantize, calibrate,
search, modify weights, or read P3. Window loss values are atomically journaled
so a resumed invocation skips completed forwards while still calling the
unchanged benchmark.eval.ppl.evaluate_ppl over the complete frozen token stream.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import transformers
import tokenizers
from transformers import AutoConfig, AutoModelForCausalLM

ROOT = Path(r"E:\quant-thesis")
BASE = ROOT / "experiments/range_search_w4/results"
OUT = BASE / "v58_m2_repaired_unified_eval"
V18 = BASE / "range_w4_fullmodel_v18"
V55 = BASE / "v55_m2_decoder_fixed_integer_repair"
V56 = BASE / "v56_m1_fast_integer_repair"
V57 = BASE / "v57_uniform_runtime_four_arm_eval"
MODEL_DIR = V55 / "R1/checkpoint/final"
MODEL_FILE = MODEL_DIR / "model.safetensors"
MAX_LENGTH = 2048
STRIDE = 512
CAP_SECONDS = 3600.0
FINISH_RESERVE_SECONDS = 180.0
PRERUN_AUDIT_CONSERVATIVE_SECONDS = 30.0
EXPECTED_MODEL_SHA = "0fa87105e36deac2ead4c871e6e0a802590418e6ccacd616929cdf91ecc8ec0c"

MANIFESTS = {
    "wikitext": (V18 / "evaluation/manifests/WIKITEXT_TEST_MANIFEST.json",
                 "acd46b20a6b84b534d5625bfab505d599001679a775d619960a40e8da4f62f46",
                 "8ffcee84dd4b684ec631cb52845e18c9c181ce6fdb8b26d65a4ffdc3d35fbda3",
                 299077, 582),
    "c4_fixed_budget": (V18 / "evaluation/manifests/C4_EVAL_MANIFEST.json",
                        "0b0d70d70d3c1e0ba62126c81a3f3e97531e58d6701236fb8b3315450b7188d4",
                        "bc9a7575527abbc41352352a0439c6bee062ea265b7bd03120704e83f4af7053",
                        131071, 253),
}

def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()

def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf8")
    tmp.replace(path)

def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf8"))

def now_utc():
    return datetime.now(timezone.utc).isoformat()

STATE_PATH = OUT / "BUDGET_STATE.json"
STATE = None

def begin_budget_segment() -> None:
    global STATE
    now = time.time()
    if STATE_PATH.exists():
        old = read_json(STATE_PATH)
        charged = float(old.get("completed_active_seconds", 0.0))
        if old.get("status") == "RUNNING" and old.get("active_segment_started_unix") is not None:
            charged += max(0.0, now - float(old["active_segment_started_unix"]))
        prior_prerun = float(old.get("pre_run_audit_charge_seconds", PRERUN_AUDIT_CONSERVATIVE_SECONDS))
    else:
        charged = PRERUN_AUDIT_CONSERVATIVE_SECONDS
        prior_prerun = PRERUN_AUDIT_CONSERVATIVE_SECONDS
    STATE = {
        "version": "v58",
        "cap_seconds": CAP_SECONDS,
        "finish_reserve_seconds": FINISH_RESERVE_SECONDS,
        "pre_run_audit_charge_seconds": prior_prerun,
        "completed_active_seconds": charged,
        "active_segment_started_unix": now,
        "active_segment_started_utc": now_utc(),
        "status": "RUNNING",
        "history_budgets_reset": False,
    }
    atomic_json(STATE_PATH, STATE)

def elapsed_seconds() -> float:
    if STATE is None:
        return 0.0
    active_start = STATE.get("active_segment_started_unix")
    if active_start is None:
        return float(STATE["completed_active_seconds"])
    return float(STATE["completed_active_seconds"]) + max(0.0, time.time() - float(active_start))

def budget_guard(reserve: float = FINISH_RESERVE_SECONDS) -> None:
    e = elapsed_seconds()
    if e + reserve >= CAP_SECONDS:
        raise TimeoutError(f"v58 active-time budget guard: elapsed={e:.1f}s reserve={reserve:.1f}s cap={CAP_SECONDS:.1f}s")

def end_budget_segment(status: str, error: str | None = None) -> None:
    global STATE
    if STATE is None:
        return
    now = time.time()
    completed = float(STATE["completed_active_seconds"]) + max(0.0, now - float(STATE["active_segment_started_unix"]))
    STATE.update({
        "completed_active_seconds": completed,
        "active_segment_started_unix": None,
        "active_segment_started_utc": None,
        "status": status,
        "ended_utc": now_utc(),
        "remaining_seconds": max(0.0, CAP_SECONDS - completed),
        "error": error,
    })
    atomic_json(STATE_PATH, STATE)

def canonical_config(config: dict) -> dict:
    ignored = {"_commit_hash", "_name_or_path", "name_or_path", "transformers_version"}
    return {k: v for k, v in config.items() if k not in ignored}

def config_hash(config: dict) -> str:
    return sha_bytes(json.dumps(config, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf8"))

def load_frozen_evaluator():
    sys.path.insert(0, str(ROOT))
    from benchmark.eval.ppl import evaluate_ppl
    return evaluate_ppl

def tensor_digest(t: torch.Tensor) -> str:
    a = t.detach().to("cpu").contiguous().numpy()
    return sha_bytes(a.tobytes(order="C"))

class WindowJournalModel(torch.nn.Module):
    """Loss-transparent wrapper that durably captures or reuses one scorer window."""
    def __init__(self, inner, input_ids, dataset_name: str, run_lock: dict, journal_dir: Path):
        super().__init__()
        self.inner = inner
        self.input_ids = input_ids
        self.dataset_name = dataset_name
        self.run_lock = run_lock
        self.journal_dir = journal_dir
        self.calls = 0
        self.n = int(input_ids.shape[1])
        self.expected_count = max(1, math.ceil((self.n - MAX_LENGTH) / STRIDE) + 1)
        self.rows = []
        self.cache_hits = 0
        self.new_forwards = 0

    def forward(self, *, input_ids, labels, use_cache=False, **kwargs):
        budget_guard()
        idx = self.calls
        if idx >= self.expected_count:
            raise RuntimeError(f"unexpected extra PPL window: {idx}")
        begin = idx * STRIDE
        end = min(begin + MAX_LENGTH, self.n)
        previous_end = 0 if idx == 0 else min((idx - 1) * STRIDE + MAX_LENGTH, self.n)
        target_length = end - previous_end
        expected_ids = self.input_ids[:, begin:end].to(input_ids.device)
        if tuple(input_ids.shape) != tuple(expected_ids.shape) or not torch.equal(input_ids, expected_ids):
            raise RuntimeError(f"frozen token window mismatch dataset={self.dataset_name} idx={idx}")
        expected_labels = expected_ids.clone()
        expected_labels[:, :-target_length] = -100
        if tuple(labels.shape) != tuple(expected_labels.shape) or not torch.equal(labels, expected_labels):
            raise RuntimeError(f"frozen scoring mask mismatch dataset={self.dataset_name} idx={idx}")
        valid = int((labels[:, 1:] != -100).sum().item())
        id_hash = tensor_digest(expected_ids)
        label_hash = tensor_digest(expected_labels)
        row_path = self.journal_dir / f"window_{idx:04d}.json"
        self.calls += 1
        if row_path.exists():
            row = read_json(row_path)
            checks = (
                row.get("run_lock_sha256") == self.run_lock["run_lock_sha256"]
                and int(row.get("window_index", -1)) == idx
                and int(row.get("begin", -1)) == begin
                and int(row.get("end", -1)) == end
                and int(row.get("valid_tokens", -1)) == valid
                and row.get("input_ids_sha256") == id_hash
                and row.get("labels_sha256") == label_hash
            )
            if not checks:
                raise RuntimeError(f"journal window source/lock mismatch: {row_path}")
            self.cache_hits += 1
            self.rows.append(row)
            return SimpleNamespace(loss=torch.tensor(float(row["window_loss"]), dtype=torch.float32, device=input_ids.device))

        budget_guard()
        outputs = self.inner(input_ids=input_ids, labels=labels, use_cache=use_cache, **kwargs)
        loss = float(outputs.loss.float().item())
        if not math.isfinite(loss):
            raise ArithmeticError(f"non-finite PPL loss at {self.dataset_name} window {idx}")
        row = {
            "dataset": self.dataset_name,
            "window_index": idx,
            "begin": begin,
            "end": end,
            "target_length": target_length,
            "valid_tokens": valid,
            "window_loss": loss,
            "window_nll": loss * valid,
            "input_ids_sha256": id_hash,
            "labels_sha256": label_hash,
            "run_lock_sha256": self.run_lock["run_lock_sha256"],
            "forward_reused": False,
            "written_utc": now_utc(),
        }
        atomic_json(row_path, row)
        self.new_forwards += 1
        self.rows.append(row)
        if idx % 25 == 0 or idx + 1 == self.expected_count:
            atomic_json(self.journal_dir.parent / "PROGRESS.json", {
                "dataset": self.dataset_name,
                "completed_windows": idx + 1,
                "total_windows": self.expected_count,
                "valid_tokens_so_far": sum(int(r["valid_tokens"]) for r in self.rows),
                "elapsed_seconds": elapsed_seconds(),
                "checkpoint_sha256": self.run_lock["checkpoint_sha256"],
                "updated_utc": now_utc(),
            })
        return outputs

def write_rows_csv(path: Path, rows: list[dict]) -> None:
    fields = ["dataset", "window_index", "begin", "end", "target_length", "valid_tokens", "window_loss", "window_nll", "input_ids_sha256", "labels_sha256", "run_lock_sha256", "forward_reused", "written_utc"]
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k) for k in fields})
    tmp.replace(path)

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "evaluation").mkdir(parents=True, exist_ok=True)
    (OUT / "reports").mkdir(parents=True, exist_ok=True)
    begin_budget_segment()
    try:
        torch.set_num_threads(4)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; v55 evaluation cannot start")
        if not MODEL_FILE.is_file():
            raise FileNotFoundError(MODEL_FILE)

        # Source, manifest, runtime, and evaluator checks are part of this run's budget.
        checkpoint_sha = sha_file(MODEL_FILE)
        lock = read_json(V55 / "FINAL_CHECKPOINT_LOCK.json")
        assert lock["model_sha256"] == checkpoint_sha == EXPECTED_MODEL_SHA
        named_csv = V55 / "NAMED_TENSOR_HASHES.csv"
        named_sha = sha_file(named_csv)
        assert named_sha == lock["named_tensor_hashes_csv_sha256"]
        v57_source_rows = list(csv.DictReader((V57 / "CHECKPOINT_SOURCE_AUDIT.csv").open(encoding="utf-8-sig", newline="")))
        v55_source = next(r for r in v57_source_rows if r["arm"] == "M2_plus_repair")
        assert v55_source["checkpoint_sha_match"] == "True"
        assert v55_source["named_tensor_manifest_matches"] == "True"
        assert v55_source["named_tensor_count_recomputed"] == "290"
        assert v55_source["named_tensor_manifest_sha256"] == named_sha

        expected_evaluator_sha = "bbe1e4c22200f1ccb3d85f35a274c44cf5d46cd606160b59b0cd9afc346bae11"
        assert sha_file(V56 / "runtime_v54.py") == expected_evaluator_sha
        core_evaluator_path = ROOT / "benchmark/eval/ppl.py"
        core_evaluator_sha = sha_file(core_evaluator_path)
        evaluate_ppl = load_frozen_evaluator()
        assert evaluate_ppl.__module__ == "benchmark.eval.ppl"

        token_sets = {}
        manifest_records = {}
        for ds, (path, expected_manifest_sha, expected_token_sha, expected_valid, expected_windows) in MANIFESTS.items():
            actual_sha = sha_file(path)
            assert actual_sha == expected_manifest_sha, f"{ds} frozen manifest hash mismatch"
            doc = read_json(path)
            tokens = doc["token_ids"]
            token_sha = sha_bytes(np.asarray(tokens, dtype="<i8").tobytes())
            assert token_sha == expected_token_sha, f"{ds} frozen token ID hash mismatch"
            assert len(tokens) - 1 == expected_valid
            window_count = max(1, math.ceil((len(tokens) - MAX_LENGTH) / STRIDE) + 1)
            assert window_count == expected_windows
            token_sets[ds] = torch.tensor(tokens, dtype=torch.long).unsqueeze(0)
            manifest_records[ds] = {
                "path": str(path), "manifest_sha256": actual_sha,
                "token_ids_sha256_i64le": token_sha, "input_tokens": len(tokens),
                "expected_valid_tokens": expected_valid, "expected_windows": expected_windows,
                "max_length": MAX_LENGTH, "stride": STRIDE,
                "add_special_tokens": False, "chat_template": False,
            }

        env = {
            "python": platform.python_version(),
            "python_executable": sys.executable,
            "pytorch": torch.__version__,
            "cuda_build": torch.version.cuda,
            "transformers": transformers.__version__,
            "tokenizers": tokenizers.__version__,
            "device": torch.cuda.get_device_name(0),
            "device_total_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
            "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
            "tf32_cudnn": torch.backends.cudnn.allow_tf32,
            "fp32_matmul_precision": torch.get_float32_matmul_precision(),
            "num_threads": torch.get_num_threads(),
            "evaluator_wrapper_sha256": expected_evaluator_sha,
            "core_evaluator_path": str(core_evaluator_path),
            "core_evaluator_sha256_current": core_evaluator_sha,
            "core_evaluator_historical_sha256": "NOT_SAVED_IN_V56_ARTIFACTS",
            "same_core_evaluator_function_module": evaluate_ppl.__module__,
        }
        source_audit = {
            "v55_model_sha256": checkpoint_sha,
            "v55_config_sha256": sha_file(MODEL_DIR / "config.json"),
            "v55_named_tensor_manifest_sha256": named_sha,
            "v55_named_tensor_count": int(lock["named_tensor_count"]),
            "v55_named_tensor_source_audit": "v57 recomputed all 290 named tensor hashes and matched the v55 manifest; direct checkpoint SHA rechecked in v58",
            "v55_original_eval_runtime": v55_source["historical_eval_runtime"],
            "v56_checkpoint_sha256": read_json(V56 / "CHECKPOINT_LOCK.json")["final_model_sha256"],
            "v56_protocol_torch": read_json(V56 / "PROTOCOL.json")["execution_runtime"]["torch_version_required"],
            "v56_runtime_python_record": "MISSING",
            "v56_runtime_transformers_record": "MISSING; config serialization transformers_version is save-time metadata only",
            "v56_evaluator_wrapper_sha256": expected_evaluator_sha,
            "v56_core_evaluator_source_hash": "NOT_RECORDED",
            "current_vs_v56_strict_runtime_identity": "NOT_PROVEN because v56 evaluation-process Python and Transformers versions are absent",
            "manifests": manifest_records,
            "historical_propagation_validity": "UNKNOWN; re-evaluation does not establish calibration propagation validity",
            "P3_read": False,
            "quantization_or_repair_run": False,
        }
        atomic_json(OUT / "ENVIRONMENT_LOCK.json", env)
        atomic_json(OUT / "SOURCE_AUDIT.json", source_audit)

        # Check expected effective configuration before spending the evaluation wall.
        config = AutoConfig.from_pretrained(str(MODEL_DIR), local_files_only=True)
        eff_config = canonical_config(config.to_dict())
        effective_config_sha = config_hash(eff_config)
        expected_effective = "a33901fd787a1385efbdd5574d88012774c97110064a3f6a2cb05d070cba9dd2"
        assert effective_config_sha == expected_effective, f"normalized effective config mismatch: {effective_config_sha}"

        # Prepare immutable run identity and per-window lock files.
        run_common = {
            "checkpoint_sha256": checkpoint_sha,
            "config_sha256": sha_file(MODEL_DIR / "config.json"),
            "effective_config_sha256": effective_config_sha,
            "evaluator_wrapper_sha256": expected_evaluator_sha,
            "core_evaluator_sha256_current": core_evaluator_sha,
            "runtime": {k: env[k] for k in ("python", "pytorch", "cuda_build", "transformers", "tokenizers", "device")},
            "max_length": MAX_LENGTH, "stride": STRIDE,
            "use_cache": False, "eval_mode": True, "dtype": "float16",
            "attention_requested": "sdpa", "tf32": False,
            "score": "sum(window_loss * valid_tokens) / sum(valid_tokens); PPL=exp(weighted_NLL)",
        }
        for ds, rec in manifest_records.items():
            run_obj = {**run_common, "dataset": ds, "manifest_sha256": rec["manifest_sha256"], "token_ids_sha256_i64le": rec["token_ids_sha256_i64le"]}
            run_sha = sha_bytes(json.dumps(run_obj, sort_keys=True, separators=(",", ":")).encode("utf8"))
            lock_obj = {**run_obj, "run_lock_sha256": run_sha}
            lock_path = OUT / "evaluation" / ds / "WINDOW_LOCK.json"
            if lock_path.exists():
                prior = read_json(lock_path)
                if prior != lock_obj:
                    raise RuntimeError(f"resume lock mismatch for {ds}")
            else:
                atomic_json(lock_path, lock_obj)

        # Cost precheck uses the measured v56 eval-only process wall, which includes model load.
        v56_wall = float(read_json(V56 / "QUALITY_EVALUATION_SOURCE.json")["new_results"] is not None and read_json(V56 / "QUALITY_EVALUATION_SOURCE.json")["eval_only_wall_seconds"])
        if elapsed_seconds() + v56_wall + FINISH_RESERVE_SECONDS >= CAP_SECONDS:
            raise TimeoutError(f"measured-pair projection does not fit remaining budget: elapsed={elapsed_seconds():.1f}, pair={v56_wall:.1f}, reserve={FINISH_RESERVE_SECONDS:.1f}")
        atomic_json(OUT / "COST_PRECHECK.json", {
            "cap_seconds": CAP_SECONDS,
            "pre_run_audit_conservative_charge_seconds": PRERUN_AUDIT_CONSERVATIVE_SECONDS,
            "elapsed_before_model_load_seconds": elapsed_seconds(),
            "v56_measured_eval_only_pair_wall_seconds_including_load": v56_wall,
            "finish_reserve_seconds": FINISH_RESERVE_SECONDS,
            "projected_remaining_after_pair_seconds": CAP_SECONDS - elapsed_seconds() - v56_wall,
            "feasible": True,
            "stage_sum_seconds_v56": sum(float(r["seconds"]) for r in csv.DictReader((V56 / "STAGE_TIMINGS.csv").open(encoding="utf8", newline="")) if r.get("mode") == "evaluation"),
            "projection_is_estimate": True,
            "history_budgets_reset": False,
        })
        atomic_json(OUT / "PROTOCOL.json", {
            "version": "v58",
            "status": "FROZEN_BEFORE_EVALUATION",
            "objective": "Re-evaluate only v55 M2+repair under current runtime using the unchanged v56 PPL evaluator and frozen WT2/C4 inputs",
            "no_quantization_repair_search_or_calibration": True,
            "no_P3": True,
            "checkpoint_sha256": checkpoint_sha,
            "named_tensor_manifest_sha256": named_sha,
            "effective_config_sha256": effective_config_sha,
            "evaluator_wrapper_sha256": expected_evaluator_sha,
            "evaluator_core_source_sha256_current": core_evaluator_sha,
            "evaluation": {"datasets": list(MANIFESTS), "max_length": MAX_LENGTH, "stride": STRIDE, "token_weighted": True, "use_cache": False, "dtype": "FP16", "attention_requested": "SDPA", "TF32": False, "progress_journal": "atomic per-window JSON"},
            "runtime_matches_v56": "strict identity unverified: v56 evaluation process Python and Transformers versions not recorded",
            "v56_result_reuse": "not used as strict main comparison; values retained as contextual reference",
            "four_arm_interaction": "not computed; historical M1/M2 runtimes differ and v56 process runtime is incompletely recorded",
            "budget": {"cap_seconds": CAP_SECONDS, "finish_reserve_seconds": FINISH_RESERVE_SECONDS, "prior_history_budgets_reset": False},
        })

        # The saved token streams are the frozen tokenizer outputs; no retokenization occurs.
        budget_guard()
        load_started = time.perf_counter()
        model = AutoModelForCausalLM.from_pretrained(
            str(MODEL_DIR), torch_dtype=torch.float16, local_files_only=True,
            low_cpu_mem_usage=True, attn_implementation="sdpa",
        ).eval().cuda()
        torch.cuda.synchronize()
        load_seconds = time.perf_counter() - load_started
        parameter = next(model.parameters())
        loaded_cfg = canonical_config(model.config.to_dict())
        loaded_effective_sha = config_hash(loaded_cfg)
        loaded_record = {
            "status": "PASS",
            "checkpoint_sha256": checkpoint_sha,
            "parameter_dtype": str(parameter.dtype),
            "parameter_device": str(parameter.device),
            "model_training_flag": bool(model.training),
            "config_effective_sha256": loaded_effective_sha,
            "config_matches_preload": loaded_effective_sha == effective_config_sha,
            "attention_requested_by_loader": "sdpa",
            "loaded_config_attention_implementation": getattr(model.config, "_attn_implementation", None),
            "lower_level_SDPA_kernel": "NOT_MEASURED; v56 artifacts do not record selected kernel",
            "use_cache_passed_by_evaluator": False,
            "load_synchronized_wall_seconds": load_seconds,
            "gpu_peak_allocated_bytes_after_load": torch.cuda.max_memory_allocated(),
            "gpu_peak_reserved_bytes_after_load": torch.cuda.max_memory_reserved(),
        }
        assert parameter.dtype == torch.float16 and parameter.device.type == "cuda" and not model.training
        assert loaded_effective_sha == expected_effective
        if getattr(model.config, "_attn_implementation", None) not in (None, "sdpa"):
            raise RuntimeError(f"loaded attention implementation is not SDPA: {model.config._attn_implementation}")
        atomic_json(OUT / "LOADER_CHECK.json", loaded_record)

        results = []
        for ds in ("wikitext", "c4_fixed_budget"):
            budget_guard()
            journal_dir = OUT / "evaluation" / ds / "windows"
            journal_dir.mkdir(parents=True, exist_ok=True)
            lock_obj = read_json(OUT / "evaluation" / ds / "WINDOW_LOCK.json")
            proxy = WindowJournalModel(model, token_sets[ds], ds, lock_obj, journal_dir)
            proxy.eval()
            eval_started = time.perf_counter()
            ppl, nll, n_tokens = evaluate_ppl(proxy, token_sets[ds], max_length=MAX_LENGTH, stride=STRIDE, progress_every=25)
            torch.cuda.synchronize()
            eval_wall = time.perf_counter() - eval_started
            exp_nll = sum(float(r["window_nll"]) for r in proxy.rows) / sum(int(r["valid_tokens"]) for r in proxy.rows)
            exp_ppl = math.exp(exp_nll)
            expected_valid, expected_windows = MANIFESTS[ds][3], MANIFESTS[ds][4]
            if len(proxy.rows) != expected_windows or sum(int(r["valid_tokens"]) for r in proxy.rows) != expected_valid or int(n_tokens) != expected_valid:
                raise RuntimeError(f"window count/token count mismatch for {ds}")
            if not (math.isclose(float(nll), exp_nll, rel_tol=0, abs_tol=1e-10) and math.isclose(float(ppl), exp_ppl, rel_tol=1e-12, abs_tol=1e-12)):
                raise ArithmeticError(f"independent NLL/PPL aggregation mismatch for {ds}")
            # Re-read all durable rows before finalizing the dataset result.
            durable_rows = [read_json(journal_dir / f"window_{i:04d}.json") for i in range(expected_windows)]
            assert all(r["run_lock_sha256"] == lock_obj["run_lock_sha256"] for r in durable_rows)
            write_rows_csv(OUT / "evaluation" / f"v58_{ds}_WINDOWS.csv", durable_rows)
            result = {
                "status": "PASS", "dataset": ds, "model_sha256": checkpoint_sha,
                "manifest_sha256": manifest_records[ds]["manifest_sha256"],
                "token_ids_sha256_i64le": manifest_records[ds]["token_ids_sha256_i64le"],
                "input_tokens": manifest_records[ds]["input_tokens"],
                "valid_tokens": sum(int(r["valid_tokens"]) for r in durable_rows),
                "windows": len(durable_rows), "nll_sum": sum(float(r["window_nll"]) for r in durable_rows),
                "weighted_nll": exp_nll, "ppl": exp_ppl,
                "max_length": MAX_LENGTH, "stride": STRIDE,
                "add_special_tokens": False, "chat_template": False,
                "evaluator_wrapper_sha256": expected_evaluator_sha,
                "evaluator_core_source_sha256_current": core_evaluator_sha,
                "eval_wall_seconds_excluding_model_load": eval_wall,
                "new_forwards_this_invocation": proxy.new_forwards,
                "resumed_window_cache_hits": proxy.cache_hits,
                "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
                "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(),
            }
            atomic_json(OUT / "evaluation" / f"v58_{ds}.json", result)
            results.append(result)
            atomic_json(OUT / "RUN_PROGRESS.json", {
                "status": "PARTIAL" if len(results) < 2 else "EVALUATIONS_COMPLETE",
                "datasets_complete": [r["dataset"] for r in results],
                "elapsed_seconds": elapsed_seconds(), "remaining_seconds": max(0.0, CAP_SECONDS - elapsed_seconds()),
                "last_completed_dataset": ds,
            })
            del proxy

        # Recalculate once more using only finalized window files.
        for result in results:
            rows = list(csv.DictReader((OUT / "evaluation" / f"v58_{result['dataset']}_WINDOWS.csv").open(encoding="utf8", newline="")))
            nll_sum = sum(float(r["window_nll"]) for r in rows)
            valid_sum = sum(int(r["valid_tokens"]) for r in rows)
            weighted = nll_sum / valid_sum
            ppl2 = math.exp(weighted)
            if abs(weighted - float(result["weighted_nll"])) > 1e-10 or abs(ppl2 - float(result["ppl"])) > 1e-12:
                raise ArithmeticError(f"final CSV aggregation mismatch for {result['dataset']}")
            result["independent_csv_recalc"] = {"nll_sum": nll_sum, "valid_tokens": valid_sum, "weighted_nll": weighted, "ppl": ppl2, "pass": True}
            atomic_json(OUT / "evaluation" / f"v58_{result['dataset']}.json", result)

        # Main comparison is strictly limited to evaluations whose runtime can be used here.
        rows_main = []
        for result in results:
            ref = read_json(V56 / "evaluation" / f"v56_{result['dataset']}.json")
            rows_main.append({
                "dataset": result["dataset"],
                "v55_M2_plus_repair_current_v58_nll": result["weighted_nll"],
                "v55_M2_plus_repair_current_v58_ppl": result["ppl"],
                "v56_M1_plus_repair_historical_v56_nll": ref["nll"],
                "v56_M1_plus_repair_historical_v56_ppl": ref["ppl"],
                "absolute_nll_difference_v55_minus_v56": result["weighted_nll"] - ref["nll"],
                "relative_ppl_change_v55_vs_v56_percent": (result["ppl"] / ref["ppl"] - 1.0) * 100.0,
                "strict_same_runtime_comparison_admissible": False,
                "reason": "v56 successful evaluation process lacks Python/Transformers runtime version records",
                "v56_value_role": "context_only_not_a_confirmed_same_runtime_control",
            })
        with (OUT / "UNIFIED_COMPARISON.csv").open("w", encoding="utf8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows_main[0]))
            w.writeheader(); w.writerows(rows_main)

        atomic_json(OUT / "FOUR_ARM_ADMISSIBILITY.json", {
            "strict_four_arm_table_admissible": False,
            "reason": "Historical M1/M2 use Transformers 4.49.0, v55 historical M2+repair uses PyTorch 2.5.1+cu121, and v56 current M1+repair evaluation lacks process-level Python/Transformers records. v58 is only a fresh v55 re-evaluation.",
            "v58_v55_result_can_be_reported": True,
            "historical_propagation_validity": "UNKNOWN",
            "descriptive_interaction_calculated": False,
        })

        total_wall = elapsed_seconds()
        atomic_json(OUT / "COST_ACCOUNTING.json", {
            "cap_seconds": CAP_SECONDS,
            "total_charged_active_seconds": total_wall,
            "pre_run_read_only_audit_conservative_seconds": STATE["pre_run_audit_charge_seconds"],
            "process_wall_including_loader_and_evaluation": time.time() - float(STATE["active_segment_started_unix"]),
            "model_load_synchronized_wall_seconds": load_seconds,
            "dataset_eval_wall_seconds_excluding_load": {r["dataset"]: r["eval_wall_seconds_excluding_model_load"] for r in results},
            "finish_reserve_seconds": FINISH_RESERVE_SECONDS,
            "historical_budgets_reset": False,
        })
        end_budget_segment("COMPLETE", None)
        atomic_json(OUT / "RUN_STATUS.json", {
            "status": "COMPLETE_V55_ACTUAL_EVALUATION",
            "evaluations_requested": 2, "evaluations_completed": 2,
            "checkpoint_sha256": checkpoint_sha,
            "P3_read": False, "quantization_or_repair_run": False, "new_calibration_run": False,
            "strict_v56_same_runtime_comparison": False,
            "all_window_files_independently_reaggregated": True,
            "charged_active_seconds": total_wall,
            "budget_cap_seconds": CAP_SECONDS,
            "remaining_seconds": max(0.0, CAP_SECONDS - total_wall),
            "history_budgets_reset": False,
        })
        print("V58_COMPLETE", json.dumps({"elapsed": total_wall, "results": [{"dataset": r["dataset"], "nll": r["weighted_nll"], "ppl": r["ppl"], "tokens": r["valid_tokens"], "windows": r["windows"]} for r in results]}, ensure_ascii=False), flush=True)
    except Exception as e:
        end_budget_segment("FAILED_OR_PARTIAL", f"{type(e).__name__}: {e}")
        atomic_json(OUT / "RUN_STATUS.json", {
            "status": "FAILED_OR_PARTIAL", "error": f"{type(e).__name__}: {e}",
            "charged_active_seconds": elapsed_seconds(), "budget_cap_seconds": CAP_SECONDS,
            "history_budgets_reset": False,
        })
        raise

if __name__ == "__main__":
    main()
