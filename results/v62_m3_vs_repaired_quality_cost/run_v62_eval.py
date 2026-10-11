from __future__ import annotations

import csv, gc, hashlib, importlib.util, json, math, os, platform, sys, threading, time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from transformers import AutoConfig, AutoModelForCausalLM

ROOT = Path(r"E:\quant-thesis")
OUT = ROOT / "results/v62_m3_vs_repaired_quality_cost"
V59_DIR = ROOT / "experiments/range_search_w4/results/v59_same_process_four_checkpoint_eval"
V58_DIR = ROOT / "experiments/range_search_w4/results/v58_m2_repaired_unified_eval"
V61_DIR = ROOT / "results/v61_m3_calibration_cost_memory"
V55_DIR = ROOT / "experiments/range_search_w4/results/v55_m2_decoder_fixed_integer_repair"
M3_DIR = V61_DIR / "M3/dense_export"
R2_DIR = V55_DIR / "R1/checkpoint/final"
CAP_SECONDS = 5400.0
CPU_PRECHECK_CHARGE_SECONDS = 180.0
RESERVE_SECONDS = 360.0
MODEL_REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"
EXPECTED_CONFIG = "a33901fd787a1385efbdd5574d88012774c97110064a3f6a2cb05d070cba9dd2"
ORDER = ["v61_M3", "v55_M2_plus_repair"]
DATASETS = ["wikitext", "c4_fixed_budget"]

def utc() -> str:
    return datetime.now(timezone.utc).isoformat()

def sha_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

def sha_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()

def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)

def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys(k for r in rows for k in r))
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader(); w.writerows(rows)
    tmp.replace(path)

def load_v59_module():
    path = V59_DIR / "run_v59_eval.py"
    spec = importlib.util.spec_from_file_location("_v59_frozen_eval_helpers", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot import frozen v59 evaluation helpers")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.OUT = OUT
    mod.CAP_SECONDS = CAP_SECONDS
    mod.FINISH_RESERVE_SECONDS = RESERVE_SECONDS
    return mod

class MemorySampler:
    def __init__(self, interval=0.5):
        self.interval = interval
        self.stop_event = threading.Event()
        self.rows: list[dict] = []
        self.phase = "PRELOAD"
        self.t0 = time.perf_counter()
        self.thread = threading.Thread(target=self._run, name="v62-memory-sampler", daemon=True)
        try:
            import psutil
            self.proc = psutil.Process(os.getpid())
            self.psutil_status = "AVAILABLE"
        except Exception as e:
            self.proc = None
            self.psutil_status = f"NOT_AVAILABLE:{type(e).__name__}"

    def start(self):
        self.thread.start()

    def _run(self):
        while not self.stop_event.is_set():
            stamp = time.perf_counter()
            row = {"elapsed_seconds": stamp - self.t0, "timestamp_utc": utc(), "phase": self.phase,
                   "pid": os.getpid(), "process_rss_bytes": None, "device_used_bytes": None,
                   "device_total_bytes": None, "device_sample_method": "torch.cuda.mem_get_info; global device, includes other processes"}
            try:
                if self.proc is not None:
                    row["process_rss_bytes"] = int(self.proc.memory_info().rss)
            except Exception as e:
                row["process_rss_error"] = type(e).__name__
            try:
                free, total = torch.cuda.mem_get_info(0)
                row["device_used_bytes"] = int(total - free)
                row["device_total_bytes"] = int(total)
            except Exception as e:
                row["device_error"] = type(e).__name__
            self.rows.append(row)
            self.stop_event.wait(self.interval)

    def close(self):
        self.stop_event.set()
        self.thread.join(timeout=2)

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if not (OUT / "CPU_PRECHECK.json").is_file():
        raise RuntimeError("CPU preflight is missing")
    pre = json.loads((OUT / "CPU_PRECHECK.json").read_text(encoding="utf-8"))
    if pre.get("status") != "PASS" or pre.get("forecast", {}).get("decision") != "WITHIN_BUDGET":
        raise RuntimeError("CPU preflight did not authorize proceeding within the frozen budget")
    if (OUT / "RUN_STATUS.json").exists() and json.loads((OUT / "RUN_STATUS.json").read_text(encoding="utf-8")).get("status") == "COMPLETE":
        raise RuntimeError("complete v62 output is immutable; refusing rerun")

    v59 = load_v59_module()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    runtime = v59.stable_runtime_snapshot()
    old_env = json.loads((V59_DIR / "PROCESS_ENVIRONMENT.json").read_text(encoding="utf-8"))
    compare_keys = ["python", "pytorch", "cuda_runtime", "transformers", "tokenizers", "gpu", "default_dtype", "threads",
                    "tf32_matmul", "tf32_cudnn", "fp32_matmul_precision", "fp16_reduced_precision_reduction",
                    "cudnn_benchmark", "deterministic_algorithms", "autocast_cuda_enabled", "autocast_cpu_enabled",
                    "model_code_sha256"]
    env_diffs = {k: {"v62": runtime.get(k), "v59": old_env.get(k)} for k in compare_keys if runtime.get(k) != old_env.get(k)}
    if env_diffs:
        raise RuntimeError(f"runtime differs from v59 frozen evaluation: {env_diffs}")
    runtime.update({"attention_loader_request": "attn_implementation=sdpa", "eval_mode": True, "use_cache": False,
                    "dtype": "float16", "evaluator_core_path": str(ROOT / "benchmark/eval/ppl.py"),
                    "evaluator_core_sha256": sha_file(ROOT / "benchmark/eval/ppl.py"),
                    "evaluator_wrapper_source": str(V58_DIR / "run_v58_eval.py"),
                    "evaluator_wrapper_sha256": sha_file(V58_DIR / "run_v58_eval.py"),
                    "v59_runner_source": str(V59_DIR / "run_v59_eval.py"),
                    "v59_runner_source_sha256": sha_file(V59_DIR / "run_v59_eval.py"),
                    "runner_source": str(Path(__file__).resolve()),
                    "runner_source_sha256": sha_file(Path(__file__).resolve()),
                    "requested_model_setting": "Luna Max requested; model-selection API unavailable to process",
                    "model_code_revision": MODEL_REVISION, "tokenizer_revision": MODEL_REVISION,
                    "process_id": os.getpid(), "same_process_order": ORDER})
    if runtime["evaluator_core_sha256"] != old_env["evaluator_core_sha256"]:
        raise RuntimeError("frozen evaluator core hash changed")
    if runtime["evaluator_wrapper_sha256"] != old_env["evaluator_wrapper_sha256"]:
        raise RuntimeError("frozen v58 evaluator wrapper hash changed")
    if runtime["v59_runner_source_sha256"] != old_env["runner_source_sha256"]:
        raise RuntimeError("frozen v59 evaluation helper source changed")
    write_json(OUT / "PROCESS_ENVIRONMENT.json", runtime)

    records = pre["checkpoints"]
    named_expected = {}
    for arm in ORDER:
        rec = records[arm]
        expected = rec["checkpoint_sha256"]
        if sha_file(Path(rec["checkpoint_path"])) != expected:
            raise RuntimeError(f"checkpoint changed after CPU lock: {arm}")
        named_rows = list(csv.DictReader(Path(rec["named_tensor_manifest_csv"]).open(encoding="utf-8-sig", newline="")))
        if len(named_rows) != 290 or sha_file(Path(rec["named_tensor_manifest_csv"])) != rec["named_tensor_manifest_sha256"]:
            raise RuntimeError(f"named tensor audit changed: {arm}")
        # The saved inventories use both spellings for the same torch dtype.
        # Normalize only the metadata label consumed by v59's exact loaded-tensor verifier.
        named_expected[arm] = {r["tensor"]: {**r, "dtype": r["dtype"].replace("torch.", "")} for r in named_rows}

    # The v59 manifests and source paths are revalidated before the lock.
    manifest_records = {}
    for ds, (path, manifest_sha, token_sha, input_n, valid_n, windows) in v59.MANIFESTS.items():
        path = Path(path)
        if sha_file(path) != manifest_sha:
            raise RuntimeError(f"frozen input manifest hash changed: {ds}")
        doc = json.loads(path.read_text(encoding="utf-8"))
        ids = np.asarray(doc["token_ids"], dtype="<i8")
        if len(ids) != input_n or sha_bytes(ids.tobytes()) != token_sha or len(ids) - 1 != valid_n:
            raise RuntimeError(f"frozen token source mismatch: {ds}")
        expected_count = max(1, math.ceil((len(ids) - v59.MAX_LENGTH) / v59.STRIDE) + 1)
        if expected_count != windows:
            raise RuntimeError(f"window count mismatch: {ds}")
        manifest_records[ds] = {"path": str(path), "manifest_sha256": manifest_sha, "token_ids_sha256_i64le": token_sha,
                                "input_tokens": input_n, "valid_tokens": valid_n, "windows": windows,
                                "max_length": v59.MAX_LENGTH, "stride": v59.STRIDE,
                                "add_special_tokens": False, "chat_template": False}

    # Persist the exact input slices and score masks used by v59 for each frozen window.
    input_index_rows = []
    archive_records = {}
    for ds, meta in manifest_records.items():
        tokens = np.asarray(json.loads(Path(meta["path"]).read_text(encoding="utf-8"))["token_ids"], dtype=np.int64)
        input_rows, label_rows, lengths, begins, ends, targets, valid_counts = [], [], [], [], [], [], []
        input_hashes, label_hashes = [], []
        previous_end = 0
        for idx in range(meta["windows"]):
            begin = idx * meta["stride"]
            end = min(begin + meta["max_length"], len(tokens))
            target_length = end - previous_end
            ids = tokens[begin:end].copy()
            labels = ids.copy()
            labels[:-target_length] = -100
            valid = int(np.count_nonzero(labels[1:] != -100))
            ids_hash = sha_bytes(np.asarray(ids, dtype="<i8").tobytes())
            labels_hash = sha_bytes(np.asarray(labels, dtype="<i8").tobytes())
            input_rows.append(ids); label_rows.append(labels); lengths.append(len(ids)); begins.append(begin); ends.append(end)
            targets.append(target_length); valid_counts.append(valid); input_hashes.append(ids_hash); label_hashes.append(labels_hash)
            input_index_rows.append({"dataset": ds, "window_index": idx, "begin": begin, "end": end,
                                     "sequence_length": len(ids), "target_length": target_length, "valid_tokens": valid,
                                     "input_ids_sha256_i64le": ids_hash, "labels_sha256_i64le": labels_hash,
                                     "input_slice_source": meta["path"], "mask_source": "v59 rule: input indices < end-target_length set to -100; score logits shifted by one",
                                     "archive": str(OUT / "evaluation/input_windows" / f"{ds}.npz")})
            previous_end = end
        max_len = meta["max_length"]
        inputs_pad = np.zeros((len(input_rows), max_len), dtype=np.int64)
        labels_pad = np.full((len(label_rows), max_len), -100, dtype=np.int64)
        for i, (x, y) in enumerate(zip(input_rows, label_rows)):
            inputs_pad[i, :len(x)] = x; labels_pad[i, :len(y)] = y
        archive_path = OUT / "evaluation/input_windows" / f"{ds}.npz"
        archive_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(archive_path, input_ids=inputs_pad, labels=labels_pad,
                            lengths=np.asarray(lengths, dtype=np.int32), begins=np.asarray(begins, dtype=np.int64),
                            ends=np.asarray(ends, dtype=np.int64), target_lengths=np.asarray(targets, dtype=np.int32),
                            valid_tokens=np.asarray(valid_counts, dtype=np.int32),
                            input_ids_sha256=np.asarray(input_hashes), labels_sha256=np.asarray(label_hashes))
        archive_records[ds] = {"path": str(archive_path), "sha256": sha_file(archive_path), "windows": len(input_rows),
                              "stored_input_token_slots": int(inputs_pad.size), "stored_mask_token_slots": int(labels_pad.size),
                              "all_slices_and_labels_are_unpadded_in_length_metadata": True}
        if sum(valid_counts) != meta["valid_tokens"]:
            raise RuntimeError(f"saved mask valid token count mismatch: {ds}")
    write_csv(OUT / "evaluation/INPUT_WINDOW_SOURCE_INDEX.csv", input_index_rows)
    write_json(OUT / "evaluation/INPUT_WINDOW_ARCHIVES.json", archive_records)
    write_csv(OUT / "CHECKPOINT_SOURCE_AUDIT.csv", [
        {"arm": arm, **{k: v for k, v in records[arm].items() if k != "effective_config"},
         "model_revision": MODEL_REVISION, "tokenizer_revision": MODEL_REVISION,
         "effective_config_matches_v59": records[arm]["effective_config_sha256"] == EXPECTED_CONFIG,
         "quantization_method": "repository M3/LWC W4A16 asymmetric affine G128" if arm == "v61_M3" else "native M2 scalar10 plus fixed-grid single-coordinate repair W4A16 symmetric G128",
         "weight_bits": 4, "activation_bits": 16, "group_size": 128,
         "symmetric": False if arm == "v61_M3" else True,
         "zero_point": "asymmetric learned affine [0,15]" if arm == "v61_M3" else 8,
         "calibration_documents": 128, "calibration_tokens_per_document": 2048,
          "calibration_manifest_sha256": "664419273f63df3c0b0746b18bb66b593a5384bbe6ba8afcd6f4905c9479aae3"} for arm in ORDER])

    # Freeze before any full-model load/forward.
    runtime_sig = {k: v for k, v in runtime.items() if k not in {"process_id", "gpu_free_memory_at_process_start_bytes"}}
    runtime_sha = sha_bytes(json.dumps(runtime_sig, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8"))
    proto = {"version": "v62", "status": "FROZEN_BEFORE_FIRST_MODEL_LOAD", "requested_model_setting": "Luna Max requested; actual service selection is not exposed to process API",
             "objective": "Same-process full WT2 test and fixed-budget C4 validation comparison of new v61 repository M3/LWC checkpoint and locked v55 M2+integer-repair checkpoint",
             "prohibited": ["training", "quantization", "integer repair", "candidate search", "new calibration", "P3", "PPL-based selection"],
             "model_order": ORDER, "dataset_order_per_model": DATASETS,
             "model_revision": MODEL_REVISION, "tokenizer_revision": MODEL_REVISION,
             "model_method_boundary": "M3 is the repository M3/LWC implementation, not a demonstrated official OmniQuant reproduction. M3 asymmetric affine grid and M2+repair symmetric grid differ; outcome compares full solutions and cannot be attributed solely to LWC or integer repair.",
             "quantization_sources": {"v61_M3": "v61 frozen protocol: W4A16 asymmetric affine [0,15], G128, per-row/per-group learnable lowbound_factor/upbound_factor (LWC), LET off; exported as dense dequant FP16 weights.",
                                      "v55_M2_plus_repair": "v55 frozen protocol: formal M2 scalar10 initialization, W4A16 symmetric G128, zero=8 fixed scale/zero/g_idx with legal integer-code repair; checkpoint stored as dense FP16 dequantized model."},
             "loader": {"dtype": "FP16", "device": "CUDA", "attn_implementation": "sdpa", "local_files_only": True, "low_cpu_mem_usage": True, "eval": True, "use_cache": False, "tf32": False},
             "runtime_signature_sha256": runtime_sha, "evaluator_core_sha256": runtime["evaluator_core_sha256"],
             "evaluator_wrapper_source": str(V58_DIR / "run_v58_eval.py"), "evaluator_wrapper_sha256": runtime["evaluator_wrapper_sha256"],
             "reused_evaluation_journal_source": str(V59_DIR / "run_v59_eval.py"), "reused_v59_runner_sha256": runtime["v59_runner_source_sha256"],
             "v62_runner_source_sha256": sha_file(Path(__file__).resolve()),
             "v62_cpu_preflight_source_sha256": sha_file(OUT / "preflight_v62.py"),
             "manifest_records": manifest_records, "input_window_archives": archive_records,
             "windows_per_dataset": {"wikitext": 582, "c4_fixed_budget": 253},
             "score": "sum(window_loss * valid_tokens)/sum(valid_tokens); PPL=exp(weighted NLL)",
             "C4_label": "fixed-budget validation stream; not standard full C4", "same_process": True,
             "resume_policy": "Each attempt has an immutable run lock with unique process nonce; if interrupted, partial windows are not mixed into a later process. A restarted attempt must complete all windows for both checkpoints within that attempt or mark the interrupted item incomplete.",
             "budget_seconds": CAP_SECONDS, "cpu_precheck_charge_seconds": CPU_PRECHECK_CHARGE_SECONDS,
             "historical_calibration_propagation_unknown": ["v45", "v46", "v49"]}
    protocol_sha = sha_bytes(json.dumps(proto, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8"))
    write_json(OUT / "FROZEN_PROTOCOL.json", proto)
    forecast = dict(pre["forecast"])
    forecast["prior_cpu_active_charge_seconds"] = CPU_PRECHECK_CHARGE_SECONDS
    forecast["projected_total_including_prior_cpu_charge_seconds"] = float(forecast["projected_active_seconds"]) + CPU_PRECHECK_CHARGE_SECONDS
    forecast["remaining_margin_after_prior_cpu_charge_seconds"] = CAP_SECONDS - forecast["projected_total_including_prior_cpu_charge_seconds"]
    forecast["decision"] = "WITHIN_BUDGET" if forecast["projected_total_including_prior_cpu_charge_seconds"] < CAP_SECONDS else "STOP_BEFORE_EVALUATION"
    write_json(OUT / "COST_PRECHECK.json", {"status": "PASS_CPU_PRECHECK", "forecast": forecast,
                "current_runtime_reference": "v59 same-process evaluator; actual 8 evaluation-stage sum 2383.0349 s, per-checkpoint pair timings recorded in CPU_PRECHECK.json",
                "historical_v56_pair_seconds": 2283.576, "v56_scope": "one checkpoint WT2+C4 successful eval-only continuation in seconds, not minutes",
                "actual_GPU_evaluation_started": False, "created_before_first_model_load": True})
    write_csv(OUT / "CPU_DIAGNOSTIC_LEDGER.csv", [
        {"item": "initial_source_and_cost_audit", "charged_seconds": 30.0, "scope": "CPU/read-only file and report inspection; conservative charge; no model forward or CUDA work"},
        {"item": "failed_preflight_attempt_1", "charged_seconds": 60.0, "scope": "CPU-only metadata/hash pass; failed on absent revision field in v61 completeness JSON; no model forward/CUDA"},
        {"item": "failed_preflight_attempt_2", "charged_seconds": 10.0, "scope": "CPU-only metadata/hash pass; failed on absent revision field in runtime JSON; no model forward/CUDA"},
        {"item": "successful_preflight_command_wall", "charged_seconds": 31.1, "scope": "CPU-only process wall; Python perf_counter body 7.772 s; no model forward/CUDA"},
        {"item": "initial_precheck_buffer", "charged_seconds": 3.9, "scope": "conservative buffer; no GPU work"},
        {"item": "prior_precheck_subtotal", "charged_seconds": 135.0, "scope": "cumulative CPU-only preflight and source work before first v62 model load"},
        {"item": "failed_v62_attempt_1_preload_and_lock_work", "charged_seconds": 30.0, "scope": "CPU/preload work before the runner's in-model timer; conservative wall charge, no evaluator forwards"},
        {"item": "failed_v62_attempt_1_model_load_and_hash_check", "charged_seconds": 6.2705027, "scope": "runner wall through full M3 load and 290 tensor hashes; GPU load but no scored forward"},
        {"item": "failure_diagnosis_and_metadata_label_fix", "charged_seconds": 8.0, "scope": "CPU inspection and exact dtype-label normalization; no weights or evaluator changes"},
        {"item": "budget_rounding_buffer", "charged_seconds": CPU_PRECHECK_CHARGE_SECONDS - 179.2705027, "scope": "conservative rounding buffer"},
        {"item": "cumulative_precheck_and_failed_attempt_charge", "charged_seconds": CPU_PRECHECK_CHARGE_SECONDS, "scope": "included in v62 5400-second active budget"},
    ])
    attempt_id = datetime.now().strftime("%Y%m%d_%H%M%S") + f"_{os.getpid()}"
    attempt_dir = OUT / "attempts" / attempt_id
    attempt_dir.mkdir(parents=True, exist_ok=False)
    write_json(attempt_dir / "FROZEN_PROTOCOL.json", proto)
    run_lock = {"attempt_id": attempt_id, "process_id": os.getpid(), "protocol_sha256": protocol_sha,
                "runtime_signature_sha256": runtime_sha, "checkpoint_sha256": {a: records[a]["checkpoint_sha256"] for a in ORDER},
                "effective_config_sha256": {a: records[a]["effective_config_sha256"] for a in ORDER},
                "input_manifest_sha256": {d: manifest_records[d]["manifest_sha256"] for d in DATASETS},
                "evaluator_core_sha256": runtime["evaluator_core_sha256"], "frozen_before_first_model_load": True,
                "created_utc": utc()}
    run_lock["run_lock_sha256"] = sha_bytes(json.dumps(run_lock, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8"))
    write_json(OUT / "RUN_LOCK.json", run_lock)
    write_json(attempt_dir / "RUN_LOCK.json", run_lock)
    write_json(OUT / "RUN_STARTED.json", {"attempt_id": attempt_id, "pid": os.getpid(), "utc": utc(), "charged_preflight_seconds": CPU_PRECHECK_CHARGE_SECONDS})

    run_start = time.perf_counter()
    run_cpu_start = time.process_time()
    stage_rows, load_rows, quality_rows = [], [], []
    sampler = MemorySampler(0.5)
    sampler.start()
    v59.OUT = attempt_dir
    v59.RUN_STATE_PATH = attempt_dir / "BUDGET_STATE.json"
    v59.RUN_STATE = {"completed_active_seconds": CPU_PRECHECK_CHARGE_SECONDS, "active_start_unix": time.time()}
    def deadline_v62(reserve=RESERVE_SECONDS):
        elapsed = CPU_PRECHECK_CHARGE_SECONDS + (time.perf_counter() - run_start)
        if elapsed + reserve >= CAP_SECONDS:
            raise TimeoutError(f"v62 cap guard: elapsed={elapsed:.1f}s reserve={reserve:.1f}s cap={CAP_SECONDS:.1f}s")
    v59.deadline = deadline_v62
    eval_ppl = v59.load_eval_function()
    if eval_ppl.__module__ != "benchmark.eval.ppl":
        raise RuntimeError("unexpected evaluator core module")
    status, error = "FAILED_OR_PARTIAL", None
    all_complete = False
    try:
        for arm in ORDER:
            deadline_v62()
            sampler.phase = f"{arm}:LOAD_HASH_EVAL"
            rec = records[arm]
            model_dir = Path(rec["checkpoint_path"]).parent
            if sha_file(Path(rec["checkpoint_path"])) != rec["checkpoint_sha256"]:
                raise RuntimeError(f"checkpoint changed before load: {arm}")
            torch.cuda.reset_peak_memory_stats()
            t = time.perf_counter()
            model = AutoModelForCausalLM.from_pretrained(str(model_dir), torch_dtype=torch.float16,
                       local_files_only=True, low_cpu_mem_usage=True, attn_implementation="sdpa").eval().cuda()
            torch.cuda.synchronize()
            load_seconds = time.perf_counter() - t
            p0 = next(model.parameters())
            cfg = v59.canonical_config(model.config.to_dict())
            cfg_sha = sha_bytes(json.dumps(cfg, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8"))
            loaded_attn = getattr(model.config, "_attn_implementation", None)
            if p0.dtype != torch.float16 or p0.device.type != "cuda" or model.training or cfg_sha != EXPECTED_CONFIG or loaded_attn != "sdpa":
                raise RuntimeError(f"loaded configuration mismatch {arm}: dtype={p0.dtype}, device={p0.device}, cfg={cfg_sha}, attn={loaded_attn}")
            t = time.perf_counter()
            named = v59.verify_loaded_named_tensors(model, arm, named_expected[arm])
            named_seconds = time.perf_counter() - t
            load_record = {"arm": arm, "checkpoint_path": str(rec["checkpoint_path"]), "checkpoint_sha256": rec["checkpoint_sha256"],
                           "loaded_named_tensor_count": named["count"], "all_loaded_tensor_hashes_match": named["all_hashes_match"],
                           "effective_config_sha256": cfg_sha, "dtype": str(p0.dtype), "device": str(p0.device),
                           "attn_implementation_requested": "sdpa", "attn_implementation_loaded": loaded_attn,
                           "eval_mode": not model.training, "use_cache_eval_argument": False, "pid": os.getpid(),
                           "load_synchronized_seconds": load_seconds, "loaded_tensor_verify_seconds": named_seconds,
                           "peak_allocated_after_load_bytes": int(torch.cuda.max_memory_allocated()),
                           "peak_reserved_after_load_bytes": int(torch.cuda.max_memory_reserved())}
            load_rows.append(load_record); write_csv(OUT / "MODEL_LOAD_AUDIT.csv", load_rows)
            for ds in DATASETS:
                deadline_v62()
                meta = manifest_records[ds]
                ids_list = json.loads(Path(meta["path"]).read_text(encoding="utf-8"))["token_ids"]
                ids = torch.tensor(ids_list, dtype=torch.long).unsqueeze(0)
                journal = attempt_dir / "evaluation" / arm / ds / "windows"
                journal.mkdir(parents=True, exist_ok=True)
                archive = archive_records[ds]
                window_lock_body = {"attempt_id": attempt_id, "run_lock_sha256": run_lock["run_lock_sha256"],
                                    "arm": arm, "model_sha256": rec["checkpoint_sha256"], "effective_config_sha256": cfg_sha,
                                    "manifest_sha256": meta["manifest_sha256"], "token_ids_sha256_i64le": meta["token_ids_sha256_i64le"],
                                    "evaluator_core_sha256": runtime["evaluator_core_sha256"], "runtime_signature_sha256": runtime_sha,
                                    "dataset": ds, "max_length": meta["max_length"], "stride": meta["stride"],
                                    "dtype": "float16", "attention": "sdpa", "tf32": False, "use_cache": False,
                                    "score": "valid-token weighted NLL then exp", "input_window_archive_sha256": archive["sha256"]}
                lock_sha = sha_bytes(json.dumps(window_lock_body, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8"))
                window_lock = {**window_lock_body, "window_lock_sha256": lock_sha}
                write_json(journal.parent / "WINDOW_LOCK.json", window_lock)
                wrapper = v59.WindowJournal(model, ids, arm, ds, {"run_lock_sha256": lock_sha, "model_sha256": rec["checkpoint_sha256"],
                                              "effective_config_sha256": cfg_sha, "manifest_sha256": meta["manifest_sha256"],
                                              "token_ids_sha256_i64le": meta["token_ids_sha256_i64le"],
                                              "evaluator_core_sha256": runtime["evaluator_core_sha256"]}, journal)
                wrapper.eval(); sampler.phase = f"{arm}:{ds}:EVALUATION"
                t = time.perf_counter()
                ppl, mean_nll, valid_total = eval_ppl(wrapper, ids, max_length=meta["max_length"], stride=meta["stride"], progress_every=25)
                torch.cuda.synchronize()
                eval_seconds = time.perf_counter() - t
                if len(wrapper.rows) != meta["windows"] or int(valid_total) != meta["valid_tokens"] or wrapper.new_forwards != meta["windows"] or wrapper.cache_hits != 0:
                    raise RuntimeError(f"incomplete or mixed-process evaluation {arm}/{ds}")
                durable = [json.loads((journal / f"window_{i:04d}.json").read_text(encoding="utf-8")) for i in range(meta["windows"])]
                if [int(r["window_index"]) for r in durable] != list(range(meta["windows"])) or {int(r["process_id"]) for r in durable} != {os.getpid()}:
                    raise RuntimeError(f"window sequence/process check failed {arm}/{ds}")
                nll_sum = sum(float(r["window_nll"]) for r in durable)
                valid_sum = sum(int(r["valid_tokens"]) for r in durable)
                nll = nll_sum / valid_sum
                ppl_independent = math.exp(nll)
                if valid_sum != meta["valid_tokens"] or not math.isclose(nll, float(mean_nll), rel_tol=0, abs_tol=1e-10) or not math.isclose(ppl_independent, float(ppl), rel_tol=1e-12, abs_tol=1e-12):
                    raise ArithmeticError(f"independent aggregate mismatch {arm}/{ds}")
                index_rows = {r["window_index"]: r for r in input_index_rows if r["dataset"] == ds}
                csv_rows = []
                for r in durable:
                    ix = index_rows[int(r["window_index"])]
                    if r["input_ids_sha256"] != ix["input_ids_sha256_i64le"] or r["labels_sha256"] != ix["labels_sha256_i64le"]:
                        raise RuntimeError(f"window input/mask source hash mismatch {arm}/{ds}/{r['window_index']}")
                    csv_rows.append({k: r[k] for k in ["arm", "dataset", "window_index", "begin", "end", "target_length", "valid_tokens", "window_loss", "window_nll", "input_ids_sha256", "labels_sha256", "model_sha256", "manifest_sha256", "token_ids_sha256_i64le", "config_effective_sha256", "evaluator_core_sha256", "process_id"]})
                out_csv = OUT / "evaluation" / f"{arm}_{ds}_WINDOWS.csv"
                write_csv(out_csv, csv_rows)
                rec_result = {"status": "PASS", "arm": arm, "dataset": ds, "checkpoint_sha256": rec["checkpoint_sha256"],
                              "effective_config_sha256": cfg_sha, "manifest_sha256": meta["manifest_sha256"],
                              "token_ids_sha256_i64le": meta["token_ids_sha256_i64le"], "input_tokens": meta["input_tokens"],
                              "valid_tokens": valid_sum, "windows": len(durable), "nll_sum": nll_sum,
                              "weighted_nll": nll, "ppl": ppl_independent, "max_length": meta["max_length"], "stride": meta["stride"],
                              "evaluator_core_sha256": runtime["evaluator_core_sha256"], "runtime_signature_sha256": runtime_sha,
                              "input_window_archive": archive["path"], "input_window_archive_sha256": archive["sha256"],
                              "eval_seconds": eval_seconds, "new_forwards": wrapper.new_forwards, "resumed_cache_hits": wrapper.cache_hits,
                              "window_process_ids": [os.getpid()], "independent_weighted_recalculation_pass": True}
                write_json(OUT / "evaluation" / f"{arm}_{ds}.json", rec_result)
                quality_rows.append(rec_result)
                stage_rows.append({"arm": arm, "dataset": ds, "stage": "evaluation", "seconds": eval_seconds,
                                   "windows": len(durable), "new_forwards": wrapper.new_forwards, "resume_hits": wrapper.cache_hits,
                                   "pid": os.getpid()})
                write_csv(OUT / "STAGE_TIMINGS.csv", stage_rows)
                write_json(attempt_dir / "RUN_PROGRESS.json", {"status": "IN_PROGRESS", "attempt_id": attempt_id,
                           "completed_evaluations": len(quality_rows), "target_evaluations": 4, "last_arm": arm,
                           "last_dataset": ds, "completed_windows_last_item": len(durable),
                           "elapsed_including_preflight_seconds": CPU_PRECHECK_CHARGE_SECONDS + time.perf_counter() - run_start,
                           "process_id": os.getpid(), "updated_utc": utc()})
                del wrapper, ids, ids_list
            load_record["peak_allocated_after_pair_bytes"] = int(torch.cuda.max_memory_allocated())
            load_record["peak_reserved_after_pair_bytes"] = int(torch.cuda.max_memory_reserved())
            load_record["rss_sampled_peak_bytes"] = max((r["process_rss_bytes"] or 0) for r in sampler.rows if arm in r["phase"])
            load_record["device_sampled_peak_bytes_global"] = max((r["device_used_bytes"] or 0) for r in sampler.rows if arm in r["phase"])
            write_csv(OUT / "MODEL_LOAD_AUDIT.csv", load_rows)
            sampler.phase = f"{arm}:UNLOAD"
            t = time.perf_counter(); del model; model = None; gc.collect(); torch.cuda.synchronize(); torch.cuda.empty_cache()
            stage_rows.append({"arm": arm, "dataset": "", "stage": "unload_gc_empty_cache", "seconds": time.perf_counter()-t,
                               "windows": 0, "new_forwards": 0, "resume_hits": 0, "pid": os.getpid()})
            write_csv(OUT / "STAGE_TIMINGS.csv", stage_rows)

        if len(quality_rows) != 4:
            raise RuntimeError(f"only {len(quality_rows)} of 4 full evaluations completed")
        # Final aggregation uses only the four complete CSVs from this run.
        comparison = []
        for ds in DATASETS:
            a = next(r for r in quality_rows if r["arm"] == ORDER[0] and r["dataset"] == ds)
            b = next(r for r in quality_rows if r["arm"] == ORDER[1] and r["dataset"] == ds)
            delta = float(a["weighted_nll"]) - float(b["weighted_nll"])
            ppl_rel = (float(a["ppl"]) / float(b["ppl"]) - 1.0) * 100.0
            comparison.append({"dataset": ds, "scope": "v62_same_process_primary", "v61_m3_nll": a["weighted_nll"],
                               "v61_m3_ppl": a["ppl"], "v55_m2_repair_nll": b["weighted_nll"],
                               "v55_m2_repair_ppl": b["ppl"], "delta_nll_v61_minus_v55": delta,
                               "absolute_nll_difference": abs(delta), "relative_ppl_change_v61_vs_v55_percent": ppl_rel,
                               "valid_tokens": a["valid_tokens"], "windows": a["windows"],
                               "better_nll": "v61_M3" if delta < 0 else ("v55_M2_plus_repair" if delta > 0 else "tie")})
        write_csv(OUT / "QUALITY_COMPARISON.csv", comparison)
        # v59 M1/M2 values are context only, never mixed into the main same-run comparison.
        hist = list(csv.DictReader((V59_DIR / "INDEPENDENT_WINDOW_RECALC.csv").open(encoding="utf-8-sig", newline="")))
        bg = [{**r, "scope": "v59 historical background, not v62 same-process primary"} for r in hist if r["arm"] in {"M1", "M2"}]
        write_csv(OUT / "V59_M1_M2_HISTORICAL_BACKGROUND.csv", bg)
        all_complete = True
        status = "COMPLETE"
    except Exception as e:
        error = f"{type(e).__name__}: {e}"
        raise
    finally:
        sampler.close()
        write_csv(OUT / "MEMORY_TIMELINE.csv", sampler.rows)
        write_csv(OUT / "attempts" / attempt_id / "MEMORY_TIMELINE.csv", sampler.rows)
        cpu_seconds = time.process_time() - run_cpu_start
        active_run = time.perf_counter() - run_start
        v62_total = CPU_PRECHECK_CHARGE_SECONDS + active_run
        mem_rows = []
        for arm in ORDER:
            arm_load = next((r for r in load_rows if r["arm"] == arm), {})
            arm_samples = [r for r in sampler.rows if arm in r["phase"]]
            mem_rows.append({"arm": arm, "scope": "evaluation process; current run", "pytorch_peak_allocated_bytes": arm_load.get("peak_allocated_after_pair_bytes", "NOT_MEASURED"),
                             "pytorch_peak_reserved_bytes": arm_load.get("peak_reserved_after_pair_bytes", "NOT_MEASURED"),
                             "device_sampled_peak_bytes_global": max((r["device_used_bytes"] or 0) for r in arm_samples) if arm_samples else "NOT_MEASURED",
                             "device_memory_scope": "global GPU used; includes desktop/other processes; sampled every 0.5s; not exact instantaneous peak",
                             "process_rss_sampled_peak_bytes": max((r["process_rss_bytes"] or 0) for r in arm_samples) if arm_samples else "NOT_MEASURED",
                             "per_process_gpu_memory": "NOT_AVAILABLE: WDDM process attribution unavailable"})
        write_csv(OUT / "QUALITY_COST_MEMORY_TABLE.csv", mem_rows)
        write_json(OUT / "attempts" / attempt_id / "ATTEMPT_ACCOUNTING.json", {"attempt_id": attempt_id, "status": status,
                    "error": error, "pid": os.getpid(), "cpu_process_seconds": cpu_seconds,
                    "active_wall_seconds_after_runner_entry": active_run, "cpu_precheck_charged_seconds": CPU_PRECHECK_CHARGE_SECONDS,
                    "cumulative_v62_active_seconds": v62_total, "cap_seconds": CAP_SECONDS,
                    "remaining_seconds": max(0.0, CAP_SECONDS-v62_total), "completed_evaluations": len(quality_rows),
                    "evaluation_stage_seconds": sum(float(r["seconds"]) for r in stage_rows if r["stage"] == "evaluation"),
                    "memory_samples": len(sampler.rows), "memory_sample_interval_seconds": sampler.interval,
                    "rss_sampling_status": sampler.psutil_status, "device_memory_sampling": "torch.cuda.mem_get_info; global, no process attribution"})
        write_csv(OUT / "STAGE_TIMINGS.csv", stage_rows)
        write_json(OUT / "RUN_STATUS.json", {"status": status if all_complete else "FAILED_OR_PARTIAL", "error": error,
                    "complete_evaluations": len(quality_rows), "target_evaluations": 4,
                    "same_process": len({pid for r in quality_rows for pid in r.get("window_process_ids", [])}) <= 1,
                    "process_ids": [os.getpid()], "charged_active_seconds_including_preflight": v62_total,
                    "budget_cap_seconds": CAP_SECONDS, "remaining_seconds": max(0.0,CAP_SECONDS-v62_total),
                    "P3_read": False, "models_modified": False, "updated_utc": utc()})
        write_json(OUT / "COST_ACCOUNTING.json", {"cap_seconds": CAP_SECONDS, "cpu_precheck_and_prior_diagnostics_charged_seconds": CPU_PRECHECK_CHARGE_SECONDS,
                    "runner_wall_seconds": active_run, "runner_cpu_process_seconds": cpu_seconds,
                    "charged_v62_active_seconds": v62_total, "remaining_seconds": max(0.0,CAP_SECONDS-v62_total),
                    "model_load_seconds": {r["arm"]: r["load_synchronized_seconds"] for r in load_rows},
                    "loaded_named_hash_seconds": {r["arm"]: r["loaded_tensor_verify_seconds"] for r in load_rows},
                    "evaluation_seconds": {f"{r['arm']}/{r['dataset']}": r["seconds"] for r in stage_rows if r["stage"] == "evaluation"},
                    "unload_seconds": {r["arm"]: r["seconds"] for r in stage_rows if r["stage"] == "unload_gc_empty_cache"},
                    "evaluation_stage_sum_seconds": sum(float(r["seconds"]) for r in stage_rows if r["stage"] == "evaluation"),
                    "setup_io_finalize_seconds_unassigned_within_runner": max(0.0, active_run - sum(float(r["seconds"]) for r in stage_rows)),
                    "v59_pair_timing_reference_seconds": pre["v59_pair_times"],
                    "v61_calibration_cost_separate_seconds": 2384.344,
                    "v55_effective_repair_cost_separate_seconds": 4240.101995,
                    "failures_and_retries": "see CPU_DIAGNOSTIC_LEDGER.csv and attempts/*/ATTEMPT_ACCOUNTING.json; no scoring resumes across process ids",
                    "historical_ledgers_reset": False, "GPU scoring activity present": bool(quality_rows)})
        if error:
            print(f"V62_PARTIAL {error}", flush=True)
        elif all_complete:
            print(json.dumps({"status": "COMPLETE", "pid": os.getpid(), "quality": comparison,
                              "runner_wall_seconds": active_run, "charged_active_seconds": v62_total,
                              "evaluation_stage_seconds": sum(float(r["seconds"]) for r in stage_rows if r["stage"] == "evaluation")}, ensure_ascii=False), flush=True)

if __name__ == "__main__":
    main()
