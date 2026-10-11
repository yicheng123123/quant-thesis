from __future__ import annotations

import csv
import gc
import hashlib
import json
import math
import os
import sys
import time
from contextlib import redirect_stdout
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
V18_PROTOCOL_PATH = HERE / "V18_EVALUATION_PROTOCOL.json"
PROTOCOL = json.loads(V18_PROTOCOL_PATH.read_text(encoding="utf-8"))
DEVICE = torch.device("cuda:0")
MAX_LENGTH = int(PROTOCOL["evaluation"]["max_length"])
STRIDE = int(PROTOCOL["evaluation"]["stride"])
EVAL_DIR = HERE / "evaluation"
RAW_DIR = EVAL_DIR / "raw"
WINDOW_DIR = EVAL_DIR / "windows"
LOG_DIR = EVAL_DIR / "logs"
EXPORT_DIR = HERE / "dense_export"
for directory in (RAW_DIR, WINDOW_DIR, LOG_DIR):
    directory.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT))
from benchmark.eval.ppl import evaluate_ppl


def sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json_atomic(path: Path, value: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


class WindowCaptureModel(torch.nn.Module):
    def __init__(self, inner, token_count: int):
        super().__init__()
        self.inner = inner
        self.token_count = token_count
        self.rows: list[dict] = []

    def forward(self, *, input_ids, labels, use_cache=False, **kwargs):
        outputs = self.inner(input_ids=input_ids, labels=labels, use_cache=use_cache, **kwargs)
        index = len(self.rows)
        begin = index * STRIDE
        end = min(begin + MAX_LENGTH, self.token_count)
        previous_end = self.rows[-1]["end"] if self.rows else 0
        valid = int((labels[:, 1:] != -100).sum().item())
        loss = float(outputs.loss.float().item())
        self.rows.append({"window_index": index, "begin": begin, "end": end,
                          "new_targets": end - previous_end, "valid_tokens": valid,
                          "window_loss": loss, "window_nll": loss * valid})
        return outputs


def load_dataset_manifest(name: str):
    if name == "wikitext":
        path = EVAL_DIR / "manifests" / "WIKITEXT_TEST_MANIFEST.json"
        expected_tokens = int(PROTOCOL["evaluation"]["wiki_text_test_input_tokens"])
        expected_scored = int(PROTOCOL["evaluation"]["wiki_text_test_scored_tokens"])
        expected_hash = PROTOCOL["source_protocol"]["v16_wikitext_manifest_sha256"]
        label = "WikiText-2 test"
    elif name == "c4_fixed_budget":
        path = EVAL_DIR / "manifests" / "C4_EVAL_MANIFEST.json"
        expected_tokens = int(PROTOCOL["evaluation"]["c4_validation_fixed_budget_input_tokens"])
        expected_scored = int(PROTOCOL["evaluation"]["c4_validation_fixed_budget_scored_tokens"])
        expected_hash = PROTOCOL["source_protocol"]["v16_c4_manifest_sha256"]
        label = "C4 validation fixed token budget"
    else:
        raise ValueError(name)
    data = path.read_bytes()
    manifest_hash = sha_bytes(data)
    if manifest_hash != expected_hash:
        raise RuntimeError(f"frozen {name} evaluation manifest hash mismatch")
    manifest = json.loads(data)
    if manifest.get("tokenizer_revision") != PROTOCOL["model"]["revision"]:
        raise RuntimeError(f"frozen {name} tokenizer revision mismatch")
    ids = manifest["token_ids"]
    if len(ids) != expected_tokens or expected_scored != expected_tokens - 1:
        raise RuntimeError(f"frozen {name} token accounting mismatch")
    token_hash = sha_bytes(np.asarray(ids, dtype="<i8").tobytes())
    return path, manifest, torch.tensor(ids, dtype=torch.long)[None], token_hash, label, expected_scored


def model_weights_hash(source: Path):
    files = sorted(source.glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"no dense safetensors weights found under {source}")
    digest = hashlib.sha256()
    details = []
    for path in files:
        file_hash = sha_file(path)
        details.append({"name": path.name, "bytes": path.stat().st_size, "sha256": file_hash})
        digest.update(path.name.encode("utf-8"))
        digest.update(bytes.fromhex(file_hash))
    return digest.hexdigest(), details


def run_one(dataset: str):
    manifest_path, dataset_manifest, token_ids, token_hash, label, expected_scored = load_dataset_manifest(dataset)
    if not (EXPORT_DIR / "config.json").is_file():
        raise FileNotFoundError(f"dense M3 export is absent: {EXPORT_DIR}")
    model_hash, weight_files = model_weights_hash(EXPORT_DIR)
    raw_path = RAW_DIR / f"M3_{dataset}.json"
    if raw_path.is_file():
        old = json.loads(raw_path.read_text(encoding="utf-8"))
        if old.get("model_weights_sha256") != model_hash or old.get("evaluation_token_ids_sha256_i64le") != token_hash:
            raise RuntimeError(f"existing M3/{dataset} result identity mismatch; preserving result and stopping")
        return old

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.synchronize(DEVICE)
    torch.cuda.reset_peak_memory_stats(DEVICE)
    load_started = time.perf_counter()
    base = AutoModelForCausalLM.from_pretrained(
        str(EXPORT_DIR), torch_dtype=torch.float16, low_cpu_mem_usage=True,
        local_files_only=True, attn_implementation="sdpa",
    ).eval().to(DEVICE)
    torch.cuda.synchronize(DEVICE)
    model_load_seconds = time.perf_counter() - load_started
    capture = WindowCaptureModel(base, int(token_ids.shape[1])).eval()
    log_path = LOG_DIR / f"M3_{dataset}.log"
    started_utc = datetime.now(timezone.utc).isoformat()
    evaluation_started = time.perf_counter()
    with log_path.open("w", encoding="utf-8", newline="") as log, redirect_stdout(log):
        ppl, average_nll, evaluated_tokens = evaluate_ppl(
            capture, token_ids, max_length=MAX_LENGTH, stride=STRIDE, progress_every=25)
    torch.cuda.synchronize(DEVICE)
    evaluation_wall = time.perf_counter() - evaluation_started
    peak_allocated = int(torch.cuda.max_memory_allocated(DEVICE))
    peak_reserved = int(torch.cuda.max_memory_reserved(DEVICE))
    rows = capture.rows
    if evaluated_tokens != expected_scored or sum(row["valid_tokens"] for row in rows) != expected_scored:
        raise RuntimeError(f"M3/{dataset} scored token count mismatch: {evaluated_tokens}/{expected_scored}")
    expected_windows = max(1, math.ceil((int(token_ids.shape[1]) - MAX_LENGTH) / STRIDE) + 1)
    if len(rows) != expected_windows:
        raise RuntimeError(f"M3/{dataset} window count mismatch: {len(rows)}/{expected_windows}")
    exact_nll = sum(row["window_nll"] for row in rows) / evaluated_tokens
    if not all(math.isfinite(x) for x in (ppl, average_nll, exact_nll)):
        raise RuntimeError(f"M3/{dataset} aggregate is nonfinite")
    if abs(average_nll - exact_nll) > 1e-10 or abs(ppl - math.exp(average_nll)) > 1e-10:
        raise RuntimeError(f"M3/{dataset} aggregate differs from exact window-weighted evaluator output")

    window_path = WINDOW_DIR / f"M3_{dataset}.csv"
    with window_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["method", "dataset", "window_index", "begin", "end",
            "new_targets", "valid_tokens", "window_loss", "window_nll"])
        writer.writeheader()
        for row in rows:
            writer.writerow({"method": "M3", "dataset": label, **row})
    result = {
        "status": "PASS", "method": "M3", "dataset": dataset, "dataset_label": label,
        "representation": "dense-dequant FP16 OmniQuant W4A16/G128",
        "model_source": str(EXPORT_DIR), "model_weights_sha256": model_hash,
        "model_weight_files": weight_files, "model_revision": PROTOCOL["model"]["revision"],
        "evaluation_manifest": str(manifest_path.relative_to(HERE)),
        "evaluation_manifest_sha256": sha_file(manifest_path),
        "dataset_tokenizer_revision": dataset_manifest["tokenizer_revision"],
        "evaluation_token_ids_sha256_i64le": token_hash,
        "input_tokens": int(token_ids.shape[1]), "evaluated_tokens": evaluated_tokens,
        "max_length": MAX_LENGTH, "stride": STRIDE,
        "add_special_tokens": False, "chat_template": False,
        "window_count": len(rows), "window_results": rows,
        "perplexity": ppl, "average_nll": average_nll,
        "exact_window_weighted_average_nll": exact_nll,
        "model_load_seconds": model_load_seconds, "evaluation_wall_seconds": evaluation_wall,
        "peak_gpu_allocated_bytes": peak_allocated, "peak_gpu_reserved_bytes": peak_reserved,
        "torch_version": torch.__version__, "transformers_version": __import__("transformers").__version__,
        "device": torch.cuda.get_device_name(0), "started_utc": started_utc,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "window_csv": str(window_path.relative_to(HERE)), "log": str(log_path.relative_to(HERE)),
    }
    write_json_atomic(raw_path, result)
    print(f"[eval:done] M3 {dataset}: PPL={ppl:.8f} NLL={average_nll:.9f} tokens={evaluated_tokens} windows={len(rows)} seconds={evaluation_wall:.1f}", flush=True)
    del capture, base, token_ids
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main() -> int:
    training = json.loads((HERE / "M3_TRAINING_SUMMARY.json").read_text(encoding="utf-8"))
    reload_check = json.loads((HERE / "DENSE_EXPORT_RELOAD_CHECK.json").read_text(encoding="utf-8"))
    if training.get("status") != "PASS" or training.get("effective_optimizer_updates") != 61440:
        raise RuntimeError("formal M3 training/checkpoint gate did not pass")
    if reload_check.get("status") != "PASS":
        raise RuntimeError("dense export/reload correctness did not pass")
    evaluator_path = ROOT / "benchmark" / "eval" / "ppl.py"
    evaluator_hash = sha_file(evaluator_path)
    frozen_copy_hash = sha_file(ROOT / "experiments" / "range_search_w4" / "results" /
                                "range_w4_omniquant_kaggle_v20_1" / "kaggle_bundle" / "evaluator" / "ppl.py")
    if evaluator_hash != frozen_copy_hash:
        raise RuntimeError("shared evaluator differs from the byte-identical v20.1 frozen evaluator copy")
    write_json_atomic(HERE / "EVALUATION_PROTOCOL.json", {
        "status": "FROZEN_FOR_M3_EVALUATION",
        "source_protocol_path": str((HERE / "V18_EVALUATION_PROTOCOL.json").relative_to(ROOT)),
        "source_protocol_sha256": sha_file(V18_PROTOCOL_PATH),
        "scope": "Reuse v18 frozen WT2/C4 token-ID manifests and sliding-window evaluator semantics only; v18 quantization settings are not adopted for M3.",
        "evaluator": "benchmark.eval.ppl.evaluate_ppl",
        "evaluator_sha256": evaluator_hash,
        "v20_1_frozen_copy_sha256": frozen_copy_hash,
        "wikitext_manifest_sha256": PROTOCOL["source_protocol"]["v16_wikitext_manifest_sha256"],
        "c4_manifest_sha256": PROTOCOL["source_protocol"]["v16_c4_manifest_sha256"],
        "max_length": MAX_LENGTH, "stride": STRIDE,
        "wikitext_input_tokens": PROTOCOL["evaluation"]["wiki_text_test_input_tokens"],
        "wikitext_scored_targets": PROTOCOL["evaluation"]["wiki_text_test_scored_tokens"],
        "c4_fixed_budget_input_tokens": PROTOCOL["evaluation"]["c4_validation_fixed_budget_input_tokens"],
        "c4_fixed_budget_scored_targets": PROTOCOL["evaluation"]["c4_validation_fixed_budget_scored_tokens"],
        "tf32": False, "dtype": "FP16", "attention": "SDPA",
        "selection_or_tuning_on_final_evaluation": False,
    })
    status_path = HERE / "M3_EVALUATION_STATUS.json"
    write_json_atomic(status_path, {"status": "EVALUATION_RUNNING", "started_utc": datetime.now(timezone.utc).isoformat()})
    results = {}
    try:
        for dataset in ("wikitext", "c4_fixed_budget"):
            results[dataset] = run_one(dataset)
            write_json_atomic(status_path, {"status": f"{dataset.upper()}_PASS", "results_completed": list(results),
                                            "updated_utc": datetime.now(timezone.utc).isoformat()})
        result = {
            "status": "PASS", "M3_WT2": results["wikitext"]["perplexity"],
            "M3_C4_fixed_budget": results["c4_fixed_budget"]["perplexity"],
            "M3_WT2_NLL": results["wikitext"]["average_nll"],
            "M3_C4_fixed_budget_NLL": results["c4_fixed_budget"]["average_nll"],
            "WT2_scored_targets": results["wikitext"]["evaluated_tokens"],
            "C4_fixed_budget_scored_targets": results["c4_fixed_budget"]["evaluated_tokens"],
            "results": results,
        }
        write_json_atomic(EVAL_DIR / "M3_SUMMARY.json", result)
        write_json_atomic(EVAL_DIR / "M3_EVAL_CORRECTNESS.json", {
            "status": "PASS", "all_token_counts_match_frozen_manifests": True,
            "all_window_weighted_nll_matches_shared_evaluator": True,
            "all_aggregates_finite": True,
            "wikitext_windows": results["wikitext"]["window_count"],
            "c4_windows": results["c4_fixed_budget"]["window_count"],
        })
        with (EVAL_DIR / "M3_RESULTS.csv").open("w", newline="", encoding="utf-8") as stream:
            fields = ["dataset", "method", "representation", "evaluated_tokens", "window_count", "average_nll", "perplexity", "evaluation_wall_seconds", "peak_gpu_allocated_bytes", "peak_gpu_reserved_bytes"]
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for item in results.values():
                writer.writerow({"dataset": item["dataset_label"], "method": "M3", "representation": item["representation"],
                                 "evaluated_tokens": item["evaluated_tokens"], "window_count": item["window_count"],
                                 "average_nll": item["average_nll"], "perplexity": item["perplexity"],
                                 "evaluation_wall_seconds": item["evaluation_wall_seconds"],
                                 "peak_gpu_allocated_bytes": item["peak_gpu_allocated_bytes"],
                                 "peak_gpu_reserved_bytes": item["peak_gpu_reserved_bytes"]})
        write_json_atomic(status_path, {"status": "PASS", "results_completed": list(results),
                                        "completed_utc": datetime.now(timezone.utc).isoformat()})
        print(json.dumps({"status": "PASS", "M3_WT2": result["M3_WT2"], "M3_C4_fixed_budget": result["M3_C4_fixed_budget"]}, indent=2), flush=True)
        return 0
    except BaseException as exc:
        write_json_atomic(status_path, {"status": "FAILED_OR_INTERRUPTED", "results_completed": list(results),
                                        "exception_type": type(exc).__name__, "message": str(exc),
                                        "updated_utc": datetime.now(timezone.utc).isoformat()})
        raise


if __name__ == "__main__":
    raise SystemExit(main())
