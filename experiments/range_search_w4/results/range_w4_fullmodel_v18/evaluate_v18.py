"""Frozen v18 M0/M1/M2 model-level WT2/C4 evaluation through shared evaluator."""
from __future__ import annotations

import csv
import gc
import hashlib
import json
import math
import os
import re
import sys
import time
from contextlib import redirect_stdout
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM

# This standalone script lives several directories below the repository root.
# Make the shared benchmark package importable when invoked by file path.
REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))
from benchmark.eval.ppl import evaluate_ppl


OUT = Path(__file__).resolve().parent
ROOT = REPO_ROOT
PROTOCOL = json.loads((OUT / "PROTOCOL.json").read_text(encoding="utf-8"))
DEVICE = torch.device("cuda")
MAX_LENGTH = int(PROTOCOL["evaluation"]["max_length"])
STRIDE = int(PROTOCOL["evaluation"]["stride"])
EVAL_DIR = OUT / "evaluation"
RAW_DIR = EVAL_DIR / "raw"
WINDOW_DIR = EVAL_DIR / "windows"
LOG_DIR = EVAL_DIR / "logs"
RAW_DIR.mkdir(parents=True, exist_ok=True)
WINDOW_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)


def sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_json_atomic(path: Path, value: dict):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


class WindowCaptureModel(torch.nn.Module):
    """Transparent wrapper recording exact per-window loss from the shared evaluator."""
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
        expected_manifest_hash = PROTOCOL["source_protocol"]["v16_wikitext_manifest_sha256"]
        label = "WikiText-2 test"
    elif name == "c4_fixed_budget":
        path = EVAL_DIR / "manifests" / "C4_EVAL_MANIFEST.json"
        expected_tokens = int(PROTOCOL["evaluation"]["c4_validation_fixed_budget_input_tokens"])
        expected_scored = int(PROTOCOL["evaluation"]["c4_validation_fixed_budget_scored_tokens"])
        expected_manifest_hash = PROTOCOL["source_protocol"]["v16_c4_manifest_sha256"]
        label = "C4 validation fixed token budget"
    else:
        raise ValueError(name)
    manifest_bytes = path.read_bytes()
    manifest_hash = sha_bytes(manifest_bytes)
    if manifest_hash != expected_manifest_hash:
        raise RuntimeError(f"frozen {name} manifest hash changed: {manifest_hash}")
    manifest = json.loads(manifest_bytes)
    if manifest.get("tokenizer_revision") != PROTOCOL["model"]["revision"]:
        raise RuntimeError(f"{name} tokenizer revision mismatch")
    token_ids = manifest["token_ids"]
    if len(token_ids) != expected_tokens or expected_scored != expected_tokens - 1:
        raise RuntimeError(f"{name} frozen token count mismatch")
    token_hash = sha_bytes(np.asarray(token_ids, dtype="<i8").tobytes())
    return path, manifest, torch.tensor(token_ids, dtype=torch.long)[None], token_hash, label, expected_scored


def model_source(method: str) -> Path:
    if method == "M0":
        return ROOT / ".hf_cache/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots" / PROTOCOL["model"]["revision"]
    return OUT / method / "checkpoint" / "final"


def model_weights_hash(source: Path):
    weight_files = sorted(source.glob("*.safetensors"))
    if not weight_files:
        raise FileNotFoundError(f"No safetensors weights at {source}")
    h = hashlib.sha256()
    details = []
    for path in weight_files:
        digest = sha_file(path)
        details.append({"name": path.name, "bytes": path.stat().st_size, "sha256": digest})
        h.update(path.name.encode("utf-8")); h.update(bytes.fromhex(digest))
    return h.hexdigest(), details


def already_completed(method: str, dataset: str, model_hash: str, token_hash: str):
    path = RAW_DIR / f"{method}_{dataset}.json"
    if not path.exists():
        return None
    existing = json.loads(path.read_text(encoding="utf-8"))
    if existing.get("model_weights_sha256") != model_hash or existing.get("evaluation_token_ids_sha256_i64le") != token_hash:
        raise RuntimeError(f"existing result identity mismatch; preserving it and stopping: {path}")
    return existing


def run_one(method: str, dataset: str):
    manifest_path, dataset_manifest, token_ids, token_hash, label, expected_scored = load_dataset_manifest(dataset)
    source = model_source(method)
    if not (source / "config.json").exists():
        raise FileNotFoundError(f"model checkpoint is absent: {source}")
    model_hash, weight_files = model_weights_hash(source)
    cached = already_completed(method, dataset, model_hash, token_hash)
    if cached is not None:
        print(f"[eval:skip] {method} {dataset} already complete with matching frozen identity", flush=True)
        return cached

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.cuda.empty_cache(); gc.collect()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    load_start = time.perf_counter()
    base = AutoModelForCausalLM.from_pretrained(
        str(source), torch_dtype=torch.float16, low_cpu_mem_usage=True,
        local_files_only=True, attn_implementation="sdpa",
    ).eval().to(DEVICE)
    torch.cuda.synchronize()
    load_seconds = time.perf_counter() - load_start
    capture = WindowCaptureModel(base, int(token_ids.shape[1])).eval()
    log_path = LOG_DIR / f"{method}_{dataset}.log"
    started_utc = datetime.now(timezone.utc).isoformat()
    eval_start = time.perf_counter()
    with log_path.open("w", encoding="utf-8", newline="") as log, redirect_stdout(log):
        ppl, average_nll, evaluated_tokens = evaluate_ppl(
            capture, token_ids, max_length=MAX_LENGTH, stride=STRIDE, progress_every=25)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - eval_start
    peak_alloc = int(torch.cuda.max_memory_allocated())
    peak_reserved = int(torch.cuda.max_memory_reserved())
    rows = capture.rows
    if evaluated_tokens != expected_scored or sum(r["valid_tokens"] for r in rows) != expected_scored:
        raise RuntimeError(f"{method}/{dataset} token count mismatch: {evaluated_tokens}/{expected_scored}")
    # evaluate_ppl stops as soon as the final sliding window covers the stream.
    # This is ceil((N - max_length) / stride) + 1, not ceil(N / stride).
    expected_windows = max(1, math.ceil((int(token_ids.shape[1]) - MAX_LENGTH) / STRIDE) + 1)
    if len(rows) != expected_windows:
        raise RuntimeError(f"{method}/{dataset} window count mismatch: {len(rows)}")
    exact_nll = sum(r["window_nll"] for r in rows) / evaluated_tokens
    if not (math.isfinite(ppl) and math.isfinite(average_nll) and math.isfinite(exact_nll)):
        raise RuntimeError(f"nonfinite {method}/{dataset} aggregate")
    if abs(average_nll - exact_nll) > 1e-10 or abs(ppl - math.exp(average_nll)) > 1e-10:
        raise RuntimeError(f"shared evaluator aggregation does not match captured exact windows {method}/{dataset}")

    row_path = WINDOW_DIR / f"{method}_{dataset}.csv"
    with row_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["method", "dataset", "window_index", "begin", "end",
            "new_targets", "valid_tokens", "window_loss", "window_nll"])
        writer.writeheader()
        for row in rows:
            writer.writerow({"method": method, "dataset": dataset, **row})
    result = {
        "status": "PASS", "method": method, "dataset": dataset, "dataset_label": label,
        "model_source": str(source), "model_weights_sha256": model_hash,
        "model_weight_files": weight_files,
        "model_revision": PROTOCOL["model"]["revision"],
        "evaluation_manifest": str(manifest_path.relative_to(OUT)),
        "evaluation_manifest_sha256": sha_file(manifest_path),
        "dataset_tokenizer_revision": dataset_manifest["tokenizer_revision"],
        "evaluation_token_ids_sha256_i64le": token_hash,
        "input_tokens": int(token_ids.shape[1]), "evaluated_tokens": evaluated_tokens,
        "max_length": MAX_LENGTH, "stride": STRIDE,
        "add_special_tokens": False, "chat_template": False,
        "window_count": len(rows), "window_results": rows,
        "perplexity": ppl, "average_nll": average_nll,
        "exact_window_weighted_average_nll": exact_nll,
        "model_load_seconds": load_seconds, "evaluation_wall_seconds": elapsed,
        "peak_gpu_allocated_bytes": peak_alloc, "peak_gpu_reserved_bytes": peak_reserved,
        "torch_version": torch.__version__, "transformers_version": __import__("transformers").__version__,
        "device": torch.cuda.get_device_name(0), "started_utc": started_utc,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "window_csv": str(row_path.relative_to(OUT)), "log": str(log_path.relative_to(OUT)),
    }
    write_json_atomic(RAW_DIR / f"{method}_{dataset}.json", result)
    print(f"[eval:done] {method} {dataset}: PPL={ppl:.8f} NLL={average_nll:.9f} "
          f"tokens={evaluated_tokens} windows={len(rows)} seconds={elapsed:.1f}", flush=True)
    del capture, base, token_ids
    gc.collect(); torch.cuda.empty_cache()
    return result


def main():
    manifest = json.loads((OUT / "FORMAL_RUN_MANIFEST.json").read_text(encoding="utf-8"))
    if manifest.get("status") not in ("QUANTIZATION_COMPLETE_EVALUATION_READY", "EVALUATION_RUNNING", "EVALUATION_PARTIAL"):
        raise RuntimeError(f"formal quantization is not complete: {manifest.get('status')}")
    for method in ("M1", "M2"):
        qcheck = json.loads((OUT / method / "QUANTIZATION_CORRECTNESS.json").read_text(encoding="utf-8"))
        reload_check = json.loads((OUT / method / "RELOAD_CORRECTNESS.json").read_text(encoding="utf-8"))
        if not qcheck.get("pass") or not reload_check.get("pass"):
            raise RuntimeError(f"{method} checkpoint correctness not PASS")
    manifest["status"] = "EVALUATION_RUNNING"
    manifest["evaluation_started_utc"] = datetime.now(timezone.utc).isoformat()
    write_json_atomic(OUT / "FORMAL_RUN_MANIFEST.json", manifest)

    results = {}
    # Frozen matrix order: WT2 first for M0/M1/M2, then fixed-budget C4.
    for dataset in ("wikitext", "c4_fixed_budget"):
        results[dataset] = {}
        for method in ("M0", "M1", "M2"):
            results[dataset][method] = run_one(method, dataset)
        manifest = json.loads((OUT / "FORMAL_RUN_MANIFEST.json").read_text(encoding="utf-8"))
        manifest["results"]["wikitext_evaluation" if dataset == "wikitext" else "c4_evaluation"] = "PASS"
        write_json_atomic(OUT / "FORMAL_RUN_MANIFEST.json", manifest)

    for dataset, key in (("wikitext", "WT2"), ("c4_fixed_budget", "C4_fixed_budget")):
        # Preserve separate corpus interpretation; never pool PPL values.
        base = results[dataset]["M0"]["perplexity"]
        native = results[dataset]["M1"]["perplexity"]
        searched = results[dataset]["M2"]["perplexity"]
        results[dataset]["comparison"] = {
            "M1_vs_M0_ppl_percent": (native / base - 1.0) * 100.0,
            "M2_vs_M0_ppl_percent": (searched / base - 1.0) * 100.0,
            "M2_vs_M1_ppl_percent": (searched / native - 1.0) * 100.0,
            "M2_strictly_better_than_M1": searched < native,
        }
    results["verdict"] = "FULLMODEL_M2_QUALITY_IMPROVED" if results["wikitext"]["M2"]["perplexity"] < results["wikitext"]["M1"]["perplexity"] else "NO_QUALITY_GAIN"
    results["quality_claim_scope"] = "primary strict comparison is frozen WikiText-2 test M2 vs M1; C4 fixed-budget is separate validation evidence"
    write_json_atomic(EVAL_DIR / "EVAL_CORRECTNESS.json", {
        "status": "PASS", "all_methods_all_datasets": True,
        "max_length": MAX_LENGTH, "stride": STRIDE,
        "all_token_counts_match_frozen_manifests": True,
        "all_window_weighted_nll_matches_shared_evaluator": True,
        "all_aggregates_finite": True,
        "results": {d: {m: {"ppl": results[d][m]["perplexity"],
                            "nll": results[d][m]["average_nll"],
                            "tokens": results[d][m]["evaluated_tokens"],
                            "windows": results[d][m]["window_count"]}
                        for m in ("M0", "M1", "M2")} for d in ("wikitext", "c4_fixed_budget")},
    })
    write_json_atomic(EVAL_DIR / "SUMMARY.json", results)
    rows = []
    for dataset in ("wikitext", "c4_fixed_budget"):
        for method in ("M0", "M1", "M2"):
            item = results[dataset][method]
            rows.append({"dataset": item["dataset_label"], "method": method,
                         "ppl": item["perplexity"], "average_nll": item["average_nll"],
                         "evaluated_tokens": item["evaluated_tokens"], "window_count": item["window_count"],
                         "evaluation_seconds": item["evaluation_wall_seconds"],
                         "peak_gpu_allocated_bytes": item["peak_gpu_allocated_bytes"],
                         "peak_gpu_reserved_bytes": item["peak_gpu_reserved_bytes"]})
    with (EVAL_DIR / "FULLMODEL_COMPARISON.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    report = f"""# v18 model-level evaluation

Evaluation correctness: **PASS**. All six M0/M1/M2 corpus runs used the frozen token-ID manifests, the shared `benchmark.eval.ppl.evaluate_ppl` function, FP16 SDPA, TF32 disabled, `max_length={MAX_LENGTH}`, and `stride={STRIDE}`. Exact per-window loss and valid-token counts are saved under `evaluation/windows/`; aggregate NLL was cross-checked against those exact window values.

## WikiText-2 test

- M0 FP16 PPL: {results['wikitext']['M0']['perplexity']:.8f} ({results['wikitext']['M0']['evaluated_tokens']} targets)
- M1 native GPTQ G128 PPL: {results['wikitext']['M1']['perplexity']:.8f} ({results['wikitext']['M1']['evaluated_tokens']} targets)
- M2 pure range search PPL: {results['wikitext']['M2']['perplexity']:.8f} ({results['wikitext']['M2']['evaluated_tokens']} targets)
- M2 vs M1: {results['wikitext']['comparison']['M2_vs_M1_ppl_percent']:+.4f}% PPL.

## C4 validation fixed token budget

- M0 FP16 PPL: {results['c4_fixed_budget']['M0']['perplexity']:.8f} ({results['c4_fixed_budget']['M0']['evaluated_tokens']} targets)
- M1 native GPTQ G128 PPL: {results['c4_fixed_budget']['M1']['perplexity']:.8f} ({results['c4_fixed_budget']['M1']['evaluated_tokens']} targets)
- M2 pure range search PPL: {results['c4_fixed_budget']['M2']['perplexity']:.8f} ({results['c4_fixed_budget']['M2']['evaluated_tokens']} targets)
- M2 vs M1: {results['c4_fixed_budget']['comparison']['M2_vs_M1_ppl_percent']:+.4f}% PPL.

C4 is the frozen **validation fixed token budget PPL**, not standard/full C4 PPL. The two datasets are reported separately.

## Frozen decision

Verdict: **{results['verdict']}**. The primary decision uses only the strict frozen WikiText-2 test comparison M2 vs M1. No method, alpha grid, calibration data, or evaluation settings were selected after seeing these scores.
"""
    (EVAL_DIR / "REPORT.md").write_text(report, encoding="utf-8")
    manifest = json.loads((OUT / "FORMAL_RUN_MANIFEST.json").read_text(encoding="utf-8"))
    manifest["status"] = "COMPLETE"
    manifest["results"]["wikitext_evaluation"] = "PASS"
    manifest["results"]["c4_evaluation"] = "PASS"
    manifest["verdict"] = results["verdict"]
    manifest["completed_utc"] = datetime.now(timezone.utc).isoformat()
    write_json_atomic(OUT / "FORMAL_RUN_MANIFEST.json", manifest)
    print(json.dumps({"status": "PASS", "verdict": results["verdict"],
                      "WT2": {m: results["wikitext"][m]["perplexity"] for m in ("M0", "M1", "M2")},
                      "C4_fixed_budget": {m: results["c4_fixed_budget"][m]["perplexity"] for m in ("M0", "M1", "M2")}}, indent=2))


if __name__ == "__main__":
    main()
