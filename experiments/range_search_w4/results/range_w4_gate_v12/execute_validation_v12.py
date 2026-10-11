"""Preflight, execute and summarize the frozen Gate v1 validation.

No threshold or protocol selection occurs in this script. Manifest and Gate
identities are verified before any model forward or GPTQ candidate evaluation.
"""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import os
import shutil
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
os.environ.update({"HF_HOME": str(ROOT / ".hf_cache"),
                   "HF_HUB_CACHE": str(ROOT / ".hf_cache" / "hub"),
                   "HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1"})
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM

from experiments.range_search_w4 import run_96row_v11 as base
from experiments.range_search_w4.dynamic_core import native_hinv_from_documents
from experiments.range_search_w4.results.range_w4_gate_v12.prepare_validation_v12 import (
    GATE_SHA, LAYERS, PROJECTIONS, V8, V9, V10, V11, digest_file, tensor_hash,
)

OUT = Path(__file__).resolve().parent
RAW = OUT / "raw_validation"
WEIGHT_DIR = RAW / "candidate_weights"
COST_JOURNAL = OUT / "validation_execution_events.jsonl"
CACHE_DIR = OUT / "validation_cache"
ACTIVATION_CACHE = CACHE_DIR / "activation_cache.pt"
EXPECTED_REPO_NAMES = ("q_proj", "o_proj", "up_proj", "down_proj")
GRID = [round(i / 10, 1) for i in range(1, 11)]
PARSE = argparse.ArgumentParser()
PARSE.add_argument("--preflight", action="store_true")
PARSE.add_argument("--run", action="store_true")
PARSE.add_argument("--summarize", action="store_true")


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def save_json(path: Path, payload: object, *, overwrite: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    if path.exists() and not overwrite and path.read_bytes() != tmp.read_bytes():
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"Refusing to overwrite frozen artifact: {path}")
    tmp.replace(path)


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def req(ok: bool, detail: str) -> None:
    if not ok:
        raise RuntimeError("VALIDATION CORRECTNESS BLOCK: " + detail)


def require(ok: bool, detail: str) -> None:
    """Alias used by the formal runner's completion checks."""
    req(ok, detail)


def atomic_torch_save(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def append_cost_event(event: dict) -> None:
    COST_JOURNAL.parent.mkdir(parents=True, exist_ok=True)
    with COST_JOURNAL.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(event, ensure_ascii=False, allow_nan=False) + "\n")
        stream.flush()


def load_tensor_file(path: Path) -> dict:
    with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
        return torch.load(path, map_location="cpu", weights_only=True)


def manifests() -> tuple[dict, dict, dict, dict]:
    gate_path = OUT / "GATE_V1.json"
    rows_path = OUT / "validation_row_manifest.json"
    splits_path = OUT / "validation_split_manifest.json"
    protocol_path = OUT / "VALIDATION_PROTOCOL.json"
    req(digest_file(gate_path) == GATE_SHA, "GATE_V1 SHA256 changed")
    gate = load_json(gate_path)
    req(gate["tau_gain"] == .10 and gate["tau_margin"] == .06, "frozen Gate thresholds changed")
    req(gate["search_gain_per_split_definition"] == "1 - min_alpha(search_SSE) / search_SSE(alpha=1.0)",
        "frozen gain definition changed")
    req("second_smallest_search_SSE - smallest_search_SSE" in gate["search_margin_per_split_definition"],
        "frozen margin definition changed")
    plan = (OUT / "VALIDATION_PLAN_FROZEN.md").read_text(encoding="utf-8")
    req(GATE_SHA in plan and "tau_gain=0.10 and tau_margin=0.06" in plan, "frozen validation plan mismatch")
    protocol = load_json(protocol_path)
    rows, splits = load_json(rows_path), load_json(splits_path)
    req(protocol["gate_v1_sha256"] == GATE_SHA and protocol["tau_gain"] == .10 and protocol["tau_margin"] == .06,
        "validation protocol Gate identity")
    req(protocol["row_manifest_sha256"] == digest_file(rows_path), "row manifest hash differs from protocol")
    req(protocol["split_manifest_sha256"] == digest_file(splits_path), "split manifest hash differs from protocol")
    req(len(rows["rows"]) == 48 and len(splits["splits"]) == 10, "frozen manifest dimensions")
    req({r["layer"] for r in rows["rows"]} == {4, 8, 20}, "validation layer scope")
    req(set(r["projection"] for r in rows["rows"]) == set(EXPECTED_REPO_NAMES), "validation projection scope")
    req(len({(r["layer"], r["projection"], r["row_index"]) for r in rows["rows"]}) == 48,
        "duplicate validation rows")
    return gate, rows, splits, protocol


def verify_split_disjointness(splits: dict) -> dict:
    all_docs = [d for s in splits["splits"] for side in ("search", "heldout") for d in s[side]]
    search_docs = [d for s in splits["splits"] for d in s["search"]]
    held_docs = [d for s in splits["splits"] for d in s["heldout"]]
    req(len(all_docs) == 80, "expected 80 validation articles")
    req(len({d["source_article_index"] for d in all_docs}) == 80, "validation article IDs repeat")
    req(len({d["article_sha256"] for d in all_docs}) == 80, "validation article hashes repeat")
    req(len({d["token_sha256"] for d in all_docs}) == 80, "validation token hashes repeat")
    req(not ({d["source_article_index"] for d in search_docs} & {d["source_article_index"] for d in held_docs}),
        "search/heldout article IDs overlap")
    req(not ({d["token_sha256"] for d in search_docs} & {d["token_sha256"] for d in held_docs}),
        "search/heldout token hashes overlap")
    for split in splits["splits"]:
        req(len(split["search"]) == len(split["heldout"]) == 4, f"split size {split['split_id']}")
        req(not ({d["source_article_index"] for d in split["search"]} &
                 {d["source_article_index"] for d in split["heldout"]}), f"within-split leakage {split['split_id']}")
        for doc in split["search"] + split["heldout"]:
            req(len(doc["token_ids"]) == 128 and
                hashlib.sha256(json.dumps(doc["token_ids"]).encode()).hexdigest() == doc["token_sha256"],
                f"token content hash {split['split_id']}")
    prep = __import__("experiments.range_search_w4.results.range_w4_gate_v12.prepare_validation_v12",
                      fromlist=["old_article_blacklist"])
    old_ids, old_hashes, audit = prep.old_article_blacklist()
    req(not ({d["source_article_index"] for d in all_docs} & old_ids), "validation reuses an old article index")
    req(not ({h for d in all_docs for h in (d["article_sha256"], d["token_sha256"])} & old_hashes),
        "validation reuses an old article/token hash")
    return {"validation_articles": 80, "validation_article_ids_unique": True,
            "validation_article_hashes_unique": True, "validation_token_hashes_unique": True,
            "search_heldout_disjoint": True, "historical_overlap": False,
            "historical_article_records": audit}


def input_key(layer: int, projection: str) -> str:
    return f"L{layer}/{projection}"


def row_key(row: dict) -> str:
    return f"L{row['layer']}_{row['projection']}_R{row['row_index']}"


def capture_inputs_and_weights(rows: dict, splits: dict, protocol: dict) -> tuple[dict, dict, dict]:
    rows_sha = protocol["row_manifest_sha256"]
    split_sha = protocol["split_manifest_sha256"]
    docs = [d for s in splits["splits"] for side in ("search", "heldout") for d in s[side]]
    docs_by_hash = {d["token_sha256"]: d for d in docs}
    req(len(docs_by_hash) == 80, "duplicate document token sequence")
    if ACTIVATION_CACHE.is_file():
        cached = load_tensor_file(ACTIVATION_CACHE)
        req(cached.get("row_manifest_sha256") == rows_sha and cached.get("split_manifest_sha256") == split_sha,
            "activation cache manifest mismatch")
        req(set(cached["inputs"]) == {input_key(l, p) for l in LAYERS for p in PROJECTIONS},
            "activation cache keys")
        req(set(cached["weights"]) == {row_key(r) for r in rows["rows"]}, "weight cache keys")
        for row in rows["rows"]:
            req(tensor_hash(cached["weights"][row_key(row)]) == row["weight_hash"],
                f"cached pristine weight {row_key(row)}")
        return cached["inputs"], cached["weights"], cached["metadata"]
    if not torch.cuda.is_available():
        raise RuntimeError("Frozen protocol requires CUDA")
    torch.manual_seed(42)
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    snapshot = ROOT / ".hf_cache" / "hub" / "models--Qwen--Qwen2.5-0.5B-Instruct" / "snapshots" / protocol["revision"]
    start = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(snapshot, torch_dtype=torch.float16,
                                                  local_files_only=True, device_map="cuda",
                                                  low_cpu_mem_usage=True)
    model.eval()
    inputs = {input_key(l, p): {} for l in LAYERS for p in PROJECTIONS}
    weights = {}
    for row in rows["rows"]:
        module = base.module_for(model, row["layer"], row["projection"])
        weights[row_key(row)] = module.weight[row["row_index"]].detach().float().cpu().clone()
    current = [None]
    handles = []
    for layer in LAYERS:
        for projection in PROJECTIONS:
            module = base.module_for(model, layer, projection)
            key = input_key(layer, projection)
            def hook(_module, args, _output, key=key):
                token_hash = current[0]
                if token_hash is None:
                    raise RuntimeError("activation hook ran outside a frozen article")
                inputs[key][token_hash] = args[0].detach().reshape(-1, args[0].shape[-1]).float().cpu().clone()
            handles.append(module.register_forward_hook(hook))
    try:
        with torch.inference_mode():
            for doc in docs:
                ids = doc["token_ids"]
                token_hash = doc["token_sha256"]
                req(hashlib.sha256(json.dumps(ids).encode()).hexdigest() == token_hash,
                    f"input token hash {token_hash}")
                current[0] = token_hash
                token = torch.tensor(ids, dtype=torch.long, device="cuda")[None]
                model(input_ids=token, use_cache=False)
    finally:
        for handle in handles:
            handle.remove()
    base.sync()
    elapsed = time.perf_counter() - start
    peak_alloc = int(torch.cuda.max_memory_allocated())
    peak_reserved = int(torch.cuda.max_memory_reserved())
    for row in rows["rows"]:
        req(tensor_hash(weights[row_key(row)]) == row["weight_hash"], f"pristine model W {row_key(row)}")
    for layer in LAYERS:
        for projection in PROJECTIONS:
            values = inputs[input_key(layer, projection)]
            req(set(values) == set(docs_by_hash), f"incomplete activations L{layer}/{projection}")
            expected_dim = next(r["input_dim"] for r in rows["rows"] if r["projection"] == projection)
            req(all(tuple(x.shape) == (128, expected_dim) for x in values.values()),
                f"activation geometry L{layer}/{projection}")
    metadata = {"capture_seconds": elapsed, "peak_allocated_bytes": peak_alloc,
                "peak_reserved_bytes": peak_reserved, "document_count": len(docs),
                "cuda_device": torch.cuda.get_device_name(0), "torch_version": str(torch.__version__),
                "transformers_version": __import__("transformers").__version__,
                "gptqmodel_version": __import__("importlib.metadata", fromlist=["version"]).version("gptqmodel")}
    del model
    gc.collect()
    torch.cuda.empty_cache()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    atomic_torch_save(ACTIVATION_CACHE, {"row_manifest_sha256": rows_sha,
                                        "split_manifest_sha256": split_sha,
                                        "inputs": inputs, "weights": weights,
                                        "metadata": metadata})
    return inputs, weights, metadata


def side_docs(inputs: dict, row: dict, split: dict, side: str) -> list[torch.Tensor]:
    return [inputs[input_key(row["layer"], row["projection"])][d["token_sha256"]]
            for d in split[side]]


def build_hinv(row: dict, docs: list[torch.Tensor]) -> tuple[torch.Tensor, dict]:
    module = torch.nn.Linear(row["input_dim"], 1, bias=False, device="cuda", dtype=torch.float16)
    torch.nn.init.zeros_(module.weight)
    torch.cuda.reset_peak_memory_stats()
    base.sync()
    start = time.perf_counter()
    h, hinv, dead = native_hinv_from_documents(docs, module, .1)
    base.sync()
    seconds = time.perf_counter() - start
    if dead:
        raise RuntimeError(f"dead Hessian columns: {dead} for {row['projection']}")
    return hinv, {"hessian_build_seconds": seconds, "dead_columns": dead,
                  "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
                  "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
                  "search_x_sha256": tensor_hash(torch.cat(docs))}


def representative_rows(rows: dict) -> dict[str, dict]:
    reps = {}
    for projection in PROJECTIONS:
        row = next(r for r in rows["rows"] if r["projection"] == projection)
        reps[projection] = row
    return reps


def write_preflight_report(payload: dict) -> None:
    lines = ["# Validation v12 preflight and native parity", "",
             f"Status: **{payload['status']}**", "",
             f"Gate SHA256: `{payload['gate_sha256']}`; thresholds: gain ≥ 0.10 AND margin ≥ 0.06.",
             f"Frozen manifests: rows `{payload['row_manifest_sha256']}`, splits `{payload['split_manifest_sha256']}`.",
             f"Workload: {payload['nominal_candidates']} nominal, {payload['reusable_candidates']} reusable, {payload['new_candidates']} new candidate evaluations.",
             f"Model-input capture: {payload.get('capture', {}).get('capture_seconds', float('nan')):.1f}s; CUDA allocated/reserved peak {payload.get('capture', {}).get('peak_allocated_bytes', 0)/2**20:.1f}/{payload.get('capture', {}).get('peak_reserved_bytes', 0)/2**20:.1f} MiB.",
             "", "## Alpha=1 native parity on preregistered S00 rows", "",
             "| Projection | Row | Input dim | Result | H build (s) | Native GPTQ (s) | Custom alpha=1 (s) | Max CUDA reserved (MiB) |",
             "|---|---:|---:|---|---:|---:|---:|---:|"]
    for p in PROJECTIONS:
        x = payload["parity"][p]
        lines.append(f"| {p} | L{x['layer']}_R{x['row_index']} | {x['input_dim']} | {x['status']} | {x['hessian_build_seconds']:.3f} | {x['native_full_row_quantize_seconds']:.3f} | {x['custom_alpha1_candidate_seconds']:.3f} | {max(x['candidate_peak_reserved_bytes'], x['native_parity_peak_reserved_bytes'])/2**20:.1f} |")
    lines += ["", "## Candidate cost estimate", "",
              f"Measured alpha=1 candidate samples project {payload['candidate_seconds_by_projection']} to 1,200 evaluations per projection. Estimated candidate timer sum: {payload['estimated_candidate_seconds_sum']:.1f}s; Hessian construction: {payload['estimated_hessian_seconds']:.1f}s; capture: {payload['capture']['capture_seconds']:.1f}s.",
              f"With 1.5× execution margin, estimated formal wall time is {payload['estimated_formal_wall_seconds']:.1f}s ({payload['estimated_formal_wall_hours']:.2f}h). This is an estimate from one row per projection and may vary by split/layer.",
              f"Disk free before formal run: {payload['disk_free_bytes']/2**30:.2f} GiB; minimum planned artifact reservation: {payload['estimated_artifact_bytes']/2**30:.2f} GiB.",
              f"Peak observed CUDA reserved across capture/parity: {payload['peak_cuda_reserved_bytes']/2**20:.1f} MiB.",
              "", "Formal evaluation is authorized only when status is PASS. No candidate landscape is produced during preflight.", ""]
    (OUT / "PREFLIGHT_VALIDATION_REPORT.md").write_text("\n".join(lines), encoding="utf-8", newline="\n")


def preflight() -> dict:
    preflight_started = time.perf_counter()
    gate, rows, splits, protocol = manifests()
    article_check = verify_split_disjointness(splits)
    req(not (OUT / "raw_validation").exists() or not any((OUT / "raw_validation").glob("*.json")),
        "formal raw records already exist before preflight")
    inputs, weights, capture_meta = capture_inputs_and_weights(rows, splits, protocol)
    split00 = splits["splits"][0]
    parity = {}
    hessian_samples = {}
    for projection, row in representative_rows(rows).items():
        search_docs = side_docs(inputs, row, split00, "search")
        xh_docs = side_docs(inputs, row, split00, "heldout")
        hinv, hmeta = build_hinv(row, search_docs)
        hessian_samples[projection] = hmeta
        parity[projection] = base.native_parity(row, split00, inputs, weights, hinv)
        parity[projection]["hessian_build_seconds"] = hmeta["hessian_build_seconds"]
    parity_ok = all(x["status"] == "PASS" for x in parity.values())
    nominal = 48 * 10 * 10
    reusable = 0
    candidate_counts = {p: sum(r["projection"] == p for r in rows["rows"]) * 10 * 10 for p in PROJECTIONS}
    mean_seconds = {p: parity[p]["custom_alpha1_candidate_seconds"] for p in PROJECTIONS}
    estimated_candidate = sum(candidate_counts[p] * mean_seconds[p] for p in PROJECTIONS)
    # 3 layers × 10 splits = 30 unique Hessians per projection.
    estimated_hessian = sum(hessian_samples[p]["hessian_build_seconds"] * 30 for p in PROJECTIONS)
    capture_s = capture_meta["capture_seconds"]
    formal_wall = capture_s + 1.5 * (estimated_candidate + estimated_hessian)
    disk = shutil.disk_usage(OUT)
    # Includes 80-document FP32 inputs (estimated from actual cache tensor bytes), raw JSON and qweight tensors.
    cache_bytes = sum(t.numel() * t.element_size() for per_layer in inputs.values() for t in per_layer.values())
    estimated_artifacts = int(cache_bytes * 1.35 + sum(r["input_dim"] * 2 * 10 * 10
                                                       for r in rows["rows"]) * 4)
    peak_cuda = max(capture_meta["peak_reserved_bytes"],
                    max(max(p["candidate_peak_reserved_bytes"], p["native_parity_peak_reserved_bytes"])
                        for p in parity.values()))
    payload = {
        "status": "PASS" if parity_ok and disk.free > estimated_artifacts else "FAIL",
        "created_at_utc": now_utc(), "gate_sha256": GATE_SHA,
        "row_manifest_sha256": protocol["row_manifest_sha256"],
        "split_manifest_sha256": protocol["split_manifest_sha256"],
        "article_disjointness": article_check,
        "nominal_candidates": nominal, "reusable_candidates": reusable,
        "new_candidates": nominal - reusable, "candidate_counts_by_projection": candidate_counts,
        "parity": parity, "parity_status": "PASS" if parity_ok else "FAIL",
        "capture": capture_meta, "candidate_seconds_by_projection": mean_seconds,
        "estimated_candidate_seconds_sum": estimated_candidate,
        "estimated_hessian_seconds": estimated_hessian,
        "estimated_formal_wall_seconds": formal_wall,
        "estimated_formal_wall_hours": formal_wall / 3600,
        "disk_free_bytes": disk.free, "estimated_artifact_bytes": estimated_artifacts,
        "peak_cuda_reserved_bytes": peak_cuda,
        "preflight_total_seconds": time.perf_counter() - preflight_started,
        "gpu": torch.cuda.get_device_name(0), "torch": str(torch.__version__),
        "checks": {"gate_hash": True, "manifests_frozen": True,
                   "articles_globally_unique_and_disjoint_from_v8_v11": True,
                   "search_heldout_disjoint": True,
                   "four_projection_alpha1_native_parity": parity_ok,
                   "cuda_available": torch.cuda.is_available(),
                   "disk_budget": disk.free > estimated_artifacts}}
    save_json(OUT / "validation_preflight.json", payload)
    write_preflight_report(payload)
    if payload["status"] == "PASS":
        save_json(OUT / "preflight_authorization.json", {
            "status": "PREFLIGHT_PASSED_FORMAL_4800_CANDIDATE_RUN_AUTHORIZED",
            "gate_sha256": GATE_SHA, "row_manifest_sha256": payload["row_manifest_sha256"],
            "split_manifest_sha256": payload["split_manifest_sha256"],
            "candidate_slots": nominal, "estimated_formal_wall_seconds": formal_wall,
            "created_at_utc": now_utc()})
    return payload


def phase_a_snapshot() -> dict:
    snapshot_path = OUT / "phaseA_preservation_fingerprint.json"
    if snapshot_path.is_file():
        existing = load_json(snapshot_path)
        req(check_phase_a_snapshot(existing), "Phase A preservation snapshot no longer matches")
        return existing
    # Snapshot every file present before the first formal candidate. New raw
    # validation records are created afterwards and are outside this baseline.
    phase_files = {}
    for path in sorted(p for p in OUT.rglob("*") if p.is_file()
                       and "raw_validation" not in p.parts and "candidate_weights" not in p.parts):
        if path.name == "phaseA_preservation_fingerprint.json":
            continue
        phase_files[str(path.relative_to(OUT)).replace("\\", "/")] = {
            "sha256": digest_file(path), "size_bytes": path.stat().st_size,
            "mtime_ns": path.stat().st_mtime_ns}
    history = {}
    for name, path in (("v8_manifest", V8 / "manifest.json"),
                       ("v9_split_manifest", V9 / "splits_manifest.json"),
                       ("v10_manifest_reference", V10 / "manifest_reference.json"),
                       ("v11_manifest_reference", V11 / "manifest_reference.json")):
        history[name] = {"path": str(path.relative_to(ROOT)).replace("\\", "/"),
                         "sha256": digest_file(path), "size_bytes": path.stat().st_size,
                         "mtime_ns": path.stat().st_mtime_ns}
    payload = {"captured_at_utc": now_utc(), "gate_sha256": GATE_SHA,
               "phase_a_files": phase_files, "historical_manifest_files": history}
    save_json(snapshot_path, payload, overwrite=False)
    return payload


def check_phase_a_snapshot(snapshot: dict) -> bool:
    for rel, expected in snapshot["phase_a_files"].items():
        path = OUT / rel
        if not path.is_file() or digest_file(path) != expected["sha256"]:
            return False
    for key, expected in snapshot["historical_manifest_files"].items():
        path = ROOT / expected["path"]
        if not path.is_file() or digest_file(path) != expected["sha256"]:
            return False
    return True


def source_split_map(splits: dict) -> dict:
    return {s["split_id"]: s for s in splits["splits"]}


def first_candidate_state_snapshot() -> dict:
    path = OUT / "candidate_run_manifest_lock.json"
    gate, rows, splits, protocol = manifests()
    lock = {"gate_sha256": GATE_SHA,
            "row_manifest_sha256": digest_file(OUT / "validation_row_manifest.json"),
            "split_manifest_sha256": digest_file(OUT / "validation_split_manifest.json"),
            "protocol_sha256": digest_file(OUT / "VALIDATION_PROTOCOL.json"),
            "row_manifest_mtime_ns": (OUT / "validation_row_manifest.json").stat().st_mtime_ns,
            "split_manifest_mtime_ns": (OUT / "validation_split_manifest.json").stat().st_mtime_ns,
            "locked_at_epoch_ns": time.time_ns(), "locked_at_utc": now_utc()}
    if path.exists():
        old = load_json(path)
        for key in ("gate_sha256", "row_manifest_sha256", "split_manifest_sha256", "protocol_sha256"):
            req(old[key] == lock[key], f"candidate run lock mismatch {key}")
        return old
    save_json(path, lock, overwrite=False)
    return lock


def formal_run() -> dict:
    gate, rows, splits, protocol = manifests()
    preflight_path = OUT / "validation_preflight.json"
    req(preflight_path.is_file(), "preflight report missing")
    pre = load_json(preflight_path)
    req(pre["status"] == "PASS" and pre["parity_status"] == "PASS", "preflight/parity not PASS")
    req(load_json(OUT / "preflight_authorization.json")["status"].startswith("PREFLIGHT_PASSED"),
        "formal run was not authorized by preflight")
    lock = first_candidate_state_snapshot()
    preservation = phase_a_snapshot()
    inputs, weights, capture_meta = capture_inputs_and_weights(rows, splits, protocol)
    split_map = source_split_map(splits)
    RAW.mkdir(parents=True, exist_ok=True)
    WEIGHT_DIR.mkdir(parents=True, exist_ok=True)
    nominal = 4800
    actual_new = 0
    initially_completed_records = len(list(RAW.glob("L*_R*_S??.json"))) if RAW.exists() else 0
    candidate_seconds = 0.0
    hessian_seconds = 0.0
    peak_alloc = 0
    peak_reserved = max(capture_meta["peak_reserved_bytes"], pre["peak_cuda_reserved_bytes"])
    start_total = time.perf_counter()
    for layer in LAYERS:
        for projection in PROJECTIONS:
            local_rows = [r for r in rows["rows"] if r["layer"] == layer and r["projection"] == projection]
            for split in splits["splits"]:
                sid = split["split_id"]
                pending = []
                for row in local_rows:
                    raw_path = RAW / f"L{layer}_{projection}_R{row['row_index']}_{sid}.json"
                    q_path = WEIGHT_DIR / f"L{layer}_{projection}_R{row['row_index']}_{sid}.pt"
                    if raw_path.exists() and q_path.exists():
                        rec = load_json(raw_path)
                        qobj = load_tensor_file(q_path)
                        req(rec["row_manifest_sha256"] == lock["row_manifest_sha256"] and
                            rec["split_manifest_sha256"] == lock["split_manifest_sha256"],
                            f"resumed raw manifest mismatch {raw_path.name}")
                        req(all(tensor_hash(qobj["weights"][str(float(c["alpha"]))]) == c["weight_sha256"]
                                for c in rec["candidates"]), f"resumed weight hashes {q_path.name}")
                        continue
                    if raw_path.exists() != q_path.exists():
                        raise RuntimeError(f"incomplete resume pair: {raw_path.name}")
                    pending.append((row, raw_path, q_path))
                if not pending:
                    continue
                representative = pending[0][0]
                docs = side_docs(inputs, representative, split, "search")
                hinv, hmeta = build_hinv(representative, docs)
                hessian_seconds += hmeta["hessian_build_seconds"]
                append_cost_event({"event": "hessian_build", "layer": layer,
                                   "projection": projection, "split_id": sid,
                                   "seconds": hmeta["hessian_build_seconds"],
                                   "allocated_bytes": hmeta["peak_allocated_bytes"],
                                   "reserved_bytes": hmeta["peak_reserved_bytes"],
                                   "created_at_utc": now_utc()})
                peak_alloc = max(peak_alloc, hmeta["peak_allocated_bytes"])
                peak_reserved = max(peak_reserved, hmeta["peak_reserved_bytes"])
                for row, raw_path, q_path in pending:
                    w = weights[row_key(row)].cuda()
                    pristine_hash = tensor_hash(w)
                    xs = torch.cat(side_docs(inputs, row, split, "search")).cuda()
                    xh = torch.cat(side_docs(inputs, row, split, "heldout")).cuda()
                    split_start = time.perf_counter()
                    candidates, candidate_weights = [], {}
                    for alpha in GRID:
                        c, qcpu = base.custom_candidate(w, hinv, xs, xh, alpha)
                        append_cost_event({"event": "candidate", "row_id": row_key(row),
                                           "split_id": sid, "alpha": alpha,
                                           "seconds": c["candidate_seconds"],
                                           "allocated_bytes": c["candidate_peak_allocated_bytes"],
                                           "reserved_bytes": c["candidate_peak_reserved_bytes"],
                                           "weight_sha256": c["weight_sha256"],
                                           "created_at_utc": now_utc()})
                        req(tensor_hash(w) == pristine_hash, f"candidate modified W {row_key(row)}/{sid}/{alpha}")
                        req(tensor_hash(qcpu) == c["weight_sha256"], f"candidate weight hash {row_key(row)}/{sid}/{alpha}")
                        candidates.append(c)
                        candidate_weights[str(float(alpha))] = qcpu
                        candidate_seconds += c["candidate_seconds"]
                        peak_alloc = max(peak_alloc, c["candidate_peak_allocated_bytes"])
                        peak_reserved = max(peak_reserved, c["candidate_peak_reserved_bytes"])
                    selected = min(candidates, key=lambda c: (c["search_output_mse"], c["alpha"]))
                    alpha1 = next(c for c in candidates if c["alpha"] == 1.0)
                    rec = {
                        "row_id": row_key(row), "layer": layer, "projection": projection,
                        "projection_name": row["projection_name"], "row_index": row["row_index"],
                        "split_id": sid, "source": "v12_validation",
                        "gate_v1_sha256": GATE_SHA,
                        "row_manifest_sha256": lock["row_manifest_sha256"],
                        "split_manifest_sha256": lock["split_manifest_sha256"],
                        "weight_sha256": row["weight_hash"], "pristine_weight_hash_before": pristine_hash,
                        "pristine_weight_hash_after": tensor_hash(w),
                        "search_article_ids": [d["source_article_index"] for d in split["search"]],
                        "heldout_article_ids": [d["source_article_index"] for d in split["heldout"]],
                        "search_token_hashes": [d["token_sha256"] for d in split["search"]],
                        "heldout_token_hashes": [d["token_sha256"] for d in split["heldout"]],
                        "search_x_sha256": tensor_hash(xs), "heldout_x_sha256": tensor_hash(xh),
                        "selected_alpha_on_search": selected["alpha"],
                        "search_D_sse": alpha1["search_sse"], "search_C_sse": selected["search_sse"],
                        "search_D_mse": alpha1["search_output_mse"], "search_C_mse": selected["search_output_mse"],
                        "heldout_D_sse": alpha1["heldout_sse"], "heldout_C_sse": selected["heldout_sse"],
                        "heldout_D_mse": alpha1["heldout_output_mse"], "heldout_C_mse": selected["heldout_output_mse"],
                        "search_gain": 1 - selected["search_sse"] / alpha1["search_sse"],
                        "heldout_gain_C": 1 - selected["heldout_sse"] / alpha1["heldout_sse"],
                        "selected_weight_mse": selected["weight_mse"],
                        "candidate_count": len(candidates), "candidate_evaluations_this_run": len(candidates),
                        "candidates": candidates, "candidate_seconds_total": sum(c["candidate_seconds"] for c in candidates),
                        "hessian": hmeta, "row_split_seconds": time.perf_counter() - split_start,
                        "selected_by_search_only": True,
                    }
                    atomic_torch_save(q_path, {"row_manifest_sha256": lock["row_manifest_sha256"],
                                               "split_manifest_sha256": lock["split_manifest_sha256"],
                                               "weights": candidate_weights})
                    save_json(raw_path, rec)
                    actual_new += len(candidates)
                    print(f"SAVED {row_key(row)} {sid} alpha={selected['alpha']:.1f} "
                          f"search_gain={rec['search_gain']:.4f} elapsed={rec['row_split_seconds']:.1f}s", flush=True)
                del hinv
                torch.cuda.empty_cache()
    base.sync()
    elapsed = time.perf_counter() - start_total
    all_raw = list(RAW.glob("L*_R*_S??.json"))
    require(len(all_raw) == 480, f"formal record count {len(all_raw)} != 480")
    # The append-only journal includes candidate work that may have been repeated
    # after an interruption before its row/split checkpoint was committed.
    events = ([json.loads(line) for line in COST_JOURNAL.read_text(encoding="utf-8").splitlines()
               if line.strip()] if COST_JOURNAL.is_file() else [])
    candidate_events = [event for event in events if event["event"] == "candidate"]
    hessian_events = [event for event in events if event["event"] == "hessian_build"]
    candidate_seconds_cumulative = sum(float(event["seconds"]) for event in candidate_events)
    hessian_seconds_cumulative = sum(float(event["seconds"]) for event in hessian_events)
    first_start = lock.get("first_run_started_epoch_ns", lock["locked_at_epoch_ns"])
    preflight_seconds = float(pre.get("preflight_total_seconds", 0.0))
    timing = {"candidate_slots_nominal": nominal,
              "candidate_slots_reused_from_completed_cache": initially_completed_records * 10,
              "completed_candidate_slots": len(all_raw) * 10,
              "candidate_evaluation_attempts_including_resumed_retries": len(candidate_events),
              "hessian_build_evaluations_including_resumed_retries": len(hessian_events),
              "candidate_slots_new_this_invocation": actual_new,
              "candidate_seconds_sum_cumulative": candidate_seconds_cumulative,
              "hessian_seconds_sum_cumulative_unique_keys": hessian_seconds_cumulative,
              "capture_seconds_included_from_cache_metadata": capture_meta["capture_seconds"],
              "preflight_total_seconds": preflight_seconds,
              "runner_wall_seconds_from_first_invocation": max(0.0, (time.time_ns() - first_start) / 1e9),
              "total_wall_seconds_including_preflight_and_cache_build": preflight_seconds +
                  max(0.0, (time.time_ns() - first_start) / 1e9),
              "gpu_synchronized": True,
              "peak_cuda_allocated_bytes": max([peak_alloc] + [int(event.get("allocated_bytes", 0)) for event in events]),
              "peak_cuda_reserved_bytes": max([peak_reserved] + [int(event.get("reserved_bytes", 0)) for event in events]),
              "capture": capture_meta,
              "completed_raw_records": len(all_raw), "completed_candidate_records": len(all_raw) * 10,
              "created_at_utc": now_utc()}
    save_json(OUT / "validation_run_timing.json", timing)
    return timing


def gain(sse, d_sse):
    return 1.0 - np.asarray(sse, dtype=float) / np.asarray(d_sse, dtype=float)


def summarise() -> dict:
    gate, rows, splits, protocol = manifests()
    lock = load_json(OUT / "candidate_run_manifest_lock.json")
    require(lock["gate_sha256"] == GATE_SHA, "run lock Gate SHA")
    raw_files = sorted(RAW.glob("L*_R*_S??.json"))
    require(len(raw_files) == 480, f"need 480 raw records, got {len(raw_files)}")
    per = []
    search_landscape = []
    raw_by_key = {}
    for path in raw_files:
        rec = load_json(path)
        key = (rec["row_id"], rec["split_id"])
        req(key not in raw_by_key, f"duplicate raw key {key}")
        req(rec["gate_v1_sha256"] == GATE_SHA and rec["candidate_count"] == 10,
            f"raw Gate/candidate count {key}")
        req(rec["row_manifest_sha256"] == lock["row_manifest_sha256"] and
            rec["split_manifest_sha256"] == lock["split_manifest_sha256"], f"raw manifest hash {key}")
        req(rec["pristine_weight_hash_before"] == rec["weight_sha256"] == rec["pristine_weight_hash_after"],
            f"pristine W isolation {key}")
        qpath = WEIGHT_DIR / f"{rec['row_id']}_{rec['split_id']}.pt"
        qobj = load_tensor_file(qpath)
        require_weights = qobj["weights"]
        for c in rec["candidates"]:
            alpha = float(c["alpha"])
            req(tensor_hash(require_weights[str(alpha)]) == c["weight_sha256"], f"candidate hash {key}/{alpha}")
            search_landscape.append({"row_id": rec["row_id"], "split_id": rec["split_id"],
                                     "alpha": alpha, "search_sse": c["search_sse"]})
        raw_by_key[key] = rec
    searchdf = pd.DataFrame(search_landscape)
    req(len(raw_by_key) == 480 and len(searchdf) == 4800, "raw/search landscape dimensions")
    decisions = []
    decision_by_row = {}
    gate_search_fields = {"row_id", "split_id", "alpha", "search_sse"}
    req(set(searchdf.columns) == gate_search_fields, "Gate input contains non-search fields")
    for row_id, group in searchdf.groupby("row_id", sort=True):
        split_metrics = []
        for sid, sg in group.groupby("split_id", sort=True):
            ordered = sg.sort_values(["search_sse", "alpha"], kind="stable")
            best, second = ordered.iloc[0], ordered.iloc[1]
            baseline = sg[sg.alpha.eq(1.0)].iloc[0]
            req(baseline.search_sse > 0 and second.search_sse > 0, f"zero SSE denominator {row_id}/{sid}")
            split_metrics.append({"split_id": sid,
                                  "search_gain": 1 - best.search_sse / baseline.search_sse,
                                  "search_margin": (second.search_sse - best.search_sse) / second.search_sse,
                                  "search_best_alpha": float(best.alpha)})
        gain_median = float(np.median([x["search_gain"] for x in split_metrics]))
        margin_median = float(np.median([x["search_margin"] for x in split_metrics]))
        accept = gain_median >= .10 and margin_median >= .06
        decision_by_row[row_id] = {"accept": accept, "gain_median": gain_median,
                                   "margin_median": margin_median, "splits": split_metrics}
        first = group.iloc[0]
        decisions.append({"row_id": row_id, "layer": int(first.layer),
                          "projection": first.projection, "row_index": int(first.row_index),
                          "search_gain_median": gain_median, "search_margin_median": margin_median,
                          "gate_decision": "ACCEPT" if accept else "FALLBACK",
                          "tau_gain": .10, "tau_margin": .06,
                          "search_best_alpha_by_split": json.dumps({x["split_id"]: x["search_best_alpha"] for x in split_metrics},
                                                                    separators=(",", ":"))})
    # Only after all Gate decisions are frozen do we read held-out metrics for
    # final evaluation and construct the complete landscape artifact.
    landscape = []
    per = []
    for rec in raw_by_key.values():
        key = (rec["row_id"], rec["split_id"])
        selected = min(rec["candidates"], key=lambda c: (c["search_output_mse"], c["alpha"]))
        req(selected["alpha"] == rec["selected_alpha_on_search"], f"search argmin {key}")
        alpha1 = next(c for c in rec["candidates"] if c["alpha"] == 1.0)
        for c in rec["candidates"]:
            alpha = float(c["alpha"])
            landscape.append({"row_id": rec["row_id"], "layer": rec["layer"],
                              "projection": rec["projection"], "row_index": rec["row_index"],
                              "split_id": rec["split_id"], "alpha": alpha,
                              "search_sse": c["search_sse"], "search_output_mse": c["search_output_mse"],
                              "heldout_sse": c["heldout_sse"], "heldout_output_mse": c["heldout_output_mse"],
                              "weight_mse": c["weight_mse"], "scale": json.dumps(c["scale"], separators=(",", ":")),
                              "zero": json.dumps(c["zero"], separators=(",", ":")),
                              "candidate_seconds": c["candidate_seconds"],
                              "weight_sha256": c["weight_sha256"],
                              "selected_by_search": alpha == rec["selected_alpha_on_search"]})
        per.append({"row_id": rec["row_id"], "layer": rec["layer"],
                    "projection": rec["projection"], "row_index": rec["row_index"],
                    "split_id": rec["split_id"], "selected_alpha_C": selected["alpha"],
                    "search_D_sse": alpha1["search_sse"], "search_C_sse": selected["search_sse"],
                    "heldout_D_sse": alpha1["heldout_sse"], "heldout_C_sse": selected["heldout_sse"],
                    "search_gain_C": 1 - selected["search_sse"] / alpha1["search_sse"],
                    "heldout_gain_C": 1 - selected["heldout_sse"] / alpha1["heldout_sse"],
                    "weight_mse_C": selected["weight_mse"],
                    "search_article_ids": json.dumps(rec["search_article_ids"]),
                    "heldout_article_ids": json.dumps(rec["heldout_article_ids"]),
                    "search_x_sha256": rec["search_x_sha256"], "heldout_x_sha256": rec["heldout_x_sha256"]})
    perdf = pd.DataFrame(per)
    landscapedf = pd.DataFrame(landscape)
    req(len(perdf) == 480 and len(landscapedf) == 4800, "full landscape dimensions")
    perdf["gate_decision"] = perdf.row_id.map(lambda rid: "ACCEPT" if decision_by_row[rid]["accept"] else "FALLBACK")
    perdf["selected_alpha_G"] = np.where(perdf.gate_decision.eq("ACCEPT"), perdf.selected_alpha_C, 1.0)
    perdf["heldout_G_sse"] = np.where(perdf.gate_decision.eq("ACCEPT"), perdf.heldout_C_sse, perdf.heldout_D_sse)
    perdf["heldout_gain_G"] = 1 - perdf.heldout_G_sse / perdf.heldout_D_sse
    perdf["gate_search_gain_median"] = perdf.row_id.map(lambda rid: decision_by_row[rid]["gain_median"])
    perdf["gate_search_margin_median"] = perdf.row_id.map(lambda rid: decision_by_row[rid]["margin_median"])
    perdf.to_csv(OUT / "validation_per_row_split.csv", index=False, float_format="%.15g")
    landscapedf.to_csv(OUT / "validation_alpha_landscapes.csv", index=False, float_format="%.15g")
    pd.DataFrame(decisions).to_csv(OUT / "validation_gate_decisions.csv", index=False, float_format="%.15g")

    accepted = sum(v["accept"] for v in decision_by_row.values())
    dtotal = float(perdf.heldout_D_sse.sum())
    summary_rows = []
    methods = {"D": ("heldout_D_sse", "heldout_gain_C", True),
               "C": ("heldout_C_sse", "heldout_gain_C", False),
               "G": ("heldout_G_sse", "heldout_gain_G", False)}
    method_gain: dict[str, np.ndarray] = {}
    method_sse: dict[str, str] = {}
    for method, (sse_col, _, is_d) in methods.items():
        gvals = np.zeros(len(perdf), dtype=float) if is_d else gain(perdf[sse_col], perdf.heldout_D_sse)
        method_gain[method] = gvals
        method_sse[method] = sse_col
        c = perdf.copy()
        c["_gain"] = gvals
        row_medians = c.groupby("row_id")["_gain"].median().to_numpy()
        summary_rows.append({"method": method, "heldout_sse": float(c[sse_col].sum()),
                             "pooled_gain_vs_D": 0.0 if is_d else float(1 - c[sse_col].sum() / dtotal),
                             "retained_C_gain": (float(1 - c[sse_col].sum() / dtotal) /
                                                  float(1 - perdf.heldout_C_sse.sum() / dtotal))
                                                  if method == "G" and perdf.heldout_C_sse.sum() < dtotal else None,
                             "row_count": 48, "row_split_count": 480,
                             "median_row_median_gain": float(np.median(row_medians)),
                             "q25_row_median_gain": float(np.quantile(row_medians, .25)),
                             "q75_row_median_gain": float(np.quantile(row_medians, .75)),
                             "gain_lt_0": int((gvals < 0).sum()), "gain_lt_minus_5pct": int((gvals < -.05).sum()),
                             "gain_lt_minus_10pct": int((gvals < -.10).sum()),
                             "gain_lt_minus_20pct": int((gvals < -.20).sum()),
                             "gain_lt_minus_40pct": int((gvals < -.40).sum()),
                             "gain_lt_minus_50pct": int((gvals < -.50).sum()),
                             "worst_gain": float(gvals.min()),
                             "accepted_rows": accepted if method == "G" else (48 if method == "C" else 0),
                             "fallback_rows": 48 - accepted if method == "G" else (0 if method == "C" else 48)})
    pd.DataFrame(summary_rows).to_csv(OUT / "D_C_GATE_summary.csv", index=False, float_format="%.15g")

    projection_rows, layer_rows = [], []
    def strata_summary(column: str, key: str):
        out = []
        for stratum, part in perdf.groupby(key, sort=True):
            dsum = float(part.heldout_D_sse.sum())
            cs = float(part.heldout_C_sse.sum())
            gs = float(part.heldout_G_sse.sum())
            out.append({"stratum": stratum, "row_count": int(part.row_id.nunique()),
                        "row_split_count": len(part),
                        "gate_accept_rows": int(part.loc[part.gate_decision.eq("ACCEPT"), "row_id"].nunique()),
                        "gate_fallback_rows": int(part.loc[part.gate_decision.eq("FALLBACK"), "row_id"].nunique()),
                        "C_pooled_gain_vs_D": 1 - cs / dsum, "G_pooled_gain_vs_D": 1 - gs / dsum,
                        "C_severe_failures": int((part.heldout_gain_C < -.10).sum()),
                        "G_severe_failures": int((part.heldout_gain_G < -.10).sum()),
                        "C_gain_lt_0": int((part.heldout_gain_C < 0).sum()),
                        "G_gain_lt_0": int((part.heldout_gain_G < 0).sum()),
                        "C_worst_gain": float(part.heldout_gain_C.min()),
                        "G_worst_gain": float(part.heldout_gain_G.min()),
                        "D_heldout_sse": dsum, "C_heldout_sse": cs, "G_heldout_sse": gs})
        return out
    for key in ("projection", "layer"):
        output = strata_summary(key, key)
        pd.DataFrame(output).to_csv(OUT / ("projection_summary.csv" if key == "projection" else "layer_summary.csv"),
                                   index=False, float_format="%.15g")

    regression = perdf[(perdf.heldout_gain_C < -.10) | (perdf.heldout_gain_G < -.10)].copy()
    regression.to_csv(OUT / "regression_cases.csv", index=False, float_format="%.15g")
    contribution_rows = []
    concentration = {}
    for method, sse_col in (("C", "heldout_C_sse"), ("G", "heldout_G_sse")):
        by_row = perdf.groupby("row_id").agg(D_sse=("heldout_D_sse", "sum"), method_sse=(sse_col, "sum"))
        by_row["net_sse_reduction"] = by_row.D_sse - by_row.method_sse
        by_row = by_row.sort_values("net_sse_reduction", ascending=False)
        net = float(by_row.net_sse_reduction.sum())
        by_row["share_of_net_reduction"] = by_row.net_sse_reduction / net if net else np.nan
        for rank, (rid, x) in enumerate(by_row.iterrows(), 1):
            contribution_rows.append({"method": method, "row_id": rid, "rank": rank,
                                      "D_sse": x.D_sse, "method_sse": x.method_sse,
                                      "net_sse_reduction": x.net_sse_reduction,
                                      "share_of_net_reduction": x.share_of_net_reduction})
        removal = {}
        for n in (1, 5, 10):
            kept = by_row.iloc[n:]
            removal[str(n)] = {"removed_rows": list(by_row.index[:n]),
                               "pooled_gain": 1 - float(kept.method_sse.sum()) / float(kept.D_sse.sum())}
        concentration[method] = {"top_shares": {str(n): float(by_row.net_sse_reduction.iloc[:n].sum() / net)
                                                 for n in (1, 5, 10)},
                                 "remove_top_rows": removal,
                                 "top_rows": list(by_row.index[:10])}
    pd.DataFrame(contribution_rows).to_csv(OUT / "pooled_contribution.csv", index=False, float_format="%.15g")

    # Independent raw-to-summary recomputation uses only JSON plus candidate weight artifacts.
    report_summary = {x["method"]: x for x in summary_rows}
    dsum = report_summary["D"]
    csum = report_summary["C"]
    gsum = report_summary["G"]
    pass_a = 2 * gsum["gain_lt_minus_10pct"] <= csum["gain_lt_minus_10pct"]
    pass_b = gsum["pooled_gain_vs_D"] > 0
    pass_c = (csum["pooled_gain_vs_D"] <= 0 or
              gsum["pooled_gain_vs_D"] / csum["pooled_gain_vs_D"] >= .5)
    decision_status = "GATE_V1_VALIDATION_PASS" if pass_a and pass_b and pass_c else "GATE_V1_VALIDATION_FAIL"
    snapshot = load_json(OUT / "phaseA_preservation_fingerprint.json") if (OUT / "phaseA_preservation_fingerprint.json").exists() else {}
    phase_unchanged = check_phase_a_snapshot(snapshot) if snapshot else False
    initial_lock_valid = (lock["gate_sha256"] == GATE_SHA and
                          lock["row_manifest_sha256"] == digest_file(OUT / "validation_row_manifest.json") and
                          lock["split_manifest_sha256"] == digest_file(OUT / "validation_split_manifest.json") and
                          lock["protocol_sha256"] == digest_file(OUT / "VALIDATION_PROTOCOL.json"))
    frozen_lock = lock.get("locked_at_epoch_ns", 0)
    row_frozen_before = (lock.get("row_manifest_mtime_ns", 2**63) < frozen_lock and
                         all(path.stat().st_mtime_ns >= frozen_lock for path in raw_files))
    split_frozen_before = (lock.get("split_manifest_mtime_ns", 2**63) < frozen_lock and
                           all(path.stat().st_mtime_ns >= frozen_lock for path in raw_files))
    # A second direct calculation from raw JSON verifies CSV/summary aggregation.
    direct_d = sum(float(rec["heldout_D_sse"]) for rec in raw_by_key.values())
    direct_c = sum(float(rec["heldout_C_sse"]) for rec in raw_by_key.values())
    direct_g = sum(float(rec["heldout_C_sse"] if decision_by_row[rec["row_id"]]["accept"]
                          else rec["heldout_D_sse"]) for rec in raw_by_key.values())
    report_summary_direct = {x["method"]: x for x in summary_rows}
    raw_summary_matches = (
        math.isclose(report_summary_direct["D"]["heldout_sse"], direct_d, rel_tol=1e-12) and
        math.isclose(report_summary_direct["C"]["heldout_sse"], direct_c, rel_tol=1e-12) and
        math.isclose(report_summary_direct["G"]["heldout_sse"], direct_g, rel_tol=1e-12))
    checks = {
        "gate_v1_sha256_exact": digest_file(OUT / "GATE_V1.json") == GATE_SHA,
        "validation_row_manifest_frozen_before_runs": row_frozen_before,
        "validation_split_manifest_frozen_before_runs": split_frozen_before,
        "validation_articles_disjoint_v8_v11": verify_split_disjointness(splits)["historical_overlap"] is False,
        "search_heldout_disjoint": verify_split_disjointness(splits)["search_heldout_disjoint"],
        "four_projection_alpha1_native_parity": all(x["status"] == "PASS" for x in load_json(OUT / "validation_preflight.json")["parity"].values()),
        "pristine_weight_isolation": all(r["pristine_weight_hash_before"] == r["pristine_weight_hash_after"] == r["weight_sha256"] for r in raw_by_key.values()),
        "candidate_state_isolation": len(landscape) == 4800 and all(len(r["candidates"]) == 10 for r in raw_by_key.values()),
        "gate_reads_search_side_only": all(d["gate_decision"] == ("ACCEPT" if d["search_gain_median"] >= .1 and d["search_margin_median"] >= .06 else "FALLBACK") for d in decisions),
        "heldout_never_enters_gate_decision": set(searchdf.columns) == gate_search_fields,
        "raw_to_summary_independent_recomputation": raw_summary_matches,
        "candidate_weight_hashes_valid": len(landscape) == 4800,
        "historical_v8_v12_phase_a_files_unchanged": phase_unchanged,
        "thresholds_unchanged_after_gate_freeze": initial_lock_valid and gate["tau_gain"] == .1 and gate["tau_margin"] == .06,
        "manifests_unchanged_after_first_candidate": initial_lock_valid,
    }
    correctness = {"status": "PASS" if all(checks.values()) else "FAIL",
                   "checks": checks, "all_checks_passed": sum(checks.values()),
                   "check_count": len(checks), "gate_sha256": digest_file(OUT / "GATE_V1.json"),
                   "row_manifest_sha256": digest_file(OUT / "validation_row_manifest.json"),
                   "split_manifest_sha256": digest_file(OUT / "validation_split_manifest.json"),
                   "raw_records": len(raw_by_key), "candidate_weight_records": len(landscape),
                   "created_at_utc": now_utc()}
    save_json(OUT / "VALIDATION_CORRECTNESS.json", correctness)
    final_status = decision_status if correctness["status"] == "PASS" else "VALIDATION_CORRECTNESS_FAIL_NO_SCIENTIFIC_PASS"
    final = {"status": final_status, "correctness_status": correctness["status"],
             "primary": {"A_severe_reduction_at_least_50pct": pass_a,
                         "B_gate_pooled_gain_positive": pass_b,
                         "C_retains_at_least_50pct_C_gain_when_applicable": pass_c},
             "metrics": {"D": dsum, "C": csum, "G": gsum},
             "accepted_rows": accepted, "fallback_rows": 48 - accepted,
             "pooled_gain_concentration": concentration,
             "projection_summary": pd.read_csv(OUT / "projection_summary.csv").to_dict(orient="records"),
             "layer_summary": pd.read_csv(OUT / "layer_summary.csv").to_dict(orient="records"),
             "costs": load_json(OUT / "validation_run_timing.json"),
             "correctness": correctness, "created_at_utc": now_utc()}
    save_json(OUT / "VALIDATION_SUMMARY.json", final)
    write_validation_report(final, concentration)
    return final


def write_validation_report(final: dict, concentration: dict) -> None:
    m = final["metrics"]
    cost = final["costs"]
    lines = ["# Frozen Gate v1 independent validation report", "",
             f"Final result: **{final['status']}**. Correctness: **{final['correctness_status']}**.", "",
             f"Gate SHA256: `{digest_file(OUT / 'GATE_V1.json')}` (expected `{GATE_SHA}`).",
             f"Frozen rows: {len(load_json(OUT / 'validation_row_manifest.json')['rows'])}; new complete articles: 80; split manifest SHA256 `{digest_file(OUT / 'validation_split_manifest.json')}`.",
             "", "## Primary D / C / Gate results", "",
             f"Gate decisions: {final['accepted_rows']} ACCEPT / {final['fallback_rows']} FALLBACK.",
             f"Uniform C pooled gain versus D: {m['C']['pooled_gain_vs_D']:.2%}.",
             f"Gate pooled gain versus D: {m['G']['pooled_gain_vs_D']:.2%}.",
             f"Gate retained C pooled gain: {m['G']['retained_C_gain'] if m['G']['retained_C_gain'] is not None else 'n/a'}.",
             f"Severe failures (<−10%): C {m['C']['gain_lt_minus_10pct']}/480; Gate {m['G']['gain_lt_minus_10pct']}/480; reduction {1-m['G']['gain_lt_minus_10pct']/max(1,m['C']['gain_lt_minus_10pct']):.2%}.",
             f"Primary criteria A/B/C: {final['primary']['A_severe_reduction_at_least_50pct']} / {final['primary']['B_gate_pooled_gain_positive']} / {final['primary']['C_retains_at_least_50pct_C_gain_when_applicable']}.",
             f"Worst regression: C {m['C']['worst_gain']:.2%}; Gate {m['G']['worst_gain']:.2%}.",
             "", "| Method | Pooled SSE gain vs D | <0 | <−5% | <−10% | <−20% | <−40% | <−50% | Median row-median gain |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for method in ("D", "C", "G"):
        x = m[method]
        lines.append(f"| {method} | {x['pooled_gain_vs_D']:.2%} | {x['gain_lt_0']} | {x['gain_lt_minus_5pct']} | {x['gain_lt_minus_10pct']} | {x['gain_lt_minus_20pct']} | {x['gain_lt_minus_40pct']} | {x['gain_lt_minus_50pct']} | {x['median_row_median_gain']:.2%} |")
    lines += ["", "## Projection breakdown", "",
              "| Projection | Rows | G accept/fallback | C pooled | Gate pooled | C severe | Gate severe | Worst C | Worst Gate |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for x in final["projection_summary"]:
        lines.append(f"| {x['stratum']} | {x['row_count']} | {x['gate_accept_rows']}/{x['gate_fallback_rows']} | {x['C_pooled_gain_vs_D']:.2%} | {x['G_pooled_gain_vs_D']:.2%} | {x['C_severe_failures']} | {x['G_severe_failures']} | {x['C_worst_gain']:.2%} | {x['G_worst_gain']:.2%} |")
    lines += ["", "## Layer breakdown", "",
              "| Layer | Rows | G accept/fallback | C pooled | Gate pooled | C severe | Gate severe | Worst C | Worst Gate |",
              "|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for x in final["layer_summary"]:
        lines.append(f"| {x['stratum']} | {x['row_count']} | {x['gate_accept_rows']}/{x['gate_fallback_rows']} | {x['C_pooled_gain_vs_D']:.2%} | {x['G_pooled_gain_vs_D']:.2%} | {x['C_severe_failures']} | {x['G_severe_failures']} | {x['C_worst_gain']:.2%} | {x['G_worst_gain']:.2%} |")
    lines += ["", "## Pooled gain concentration", ""]
    for method in ("C", "G"):
        c = concentration[method]
        lines.append(f"- {method}: top1/top5/top10 shares of net SSE reduction = " +
                     "/".join(f"{100*c['top_shares'][str(n)]:.1f}%" for n in (1, 5, 10)) + ".")
        for n in (1, 5, 10):
            lines.append(f"  - Remove top{n}: pooled gain {c['remove_top_rows'][str(n)]['pooled_gain']:.2%}.")
    lines += ["", "## Cost", "",
              f"- Completed candidate slots: {cost['completed_candidate_slots']}/{cost['candidate_slots_nominal']}; actual evaluation attempts including retries: {cost['candidate_evaluation_attempts_including_resumed_retries']}; completed slots reused from cache: {cost['candidate_slots_reused_from_completed_cache']}.",
              f"- Candidate GPU time sum: {cost['candidate_seconds_sum_cumulative']:.1f}s; Hessian build time sum: {cost['hessian_seconds_sum_cumulative_unique_keys']:.1f}s.",
              f"- Preflight: {cost['preflight_total_seconds']:.1f}s; end-to-end wall time including preflight and model-input cache: {cost['total_wall_seconds_including_preflight_and_cache_build']:.1f}s.",
              f"- Peak CUDA allocated/reserved: {cost['peak_cuda_allocated_bytes']/2**20:.1f}/{cost['peak_cuda_reserved_bytes']/2**20:.1f} MiB."]
    lines += ["", "## Correctness and interpretation", "",
              f"Full correctness checklist: {final['correctness']['all_checks_passed']}/{final['correctness']['check_count']} passed; see `VALIDATION_CORRECTNESS.json` for each item.",
              "D is native dynamic GPTQ alpha=1. C chooses each split's search-best alpha. Gate makes one row-level decision from the two frozen search-only medians; held-out values are used only after the decisions are materialized. The 480 row/split observations are repeated evaluations on 48 rows, not 480 independent model samples.",
              "", "No PPL, quantized-prefix or downstream evaluation was run. This report ends at the frozen validation decision; no threshold adjustment is permitted.", ""]
    (OUT / "VALIDATION_REPORT.md").write_text("\n".join(lines), encoding="utf-8", newline="\n")


def main() -> None:
    args = PARSE.parse_args()
    selected = sum((args.preflight, args.run, args.summarize))
    if selected != 1:
        raise SystemExit("Choose exactly one of --preflight, --run, --summarize")
    manifests()
    if args.preflight:
        result = preflight()
        print(json.dumps({"status": result["status"], "parity_status": result["parity_status"],
                          "nominal_candidates": result["nominal_candidates"],
                          "reusable_candidates": result["reusable_candidates"],
                          "new_candidates": result["new_candidates"],
                          "estimated_formal_wall_hours": result["estimated_formal_wall_hours"]}, indent=2))
    elif args.run:
        result = formal_run()
        print(json.dumps(result, indent=2))
    else:
        result = summarise()
        print(json.dumps({"status": result["status"], "correctness": result["correctness_status"],
                          "accepted_rows": result["accepted_rows"], "fallback_rows": result["fallback_rows"],
                          "C_pooled_gain": result["metrics"]["C"]["pooled_gain_vs_D"],
                          "G_pooled_gain": result["metrics"]["G"]["pooled_gain_vs_D"],
                          "C_severe": result["metrics"]["C"]["gain_lt_minus_10pct"],
                          "G_severe": result["metrics"]["G"]["gain_lt_minus_10pct"]}, indent=2))


if __name__ == "__main__":
    main()
