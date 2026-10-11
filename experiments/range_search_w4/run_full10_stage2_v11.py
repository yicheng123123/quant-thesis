"""Resumable v11 Stage 2: frozen 96 rows on S03-S09 only.

This runner writes new experiment artifacts only beneath
experiments/range_search_w4/results/range_w4_96row_v11/. It never edits the
frozen Stage 1 records or the v8-v10 source runs.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import math
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
os.environ.update({
    "HF_HOME": str(ROOT / ".hf_cache"),
    "HF_HUB_CACHE": str(ROOT / ".hf_cache" / "hub"),
    "HF_HUB_OFFLINE": "1",
    "HF_DATASETS_OFFLINE": "1",
})
sys.path.insert(0, str(ROOT))

import torch
from transformers import AutoModelForCausalLM

from experiments.range_search_w4 import run_96row_v11 as base
from experiments.range_search_w4.dynamic_core import native_hinv_from_documents
from experiments.range_search_w4.run_task0_parity import tensor_hash

HERE = Path(__file__).resolve().parent
OUT = HERE / "results" / "range_w4_96row_v11"
V9 = HERE / "results" / "range_w4_l23r448_v9"
V10 = HERE / "results" / "range_w4_sixrow_v10"
STAGE2 = OUT / "raw" / "stage2"
WEIGHTS = STAGE2 / "weights"
HESSIAN = OUT / "stage2" / "hessian_cache"
ACTIVATION_CACHE = OUT / "stage2_activation_cache.pt"
PREFLIGHT_PATH = OUT / "stage2_preflight.json"
TIMING_PATH = OUT / "stage2_run_timing.json"
SPLIT_IDS = tuple(f"S{i:02d}" for i in range(3, 10))
EXPECTED_ROWS = 96
EXPECTED_SPLIT_HASH = base.EXPECTED_V9_MANIFEST
EXPECTED_ROW_HASH = "c56eeaa1d1af6e4ed8dd78f811c811c66758b215659a98bf8550052c786e6106"


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def save_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def load_tensor_cache(path: Path) -> dict:
    with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
        return torch.load(path, map_location="cpu", weights_only=True)


def verify_stage1_preservation() -> dict:
    reference = base.read_json(OUT / "stage1_preservation_fingerprint.json")
    checks = {}
    for rel, expected in reference["files"].items():
        path = OUT / rel
        checks[rel] = path.is_file() and digest(path.read_bytes()) == expected
    return {"status": "PASS" if all(checks.values()) else "FAIL",
            "checked_file_count": len(checks), "failed_files": [k for k, v in checks.items() if not v]}


def validate_amendment_and_manifests() -> tuple[dict, dict, dict]:
    amendment = OUT / "STAGE2_PROTOCOL_AMENDMENT.md"
    if not amendment.is_file():
        raise RuntimeError("frozen Stage 2 amendment missing")
    amendment_text = amendment.read_text(encoding="utf-8")
    required_phrases = ("before any v11 S03–S09 activation capture or candidate evaluation",
                        "47 `Stage1 candidate-positive`, 48 `Stage1 mixed-risk`, and 1 `Stage1 neutral/negative`",
                        "This amendment was made before any S03–S09 run")
    if any(s not in amendment_text for s in required_phrases):
        raise RuntimeError("protocol amendment lacks required pre-run rationale")
    rows = base.read_json(OUT / "row_manifest.json")
    protocol = base.read_json(OUT / "protocol.json")
    manifest_path = V9 / "splits_manifest.json"
    if digest(manifest_path.read_bytes()) != EXPECTED_SPLIT_HASH:
        raise RuntimeError("frozen v9 split manifest hash changed")
    if digest((OUT / "row_manifest.json").read_bytes()) != EXPECTED_ROW_HASH:
        raise RuntimeError("frozen row manifest changed")
    if len(rows.get("rows", [])) != EXPECTED_ROWS:
        raise RuntimeError("expected exactly 96 frozen rows")
    if protocol.get("row_manifest_sha256") != EXPECTED_ROW_HASH:
        raise RuntimeError("Stage 1 protocol references another row manifest")
    v9 = base.read_json(manifest_path)
    if [s["split_id"] for s in v9["splits"]] != [f"S{i:02d}" for i in range(10)]:
        raise RuntimeError("v9 frozen split order is not S00-S09")
    # Protocol invariants inherited from the completed v11 Stage 1.
    frozen = {
        "model": "Qwen/Qwen2.5-0.5B-Instruct",
        "revision": base.REVISION,
        "model_dtype": "float16",
        "math_dtype": "float32",
        "weight_quantization": "symmetric uniform W4A16",
        "group_size": 128, "block_size": 128, "damp_percent": 0.1,
        "desc_act": False, "static_groups": False, "true_sequential": True,
        "alpha_grid": base.GRID,
    }
    for key, value in frozen.items():
        if protocol.get(key) != value:
            raise RuntimeError(f"frozen protocol mismatch: {key}: {protocol.get(key)!r} != {value!r}")
    return rows, v9, protocol


def check_split_separation(v9: dict) -> dict:
    checks = {}
    split_map = {s["split_id"]: s for s in v9["splits"]}
    all_expected = {f"S{i:02d}" for i in range(10)}
    checks["all_10_frozen_splits_present"] = set(split_map) == all_expected
    for sid in SPLIT_IDS:
        split = split_map[sid]
        search, held = split["search"], split["heldout"]
        search_articles = {d["source_article_index"] for d in search}
        held_articles = {d["source_article_index"] for d in held}
        search_tokens = {d["token_sha256"] for d in search}
        held_tokens = {d["token_sha256"] for d in held}
        checks[f"{sid}_search_heldout_article_disjoint"] = not (search_articles & held_articles)
        checks[f"{sid}_search_heldout_token_disjoint"] = not (search_tokens & held_tokens)
        checks[f"{sid}_article_count_4_each"] = len(search) == 4 and len(held) == 4
        checks[f"{sid}_token_ids_hash_valid"] = all(
            digest(json.dumps(d["token_ids"]).encode()) == d["token_sha256"] for d in search + held)
    return {"status": "PASS" if all(checks.values()) else "FAIL", "checks": checks}


def documents_for_stage2(v9: dict) -> dict:
    docs = {}
    split_map = {s["split_id"]: s for s in v9["splits"]}
    for sid in SPLIT_IDS:
        for side in ("search", "heldout"):
            for doc in split_map[sid][side]:
                token_hash = doc["token_sha256"]
                if digest(json.dumps(doc["token_ids"]).encode()) != token_hash:
                    raise RuntimeError(f"frozen token IDs failed hash check: {sid}/{side}/{token_hash}")
                prior = docs.get(token_hash)
                if prior is not None and prior["token_ids"] != doc["token_ids"]:
                    raise RuntimeError("token hash collision in frozen manifest")
                docs[token_hash] = doc
    if len(docs) != 56:
        raise RuntimeError(f"expected 56 unique S03-S09 docs, found {len(docs)}")
    return docs


@torch.inference_mode()
def capture_stage2_inputs(rows: dict, v9: dict) -> tuple[dict, dict, dict]:
    row_hash = digest((OUT / "row_manifest.json").read_bytes())
    docs_by_hash = documents_for_stage2(v9)
    row_cache = load_tensor_cache(OUT / "activation_cache.pt")
    if row_cache.get("row_manifest_sha256") != row_hash or row_cache.get("v9_manifest_sha256") != EXPECTED_SPLIT_HASH:
        raise RuntimeError("Stage 1 activation cache provenance mismatch")
    weights = row_cache["weights"]
    expected_weight_hashes = {base.row_key(r): r["weight_hash"] for r in rows["rows"]}
    actual_weight_hashes = {k: tensor_hash(v) for k, v in weights.items()}
    if actual_weight_hashes != expected_weight_hashes:
        raise RuntimeError("pristine W cache hashes differ from frozen row manifest")
    if ACTIVATION_CACHE.exists():
        cache = load_tensor_cache(ACTIVATION_CACHE)
        if cache.get("row_manifest_sha256") != row_hash or cache.get("v9_manifest_sha256") != EXPECTED_SPLIT_HASH:
            raise RuntimeError("Stage 2 activation cache provenance mismatch")
        if cache.get("split_ids") != list(SPLIT_IDS) or cache.get("document_token_hashes") != sorted(docs_by_hash):
            raise RuntimeError("Stage 2 activation cache document index mismatch")
        inputs = cache["inputs"]
        if any(set(inputs[base.input_key(l, p)]) != set(docs_by_hash)
               for l in base.LAYERS for p in base.PROJECTIONS):
            raise RuntimeError("Stage 2 activation cache is incomplete")
        return inputs, weights, cache["metadata"]

    if not torch.cuda.is_available():
        raise RuntimeError("Stage 2 requires CUDA")
    torch.manual_seed(42)
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        base.SNAPSHOT, torch_dtype=torch.float16, local_files_only=True,
        device_map="cuda", low_cpu_mem_usage=True,
    )
    model.eval()
    inputs = {base.input_key(layer, proj): {} for layer in base.LAYERS for proj in base.PROJECTIONS}
    current_hash = [None]
    handles = []
    for layer in base.LAYERS:
        for projection in base.PROJECTIONS:
            module = base.module_for(model, layer, projection)
            key = base.input_key(layer, projection)

            def hook(_module, args, _output, cache_key=key):
                token_hash = current_hash[0]
                if token_hash is None:
                    raise RuntimeError("projection hook ran without an active frozen document")
                inputs[cache_key][token_hash] = args[0].detach().reshape(-1, args[0].shape[-1]).float().cpu().clone()

            handles.append(module.register_forward_hook(hook))
    try:
        for token_hash, doc in docs_by_hash.items():
            current_hash[0] = token_hash
            ids = torch.tensor(doc["token_ids"], dtype=torch.long, device="cuda")[None]
            model(input_ids=ids, use_cache=False)
    finally:
        for handle in handles:
            handle.remove()
    base.sync()
    metadata = {
        "capture_seconds": time.perf_counter() - started,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "document_count": len(docs_by_hash), "device_name": torch.cuda.get_device_name(0),
        "torch_version": str(torch.__version__),
        "transformers_version": importlib.metadata.version("transformers"),
        "gptqmodel_version": importlib.metadata.version("gptqmodel"),
        "weight_hashes": actual_weight_hashes,
    }
    for key, values in inputs.items():
        if set(values) != set(docs_by_hash):
            raise RuntimeError(f"projection input capture incomplete for {key}: {len(values)} docs")
    del model
    torch.cuda.empty_cache()
    cache = {"row_manifest_sha256": row_hash, "v9_manifest_sha256": EXPECTED_SPLIT_HASH,
             "split_ids": list(SPLIT_IDS), "document_token_hashes": sorted(docs_by_hash),
             "inputs": inputs, "metadata": metadata}
    ACTIVATION_CACHE.parent.mkdir(parents=True, exist_ok=True)
    tmp = ACTIVATION_CACHE.with_suffix(".pt.tmp")
    torch.save(cache, tmp)
    tmp.replace(ACTIVATION_CACHE)
    return inputs, weights, metadata


def get_hinv_stage2(row: dict, split: dict, inputs: dict, weights: dict) -> tuple[torch.Tensor, dict]:
    layer, projection, sid = row["layer"], row["projection"], split["split_id"]
    docs = base.docs_for_side(inputs, layer, projection, split, "search")
    x = torch.cat(docs)
    input_sha = tensor_hash(x)
    path = HESSIAN / f"hinv_L{layer}_{projection}_{sid}.pt"
    if path.exists():
        obj = torch.load(path, map_location="cpu", weights_only=True)
        if (obj.get("row_manifest_sha256") != EXPECTED_ROW_HASH
                or obj.get("v9_manifest_sha256") != EXPECTED_SPLIT_HASH
                or obj.get("search_x_sha256") != input_sha):
            raise RuntimeError(f"Stage 2 Hessian cache mismatch: {path.name}")
        return obj["hinv"].to("cuda"), {**{k: v for k, v in obj.items() if k != "hinv"}, "cache_reused": True}
    sample = next(r for r in base.row_manifest_global["rows"]
                  if r["layer"] == layer and r["projection"] == projection)
    w = weights[base.row_key(sample)].cuda()
    module = torch.nn.Linear(row["input_dim"], 1, bias=False, device="cuda", dtype=torch.float16)
    module.weight.data.copy_(w[None].half())
    base.empty_peak_stats()
    (h, hinv, dead), seconds = base.timed_call(lambda: native_hinv_from_documents(docs, module, .1))
    if dead:
        raise RuntimeError(f"dead GPTQ columns for {layer}/{projection}/{sid}: {dead}")
    item = {
        "layer": layer, "projection": projection, "split_id": sid,
        "row_manifest_sha256": EXPECTED_ROW_HASH, "v9_manifest_sha256": EXPECTED_SPLIT_HASH,
        "search_x_sha256": input_sha, "input_dim": row["input_dim"], "dead_columns": dead,
        "hessian_build_seconds": seconds,
        "hessian_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "hessian_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "diag_h_summary": base.tensor_summary(h.diagonal()),
        "diag_hinv_cholesky_summary": base.tensor_summary(hinv.diagonal()), "cache_reused": False,
    }
    HESSIAN.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".pt.tmp")
    torch.save({**item, "hinv": hinv.detach().cpu()}, tmp)
    tmp.replace(path)
    return hinv, item


def check_v10_reuse(rows: dict, v9: dict, inputs: dict) -> dict:
    """Return only exact v10 records whose manifest, X and candidate weights match."""
    expected_protocol = base.read_json(V10 / "protocol.json")
    invariant_protocol = (
        expected_protocol.get("revision") == base.REVISION
        and expected_protocol.get("quantization") == "W4A16 symmetric G128"
        and expected_protocol.get("group_size") == 128
        and expected_protocol.get("block_size") == 128
        and expected_protocol.get("damp_percent") == .1
        and expected_protocol.get("desc_act") is False
        and expected_protocol.get("alpha_grid") == base.GRID
        and expected_protocol.get("frozen_splits_manifest_sha256") == EXPECTED_SPLIT_HASH
    )
    if not invariant_protocol:
        return {}
    split_map = {s["split_id"]: s for s in v9["splits"]}
    reused = {}
    for row in rows["rows"]:
        if row["layer"] != 12 or row["projection"] != "q_proj" or row["row_index"] != 448:
            continue
        for sid in SPLIT_IDS:
            split = split_map[sid]
            src_path = V10 / f"L{row['layer']}R{row['row_index']}_{sid}.json"
            weight_path = V10 / f"weights_L{row['layer']}R{row['row_index']}_{sid}.pt"
            if not src_path.is_file() or not weight_path.is_file():
                continue
            src = base.read_json(src_path)
            expected = {
                "row": f"L{row['layer']}R{row['row_index']}", "layer": row["layer"],
                "row_index": row["row_index"], "projection": "self_attn.q_proj", "split_id": sid,
                "manifest_sha256": EXPECTED_SPLIT_HASH, "weight_sha256": row["weight_hash"],
                "search_article_ids": [d["source_article_index"] for d in split["search"]],
                "heldout_article_ids": [d["source_article_index"] for d in split["heldout"]],
            }
            if any(src.get(k) != v for k, v in expected.items()):
                continue
            if src.get("source") != "v10_run" or [float(c["alpha"]) for c in src.get("candidates", [])] != base.GRID:
                continue
            xs = base.x_for_split(inputs, row["layer"], row["projection"], split, "search", device="cpu")
            xh = base.x_for_split(inputs, row["layer"], row["projection"], split, "heldout", device="cpu")
            if src.get("search_x_sha256") != tensor_hash(xs) or src.get("heldout_x_sha256") != tensor_hash(xh):
                continue
            weights_obj = torch.load(weight_path, map_location="cpu", weights_only=True)
            qweights = weights_obj.get("weights", {})
            if weights_obj.get("row_manifest_sha256") not in (None, EXPECTED_ROW_HASH):
                continue
            if any(str(c["alpha"]) not in qweights or tensor_hash(qweights[str(c["alpha"])]) != c["weight_sha256"]
                   for c in src["candidates"]):
                continue
            reused[(row["layer"], row["projection"], row["row_index"], sid)] = {
                "source": src, "raw_path": src_path, "weights_path": weight_path,
                "raw_sha256": digest(src_path.read_bytes()),
                "weights_sha256": digest(weight_path.read_bytes()),
            }
    return reused


def stage2_existing(rows: dict, v9: dict, inputs: dict) -> dict:
    split_map = {s["split_id"]: s for s in v9["splits"]}
    existing = {}
    for row in rows["rows"]:
        for sid in SPLIT_IDS:
            slug = base.result_slug(row["layer"], row["projection"], row["row_index"], sid)
            path = STAGE2 / f"{slug}.json"
            if not path.exists():
                continue
            record = base.read_json(path)
            split = split_map[sid]
            xs = base.x_for_split(inputs, row["layer"], row["projection"], split, "search", device="cpu")
            xh = base.x_for_split(inputs, row["layer"], row["projection"], split, "heldout", device="cpu")
            checks = {
                "row_id": base.row_key(row), "split_id": sid,
                "row_manifest_sha256": EXPECTED_ROW_HASH, "v9_manifest_sha256": EXPECTED_SPLIT_HASH,
                "weight_sha256": row["weight_hash"],
                "search_article_ids": [d["source_article_index"] for d in split["search"]],
                "heldout_article_ids": [d["source_article_index"] for d in split["heldout"]],
                "search_x_sha256": tensor_hash(xs), "heldout_x_sha256": tensor_hash(xh),
            }
            if any(record.get(k) != v for k, v in checks.items()):
                raise RuntimeError(f"existing Stage 2 record provenance mismatch: {path.name}")
            if [float(c["alpha"]) for c in record.get("candidates", [])] != base.GRID:
                raise RuntimeError(f"existing Stage 2 candidate grid mismatch: {path.name}")
            if record.get("source") not in ("v11_stage2", "existing_v10"):
                raise RuntimeError(f"unsupported Stage 2 record source: {path.name}")
            weight_pointer = record.get("weights_path") or record.get("source_weights_path")
            if not weight_pointer:
                raise RuntimeError(f"Stage 2 candidate weights pointer missing: {path.name}")
            wp = ROOT / weight_pointer
            if not wp.is_file():
                raise RuntimeError(f"Stage 2 candidate weights missing: {wp}")
            obj = torch.load(wp, map_location="cpu", weights_only=True)
            qweights = obj.get("weights", {})
            if any(tensor_hash(qweights[str(c["alpha"])]) != c["weight_sha256"] for c in record["candidates"]):
                raise RuntimeError(f"Stage 2 candidate weights hash mismatch: {wp.name}")
            existing[(row["layer"], row["projection"], row["row_index"], sid)] = record
    return existing


def run_preflight() -> None:
    rows, v9, protocol = validate_amendment_and_manifests()
    preservation_before = verify_stage1_preservation()
    if preservation_before["status"] != "PASS":
        raise RuntimeError(f"Stage 1 preservation check failed: {preservation_before['failed_files'][:5]}")
    split_sep = check_split_separation(v9)
    if split_sep["status"] != "PASS":
        raise RuntimeError("S03-S09 split separation or token hash check failed")
    base.row_manifest_global = rows
    inputs, weights, capture = capture_stage2_inputs(rows, v9)
    reusable_sources = check_v10_reuse(rows, v9, inputs)
    existing = stage2_existing(rows, v9, inputs)
    nominal = EXPECTED_ROWS * len(SPLIT_IDS) * len(base.GRID)
    # Existing Stage 2 records take precedence; matching v10 sources are only
    # counted as reusable work when that row/split has not already been saved.
    reused = {k: v for k, v in reusable_sources.items() if k not in existing}
    remaining = EXPECTED_ROWS * len(SPLIT_IDS) - len(existing) - len(reused)
    new_candidates = remaining * len(base.GRID)

    # One native alpha=1 parity spot per projection, fixed to the first row
    # in each projection and S03 before seeing any Stage 2 result.
    split_map = {s["split_id"]: s for s in v9["splits"]}
    parity = []
    for proj in base.PROJECTIONS:
        row = next(r for r in rows["rows"] if r["projection"] == proj)
        hinv, _ = get_hinv_stage2(row, split_map["S03"], inputs, weights)
        parity.append(base.native_parity(row, split_map["S03"], inputs, weights, hinv))
    if not all(p["status"] == "PASS" for p in parity):
        raise RuntimeError("S03-S09 alpha=1 native parity failed; formal Stage 2 stopped")

    # Repeating an alpha candidate on the same pristine tensors proves that a
    # previous candidate did not mutate W or Hinv. These two diagnostic calls
    # are counted separately from the 6,720-point landscape.
    sample = next(r for r in rows["rows"] if r["projection"] == "q_proj")
    hinv, _ = get_hinv_stage2(sample, split_map["S03"], inputs, weights)
    xs = base.x_for_split(inputs, sample["layer"], sample["projection"], split_map["S03"], "search")
    xh = base.x_for_split(inputs, sample["layer"], sample["projection"], split_map["S03"], "heldout")
    w = weights[base.row_key(sample)].cuda()
    w_before, h_before = tensor_hash(w), tensor_hash(hinv)
    a, qa = base.custom_candidate(w, hinv, xs, xh, .7)
    b, qb = base.custom_candidate(w, hinv, xs, xh, .7)
    candidate_isolation = {
        "same_output_hash": a["weight_sha256"] == b["weight_sha256"] == tensor_hash(qa) == tensor_hash(qb),
        "same_scale": a["scale"] == b["scale"], "same_zero": a["zero"] == b["zero"],
        "same_search_sse": a["search_sse"] == b["search_sse"],
        "pristine_weight_unchanged": tensor_hash(w) == w_before,
        "hinv_unchanged": tensor_hash(hinv) == h_before,
    }
    if not all(candidate_isolation.values()):
        raise RuntimeError(f"candidate-state isolation failed: {candidate_isolation}")

    # Estimate from actual Stage 1 per-alpha timing; v10 reuse and any valid
    # resumable Stage 2 records are excluded from new formal work.
    stage1_raw = [base.read_json(p) for p in (OUT / "raw").glob("L*_S??.json")]
    seconds_by_proj = defaultdict(list)
    for rec in stage1_raw:
        if rec["source"] == "v11_run":
            for c in rec["candidates"]:
                seconds_by_proj[rec["projection_short"]].append(float(c["candidate_seconds"]))
    new_count_by_proj = Counter()
    for row in rows["rows"]:
        n = sum((row["layer"], row["projection"], row["row_index"], sid) not in reused
                and (row["layer"], row["projection"], row["row_index"], sid) not in existing
                for sid in SPLIT_IDS) * len(base.GRID)
        new_count_by_proj[row["projection"]] += n
    projected_candidate_seconds = {
        proj: (sum(seconds_by_proj[proj]) / len(seconds_by_proj[proj]) * new_count_by_proj[proj]
               if seconds_by_proj[proj] else None) for proj in base.PROJECTIONS}
    estimated_raw = sum(v for v in projected_candidate_seconds.values() if v is not None)
    estimate_safety = 1.5 * estimated_raw
    if (new_candidates > 0 and estimated_raw <= 0) or any(v is None for v in projected_candidate_seconds.values()):
        raise RuntimeError("unable to derive Stage 2 candidate cost from Stage 1 timings")

    checks = {
        "row_manifest_hash_unchanged": digest((OUT / "row_manifest.json").read_bytes()) == EXPECTED_ROW_HASH,
        "split_manifest_hash_unchanged": digest((V9 / "splits_manifest.json").read_bytes()) == EXPECTED_SPLIT_HASH,
        "stage1_preserved_before_stage2": preservation_before["status"] == "PASS",
        "stage2_search_heldout_separation": split_sep["status"] == "PASS",
        "pristine_weight_hashes_match": actual_weight_hashes_match(rows, weights),
        "candidate_state_isolation": all(candidate_isolation.values()),
        "alpha1_native_parity_all_projections": all(p["status"] == "PASS" for p in parity),
        "each_v10_reused_record_passed_hash_gate": all(
            bool(entry["raw_sha256"] and entry["weights_sha256"]) for entry in reusable_sources.values()),
        "row_split_accounting_complete": len(existing) + len(reused) + remaining == EXPECTED_ROWS * len(SPLIT_IDS),
    }
    if not all(checks.values()):
        raise RuntimeError(f"preflight correctness failure: {checks}")
    result = {
        "status": "PASS", "row_manifest_sha256": EXPECTED_ROW_HASH,
        "v9_manifest_sha256": EXPECTED_SPLIT_HASH, "split_ids": list(SPLIT_IDS),
        "nominal_candidate_evaluations": nominal,
        "existing_stage2_candidate_evaluations": len(existing) * len(base.GRID),
        "reusable_v10_candidate_evaluations": len(reused) * len(base.GRID),
        "new_candidate_evaluations": new_candidates,
        "existing_stage2_row_splits": len(existing), "reusable_v10_row_splits": len(reused),
        "new_row_splits": remaining,
        "reused_v10_row_split_ids": [f"{base.row_key(next(r for r in rows['rows'] if r['layer']==k[0] and r['projection']==k[1] and r['row_index']==k[2]))}/{k[3]}" for k in sorted(reused)],
        "new_candidate_counts_by_projection": dict(new_count_by_proj),
        "stage1_measured_mean_candidate_seconds": {p: sum(v)/len(v) for p, v in seconds_by_proj.items()},
        "estimated_candidate_seconds_by_projection": projected_candidate_seconds,
        "estimated_candidate_seconds_raw": estimated_raw,
        "estimated_candidate_wall_seconds_with_1_5x_margin": estimate_safety,
        "estimated_candidate_wall_hours_with_margin": estimate_safety / 3600,
        "diagnostic_evaluations_outside_formal_count": {
            "custom_alpha1_in_native_parity": len(parity), "repeat_alpha07_isolation": 2},
        "capture": capture, "parity": parity, "candidate_isolation": candidate_isolation,
        "split_separation": split_sep, "stage1_preservation": preservation_before,
        "checks": checks, "python_torch_cuda": {
            "python": sys.version, "torch": str(torch.__version__),
            "cuda_available": torch.cuda.is_available(),
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "total_memory_bytes": torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else None,
        },
    }
    # Preserve the pre-execution preflight as the run's frozen planning record.
    # Later checks write a separate resume snapshot instead of overwriting it.
    preflight_output = PREFLIGHT_PATH if not PREFLIGHT_PATH.exists() else OUT / "stage2_resume_preflight.json"
    result["preflight_artifact"] = str(preflight_output.relative_to(ROOT))
    save_json(preflight_output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


def actual_weight_hashes_match(rows: dict, weights: dict) -> bool:
    return {base.row_key(r): tensor_hash(weights[base.row_key(r)]) for r in rows["rows"]} == {
        base.row_key(r): r["weight_hash"] for r in rows["rows"]}


def run_stage2() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("Stage 2 requires CUDA")
    rows, v9, _protocol = validate_amendment_and_manifests()
    preservation = verify_stage1_preservation()
    if preservation["status"] != "PASS":
        raise RuntimeError(f"Stage 1 files changed before Stage 2: {preservation['failed_files'][:5]}")
    if not PREFLIGHT_PATH.is_file():
        raise RuntimeError("Stage 2 preflight is missing; run --preflight first")
    preflight = base.read_json(PREFLIGHT_PATH)
    if preflight.get("status") != "PASS" or not all(preflight.get("checks", {}).values()):
        raise RuntimeError("Stage 2 preflight did not pass")
    base.row_manifest_global = rows
    inputs, weights, capture = capture_stage2_inputs(rows, v9)
    split_map = {s["split_id"]: s for s in v9["splits"]}
    reused = check_v10_reuse(rows, v9, inputs)
    existing = stage2_existing(rows, v9, inputs)
    if len(existing) == EXPECTED_ROWS * len(SPLIT_IDS):
        preservation_after = verify_stage1_preservation()
        if preservation_after["status"] != "PASS":
            raise RuntimeError("Stage 1 preservation failed while verifying completed Stage 2")
        prior_timing = base.read_json(TIMING_PATH) if TIMING_PATH.is_file() else {}
        print(json.dumps({"stage2": "ALREADY_COMPLETE", "row_split_records": len(existing),
                          "durable_new_candidate_evaluations": sum(
                              int(r["candidate_evaluations_this_run"]) for r in existing.values()
                              if r["source"] == "v11_stage2"),
                          "prior_timing_preserved": bool(prior_timing),
                          "stage1_preservation": preservation_after}, ensure_ascii=False, indent=2), flush=True)
        return
    start = time.perf_counter()
    candidate_seconds = 0.0
    hessian_seconds = 0.0
    newly_evaluated = 0
    newly_completed_records = 0
    max_alloc = 0
    max_reserved = 0
    STAGE2.mkdir(parents=True, exist_ok=True)
    WEIGHTS.mkdir(parents=True, exist_ok=True)
    for row in rows["rows"]:
        layer, projection, index = row["layer"], row["projection"], row["row_index"]
        row_id = base.row_key(row)
        w = weights[row_id].cuda()
        if tensor_hash(w) != row["weight_hash"]:
            raise RuntimeError(f"pristine W mismatch: {row_id}")
        for sid in SPLIT_IDS:
            key = (layer, projection, index, sid)
            slug = base.result_slug(layer, projection, index, sid)
            raw_path, weight_path = STAGE2 / f"{slug}.json", WEIGHTS / f"{slug}.pt"
            split = split_map[sid]
            xs = base.x_for_split(inputs, layer, projection, split, "search")
            xh = base.x_for_split(inputs, layer, projection, split, "heldout")
            if key in existing:
                continue
            if key in reused:
                src_entry = reused[key]
                src = src_entry["source"]
                candidates = [{k: c[k] for k in (
                    "alpha", "search_sse", "heldout_sse", "search_output_mse", "heldout_output_mse",
                    "weight_mse", "scale", "zero", "candidate_seconds", "weight_sha256",
                    "candidate_peak_allocated_bytes", "candidate_peak_reserved_bytes") if k in c}
                    for c in src["candidates"]]
                selected = min(candidates, key=lambda c: c["search_output_mse"])
                baseline = next(c for c in candidates if float(c["alpha"]) == 1.0)
                record = {
                    "row_id": row_id, "layer": layer, "projection": row["projection_name"],
                    "projection_short": projection, "row_index": index, "output_dim": row["output_dim"],
                    "input_dim": row["input_dim"], "split_id": sid, "source": "existing_v10",
                    "source_v10_raw": str(src_entry["raw_path"].relative_to(ROOT)),
                    "source_v10_raw_sha256": src_entry["raw_sha256"],
                    "source_weights_path": str(src_entry["weights_path"].relative_to(ROOT)),
                    "source_weights_sha256": src_entry["weights_sha256"],
                    "row_manifest_sha256": EXPECTED_ROW_HASH, "v9_manifest_sha256": EXPECTED_SPLIT_HASH,
                    "weight_sha256": row["weight_hash"],
                    "search_article_ids": [d["source_article_index"] for d in split["search"]],
                    "heldout_article_ids": [d["source_article_index"] for d in split["heldout"]],
                    "search_x_sha256": tensor_hash(xs), "heldout_x_sha256": tensor_hash(xh),
                    "selected_alpha": selected["alpha"],
                    "search_D_sse": baseline["search_sse"], "search_D_mse": baseline["search_output_mse"],
                    "search_C_sse": selected["search_sse"], "search_C_mse": selected["search_output_mse"],
                    "heldout_D_sse": baseline["heldout_sse"], "heldout_D_mse": baseline["heldout_output_mse"],
                    "heldout_C_sse": selected["heldout_sse"], "heldout_C_mse": selected["heldout_output_mse"],
                    "search_gain": 1 - selected["search_sse"] / baseline["search_sse"],
                    "held_gain": 1 - selected["heldout_sse"] / baseline["heldout_sse"],
                    "candidate_evaluations_this_run": 0, "candidate_seconds_total": 0.0,
                    "hessian_seconds": 0.0, "candidates": candidates,
                }
                save_json(raw_path, record)
                newly_completed_records += 1
                print(f"REUSED {row_id} {sid} from v10", flush=True)
                continue

            hinv, hdiag = get_hinv_stage2(row, split, inputs, weights)
            if not hdiag.get("cache_reused", False):
                hessian_seconds += float(hdiag["hessian_build_seconds"])
            pristine_hash, hinv_hash = tensor_hash(w), tensor_hash(hinv)
            candidates, qweights = [], {}
            this_seconds = 0.0
            for alpha in base.GRID:
                item, q = base.custom_candidate(w, hinv, xs, xh, alpha)
                candidates.append(item)
                qweights[str(alpha)] = q
                this_seconds += item["candidate_seconds"]
                candidate_seconds += item["candidate_seconds"]
                newly_evaluated += 1
                max_alloc = max(max_alloc, item["candidate_peak_allocated_bytes"])
                max_reserved = max(max_reserved, item["candidate_peak_reserved_bytes"])
            if tensor_hash(w) != pristine_hash or tensor_hash(hinv) != hinv_hash:
                raise RuntimeError(f"candidate state mutated for {row_id}/{sid}")
            chosen = min(candidates, key=lambda c: c["search_output_mse"])
            baseline = next(c for c in candidates if float(c["alpha"]) == 1.0)
            torch_tmp = weight_path.with_suffix(".pt.tmp")
            torch.save({"row_manifest_sha256": EXPECTED_ROW_HASH, "weights": qweights}, torch_tmp)
            torch_tmp.replace(weight_path)
            record = {
                "row_id": row_id, "layer": layer, "projection": row["projection_name"],
                "projection_short": projection, "row_index": index, "output_dim": row["output_dim"],
                "input_dim": row["input_dim"], "split_id": sid, "source": "v11_stage2",
                "row_manifest_sha256": EXPECTED_ROW_HASH, "v9_manifest_sha256": EXPECTED_SPLIT_HASH,
                "weight_sha256": pristine_hash,
                "search_article_ids": [d["source_article_index"] for d in split["search"]],
                "heldout_article_ids": [d["source_article_index"] for d in split["heldout"]],
                "search_x_sha256": tensor_hash(xs), "heldout_x_sha256": tensor_hash(xh),
                "selected_alpha": chosen["alpha"],
                "search_D_sse": baseline["search_sse"], "search_D_mse": baseline["search_output_mse"],
                "search_C_sse": chosen["search_sse"], "search_C_mse": chosen["search_output_mse"],
                "heldout_D_sse": baseline["heldout_sse"], "heldout_D_mse": baseline["heldout_output_mse"],
                "heldout_C_sse": chosen["heldout_sse"], "heldout_C_mse": chosen["heldout_output_mse"],
                "search_gain": 1 - chosen["search_sse"] / baseline["search_sse"],
                "held_gain": 1 - chosen["heldout_sse"] / baseline["heldout_sse"],
                "candidate_evaluations_this_run": len(base.GRID), "candidate_seconds_total": this_seconds,
                "hessian_seconds": float(hdiag["hessian_build_seconds"]),
                "weights_path": str(weight_path.relative_to(ROOT)), "candidates": candidates,
            }
            save_json(raw_path, record)
            newly_completed_records += 1
            print(f"SAVED {row_id} {sid} alpha={chosen['alpha']:.1f} held_gain={record['held_gain']:+.5f}", flush=True)

    # Reconcile durable records rather than trusting transient counters.
    records = [base.read_json(p) for p in STAGE2.glob("L*_S??.json")]
    if len(records) != EXPECTED_ROWS * len(SPLIT_IDS):
        raise RuntimeError(f"Stage 2 incomplete: expected 672 records, got {len(records)}")
    durable_new = sum(int(r["candidate_evaluations_this_run"]) for r in records if r["source"] == "v11_stage2")
    durable_reuse = sum(len(base.GRID) for r in records if r["source"] == "existing_v10")
    # An interrupted/resumed run may have created records in an earlier invocation.
    actual_invocation_seconds = time.perf_counter() - start
    timing = {
        "row_manifest_sha256": EXPECTED_ROW_HASH, "v9_manifest_sha256": EXPECTED_SPLIT_HASH,
        "row_split_records": len(records), "new_candidate_evaluations_this_invocation": newly_evaluated,
        "new_candidate_evaluations_durable_total": durable_new,
        "reusable_v10_candidate_evaluations": durable_reuse,
        "candidate_seconds_this_invocation": candidate_seconds,
        "hessian_seconds_this_invocation": hessian_seconds,
        "invocation_seconds": actual_invocation_seconds, "gpu_synchronized": True,
        "newly_completed_records_this_invocation": newly_completed_records,
        "max_candidate_peak_allocated_bytes": max_alloc,
        "max_candidate_peak_reserved_bytes": max_reserved,
        "activation_capture_seconds": capture.get("capture_seconds"),
        "stage2_activation_cache_bytes": ACTIVATION_CACHE.stat().st_size,
    }
    save_json(TIMING_PATH, timing)
    post_preservation = verify_stage1_preservation()
    if post_preservation["status"] != "PASS":
        raise RuntimeError(f"Stage 1 artifacts changed during Stage 2: {post_preservation['failed_files'][:5]}")
    print(json.dumps({"stage2": "DONE", **timing,
                      "stage1_preservation": post_preservation}, ensure_ascii=False, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--preflight", action="store_true")
    group.add_argument("--run-stage2", action="store_true")
    args = parser.parse_args()
    if args.preflight:
        run_preflight()
    else:
        run_stage2()


if __name__ == "__main__":
    main()
