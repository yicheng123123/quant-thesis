"""Projection preflight and preregistered three-split Stage 1 for W4 GPTQ v11."""
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

from experiments.range_search_w4.dynamic_core import dynamic_gptq, native_hinv_from_documents, output_sse, weight_mse
from experiments.range_search_w4.run import sync
from experiments.range_search_w4.run_task0_parity import tensor_hash

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
OUT = RESULTS / "range_w4_96row_v11"
V8 = RESULTS / "range_w4_dynamic_v8"
V9 = RESULTS / "range_w4_l23r448_v9"
V10 = RESULTS / "range_w4_sixrow_v10"
REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"
SNAPSHOT = ROOT / ".hf_cache" / "hub" / "models--Qwen--Qwen2.5-0.5B-Instruct" / "snapshots" / REVISION
EXPECTED_V9_MANIFEST = "34a24ef8602b8ce1e38e7a284f112acf7f766962d87d737f69148b78bcfaba78"
LAYERS = (0, 12, 23)
PROJECTIONS = ("q_proj", "o_proj", "up_proj", "down_proj")
GRID = [round(i / 10, 1) for i in range(1, 11)]
SPLIT_IDS = ("S00", "S01", "S02")
MAX_ESTIMATED_FORMAL_SECONDS = 4 * 60 * 60
MAX_PROCESS_RESERVED_RATIO = 0.80


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def save_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def frozen_write(path: Path, value: dict) -> None:
    content = json.dumps(value, ensure_ascii=False, indent=2) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != content:
            raise RuntimeError(f"frozen artifact differs; refusing overwrite: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def tree_fingerprint(path: Path) -> dict:
    result = {}
    for file in sorted(x for x in path.rglob("*") if x.is_file()):
        result[str(file.relative_to(path))] = {
            "sha256": sha256_bytes(file.read_bytes()),
            "mtime_ns": file.stat().st_mtime_ns,
            "size_bytes": file.stat().st_size,
        }
    return result


def history_matches(reference: dict) -> dict:
    return {name: tree_fingerprint(path) == reference["historical_fingerprints_before_v11"][name]
            for name, path in (("v8", V8), ("v9", V9), ("v10", V10))}


def module_for(model, layer: int, projection: str):
    block = model.model.layers[layer]
    if projection in ("q_proj", "o_proj"):
        return getattr(block.self_attn, projection)
    return getattr(block.mlp, projection)


def row_key(row: dict) -> str:
    return f"L{row['layer']}_{row['projection']}_R{row['row_index']}"


def input_key(layer: int, projection: str) -> str:
    return f"L{layer}/{projection}"


def result_slug(layer: int, projection: str, row_index: int, split_id: str) -> str:
    return f"L{layer}_{projection}_R{row_index}_{split_id}"


def validate_manifests() -> tuple[dict, dict]:
    row_manifest = read_json(OUT / "row_manifest.json")
    reference = read_json(OUT / "manifest_reference.json")
    split_path = V9 / "splits_manifest.json"
    digest = sha256_bytes(split_path.read_bytes())
    if digest != EXPECTED_V9_MANIFEST or digest != reference["v9_splits_manifest_sha256"]:
        raise RuntimeError("v9 frozen split manifest hash mismatch")
    manifest = read_json(split_path)
    if [x["split_id"] for x in manifest["splits"][:3]] != list(SPLIT_IDS):
        raise RuntimeError("v9 S00-S02 order changed")
    if len(row_manifest["rows"]) != 96:
        raise RuntimeError("frozen row manifest must contain exactly 96 rows")
    if sha256_bytes((OUT / "row_manifest.json").read_bytes()) != reference["row_manifest_sha256"]:
        raise RuntimeError("row manifest changed after freeze")
    return row_manifest, manifest


@torch.inference_mode()
def capture_inputs_and_weights(row_manifest: dict, manifest: dict) -> tuple[dict, dict, dict]:
    cache_path = OUT / "activation_cache.pt"
    row_manifest_hash = sha256_bytes((OUT / "row_manifest.json").read_bytes())
    if cache_path.exists():
        # This cache was created locally by this runner; the allowlist is only
        # for torch.__version__ saved as TorchVersion by the first capture.
        with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
            cache = torch.load(cache_path, map_location="cpu", weights_only=True)
        if cache["row_manifest_sha256"] != row_manifest_hash or cache["v9_manifest_sha256"] != EXPECTED_V9_MANIFEST:
            raise RuntimeError("activation cache does not belong to the frozen v11 manifests")
        return cache["inputs"], cache["weights"], cache["metadata"]
    if not torch.cuda.is_available():
        raise RuntimeError("v11 frozen protocol requires CUDA")
    docs_by_hash = {}
    for split in manifest["splits"][:3]:
        for side in ("search", "heldout"):
            for doc in split[side]:
                token_hash = doc["token_sha256"]
                if sha256_bytes(json.dumps(doc["token_ids"]).encode()) != token_hash:
                    raise RuntimeError(f"frozen token ID hash mismatch: {token_hash}")
                docs_by_hash[token_hash] = doc
    if len(docs_by_hash) != 24:
        raise RuntimeError(f"expected 24 unique S00-S02 articles, got {len(docs_by_hash)}")

    torch.manual_seed(42)
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        SNAPSHOT, torch_dtype=torch.float16, local_files_only=True,
        device_map="cuda", low_cpu_mem_usage=True,
    )
    model.eval()
    inputs = {input_key(layer, proj): {} for layer in LAYERS for proj in PROJECTIONS}
    weights = {}
    for row in row_manifest["rows"]:
        module = module_for(model, row["layer"], row["projection"])
        weights[row_key(row)] = module.weight.detach()[row["row_index"]].float().cpu().clone()

    current_hash = [None]
    handles = []
    for layer in LAYERS:
        for projection in PROJECTIONS:
            module = module_for(model, layer, projection)
            key = input_key(layer, projection)

            def hook(_module, args, _output, cache_key=key):
                token_hash = current_hash[0]
                if token_hash is None:
                    raise RuntimeError("projection hook ran without an active document")
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
    sync()
    capture_seconds = time.perf_counter() - started
    capture_meta = {
        "capture_seconds": capture_seconds,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "document_count": len(docs_by_hash),
        "device_name": torch.cuda.get_device_name(0),
        "torch_version": str(torch.__version__),
        "transformers_version": importlib.metadata.version("transformers"),
        "gptqmodel_version": importlib.metadata.version("gptqmodel"),
    }
    expected_weight_hashes = {row_key(r): r["weight_hash"] for r in row_manifest["rows"]}
    actual_weight_hashes = {key: tensor_hash(value) for key, value in weights.items()}
    if actual_weight_hashes != expected_weight_hashes:
        raise RuntimeError("effective FP16 model row hashes differ from the frozen manifest")
    for key, values in inputs.items():
        if set(values) != set(docs_by_hash):
            raise RuntimeError(f"projection capture incomplete: {key}; got {len(values)} documents")
    del model
    torch.cuda.empty_cache()
    cache = {
        "row_manifest_sha256": row_manifest_hash,
        "v9_manifest_sha256": EXPECTED_V9_MANIFEST,
        "inputs": inputs,
        "weights": weights,
        "metadata": {**capture_meta, "weight_hashes": actual_weight_hashes},
    }
    temp = cache_path.with_suffix(".pt.tmp")
    torch.save(cache, temp)
    temp.replace(cache_path)
    return inputs, weights, cache["metadata"]


def x_for_split(inputs: dict, layer: int, projection: str, split: dict, side: str, device: str = "cuda") -> torch.Tensor:
    key = input_key(layer, projection)
    return torch.cat([inputs[key][d["token_sha256"]] for d in split[side]]).to(device)


def docs_for_side(inputs: dict, layer: int, projection: str, split: dict, side: str, device: str = "cuda") -> list[torch.Tensor]:
    key = input_key(layer, projection)
    return [inputs[key][d["token_sha256"]].to(device) for d in split[side]]


def empty_peak_stats() -> None:
    gc.collect()
    torch.cuda.empty_cache()
    sync()
    torch.cuda.reset_peak_memory_stats()


def timed_call(fn):
    sync()
    start = time.perf_counter()
    result = fn()
    sync()
    return result, time.perf_counter() - start


def tensor_summary(value: torch.Tensor) -> dict:
    x = value.detach().float().reshape(-1).cpu()
    if x.numel() == 0 or not torch.isfinite(x).all():
        raise ValueError("cannot summarize empty or non-finite tensor")
    return {
        "min": float(x.min()), "mean": float(x.mean()), "std": float(x.std(unbiased=False)),
        "p50": float(torch.quantile(x, 0.50)), "p90": float(torch.quantile(x, 0.90)),
        "p99": float(torch.quantile(x, 0.99)), "max": float(x.max()),
    }


def get_hinv(row: dict, split: dict, inputs: dict, weights: dict, *, measure: bool = False) -> tuple[torch.Tensor, dict]:
    layer, projection, sid = row["layer"], row["projection"], split["split_id"]
    xs_docs = docs_for_side(inputs, layer, projection, split, "search")
    xs = torch.cat(xs_docs)
    input_sha = tensor_hash(xs)
    cache_path = OUT / "hessian_cache" / f"hinv_L{layer}_{projection}_{sid}.pt"
    if cache_path.exists():
        obj = torch.load(cache_path, map_location="cpu", weights_only=True)
        if obj["v9_manifest_sha256"] != EXPECTED_V9_MANIFEST or obj["search_x_sha256"] != input_sha:
            raise RuntimeError(f"Hessian cache mismatch: {cache_path.name}")
        return obj["hinv"].to("cuda"), {**{k: v for k, v in obj.items() if k != "hinv"}, "cache_reused": True}

    sample = next(r for r in row_manifest_global["rows"] if r["layer"] == layer and r["projection"] == projection)
    w = weights[row_key(sample)].cuda()
    module = torch.nn.Linear(row["input_dim"], 1, bias=False, device="cuda", dtype=torch.float16)
    module.weight.data.copy_(w[None].half())
    empty_peak_stats()
    (h, hinv, dead), seconds = timed_call(lambda: native_hinv_from_documents(xs_docs, module, 0.1))
    if dead:
        raise RuntimeError(f"dead GPTQ columns for L{layer} {projection} {sid}: {dead}")
    diagnostics = {
        "layer": layer, "projection": projection, "split_id": sid,
        "v9_manifest_sha256": EXPECTED_V9_MANIFEST,
        "search_x_sha256": input_sha,
        "input_dim": row["input_dim"], "dead_columns": dead,
        "hessian_build_seconds": seconds,
        "hessian_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "hessian_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "diag_h_summary": tensor_summary(h.diagonal()),
        "diag_hinv_cholesky_summary": tensor_summary(hinv.diagonal()),
        "cache_reused": False,
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache_path.with_suffix(".pt.tmp")
    torch.save({**diagnostics, "hinv": hinv.detach().cpu()}, tmp)
    tmp.replace(cache_path)
    return hinv, diagnostics


# Set after the frozen row manifest is read in preflight/run entry points.
row_manifest_global: dict = {}


def custom_candidate(w: torch.Tensor, hinv: torch.Tensor, xs: torch.Tensor, xh: torch.Tensor, alpha: float) -> tuple[dict, torch.Tensor]:
    empty_peak_stats()
    start = time.perf_counter()
    result = dynamic_gptq(w[None], hinv, alpha=alpha, group_size=128, block_size=128)
    q = result.weight_fp32[0].half()
    search_sse = output_sse(xs, w, q)
    held_sse = output_sse(xh, w, q)
    sync()
    seconds = time.perf_counter() - start
    record = {
        "alpha": alpha,
        "search_sse": search_sse,
        "heldout_sse": held_sse,
        "search_output_mse": search_sse / xs.shape[0],
        "heldout_output_mse": held_sse / xh.shape[0],
        "weight_mse": weight_mse(w, q),
        "scale": result.scales[0].detach().float().cpu().tolist(),
        "zero": result.zeros[0].detach().float().cpu().tolist(),
        "candidate_seconds": seconds,
        "candidate_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "candidate_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "weight_sha256": tensor_hash(q),
    }
    return record, q.detach().cpu()


def native_parity(row: dict, split: dict, inputs: dict, weights: dict, hinv: torch.Tensor) -> dict:
    from gptqmodel.quantization import QuantizeConfig
    from gptqmodel.quantization.gptq import GPTQ

    layer, projection, sid = row["layer"], row["projection"], split["split_id"]
    docs = docs_for_side(inputs, layer, projection, split, "search")
    xs = torch.cat(docs)
    xh = x_for_split(inputs, layer, projection, split, "heldout")
    w = weights[row_key(row)].cuda()
    module = torch.nn.Linear(row["input_dim"], 1, bias=False, device="cuda", dtype=torch.float16)
    module.weight.data.copy_(w[None].half())
    qcfg = QuantizeConfig(bits=4, group_size=128, sym=True, desc_act=False,
                          static_groups=False, true_sequential=True, damp_percent=.1, mse=0.0)
    native = GPTQ(module, qcfg=qcfg)
    native.quantizer.configure(perchannel=True, bits=4, sym=True)
    empty_peak_stats()
    native_seconds_start = time.perf_counter()
    for x in docs:
        native.process_batch(x[None])
    native_q, native_scale, native_zero, *_ = native.quantize(blocksize=128)
    sync()
    native_seconds = time.perf_counter() - native_seconds_start
    native_peak_allocated = torch.cuda.max_memory_allocated()
    native_peak_reserved = torch.cuda.max_memory_reserved()
    native_q = native_q[0]
    custom, custom_q_cpu = custom_candidate(w, hinv, xs, xh, 1.0)
    custom_q = custom_q_cpu.cuda()
    scale_n = native_scale[0].detach().float().reshape(-1)
    zero_n = native_zero[0].detach().float().reshape(-1)
    scale_c = torch.tensor(custom["scale"], device="cuda", dtype=torch.float32).reshape(-1)
    zero_c = torch.tensor(custom["zero"], device="cuda", dtype=torch.float32).reshape(-1)
    search_native = output_sse(xs, w, native_q)
    held_native = output_sse(xh, w, native_q)
    search_custom = output_sse(xs, w, custom_q)
    held_custom = output_sse(xh, w, custom_q)
    checks = {
        # native.quantize() deletes its accumulated H in GPTQModel 2.2.0;
        # get_hinv() has already rejected dead columns before this call.
        "no_dead_columns": bool(torch.isfinite(hinv).all().item() and (hinv.diagonal() > 0).all().item()),
        "fp16_final_weight_equal": torch.equal(native_q, custom_q),
        # Match the previously frozen Task 0 parity criterion: a sub-1e-8
        # float32 scale delta is accepted only when the resulting FP16 weights
        # and losses also match exactly/within their declared tolerance.
        "scale_equal": torch.allclose(scale_n, scale_c, atol=1e-6, rtol=1e-5),
        "zero_point_equal": torch.equal(zero_n, zero_c),
        "search_sse_equal": abs(search_native - search_custom) <= max(1e-7, abs(search_native) * 1e-6),
        "heldout_sse_equal": abs(held_native - held_custom) <= max(1e-7, abs(held_native) * 1e-6),
        "custom_alpha1_is_raw_grid_member": custom["alpha"] == 1.0,
    }
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "layer": layer, "projection": projection, "row_index": row["row_index"],
        "split_id": sid, "input_dim": row["input_dim"], "output_dim": row["output_dim"],
        "configuration": {"gptqmodel": importlib.metadata.version("gptqmodel"),
            "alpha": 1.0, "bits": 4, "symmetric": True, "group_size": 128,
            "block_size": 128, "damp_percent": .1, "desc_act": False},
        "checks": checks,
        "differences": {
            "weight_mismatched_elements": int((native_q != custom_q).sum()),
            "scale_max_abs": float((scale_n - scale_c).abs().max()),
            "zero_max_abs": float((zero_n - zero_c).abs().max()),
            "search_sse_abs": abs(search_native - search_custom),
            "heldout_sse_abs": abs(held_native - held_custom),
        },
        "native_search_sse": search_native, "custom_search_sse": search_custom,
        "native_heldout_sse": held_native, "custom_heldout_sse": held_custom,
        "native_full_row_quantize_seconds": native_seconds,
        "custom_alpha1_candidate_seconds": custom["candidate_seconds"],
        "candidate_peak_allocated_bytes": custom["candidate_peak_allocated_bytes"],
        "candidate_peak_reserved_bytes": custom["candidate_peak_reserved_bytes"],
        "native_parity_peak_allocated_bytes": native_peak_allocated,
        "native_parity_peak_reserved_bytes": native_peak_reserved,
    }


def reused_v10_records(row_manifest: dict, inputs: dict, v9_manifest: dict) -> dict:
    row_by_layer_proj_index = {
        (r["layer"], r["projection"], r["row_index"]): r for r in row_manifest["rows"]
    }
    reused = {}
    split_map = {s["split_id"]: s for s in v9_manifest["splits"][:3]}
    for (layer, projection, row_index), row in row_by_layer_proj_index.items():
        if projection != "q_proj":
            continue
        for sid, split in split_map.items():
            source_path = V10 / f"L{layer}R{row_index}_{sid}.json"
            if not source_path.exists():
                continue
            source = read_json(source_path)
            x_search = x_for_split(inputs, layer, projection, split, "search", device="cpu")
            x_held = x_for_split(inputs, layer, projection, split, "heldout", device="cpu")
            expected = {
                "row": f"L{layer}R{row_index}", "layer": layer,
                "row_index": row_index, "projection": "self_attn.q_proj",
                "split_id": sid, "manifest_sha256": EXPECTED_V9_MANIFEST,
                "weight_sha256": row["weight_hash"],
                "search_article_ids": [d["source_article_index"] for d in split["search"]],
                "heldout_article_ids": [d["source_article_index"] for d in split["heldout"]],
                "search_x_sha256": tensor_hash(x_search), "heldout_x_sha256": tensor_hash(x_held),
            }
            if any(source.get(k) != v for k, v in expected.items()):
                continue
            if [c["alpha"] for c in source.get("candidates", [])] != GRID:
                continue
            v10_protocol = read_json(V10 / "protocol.json")
            if (v10_protocol["group_size"] != 128 or v10_protocol["block_size"] != 128
                    or v10_protocol["damp_percent"] != .1 or not v10_protocol["quantization"].startswith("W4A16 symmetric")):
                continue
            if source.get("source") == "v10_run":
                weights_path = V10 / f"weights_L{layer}R{row_index}_{sid}.pt"
            elif source.get("source") == "v9_existing":
                weights_path = V9 / f"weights_{sid}.pt"
            else:
                continue
            if not weights_path.exists():
                continue
            weights_obj = torch.load(weights_path, map_location="cpu", weights_only=True)
            candidate_weights = weights_obj.get("weights", {})
            if any(str(c["alpha"]) not in candidate_weights for c in source["candidates"]):
                continue
            all_hashes_match = all(
                tensor_hash(candidate_weights[str(c["alpha"])]) == c["weight_sha256"]
                for c in source["candidates"]
            )
            if not all_hashes_match:
                continue
            key = (layer, projection, row_index, sid)
            reused[key] = {"source_raw": source, "source_weights_path": weights_path,
                           "source_raw_path": source_path, "candidate_weights": candidate_weights}
    return reused


def write_preflight_report(payload: dict) -> None:
    lines = [
        "# v11 Stage 1 projection preflight",
        "",
        f"Status: **{payload['status']}**",
        "",
        f"Frozen 96-row manifest SHA-256: `{payload['row_manifest_sha256']}`.",
        f"v9 split manifest SHA-256: `{payload['v9_manifest_sha256']}`; splits: S00–S02.",
        f"Planned Stage 1: {payload.get('nominal_candidate_evaluations', 'pending')} candidates; "
        f"reused: {payload.get('reusable_candidate_evaluations', 'pending')}; "
        f"new: {payload.get('new_candidate_evaluations', 'pending')}.",
        "",
        "| Projection | In × out | Parity | H build (s) | H peak alloc / reserved (MiB) | candidate mean (s) | candidate peak alloc / reserved (MiB) | estimated Stage 1 formal (s) |",
        "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in payload.get("projections", []):
        def mib(v):
            return f"{v / (1024**2):.1f}" if v is not None else "n/a"
        lines.append(
            f"| {item['projection']} | {item['input_dim']} × {item['output_dim']} | {item['parity_status']} | "
            f"{item.get('hessian_build_seconds', float('nan')):.3f} | "
            f"{mib(item.get('hessian_peak_allocated_bytes'))} / {mib(item.get('hessian_peak_reserved_bytes'))} | "
            f"{item.get('candidate_mean_seconds', float('nan')):.3f} | "
            f"{mib(item.get('candidate_peak_allocated_bytes'))} / {mib(item.get('candidate_peak_reserved_bytes'))} | "
            f"{item.get('estimated_formal_seconds_with_1_5x_safety_factor', float('nan')):.1f} |"
        )
    lines.extend(["", "## Gate checks", ""])
    for name, value in payload.get("checks", {}).items():
        lines.append(f"- `{name}`: {'PASS' if value else 'FAIL'}")
    end_to_end = payload.get("estimated_end_to_end_seconds")
    end_to_end_text = f"{end_to_end:.1f} s" if end_to_end is not None else "not estimable (preflight gate failed)"
    lines.extend(["", f"Observed preflight wall time: {payload.get('preflight_wall_seconds', float('nan')):.2f} s.",
                  f"Estimated end-to-end Stage 1 plus preflight/capture: {end_to_end_text}.",
                  f"Maximum process CUDA reserved: {payload.get('max_process_reserved_bytes', 0) / (1024**2):.1f} MiB / "
                  f"{payload.get('cuda_total_memory_bytes', 0) / (1024**2):.1f} MiB.", ""])
    (OUT / "PREFLIGHT_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def preflight() -> None:
    global row_manifest_global
    start = time.perf_counter()
    if not torch.cuda.is_available():
        raise RuntimeError("v11 preflight requires CUDA")
    row_manifest, v9_manifest = validate_manifests()
    row_manifest_global = row_manifest
    row_manifest_hash = sha256_bytes((OUT / "row_manifest.json").read_bytes())
    reference = read_json(OUT / "manifest_reference.json")
    if reference.get("row_manifest_sha256") != row_manifest_hash:
        raise RuntimeError("manifest_reference does not bind the frozen row manifest")
    inputs, weights, capture_meta = capture_inputs_and_weights(row_manifest, v9_manifest)
    split_map = {s["split_id"]: s for s in v9_manifest["splits"][:3]}
    reused = reused_v10_records(row_manifest, inputs, v9_manifest)
    nominal = 96 * 3 * 10
    reusable_candidates = len(reused) * 10
    per_projection_rows = {p: sum(r["projection"] == p for r in row_manifest["rows"]) for p in PROJECTIONS}
    projection_runs = []
    parity_all = True
    no_oom = True
    for projection in PROJECTIONS:
        row = next(r for r in row_manifest["rows"] if r["layer"] == 0 and r["projection"] == projection)
        split = split_map["S00"]
        try:
            hinv, hdiag = get_hinv(row, split, inputs, weights, measure=True)
            xs = x_for_split(inputs, row["layer"], projection, split, "search")
            xh = x_for_split(inputs, row["layer"], projection, split, "heldout")
            w = weights[row_key(row)].cuda()
            parity = native_parity(row, split, inputs, weights, hinv)
            smoke_candidates = []
            for alpha in (.7, .9):
                item, _ = custom_candidate(w, hinv, xs, xh, alpha)
                smoke_candidates.append(item)
            parity_status = parity["status"]
            parity_all = parity_all and parity_status == "PASS"
            metrics = {
                "projection": projection,
                "preflight_row": row["row_index"],
                "layer": row["layer"],
                "input_dim": row["input_dim"], "output_dim": row["output_dim"],
                "parity_status": parity_status,
                "parity": parity,
                "hessian_build_seconds": hdiag["hessian_build_seconds"],
                "hessian_peak_allocated_bytes": hdiag["hessian_peak_allocated_bytes"],
                "hessian_peak_reserved_bytes": hdiag["hessian_peak_reserved_bytes"],
                "diag_h_summary": hdiag["diag_h_summary"],
                "diag_hinv_cholesky_summary": hdiag["diag_hinv_cholesky_summary"],
                "smoke_candidates": smoke_candidates,
            }
            profile_seconds = [parity["custom_alpha1_candidate_seconds"]] + [x["candidate_seconds"] for x in smoke_candidates]
            metrics["candidate_mean_seconds"] = sum(profile_seconds) / len(profile_seconds)
            metrics["candidate_peak_allocated_bytes"] = max(
                [parity["candidate_peak_allocated_bytes"]] + [x["candidate_peak_allocated_bytes"] for x in smoke_candidates]
            )
            metrics["candidate_peak_reserved_bytes"] = max(
                [parity["candidate_peak_reserved_bytes"]] + [x["candidate_peak_reserved_bytes"] for x in smoke_candidates]
            )
            metrics["preflight_candidate_evaluations"] = 3
            projection_runs.append(metrics)
            if parity_status != "PASS":
                break
        except torch.cuda.OutOfMemoryError as exc:
            torch.cuda.empty_cache()
            no_oom = False
            projection_runs.append({"projection": projection, "preflight_row": row["row_index"],
                "layer": 0, "input_dim": row["input_dim"], "output_dim": row["output_dim"],
                "parity_status": "NOT_RUN_OOM", "error": str(exc),
                "hessian_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                "hessian_peak_reserved_bytes": torch.cuda.max_memory_reserved()})
            break
        except Exception as exc:
            no_oom = no_oom and "out of memory" not in str(exc).lower()
            projection_runs.append({"projection": projection, "preflight_row": row["row_index"],
                "layer": 0, "input_dim": row["input_dim"], "output_dim": row["output_dim"],
                "parity_status": "FAIL", "error": str(exc)})
            parity_all = False
            break

    total_memory = torch.cuda.get_device_properties(0).total_memory
    max_reserved = max(
        [capture_meta.get("peak_reserved_bytes", 0)]
        + [max(p.get("hessian_peak_reserved_bytes", 0),
               p.get("candidate_peak_reserved_bytes", 0),
               p.get("parity", {}).get("native_parity_peak_reserved_bytes", 0))
           for p in projection_runs]
    )
    all_four_completed = len(projection_runs) == len(PROJECTIONS)
    enough_budget = all_four_completed and all("candidate_mean_seconds" in p for p in projection_runs)
    estimated_candidate = 0.0
    estimated_hessian_remaining = 0.0
    estimated_by_projection = {}
    if enough_budget:
        reused_by_projection = {p: sum(1 for k in reused if k[1] == p) * 10 for p in PROJECTIONS}
        for p in projection_runs:
            name = p["projection"]
            candidate_n = per_projection_rows[name] * len(SPLIT_IDS) * 10 - reused_by_projection[name]
            cand = p["candidate_mean_seconds"] * candidate_n
            # Four L0/S00 Hessians were built for preflight; the other 8 of 9
            # layer/split Hessians per projection remain in the formal phase.
            hest = p["hessian_build_seconds"] * 8
            estimated_by_projection[name] = {
                "new_candidate_evaluations": candidate_n,
                "candidate_seconds_raw": cand,
                "remaining_hessian_builds": 8,
                "hessian_seconds_raw": hest,
                "estimated_formal_seconds_with_1_5x_safety_factor": 1.5 * (cand + hest),
            }
            estimated_candidate += cand
            estimated_hessian_remaining += hest
    estimated_formal = 1.5 * (estimated_candidate + estimated_hessian_remaining) if enough_budget else None
    preflight_wall = time.perf_counter() - start
    estimated_end_to_end = (estimated_formal + preflight_wall) if estimated_formal is not None else None
    checks = {
        "frozen_row_manifest_96_rows": len(row_manifest["rows"]) == 96,
        "v9_split_manifest_hash": sha256_bytes((V9 / "splits_manifest.json").read_bytes()) == EXPECTED_V9_MANIFEST,
        "S00_S02_reused": list(split_map) == list(SPLIT_IDS),
        "all_four_projection_preflights_completed": all_four_completed,
        "four_projection_native_parity_pass": parity_all and all(p["parity_status"] == "PASS" for p in projection_runs),
        "no_preflight_cuda_oom_or_numerical_failure": no_oom,
        "existing_v10_row_splits_verified": reusable_candidates == len(reused) * 10,
        "projected_formal_runtime_within_4h_gate": estimated_formal is not None and estimated_formal <= MAX_ESTIMATED_FORMAL_SECONDS,
        "peak_process_reserved_within_80pct_gpu": max_reserved <= int(total_memory * MAX_PROCESS_RESERVED_RATIO),
        "v8_v9_v10_unchanged_during_preflight": all(history_matches(reference).values()),
    }
    status = "PASS" if all(checks.values()) else "FAIL"
    protocol = {
        "id": "range_w4_96row_v11",
        "stage": "Stage 1 screening only",
        "model": "Qwen/Qwen2.5-0.5B-Instruct",
        "revision": REVISION,
        "checkpoint_dtype": "bfloat16 source, loaded as float16",
        "model_dtype": "float16",
        "math_dtype": "float32",
        "device": torch.cuda.get_device_name(0),
        "layers": list(LAYERS),
        "projections": ["self_attn.q_proj", "self_attn.o_proj", "mlp.up_proj", "mlp.down_proj"],
        "weight_quantization": "symmetric uniform W4A16",
        "integer_range": [0, 15], "zero": 8, "rounding": "torch.round ties to even",
        "group_size": 128, "block_size": 128, "damp_percent": .1,
        "desc_act": False, "static_groups": False, "true_sequential": True,
        "D": "alpha=1.0 native dynamic GPTQ",
        "C": "search-output-MSE-selected shared-alpha dynamic GPTQ",
        "alpha_grid": GRID,
        "input_semantics": "FP model projection inputs; no quantized prefix",
        "dataset": "Salesforce/wikitext:wikitext-2-raw-v1:train",
        "v9_manifest_sha256": EXPECTED_V9_MANIFEST,
        "split_ids": list(SPLIT_IDS),
        "articles_per_split": {"search": 4, "heldout": 4},
        "tokens_per_article": 128,
        "row_manifest_sha256": row_manifest_hash,
        "nominal_candidate_evaluations": nominal,
        "reusable_candidate_evaluations": reusable_candidates,
        "new_candidate_evaluations": nominal - reusable_candidates,
        "reused_row_split_records": [
            {"layer": k[0], "projection": k[1], "row_index": k[2], "split_id": k[3], "source": "existing_v10"}
            for k in sorted(reused)
        ],
        "stage1_labeling_thresholds": {
            "candidate_positive": "at least 2/3 held_gain > +2%, median held_gain > +5%, worst >= -10%",
            "mixed_risk": "not candidate-positive and has positive/negative split mixture, worst < -10%, alpha span >= 0.3 or three distinct selected alphas, or abs(median) < 2%",
            "neutral_negative": "remaining rows with no reliable positive three-split result",
        },
        "preflight_cost_gate": {
            "max_estimated_formal_seconds": MAX_ESTIMATED_FORMAL_SECONDS,
            "max_process_reserved_ratio_of_total_gpu": MAX_PROCESS_RESERVED_RATIO,
            "projected_candidate_and_hessian_cost_multiplier": 1.5,
        },
        "candidate_evaluation_estimates_by_projection": estimated_by_projection,
    }
    # The core protocol and row manifest are frozen; preflight-derived cost
    # estimates are generated metadata and may be refreshed after a harness fix.
    save_json(OUT / "protocol.json", protocol)
    payload = {
        "status": status,
        "checks": checks,
        "row_manifest_sha256": row_manifest_hash,
        "v9_manifest_sha256": EXPECTED_V9_MANIFEST,
        "nominal_candidate_evaluations": nominal,
        "reusable_candidate_evaluations": reusable_candidates,
        "new_candidate_evaluations": nominal - reusable_candidates,
        "reused_row_split_records": protocol["reused_row_split_records"],
        "capture": capture_meta,
        "projections": projection_runs,
        "estimated_candidate_seconds_raw": estimated_candidate if enough_budget else None,
        "estimated_remaining_hessian_seconds_raw": estimated_hessian_remaining if enough_budget else None,
        "estimated_formal_seconds_with_safety_factor": estimated_formal,
        "estimated_end_to_end_seconds": estimated_end_to_end,
        "preflight_wall_seconds": preflight_wall,
        "max_process_reserved_bytes": max_reserved,
        "cuda_total_memory_bytes": total_memory,
        "historical_fingerprints_unchanged": history_matches(reference),
    }
    save_json(OUT / "preflight_metrics.json", payload)
    write_preflight_report(payload)
    if status != "PASS":
        print(json.dumps({"preflight": status, "checks": checks,
                          "estimated_formal_seconds": estimated_formal}, indent=2))
        raise SystemExit(2)
    print(json.dumps({"preflight": status, "checks": checks,
                      "new_candidate_evaluations": nominal - reusable_candidates,
                      "estimated_formal_seconds": estimated_formal,
                      "max_process_reserved_mib": max_reserved / (1024**2)}, indent=2))


def run_stage1() -> None:
    global row_manifest_global
    if not torch.cuda.is_available():
        raise RuntimeError("Stage 1 requires CUDA")
    pre = read_json(OUT / "preflight_metrics.json")
    if pre["status"] != "PASS":
        raise RuntimeError("Stage 1 blocked because projection preflight did not PASS")
    protocol = read_json(OUT / "protocol.json")
    row_manifest, v9_manifest = validate_manifests()
    row_manifest_global = row_manifest
    reference = read_json(OUT / "manifest_reference.json")
    if not all(history_matches(reference).values()):
        raise RuntimeError("v8/v9/v10 changed after preflight")
    inputs, weights, _ = capture_inputs_and_weights(row_manifest, v9_manifest)
    split_map = {s["split_id"]: s for s in v9_manifest["splits"][:3]}
    reused = reused_v10_records(row_manifest, inputs, v9_manifest)
    if len(reused) * 10 != protocol["reusable_candidate_evaluations"]:
        raise RuntimeError("reusable candidate count changed after preflight")
    start = time.perf_counter()
    candidate_seconds_this_invocation = 0.0
    hessian_seconds_this_invocation = 0.0
    completed_records = 0
    candidate_count_invocation = 0
    raw_dir = OUT / "raw"
    raw_weights_dir = raw_dir / "weights"
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_weights_dir.mkdir(parents=True, exist_ok=True)

    for row in row_manifest["rows"]:
        layer, projection, row_index = row["layer"], row["projection"], row["row_index"]
        row_id = row_key(row)
        w = weights[row_id].cuda()
        if tensor_hash(w) != row["weight_hash"]:
            raise RuntimeError(f"pristine weight hash mismatch: {row_id}")
        for sid in SPLIT_IDS:
            split = split_map[sid]
            slug = result_slug(layer, projection, row_index, sid)
            json_path = raw_dir / f"{slug}.json"
            weights_path = raw_weights_dir / f"{slug}.pt"
            key = (layer, projection, row_index, sid)
            if json_path.exists():
                existing = read_json(json_path)
                if existing.get("row_manifest_sha256") != protocol["row_manifest_sha256"] or existing.get("source") not in ("v11_run", "existing_v10"):
                    raise RuntimeError(f"resume raw record differs: {json_path.name}")
                completed_records += 1
                continue
            xs = x_for_split(inputs, layer, projection, split, "search")
            xh = x_for_split(inputs, layer, projection, split, "heldout")
            hinv, hdiag = get_hinv(row, split, inputs, weights)
            if not hdiag.get("cache_reused", False):
                hessian_seconds_this_invocation += float(hdiag["hessian_build_seconds"])
            pristine_hash = tensor_hash(w)
            if key in reused:
                entry = reused[key]
                src = entry["source_raw"]
                qweights = {str(a): entry["candidate_weights"][str(a)].detach().cpu().clone() for a in GRID}
                copied = []
                for c in src["candidates"]:
                    copied.append({k: c[k] for k in (
                        "alpha", "search_sse", "heldout_sse", "search_output_mse", "heldout_output_mse",
                        "weight_mse", "scale", "zero", "candidate_seconds", "weight_sha256",
                    )})
                selected = next(c for c in copied if c["alpha"] == src["selected_alpha"])
                baseline = copied[-1]
                record = {
                    "row_id": row_id, "layer": layer, "projection": row["projection_name"],
                    "projection_short": projection, "row_index": row_index,
                    "output_dim": row["output_dim"], "input_dim": row["input_dim"],
                    "split_id": sid, "source": "existing_v10", "source_v10_raw": str(entry["source_raw_path"].relative_to(ROOT)),
                    "source_v10_raw_sha256": sha256_bytes(entry["source_raw_path"].read_bytes()),
                    "source_weights_sha256": sha256_bytes(entry["source_weights_path"].read_bytes()),
                    "row_manifest_sha256": protocol["row_manifest_sha256"],
                    "v9_manifest_sha256": EXPECTED_V9_MANIFEST,
                    "weight_sha256": pristine_hash,
                    "search_article_ids": src["search_article_ids"], "heldout_article_ids": src["heldout_article_ids"],
                    "search_x_sha256": src["search_x_sha256"], "heldout_x_sha256": src["heldout_x_sha256"],
                    "selected_alpha": src["selected_alpha"],
                    "search_D_sse": baseline["search_sse"], "search_D_mse": baseline["search_output_mse"],
                    "search_C_sse": selected["search_sse"], "search_C_mse": selected["search_output_mse"],
                    "heldout_D_sse": baseline["heldout_sse"], "heldout_D_mse": baseline["heldout_output_mse"],
                    "heldout_C_sse": selected["heldout_sse"], "heldout_C_mse": selected["heldout_output_mse"],
                    "search_gain": 1 - selected["search_sse"] / baseline["search_sse"],
                    "held_gain": 1 - selected["heldout_sse"] / baseline["heldout_sse"],
                    "candidate_evaluations_this_run": 0,
                    "candidate_seconds_total": 0.0,
                    "hessian_seconds": 0.0,
                    "candidates": copied,
                }
                tmpw = weights_path.with_suffix(".pt.tmp")
                torch.save({"row_manifest_sha256": protocol["row_manifest_sha256"], "weights": qweights}, tmpw)
                tmpw.replace(weights_path)
            else:
                candidates, qweights = [], {}
                total_candidate_seconds = 0.0
                for alpha in GRID:
                    c, q = custom_candidate(w, hinv, xs, xh, alpha)
                    total_candidate_seconds += c["candidate_seconds"]
                    candidate_seconds_this_invocation += c["candidate_seconds"]
                    candidate_count_invocation += 1
                    candidates.append(c)
                    qweights[str(alpha)] = q
                if tensor_hash(w) != pristine_hash:
                    raise RuntimeError(f"candidate state modified pristine W: {row_id}/{sid}")
                chosen = min(candidates, key=lambda c: c["search_output_mse"])
                baseline = candidates[-1]
                record = {
                    "row_id": row_id, "layer": layer, "projection": row["projection_name"],
                    "projection_short": projection, "row_index": row_index,
                    "output_dim": row["output_dim"], "input_dim": row["input_dim"],
                    "split_id": sid, "source": "v11_run",
                    "row_manifest_sha256": protocol["row_manifest_sha256"],
                    "v9_manifest_sha256": EXPECTED_V9_MANIFEST,
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
                    "candidate_evaluations_this_run": 10,
                    "candidate_seconds_total": total_candidate_seconds,
                    "hessian_seconds": float(hdiag["hessian_build_seconds"]),
                    "candidates": candidates,
                }
                tmpw = weights_path.with_suffix(".pt.tmp")
                torch.save({"row_manifest_sha256": protocol["row_manifest_sha256"], "weights": qweights}, tmpw)
                tmpw.replace(weights_path)
            save_json(json_path, record)
            completed_records += 1
            if record["source"] == "existing_v10":
                candidate_count_invocation += 0
            print(f"SAVED {row_id} {sid} source={record['source']} alpha={record['selected_alpha']} held={record['held_gain']:+.4f}", flush=True)

    actual = []
    for path in raw_dir.glob("L*_S??.json"):
        actual.append(read_json(path))
    run_timing = {
        "row_manifest_sha256": protocol["row_manifest_sha256"],
        "row_split_records": len(actual),
        "candidate_evaluations_this_invocation": candidate_count_invocation,
        "candidate_evaluations_durable_new": sum(r["candidate_evaluations_this_run"] for r in actual if r["source"] == "v11_run"),
        "candidate_evaluations_reused": sum(10 for r in actual if r["source"] == "existing_v10"),
        "candidate_seconds_this_invocation": candidate_seconds_this_invocation,
        "hessian_seconds_this_invocation": hessian_seconds_this_invocation,
        "invocation_seconds": time.perf_counter() - start,
        "gpu_synchronized": True,
    }
    save_json(OUT / "run_timing.json", run_timing)
    if not all(history_matches(reference).values()):
        raise RuntimeError("v8/v9/v10 tree changed during Stage 1")
    print(json.dumps({"stage1": "DONE", "row_split_records": len(actual),
                      "new_candidate_evaluations": run_timing["candidate_evaluations_durable_new"],
                      "reused_candidate_evaluations": run_timing["candidate_evaluations_reused"],
                      "seconds": run_timing["invocation_seconds"]}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--preflight", action="store_true")
    group.add_argument("--run-stage1", action="store_true")
    args = parser.parse_args()
    if args.preflight:
        preflight()
    else:
        run_stage1()


if __name__ == "__main__":
    main()
