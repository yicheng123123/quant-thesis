"""Capture frozen v14 inputs, run parity/cost preflight, then 1,920 candidates.

The candidate path reads search inputs only. Held-out scoring starts only after
all alpha search losses for a row/K have been frozen.
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
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
os.environ.update({"HF_HOME": str(ROOT / ".hf_cache"),
                   "HF_HUB_CACHE": str(ROOT / ".hf_cache" / "hub"),
                   "HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1"})

import torch
from transformers import AutoModelForCausalLM
from gptqmodel.quantization import QuantizeConfig
from gptqmodel.quantization.gptq import GPTQ

from experiments.range_search_w4.dynamic_core import dynamic_gptq, native_hinv_from_documents, output_sse

OUT = Path(__file__).resolve().parent
ROW_PATH = OUT / "row_manifest.json"
ARTICLE_PATH = OUT / "article_manifest.json"
PROTOCOL_PATH = OUT / "PROTOCOL.json"
RAW = OUT / "raw"
CAND_DIR = RAW / "candidate_weights"
CACHE = OUT / "cache"
HESS_DIR = CACHE / "hessian"
ACT_CACHE = CACHE / "activation_cache.pt"
REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"
SNAPSHOT = ROOT / ".hf_cache" / "hub" / "models--Qwen--Qwen2.5-0.5B-Instruct" / "snapshots" / REVISION
GRID = [round(i / 10, 1) for i in range(1, 11)]
K_VALUES = (4, 8, 16, 32)


def sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def tensor_hash(x: torch.Tensor) -> str:
    return sha_bytes(x.detach().contiguous().cpu().numpy().tobytes())


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def load_torch(path: Path) -> dict:
    with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
        return torch.load(path, map_location="cpu", weights_only=True)


def atomic_save(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    tmp.replace(path)


def sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def row_key(row: dict) -> str:
    return f"L{row['layer']}_{row['projection']}_R{row['row_index']}"


def module_key(layer: int, projection: str) -> str:
    return f"L{layer}/{projection}"


def validate_frozen() -> tuple[dict, dict, dict]:
    rows, articles, protocol = load_json(ROW_PATH), load_json(ARTICLE_PATH), load_json(PROTOCOL_PATH)
    assert len(rows["rows"]) == 48 and len(articles["calibration"]) == 32 and len(articles["heldout"]) == 32
    assert protocol["status"] == "FROZEN_BEFORE_PREFLIGHT_AND_CANDIDATES"
    assert sha_file(ROW_PATH) == protocol["row_manifest_sha256"]
    assert sha_file(ARTICLE_PATH) == protocol["article_manifest_sha256"]
    assert articles["K_membership"] == {f"K{k}": [f"C{i:02d}" for i in range(1, k + 1)] for k in K_VALUES}
    return rows, articles, protocol


def module_for(model, layer: int, projection: str):
    block = model.model.layers[layer]
    return getattr(block.self_attn, projection) if projection in ("q_proj", "o_proj") else getattr(block.mlp, projection)


@torch.inference_mode()
def capture_activations(rows: dict, articles: dict) -> dict:
    if ACT_CACHE.exists():
        cache = load_torch(ACT_CACHE)
        meta_path = CACHE / "activation_cache_metadata.json"
        if meta_path.exists():
            cache["metadata"].update(load_json(meta_path))
        assert cache["row_manifest_sha256"] == sha_file(ROW_PATH)
        assert cache["article_manifest_sha256"] == sha_file(ARTICLE_PATH)
        assert set(cache["inputs"]["L2/q_proj"]) == {d["token_sha256"] for d in articles["calibration"] + articles["heldout"]}
        print("activation cache verified and reused", flush=True)
        return cache
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required by the frozen v14 protocol")
    docs = articles["calibration"] + articles["heldout"]
    doc_by_hash = {d["token_sha256"]: d for d in docs}
    assert len(doc_by_hash) == 64
    torch.manual_seed(20261003)
    gc.collect(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(SNAPSHOT, torch_dtype=torch.float16,
                                                local_files_only=True, device_map="cuda",
                                                low_cpu_mem_usage=True)
    model.eval()
    model_load_seconds = time.perf_counter() - t0
    inputs = {module_key(layer, proj): {} for layer in (2, 6, 14, 18)
              for proj in ("q_proj", "o_proj", "up_proj", "down_proj")}
    weights = {}
    for row in rows["rows"]:
        mod = module_for(model, row["layer"], row["projection"])
        weights[row_key(row)] = mod.weight.detach()[row["row_index"]].float().cpu().clone()
    weight_hashes = {k: tensor_hash(v) for k, v in weights.items()}
    expected = {row_key(r): r["weight_hash"] for r in rows["rows"]}
    if weight_hashes != expected:
        raise RuntimeError("loaded model row hashes differ from frozen row manifest")
    active_hash = [None]
    handles = []
    for layer in (2, 6, 14, 18):
        for projection in ("q_proj", "o_proj", "up_proj", "down_proj"):
            mod = module_for(model, layer, projection)
            key = module_key(layer, projection)
            def hook(_module, args, _output, cache_key=key):
                token_hash = active_hash[0]
                if token_hash is None:
                    raise RuntimeError("activation hook ran without a frozen article")
                x = args[0].detach().reshape(-1, args[0].shape[-1]).float().cpu().clone()
                inputs[cache_key][token_hash] = x
            handles.append(mod.register_forward_hook(hook))
    forward_seconds = {}
    forward_start = time.perf_counter()
    try:
        for i, doc in enumerate(docs):
            active_hash[0] = doc["token_sha256"]
            token_ids = torch.tensor(doc["token_ids"], dtype=torch.long, device="cuda")[None]
            sync(); start = time.perf_counter()
            model(input_ids=token_ids, use_cache=False)
            sync(); forward_seconds[doc["article_id"]] = time.perf_counter() - start
            if (i + 1) % 8 == 0:
                print(f"activation capture {i+1}/64 articles", flush=True)
    finally:
        for h in handles: h.remove()
    forward_total_seconds = time.perf_counter() - forward_start
    for key, values in inputs.items():
        if set(values) != set(doc_by_hash):
            raise RuntimeError(f"incomplete activation cache for {key}: {len(values)}/64")
    cache_obj = {"row_manifest_sha256": sha_file(ROW_PATH),
                 "article_manifest_sha256": sha_file(ARTICLE_PATH),
                 "inputs": inputs, "weights": weights,
                 "metadata": {"capture_status": "PASS", "device": torch.cuda.get_device_name(0),
                              "torch_version": str(torch.__version__),
                              "transformers_version": importlib.metadata.version("transformers"),
                              "gptqmodel_version": importlib.metadata.version("gptqmodel"),
                              "model_load_seconds": model_load_seconds,
                              "forward_seconds_total": forward_total_seconds,
                              "forward_seconds_by_article": forward_seconds,
                              "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
                              "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
                              "weight_hashes": weight_hashes,
                              "article_count": 64}}
    del model
    torch.cuda.empty_cache(); gc.collect()
    save_start = time.perf_counter()
    atomic_save(ACT_CACHE, cache_obj)
    cache_obj["metadata"]["cache_save_seconds"] = time.perf_counter() - save_start
    save_json(CACHE / "activation_cache_metadata.json", cache_obj["metadata"])
    return cache_obj


def article_docs(cache: dict, row: dict, articles: list[dict], device="cuda") -> list[torch.Tensor]:
    key = module_key(row["layer"], row["projection"])
    return [cache["inputs"][key][d["token_sha256"]].to(device) for d in articles]


def make_module(weight: torch.Tensor) -> torch.nn.Linear:
    module = torch.nn.Linear(weight.numel(), 1, bias=False, device="cuda", dtype=torch.float16)
    module.weight.data.copy_(weight.to(device="cuda", dtype=torch.float16)[None])
    return module


def hinv_filename(layer: int, projection: str, k: int) -> Path:
    return HESS_DIR / f"L{layer}_{projection}_K{k}.pt"


def build_hinv(row: dict, k: int, cache: dict, articles: dict) -> tuple[torch.Tensor, dict]:
    cal = articles["calibration"][:k]
    docs_cpu = article_docs(cache, row, cal, device="cpu")
    xcat = torch.cat(docs_cpu)
    input_sha = tensor_hash(xcat)
    path = hinv_filename(row["layer"], row["projection"], k)
    if path.exists():
        obj = load_torch(path)
        if obj["input_sha256"] != input_sha or obj["row_manifest_sha256"] != sha_file(ROW_PATH) or obj["article_manifest_sha256"] != sha_file(ARTICLE_PATH):
            raise RuntimeError(f"v14 Hessian cache input/provenance mismatch: {path.name}")
        return obj["hinv"].to("cuda"), {k0: v for k0, v in obj.items() if k0 != "hinv"} | {"cache_reused": True}
    weight = cache["weights"][row_key(row)]
    module = make_module(weight)
    docs_gpu = [x.to("cuda", dtype=torch.float32) for x in docs_cpu]
    gc.collect(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(); sync()
    start = time.perf_counter()
    h, hinv, dead = native_hinv_from_documents(docs_gpu, module, 0.1)
    sync(); elapsed = time.perf_counter() - start
    if dead:
        raise RuntimeError(f"dead Hessian columns: {dead} in L{row['layer']}/{row['projection']}/K{k}")
    hinv_cpu = hinv.detach().cpu().contiguous()
    meta = {"layer": row["layer"], "projection": row["projection"], "K": k,
            "row_manifest_sha256": sha_file(ROW_PATH), "article_manifest_sha256": sha_file(ARTICLE_PATH),
            "input_sha256": input_sha, "hinv_sha256": tensor_hash(hinv_cpu),
            "input_dim": row["input_dim"], "tokens": k * 128,
            "hessian_build_seconds": elapsed, "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()), "dead_columns": dead,
            "cache_reused": False}
    atomic_save(path, {**meta, "hinv": hinv_cpu})
    del h, hinv, module, docs_gpu, docs_cpu, xcat
    torch.cuda.empty_cache(); gc.collect()
    return load_torch(path)["hinv"].to("cuda"), meta


def native_parity(row: dict, cache: dict, articles: dict, hinv_helper: torch.Tensor) -> dict:
    cal_docs = article_docs(cache, row, articles["calibration"][:4])
    held_docs = article_docs(cache, row, articles["heldout"])
    xs = torch.cat(cal_docs)
    xh = torch.cat(held_docs)
    w = cache["weights"][row_key(row)].cuda()
    module = make_module(w)
    qcfg = QuantizeConfig(bits=4, group_size=128, sym=True, desc_act=False,
                          static_groups=False, true_sequential=True, damp_percent=.1, mse=0.0)
    native = GPTQ(module, qcfg=qcfg)
    native.quantizer.configure(perchannel=True, bits=4, sym=True)
    for x in cal_docs:
        native.process_batch(x[None])
    native_entries, native_scales, native_zeros, inverse_results = [], [], [], []
    find_params = native.quantizer.find_params
    def trace_find(x, weight=False):
        if x.shape[-1] == 128:
            native_entries.append(x.detach().float().cpu().clone())
        find_params(x, weight=weight)
        if x.shape[-1] == 128:
            native_scales.append(native.quantizer.scale.detach().float().cpu().clone())
            native_zeros.append(native.quantizer.zero.detach().float().cpu().clone())
    native.quantizer.find_params = trace_find
    native_inverse = native.block_cholesky_inverse
    def trace_inverse(chol, *args, **kwargs):
        result = native_inverse(chol, *args, **kwargs)
        inverse_results.append(result.detach().clone())
        return result
    native.block_cholesky_inverse = trace_inverse
    sync(); start = time.perf_counter()
    qnative, snative, znative, group_idx, *_ = native.quantize(blocksize=128)
    sync(); native_seconds = time.perf_counter() - start
    if len(inverse_results) != 1:
        raise RuntimeError("native GPTQ parity did not expose exactly one inverse")
    hinv_native = torch.linalg.cholesky(inverse_results[0], upper=True)
    custom = dynamic_gptq(w[None], hinv_native, alpha=1.0, group_size=128, block_size=128, trace=True)
    qcustom = custom.weight_fp32[0].half()
    qnative = qnative[0]
    entry_diffs = [float((a.float().cpu() - b.float().cpu()).abs().max())
                   for a, b in zip(native_entries, custom.group_inputs)]
    sn = snative[0].detach().float().reshape(-1)
    zn = znative[0].detach().float().reshape(-1)
    sc = custom.scales[0].detach().float().cpu().reshape(-1)
    zc = custom.zeros[0].detach().float().cpu().reshape(-1)
    se_n, se_c = output_sse(xs, w, qnative), output_sse(xs, w, qcustom)
    he_n, he_c = output_sse(xh, w, qnative), output_sse(xh, w, qcustom)
    group_count = math.ceil(row["input_dim"] / 128)
    group_idx_ok = torch.equal(group_idx.cpu(), torch.arange(row["input_dim"], dtype=group_idx.dtype) // 128)
    hinv_same = torch.allclose(hinv_native, hinv_helper, atol=1e-4, rtol=1e-5)
    checks = {"group_count_and_boundaries": group_idx_ok and len(native_entries) == group_count == len(custom.group_inputs),
              "compensated_group_entry_weights": len(entry_diffs) == group_count and max(entry_diffs) <= 1e-5,
              "scale_equal": torch.allclose(sn.cpu(), sc, atol=1e-6, rtol=1e-5),
              "zero_equal": torch.equal(zn.cpu(), zc),
              "final_fp16_weight_equal": torch.equal(qnative, qcustom),
              "search_sse_equal": abs(se_n - se_c) <= max(1e-7, abs(se_n) * 1e-6),
              "heldout_sse_equal": abs(he_n - he_c) <= max(1e-7, abs(he_n) * 1e-6),
              "native_hessian_matches_helper": hinv_same}
    return {"status": "PASS" if all(checks.values()) else "FAIL", "layer": row["layer"],
            "projection": row["projection"], "row_index": row["row_index"], "K": 4,
            "checks": checks, "configuration": {"alpha": 1.0, "bits": 4, "symmetric": True,
                "group_size": 128, "block_size": 128, "damp_percent": 0.1, "desc_act": False,
                "gptqmodel_version": importlib.metadata.version("gptqmodel")},
            "group_count": group_count, "group_entry_max_abs": max(entry_diffs),
            "scale_max_abs": float((sn.cpu() - sc).abs().max()),
            "zero_max_abs": float((zn.cpu() - zc).abs().max()),
            "fp16_weight_mismatches": int((qnative.cpu() != qcustom.cpu()).sum()),
            "native_search_sse": se_n, "custom_search_sse": se_c,
            "native_heldout_sse": he_n, "custom_heldout_sse": he_c,
            "native_quantize_seconds": native_seconds,
            "native_peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "native_peak_reserved_bytes": int(torch.cuda.max_memory_reserved())}


def eval_article_sse(xdocs: list[torch.Tensor], w: torch.Tensor, q: torch.Tensor) -> tuple[list[float], float, int, int]:
    gc.collect(); torch.cuda.reset_peak_memory_stats(); sync(); start = time.perf_counter()
    delta = q.float() - w.float()
    vals = []
    for x in xdocs:
        err = x.float() @ delta
        vals.append(float(err.square().sum().item()))
    sync(); elapsed = time.perf_counter() - start
    return vals, elapsed, int(torch.cuda.max_memory_allocated()), int(torch.cuda.max_memory_reserved())


def preflight(rows: dict, articles: dict, cache: dict) -> tuple[dict, dict]:
    projection_runs, hessian_runs, candidate_runs = [], [], []
    projection_reps = {}
    for projection in ("q_proj", "o_proj", "up_proj", "down_proj"):
        row = next(r for r in rows["rows"] if r["layer"] == 2 and r["projection"] == projection and r["bucket"] == "low")
        projection_reps[projection] = row
        h_by_k = {}
        for k in K_VALUES:
            h, meta = build_hinv(row, k, cache, articles)
            h_by_k[k] = h
            hessian_runs.append(meta)
        parity = native_parity(row, cache, articles, h_by_k[4])
        parity["alpha1_candidate_evaluation_count"] = 1
        projection_runs.append(parity)
        pre_candidates = []
        cal_x = torch.cat(article_docs(cache, row, articles["calibration"][:4]))
        w = cache["weights"][row_key(row)].cuda()
        h_before = tensor_hash(h_by_k[4].detach().cpu())
        for alpha in GRID:
            torch.cuda.reset_peak_memory_stats(); sync(); start = time.perf_counter()
            result = dynamic_gptq(w[None], h_by_k[4], alpha=alpha, group_size=128, block_size=128)
            q = result.weight_fp32[0].half()
            search_sse = output_sse(cal_x, w, q)
            sync(); elapsed = time.perf_counter() - start
            item = {"alpha": alpha, "search_sse": search_sse, "search_mse": search_sse / cal_x.shape[0],
                    "weight_sha256": tensor_hash(q), "scales": result.scales[0].float().cpu().tolist(),
                    "zeros": result.zeros[0].float().cpu().tolist(), "candidate_seconds": elapsed,
                    "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
                    "peak_reserved_bytes": int(torch.cuda.max_memory_reserved())}
            pre_candidates.append((item, q.detach().clone()))
            candidate_runs.append({"projection": projection, "alpha": alpha, **{k0: v for k0, v in item.items() if k0 != "scales" and k0 != "zeros"}})
            del result, q
        if tensor_hash(w) != row["weight_hash"] or tensor_hash(h_by_k[4].detach().cpu()) != h_before:
            raise RuntimeError("preflight candidate mutated pristine W or shared Hinv")
        chosen = min((x[0] for x in pre_candidates), key=lambda x: (x["search_sse"], x["alpha"]))
        q_by_alpha = {it["alpha"]: q for it, q in pre_candidates}
        xheld = article_docs(cache, row, articles["heldout"])
        d_alpha = next(x[0]["alpha"] for x in pre_candidates if x[0]["alpha"] == 1.0)
        _, held_eval_time, held_alloc, held_reserved = eval_article_sse(xheld, w, q_by_alpha[d_alpha])
        if chosen["alpha"] == 1.0:
            held_eval_time *= 1
        else:
            _, t2, a2, r2 = eval_article_sse(xheld, w, q_by_alpha[chosen["alpha"]])
            held_eval_time += t2; held_alloc = max(held_alloc, a2); held_reserved = max(held_reserved, r2)
        candidate_runs.append({"projection": projection, "selected_alpha_search_only": chosen["alpha"],
                               "candidate10_wall_seconds": sum(x[0]["candidate_seconds"] for x in pre_candidates),
                               "per_candidate_mean_seconds": sum(x[0]["candidate_seconds"] for x in pre_candidates) / 10,
                               "heldout_eval_pair_seconds": held_eval_time,
                               "heldout_peak_allocated_bytes": held_alloc,
                               "heldout_peak_reserved_bytes": held_reserved})
        del h_by_k, w, cal_x, xheld, pre_candidates, q_by_alpha
        torch.cuda.empty_cache(); gc.collect()
    capture = cache["metadata"]
    forward_by = capture["forward_seconds_by_article"]
    cost_estimates = {}
    for projection in projection_reps:
        hs = {m["K"]: m for m in hessian_runs if m["projection"] == projection}
        cands = next(x for x in candidate_runs if x["projection"] == projection and "candidate10_wall_seconds" in x)
        h_est = sum(hs[k]["hessian_build_seconds"] * 4 for k in K_VALUES)
        cand_est = cands["per_candidate_mean_seconds"] * 480
        held_est = cands["heldout_eval_pair_seconds"] * 48
        cost_estimates[projection] = {"estimated_hessian_seconds_all_layers": h_est,
                                      "estimated_candidate_seconds_480_slots": cand_est,
                                      "estimated_heldout_eval_seconds_48_rowK_pairs": held_est,
                                      "candidate_runtime_sample_10_seconds": cands["candidate10_wall_seconds"]}
    # Coverage-specific activation forward attribution: C1..CK plus shared H01..H32.
    activation_by_k = {k: sum(forward_by[f"C{i:02d}"] for i in range(1, k + 1)) +
                       sum(forward_by[f"H{i:02d}"] for i in range(1, 33)) for k in K_VALUES}
    global_cost = {"activation_forward_seconds_by_K": activation_by_k,
                   "actual_shared_64_article_capture_wall_seconds": capture["model_load_seconds"] + capture["forward_seconds_total"] + capture.get("cache_save_seconds", 0.0),
                   "estimated_formal_seconds": capture["model_load_seconds"] + capture["forward_seconds_total"] + capture.get("cache_save_seconds", 0.0) +
                       sum(sum(v.values()) for v in cost_estimates.values()),
                   "estimated_candidate_count": 1920,
                   "estimated_candidate_count_by_projection": {p: 480 for p in projection_reps}}
    result = {"status": "PASS" if all(p["status"] == "PASS" for p in projection_runs) else "FAIL",
              "row_manifest_sha256": sha_file(ROW_PATH), "article_manifest_sha256": sha_file(ARTICLE_PATH),
              "activation_cache_sha256": sha_file(ACT_CACHE), "projection_parity": projection_runs,
              "hessian_builds": hessian_runs, "candidate_timing_runs": candidate_runs,
              "cost_estimate_by_projection": cost_estimates, "cost_estimate": global_cost,
              "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    save_json(OUT / "PREFLIGHT.json", result)
    lines = ["# v14 parity and cost preflight", "", f"Status: **{result['status']}**.",
             "", f"Frozen rows SHA-256: `{result['row_manifest_sha256']}`.",
             f"Frozen articles SHA-256: `{result['article_manifest_sha256']}`.",
             "", "## Native alpha=1 parity by projection (K4)", ""]
    for p in projection_runs:
        lines.append(f"- `{p['projection']}` L2/R{p['row_index']}: {p['status']}; checks={p['checks']}; max scale delta={p['scale_max_abs']:.3g}; group-entry max delta={p['group_entry_max_abs']:.3g}; weight mismatches={p['fp16_weight_mismatches']}; native/custom search SSE={p['native_search_sse']:.9g}/{p['custom_search_sse']:.9g}; held-out SSE={p['native_heldout_sse']:.9g}/{p['custom_heldout_sse']:.9g}.")
    lines += ["", "## Hessian build sample by projection and K", ""]
    for p in projection_reps:
        items = [m for m in hessian_runs if m["projection"] == p]
        lines.append(f"- `{p}`: " + "; ".join(f"K{m['K']}={m['hessian_build_seconds']:.2f}s, alloc={m['peak_allocated_bytes']/2**20:.1f} MiB, reserved={m['peak_reserved_bytes']/2**20:.1f} MiB" for m in items))
    lines += ["", "## Workload estimate", "",
              f"- Nominal / reusable / new formal candidates: 1920 / 0 / 1920.",
              f"- Estimated formal compute total (preflight formula, no inference): {global_cost['estimated_formal_seconds']/60:.1f} min, before analysis/file overhead.",
              f"- Actual shared 64-article activation-capture wall: {global_cost['actual_shared_64_article_capture_wall_seconds']/60:.1f} min.",
              "- Candidate runtime samples (10 alphas, one L2 row at K4): " + ", ".join(f"{p} {next(x['candidate10_wall_seconds'] for x in candidate_runs if x['projection']==p and 'candidate10_wall_seconds' in x):.2f}s" for p in projection_reps) + ".",
              "", "Candidate timings cover custom dynamic GPTQ and search-side loss only. Held-out is used in parity alone at this stage. If any projection parity fails, the formal candidate stage stops.", ""]
    (OUT / "PREFLIGHT_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    return result, {"activation_forward_seconds_by_K": activation_by_k}


def candidate_path(rid: str, k: int, alpha: float) -> Path:
    return CAND_DIR / f"{rid}_K{k}_a{int(round(alpha * 10)):02d}.pt"


def raw_path(rid: str, k: int) -> Path:
    return RAW / f"{rid}_K{k}.json"


def run_candidates(rows: dict, articles: dict, cache: dict) -> None:
    RAW.mkdir(parents=True, exist_ok=True); CAND_DIR.mkdir(parents=True, exist_ok=True)
    start_all = time.perf_counter()
    hessian_summary = []
    # Build/reuse each H_K once per layer/projection; rows in a projection share X and H.
    hinv_map = {}
    for layer in (2, 6, 14, 18):
        for projection in ("q_proj", "o_proj", "up_proj", "down_proj"):
            rep = next(r for r in rows["rows"] if r["layer"] == layer and r["projection"] == projection and r["bucket"] == "low")
            for k in K_VALUES:
                hinv, meta = build_hinv(rep, k, cache, articles)
                hinv_map[(layer, projection, k)] = hinv
                hessian_summary.append(meta)
                print(f"H ready L{layer}/{projection}/K{k} reused={meta.get('cache_reused',False)}", flush=True)
    save_json(OUT / "formal_hessian_timing.json", {"builds": hessian_summary,
              "unique_hessian_count": len(hessian_summary), "wall_seconds": time.perf_counter()-start_all})
    n_complete = 0
    for row in rows["rows"]:
        rid = row_key(row)
        w_cpu = cache["weights"][rid]
        assert tensor_hash(w_cpu) == row["weight_hash"]
        for k in K_VALUES:
            rp = raw_path(rid, k)
            if rp.exists():
                existing = load_json(rp)
                if existing.get("status") == "COMPLETE":
                    for cand in existing["candidates"]:
                        cp = candidate_path(rid, k, cand["alpha"])
                        qobj = load_torch(cp)
                        assert tensor_hash(qobj["weight"]) == cand["weight_sha256"]
                    n_complete += 1
                    continue
                record = existing
            else:
                record = {"status": "IN_PROGRESS", "row_id": rid, "layer": row["layer"],
                          "projection": row["projection"], "row_index": row["row_index"], "bucket": row["bucket"],
                          "K": k, "row_manifest_sha256": sha_file(ROW_PATH),
                          "article_manifest_sha256": sha_file(ARTICLE_PATH),
                          "pristine_weight_sha256": row["weight_hash"],
                          "calibration_article_ids": [d["article_id"] for d in articles["calibration"][:k]],
                          "heldout_article_ids": [d["article_id"] for d in articles["heldout"]],
                          "candidates": [], "selected_by_search_only": True}
            prior = {float(c["alpha"]): c for c in record["candidates"]}
            hinv = hinv_map[(row["layer"], row["projection"], k)]
            hinv_hash = tensor_hash(hinv.detach().cpu())
            docs_s = article_docs(cache, row, articles["calibration"][:k])
            xs = torch.cat(docs_s)
            w = w_cpu.cuda()
            row_start = time.perf_counter()
            for alpha in GRID:
                if alpha in prior:
                    continue
                torch.cuda.reset_peak_memory_stats(); sync(); cand_start = time.perf_counter()
                result = dynamic_gptq(w[None], hinv, alpha=alpha, group_size=128, block_size=128)
                q = result.weight_fp32[0].half()
                search_sse = output_sse(xs, w, q)
                sync(); candidate_seconds = time.perf_counter() - cand_start
                item = {"alpha": alpha, "search_sse": search_sse,
                        "search_mse": search_sse / xs.shape[0],
                        "weight_mse": float((q.float() - w.float()).square().mean().item()),
                        "scales": result.scales[0].detach().float().cpu().tolist(),
                        "zeros": result.zeros[0].detach().float().cpu().tolist(),
                        "weight_sha256": tensor_hash(q), "candidate_seconds": candidate_seconds,
                        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
                        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
                        "pristine_weight_hash_after": tensor_hash(w),
                        "hinv_sha256": tensor_hash(hinv.detach().cpu())}
                if item["pristine_weight_hash_after"] != row["weight_hash"] or item["hinv_sha256"] != hinv_hash:
                    raise RuntimeError(f"candidate state contamination at {rid}/K{k}/alpha{alpha}")
                atomic_save(candidate_path(rid, k, alpha), {"row_manifest_sha256": sha_file(ROW_PATH),
                            "article_manifest_sha256": sha_file(ARTICLE_PATH), "row_id": rid, "K": k,
                            "alpha": alpha, "weight": q.detach().cpu().contiguous()})
                record["candidates"].append(item)
                record["candidates"].sort(key=lambda x: x["alpha"])
                record["candidate_count_completed"] = len(record["candidates"])
                save_json(rp, record)
                del result, q
            assert len(record["candidates"]) == 10
            selected = min(record["candidates"], key=lambda c: (c["search_sse"], c["alpha"]))
            d_item = next(c for c in record["candidates"] if c["alpha"] == 1.0)
            selected_index = float(selected["alpha"])
            held_docs = article_docs(cache, row, articles["heldout"])
            qd = load_torch(candidate_path(rid, k, 1.0))["weight"].cuda()
            qc = load_torch(candidate_path(rid, k, selected_index))["weight"].cuda()
            d_article_sse, d_time, d_alloc, d_res = eval_article_sse(held_docs, w, qd)
            if selected_index == 1.0:
                c_article_sse, c_time, c_alloc, c_res = d_article_sse, 0.0, d_alloc, d_res
            else:
                c_article_sse, c_time, c_alloc, c_res = eval_article_sse(held_docs, w, qc)
            def side(vals):
                return {"article_sse": vals, "article_mse": [x / 128 for x in vals],
                        "pooled_sse": float(sum(vals)), "pooled_mse": float(sum(vals) / (32 * 128))}
            record["selected_alpha"] = selected_index
            record["search_gain_vs_D"] = 1 - selected["search_sse"] / d_item["search_sse"]
            record["heldout_D"] = side(d_article_sse)
            record["heldout_C"] = side(c_article_sse)
            record["heldout_gain_C_vs_D"] = 1 - record["heldout_C"]["pooled_sse"] / record["heldout_D"]["pooled_sse"]
            record["heldout_evaluation_seconds"] = d_time + c_time
            record["heldout_peak_allocated_bytes"] = max(d_alloc, c_alloc)
            record["heldout_peak_reserved_bytes"] = max(d_res, c_res)
            record["candidate_seconds_total"] = sum(c["candidate_seconds"] for c in record["candidates"])
            record["row_K_wall_seconds"] = time.perf_counter() - row_start
            record["hessian_sha256"] = hinv_hash
            record["search_input_sha256"] = tensor_hash(xs.detach().cpu())
            record["status"] = "COMPLETE"
            save_json(rp, record)
            del xs, docs_s, held_docs, w, qd, qc, hinv
            torch.cuda.empty_cache(); gc.collect()
            n_complete += 1
            print(f"completed {n_complete}/192 row-K records: {rid}/K{k}, alpha={selected_index:.1f}, gain={record['heldout_gain_C_vs_D']:+.3%}", flush=True)
            hinv = hinv_map[(row["layer"], row["projection"], k)]
    save_json(OUT / "formal_run_timing.json", {"completed_row_K_records": n_complete,
              "formal_candidate_evaluations": n_complete * 10,
              "wall_seconds_including_hessian_builds": time.perf_counter() - start_all})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight-only", action="store_true", help="stop after successful preflight")
    args = parser.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    rows, articles, protocol = validate_frozen()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    cache = capture_activations(rows, articles)
    preflight_path = OUT / "PREFLIGHT.json"
    if preflight_path.exists():
        pre = load_json(preflight_path)
        assert pre["row_manifest_sha256"] == sha_file(ROW_PATH) and pre["article_manifest_sha256"] == sha_file(ARTICLE_PATH)
        print("preflight artifact exists", pre["status"], flush=True)
    else:
        pre, _ = preflight(rows, articles, cache)
    if pre["status"] != "PASS":
        save_json(OUT / "correctness_checks.json", {"status": "STOP_PREFLIGHT_FAIL", "projection_parity": pre["projection_parity"]})
        raise SystemExit("A projection parity check failed; formal candidates were not started")
    if args.preflight_only:
        return
    run_candidates(rows, articles, cache)


if __name__ == "__main__":
    main()
