"""Frozen ten-split L23 R448 shared-alpha dynamic GPTQ experiment.

Only reads v8 parity evidence. Per-split JSON is the atomic resume marker.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM

from experiments.range_search_w4.dynamic_core import (
    dynamic_gptq, native_hinv_from_documents, output_sse, weight_mse,
)
from experiments.range_search_w4.prepare_v9_splits import OUTPUT, V8
from experiments.range_search_w4.run import SNAPSHOT, sync
from experiments.range_search_w4.run_task0_parity import tensor_hash

EXPECTED_SPLITS_SHA256 = "34a24ef8602b8ce1e38e7a284f112acf7f766962d87d737f69148b78bcfaba78"


def save_json(path: Path, value: object) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def timed(fn):
    sync()
    start = time.perf_counter()
    result = fn()
    sync()
    return result, time.perf_counter() - start


@torch.inference_mode()
def capture_inputs(manifest: dict, manifest_hash: str) -> tuple[dict[str, torch.Tensor], torch.Tensor, dict]:
    cache_path = OUTPUT / "activation_cache.pt"
    if cache_path.exists():
        cached = torch.load(cache_path, map_location="cpu", weights_only=True)
        if cached["manifest_sha256"] != manifest_hash:
            raise RuntimeError("activation cache uses a different frozen split manifest")
        return cached["inputs"], cached["weight"], cached["metadata"]
    torch.manual_seed(42)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        SNAPSHOT, torch_dtype=torch.float16, local_files_only=True,
        device_map="cuda", low_cpu_mem_usage=True)
    model.eval()
    module = model.model.layers[23].self_attn.q_proj
    weight = module.weight.detach().float().cpu().clone()[448]
    inputs = {}
    current_id = [None]

    def hook(_module, args, _output):
        inputs[current_id[0]] = args[0].detach().reshape(-1, args[0].shape[-1]).float().cpu().clone()

    handle = module.register_forward_hook(hook)
    for split in manifest["splits"]:
        for side in ("search", "heldout"):
            for doc in split[side]:
                key = doc["token_sha256"]
                if key in inputs:
                    raise RuntimeError("duplicate article/token sequence")
                ids = doc["token_ids"]
                if hashlib.sha256(json.dumps(ids).encode()).hexdigest() != key:
                    raise RuntimeError("token hash mismatch before model capture")
                current_id[0] = key
                model(input_ids=torch.tensor(ids, dtype=torch.long, device="cuda")[None],
                      use_cache=False)
                if inputs[key].shape != (128, weight.numel()):
                    raise RuntimeError("unexpected projection input shape")
    handle.remove()
    sync()
    metadata = {"capture_seconds": time.perf_counter() - started,
                "capture_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                "capture_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                "weight_sha256": tensor_hash(weight), "documents": len(inputs)}
    del model
    torch.cuda.empty_cache()
    temp = cache_path.with_suffix(".pt.tmp")
    torch.save({"manifest_sha256": manifest_hash, "inputs": inputs,
                "weight": weight, "metadata": metadata}, temp)
    temp.replace(cache_path)
    return inputs, weight, metadata


def activation_stats(xs: torch.Tensor, xh: torch.Tensor) -> dict:
    rms_s = xs.square().mean(0).sqrt()
    rms_h = xh.square().mean(0).sqrt()
    rel = (rms_h - rms_s).abs() / rms_s.clamp_min(1e-12)
    gram_s = xs.T @ xs / xs.shape[0]
    gram_h = xh.T @ xh / xh.shape[0]
    return {"global_rms_ratio": float(xh.square().mean().sqrt() / xs.square().mean().sqrt()),
            "feature_rms_relative_change_median": float(rel.median()),
            "feature_rms_relative_change_p90": float(torch.quantile(rel, .9)),
            "gram_relative_frobenius_change": float(torch.linalg.vector_norm(gram_h - gram_s) /
                                                     torch.linalg.vector_norm(gram_s)),
            "search_mean": float(xs.mean()), "heldout_mean": float(xh.mean()),
            "search_std": float(xs.std()), "heldout_std": float(xh.std()),
            "search_p01": float(torch.quantile(xs.flatten(), .01)),
            "search_p99": float(torch.quantile(xs.flatten(), .99)),
            "heldout_p01": float(torch.quantile(xh.flatten(), .01)),
            "heldout_p99": float(torch.quantile(xh.flatten(), .99))}


def main() -> None:
    parity = json.loads((V8 / "dynamic_parity_report.json").read_text(encoding="utf-8"))
    native_rows = json.loads((V8 / "native_six_row_check.json").read_text(encoding="utf-8"))
    if parity["status"] != "PASS" or native_rows["status"] != "PASS":
        raise RuntimeError("v8 native dynamic GPTQ parity gate is not passed")
    manifest_path = OUTPUT / "splits_manifest.json"
    manifest_bytes = manifest_path.read_bytes()
    manifest_hash = hashlib.sha256(manifest_bytes).hexdigest()
    if manifest_hash != EXPECTED_SPLITS_SHA256:
        raise RuntimeError("frozen split manifest changed after preregistration")
    manifest = json.loads(manifest_bytes)
    if len(manifest["splits"]) != 10:
        raise RuntimeError("expected ten preregistered splits")
    all_docs = [d for s in manifest["splits"] for key in ("search", "heldout") for d in s[key]]
    if len({d["source_article_index"] for d in all_docs}) != 80:
        raise RuntimeError("articles overlap across splits")
    run_start = time.perf_counter()
    inputs, weight_cpu, capture_meta = capture_inputs(manifest, manifest_hash)
    xh_all = {s["split_id"]: torch.cat([inputs[d["token_sha256"]] for d in s["heldout"]]).cuda()
              for s in manifest["splits"]}
    w = weight_cpu.cuda()
    if tensor_hash(w) != capture_meta["weight_sha256"]:
        raise RuntimeError("weight cache hash mismatch")
    module = torch.nn.Linear(w.numel(), 1, bias=False, device="cuda", dtype=torch.float16)
    module.weight.data.copy_(w[None].half())
    did_work = False
    for split in manifest["splits"]:
        sid = split["split_id"]
        result_path = OUTPUT / f"{sid}.json"
        weights_path = OUTPUT / f"weights_{sid}.pt"
        if result_path.exists() and weights_path.exists():
            prior = json.loads(result_path.read_text(encoding="utf-8"))
            if prior["manifest_sha256"] != manifest_hash or prior["weight_sha256"] != capture_meta["weight_sha256"]:
                raise RuntimeError(f"resume input mismatch: {sid}")
            print("RESUME", sid, flush=True)
            continue
        xs_docs = [inputs[d["token_sha256"]].cuda() for d in split["search"]]
        xs = torch.cat(xs_docs)
        xh = xh_all[sid]
        torch.cuda.reset_peak_memory_stats()
        split_start = time.perf_counter()
        (_, hinv, dead), h_seconds = timed(lambda: native_hinv_from_documents(xs_docs, module, .1))
        if dead:
            raise RuntimeError(f"dead Hessian columns in {sid}")
        pristine_hash = tensor_hash(w)
        candidates, qweights = [], {}
        for alpha in [round(i / 10, 1) for i in range(1, 11)]:
            result, seconds = timed(lambda a=alpha: dynamic_gptq(w[None], hinv, alpha=a))
            q = result.weight_fp32[0].half()
            qs = q.float()
            own_search_sse = output_sse(xs, w, qs)
            heldout_sse = {other_sid: output_sse(other_xh, w, qs)
                           for other_sid, other_xh in xh_all.items()}
            candidates.append({"alpha": alpha, "search_sse": own_search_sse,
                "search_output_mse": own_search_sse / xs.shape[0],
                "heldout_sse": heldout_sse[sid],
                "heldout_output_mse": heldout_sse[sid] / xh.shape[0],
                "cross_heldout_sse": heldout_sse,
                "weight_mse": weight_mse(w, qs),
                "scale": result.scales[0].tolist(), "zero": result.zeros[0].tolist(),
                "candidate_seconds": seconds, "weight_sha256": tensor_hash(q)})
            qweights[str(alpha)] = q.cpu()
        if tensor_hash(w) != pristine_hash:
            raise RuntimeError(f"candidate contaminated pristine W in {sid}")
        selected = min(candidates, key=lambda c: c["search_output_mse"])
        baseline = candidates[-1]
        if baseline["alpha"] != 1.0:
            raise RuntimeError("alpha=1 baseline absent")
        record = {"split_id": sid, "manifest_sha256": manifest_hash,
                  "weight_sha256": pristine_hash,
                  "search_article_ids": [d["source_article_index"] for d in split["search"]],
                  "heldout_article_ids": [d["source_article_index"] for d in split["heldout"]],
                  "search_x_sha256": tensor_hash(xs), "heldout_x_sha256": tensor_hash(xh),
                  "search_tokens": xs.shape[0], "heldout_tokens": xh.shape[0],
                  "selected_alpha_on_search": selected["alpha"],
                  "search_gain": 1 - selected["search_sse"] / baseline["search_sse"],
                  "heldout_gain": 1 - selected["heldout_sse"] / baseline["heldout_sse"],
                  "candidates": candidates, "activation": activation_stats(xs, xh),
                  "hessian_seconds": h_seconds,
                  "candidate_evaluations": len(candidates),
                  "candidate_seconds_total": sum(c["candidate_seconds"] for c in candidates),
                  "split_seconds": time.perf_counter() - split_start,
                  "peak_allocated_bytes": torch.cuda.max_memory_allocated()}
        weight_temp = weights_path.with_suffix(".pt.tmp")
        torch.save({"manifest_sha256": manifest_hash, "weights": qweights}, weight_temp)
        weight_temp.replace(weights_path)
        save_json(result_path, record)
        did_work = True
        print("SAVED", sid, "alpha", selected["alpha"],
              "heldout_gain", round(record["heldout_gain"], 6), flush=True)
    if did_work or not (OUTPUT / "run_timing.json").exists():
        save_json(OUTPUT / "run_timing.json", {"manifest_sha256": manifest_hash,
                  "capture": capture_meta, "invocation_seconds": time.perf_counter() - run_start,
                  "source": "GPU synchronized candidate/Hessian timings; capture includes model load and 80 forwards"})


if __name__ == "__main__":
    main()
