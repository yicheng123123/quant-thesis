"""Run five remaining rows on v9 frozen splits; import L23R448 from v9."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM

from experiments.range_search_w4.dynamic_core import dynamic_gptq, native_hinv_from_documents, output_sse, weight_mse
from experiments.range_search_w4.prepare_v9_splits import OUTPUT as V9, V8
from experiments.range_search_w4.run import SNAPSHOT, sync
from experiments.range_search_w4.run_task0_parity import historical_fingerprint, tensor_hash

HERE = Path(__file__).resolve().parent
OUT = HERE / "results" / "range_w4_sixrow_v10"
EXPECTED_SPLITS_SHA256 = "34a24ef8602b8ce1e38e7a284f112acf7f766962d87d737f69148b78bcfaba78"
GRID = [round(i / 10, 1) for i in range(1, 11)]
LAYERS = (0, 12, 23)
ROWS = (0, 448)


def save_json(path: Path, value: object) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def save_frozen_json(path: Path, value: object) -> None:
    content = json.dumps(value, ensure_ascii=False, indent=2) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != content:
            raise RuntimeError(f"preflight artifact differs; refusing replacement: {path.name}")
        return
    path.write_text(content, encoding="utf-8")


def timed(fn):
    sync()
    start = time.perf_counter()
    result = fn()
    sync()
    return result, time.perf_counter() - start


def tree_fingerprint(root: Path) -> dict:
    result = {}
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        rel = str(path.relative_to(root))
        result[rel] = {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                       "mtime_ns": path.stat().st_mtime_ns, "size_bytes": path.stat().st_size}
    return result


def load_frozen_manifest() -> tuple[dict, str]:
    path = V9 / "splits_manifest.json"
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != EXPECTED_SPLITS_SHA256:
        raise RuntimeError(f"v9 frozen manifest SHA mismatch: {digest}")
    manifest = json.loads(raw)
    if len(manifest["splits"]) != 10:
        raise RuntimeError("expected ten frozen splits")
    all_docs = [d for s in manifest["splits"] for side in ("search", "heldout") for d in s[side]]
    if len(all_docs) != 80 or len({d["source_article_index"] for d in all_docs}) != 80:
        raise RuntimeError("frozen split article count/uniqueness mismatch")
    if len({d["token_sha256"] for d in all_docs}) != 80:
        raise RuntimeError("frozen split token sequences are not unique")
    for split in manifest["splits"]:
        if len(split["search"]) != 4 or len(split["heldout"]) != 4:
            raise RuntimeError(f"wrong document counts: {split['split_id']}")
        if {d["token_sha256"] for d in split["search"]} & {d["token_sha256"] for d in split["heldout"]}:
            raise RuntimeError(f"search and heldout overlap: {split['split_id']}")
    return manifest, digest


@torch.inference_mode()
def capture_all(manifest: dict, manifest_hash: str) -> tuple[dict, dict, dict]:
    cache = OUT / "activation_cache.pt"
    if cache.exists():
        value = torch.load(cache, map_location="cpu", weights_only=True)
        if value["manifest_sha256"] != manifest_hash:
            raise RuntimeError("activation cache manifest mismatch")
        return value["inputs"], value["weights"], value["metadata"]
    torch.manual_seed(42)
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(SNAPSHOT, torch_dtype=torch.float16,
        local_files_only=True, device_map="cuda", low_cpu_mem_usage=True)
    model.eval()
    inputs: dict[str, dict[str, torch.Tensor]] = {str(layer): {} for layer in LAYERS}
    weights = {str(layer): {} for layer in LAYERS}
    for layer in LAYERS:
        module = model.model.layers[layer].self_attn.q_proj
        for row in ROWS:
            weights[str(layer)][str(row)] = module.weight.detach()[row].float().cpu().clone()
    weights["0"]["full_for_spot"] = model.model.layers[0].self_attn.q_proj.weight.detach().cpu().clone()
    current = [None]
    handles = []
    for layer in LAYERS:
        def hook(_module, args, _output, key=str(layer)):
            inputs[key][current[0]] = args[0].detach().reshape(-1, args[0].shape[-1]).float().cpu().clone()
        handles.append(model.model.layers[layer].self_attn.q_proj.register_forward_hook(hook))
    for split in manifest["splits"]:
        for side in ("search", "heldout"):
            for doc in split[side]:
                token_hash = doc["token_sha256"]
                current[0] = token_hash
                if hashlib.sha256(json.dumps(doc["token_ids"]).encode()).hexdigest() != token_hash:
                    raise RuntimeError("token hash mismatch")
                ids = torch.tensor(doc["token_ids"], dtype=torch.long, device="cuda")[None]
                model(input_ids=ids, use_cache=False)
                if any(token_hash not in inputs[str(layer)] for layer in LAYERS):
                    raise RuntimeError("projection hook missed a frozen article")
    for handle in handles:
        handle.remove()
    sync()
    metadata = {"capture_seconds": time.perf_counter() - start,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(), "document_count": 80,
        "weight_row_hashes": {str(layer): {str(row): tensor_hash(weights[str(layer)][str(row)])
                              for row in ROWS} for layer in LAYERS}}
    del model
    torch.cuda.empty_cache()
    tmp = cache.with_suffix(".pt.tmp")
    torch.save({"manifest_sha256": manifest_hash, "inputs": inputs,
                "weights": weights, "metadata": metadata}, tmp)
    tmp.replace(cache)
    return inputs, weights, metadata


def run_spot_parity(inputs: dict, weights: dict, manifest: dict) -> dict:
    from gptqmodel.quantization import QuantizeConfig
    from gptqmodel.quantization.gptq import GPTQ

    split = manifest["splits"][0]
    docs = [inputs["0"][d["token_sha256"]].cuda() for d in split["search"]]
    xs = torch.cat(docs)
    xheld = torch.cat([inputs["0"][d["token_sha256"]] for d in split["heldout"]]).cuda()
    full_w = weights["0"]["full_for_spot"].cuda()
    module = torch.nn.Linear(full_w.shape[1], full_w.shape[0], bias=False,
                             device="cuda", dtype=torch.float16)
    module.weight.data.copy_(full_w)
    cfg = QuantizeConfig(bits=4, group_size=128, sym=True, desc_act=False,
                         static_groups=False, true_sequential=True,
                         damp_percent=.1, mse=0.0)
    native = GPTQ(module, qcfg=cfg)
    native.quantizer.configure(perchannel=True, bits=4, sym=True)
    for x in docs:
        native.process_batch(x[None])
    native_q, native_scale, native_zero, *_ = native.quantize(blocksize=128)
    _, hinv, dead = native_hinv_from_documents(docs, module, .1)
    wrow = weights["0"]["0"].cuda()
    custom = dynamic_gptq(wrow[None], hinv, alpha=1.0)
    custom_q = custom.weight_fp32[0].half()
    diff = native_q[0].float() - custom_q.float()
    checks = {"no_dead_columns": dead == 0,
        "fp16_weight_equal": torch.equal(native_q[0], custom_q),
        "scale_equal": torch.equal(native_scale[0].float(), custom.scales[0]),
        "zero_equal": torch.equal(native_zero[0].float(), custom.zeros[0]),
        "search_output_mse_equal": abs(output_sse(xs, wrow, native_q[0]) -
                                        output_sse(xs, wrow, custom_q)) < 1e-7,
        "heldout_output_mse_equal": abs(output_sse(xheld, wrow, native_q[0]) -
                                         output_sse(xheld, wrow, custom_q)) < 1e-7}
    return {"status": "PASS" if all(checks.values()) else "FAIL", "row": "L0R0",
        "split_id": "S00", "configuration": {"gptqmodel": "2.2.0", "alpha": 1.0,
        "bits": 4, "symmetric": True, "group_size": 128, "block_size": 128,
        "damp_percent": .1, "desc_act": False}, "checks": checks,
        "differences": {"weight_max_abs": float(diff.abs().max()),
            "weight_mismatched_elements": int((diff != 0).sum()),
            "scale_max_abs": float((native_scale[0].float() - custom.scales[0]).abs().max()),
            "zero_max_abs": float((native_zero[0].float() - custom.zeros[0]).abs().max())},
        "native_search_sse": output_sse(xs, wrow, native_q[0]),
        "custom_search_sse": output_sse(xs, wrow, custom_q),
        "native_heldout_sse": output_sse(xheld, wrow, native_q[0]),
        "custom_heldout_sse": output_sse(xheld, wrow, custom_q)}


def import_l23r448(manifest_hash: str) -> list[dict]:
    imported = []
    for i in range(10):
        sid = f"S{i:02d}"
        source_path = V9 / f"{sid}.json"
        source = json.loads(source_path.read_text(encoding="utf-8"))
        if source["manifest_sha256"] != manifest_hash:
            raise RuntimeError(f"v9 raw manifest hash mismatch for {sid}")
        if source["weight_sha256"] != json.loads((V8 / "layer23_row448.json").read_text(
                encoding="utf-8"))["pristine_weight_sha256"]:
            raise RuntimeError("v8/v9 L23R448 weight hash mismatch")
        candidates = [{"alpha": c["alpha"], "search_sse": c["search_sse"],
            "heldout_sse": c["heldout_sse"], "search_output_mse": c["search_output_mse"],
            "heldout_output_mse": c["heldout_output_mse"], "weight_mse": c["weight_mse"],
            "scale": c["scale"], "zero": c["zero"], "candidate_seconds": c["candidate_seconds"],
            "weight_sha256": c["weight_sha256"]} for c in source["candidates"]]
        selected = next(c for c in candidates if c["alpha"] == source["selected_alpha_on_search"])
        baseline = candidates[-1]
        imported.append({"row": "L23R448", "layer": 23, "projection": "self_attn.q_proj",
            "split_id": sid, "source": "v9_existing", "source_v9_file": source_path.name,
            "source_v9_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
            "manifest_sha256": manifest_hash, "weight_sha256": source["weight_sha256"],
            "search_article_ids": source["search_article_ids"],
            "heldout_article_ids": source["heldout_article_ids"],
            "selected_alpha": source["selected_alpha_on_search"],
            "search_D_mse": baseline["search_output_mse"],
            "search_C_mse": selected["search_output_mse"],
            "heldout_D_mse": baseline["heldout_output_mse"],
            "heldout_C_mse": selected["heldout_output_mse"],
            "search_gain": source["search_gain"], "heldout_gain": source["heldout_gain"],
            "candidate_evaluations_this_run": 0, "candidates": candidates})
    return imported


def preflight() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("frozen v10 protocol requires CUDA")
    OUT.mkdir(parents=True, exist_ok=True)
    manifest, manifest_hash = load_frozen_manifest()
    frozen_v8 = tree_fingerprint(V8)
    frozen_v9 = tree_fingerprint(V9)
    v8_report = json.loads((V8 / "summary.json").read_text(encoding="utf-8"))
    v9_checks = json.loads((V9 / "correctness_checks.json").read_text(encoding="utf-8"))
    if v9_checks["status"] != "PASS":
        raise RuntimeError("v9 summary checks are not PASS")
    save_frozen_json(OUT / "protocol.json", {"id": "range_w4_sixrow_v10",
        "source_protocol": "range_w4_dynamic_v8 + range_w4_l23r448_v9",
        "model": "Qwen/Qwen2.5-0.5B-Instruct",
        "revision": "7ae557604adf67be50417f59c2c2f167def9a775",
        "layers": [0, 12, 23], "projection": "self_attn.q_proj", "rows": [0, 448],
        "weight_dtype": "float16", "math_dtype": "float32", "quantization": "W4A16 symmetric G128",
        "integer_range": [0, 15], "zero": 8, "rounding": "round-to-even",
        "group_size": 128, "block_size": 128, "damp_percent": .1,
        "desc_act": False, "static_groups": False, "true_sequential": True,
        "input_semantics": "FP model projection inputs", "alpha_grid": GRID,
        "method": "D=alpha1 native dynamic; C=search-MSE-selected shared-alpha dynamic GPTQ",
        "frozen_splits_manifest_sha256": manifest_hash,
        "n_new_rows": 5, "n_splits": 10, "n_alpha": 10,
        "expected_new_candidate_evaluations": 500})
    save_frozen_json(OUT / "manifest_reference.json", {"v9_splits_manifest_sha256": manifest_hash,
        "split_ids": [s["split_id"] for s in manifest["splits"]],
        "split_article_token_hashes": [{"split_id": s["split_id"],
            "search": [{k: d[k] for k in ("source_article_index", "source_row_index", "article_sha256", "token_sha256")} for d in s["search"]],
            "heldout": [{k: d[k] for k in ("source_article_index", "source_row_index", "article_sha256", "token_sha256")} for d in s["heldout"]]} for s in manifest["splits"]],
        "v8_summary_sha256": hashlib.sha256((V8 / "summary.json").read_bytes()).hexdigest(),
        "v9_summary_sha256": hashlib.sha256((V9 / "summary.json").read_bytes()).hexdigest(),
        "v8_file_fingerprints_before": frozen_v8, "v9_file_fingerprints_before": frozen_v9})
    inputs, weights, capture_meta = capture_all(manifest, manifest_hash)
    weight_checks = {}
    for layer in LAYERS:
        for row in ROWS:
            key = f"L{layer}R{row}"
            v8_raw = json.loads((V8 / f"layer{layer}_row{row}.json").read_text(encoding="utf-8"))
            weight_checks[key] = tensor_hash(weights[str(layer)][str(row)]) == v8_raw["pristine_weight_sha256"]
    spot = run_spot_parity(inputs, weights, manifest)
    save_json(OUT / "spot_parity.json", spot)
    imported = import_l23r448(manifest_hash)
    import_checks = {}
    for r in imported:
        source = json.loads((V9 / r["source_v9_file"]).read_text(encoding="utf-8"))
        import_checks[r["split_id"]] = (r["selected_alpha"] == source["selected_alpha_on_search"] and
            r["candidates"][-1]["alpha"] == 1.0 and len(r["candidates"]) == 10 and
            all(a["search_output_mse"] == b["search_output_mse"] and
                a["heldout_output_mse"] == b["heldout_output_mse"]
                for a, b in zip(r["candidates"], source["candidates"])))
    checks = {"v9_manifest_hash": manifest_hash == EXPECTED_SPLITS_SHA256,
        "ten_splits_article_token_hashes_reused": len(imported) == 10 and len(manifest["splits"]) == 10,
        "split_search_heldout_disjoint": all(not ({d["token_sha256"] for d in s["search"]} &
                                                    {d["token_sha256"] for d in s["heldout"]}) for s in manifest["splits"]),
        "all_v8_row_weight_hashes_match_captured_model": all(weight_checks.values()),
        "spot_native_dynamic_parity": spot["status"] == "PASS",
        "l23r448_import_exact": all(import_checks.values()),
        "planned_candidate_count_is_500": 5 * 10 * 10 == 500,
        "v8_v9_initial_fingerprints_recorded": bool(frozen_v8) and bool(frozen_v9),
        "activation_capture_has_80_docs_each_layer": all(len(inputs[str(layer)]) == 80 for layer in LAYERS)}
    save_json(OUT / "preflight_checks.json", {"status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks, "row_weight_hash_checks": weight_checks,
        "l23r448_import_checks": import_checks, "spot_parity": spot,
        "capture": capture_meta})
    if not all(checks.values()):
        raise RuntimeError("v10 preflight failed; formal candidates not started")
    for record in imported:
        save_json(OUT / f"{record['row']}_{record['split_id']}.json", record)
    print("PREFLIGHT PASS; five rows x ten splits x ten alphas = 500 candidate evaluations planned")


def run_formal() -> None:
    pre = json.loads((OUT / "preflight_checks.json").read_text(encoding="utf-8"))
    if pre["status"] != "PASS":
        raise RuntimeError("preflight is not PASS")
    ref = json.loads((OUT / "manifest_reference.json").read_text(encoding="utf-8"))
    manifest, manifest_hash = load_frozen_manifest()
    if ref["v9_splits_manifest_sha256"] != manifest_hash:
        raise RuntimeError("manifest changed after preflight")
    cache = torch.load(OUT / "activation_cache.pt", map_location="cpu", weights_only=True)
    if cache["manifest_sha256"] != manifest_hash:
        raise RuntimeError("activation cache manifest mismatch")
    inputs, weights = cache["inputs"], cache["weights"]
    run_start = time.perf_counter()
    total_candidates = 0
    hessian_seconds = 0.0
    for layer in LAYERS:
        module = torch.nn.Linear(896, 1, bias=False, device="cuda", dtype=torch.float16)
        module.weight.data.copy_(weights[str(layer)]["0"].cuda()[None].half())
        for split in manifest["splits"]:
            sid = split["split_id"]
            xs_docs = [inputs[str(layer)][d["token_sha256"]].cuda() for d in split["search"]]
            xs = torch.cat(xs_docs)
            xh = torch.cat([inputs[str(layer)][d["token_sha256"]] for d in split["heldout"]]).cuda()
            hcache = OUT / f"hinv_L{layer}_{sid}.pt"
            if hcache.exists():
                hobj = torch.load(hcache, map_location="cuda", weights_only=True)
                if hobj["manifest_sha256"] != manifest_hash or hobj["search_x_sha256"] != tensor_hash(xs):
                    raise RuntimeError(f"Hessian cache mismatch {layer}/{sid}")
                hinv, elapsed_h = hobj["hinv"], hobj["seconds"]
            else:
                (h, hinv, dead), elapsed_h = timed(lambda: native_hinv_from_documents(xs_docs, module, .1))
                if dead:
                    raise RuntimeError(f"dead columns: L{layer} {sid}")
                tmp = hcache.with_suffix(".pt.tmp")
                torch.save({"manifest_sha256": manifest_hash, "search_x_sha256": tensor_hash(xs),
                            "hinv": hinv.cpu(), "seconds": elapsed_h}, tmp)
                tmp.replace(hcache)
            hessian_seconds += elapsed_h
            for row in ROWS:
                row_name = f"L{layer}R{row}"
                path = OUT / f"{row_name}_{sid}.json"
                if path.exists():
                    existing = json.loads(path.read_text(encoding="utf-8"))
                    if existing.get("source") != "v10_run" or existing["manifest_sha256"] != manifest_hash:
                        continue  # L23R448 v9 import is immutable.
                    print("RESUME", row_name, sid, flush=True)
                    continue
                w = weights[str(layer)][str(row)].cuda()
                pristine_hash = tensor_hash(w)
                candidates = []
                qweights = {}
                for alpha in GRID:
                    result, seconds = timed(lambda a=alpha: dynamic_gptq(w[None], hinv, alpha=a))
                    q = result.weight_fp32[0].half()
                    search_sse = output_sse(xs, w, q)
                    held_sse = output_sse(xh, w, q)
                    candidates.append({"alpha": alpha, "search_sse": search_sse,
                        "heldout_sse": held_sse, "search_output_mse": search_sse / xs.shape[0],
                        "heldout_output_mse": held_sse / xh.shape[0], "weight_mse": weight_mse(w, q),
                        "scale": result.scales[0].tolist(), "zero": result.zeros[0].tolist(),
                        "candidate_seconds": seconds, "weight_sha256": tensor_hash(q)})
                    qweights[str(alpha)] = q.cpu()
                if tensor_hash(w) != pristine_hash:
                    raise RuntimeError(f"pristine W changed in {row_name} {sid}")
                selected = min(candidates, key=lambda c: c["search_output_mse"])
                d = candidates[-1]
                record = {"row": row_name, "layer": layer, "row_index": row,
                    "projection": "self_attn.q_proj", "split_id": sid, "source": "v10_run",
                    "manifest_sha256": manifest_hash, "weight_sha256": pristine_hash,
                    "search_article_ids": [d["source_article_index"] for d in split["search"]],
                    "heldout_article_ids": [d["source_article_index"] for d in split["heldout"]],
                    "search_x_sha256": tensor_hash(xs), "heldout_x_sha256": tensor_hash(xh),
                    "selected_alpha": selected["alpha"],
                    "search_D_mse": d["search_output_mse"], "search_C_mse": selected["search_output_mse"],
                    "heldout_D_mse": d["heldout_output_mse"], "heldout_C_mse": selected["heldout_output_mse"],
                    "search_gain": 1 - selected["search_sse"] / d["search_sse"],
                    "heldout_gain": 1 - selected["heldout_sse"] / d["heldout_sse"],
                    "candidate_evaluations_this_run": 10,
                    "candidate_seconds_total": sum(c["candidate_seconds"] for c in candidates),
                    "hessian_seconds": elapsed_h, "candidates": candidates}
                weights_path = OUT / f"weights_{row_name}_{sid}.pt"
                tmp_w = weights_path.with_suffix(".pt.tmp")
                torch.save({"manifest_sha256": manifest_hash, "weights": qweights}, tmp_w)
                tmp_w.replace(weights_path)
                save_json(path, record)
                total_candidates += 10
                print("SAVED", row_name, sid, "alpha", selected["alpha"],
                      "held gain", f"{record['heldout_gain']:.4f}", flush=True)
        del module
        torch.cuda.empty_cache()
    # Derive actual count from durable records so interruption/resume stays auditable.
    run_records = [json.loads(p.read_text(encoding="utf-8")) for p in OUT.glob("L*_S??.json")
                   if json.loads(p.read_text(encoding="utf-8")).get("source") == "v10_run"]
    save_json(OUT / "run_timing.json", {"manifest_sha256": manifest_hash,
        "candidate_evaluations_this_invocation": total_candidates,
        "candidate_evaluations_durable": sum(r["candidate_evaluations_this_run"] for r in run_records),
        "candidate_seconds_durable": sum(r["candidate_seconds_total"] for r in run_records),
        "hessian_seconds_this_invocation": hessian_seconds,
        "invocation_seconds": time.perf_counter() - run_start,
        "gpu_synchronized": True})
    print("FORMAL DONE", sum(r["candidate_evaluations_this_run"] for r in run_records), "new GPTQ candidates")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if args.preflight == args.run:
        parser.error("choose exactly one of --preflight or --run")
    if args.preflight:
        preflight()
    else:
        run_formal()


if __name__ == "__main__":
    main()
