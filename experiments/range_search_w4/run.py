"""Run document-disjoint local W4 range diagnostics; no checkpoint is modified."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
os.environ.update({
    "HF_HOME": str(ROOT / ".hf_cache"),
    "HF_HUB_CACHE": str(ROOT / ".hf_cache" / "hub"),
    "HF_HUB_OFFLINE": "1",
    "HF_DATASETS_OFFLINE": "1",
})
sys.path.insert(0, str(ROOT))

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from experiments.range_search_w4.core import (
    GRID, candidate_grids, gptq_candidates, group_errors, make_grid,
    output_mse, prepare_hinv, quantize, weight_mse,
)

HERE = Path(__file__).resolve().parent
PROTOCOL = json.loads((HERE / "protocol.json").read_text(encoding="utf-8"))
SNAPSHOT = ROOT / ".hf_cache" / "hub" / "models--Qwen--Qwen2.5-0.5B-Instruct" / "snapshots" / PROTOCOL["model_revision"]
RESULTS_ROOT = HERE / "results"


def sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def select_documents(tokenizer):
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train", cache_dir=str(ROOT / ".hf_cache" / "datasets"))
    articles = []
    current = []
    start = None
    for i, record in enumerate(ds):
        line = record["text"]
        if re.match(r"^= [^=].* =$", line.strip()) and ";" not in line:
            if current:
                articles.append((start, "\n".join(current)))
            current, start = [line], i
        elif current:
            current.append(line)
    if current:
        articles.append((start, "\n".join(current)))
    eligible = []
    for source_index, (row_index, article) in enumerate(articles):
        ids = tokenizer(article, add_special_tokens=False)["input_ids"]
        if len(ids) >= PROTOCOL["tokens_per_document"]:
            ids = ids[:PROTOCOL["tokens_per_document"]]
            digest = hashlib.sha256(json.dumps(ids).encode()).hexdigest()
            eligible.append({"source_article_index": source_index, "source_row_index": row_index, "article_sha256": hashlib.sha256(article.encode()).hexdigest(), "sha256": digest, "ids": ids})
        if len(eligible) == 8:
            break
    if len(eligible) != 8 or len({d["source_article_index"] for d in eligible}) != 8:
        raise RuntimeError("eight disjoint articles unavailable")
    return eligible


@torch.inference_mode()
def capture():
    if not torch.cuda.is_available():
        raise RuntimeError("this frozen protocol requires CUDA")
    torch.manual_seed(PROTOCOL["seed"])
    torch.cuda.reset_peak_memory_stats()
    tokenizer = AutoTokenizer.from_pretrained(SNAPSHOT, local_files_only=True)
    docs = select_documents(tokenizer)
    model = AutoModelForCausalLM.from_pretrained(SNAPSHOT, torch_dtype=torch.float16, local_files_only=True, device_map="cuda", low_cpu_mem_usage=True)
    model.eval()
    captured = {str(i): [] for i in PROTOCOL["layers"]}
    handles = []
    for layer in PROTOCOL["layers"]:
        module = model.model.layers[layer].self_attn.q_proj
        def hook(_module, inputs, _output, key=str(layer)):
            captured[key].append(inputs[0].detach().reshape(-1, inputs[0].shape[-1]).float().cpu())
        handles.append(module.register_forward_hook(hook))
    for doc in docs:
        ids = torch.tensor(doc["ids"], device="cuda").unsqueeze(0)
        model(input_ids=ids, use_cache=False)
    for h in handles:
        h.remove()
    weights = {str(i): model.model.layers[i].self_attn.q_proj.weight.detach().float().cpu().clone() for i in PROTOCOL["layers"]}
    peak = {"allocated_bytes": torch.cuda.max_memory_allocated(), "reserved_bytes": torch.cuda.max_memory_reserved()}
    del model
    torch.cuda.empty_cache()
    return docs, captured, weights, peak


def metric_record(name, g, w, q, xs, xh, count, seconds, peak, *, selected_on):
    return {
        "method": name, "selected_on": selected_on,
        "alpha": g.alpha, "beta": g.beta, "scale": g.scale, "zero": g.zero,
        "actual_low": g.low, "actual_high": g.high,
        "search_output_mse": output_mse(xs, w, q),
        "heldout_output_mse": output_mse(xh, w, q),
        "weight_mse": weight_mse(w, q),
        "candidate_evaluations": count, "evaluation_seconds": seconds,
        "peak_allocated_bytes": peak,
    }


def evaluate_row(w, xs, xh, hinv, *, sym):
    start = time.perf_counter()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    grids, nominal = candidate_grids(w, sym=sym)
    dense_grids, dense_nominal = (candidate_grids(w, sym=False, dense_single=True) if not sym else ([], 0))
    all_grids = grids + dense_grids
    rtn = torch.stack([quantize(w, g) for g in all_grids])
    error = rtn - w[None, :]
    weight_losses = error.square().mean(1)
    output_losses = (xs @ error.T).square().mean(0)
    gptq = gptq_candidates(w, [[g] for g in all_grids], hinv, chunk_size=16)
    gptq_losses = (xs @ (gptq - w[None, :]).T).square().mean(0)
    sync()
    elapsed = time.perf_counter() - start
    peak = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
    split = len(grids)
    families = {"single_10": list(range(split))} if sym else {
        "independent_100": list(range(split)),
        "single_10": [i for i in range(split) if grids[i].alpha == grids[i].beta],
        "single_100": list(range(split, len(all_grids))),
    }
    records = []
    for family, indices in families.items():
        baseline_idx = next(i for i in indices if all_grids[i].alpha == 1 and all_grids[i].beta == 1)
        selections = {
            "A_minmax_fixed_gptq": baseline_idx,
            "B_rtn_weight_fixed_gptq": min(indices, key=lambda i: float(weight_losses[i])),
            "B_rtn_output_fixed_gptq": min(indices, key=lambda i: float(output_losses[i])),
            "C_gptq_output_fixed_gptq": min(indices, key=lambda i: float(gptq_losses[i])),
        }
        for name, index in selections.items():
            rec = metric_record(f"{family}/{name}", all_grids[index], w, gptq[index], xs, xh, len(indices), elapsed, peak, selected_on=name.split("_")[1])
            rec.update({"rtn_search_output_mse": float(output_losses[index]), "rtn_weight_mse": float(weight_losses[index]), "gptq_search_output_mse": float(gptq_losses[index]), "nominal_candidates": 10 if family == "single_10" else (dense_nominal if family == "single_100" else nominal), "unique_candidates": len(indices), "shared_evaluated_candidates_all_families": len(all_grids), "evaluation_seconds_scope": "all candidates of this quantizer format, shared across four method selections", "quantizer_format": "symmetric" if sym else "asymmetric"})
            records.append(rec)
    return records


def group_diagnostic(w, xs, xh, hinv, group_size=128):
    start = time.perf_counter()
    n = (w.numel() + group_size - 1) // group_size
    selected = []
    minmax = []
    group_choices = []
    group_candidate_counts = []
    for group in range(n):
        s = group * group_size
        e = min(s + group_size, w.numel())
        choices, nominal = candidate_grids(w[s:e], sym=True)
        group_choices.append(choices)
        group_candidate_counts.append({"group": group, "nominal": nominal, "unique": len(choices)})
        errors = [output_mse(xs[:, s:e], w[s:e], quantize(w[s:e], g)) for g in choices]
        selected.append(choices[min(range(len(choices)), key=lambda i: errors[i])])
        minmax.append(make_grid(w[s:e], sym=True))
    def rtn(gs):
        return torch.cat([quantize(w[i * group_size:(i + 1) * group_size], g) for i, g in enumerate(gs)])
    q0, q1 = rtn(minmax), rtn(selected)
    records = {}
    records["group_candidate_counts"] = group_candidate_counts
    for name, q in (("minmax", q0), ("independent_group_search", q1)):
        records[name] = {"search": group_errors(xs, w, q, group_size), "heldout": group_errors(xh, w, q, group_size), "weight_mse": weight_mse(w, q)}
    ranking = []
    for group, choices in enumerate(group_choices):
        s, e = group * group_size, min((group + 1) * group_size, w.numel())
        candidates = [quantize(w[s:e], g) for g in choices]
        item = {"group": group}
        for name, x in (("search", xs), ("heldout", xh)):
            local = [output_mse(x[:, s:e], w[s:e], qg) for qg in candidates]
            full = [output_mse(x, w, torch.cat((q0[:s], qg, q0[e:]))) for qg in candidates]
            li = min(range(len(choices)), key=lambda i: local[i])
            fi = min(range(len(choices)), key=lambda i: full[i])
            item[name] = {"independent_alpha": choices[li].alpha, "full_alpha_with_other_groups_minmax": choices[fi].alpha, "ranking_disagrees": li != fi}
        ranking.append(item)
    records["ranking_diagnostic"] = ranking
    # The coordinate diagnostic only replaces a group's RTN weights. The
    # residual cache is valid for this semantics, never for a GPTQ rerun.
    current = selected.copy()
    q = q1.clone()
    residual = xs @ (q - w)
    accepted = []
    for group in range(n):
        s, e = group * group_size, min((group + 1) * group_size, w.numel())
        choices, _ = candidate_grids(w[s:e], sym=True)
        current_loss = float(residual.square().mean())
        for g in choices:
            replacement = quantize(w[s:e], g)
            trial_residual = residual + xs[:, s:e] @ (replacement - q[s:e])
            trial_loss = float(trial_residual.square().mean())
            if trial_loss < current_loss - 1e-12:
                best = (g, replacement, trial_residual, trial_loss)
                current_loss = trial_loss
        if current_loss < float(residual.square().mean()) - 1e-12:
            g, replacement, residual, loss = best
            q[s:e] = replacement
            current[group] = g
            accepted.append({"group": group, "alpha": g.alpha, "search_full_mse_after": loss})
    if not torch.allclose(residual, xs @ (q - w), rtol=1e-4, atol=1e-5):
        raise AssertionError("cached residual differs from direct recomputation")
    records["coordinate_replace_one_sweep"] = {"search": group_errors(xs, w, q, group_size), "heldout": group_errors(xh, w, q, group_size), "accepted": accepted, "semantics": "RTN replace one group only"}
    # GPTQ may change all later groups when a grid changes: recompute both.
    qg0, qg1 = gptq_candidates(w, [minmax, selected], hinv, group_size=group_size, chunk_size=2)
    for name, qg in (("gptq_minmax_external", qg0), ("gptq_independent_external", qg1)):
        records[name] = {"search": group_errors(xs, w, qg, group_size), "heldout": group_errors(xh, w, qg, group_size), "weight_mse": weight_mse(w, qg), "semantics": "full GPTQ rerun for changed grid"}
    sync()
    records["cost"] = {
        "independent_rtn_candidate_evaluations": n * 10,
        "ranking_metric_evaluations": n * 10 * 4,
        "coordinate_rtn_candidate_evaluations": n * 10,
        "full_gptq_candidate_evaluations": 2,
        "total_metric_evaluations": n * 10 * 6 + 2,
        "seconds_including_cache": time.perf_counter() - start,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0,
    }
    return records


def atomic_json(path, payload):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("smoke", "representative"), default="smoke")
    parser.add_argument("--run-name", default=PROTOCOL["id"], help="new isolated result directory name")
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.run_name):
        raise ValueError("run-name must be a simple directory name")
    results = RESULTS_ROOT / args.run_name
    results.mkdir(parents=True, exist_ok=True)
    layers = PROTOCOL["layers"][:1] if args.phase == "smoke" else PROTOCOL["layers"]
    row_ids = PROTOCOL["row_indices"][:1] if args.phase == "smoke" else PROTOCOL["row_indices"]
    if all((results / f"layer{layer}_row{row_id}.json").exists() for layer in layers for row_id in row_ids):
        cached_manifest = results / "manifest.json"
        if not cached_manifest.exists() or json.loads(cached_manifest.read_text(encoding="utf-8"))["protocol"] != PROTOCOL:
            raise RuntimeError("cached rows lack the current frozen protocol manifest")
        print(f"all requested rows already cached in {results}; no files changed")
        return
    started = time.perf_counter()
    docs, captured, weights, capture_peak = capture()
    sync()
    capture_seconds = time.perf_counter() - started
    manifest = {"protocol": PROTOCOL, "documents": [{k: v for k, v in d.items() if k != "ids"} for d in docs], "capture_seconds": capture_seconds, "capture_peak_cuda": capture_peak, "torch": torch.__version__, "device": torch.cuda.get_device_name(0)}
    prior_manifest = results / "manifest.json"
    previous = None
    if prior_manifest.exists():
        previous = json.loads(prior_manifest.read_text(encoding="utf-8"))
        if previous["protocol"] != PROTOCOL or previous["documents"] != manifest["documents"]:
            raise RuntimeError("existing run manifest differs from frozen protocol or documents")
    else:
        atomic_json(prior_manifest, manifest)
    rows = []
    hessian_seconds = {}
    for layer in layers:
        key = str(layer)
        xs = torch.cat(captured[key][:4]).cuda()
        xh = torch.cat(captured[key][4:]).cuda()
        h_start = time.perf_counter()
        hinv = prepare_hinv(xs, document_count=4, damp_percent=PROTOCOL["damp_percent"])
        sync()
        hessian_seconds[key] = time.perf_counter() - h_start
        for row_id in row_ids:
            target = results / f"layer{layer}_row{row_id}.json"
            if target.exists():
                payload = json.loads(target.read_text(encoding="utf-8"))
                if payload["document_hashes"] != [d["sha256"] for d in docs]:
                    raise RuntimeError("cached row has a different document manifest")
            else:
                w = weights[key][row_id].cuda()
                t0 = time.perf_counter()
                sym = evaluate_row(w, xs, xh, hinv, sym=True)
                asym = evaluate_row(w, xs, xh, hinv, sym=False)
                group = group_diagnostic(w, xs, xh, hinv)
                sync()
                payload = {"layer": layer, "row": row_id, "document_hashes": [d["sha256"] for d in docs], "symmetric": sym, "asymmetric_format_control": asym, "group_diagnostic": group, "row_total_seconds": time.perf_counter() - t0}
                atomic_json(target, payload)
            for rec in payload["symmetric"] + payload["asymmetric_format_control"]:
                rows.append({"layer": layer, "row": row_id, **rec})
            print(f"completed layer={layer} row={row_id} seconds={payload['row_total_seconds']:.1f}", flush=True)
    if args.phase == "representative":
        method_file = results / "methods.csv"
        if not method_file.exists():
            with method_file.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
    invocation_seconds = time.perf_counter() - started
    manifest["total_seconds_this_invocation"] = invocation_seconds
    manifest["total_seconds_all_invocations"] = invocation_seconds + (previous.get("total_seconds_all_invocations", previous.get("total_seconds_this_invocation", 0)) if previous else 0)
    manifest["capture_seconds"] += previous.get("capture_seconds", 0) if previous else 0
    if previous:
        manifest["capture_peak_cuda"]["allocated_bytes"] = max(manifest["capture_peak_cuda"]["allocated_bytes"], previous["capture_peak_cuda"]["allocated_bytes"])
        manifest["capture_peak_cuda"]["reserved_bytes"] = max(manifest["capture_peak_cuda"]["reserved_bytes"], previous["capture_peak_cuda"]["reserved_bytes"])
        for key, value in previous.get("hessian_seconds_by_layer", {}).items():
            hessian_seconds[key] = hessian_seconds.get(key, 0) + value
    manifest["hessian_seconds_by_layer"] = hessian_seconds
    manifest["completed_rows"] = [[r["layer"], r["row"]] for r in rows if r["method"] == "single_10/A_minmax_fixed_gptq" and r["quantizer_format"] == "symmetric"]
    atomic_json(prior_manifest, manifest)


if __name__ == "__main__":
    main()
