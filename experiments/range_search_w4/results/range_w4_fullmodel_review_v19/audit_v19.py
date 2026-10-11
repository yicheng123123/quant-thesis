"""Read-only, CPU-side independent audit of frozen v18 artifacts.

Run from the repository root with .venv-gptq/Scripts/python.exe. This never
loads a model for inference or runs a GPTQ candidate.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open


HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent
V18 = RESULTS / "range_w4_fullmodel_v18"
V16 = RESULTS / "range_w4_fullmodel_baseline_v16"
V171 = RESULTS / "range_w4_m2_parity_v17_1"
V173 = RESULTS / "range_w4_m2_runtime_v17_3"
GRID = np.round(np.arange(1, 11) / 10, 1)
NAMES = ("self_attn.k_proj", "self_attn.v_proj", "self_attn.q_proj", "self_attn.o_proj", "mlp.up_proj", "mlp.gate_proj", "mlp.down_proj")


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def save(name, data):
    (HERE / name).write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()


def hbytes(b):
    return hashlib.sha256(b).hexdigest()


def tbytes(t):
    return t.detach().cpu().contiguous().numpy().tobytes()


def state_hash(s):
    h = hashlib.sha256()
    for key in ("weight", "scales", "zeros", "g_idx", "selected_alpha"):
        t = s.get(key)
        if t is not None:
            h.update(key.encode())
            h.update(tbytes(t))
    return h.hexdigest()


def layer_hash(modules):
    h = hashlib.sha256()
    for name in sorted(modules):
        h.update(name.encode()); h.update(modules[name]["state_sha256"].encode("ascii"))
    return h.hexdigest()


def alpha_hash(modules):
    h = hashlib.sha256()
    for name in sorted(modules):
        h.update(name.encode("utf-8"))
        h.update(modules[name]["selected_alpha"].contiguous().numpy().astype("<f8", copy=False).tobytes())
    return h.hexdigest()


def code_bundle():
    h = hashlib.sha256()
    for p in (V18 / "run_formal.py", V18 / "run_precheck.py", V171 / "native_m2.py", V173 / "optimized_native_m2.py"):
        h.update(p.name.encode()); h.update(bytes.fromhex(sha(p)))
    return h.hexdigest()


def check_data():
    files = ((V16 / "CALIBRATION_MANIFEST.json", V18 / "CALIBRATION_MANIFEST.json"),
             (V16 / "WIKITEXT_TEST_MANIFEST.json", V18 / "evaluation/manifests/WIKITEXT_TEST_MANIFEST.json"),
             (V16 / "C4_EVAL_MANIFEST.json", V18 / "evaluation/manifests/C4_EVAL_MANIFEST.json"))
    hashes = [{"v16": str(a), "v18": str(b), "sha256": sha(a), "byte_equal": sha(a) == sha(b)} for a, b in files]
    c, w, c4 = [load(a) for a, _ in files]
    samples = c["samples"]
    ids = [s["source_article_index"] for s in samples]
    doc_hashes = [s["article_sha256"] for s in samples]
    token_sha_ok = all(hbytes(json.dumps(s["token_ids"]).encode()) == s["token_sha256"] for s in samples)
    # Frozen v16 source-level disjointness was independently recorded before v18.
    v16_verify = load(V16 / "VERIFY_AUDIT.json")
    eval_token_hashes_ok = all(hbytes(json.dumps(d["token_ids"]).encode()) == d["token_sha256"] for d in (w,c4))
    c4_docs = c4["documents"]
    c4_indices = [d["shard_document_index"] for d in c4_docs]
    c4_doc_hashes = [d["text_sha256"] for d in c4_docs]
    c4_docs_unique = len(set(c4_indices)) == len(c4_indices) and len(set(c4_doc_hashes)) == len(c4_docs)
    out = {"status": "PASS" if all(x["byte_equal"] for x in hashes) and len(set(ids)) == len(set(doc_hashes)) == 128 and token_sha_ok and eval_token_hashes_ok and c4_docs_unique else "FAIL",
           "v16_to_v18_frozen_manifest_hashes": hashes,
           "calibration": {"dataset": c["dataset"], "split": c["split"], "n_documents": len(samples),
                           "unique_source_article_ids": len(set(ids)), "unique_full_article_hashes": len(set(doc_hashes)),
                           "tokens_each": sorted({len(s["token_ids"]) for s in samples}), "per_sample_token_hashes_valid": token_sha_ok,
                           "source_order_first_last_ids": [ids[0], ids[-1]]},
           "evaluation": {"wikitext_split": w.get("split"), "c4_split": c4.get("split"),
                          "wikitext_input_tokens": len(w["token_ids"]), "c4_input_tokens": len(c4["token_ids"]),
                          "both_token_order_hashes_valid":eval_token_hashes_ok,
                          "c4_manifest_documents":len(c4_docs),"c4_unique_document_indices_and_text_hashes":c4_docs_unique,
                          "wikitext_test_per_document_ids_available_in_manifest":False},
           "v16_pre_result_disjointness_audit": {"status":v16_verify["status"], "checks":{k:v for k,v in v16_verify["checks"].items() if k in ("calibration128_unique_whole_articles","calibration_disjoint_all_v8_v15_articles","cal_full_article_hashes_disjoint_final_eval_documents","E1_exact_test_tokens_rebuilt","E2_exact_bounded_tokens_rebuilt","source_arrow_hashes_fixed")}},
           "scope_limit": "Exact frozen artifacts and declared source split/disjointness are checked; semantic duplicates or pretraining overlap are not established."}
    save("DATA_LEAKAGE_AUDIT.json", out)
    return out


def check_eval():
    results = {}
    errors = []
    for ds, filename, input_n, scored_n, expected_windows in (("wikitext", "WIKITEXT_TEST_MANIFEST.json", 299078, 299077, 582),
                                                                 ("c4_fixed_budget", "C4_EVAL_MANIFEST.json", 131072, 131071, 253)):
        m = load(V18 / "evaluation/manifests" / filename)
        ids = np.asarray(m["token_ids"], dtype="<i8")
        token_hash = hbytes(ids.tobytes())
        results[ds] = {}
        for method in ("M0", "M1", "M2"):
            raw = load(V18 / "evaluation/raw" / f"{method}_{ds}.json")
            with (V18 / "evaluation/windows" / f"{method}_{ds}.csv").open(newline="", encoding="utf-8") as f:
                windows = list(csv.DictReader(f))
            total_nll = math.fsum(float(x["window_nll"]) for x in windows)
            total_tokens = sum(int(x["valid_tokens"]) for x in windows)
            nll = total_nll / total_tokens
            ppl = math.exp(nll)
            endpoints = [(int(x["begin"]), int(x["end"]), int(x["new_targets"]), int(x["valid_tokens"])) for x in windows]
            expected = []
            prev_end = 0
            for begin in range(0, input_n, 512):
                end = min(begin + 2048, input_n)
                new = end - prev_end
                expected.append((begin, end, new, new - 1 if begin == 0 else new))
                prev_end = end
                if end == input_n:
                    break
            ok = (len(ids) == input_n and len(windows) == expected_windows and total_tokens == scored_n and endpoints == expected
                  and token_hash == raw["evaluation_token_ids_sha256_i64le"]
                  and abs(nll - raw["average_nll"]) < 1e-12 and abs(ppl - raw["perplexity"]) < 1e-10
                  and raw["evaluation_manifest_sha256"] == sha(V18 / "evaluation/manifests" / filename)
                  and all(abs(float(x["window_loss"]) * int(x["valid_tokens"]) - float(x["window_nll"])) < 1e-8 for x in windows))
            if not ok: errors.append(f"{ds}:{method}")
            results[ds][method] = {"pass": ok, "mean_nll_recomputed": nll, "ppl_recomputed": ppl,
                                   "scored_tokens": total_tokens, "windows": len(windows), "token_ids_sha256_i64le": token_hash,
                                   "first_window": endpoints[0], "tail_window": endpoints[-1], "model_source": raw["model_source"],
                                   "model_weights_sha256": raw["model_weights_sha256"],
                                   "model_weight_files":raw["model_weight_files"]}
    save("EVALUATOR_AUDIT.json", {"status": "PASS" if not errors else "FAIL", "errors": errors,
                                  "method": "independent CSV weighted-NLL recomputation; independently reconstructed stop-at-tail boundaries and token counts", "runs": results})
    q = {}
    for ds, rr in results.items():
        m0, m1, m2 = [rr[x] for x in ("M0", "M1", "M2")]
        p0, p1, p2 = [x["ppl_recomputed"] for x in (m0, m1, m2)]
        n0, n1, n2 = [x["mean_nll_recomputed"] for x in (m0, m1, m2)]
        q[ds] = {"M0_PPL": p0, "M1_PPL": p1, "M2_PPL": p2, "M0_NLL": n0, "M1_NLL": n1, "M2_NLL": n2,
                 "M2_minus_M1_PPL": p2-p1, "M2_vs_M1_relative_PPL_change_percent": 100*(p2/p1-1),
                 "M2_minus_M1_NLL": n2-n1,
                 "PPL_degradation_recovered_fraction": (p1-p2)/(p1-p0),
                 "NLL_degradation_recovered_fraction": (n1-n2)/(n1-n0)}
    save("QUALITY_GAIN_RECALC.json", q)
    return results, q, errors


def check_alpha():
    counts = Counter(); proj = defaultdict(list)
    improved = tied = worse = 0
    gains = []
    with (V18 / "M2/selected_alpha.csv").open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            a = float(row["selected_alpha"]); counts[a] += 1
            p = row["module"].split(".")[-1]
            proj[p].append(a)
            selected, base = float(row["search_output_mse"]), float(row["alpha1_output_mse"])
            if selected < base: improved += 1
            elif selected == base: tied += 1
            else: worse += 1
            gains.append(float(row["relative_output_gain"]))
    total = sum(counts.values())
    out = {"rows": total, "counts": {str(k): counts[k] for k in GRID}, "fractions": {str(k): counts[k]/total for k in GRID},
           "mean_alpha": sum(k*v for k,v in counts.items())/total, "median_alpha": float(np.median(np.repeat(list(counts), list(counts.values())))),
           "strictly_improved_from_csv": improved, "exact_ties_from_csv": tied, "worse_from_csv": worse,
           "alpha_less_than_one": total-counts[1.0], "alpha_equal_one": counts[1.0],
           "mean_row_relative_gain": float(np.mean(gains)), "selection_tie_correspondence": improved == total-counts[1.0] and tied == counts[1.0] and worse == 0}
    save("ALPHA_RECALC.json", out)
    with (HERE / "PROJECTION_RECALC.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["projection", "rows", "mean_alpha", "alpha_1_fraction", "alpha_le_0_7_fraction"] + [f"alpha_{a:.1f}_count" for a in GRID])
        for p in sorted(proj):
            aa = np.asarray(proj[p]); c = Counter(aa)
            w.writerow([p, len(aa), float(aa.mean()), float((aa==1).mean()), float((aa<=0.7).mean())] + [c[a] for a in GRID])
    return out


def check_checkpoints(codehash, calhash, protocolhash):
    checks = Counter(); errors = []; examples = []; method_info = {}
    parity_layers = {0, 6, 12, 18, 23}
    for method in ("M1", "M2"):
        chain = "0"*64; previous_next = None; seen_rows = seen_modules = 0
        final_file = V18 / method / "checkpoint/final/model.safetensors"
        with safe_open(str(final_file), framework="pt", device="cpu") as sf:
            for layer in range(24):
                payload = torch.load(V18 / "checkpoints" / method / f"layer_{layer:02d}.pt", map_location="cpu", weights_only=False)
                meta, modules = payload["metadata"], payload["modules"]
                def require(condition, label):
                    checks[label] += 1
                    if not condition: errors.append(f"{method}:L{layer}:{label}")
                require(meta["method"] == method and meta["layer"] == layer, "layer_identity")
                require(meta["code_bundle_sha256"] == codehash and meta["calibration_manifest_sha256"] == calhash and meta["protocol_sha256"] == protocolhash, "frozen_hash_metadata")
                require(tuple(modules) == NAMES and meta["module_count"] == 7, "module_order_and_count")
                bounds = meta["boundary_activation_hashes"]
                if previous_next is not None: require(bounds["qkv_input_sha256"] == previous_next, "cross_layer_self_prefix_hash")
                previous_next = bounds.get("next_layer_qkv_input_sha256")
                for n in NAMES:
                    s = modules[n]; rows, cols = s["weight"].shape; groups = math.ceil(cols/128)
                    seen_rows += rows; seen_modules += 1
                    require(s["state_sha256"] == state_hash(s), "module_state_hash")
                    require(s["scales"].shape == s["zeros"].shape == (rows,groups), "per_row_group_scale_zero_shape")
                    require(s["g_idx"].shape == (cols,) and torch.equal(s["g_idx"].long(), torch.arange(cols)//128), "g_idx_G128_mapping")
                    require(bool(torch.all(s["zeros"] == 8)), "symmetric_zero_point_8")
                    require(bool(torch.isfinite(s["weight"]).all() and torch.isfinite(s["scales"]).all() and torch.all(s["scales"]>0)), "finite_positive_quantizer")
                    require(s["input_sha256"] == bounds[{"self_attn.k_proj":"qkv_input_sha256", "self_attn.v_proj":"qkv_input_sha256", "self_attn.q_proj":"qkv_input_sha256", "self_attn.o_proj":"o_input_sha256", "mlp.up_proj":"up_gate_input_sha256", "mlp.gate_proj":"up_gate_input_sha256", "mlp.down_proj":"down_input_sha256"}[n]], "shared_dependency_input_hash")
                    require(torch.equal(sf.get_tensor(f"model.layers.{layer}.{n}.weight"), s["weight"]), "final_checkpoint_weight_equals_layer_commit")
                    if method == "M2":
                        losses = s["candidate_losses"].numpy().astype(np.float64)
                        alpha = s["selected_alpha"].numpy()
                        best = s["best_loss"].numpy()
                        alpha1 = s["alpha1_loss"].numpy()
                        # v18 scans ascending alpha and replaces on equality with the larger alpha.
                        minloss = losses.min(axis=0)
                        require(losses.shape == (10,rows) and np.allclose(best,minloss,rtol=0,atol=1e-6), "best_loss_is_minimum_candidate")
                        require(np.allclose(alpha1,losses[9],rtol=0,atol=1e-6), "alpha1_loss_is_grid_endpoint")
                        require(np.all(np.isin(np.round(alpha,1),GRID)), "selected_alpha_in_grid")
                        idx = np.rint(alpha*10).astype(int)-1
                        require(np.allclose(best,losses[idx,np.arange(rows)],rtol=0,atol=1e-6), "selected_alpha_loss_matches_candidate")
                        if layer in parity_layers and n in ("self_attn.q_proj","self_attn.k_proj","self_attn.o_proj","mlp.up_proj","mlp.down_proj"):
                            for r in (0,rows//2,rows-1):
                                w = s["weight"][r].float(); g = s["g_idx"].long()
                                sc = s["scales"][r].float()[g]; z = s["zeros"][r].float()[g]
                                decoded = (((w/sc+z).round().clamp(0,15)-z)*sc).half()
                                exact = torch.equal(decoded,s["weight"][r])
                                require(exact, "sampled_row_integer_lattice_dequant")
                                examples.append({"layer":layer,"projection":n,"row":r,"selected_alpha":float(alpha[r]),"lattice_exact":exact})
                require(meta["row_count"] == sum(x["weight"].shape[0] for x in modules.values()), "layer_row_count")
                lh = layer_hash(modules); ah = alpha_hash(modules)
                require(meta["layer_quant_state_sha256"] == lh and meta["selected_alpha_table_sha256"] == ah, "layer_and_alpha_table_hash")
                require(meta["previous_model_state_sha256"] == chain, "previous_model_state_chain")
                chain = hbytes((chain+lh).encode("ascii"))
                require(meta["model_state_sha256"] == chain, "model_state_chain")
        method_info[method] = {"layers":24,"modules":seen_modules,"rows":seen_rows,"final_safetensors_sha256":sha(final_file),"last_model_state_sha256":chain}
    save("CHECKPOINT_AUDIT.json", {"status":"PASS" if not errors else "FAIL", "checks":dict(checks),"errors":errors,"methods":method_info,
                                   "method":"all 48 layer checkpoints individually deserialized; state/chain hashes, per-row metadata and final safetensors weights rechecked"})
    save("ROW_STITCH_FORMAL_AUDIT.json", {"status":"PASS" if not errors else "FAIL", "metadata_checks":{k:v for k,v in checks.items() if k in ("per_row_group_scale_zero_shape","g_idx_G128_mapping","symmetric_zero_point_8","selected_alpha_loss_matches_candidate","sampled_row_integer_lattice_dequant","final_checkpoint_weight_equals_layer_commit")},
           "preselected_samples":examples,"formal_semantics":"G128 scale/zero are [output_row, group], g_idx is input-column-only. With desc_act=false/static_groups=false and row-separable GPTQ compensation, selecting independent output rows from full candidates is legal.",
           "candidate_weight_provenance_limit":"v18 stores all candidate losses but only the stitched selected weight, not all ten full candidate weight tensors. Exact selected-row source tensors therefore cannot be reloaded from v18 alone; v17.3 saved candidate cache and independent selected-row parity provide prior spot evidence. In v17.3 weight/zero/g_idx/local output were exact; a few scale comparisons were allclose (rtol=1e-6, atol=1e-9), not bitwise exact."})
    return checks, errors, method_info


def main():
    codehash = code_bundle(); protocolhash = sha(V18/"PROTOCOL.json"); calhash = sha(V18/"CALIBRATION_MANIFEST.json")
    formal = load(V18/"FORMAL_RUN_MANIFEST.json")
    data = check_data(); ev, quality, eval_errors = check_eval(); alpha = check_alpha()
    ckchecks, ckerrors, ckmethods = check_checkpoints(codehash,calhash,protocolhash)
    # Inspect formal timings for the exact parity reference coverage and dependency sequence.
    with (V18/"FORMAL_TIMING.csv").open(newline="",encoding="utf-8") as f: timing = list(csv.DictReader(f))
    parity = []
    for t in timing:
        if "parity" in t["stage"].lower() or "parity" in t["detail"].lower():
            parity.append({k:t[k] for k in ("method","stage","layer","module","alpha","detail")})
    parity_all_true = all(all(json.loads(x["detail"]).values()) for x in parity)
    parity_coverage = {(int(x["layer"]),x["module"].split(".",3)[3]) for x in parity}
    full_coverage = all(set(NAMES) == {p for l,p in parity_coverage if l==layer} for layer in (0,6,12,18,23))
    save("NATIVE_SEMANTICS_AUDIT.json", {"status":"PASS" if codehash==formal["code_bundle_sha256"] and len(parity)==35 and parity_all_true and full_coverage else "FAIL",
          "code_bundle_sha256_recomputed":codehash,"matches_formal_manifest":codehash==formal["code_bundle_sha256"],
          "alpha1_native_parity_timing_records":len(parity),"all_weight_scale_zero_g_idx_exact":parity_all_true,"five_layers_seven_projections_each":full_coverage,"parity_coverage":parity,
          "source_semantics":"v17.3 prepare_native_state caches only alpha-independent H inverse; each candidate creates fresh native GPTQ core. For alpha=1 no find_params hook; for alpha<1 hook scales dynamic per-G128 scale after find_params on compensated current group. Native compensation and g_idx path remain unchanged.",
          "spot_limit":"35 full-module checks at layers 0,6,12,18,23 and seven projections; not a rerun of native reference for the other 133 modules."})
    first_m1 = torch.load(V18/"checkpoints/M1/layer_00.pt",map_location="cpu",weights_only=False)
    first_m2 = torch.load(V18/"checkpoints/M2/layer_00.pt",map_location="cpu",weights_only=False)
    same_initial = first_m1["metadata"]["boundary_activation_hashes"]["qkv_input_sha256"] == first_m2["metadata"]["boundary_activation_hashes"]["qkv_input_sha256"] and first_m1["modules"]["self_attn.k_proj"]["hessian_sha256"] == first_m2["modules"]["self_attn.k_proj"]["hessian_sha256"]
    save("BASELINE_INDEPENDENCE.json", {"status":"PASS" if ckmethods["M1"]["last_model_state_sha256"] != ckmethods["M2"]["last_model_state_sha256"] and same_initial else "FAIL",
          "M1_source":"run_formal.run_method('M1') invokes load_model(original FP16 checkpoint) independently and quantize_native_module; no M2 candidate is used",
          "M2_source":"run_formal.run_method('M2') invokes a second load_model(original FP16 checkpoint) and commits its own stitched module/layer checkpoints",
          "M1_last_chain":ckmethods["M1"]["last_model_state_sha256"],"M2_last_chain":ckmethods["M2"]["last_model_state_sha256"],
          "same_initial_layer0_qkv_input_and_k_hessian_hashes":same_initial,
          "distinct_final_weight_hashes":ckmethods["M1"]["final_safetensors_sha256"]!=ckmethods["M2"]["final_safetensors_sha256"]})
    save("FORMAL_SELF_PREFIX_AUDIT.json", {"status":"PASS" if not ckerrors else "FAIL",
          "checked_boundary_hash_count":ckchecks["shared_dependency_input_hash"],"cross_layer_transition_hash_count":ckchecks["cross_layer_self_prefix_hash"],
          "dependency_order":["q/k/v shared norm input; commit all three","capture o input from committed q/k/v","commit o; capture up/gate shared input","commit up and gate; capture down input","commit down; forward to next-layer qkv input"],
          "source":"v18 run_formal.run_method, run_block_capture, run_block_to_next and 48 layer checkpoint boundary hashes",
          "limit":"Hash chain and source control flow support formal self-prefix; this audit does not recompute 128x2048 GPU activations."})
    reloads = {m:load(V18/m/"RELOAD_CORRECTNESS.json") for m in ("M1","M2")}
    file_hashes_ok = all(ev[ds][m]["model_weight_files"][0]["sha256"]==ckmethods[m]["final_safetensors_sha256"] for ds in ev for m in ("M1","M2"))
    same_stream_weights_ok = all(ev["wikitext"][m]["model_weights_sha256"]==ev["c4_fixed_budget"][m]["model_weights_sha256"] for m in ("M0","M1","M2"))
    timing_counts = {stage:sum(t["method"]=="M2" and t["stage"]==stage for t in timing) for stage in ("full_linear_alpha_candidate_gptq","alpha1_native_parity_reference","full_linear_row_stitch")}
    timing_ok = timing_counts == {"full_linear_alpha_candidate_gptq":1680,"alpha1_native_parity_reference":35,"full_linear_row_stitch":168}
    manifest_ok = data["status"]=="PASS" and not eval_errors and not ckerrors and alpha["selection_tie_correspondence"] and alpha["rows"]==304128 and all(x["pass"] and x["max_abs_logit_difference"]==0 for x in reloads.values())
    checks = {"frozen_data":data["status"]=="PASS","evaluation_recalc":not eval_errors,"checkpoint_and_stitch":not ckerrors,
              "alpha_table":alpha["selection_tie_correspondence"] and alpha["rows"]==304128,
              "code_bundle_unchanged":codehash==formal["code_bundle_sha256"],"reload_logits_exact":all(x["pass"] and x["max_abs_logit_difference"]==0 for x in reloads.values()),
              "parity_35_exact":parity_all_true and full_coverage,"final_model_file_matches_eval":file_hashes_ok,"same_weight_sources_for_both_eval_streams":same_stream_weights_ok,"timing_execution_counts":timing_ok}
    save("correctness_checks.json", {"status":"PASS" if manifest_ok and all(checks.values()) else "FAIL","checks":checks,"checkpoint_errors":ckerrors,"evaluation_errors":eval_errors,"timing_counts":timing_counts})
    print(json.dumps({"status":"PASS" if manifest_ok and all(checks.values()) else "FAIL", "checks":checks,"quality":quality,"alpha":alpha,"parity_records":len(parity)},indent=2))


if __name__ == "__main__": main()
