"""CPU-only, read-only source audit and v51 protocol freeze."""
import csv, hashlib, json, math, shutil, time
from pathlib import Path
import torch
from safetensors import safe_open

ROOT = Path(r"E:\quant-thesis")
BASE = ROOT / "experiments/range_search_w4/results"
OUT = BASE / "v51_m2_fast_integer_repair_fullmodel"
V18 = BASE / "range_w4_fullmodel_v18"
V42 = BASE / "v42_m2_single_integer_repair_fullmodel"
V45 = BASE / "v45_m2_single_integer_repair_fullmodel"
V46 = BASE / "v46_m1_single_integer_repair_fullmodel"
V48 = BASE / "v48_fast_numerical_acceptance"
V49 = BASE / "v49_fullmodel_fast_integer_repair"
EXPECTED_REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"
EXPECTED_CAL = "664419273f63df3c0b0746b18bb66b593a5384bbe6ba8afcd6f4905c9479aae3"

torch.set_num_threads(4)

def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()

def tensor_sha(t):
    return hashlib.sha256(t.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()

def pack(c):
    return (c[:, 0::2] | (c[:, 1::2] << 4)).contiguous()

def unpack(p):
    return torch.stack((p & 15, p >> 4), dim=-1).flatten(1).contiguous()

def dump(name, obj):
    path = OUT / name
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)

def main():
    start = time.perf_counter()
    assert OUT.is_dir() and all((OUT / p).is_dir() for p in ["original_codes", "formal/groups", "formal/rows", "R1/checkpoint/final", "evaluation", "reports", "diagnostics", "recovery"])
    assert not (OUT / "GPU_START.json").exists() and not (OUT / "PROTOCOL.json").exists(), "v51 already initialized; preserve and inspect"

    meta = json.loads((V18 / "M2/QUANTIZATION_METADATA.json").read_text(encoding="utf-8"))
    formal = json.loads((V18 / "FORMAL_RUNTIME.json").read_text(encoding="utf-8"))
    run_manifest = json.loads((V18 / "FORMAL_RUN_MANIFEST.json").read_text(encoding="utf-8"))
    calpath = V18 / "CALIBRATION_MANIFEST.json"
    m2path = V18 / "M2/checkpoint/final/model.safetensors"
    m2config = V18 / "M2/checkpoint/final/config.json"
    pristine = ROOT / ".hf_cache/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots" / EXPECTED_REVISION
    pristine_weights = pristine / "model.safetensors"
    pristine_index = pristine / "model.safetensors.index.json"
    assert sha(calpath) == EXPECTED_CAL == meta["calibration_manifest_sha256"]
    assert sha(m2path) == formal["m2"]["final_model_sha256"]
    assert meta["model_revision"] == EXPECTED_REVISION == formal["model_revision"] if "model_revision" in formal else meta["model_revision"] == EXPECTED_REVISION
    assert meta["all_expected_modules"] == 168 and meta["all_expected_output_rows"] == 304128
    assert meta["format"]["weight_bits"] == 4 and meta["format"]["activation_bits"] == 16
    assert meta["format"]["group_size"] == 128 and meta["format"]["symmetric"] is True
    assert meta["format"]["desc_act"] is False and meta["format"]["static_groups"] is False
    assert run_manifest["status"] == "COMPLETE" and run_manifest["results"]["m2"] == "COMPLETE_RELOAD_PASS"
    assert pristine_weights.exists() or pristine_index.exists(), "pristine FP checkpoint weight source absent"
    fp_files = [pristine_weights] if pristine_weights.exists() else [pristine / n for n in sorted(set(json.loads(pristine_index.read_text())["weight_map"].values()))]
    assert all(p.exists() for p in fp_files)

    # Validate token identity and dimensions from the frozen 128 x 2048 manifest.
    cal = json.loads(calpath.read_text(encoding="utf-8"))
    assert len(cal["samples"]) == 128 and all(len(s["token_ids"]) == 2048 for s in cal["samples"])
    token_bytes = b"".join(torch.tensor(s["token_ids"], dtype=torch.int64).numpy().astype("<i8", copy=False).tobytes() for s in cal["samples"])
    token_sha = hashlib.sha256(token_bytes).hexdigest()
    saved_token_sha = torch.load(V18 / "checkpoints/M2/layer_00.pt", map_location="cpu", weights_only=False)["metadata"]["calibration_ordered_token_ids_sha256"]
    assert token_sha == saved_token_sha

    v42_audit = json.loads((V42 / "SOURCE_AUDIT.json").read_text(encoding="utf-8"))
    v45_audit = json.loads((V45 / "SOURCE_AUDIT.json").read_text(encoding="utf-8"))
    v42_rows = list(csv.DictReader((V42 / "SOURCE_MODULES.csv").open(encoding="utf-8", newline="")))
    v45_sources = v45_audit["sources"]
    assert v42_audit["pass"] and v45_audit["pass"] and len(v45_sources) == 24 and len(v42_rows) == 168
    assert v42_audit["model_sha256"] == v45_audit["model_sha256"] == sha(m2path)
    assert v42_audit["calibration_sha256"] == v45_audit["calibration_sha256"] == EXPECTED_CAL
    assert "Unique bitwise enumeration" in v45_audit["code_provenance"] and "no approximate" in v45_audit["code_provenance"]
    source_rows = {(int(r["layer"]), r["module"]): r for r in v42_rows}
    staged_layers = []
    module_rows = []
    with safe_open(str(m2path), framework="pt", device="cpu") as sf:
        for li, source_record in enumerate(v45_sources):
            native = V18 / f"checkpoints/M2/layer_{li:02d}.pt"
            previous = V42 / f"original_codes/layer_{li:02d}.pt"
            v45_codes = V45 / f"original_codes/layer_{li:02d}.pt"
            expected_layer_sha = source_record["derived_sha256"]
            assert source_record["source_sha256"] == sha(native)
            assert source_record["derived"] == str(previous) and sha(previous) == expected_layer_sha
            assert sha(v45_codes) == expected_layer_sha, f"v45 M2 original code source changed at layer {li}"
            staged = OUT / f"original_codes/layer_{li:02d}.pt"
            shutil.copy2(v45_codes, staged)
            assert sha(staged) == expected_layer_sha
            payload = torch.load(native, map_location="cpu", weights_only=False)
            assert payload["metadata"]["method"] == "M2" and payload["metadata"]["layer"] == li
            assert payload["metadata"]["calibration_manifest_sha256"] == EXPECTED_CAL
            modules = torch.load(staged, map_location="cpu", weights_only=True)
            assert set(modules) == set(payload["modules"])
            for name, state in payload["modules"].items():
                src = modules[name]
                # Revalidate saved module state hash; no approximate inverse from W is used here.
                hh = hashlib.sha256()
                for key in ("weight", "scales", "zeros", "g_idx", "selected_alpha"):
                    hh.update(key.encode()); hh.update(state[key].contiguous().numpy().tobytes())
                assert hh.hexdigest() == state["state_sha256"]
                assert torch.equal(sf.get_tensor(f"model.layers.{li}.{name}.weight").view(torch.int16), state["weight"].view(torch.int16))
                code = src["codes"].to(torch.uint8)
                scales = state["scales"]
                zeros = state["zeros"]
                g_idx = state["g_idx"].long()
                w = state["weight"]
                assert code.shape == w.shape and bool(((code >= 0) & (code <= 15)).all())
                assert torch.equal(src["scales"], scales) and torch.equal(src["zeros"], zeros) and torch.equal(src["g_idx"], state["g_idx"])
                assert torch.equal(zeros, torch.full_like(zeros, 8))
                assert torch.equal(g_idx, torch.arange(w.shape[1]) // 128)
                assert torch.equal(unpack(pack(code)), code)
                assert torch.equal(src["packed_u4"], pack(code))
                # Exact bitwise grid validation against all 16 legal values in row chunks.
                unique_counts = torch.zeros_like(code)
                for lo in range(0, w.shape[0], 128):
                    wc = w[lo:lo+128].view(torch.int16)
                    sc = scales[lo:lo+128, g_idx]
                    cc = code[lo:lo+128]
                    effective = (sc * (cc.float() - 8)).half()
                    assert torch.equal(effective.view(torch.int16), wc)
                    counts = torch.zeros_like(cc)
                    for k in range(16):
                        counts += ((sc * (k - 8)).half().view(torch.int16) == wc).to(torch.uint8)
                    assert bool((counts == 1).all()), f"M2 grid does not uniquely recover code: {li}/{name}"
                    unique_counts[lo:lo+128] = counts
                recorded = source_rows[(li, name)]
                assert recorded["code_sha256"] == tensor_sha(code)
                assert recorded["exact_unique_mapping"] == "True" and recorded["pack_roundtrip"] == "True"
                module_rows.append({"layer": li, "module": name, "rows": code.shape[0], "input_dim": code.shape[1], "code_sha256": tensor_sha(code), "scale_sha256": tensor_sha(scales), "zero_sha256": tensor_sha(zeros), "g_idx_sha256": tensor_sha(g_idx.int()), "exact_m2_grid_bitwise": True, "all16_code_mapping_unique": True, "pack_roundtrip": True, "m2_state_sha256": state["state_sha256"], "m2_selected_alpha_sha256": tensor_sha(state["selected_alpha"])})
            staged_layers.append({"layer": li, "m2_native_checkpoint": str(native), "native_checkpoint_sha256": sha(native), "v42_exact_code_file_sha256": expected_layer_sha, "v45_staged_source_sha256": sha(v45_codes), "v51_staged_sha256": sha(staged), "modules": len(modules)})
    assert len(module_rows) == 168 and sum(x["rows"] for x in module_rows) == 304128

    # Frozen v49 numerical implementation and helper provenance.
    v49_runner = V49 / "run_v49.py"
    runner_bytes = v49_runner.read_bytes()
    source_hashes = {
        "v49_run_v49_sha256": sha(v49_runner),
        "v48_fast_accept_sha256": sha(V48 / "fast_accept.py"),
        "v48_acceptance_rules_sha256": sha(V48 / "ACCEPTANCE_RULES.md"),
        "v48_runner_sha256": sha(V48 / "run_v48.py"),
        "v46_runtime_sha256": sha(V46 / "runtime_v46.py"),
        "v45_repair_helpers_sha256": sha(V46 / "repair_v45.py"),
        "v45_precision_check_sha256": sha(V46 / "precision_check.py"),
        "v49_runner_template_sha256": sha(v49_runner),
    }
    expected_v49 = json.loads((V49 / "FINAL_RUNNER_LOCK.json").read_text(encoding="utf-8"))
    assert source_hashes["v48_fast_accept_sha256"] == expected_v49["fast_accept_sha256"]
    assert source_hashes["v48_acceptance_rules_sha256"] == expected_v49["frozen_rules_sha256"]
    assert source_hashes["v49_run_v49_sha256"] == expected_v49["runner_sha256"]
    # These result source files must exist and retain audited hashes before quality comparison.
    ppl_audit = json.loads((V46 / "PPL_SOURCE_AUDIT.json").read_text(encoding="utf-8"))
    assert ppl_audit["pass"] and not ppl_audit["P3_read"]
    for record in ppl_audit["historical_source_records"].values():
        assert sha(record["result_path"]) == record["result_sha256"]
        assert sha(record["windows_path"]) == record["windows_sha256"]
    v49_ppl = {}
    v49_lock = json.loads((V49 / "CHECKPOINT_LOCK.json").read_text(encoding="utf-8"))
    for ds in ("wikitext", "c4_fixed_budget"):
        rp = V49 / f"R1_{ds}.json"
        wp = V49 / f"R1_{ds}_WINDOWS.csv"
        rr = json.loads(rp.read_text(encoding="utf-8"))
        assert rr["model_sha256"] == v49_lock["final_model_sha256"]
        v49_ppl[ds] = {"result_path": str(rp), "result_sha256": sha(rp), "windows_path": str(wp), "windows_sha256": sha(wp), "ppl": rr["ppl"], "nll": rr["nll"], "valid_tokens": rr["scored_tokens"], "windows": rr["windows"]}

    m2_build_seconds = float(formal["m2"]["wall_seconds_including_load_replay_save"])
    assert abs(m2_build_seconds - 2115.9353341000096) < 1e-6
    v49_blocks = list(csv.DictReader((V49 / "BLOCK_TIMINGS.csv").open(encoding="utf-8", newline="")))
    measured_blocks = [float(x["seconds"]) for x in v49_blocks if int(x["new_groups"]) > 0]
    assert measured_blocks
    v49_max_block = max(measured_blocks)
    reserve = 600.0
    setup = 180.0
    safety = 1.5
    projected = 24 * safety * v49_max_block + reserve + setup
    assert projected < 10800, f"conservative projection {projected:.1f}s exceeds 10800s"

    protocol = {
        "version": "v51", "status": "FROZEN_BEFORE_GPU",
        "requested_model_setting": "Luna Max requested; current interface cannot verify or switch settings.",
        "objective": "Complete formal M2 scalar10 initialization plus exact v49/v48 frozen fast single-coordinate integer repair; no v50 path.",
        "initialization": {"method": "M2 scalar10", "checkpoint": str(V18 / "M2/checkpoint/final"), "model_sha256": sha(m2path), "config_sha256": sha(m2config), "model_revision": EXPECTED_REVISION, "quantization_metadata_sha256": sha(V18 / "M2/QUANTIZATION_METADATA.json"), "quantization_correctness_sha256": sha(V18 / "M2/QUANTIZATION_CORRECTNESS.json"), "formal_runtime_sha256": sha(V18 / "FORMAL_RUNTIME.json"), "build_wall_seconds_historical": m2_build_seconds},
        "calibration": {"manifest_sha256": EXPECTED_CAL, "ordered_token_ids_sha256": token_sha, "documents": 128, "tokens_each": 2048, "evaluation_separated": True},
        "quantization": {"weight_bits": 4, "activation_bits": 16, "group_size": 128, "symmetric": True, "zero": 8, "codes": [0, 15], "scale_zero_g_idx_fixed": True, "dequant_grid": "FP16 round(scales[row,g_idx[col]] * (code-8))", "full_model_linear_modules": 168, "output_rows": 304128},
        "integer_code_provenance": {"source": "v18 M2 trusted dense FP16 layer state plus its exact saved scales/zeros/g_idx; codes reused from the v42/v45 all-16-value bitwise enumeration audit", "no_approximate_weight_rounding": True, "all16_mapping_unique_revalidated": True, "M2_file_sha256": v45_audit["model_sha256"], "staged_layer_files": staged_layers, "module_grid_audit": "M2_MODULE_GRID_AUDIT.csv"},
        "algorithm_source": {"v49_runner_template_sha256": source_hashes["v49_run_v49_sha256"], "v48_fast_accept_path": str(V48 / "fast_accept.py"), "fast_accept_sha256": source_hashes["v48_fast_accept_sha256"], "acceptance_rules": str(V48 / "ACCEPTANCE_RULES.md"), "acceptance_rules_sha256": source_hashes["v48_acceptance_rules_sha256"], "v46_runtime_sha256": source_hashes["v46_runtime_sha256"], "v45_helper_sha256": source_hashes["v45_repair_helpers_sha256"], "v45_precision_sha256": source_hashes["v45_precision_check_sha256"], "acceptance": "exact v49 wrapper dispatches to v48 fast_accept.batch_repair; FP64 quadratic proposal with v48 empirical envelope, fixed raw-X FP64 uncertain-candidate checks, p/B refresh at 1,8,16,24,32; no certificate", "core_algorithm_modified": False, "v50_gram_path_used": False},
        "fixed_search": {"neighborhood": "legal actual FP16 grid deltas for +/-1 code", "max_accepted_per_row": 32, "candidate_budget": "128 * input dimension per row", "tie_breaking": "v45 deterministic row/coordinate/offset ordering", "C": "full undamped FP64 Gram, original v49 builder", "sibling_inputs_and_C": "shared only for sibling linears with the same current-run input", "propagation": ["qkv", "o", "up_gate", "down", "next block"], "group_commit": "atomic after all sibling matrices and checks pass", "prefix_cache": "none from v45 or v49; reconstruct and propagate current v51 model"},
        "restore": {"start": "original formal M2 each process", "committed_group": "restore only a complete atomic checkpoint after protocol/grid/code/packed/dequantized-weight and current-prefix input hash checks", "failed_group": "discard incomplete scratch and rerun that group from original M2 parameters", "group_checkpoints": 96},
        "output": {"root": str(OUT), "model_dir": "R1/checkpoint/final (generic inherited v46 runtime folder label; represents only v51 M2+v49 model)", "results_are_isolated": True},
        "evaluation": {"evaluator": "unchanged v46 runtime.evaluate", "max_length": 2048, "stride": 512, "token_weighted_valid_labels": True, "P3_read": False, "reuse_rule": "reuse old R2 PPL only if every named dense model tensor and inference config equal v45 original R2; otherwise one WT2 test and one fixed-budget C4 validation evaluation"},
        "budget": {"hard_gpu_seconds": 10800, "finish_reserve_seconds": reserve, "setup_reserve_seconds": setup, "projection_only_not_measured": True, "projection_method": f"24 blocks * 1.5 * v49 maximum successful block wall ({v49_max_block:.6f}s) + 600s final save/reload/eval reserve + 180s startup/setup", "projected_seconds": projected, "measured_v49_successful_max_block_seconds": v49_max_block, "v49_measured_block_rows": len(measured_blocks), "m2_build_cost_historical_outside_this_job_seconds": m2_build_seconds},
        "freeze_epoch": time.time()
    }
    dump("PROTOCOL.json", protocol)
    (OUT / "M2_MODULE_GRID_AUDIT.csv").write_text("", encoding="utf-8")
    with (OUT / "M2_MODULE_GRID_AUDIT.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(module_rows[0]))
        w.writeheader(); w.writerows(module_rows)
    dump("SOURCE_AUDIT.json", {
        "pass": True, "cpu_seconds": time.perf_counter() - start,
        "m2_model_sha256": sha(m2path), "m2_config_sha256": sha(m2config), "pristine_fp_weight_files": [{"path": str(p), "sha256": sha(p)} for p in fp_files],
        "calibration_manifest_sha256": sha(calpath), "ordered_token_ids_sha256": token_sha,
        "v42_source_audit_sha256": sha(V42 / "SOURCE_AUDIT.json"), "v45_source_audit_sha256": sha(V45 / "SOURCE_AUDIT.json"),
        "v46_ppl_source_audit_sha256": sha(V46 / "PPL_SOURCE_AUDIT.json"), "v49_checkpoint_lock_sha256": sha(V49 / "CHECKPOINT_LOCK.json"),
        "source_files": source_hashes, "m2_original_code_files": staged_layers, "modules": 168, "output_rows": 304128,
        "unique_exact_m2_grid_mappings": True, "m2_pack_reload_grid_matches_dense_weights_bitwise": True,
        "m2_formal_ppl_sources": ppl_audit["historical_source_records"], "v49_r1_ppl_sources": v49_ppl,
        "original_m2_build_seconds": m2_build_seconds, "v49_repair_cost_reference": {"formal_repair_wall_seconds": 4543.0753269, "source": str(V49 / "REPORT.md"), "not_attributed_as_initialization_only": True},
        "gpu_used": False, "historical_results_read_only": True
    })
    dump("BUDGET_PRECHECK.json", {"gpu_job_cap_seconds": 10800, "projection_seconds": projected, "projection_not_measurement": True, "v49_max_successful_block_seconds": v49_max_block, "block_multiplier": safety, "finish_reserve_seconds": reserve, "setup_reserve_seconds": setup, "m2_historical_build_seconds_not_in_new_job": m2_build_seconds, "go": True})
    dump("PREPARATION.json", {"status": "CPU_SOURCE_AUDIT_PASS", "modules": 168, "rows": 304128, "m2_code_grid_unique_and_bitwise": True, "output_tree_precreated_and_writable": True, "gpu_started": False, "completed_seconds": time.perf_counter() - start})
    print("V51_CPU_PREFLIGHT_PASS", round(time.perf_counter() - start, 3), "seconds", "projection", round(projected, 3), "seconds", flush=True)

if __name__ == "__main__":
    main()
