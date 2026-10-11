"""CPU-only source audit and protocol freeze for the v32 local 2x2 diagnostic."""
from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
RESULTS = ROOT / "experiments/range_search_w4/results"
V24 = RESULTS / "range_w4_structured_group_e2_v24"
V31 = RESULTS / "range_w4_generation_selection_split_v31"
MODEL = ROOT / ".hf_cache/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775/model.safetensors"


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def dump(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", "utf-8")


def choose(values: dict[float, float]) -> float:
    lowest = min(values.values())
    return max(a for a, value in values.items() if value == lowest)


def main() -> None:
    v24 = json.loads((V24 / "PROTOCOL.json").read_text("utf-8"))
    v31 = json.loads((V31 / "FROZEN_PROTOCOL.json").read_text("utf-8"))
    v31_audit = json.loads((V31 / "DATA_RELATION_AUDIT.json").read_text("utf-8"))
    v31_checks = json.loads((V31 / "ANALYSIS_CHECKS.json").read_text("utf-8"))
    v31_sources = json.loads((V31 / "SOURCE_HASHES.json").read_text("utf-8"))
    if v31_audit["status"] != "PASS" or v31_checks["status"] != "PASS":
        raise RuntimeError("v31 baseline source audit is not PASS")
    if sha_file(V24 / "PROTOCOL.json") != v31_sources["v24_protocol"]:
        raise RuntimeError("v24 protocol changed since v31 audit")
    if sha_file(MODEL) != v31_sources["base_model_weights"]:
        raise RuntimeError("base checkpoint changed since v31 audit")
    if v24["model"] != v31["inheritance"]["v24_model"]:
        raise RuntimeError("model metadata mismatch between v24 and v31")
    if v24["quantization"] != v31["inheritance"]["v24_quantization"]:
        raise RuntimeError("quantizer metadata mismatch between v24 and v31")
    if v24["targets"]["rows"] != v31["inheritance"]["v24_fixed_rows"]:
        raise RuntimeError("fixed rows changed")
    if v24["methods"]["M2_original_scalar"]["alpha_grid"] != v31["inheritance"]["v24_scalar_grid"]:
        raise RuntimeError("original scalar10 grid changed")

    input_paths = {}
    input_hashes = {}
    for layer in (0, 12, 23):
        for projection in ("up_proj", "down_proj"):
            p = ((RESULTS / "range_w4_block_objective_e1_v23/cache") if projection == "up_proj" else (V24 / "cache")) / f"input_L{layer}_{projection}.f16"
            relative = p.relative_to(ROOT).as_posix()
            expected = v31_sources["activation_inputs"][relative]
            current = sha_file(p)
            if current != expected:
                raise RuntimeError(f"activation input cache changed: {p}")
            input_paths[f"L{layer}_{projection}"] = relative
            input_hashes[f"L{layer}_{projection}"] = current

    # v31 long table contains the already audited v24 original search/held scores.
    long_rows = read_csv(V31 / "CROSS_GROUP_ROW_ALPHA_SCORES.csv")
    selections = {(r["bank"], int(r["layer"]), r["projection"], int(r["row"])): float(r["original_alpha"])
                  for r in read_csv(V31 / "SELECTIONS_BY_ROW.csv")}
    old: dict[tuple, dict[str, float]] = {}
    for r in long_rows:
        bank = r["bank"]
        group = r["evaluation_group"]
        if group not in (("A", "B") if bank == "S0" else ("C", "D")):
            continue
        if r["score_source"] != "v30_audited_v24_saved_score":
            raise RuntimeError("original source score is not the audited v24/v30 score")
        key = (bank, int(r["layer"]), r["projection"], int(r["row"]), float(r["alpha"]))
        old.setdefault(key, {})[group] = float(r["score_mse"])
    if len(old) != 480 or any(len(v) != 2 for v in old.values()):
        raise RuntimeError("incomplete original scalar10 source bank")
    if len(selections) != 48:
        raise RuntimeError("expected 48 original search selections")
    baseline = []
    for key, scores in sorted(old.items()):
        bank, layer, projection, row, alpha = key
        original_selector = "A" if bank == "S0" else "C"
        held_group = "B" if bank == "S0" else "D"
        alphas = {a: old[(bank, layer, projection, row, a)][original_selector]
                  for a in v24["methods"]["M2_original_scalar"]["alpha_grid"]}
        selected = choose(alphas)
        if selected != selections[(bank, layer, projection, row)]:
            raise RuntimeError(f"v24 original scalar selection not reproduced: {key}")
        baseline.append({"bank": bank, "layer": layer, "projection": projection, "row": row,
                         "alpha": alpha, "search_group": original_selector, "held_group": held_group,
                         "search_mse": scores[original_selector], "held_mse": scores[held_group],
                         "original_selected_alpha": selected})
    with (HERE / "ORIGINAL_BANK_REUSED.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(baseline[0]))
        writer.writeheader(); writer.writerows(baseline)

    split_docs = {"S0": {"search": list(range(0, 16)), "held": list(range(16, 32))},
                  "S1": {"search": list(range(32, 48)), "held": list(range(48, 64))}}
    source_hashes = {
        "v24_protocol": sha_file(V24 / "PROTOCOL.json"),
        "v24_article_manifest": sha_file(V24 / "E2_DOCUMENT_MANIFEST.json"),
        "v24_held_results": sha_file(V24 / "HELD_RESULTS.csv"),
        "v24_held_repeat_noise": sha_file(V24 / "HELD_REPEAT_NOISE.json"),
        "v31_frozen_protocol": sha_file(V31 / "FROZEN_PROTOCOL.json"),
        "v31_data_relation_audit": sha_file(V31 / "DATA_RELATION_AUDIT.json"),
        "v31_original_choice_checks": sha_file(V31 / "ANALYSIS_CHECKS.json"),
        "v31_original_and_cross_score_table": sha_file(V31 / "CROSS_GROUP_ROW_ALPHA_SCORES.csv"),
        "v31_original_selections": sha_file(V31 / "SELECTIONS_BY_ROW.csv"),
        "base_model_weights": sha_file(MODEL),
        "activation_inputs": input_hashes,
        "original_bank_reused": sha_file(HERE / "ORIGINAL_BANK_REUSED.csv"),
    }
    protocol = {
        "name": "v32 Fixed Channel Scaling x Scalar Range Search Local 2x2 Diagnostic",
        "status": "FROZEN_BEFORE_GPU",
        "interpretation": "Exploratory screening on previously viewed articles and fixed v24 rows; no blind generalization claim",
        "model": v24["model"], "quantization": v24["quantization"], "targets": v24["targets"],
        "split_documents": split_docs,
        "article_unit": "16 complete 2048-token WT2 train articles per search or held group; no token splitting",
        "prefix": "fixed v18 M2 quantized prefix, identical v23/v24 cached inputs for all four arms",
        "original_weight": "base model checkpoint tensor converted to FP16 exactly as v24; verify pristine SHA against v24 sidecar",
        "input_paths": input_paths,
        "input_sha256": input_hashes,
        "arms": {
            "G00": "original coordinates, native dynamic GPTQ alpha=1; saved v24 score/candidate reused",
            "G01": "original coordinates, v24 scalar10 search-selected alpha; saved v24 score/candidate reused",
            "G10": "one frozen activation-derived D, transformed-coordinate dynamic GPTQ alpha=1",
            "G11": "same frozen D, transformed-coordinate dynamic GPTQ scalar10 search-selected alpha",
        },
        "alpha_grid": v24["methods"]["M2_original_scalar"]["alpha_grid"],
        "scale_definition": {
            "source": "only the split's 16 search articles and same frozen input cache",
            "a_j": "mean(abs(X[:,j])) over 32768 search tokens; FP16 cache converted to FP32; FP64 reduction",
            "d_unscaled_j": "sqrt(max(a_j, 1e-8)) in FP64",
            "normalization": "divide by sqrt(min(d_unscaled)*max(d_unscaled)) in FP64",
            "stored_D_dtype": "FP32",
            "Wprime": "(W_FP16.float()*D_FP32).to(FP16), contiguous, full Linear",
            "Xprime": "(X_FP16.float()/D_FP32).to(FP16), each complete document",
            "alpha_dependent_D": False,
            "held_in_D_or_H_or_selection": False,
        },
        "hessian": {
            "source": "fresh native GPTQ.process_batch on Xprime FP16 search documents, in exact document order",
            "initialization": "FP32 zero square matrix and nsamples=0",
            "preparation": "same v17.3 prepare_native_state: dead-column, damp=0.1*mean(diag), lower Cholesky, inverse, native final Cholesky",
            "unscaled_H_reuse_in_scaled_branch": False,
            "saved_dtype": "FP32",
        },
        "scoring": {
            "target": "same original unquantized reference X_FP16.float()@W_FP16.float().T; bias cancels",
            "scaled_error": "Xprime_FP16.float()@(Qprime_FP16.float()-Wprime_FP16.float()).T + [Xprime_FP16.float()@Wprime_FP16.float().T - X_FP16.float()@W_FP16.float().T]",
            "input_chunk": "2 complete documents = 4096 tokens; 4 fixed output rows; FP32 GEMM and FP64 squared-sum accumulation",
            "normalization": "divide by 16*2048 tokens for each row",
            "selection": "argmin on search MSE over the original 10 alphas; exact ties choose larger alpha",
            "held_use": "score only G10 alpha1 and G11 search-selected row candidates after selection lock",
            "baseline": "reused v24/v31 original scalar10 search and held scores with exact source check",
            "Qeff": "optional diagnostic Qprime*D^-1 in FP32 only; not a packable original G128 weight",
        },
        "preflight": {
            "D_identity_targets": ["L0_up_proj_S0", "L0_down_proj_S0"],
            "D_identity_requirements": "rebuilt native H exact hash and alpha1 selected row Q/scales/zeros/g_idx exact vs v24",
            "unquantized_equivalence": "analytic identity, FP64 and FP32 spot on first 128 search tokens, FP16 conversion drift over full search and held scoring groups",
            "scaled_native_alpha1_parity": "for every target/split, compare optimized alpha1 to independent native_dynamic_gptq at identical Wprime/Hprime",
            "candidate_independence": "immutable pristine Wprime and prepared H; verify unchanged hashes after candidates and reversed-order spot",
        },
        "budget": {"new_gpu_wall_seconds_max": 1200, "count_cuda_init_warmup_parity_hessian_factor_candidates_scoring_io": True,
                   "expected_new_scaled_full_linear_candidates": 120,
                   "expected_additional_scaled_native_alpha1_parity_calls": 12,
                   "expected_D_identity_alpha1_calls": 2},
        "metrics": {
            "interaction": "I=(L10-L11)-(L00-L01) on identical held documents and fixed rows; lower loss is better",
            "primary": ["G10-vs-G00", "G11-vs-G10", "G11-vs-G01"],
            "wtl_epsilon": "v24 HELD_REPEAT_NOISE.json per unit; frozen prior to v32",
            "pooled_SSE": "sum over fixed rows of per-row MSE*32768; never called whole-Linear output SSE",
        },
        "prohibitions": ["gradient or learning", "AWQ implementation claim", "scaling-strength search", "46-point grid",
                         "whole-model quantization", "PPL", "held-based D tuning", "public quantizer changes", "checkpoint overwrite"],
        "source_sha256": source_hashes,
    }
    dump(HERE / "SOURCE_AUDIT.json", {
        "status": "PASS", "checks": [
            "v24/v31 model, quantizer, rows, original grid agree",
            "all six current input cache hashes equal v31 audited hashes",
            "base checkpoint hash equals v31 audited hash",
            "480 original scalar10 search/held candidate row scores available",
            "all 48 original alpha selections reproduced exactly",
            "v31 data relation audit and original choice checks PASS",
        ],
        "source_sha256": source_hashes,
        "new_GPU_work_started": False,
    })
    dump(HERE / "FROZEN_PROTOCOL.json", protocol)
    (HERE / "raw/scales").mkdir(parents=True, exist_ok=True)
    (HERE / "raw/hessians").mkdir(parents=True, exist_ok=True)
    (HERE / "raw/candidates").mkdir(parents=True, exist_ok=True)
    print(json.dumps({"status": "FROZEN_BEFORE_GPU", "baseline_rows": len(baseline),
                      "protocol_sha256": sha_file(HERE / "FROZEN_PROTOCOL.json")}, indent=2))


if __name__ == "__main__":
    main()
