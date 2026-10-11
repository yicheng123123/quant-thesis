"""Freeze the v24 E2 protocol and reuse the v22 deterministic document split."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[3]
RESULTS = ROOT / "experiments/range_search_w4/results"
V22 = RESULTS / "range_w4_research_gap_v22"
E1 = RESULTS / "range_w4_block_objective_e1_v23"
V18 = RESULTS / "range_w4_fullmodel_v18"
MODEL = ROOT / ".hf_cache/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775"


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def stable_write(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text(encoding="utf-8") != payload:
            raise RuntimeError(f"Refusing to overwrite a different frozen artifact: {path}")
        return
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(payload, encoding="utf-8", newline="\n")
    tmp.replace(path)


def write_json(path: Path, payload) -> None:
    stable_write(path, json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def main() -> None:
    required = [
        V22 / "PARAMETERIZATION_ASSESSMENT.md",
        V22 / "MINIMAL_DISCRIMINATING_EXPERIMENTS.md",
        V22 / "M3_RANGE_DIAGNOSTIC.md",
        V22 / "BACKUP_DIRECTION.md",
        V22 / "REPORT.md",
        E1 / "E1_DOCUMENT_MANIFEST.json",
        E1 / "PROTOCOL.json",
        E1 / "cache/PREFIX_CACHE_MANIFEST.json",
        V18 / "M2/checkpoint/final/model.safetensors",
        V18 / "PROTOCOL.json",
        RESULTS / "range_w4_fullmodel_baseline_v16/WIKITEXT_TEST_MANIFEST.json",
        RESULTS / "range_w4_fullmodel_baseline_v16/C4_EVAL_MANIFEST.json",
        MODEL / "config.json",
        MODEL / "model.safetensors",
    ]
    missing = [str(p) for p in required if not p.is_file()]
    if missing:
        raise FileNotFoundError("Required frozen source is missing: " + ", ".join(missing))

    e1_path = E1 / "E1_DOCUMENT_MANIFEST.json"
    e1 = json.loads(e1_path.read_text(encoding="utf-8"))
    samples = e1["samples"]
    if len(samples) != 64 or e1.get("selection_domain") != "v22-next-diagnostic-20261022:":
        raise RuntimeError("v23 E1 documents do not match the v22 frozen selection domain/count")
    if e1.get("selection_rule_source") != "v22 MINIMAL_DISCRIMINATING_EXPERIMENTS.md":
        raise RuntimeError("E1 document manifest does not cite the frozen v22 selection rule")
    if e1.get("split_rule", {}).get("semantics") != "whole documents; no document or token fragments shared between search and held":
        raise RuntimeError("E1 split semantics do not match v22")
    expected_splits = (["S0_search"] * 16 + ["S0_held"] * 16 +
                       ["S1_search"] * 16 + ["S1_held"] * 16)
    if [s.get("split") for s in samples] != expected_splits:
        raise RuntimeError("E1 document order/splits differ from frozen v22 positions")
    article_hashes = [s["article_sha256"] for s in samples]
    token_hashes = [s["token_sha256"] for s in samples]
    if len(set(article_hashes)) != 64 or len(set(token_hashes)) != 64:
        raise RuntimeError("Frozen document set contains repeated articles or token sequences")
    for sample in samples:
        if len(sample["token_ids"]) != 2048:
            raise RuntimeError("Frozen article sample has the wrong token length")
        token_hash = hashlib.sha256(json.dumps(sample["token_ids"], separators=(",", ":")).encode()).hexdigest()
        # E1 stored the same compact JSON list encoding; validate separately below using its own hash field.
        if sample.get("token_sha256") != token_hash:
            # Older Python JSON spacing is part of the E1 manifest convention.
            token_hash = hashlib.sha256(json.dumps(sample["token_ids"]).encode("utf-8")).hexdigest()
            if sample.get("token_sha256") != token_hash:
                raise RuntimeError("Frozen token sequence hash mismatch")
    ordered_ids_hash = hashlib.sha256(
        b"".join(int(t).to_bytes(8, "little", signed=True) for s in samples for t in s["token_ids"])
    ).hexdigest()
    if ordered_ids_hash != e1.get("ordered_token_ids_sha256_i64le"):
        raise RuntimeError("Frozen ordered token id hash mismatch")

    # The E1 manifest is the materialized output of the exact v22 deterministic
    # document-selection rule. E2 reuses these same complete documents/splits.
    e2 = dict(e1)
    e2["manifest_id"] = "range_w4_structured_group_e2_v24_documents"
    e2["derived_from_manifest"] = {
        "path": str(e1_path.relative_to(ROOT).as_posix()),
        "sha256": sha_file(e1_path),
        "reuse_semantics": "same 64 complete articles, token sequences, ordering and S0/S1 search/held split from the v22 frozen selection rule",
    }
    e2["selection_rule_source"] = "v22 MINIMAL_DISCRIMINATING_EXPERIMENTS.md; materialized in E1 v23 and reused without resampling"
    e2["held_use"] = "held samples are scored only after search selections are locked; no held labels enter region, grid or alpha selection"
    write_json(OUT / "E2_DOCUMENT_MANIFEST.json", e2)

    source_files = [
        V22 / "PARAMETERIZATION_ASSESSMENT.md",
        V22 / "MINIMAL_DISCRIMINATING_EXPERIMENTS.md",
        V22 / "M3_RANGE_DIAGNOSTIC.md",
        V22 / "BACKUP_DIRECTION.md",
        V22 / "REPORT.md",
        e1_path,
        E1 / "PROTOCOL.json",
        E1 / "cache/PREFIX_CACHE_MANIFEST.json",
        V18 / "PROTOCOL.json",
        V18 / "M2/checkpoint/final/model.safetensors",
        RESULTS / "range_w4_fullmodel_baseline_v16/WIKITEXT_TEST_MANIFEST.json",
        RESULTS / "range_w4_fullmodel_baseline_v16/C4_EVAL_MANIFEST.json",
        MODEL / "config.json",
        MODEL / "model.safetensors",
    ]
    source_hashes = {p.relative_to(ROOT).as_posix(): sha_file(p) for p in source_files}
    protocol = {
        "name": "v24 E2 Structured 2-Region Group-Alpha Diagnostic",
        "status": "FROZEN_BEFORE_E2_SEARCH",
        "verdict": "NOT_RUN",
        "model": {
            "id": "Qwen/Qwen2.5-0.5B-Instruct",
            "revision": "7ae557604adf67be50417f59c2c2f167def9a775",
            "dtype": "FP16",
            "attention": "SDPA",
            "device": "NVIDIA GeForce RTX 4060 Laptop GPU",
            "tf32": False,
        },
        "quantization": {
            "weight_bits": 4,
            "activation_bits": 16,
            "symmetric": True,
            "group_size": 128,
            "dynamic_groups": True,
            "block_size": 128,
            "damp_percent": 0.1,
            "desc_act": False,
            "static_groups": False,
            "native_gptq_compensation": True,
            "alpha_1_is_strict_native_noop": True,
            "model_dtype": "FP16",
            "hessian_and_gptq_math": "FP32",
            "quantizer_or_compensation_modified": False,
        },
        "targets": {
            "layers": [0, 12, 23],
            "projections": ["up_proj", "down_proj"],
            "rows": {"up_proj": [0, 224, 448, 672], "down_proj": [0, 224, 448, 672]},
            "unit_count": 48,
        },
        "documents": {
            "manifest": "E2_DOCUMENT_MANIFEST.json",
            "source": "WikiText-2 train; reuse the exact 64 complete documents selected under v22's frozen SHA ordering, materialized in E1 v23",
            "tokens_per_document": 2048,
            "split_ids": {
                "S0": {"search_positions": [0, 15], "held_positions": [16, 31]},
                "S1": {"search_positions": [32, 47], "held_positions": [48, 63]},
            },
            "selection_unit": "complete article/document; no token fragments are split across search and held",
            "test_or_c4_used_for_selection": False,
        },
        "objective": {
            "name": "frozen M2 local projection output reconstruction MSE",
            "formula": "mean((X @ (Q_FP16.float()-W_FP16.float()).T)^2) per selected output row",
            "selection_data": "search documents only",
            "held_data_use": "final metrics and retrospective oracle only after search lock",
        },
        "methods": {
            "M1_native": {"selection": "alpha=1.0", "candidate_source": "shared full-Linear candidate bank"},
            "M2_original_scalar": {"alpha_grid": [i / 10 for i in range(1, 11)], "selection": "per-row local search MSE"},
            "dense_scalar_control": {
                "definition": "original 10-point grid plus alpha=0.1+0.9*j/37 for j=1..36",
                "additional_alpha_count": 36,
                "total_unique_alpha_count": 46,
                "selection": "per-row local search MSE",
            },
            "structured_2region": {
                "grid": [i / 10 for i in range(1, 11)],
                "initialization": "M2 original scalar selected alpha copied to both regions",
                "sweeps": 2,
                "order": ["region_A", "region_B"],
                "per_region_alternatives_per_sweep": 9,
                "acceptance": "strict search-loss improvement only; ties retain current pair; among tied improving alternatives choose larger alpha",
                "each_candidate": "rerun GPTQ from pristine selected row weight and immutable split-specific H/Hinv; native compensation runs through every subsequent group",
            },
        },
        "regions": {
            "assignment_source": "v22 E2 frozen group-index partition",
            "input_dim_896_group_count_7": {"A_group_indices": [0, 1, 2], "B_group_indices": [3, 4, 5, 6]},
            "input_dim_4864_group_count_38": {"A_group_indices": list(range(19)), "B_group_indices": list(range(19, 38))},
            "threshold_or_statistic_search": False,
            "descriptive_statistics_only": ["pristine group max_abs/range", "native alpha=1 group reconstruction error", "search H diagonal mean"],
        },
        "scoring_and_budget": {
            "score_positions_per_selected_row": {"M1": 1, "M2_original": 10, "dense_scalar": 46, "structured_2region_max": 46},
            "full_linear_candidate_calls": {"M2_original_base": 120, "dense_scalar_additional": 432, "dense_scalar_total_unique_bank": 552},
            "structured_additional_per_row_gptq_max": 1728,
            "row_split_units": 48,
            "count_repeated_scheduled_structured_tuples": True,
            "deduplicate_exact_tuple_state_for_cache": True,
            "score_evaluations_and_gptq_executions_are_distinct": True,
        },
        "primary_comparisons": [
            "structured_2region vs M2_original_scalar",
            "structured_2region vs dense_scalar_control",
        ],
        "promotion_gate": "v22 E2: improve pooled held row SSE beyond frozen epsilon vs both scalar baselines in both splits; at least 4/6 layer-projection targets positive; no new < -10% native-alpha1 failure case",
        "prohibitions": ["block objective", "PPL", "C4 evaluation", "AWQ scaling", "more than two regions", "adaptive/threshold search", "second model", "full-model method"],
        "frozen_source_sha256": source_hashes,
        "source_manifest_reuse": "v23 E1 document/prefix artifacts are read-only inputs; v8-v23 artifacts are not modified",
    }
    write_json(OUT / "PROTOCOL.json", protocol)

    region_doc = """# E2 Region Assignment Protocol\n\n**Frozen before E2 candidate scoring.** Use the exact v22 E2 group-index partition. v22 froze a concrete contiguous assignment, so the v24 fallback to select a statistic/threshold is not invoked. No M3 quality result, search loss, held loss, activation label, or threshold is used to assign groups.\n\n- For input width 896, G128 creates 7 groups: Region A = groups 0–2; Region B = groups 3–6.\n- For input width 4864, G128 creates 38 groups: Region A = groups 0–18; Region B = groups 19–37.\n- The assignment is identical for every output row at the same target shape. Each region receives its own shared alpha for the row.\n- The group order is native input-column order under `desc_act=False` and `static_groups=False`. A candidate changes the alpha hook at native dynamic group entry and reruns GPTQ compensation from the pristine row.\n\nPristine group max-absolute weight/range, alpha=1 group reconstruction error, and search-H diagonal summaries may be reported descriptively after the assignment is locked. They do not change A/B membership. Held data is excluded from region definition and all search decisions.\n"""
    stable_write(OUT / "REGION_ASSIGNMENT_PROTOCOL.md", region_doc)

    budget = {
        "status": "PLANNED_PRE_PREFLIGHT",
        "units": 48,
        "grid": [i / 10 for i in range(1, 11)],
        "dense_grid_additional": [0.1 + 0.9 * j / 37 for j in range(1, 37)],
        "dense_grid_unique_total": 46,
        "search_score_positions": {
            "M1_native_alpha1": 48,
            "M2_original_scalar_10": 480,
            "dense_scalar_46": 2208,
            "structured_2region_max_46": 2208,
            "structured_incremental_beyond_shared_10": 1728,
        },
        "candidate_gptq_executions": {
            "full_linear_shared_alpha_base_10": 120,
            "full_linear_dense_additional_36": 432,
            "full_linear_unique_total_for_scalar_bank": 552,
            "per_row_structured_additional_max": 1728,
            "full_model_gptq_executions": 0,
        },
        "held_scoring_plan": {
            "selected_method_candidates": "M1/M2/dense/structured search selections only",
            "retrospective_oracle": "all unique dense scalar alpha candidates and all unique structured tuples actually evaluated; no held values are read before selection lock",
            # The ten diagonal structured tuples reuse the dense scalar bank;
            # only the at-most 36 off-diagonal tuples add held score positions.
            "held_score_positions_upper_bound": 48 * (46 + 36),
        },
        "structured_scheduled_slots": 48 * 36,
        "structured_unique_tuple_executions_after_cache": "at most 1728; actual count recorded from cache and event log",
        "notes": [
            "A score position is one selected row by one candidate loss. It is not a GPTQ call.",
            "Full-Linear candidate runs produce outputs for all rows, but only the four preregistered rows are scored/stored.",
            "Structured updates are row-specific GPTQ executions with fresh compensation state; no full-Linear shared-candidate shortcut is used for unequal alpha_A/alpha_B.",
            "Exact repeated structured tuples are cacheable; scheduled and unique tuples are both recorded.",
        ],
    }
    budget_path = OUT / "E2_EXECUTION_BUDGET.json"
    if budget_path.exists():
        existing_budget = json.loads(budget_path.read_text(encoding="utf-8"))
        mismatches = {key: {"expected": value, "actual": existing_budget.get(key)}
                      for key, value in budget.items() if key != "status" and existing_budget.get(key) != value}
        if mismatches:
            raise RuntimeError(f"Existing E2 budget differs from frozen plan: {mismatches}")
    else:
        write_json(budget_path, budget)
    print(json.dumps({"status": "FROZEN", "e2_manifest_sha256": sha_file(OUT / "E2_DOCUMENT_MANIFEST.json"),
                      "protocol_sha256": sha_file(OUT / "PROTOCOL.json"),
                      "source_e1_manifest_sha256": sha_file(e1_path), "samples": len(samples)}, indent=2))


if __name__ == "__main__":
    main()
