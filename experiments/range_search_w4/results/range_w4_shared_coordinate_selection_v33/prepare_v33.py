"""Freeze v33's CPU-only shared-coordinate selection protocol from v32/v24."""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path


HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent
V32 = RESULTS / "range_w4_fixed_channel_scale_scalar_v32"
V24 = RESULTS / "range_w4_structured_group_e2_v24"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    started = time.perf_counter()
    if (HERE / "FROZEN_PROTOCOL.json").exists():
        raise RuntimeError("refusing to replace an existing v33 frozen protocol")
    p32 = json.loads((V32 / "FROZEN_PROTOCOL.json").read_text("utf-8"))
    correct = json.loads((V32 / "CORRECTNESS.json").read_text("utf-8"))
    source = json.loads((V32 / "SOURCE_AUDIT.json").read_text("utf-8"))
    v24 = json.loads((V24 / "PROTOCOL.json").read_text("utf-8"))
    lock24 = json.loads((V24 / "SEARCH_SELECTION_LOCK.json").read_text("utf-8"))
    if correct.get("status") != "PASS" or source.get("status") != "PASS":
        raise RuntimeError("v32 source/correctness audit is not PASS")
    if p32.get("quantization", {}).get("alpha_1_is_strict_native_noop") is not True:
        raise RuntimeError("unexpected v32 quantization semantics")
    if p32.get("alpha_grid") != v24.get("methods", {}).get("M2_original_scalar", {}).get("alpha_grid"):
        raise RuntimeError("v32/v24 scalar10 grids differ")
    if lock24.get("held_indices_not_read_for_selection") is not True:
        raise RuntimeError("v24 source selection lock does not establish held separation")

    source_files = {
        "v32_protocol": V32 / "FROZEN_PROTOCOL.json",
        "v32_correctness": V32 / "CORRECTNESS.json",
        "v32_source_audit": V32 / "SOURCE_AUDIT.json",
        "v32_scaled_search_scores": V32 / "SCALED_CANDIDATE_SEARCH_SCORES.csv",
        "v24_protocol": V24 / "PROTOCOL.json",
        "v24_search_scores": V24 / "SEARCH_RESULTS.csv",
        "v24_search_lock": V24 / "SEARCH_SELECTION_LOCK.json",
    }
    missing = [name for name, path in source_files.items() if not path.is_file()]
    if missing:
        raise RuntimeError(f"missing frozen source files: {missing}")

    protocol = {
        "name": "v33 Shared Coordinate Selection Diagnostic",
        "status": "FROZEN_BEFORE_SELECTION",
        "interpretation": "Retrospective exploratory diagnostic on previously viewed articles; no new quality validation",
        "compute_scope": "CPU artifact analysis only; no CUDA calls, model forwards, GPTQ, PPL, or whole-model quantization",
        "cpu_wall_budget_seconds": 300,
        "inherits": {
            "model": p32["model"],
            "quantization": p32["quantization"],
            "alpha_grid": p32["alpha_grid"],
            "targets": p32["targets"],
            "split_documents": p32["split_documents"],
            "article_unit": p32["article_unit"],
            "prefix": p32["prefix"],
            "scoring": p32["scoring"],
        },
        "selection_unit": "one target x split, with four fixed output rows sharing one coordinate choice D",
        "coordinate_choices": {
            "identity": "D=I, original-coordinate v24/v32 scalar10 candidate bank",
            "fixed": "the already frozen v32 target x split activation-derived D; no new D or strength search",
            "rowwise_D_selection": False,
        },
        "selection_rule": {
            "per_coordinate_per_row": "argmin over the original 10 alpha search MSE values; exact ties choose larger alpha",
            "coordinate_score": "sum over the four fixed rows of per-row minimum search MSE * 32768 tokens",
            "coordinate_choice": "choose the smaller coordinate score; exact equality selects D=I with zero tolerance",
            "data_access": "selection process consumes search_mse values only; no held score values are read before SEARCH_SELECTION_LOCK.json is written",
            "expected_identity_inclusion": "shared minimum search score must be <= G01 sum because D=I arm is in the candidate set",
        },
        "held_analysis": {
            "timing": "separate command after frozen search choices and lock are written",
            "baseline": "v32 G01 held result for each row",
            "selected_fixed": "v32 G11 held result at the already selected fixed D row alpha",
            "selected_identity": "v32 G01 held result at the already selected identity-coordinate row alpha",
            "epsilon": "v32/v24 frozen per-row repeat-noise epsilon; no new tolerance",
            "severe_regression_threshold": "strictly worse than -10% relative to G01, regardless of WTL epsilon",
            "aggregation": "four-row SSE within each target x split; row-wise diagnostics remain paired units",
        },
        "source_files_sha256": {name: sha(path) for name, path in source_files.items()},
        "held_score_sources_excluded_until_selection_lock": [
            "v32 raw/*_HELD_RESULTS.json",
            "v24 HELD_RESULTS.csv",
            "v32 FOUR_ARM_ROW_RESULTS.csv",
        ],
        "prohibitions": [
            "new D computation", "scaling strength search", "GPU", "model forward", "GPTQ",
            "PPL", "whole-model quantization", "held-based selection or rule changes",
            "row-specific D", "third coordinate system", "held oracle selection",
        ],
    }
    (HERE / "FROZEN_PROTOCOL.json").write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + "\n", "utf-8")
    (HERE / "BUDGET.json").write_text(json.dumps({
        "budget_seconds": protocol["cpu_wall_budget_seconds"],
        "stages": {"prepare_freeze": time.perf_counter() - started},
        "cumulative_wall_seconds": time.perf_counter() - started,
        "gpu_calls": 0,
        "model_forwards": 0,
        "gptq_executions": 0,
        "ppl_runs": 0,
    }, ensure_ascii=False, indent=2) + "\n", "utf-8")
    print(json.dumps({"status": protocol["status"], "source_files": len(source_files),
                      "cpu_wall_budget_seconds": protocol["cpu_wall_budget_seconds"]}, indent=2))


if __name__ == "__main__":
    main()
