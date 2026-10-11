"""CPU-only post-stop analysis for the partial S0 held artifacts and S1 failure."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from transformers.models.qwen2.modeling_qwen2 import Qwen2RMSNorm

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
RAW = HERE / "raw"
CACHE = HERE / "cache"
CORRECT = HERE / "correctness"
MODEL = ROOT / ".hf_cache/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775"
PREFIX = ROOT / "experiments/range_search_w4/results/range_w4_fullmodel_v18/M2/checkpoint/final/model.safetensors"


def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def paired(base: np.ndarray, cand: np.ndarray) -> dict:
    change = (cand.astype(np.float64) / base.astype(np.float64) - 1.0) * 100.0
    improvement = -change
    return {
        "baseline_mse_mean": float(np.mean(base)),
        "candidate_mse_mean": float(np.mean(cand)),
        "relative_loss_change_pct_candidate_over_baseline": float(np.mean(cand) / np.mean(base) * 100.0 - 100.0),
        "relative_improvement_pct": float(100.0 - np.mean(cand) / np.mean(base) * 100.0),
        "paired_W_T_L_lower_is_better_exact_ties": {
            "wins": int(np.sum(cand < base)), "ties": int(np.sum(cand == base)), "losses": int(np.sum(cand > base))
        },
        "paired_median_improvement_pct": float(np.median(improvement)),
        "worst_row_degradation_pct": float(np.max(change)),
        "rows_degraded_gt_10pct": int(np.sum(change > 10.0)),
        "rows_degraded_gt_50pct": int(np.sum(change > 50.0)),
    }


def main() -> None:
    result_path = RAW / "S0_held_result.json"
    npz_path = RAW / "S0_held_mlp_outputs.npz"
    if not result_path.is_file() or not npz_path.is_file():
        raise FileNotFoundError("partial S0 held outputs are missing")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    archive = np.load(npz_path, allow_pickle=False)
    reference = archive["reference"].astype(np.float32)
    arms = {name: archive[name].astype(np.float32) for name in ("G00", "G01", "G10", "G11")}
    row_mse = {}
    article_mse = {}
    for arm, output in arms.items():
        err = output - reference
        err64 = err.astype(np.float64)
        row_mse[arm] = (np.square(err64).sum(axis=(0, 1)) / (output.shape[0] * output.shape[1])).astype(np.float64)
        article_mse[arm] = np.square(err64).mean(axis=(1, 2))
        stored = float(result["arm_full_mlp_mse"][arm])
        recomputed = float(np.square(err64).mean())
        if abs(stored - recomputed) > max(1e-10, abs(stored) * 1e-7):
            raise RuntimeError(f"S0 held NPZ/result mismatch {arm}: {stored} vs {recomputed}")

    row_path = RAW / "S0_FULL_MLP_ROW_HELD_PARTIAL.csv"
    with row_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["split", "arm", "projection", "row", "mse"])
        for arm in arms:
            for row, loss in enumerate(row_mse[arm]):
                writer.writerow(["S0", arm, "full_mlp_output", row, f"{loss:.17g}"])
    row_contrast_path = RAW / "S0_FULL_MLP_ROW_CONTRASTS_PARTIAL.csv"
    with row_contrast_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["split", "contrast", "row", "baseline_arm", "candidate_arm", "baseline_mse", "candidate_mse", "relative_loss_change_pct"])
        for contrast, baseline, candidate in (("G10_vs_G00", "G00", "G10"), ("G11_vs_G10", "G10", "G11"), ("G11_vs_G01", "G01", "G11")):
            for row, (b, c) in enumerate(zip(row_mse[baseline], row_mse[candidate])):
                writer.writerow(["S0", contrast, row, baseline, candidate, f"{b:.17g}", f"{c:.17g}", f"{(c/b-1)*100:.12g}"])

    article_path = RAW / "S0_ARTICLE_HELD_CONTRASTS.csv"
    article_rows = result["article_metrics"]
    by_arm = {a: [float(r["full_mlp_mse"]) for r in article_rows if r["arm"] == a] for a in arms}
    with article_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["split", "group", "article_id", "article_sha256", "G00_mse", "G01_mse", "G10_mse", "G11_mse"])
        ref_rows = [r for r in article_rows if r["arm"] == "G00"]
        for i, ref_row in enumerate(ref_rows):
            writer.writerow(["S0", ref_row["group"], ref_row["article_id"], ref_row["article_sha256"],
                             *(by_arm[a][i] for a in ("G00", "G01", "G10", "G11"))])

    contrasts = {
        "G10_vs_G00_scaling_transfer": paired(row_mse["G00"], row_mse["G10"]),
        "G11_vs_G10_scalar_search_after_scaling": paired(row_mse["G10"], row_mse["G11"]),
        "G11_vs_G01_combination_vs_existing_scalar10": paired(row_mse["G01"], row_mse["G11"]),
    }
    article_contrasts = {
        "G10_vs_G00_scaling_transfer": paired(np.asarray(by_arm["G00"]), np.asarray(by_arm["G10"])),
        "G11_vs_G10_scalar_search_after_scaling": paired(np.asarray(by_arm["G10"]), np.asarray(by_arm["G11"])),
        "G11_vs_G01_combination_vs_existing_scalar10": paired(np.asarray(by_arm["G01"]), np.asarray(by_arm["G11"])),
    }
    article_concentration = {}
    for name, (base_arm, cand_arm) in {
        "G10_vs_G00": ("G00", "G10"), "G11_vs_G10": ("G10", "G11"), "G11_vs_G01": ("G01", "G11")
    }.items():
        gain = np.asarray(by_arm[base_arm]) - np.asarray(by_arm[cand_arm])
        net = float(gain.sum())
        idx = int(np.argmax(gain))
        article_concentration[name] = {
            "net_sse_improvement_sum_article_mse_units": net,
            "largest_positive_article_id": ref_rows[idx]["article_id"],
            "largest_article_improvement": float(gain[idx]),
            "largest_positive_contribution_fraction_of_net": (float(max(gain[idx], 0.0) / net) if net > 0 else None),
            "net_after_removing_largest_positive_article": float(net - max(gain[idx], 0.0)),
        }

    lock = json.loads((HERE / "final/S0_SELECTION_LOCK.json").read_text(encoding="utf-8"))
    alpha_hist = lock["selected_tensors"]
    selected_counts = {}
    for arm in ("G00", "G01", "G10", "G11"):
        selected_counts[arm] = {
            proj: alpha_hist[f"{arm}.{proj}"]["alpha_histogram"] for proj in ("gate", "up", "down")
        }

    # A CPU-only RMSNorm replay checks the saved P0/P2 activation pairs.  It is
    # forensic evidence, not a replacement for the failing CUDA D=I check.
    with safe_open(str(MODEL / "model.safetensors"), framework="pt", device="cpu") as f:
        base_norm = f.get_tensor("model.layers.12.post_attention_layernorm.weight").half()
    with safe_open(str(PREFIX), framework="pt", device="cpu") as f:
        prefix_norm = f.get_tensor("model.layers.12.post_attention_layernorm.weight")
    config = json.loads((MODEL / "config.json").read_text(encoding="utf-8"))
    rms = Qwen2RMSNorm(896, eps=config["rms_norm_eps"]).half().cpu()
    rms.weight.data.copy_(base_norm)
    norm_diagnostics = {
        "scope": "CPU-only replay of stored pre/post-norm capture pair; does not modify or replace frozen CUDA correctness check",
        "base_prefix_norm_weight_equal": bool(torch.equal(base_norm, prefix_norm)),
        "base_norm_sha256_fp16": sha_bytes(base_norm.contiguous().numpy().tobytes()),
        "prefix_norm_sha256_native": sha_bytes(prefix_norm.contiguous().numpy().tobytes()),
        "rms_norm_eps": config["rms_norm_eps"],
        "groups": {},
    }
    for split, group in (("S0", "P0"), ("S1", "P2")):
        shape = (8, 2048, 896)
        pre = np.memmap(CACHE / f"{split}_{group}_prenorm.f16", dtype=np.float16, mode="r", shape=shape)
        post = np.memmap(CACHE / f"{split}_{group}_postnorm.f16", dtype=np.float16, mode="r", shape=shape)
        per_doc = []
        for doc in range(8):
            calc = rms(torch.from_numpy(np.array(pre[doc], copy=True)))
            captured = torch.from_numpy(np.array(post[doc], copy=True))
            diff = (calc - captured).abs()
            per_doc.append({"doc_index_in_group": doc, "max_abs": float(diff.max()), "mean_abs": float(diff.float().mean()),
                            "exact_equal_elements": int((diff == 0).sum()), "elements": int(diff.numel()),
                            "pre_sha256": sha_bytes(np.asarray(pre[doc]).tobytes()),
                            "post_sha256": sha_bytes(np.asarray(post[doc]).tobytes())})
        norm_diagnostics["groups"][group] = per_doc
        pre._mmap.close(); post._mmap.close()

    # Execution-accounting reconciliation uses completed ledger rows plus the
    # explicit failed parity record.  Planned per-split allocations are not
    # substituted for actual split placement.
    with (RAW / "BUDGET_LEDGER.csv").open(newline="", encoding="utf-8") as f:
        ledger_rows = list(csv.DictReader(f))
    stage_seconds = {}
    for row in ledger_rows:
        stage = row["stage"]
        stage_seconds[stage] = stage_seconds.get(stage, 0.0) + float(row["wall_seconds"] or 0.0)
    max_alloc = max(int(r["peak_allocated_bytes"] or 0) for r in ledger_rows)
    max_reserved = max(int(r["peak_reserved_bytes"] or 0) for r in ledger_rows)
    attempted_counts = {
        "planned_full_Linear_GPTQ": 144,
        "planned_main_candidate_GPTQ": 124,
        "planned_alpha1_parity_GPTQ": 16,
        "planned_down_isolation_replay_GPTQ": 4,
        "actual_main_candidate_GPTQ_completed": sum(1 for r in ledger_rows if r["stage"] == "full_Linear_dynamic_GPTQ_candidate"),
        "actual_alpha1_parity_GPTQ_logged_pass": sum(1 for r in ledger_rows if r["stage"] == "alpha1_independent_native_parity"),
        "actual_alpha1_parity_GPTQ_failed_after_call": 1,
        "actual_down_isolation_replay_GPTQ": sum(1 for r in ledger_rows if r["stage"] == "down_candidate_isolation_replay"),
    }
    attempted_counts["actual_full_Linear_GPTQ_executions_including_failed_parity"] = (
        attempted_counts["actual_main_candidate_GPTQ_completed"]
        + attempted_counts["actual_alpha1_parity_GPTQ_logged_pass"]
        + attempted_counts["actual_alpha1_parity_GPTQ_failed_after_call"]
        + attempted_counts["actual_down_isolation_replay_GPTQ"]
    )
    attempt_rows = [
        [1, "protocol_id guard", "FAIL before model/GPTQ", 0, 0, 0, "preflight launch; conservative 14s charged"],
        [2, "down coordinate spot input shape", "FAIL after S0 D selector", 0, 0, 0, "shape bug; no GPTQ candidate started"],
        [3, "temporary arm construction", "FAIL after S0 gate/up banks", 40, 4, 0, "KeyError G00; banks persisted"],
        [4, "norm set_weight call", "FAIL before down G00", 0, 0, 0, "Parameter passed where module was expected"],
        [5, "down G00 alpha1 parity comparison", "FAIL after GPTQ call", 1, 1, 0, "native GPTQ ran; CPU/CUDA compare raised before parity score; extra call"],
        [6, "down parity return", "FAIL after G00 parity and bank persisted", 0, 0, 0, "NameError cmp; no repeated GPTQ on recovery"],
        [7, "row-stitch g_idx device comparison", "FAIL after all S0 candidate banks/replays", 0, 0, 2, "CPU g_idx compared with CUDA expected; no held opened"],
        [8, "S1 D=I norm-output regression", "CORRECTNESS_FAIL; STOP", 0, 0, 2, "S0 replays repeated on recovery; S0 held evaluated; S1 stopped before selector/held"],
    ]
    with (RAW / "ATTEMPT_LOG.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["launch", "failure_or_stop_point", "outcome", "new_main_GPTQ_calls", "native_parity_GPTQ_calls", "down_replay_GPTQ_calls", "detail"])
        writer.writerows(attempt_rows)
    execution_audit = {
        "planned_counts": {k: attempted_counts[k] for k in ("planned_full_Linear_GPTQ", "planned_main_candidate_GPTQ", "planned_alpha1_parity_GPTQ", "planned_down_isolation_replay_GPTQ")},
        "actual_counts": attempted_counts,
        "actual_full_GPTQ_by_split": {"S0": {"main_candidates": 62, "alpha1_parity_calls_including_failed_compare": 9, "down_replays": 4, "total": 75}, "S1": {"main_candidates": 0, "alpha1_parity_calls": 0, "down_replays": 0, "total": 0}},
        "main_row_alpha_slots_completed": {"S0": 214272, "S1": 0},
        "independent_parity_direct_score_row_slots_completed": 22144,
        "replay_score_row_slots_completed": 4 * 896,
        "failed_parity_call_note": "One S0/G00 down native alpha1 GPTQ call completed but the first cross-device comparison raised; a retry then passed. The failed call added one actual execution beyond the 144-call frozen plan.",
        "replay_allocation_note": "The frozen plan allocates two replays per split; resume caused all four replay executions to land in S0 (two repeated), with zero in S1.",
    }
    cost_audit = {
        "cumulative_gpu_wall_seconds_from_state": float(json.loads((RAW / "RUN_STATE.json").read_text(encoding="utf-8").replace("\ufeff", ""))["gpu_seconds_used"]),
        "external_device_probe_seconds_included": 5.2,
        "hard_limit_seconds": 1800,
        "stop_compute_threshold_seconds": 1620,
        "budget_ledger_recorded_stage_seconds_sum_nonexclusive_caveat": float(sum(float(r["wall_seconds"] or 0.0) for r in ledger_rows)),
        "max_peak_allocated_bytes": max_alloc,
        "max_peak_reserved_bytes": max_reserved,
        "stage_seconds": stage_seconds,
        "unseparately_timed": ["down-H construction is performed inside candidate_and_parity_down before damping-preparation timing", "some no-op correctness stamps report 0 seconds", "failed native parity GPTQ is included in RUNTIME_ERROR tail rather than a standalone timing"],
    }

    report = {
        "status": "PARTIAL_S0_HELD_ONLY_OVERALL_CORRECTNESS_FAIL",
        "held_group": "P1",
        "held_articles": 8,
        "held_token_positions": 16384,
        "selected_D_ratio": lock["D_selection"]["selected_ratio"],
        "selected_D_sha256": lock["D_selection"]["selected_D_sha256"],
        "selected_alpha_histograms": selected_counts,
        "full_mlp_mse_recomputed_from_saved_output_npz": {arm: float(row_mse[arm].mean()) for arm in arms},
        "full_mlp_mse_stored_by_runner": result["arm_full_mlp_mse"],
        "contrasts_full_mlp_output_rows": contrasts,
        "contrasts_per_article_full_mlp": article_contrasts,
        "article_gain_concentration": article_concentration,
        "v34_primary_minimum_improvement_gate_S0_only": {
            "G11_vs_G01_relative_improvement_pct": contrasts["G11_vs_G01_combination_vs_existing_scalar10"]["relative_improvement_pct"],
            "passes_1pct_in_S0": contrasts["G11_vs_G01_combination_vs_existing_scalar10"]["relative_improvement_pct"] >= 1.0,
            "overall_gate_status": "NOT_PASS; S0 partial contrast below threshold; S1 not evaluated",
        },
        "v34_fold_error_gate": {
            "stored_S0_fold_error_mse": result["held"]["fold_error_mse"],
            "valid": False,
            "reason": "Static code audit: selected_fold_correctness/eval_held divide norm weight by D but leave pristine gate/up weights unscaled; this is not the frozen W'=WD, X'=XD^-1 FP16 fold. No replacement forward was run after stop.",
        },
        "projection_tail_gate": {
            "status": "NOT_AVAILABLE_AFTER_STOP",
            "reason": "per-projection row metrics were held in process memory and consolidated only after both splits; S1 correctness failure stopped before CSV finalization. Saved NPZ contains complete MLP outputs only.",
        },
        "s1_D_identity_CPU_forensic": norm_diagnostics,
        "execution_audit": execution_audit,
        "cost_audit": cost_audit,
        "correctness": {
            "S0": "PASS through native alpha1 parity, down replay, row stitch/commit, and S0 selection lock; S0 held evaluation completed",
            "S1": "FAIL: D=I norm output differs from frozen prefix capture; stopped before AWQ selector, GPTQ candidate generation, selection lock, or P3 held access",
            "overall": "FAIL / stop; no overall quality gate pass claim",
        },
        "ppl": "NOT_RUN",
        "full_model_quantization": "NOT_RUN",
    }
    write_json(RAW / "PARTIAL_ANALYSIS.json", report)
    write_json(RAW / "ACTUAL_EXECUTION_AUDIT.json", execution_audit)
    write_json(RAW / "COST_AUDIT.json", cost_audit)
    write_json(CORRECT / "STOP_DIAGNOSIS.json", {
        "overall_status": "CORRECTNESS_FAIL_STOPPED",
        "S0": {
            "status": "PASS through local gates and held evaluation",
            "passed": ["D=I regression", "gate/up alpha1 native parity", "down alpha1 parameter/group parity", "candidate isolation replays (G01/G11)", "row stitch and actual module commit", "selection lock before P1 held read", "complete L12 held MLP forward and L13 boundary capture"],
            "gap": "G00 down alpha1 parity fields do not include direct-row-output-score comparison; other S0 down arms do.",
        },
        "S1": {
            "status": "FAIL at D=I before AWQ selector or GPTQ candidates",
            "failure": "D=I norm output differs from frozen prefix capture",
            "held_group_P3_opened": False,
            "cuda_failure_diagnostic_saved_by_runner": False,
            "CPU_capture_pair_diagnostic": norm_diagnostics["groups"]["P2"],
            "base_and_prefix_norm_weights_equal": norm_diagnostics["base_prefix_norm_weight_equal"],
            "cause": "Not localized. CPU replay of saved pre/post capture pairs shows large P2 discrepancies despite equal base/prefix norm weights; no tolerance was changed and no GPU retry was run.",
        },
        "static_correctness_issue_found_after_stop": {
            "issue": "Both selected_fold_correctness and eval_held fold-error reference divide the norm weight by D but do not multiply pristine gate/up weights by D before the MLP forward.",
            "locations": ["run_v36.py:1045-1054", "run_v36.py:1117-1136"],
            "consequence": "Stored fold-error MSE is not the FP16 W'=WD, X'=XD^-1 fold error required by the protocol; the fold-error quality gate is invalid for this run.",
            "rerun_after_discovery": False,
        },
        "quality_or_threshold_adjustments": False,
        "GPU_rerun_after_S1_failure": False,
    })
    archive.close()


if __name__ == "__main__":
    main()
