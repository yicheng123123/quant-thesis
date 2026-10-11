from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
REPO = HERE.parents[4]
V18 = REPO / "experiments" / "range_search_w4" / "results" / "range_w4_fullmodel_v18"


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def main():
    training = read_json(HERE / "M3_TRAINING_SUMMARY.json")
    reload_check = read_json(HERE / "DENSE_EXPORT_RELOAD_CHECK.json")
    eval_summary = read_json(HERE / "evaluation" / "M3_SUMMARY.json")
    eval_check = read_json(HERE / "evaluation" / "M3_EVAL_CORRECTNESS.json")
    preflight = read_json(ROOT / "LOCAL_PREFLIGHT" / "correctness_checks.json")
    preflight_verdict = read_json(ROOT / "LOCAL_PREFLIGHT" / "PREFLIGHT_VERDICT.json")
    preflight_cost = read_json(ROOT / "LOCAL_PREFLIGHT" / "LOCAL_COST_RECALC.json")
    model = read_json(ROOT / "LOCAL_MODEL_LOAD_CHECK.json")
    tokenizer = read_json(ROOT / "TOKENIZER_CHECK.json")
    parity = read_json(ROOT / "QUANT_OFF_PARITY.json")
    source_audit = read_json(HERE / "OMNIQUANT_SOURCE_AUDIT.json")
    resume_smoke = read_json(HERE / "RESUME_SMOKE" / "RESUME_SMOKE.json")
    atomic_save_smoke = read_json(HERE / "ATOMIC_SAVE_SMOKE.json")
    recovery = read_json(HERE / "RUNNER_RECOVERY_NOTE.json")
    attempt1 = read_json(HERE / "ATTEMPT_01_MANIFEST.json")
    latest = read_json(HERE / "checkpoints" / "LATEST.json")
    checkpoint_dir = HERE / "checkpoints" / latest["checkpoint_dir"]
    checkpoint = read_json(checkpoint_dir / "CHECKPOINT_MANIFEST.json")
    if checkpoint["completed_layer_count"] != 24 or checkpoint["effective_optimizer_updates_in_checkpoint"] != 61440:
        raise RuntimeError("final checkpoint does not contain all 24 blocks / 61,440 effective updates")
    for key in ("state", "rng"):
        file = checkpoint_dir / checkpoint[f"{key}_file"]
        if file.stat().st_size != checkpoint[f"{key}_file_bytes"] or sha_file(file) != checkpoint[f"{key}_file_sha256"]:
            raise RuntimeError(f"final checkpoint {key} artifact failed its committed hash")

    conditions = {
        "frozen_model_identity": model.get("status") == "PASS" and tokenizer.get("status") == "PASS",
        "preflight_correctness": preflight.get("status") == "PASS",
        "preflight_local_go": preflight_verdict.get("verdict") == "LOCAL_M3_FORMAL_GO",
        "preflight_24h_and_memory_gates": preflight_cost.get("memory_gate_pass") is True and preflight_cost.get("runtime_gate_pass_le_24h") is True,
        "quant_off_native_port_parity": parity.get("status") == "PASS",
        "isolated_source_audit": source_audit.get("status") == "PASS" and source_audit.get("quantizer_source_modified") is False and source_audit.get("optimizer_or_loss_math_modified") is False,
        "block_resume_smoke": resume_smoke.get("status") == "PASS" and resume_smoke.get("lwc_state_tensors_equal") and resume_smoke.get("quantized_tensors_equal") and resume_smoke.get("logits_bitwise_equal"),
        "atomic_checkpoint_save_smoke": atomic_save_smoke.get("status") == "PASS",
        "checkpoint_recovery_accounted": recovery.get("status") == "CHECKPOINT_IO_FIX_BEFORE_RESTART" and attempt1.get("optimizer_updates_attempted") == 2560 and attempt1.get("completed_blocks_committed") == 0,
        "formal_training": training.get("status") == "PASS" and training.get("effective_optimizer_updates") == 61440 and training.get("completed_blocks") == 24,
        "formal_step_checks": training.get("training_step_check_counts", {}).get("steps") == 61440 and training.get("training_step_check_counts", {}).get("finite_losses") == 61440 and training.get("training_step_check_counts", {}).get("finite_gradients") == 61440,
        "dense_export_reload": reload_check.get("status") == "PASS",
        "evaluation_correctness": eval_check.get("status") == "PASS",
    }
    if not all(conditions.values()):
        raise RuntimeError(f"final correctness conditions failed: {conditions}")

    attempts = {}
    with (HERE / "logs" / "training_steps.csv").open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            attempts[row["attempt_id"]] = attempts.get(row["attempt_id"], 0) + 1
    total_attempted_updates = sum(attempts.values())
    if attempts.get(training["attempt_id"]) != 61440:
        raise RuntimeError("successful attempt step log is not exactly 61,440 rows")
    if total_attempted_updates != 64000:
        raise RuntimeError(f"attempted optimizer update log count is unexpected: {total_attempted_updates}")

    checks = {
        "status": "PASS", "checks": conditions,
        "model_id": model.get("model_id"), "model_revision": model.get("revision"),
        "model_manifest_sha256": sha_file(ROOT / "LOCAL_MODEL_MANIFEST.json"),
        "calibration_manifest_sha256": training.get("calibration_manifest_sha256"),
        "ordered_calibration_token_hashes_sha256": training.get("ordered_token_hashes_sha256"),
        "official_omniquant_commit": "feffe8ea87d80f7bb57b6e25e7cff9dc950fcc14",
        "official_quantizer_and_optimizer_math_unmodified": True,
        "formal_effective_optimizer_updates": training["effective_optimizer_updates"],
        "all_attempted_updates_including_uncommitted_first_attempt": total_attempted_updates,
        "uncommitted_updates_excluded_from_final_checkpoint": attempt1["optimizer_updates_attempted"],
        "dense_export_reload_max_abs_logit_diff": reload_check["max_abs_diff"],
        "wt2_scored_targets": eval_summary["WT2_scored_targets"],
        "c4_fixed_budget_scored_targets": eval_summary["C4_fixed_budget_scored_targets"],
    }
    write_json(ROOT / "correctness_checks.json", checks)

    with (V18 / "FULLMODEL_COMPARISON.csv").open(newline="", encoding="utf-8") as stream:
        baseline_rows = list(csv.DictReader(stream))
    wt2 = eval_summary["results"]["wikitext"]
    c4 = eval_summary["results"]["c4_fixed_budget"]
    m3_alloc = int(training["peak_allocated_bytes"])
    methods = {row["Method"]: row for row in baseline_rows}
    output_rows = []
    for method in ("M0", "M1", "M2"):
        row = methods[method]
        output_rows.append({
            "Method": method,
            "Algorithm": row["Precision/Quant"],
            "Quant": row["Precision/Quant"],
            "GPU": "NVIDIA GeForce RTX 4060 Laptop GPU",
            "Calibration samples": "" if method == "M0" else 128,
            "Calibration tokens": "" if method == "M0" else 262144,
            "Calibration wall seconds": row["Calibration wall seconds"],
            "Peak allocated bytes": row["Calibration peak GPU allocated bytes"],
            "WT2 NLL": row["WT2 NLL"], "WT2 PPL": row["WT2 PPL"],
            "C4 fixed-budget NLL": row["C4 fixed-budget NLL"], "C4 fixed-budget PPL": row["C4 fixed-budget PPL"],
            "Representation": row["Representation"], "Status": "REUSED_V18_ARTIFACT",
        })
    output_rows.append({
        "Method": "M3",
        "Algorithm": "Official OmniQuant W4A16/G128; LWC on; LET off; 20 epochs",
        "Quant": "W4A16 asymmetric affine G128 [0,15]",
        "GPU": "NVIDIA GeForce RTX 4060 Laptop GPU",
        "Calibration samples": 128, "Calibration tokens": 262144,
        "Calibration wall seconds": training["calibration_wall_seconds"],
        "Peak allocated bytes": m3_alloc,
        "WT2 NLL": wt2["average_nll"], "WT2 PPL": wt2["perplexity"],
        "C4 fixed-budget NLL": c4["average_nll"], "C4 fixed-budget PPL": c4["perplexity"],
        "Representation": "dense-dequant FP16", "Status": "NEW_LOCAL_FORMAL_M3",
    })
    fields = ["Method", "Algorithm", "Quant", "GPU", "Calibration samples", "Calibration tokens",
              "Calibration wall seconds", "Peak allocated bytes", "WT2 NLL", "WT2 PPL",
              "C4 fixed-budget NLL", "C4 fixed-budget PPL", "Representation", "Status"]
    for path in (ROOT / "M0_M1_M2_M3.csv", HERE / "M0_M1_M2_M3.csv"):
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(output_rows)

    comparison = {}
    for dataset, current in (("WT2", float(wt2["perplexity"])), ("C4_fixed_budget", float(c4["perplexity"]))):
        m2_key = "WT2 PPL" if dataset == "WT2" else "C4 fixed-budget PPL"
        m1 = float(methods["M1"][m2_key])
        m2 = float(methods["M2"][m2_key])
        m0 = float(methods["M0"][m2_key])
        comparison[dataset] = {
            "M3_PPL": current, "M0_PPL": m0, "M1_PPL": m1, "M2_PPL": m2,
            "M3_vs_M2_percent": (current / m2 - 1.0) * 100.0,
            "M3_vs_M1_percent": (current / m1 - 1.0) * 100.0,
            "M3_lower_than_M2": current < m2,
            "M3_lower_than_M1": current < m1,
        }
    write_json(HERE / "M3_COMPARISON_TO_V18.json", {
        "status": "PASS", "scope": "same frozen model and evaluation token manifests; method-level comparison, not a causal mechanism claim",
        "M3": comparison,
        "historical_baseline_source": str((V18 / "FULLMODEL_COMPARISON.csv").relative_to(REPO)),
        "historical_baseline_sha256": sha_file(V18 / "FULLMODEL_COMPARISON.csv"),
        "M3_eval_summary_sha256": sha_file(HERE / "evaluation" / "M3_SUMMARY.json"),
    })

    report = [
        "# v21.1 Local Frozen Model Snapshot Repair and M3",
        "",
        "**Status: `M3_FORMAL_BASELINE_COMPLETED_LOCAL`**",
        "",
        "## Protocol and correctness",
        "",
        f"- Model: `{model.get('model_id')}` at exact revision `{model.get('revision')}`, FP16, local frozen snapshot; tokenizer identity and the 128 × 2048 WikiText-2 train token hashes match the frozen manifests.",
        "- Quantization: official OmniQuant commit `feffe8ea87d80f7bb57b6e25e7cff9dc950fcc14`, W4A16 asymmetric affine G128, LWC on, LET off, 20 epochs, batch size 1. The official quantizer, reconstruction loss, AMP, AdamW, and compensation code were not modified.",
        f"- Formal effective work: {training['completed_blocks']} blocks × 20 epochs × 128 samples = {training['effective_optimizer_updates']:,} optimizer updates. Finite loss and finite LWC gradients were checked for every committed-run update; quantized weights, scales, and zero points were checked at each block boundary.",
        f"- Quant-off native/port parity: **PASS**. Block-boundary interruption/resume smoke: **PASS**, including bitwise-equal LWC state, quantized weights/scales/zeros, and 64-token logits.",
        f"- Dense-dequant export/reload: **PASS**; maximum logit difference `{reload_check['max_abs_diff']:.8g}` over the frozen 64-token probe.",
        "",
        "## Training cost",
        "",
        f"- Accepted M3 calibration wall: **{training['calibration_wall_seconds']:.1f} s ({training['calibration_wall_hours']:.3f} h)** for the full official layer loop; model load `{training['model_load_seconds']:.1f} s` is reported separately.",
        f"- Formal peak GPU memory: `{training['peak_allocated_gib']:.3f} GiB` allocated, `{training['peak_reserved_gib']:.3f} GiB` reserved.",
        f"- A Windows read-only-handle `fsync` error occurred after the first L0 block had run but before any checkpoint committed. Its {attempt1['optimizer_updates_attempted']:,} updates were discarded; the first block was rerun from pristine weights. Total attempted optimizer updates across attempts were {total_attempted_updates:,}; the final checkpoint contains exactly {training['effective_optimizer_updates']:,} effective updates. The failed attempt and temporary checkpoint are preserved.",
        f"- Preflight estimate before formal run: `{preflight_cost['conservative_total_hours']:.3f} h` conservative; actual accepted calibration wall is reported above. The failure/restart cost is separately disclosed.",
        "",
        "## Evaluation",
        "",
        f"- WikiText-2 test: **{wt2['perplexity']:.6f} PPL**, `{wt2['evaluated_tokens']:,}` targets, `{wt2['window_count']}` windows.",
        f"- C4 validation fixed token budget: **{c4['perplexity']:.6f} PPL**, `{c4['evaluated_tokens']:,}` targets, `{c4['window_count']}` windows. This is not full C4 PPL.",
        f"- Relative to the reused v18 M2 artifact: WT2 `{comparison['WT2']['M3_vs_M2_percent']:+.3f}%`; C4 fixed-budget `{comparison['C4_fixed_budget']['M3_vs_M2_percent']:+.3f}%` (negative means lower PPL). M0/M1/M2 values were reused; they were not rerun.",
        "- M3 versus M2 is a method-level comparison: M3 uses OmniQuant LWC with asymmetric affine quantization, while M2 is the frozen GPTQ range-search method. This result does not isolate a single mechanism or establish generalization beyond the evaluated model and manifests.",
        "- The same `benchmark.eval.ppl.evaluate_ppl` implementation, FP16/SDPA, TF32-off settings, 2048 context, 512 stride, and frozen v18 WT2/C4 token-ID manifests were used. No evaluation data was used for training or parameter selection.",
        "",
        "## Artifact map",
        "",
        "- Raw M3 evaluation: `M3/evaluation/raw/`; per-window rows: `M3/evaluation/windows/`.",
        "- Training logs and per-block health: `M3/logs/`; resumable checkpoints: `M3/checkpoints/`.",
        "- Dense-dequant FP16 model: `M3/dense_export/`; export identity: `M3/DENSE_EXPORT_MANIFEST.json`.",
        "- Frozen configs, manifests, code hashes, and checks: `PROTOCOL.json`, `M3/FORMAL_PROTOCOL_LOCK.json`, `LOCAL_MODEL_MANIFEST.json`, and `correctness_checks.json`.",
    ]
    report_text = "\n".join(report) + "\n"
    (ROOT / "REPORT.md").write_text(report_text, encoding="utf-8")
    (HERE / "RUN_REPORT.md").write_text(report_text, encoding="utf-8")
    write_json(ROOT / "FINAL_STATUS.json", {
        "status": "M3_FORMAL_BASELINE_COMPLETED_LOCAL",
        "effective_optimizer_updates": training["effective_optimizer_updates"],
        "all_attempted_updates": total_attempted_updates,
        "formal_calibration_wall_seconds": training["calibration_wall_seconds"],
        "formal_peak_allocated_bytes": training["peak_allocated_bytes"],
        "wt2_ppl": wt2["perplexity"], "c4_fixed_budget_ppl": c4["perplexity"],
        "correctness": "PASS", "evaluation": "PASS",
    })
    print("Final v21.1 M3 report and comparison artifacts written.")


if __name__ == "__main__":
    main()
