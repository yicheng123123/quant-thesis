"""Freeze v36 article/token manifest, protocol, source hashes and workload.

CPU only: no torch, CUDA, model loading, forward pass, candidate generation,
GPTQ, or quality scoring. Article order inherits v35's already frozen key.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
RESULTS = ROOT / "experiments/range_search_w4/results"
V34 = RESULTS / "range_w4_awq_scaling_gptq_compat_v34"
V35 = RESULTS / "range_w4_awq_gptq_v35"
V18 = RESULTS / "range_w4_fullmodel_v18"
PARQUET = ROOT / ".hf_cache/hub/datasets--Salesforce--wikitext/snapshots/b08601e04326c79dfdd32d625aee71d232d685c3/wikitext-2-raw-v1/train-00000-of-00001.parquet"
TOKENIZER = ROOT / ".hf_cache/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775"
SEQ = 2048
GROUPS = ("P0", "P1", "P2", "P3")
ALPHAS = [round(i / 10, 1) for i in range(1, 11)]


def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, value) -> None:
    payload = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != payload:
            raise RuntimeError(f"Refusing to overwrite differing v36 artifact: {path}")
        return
    path.write_text(payload, encoding="utf-8", newline="\n")


def write_csv(path: Path, rows: list[dict], columns: list[str]) -> None:
    if path.exists():
        raise RuntimeError(f"Refusing to overwrite existing v36 artifact: {path}")
    with path.open("x", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def article_iter(records):
    current, start, article_id = [], None, 0
    for row_index, record in enumerate(records):
        line = record["text"]
        if re.match(r"^= [^=].* =$", line.strip()) and ";" not in line:
            if current:
                yield article_id, start, "\n".join(current)
                article_id += 1
            current, start = [line], row_index
        elif current:
            current.append(line)
    if current:
        yield article_id, start, "\n".join(current)


def main() -> None:
    for p in (V34 / "NEXT_EXPERIMENT_PROTOCOL.json", V35 / "NEW_ARTICLE_MANIFEST.json",
              V35 / "SOURCE_HASHES.json", V35 / "PRECHECK.json", PARQUET,
              TOKENIZER / "tokenizer.json", TOKENIZER / "tokenizer_config.json"):
        if not p.is_file():
            raise FileNotFoundError(p)

    v34 = json.loads((V34 / "NEXT_EXPERIMENT_PROTOCOL.json").read_text(encoding="utf-8"))
    v35_manifest = json.loads((V35 / "NEW_ARTICLE_MANIFEST.json").read_text(encoding="utf-8"))
    v35_precheck = json.loads((V35 / "PRECHECK.json").read_text(encoding="utf-8"))
    if v35_manifest["status"] != "STOP_INSUFFICIENT_DISJOINT_ARTICLES" or len(v35_manifest["articles"]) != 34:
        raise RuntimeError("v35 source is not the frozen 34-article STOP manifest")
    if v35_precheck["eligible_new_whole_articles"] != 34 or v35_precheck["GPU_executions"] != 0:
        raise RuntimeError("v35 precheck status/count mismatch")
    articles = sorted(v35_manifest["articles"], key=lambda x: x["eligible_rank_0based"])
    if [x["eligible_rank_0based"] for x in articles] != list(range(34)):
        raise RuntimeError("v35 deterministic order is not contiguous 0..33")

    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer

    if sha_file(PARQUET) != v35_manifest["dataset_source_sha256"]:
        raise RuntimeError("WT2 train source hash differs from the frozen v35 source")
    tok_hashes = {k: sha_file(TOKENIZER / k) for k in ("tokenizer.json", "tokenizer_config.json")}
    if tok_hashes != v35_manifest["tokenizer_files_sha256"]:
        raise RuntimeError("tokenizer file hash mismatch with v35 manifest")
    table = pq.read_table(PARQUET, columns=["text"]).to_pylist()
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER), local_files_only=True)
    expected = {int(x["source_article_index"]): x for x in articles}
    recovered = {}
    for article_id, row_index, text in article_iter(table):
        record = expected.get(article_id)
        if record is None:
            continue
        full_hash = sha_bytes(text.encode("utf-8"))
        if full_hash != record["article_sha256"] or row_index != record["source_row_index"]:
            raise RuntimeError(f"v35 article identity mismatch at frozen ID {article_id}")
        token_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        if len(token_ids) != record["full_article_tokens"] or len(token_ids) < SEQ:
            raise RuntimeError(f"token count mismatch at article {article_id}")
        token_ids = token_ids[:SEQ]
        token_hash = sha_bytes(json.dumps(token_ids).encode("utf-8"))
        if token_hash != record["token_sha256"]:
            raise RuntimeError(f"v35 token hash mismatch at article {article_id}")
        recovered[article_id] = (token_ids, token_hash)
    if set(recovered) != set(expected):
        raise RuntimeError(f"did not rematerialize all v35 articles: {len(recovered)}/34")

    chosen = articles[:32]
    unused = articles[32:]
    group_for_rank = {rank: GROUPS[rank // 8] for rank in range(32)}
    split_for_group = {"P0": ("S0", "search"), "P1": ("S0", "held"),
                       "P2": ("S1", "search"), "P3": ("S1", "held")}
    frozen_records = []
    for entry in chosen:
        rank = int(entry["eligible_rank_0based"])
        group = group_for_rank[rank]
        split, role = split_for_group[group]
        frozen_records.append({**entry, "group": group, "split": split, "role": role,
                               "token_ids_sha256_reverified": recovered[int(entry["source_article_index"])][1]})
    cache_manifest = {group: [] for group in GROUPS}
    for item in frozen_records:
        aid = int(item["source_article_index"])
        cache_manifest[item["group"]].append({"source_article_index": aid,
                                                 "token_ids": recovered[aid][0],
                                                 "token_sha256": recovered[aid][1]})

    protocol = json.loads(json.dumps(v34))
    protocol["protocol_id"] = "v36_l12_awq_selector_dynamic_gptq_32_new_articles"
    protocol["status"] = "FROZEN_BEFORE_MODEL_OR_GPU"
    protocol["data"]["assignment"] = "v35 eligible_rank order: ranks 0..7 P0, 8..15 P1, 16..23 P2, 24..31 P3; S0=P0 search/P1 held; S1=P2 search/P3 held"
    protocol["data"]["articles_per_group"] = 8
    protocol["data"]["tokens_per_group"] = 8 * SEQ
    protocol["data"]["total_unique_articles"] = 32
    protocol["data"]["pre_gpu_exclusion"] = "Use only v35's 34 verified disjoint whole articles in its deterministic ordering_key_sha256 order. Freeze first 32 before any model/GPU; ranks 32-33 unused. No text/error-driven reorder, no backfill. Preserve v34 tokenizer/revision/special-token/length/exclusion rules."
    protocol["data"]["v36_scale_change"] = {
        "v34_plan": {"search_articles_per_split": 16, "held_articles_per_split": 16},
        "v36": {"search_articles_per_split": 8, "held_articles_per_split": 8},
        "reason": "v35's independent preflight found only 34 eligible articles; v36 is a new smaller-scale protocol and does not make v35 pass",
        "strength": "lower sample support than v34; articles remain whole and independent units; tokens are not independent replicates"
    }
    protocol["data"]["group_counts"] = {g: 8 for g in GROUPS}
    protocol["workload"]["gpu_budget"] = json.loads(json.dumps(v34["workload"]["gpu_budget"]))
    protocol["workload"]["gpu_budget"]["cumulative_hard_cap_minutes"] = 30
    protocol["workload"]["gpu_budget"]["abort"] = "Hard cumulative GPU wall 1800s includes model load, warmup, correctness, all candidates, forward/score, retries, cache/checkpoint I/O. Stop new GPU work at 1620s to retain 180s finalization reserve; no extension."
    protocol["held_gate"]["sample_size_application"] = {
        "frozen_v34_numeric_gates_unchanged": True,
        "why_no_sample_dependent_clause_adjustment": "The v34 gates are fixed relative effect-size and row/projection-tail thresholds; none sets a minimum document count or uses a sample-count denominator. In v36 they are applied to aggregate scores from 8 held whole articles (16384 token positions per split), so uncertainty is larger and must be reported; thresholds are not recalibrated after held access."
    }
    protocol["execution_status"] = {"gpu_experiment": "FROZEN_NOT_STARTED", "harness": "v36 independent harness", "runtime": True}
    protocol["v36_artifacts"] = "experiments/range_search_w4/results/range_w4_awq_gptq_v36 only"

    # Rebuild v34 job arithmetic with v36 token/document scale. Candidate
    # counts and row-alpha slots are invariant; token/output work is not.
    jobs = []
    for split in ("S0", "S1"):
        for ratio_i in range(20):
            jobs.append({"split": split, "phase": "search", "task_type": "AWQ_RTN_D_selector_ratio",
                         "arm": "G10_and_G11_shared", "linear": "gate_proj+up_proj; down FP16",
                         "shape": "gate/up=[4864,896]; down=[896,4864]", "candidate_count": 1,
                         "candidate_group_count": 20, "scored_rows": 896,
                         "tokens_per_candidate": 8 * SEQ, "row_alpha_score_slots": 0,
                         "score_tensor_elements": 8 * SEQ * 896,
                         "cache_reuse": "one captured postnorm search X/reference reused; pristine weights restored per ratio",
                         "additional_gptq_calls": 0, "detail": f"ratio={ratio_i*0.05:.2f}; full MLP output score"})
        for coord, bank in (("original", "G00/G01 shared"), ("AWQ_D", "G10/G11 shared")):
            for linear in ("gate_proj", "up_proj"):
                jobs.append({"split": split, "phase": "search", "task_type": "main_GPTQ_candidate_bank",
                             "arm": bank, "linear": linear, "shape": "[4864,896]", "candidate_count": 10,
                             "candidate_group_count": 10, "scored_rows": 4864,
                             "tokens_per_candidate": 8 * SEQ, "row_alpha_score_slots": 10 * 4864,
                             "score_tensor_elements": 10 * 4864 * 8 * SEQ,
                             "cache_reuse": f"one H from its actual search input per coordinate; all rows scored once/candidate; {coord}",
                             "additional_gptq_calls": 0, "detail": "alpha=1 is native baseline within bank"})
        for arm, count in (("G00", 1), ("G01", 10), ("G10", 1), ("G11", 10)):
            jobs.append({"split": split, "phase": "search", "task_type": "main_GPTQ_down_candidate",
                         "arm": arm, "linear": "down_proj", "shape": "[896,4864]", "candidate_count": count,
                         "candidate_group_count": count, "scored_rows": 896,
                         "tokens_per_candidate": 8 * SEQ, "row_alpha_score_slots": count * 896,
                         "score_tensor_elements": count * 896 * 8 * SEQ,
                         "cache_reuse": "arm-specific gate/up committed downstream input and H; fresh GPTQ candidate state",
                         "additional_gptq_calls": 0, "detail": "down input/H recaptured after whole gate/up stitch"})
        for name, count in (("native alpha1 parity", 8), ("down candidate isolation replay", 2)):
            jobs.append({"split": split, "phase": "correctness", "task_type": name,
                         "arm": "correctness", "linear": "four coordinate gate/up states and four arm down inputs" if count == 8 else "down_proj",
                         "shape": "full native Linear shape", "candidate_count": count,
                         "candidate_group_count": count, "scored_rows": 0, "tokens_per_candidate": 0,
                         "row_alpha_score_slots": 0, "score_tensor_elements": 0,
                         "cache_reuse": "same H/input semantics; pristine weight and fresh candidate state",
                         "additional_gptq_calls": count, "detail": "v34 frozen checks; 8 parity and 2 replays per split"})
    write_csv(OUT / "JOB_TABLE.csv", jobs,
              ["split", "phase", "task_type", "arm", "linear", "shape", "candidate_count",
               "candidate_group_count", "scored_rows", "tokens_per_candidate", "row_alpha_score_slots",
               "score_tensor_elements", "cache_reuse", "additional_gptq_calls", "detail"])

    main_gptq = sum(x["candidate_count"] for x in jobs if x["task_type"].startswith("main_GPTQ"))
    score_slots = sum(x["row_alpha_score_slots"] for x in jobs)
    check_calls = sum(x["additional_gptq_calls"] for x in jobs)
    output_elements = sum(x["score_tensor_elements"] for x in jobs)
    gptq_output_elements = sum(x["score_tensor_elements"] for x in jobs if x["task_type"].startswith("main_GPTQ"))
    selector_output_elements = sum(x["score_tensor_elements"] for x in jobs if x["task_type"] == "AWQ_RTN_D_selector_ratio")
    # v34 protocol arithmetic, intentionally unchanged by a smaller document split.
    if (main_gptq, score_slots, check_calls) != (124, 428544, 20):
        raise RuntimeError(f"v36 GPTQ/score slot count mismatch: {main_gptq}/{score_slots}/{check_calls}")
    if (8 * SEQ * 896 * 20 * 2) != 587202560:
        raise RuntimeError("AWQ selector output element count mismatch")
    awq_rows = sum(x["candidate_count"] for x in jobs if x["task_type"] == "AWQ_RTN_D_selector_ratio")
    # Correctness jobs count separately; parity calls and row-score slots do not
    # become candidate executions or added alpha choices.
    audit = {
        "status": "PASS_COUNT_RECONCILIATION; EXECUTION_NOT_RUN",
        "data": {"new_articles": 32, "unused_eligible_articles": 2, "documents_per_group": 8,
                 "search_documents_per_split": 8, "held_documents_per_split": 8,
                 "tokens_per_document": SEQ, "search_token_positions_per_split": 8*SEQ,
                 "held_token_positions_per_split": 8*SEQ},
        "counts": {"AWQ_ratio_candidates_total": awq_rows, "RTN_fakequant_Linear_ops": 80,
                   "full_MLP_selector_forwards": 40, "reference_MLP_forwards": 2,
                   "main_full_Linear_GPTQ_executions": main_gptq,
                   "alpha1_parity_GPTQ_calls": 16, "down_isolation_replay_calls": 4,
                   "total_full_Linear_GPTQ_if_all_run": main_gptq+check_calls,
                   "row_alpha_score_slots": score_slots,
                   "selector_full_MLP_output_elements": 587202560,
                   "GPTQ_local_score_output_elements": gptq_output_elements,
                   "AWQ_selector_output_score_elements": selector_output_elements,
                   "candidate_score_elements_total": output_elements,
                   "unique_H_per_split": 6, "unique_H_total": 12,
                   "prefix_capture_sets": 4, "search_gate_up_to_down_refreshes": 8,
                   "held_full_MLP_arm_forwards": 8, "L13_boundary_captures": 8},
        "cost_units": {
            "candidate_counts_are_invariant_to_document_count": True,
            "search_tokens_per_candidate_v34": 32768,
            "search_tokens_per_candidate_v36": 16384,
            "held_tokens_per_arm_split_v34": 32768,
            "held_tokens_per_arm_split_v36": 16384,
            "warning": "Article count is halved, but GPTQ executions, alpha choices, RTN ratio candidates, H count, prefix set count and candidate score slots are not halved; activation/H and forward/scoring token workload is lower."
        },
        "job_table": {"rows": len(jobs), "main_GPTQ_calls": main_gptq,
                      "row_alpha_score_slots": score_slots, "correctness_GPTQ_calls": check_calls,
                      "candidate_score_tensor_elements": output_elements},
        "quality_gate": "v34 numerical gates unchanged; applied to 8 held whole articles per split/16384 token positions; higher uncertainty explicitly reported",
        "gpu_budget_seconds": 1800,
        "stop_compute_by_seconds": 1620,
    }

    write_json(OUT / "manifests" / "ARTICLE_MANIFEST.json", {
        "manifest_id": "v36_frozen_32_of_v35_34_deterministic_order",
        "status": "FROZEN_BEFORE_MODEL_OR_GPU",
        "source_manifest": "v35 NEW_ARTICLE_MANIFEST.json",
        "source_manifest_sha256": sha_file(V35 / "NEW_ARTICLE_MANIFEST.json"),
        "dataset_source_sha256": v35_manifest["dataset_source_sha256"],
        "tokenizer_revision": v35_manifest["tokenizer_revision"],
        "tokenizer_files_sha256": tok_hashes,
        "add_special_tokens": False, "tokens_per_article": SEQ,
        "ordering": "exact v35 eligible_rank_0based ordering; no text/model selection",
        "groups": {g: [x for x in frozen_records if x["group"] == g] for g in GROUPS},
        "selected_articles": frozen_records,
        "not_used": unused,
        "group_hashes": {g: sha_bytes(json.dumps(cache_manifest[g], separators=(",", ":")).encode("utf-8")) for g in GROUPS},
    })
    write_json(OUT / "cache" / "tokens_by_group.json", cache_manifest)
    write_json(OUT / "manifests" / "FROZEN_PROTOCOL.json", protocol)
    write_json(OUT / "JOB_TABLE_AUDIT.json", audit)

    source_paths = [
        V34 / "NEXT_EXPERIMENT_PROTOCOL.json", V34 / "REPORT.md", V34 / "SOURCE_AUDIT.md", V34 / "COMPARISON_TABLE.csv",
        V35 / "NEW_ARTICLE_MANIFEST.json", V35 / "FROZEN_PROTOCOL_V35.json", V35 / "JOB_TABLE.csv",
        V35 / "JOB_TABLE_AUDIT.json", V35 / "PRECHECK.json", V35 / "REPORT.md", V35 / "SOURCE_HASHES.json",
        V18 / "PROTOCOL.json", V18 / "run_formal.py", V18 / "FORMAL_TIMING.csv",
        V18 / "CALIBRATION_MANIFEST.json", V18 / "M2/RUN_REPORT.json",
        RESULTS / "range_w4_m2_runtime_v17_3/optimized_native_m2.py",
        V18 / "M2/checkpoint/final/model.safetensors", TOKENIZER / "model.safetensors",
        PARQUET, TOKENIZER / "tokenizer.json", TOKENIZER / "tokenizer_config.json", Path(__file__),
    ]
    source_hashes = {p.relative_to(ROOT).as_posix(): (sha_file(p) if p.is_file() else "MISSING") for p in source_paths}
    write_json(OUT / "manifests" / "SOURCE_HASHES.json", {
        "git_head": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip(),
        "worktree_dirty_at_freeze": True,
        "v35_status_preserved": v35_manifest["status"],
        "paths": source_hashes,
        "official_awq": v34["provenance"]["official_upstream"],
        "note": "All v35 and historical paths are source-only; v36 outputs are isolated. The token cache is frozen provenance only; runtime must not evaluate held until lock."
    })
    write_json(OUT / "raw" / "RUN_STATE.json", {
        "status": "FROZEN_PREFLIGHT_COMPLETE_GPU_NOT_STARTED",
        "protocol_sha256": sha_file(OUT / "manifests" / "FROZEN_PROTOCOL.json"),
        "article_manifest_sha256": sha_file(OUT / "manifests" / "ARTICLE_MANIFEST.json"),
        "token_cache_sha256": sha_file(OUT / "cache" / "tokens_by_group.json"),
        "job_table_sha256": sha_file(OUT / "JOB_TABLE.csv"),
        "job_audit_sha256": sha_file(OUT / "JOB_TABLE_AUDIT.json"),
        "gpu_seconds_used": 0, "completed_tasks": [], "held_access": "NOT_OPENED_BY_MODEL_RUNTIME",
        "resume_rule": "Require all four freeze hashes to match before any resume; only completed full dependency groups may be skipped."
    })
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
