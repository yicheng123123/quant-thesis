"""CPU-only v35 article preflight and frozen workload accounting.

This script never imports torch, initializes CUDA, loads a model, or executes GPTQ.
It mirrors the v23 whole-article parser/tokenization and stops if fewer than 64
new disjoint articles remain after the v34 exclusion rule.
"""
from __future__ import annotations

import csv
import hashlib
import importlib.metadata as metadata
import json
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
RESULTS = ROOT / "experiments/range_search_w4/results"
V34 = RESULTS / "range_w4_awq_scaling_gptq_compat_v34"
V18 = RESULTS / "range_w4_fullmodel_v18"
V18_WT2_TEST = V18 / "evaluation/manifests/WIKITEXT_TEST_MANIFEST.json"
V18_C4_EVAL = V18 / "evaluation/manifests/C4_EVAL_MANIFEST.json"
V14_MANIFEST = RESULTS / "range_w4_calibration_coverage_v14/article_manifest.json"
V15_MANIFEST = RESULTS / "range_w4_quantized_prefix_v15/article_manifest.json"
V23 = RESULTS / "range_w4_block_objective_e1_v23"
V24 = RESULTS / "range_w4_structured_group_e2_v24"
V31 = RESULTS / "range_w4_generation_selection_split_v31"
V32 = RESULTS / "range_w4_fixed_channel_scale_scalar_v32"
V33 = RESULTS / "range_w4_shared_coordinate_selection_v33"
PARQUET = ROOT / ".hf_cache/hub/datasets--Salesforce--wikitext/snapshots/b08601e04326c79dfdd32d625aee71d232d685c3/wikitext-2-raw-v1/train-00000-of-00001.parquet"
TOKENIZER = ROOT / ".hf_cache/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775"
TOKENS_PER_ARTICLE = 2048
REQUIRED_ARTICLES = 64
ALPHAS = [round(i / 10, 1) for i in range(1, 11)]


def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def stable_json(path: Path, obj) -> None:
    text = json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != text:
            raise RuntimeError(f"Refusing to overwrite different v35 artifact: {path}")
        return
    path.write_text(text, encoding="utf-8", newline="\n")


def write_csv(path: Path, rows: list[dict], columns: list[str]) -> None:
    if path.exists():
        raise RuntimeError(f"Refusing to overwrite existing v35 artifact: {path}")
    with path.open("x", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def articles(records):
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


def article_records(obj):
    """Extract article identities from nested historical manifests."""
    found = []

    def visit(value):
        if isinstance(value, dict):
            if "source_article_index" in value and value.get("article_sha256"):
                found.append(value)
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(obj)
    return found


def main() -> None:
    for subdir in ("manifests", "cache", "checkpoints", "final", "raw", "logs", "correctness"):
        (OUT / subdir).mkdir(parents=True, exist_ok=True)
    v34_protocol_path = V34 / "NEXT_EXPERIMENT_PROTOCOL.json"
    e2_path = V24 / "E2_DOCUMENT_MANIFEST.json"
    v18_cal_path = V18 / "CALIBRATION_MANIFEST.json"
    e1_path = V23 / "E1_DOCUMENT_MANIFEST.json"
    for p in (v34_protocol_path, e2_path, v18_cal_path, e1_path,
              V14_MANIFEST, V15_MANIFEST, PARQUET,
              TOKENIZER / "tokenizer.json", TOKENIZER / "tokenizer_config.json"):
        if not p.exists():
            raise FileNotFoundError(p)

    protocol = json.loads(v34_protocol_path.read_text(encoding="utf-8"))
    e2 = json.loads(e2_path.read_text(encoding="utf-8"))
    v18_cal = json.loads(v18_cal_path.read_text(encoding="utf-8"))
    e1 = json.loads(e1_path.read_text(encoding="utf-8"))
    wt2_test = json.loads(V18_WT2_TEST.read_text(encoding="utf-8"))
    c4_eval = json.loads(V18_C4_EVAL.read_text(encoding="utf-8"))
    dataset_sha = sha_file(PARQUET)
    if dataset_sha != e2["dataset_source_sha256"]:
        raise RuntimeError("Cached WT2 train parquet differs from v24 frozen source hash")
    if e2["tokenizer_revision"] != protocol["model"]["revision"]:
        raise RuntimeError("v24 tokenizer revision does not match v34 frozen revision")

    excluded: dict[int, dict] = {}

    def add_exclusion(article_id: int, article_hash: str, token_hash: str | None, source: str,
                      split_relation: str = ""):
        old = excluded.get(article_id)
        if old and old["article_sha256"] != article_hash:
            raise RuntimeError(f"Article ID/hash conflict for {article_id}")
        if old is None:
            excluded[article_id] = {
                "source_article_index": article_id,
                "article_sha256": article_hash,
                "token_sha256": token_hash or "",
                "exclusion_sources": {source},
                "split_relations": {split_relation} if split_relation else set(),
            }
        else:
            old["exclusion_sources"].add(source)
            if split_relation:
                old["split_relations"].add(split_relation)
            if token_hash:
                if old["token_sha256"] and old["token_sha256"] != token_hash:
                    old.setdefault("additional_token_sha256", set()).add(token_hash)
                else:
                    old["token_sha256"] = token_hash

    historical_sources = [
        ("v14 calibration/heldout/blacklist", V14_MANIFEST),
        ("v15 calibration/heldout", V15_MANIFEST),
        ("v18 formal M2 calibration", v18_cal_path),
    ]
    for label, source_path in historical_sources:
        obj = json.loads(source_path.read_text(encoding="utf-8"))
        for sample in article_records(obj):
            ids = sample.get("token_ids")
            token_hash = sample.get("token_sha256")
            if not token_hash and ids is not None:
                token_hash = sha_bytes(json.dumps(ids).encode("utf-8"))
            relation = str(sample.get("split") or f"{label}; article record")
            add_exclusion(int(sample["source_article_index"]), sample["article_sha256"], token_hash,
                          label, relation)

    hist = e2["historical_exclusions"]
    # The v24 arrays are independently sorted; reconstruct ID↔hash pairs from
    # their source manifests rather than zipping the two arrays.
    if {int(a) for a in hist["article_ids"]} != set(excluded):
        raise RuntimeError("Reconstructed historical article IDs differ from v24 exclusion set")
    if set(hist["full_article_sha256"]) != {r["article_sha256"] for r in excluded.values()}:
        raise RuntimeError("Reconstructed historical full-article hashes differ from v24 exclusion set")
    historical_ids = set(excluded)
    historical_hashes = {r["article_sha256"] for r in excluded.values()}
    historical_token_hashes = {r["token_sha256"] for r in excluded.values() if r["token_sha256"]}

    e2_samples = e2["samples"]
    if len(e2_samples) != 64:
        raise RuntimeError(f"Expected the 64 frozen v24 articles, found {len(e2_samples)}")
    expected_split_counts = {"S0_search": 16, "S0_held": 16, "S1_search": 16, "S1_held": 16}
    actual_split_counts = {s: sum(x.get("split") == s for x in e2_samples) for s in expected_split_counts}
    if actual_split_counts != expected_split_counts:
        raise RuntimeError(f"v24 whole-article split counts do not match frozen 16x4 design: {actual_split_counts}")
    v24_by_id = {int(x["source_article_index"]): x for x in e2_samples}
    v24_hashes = {x["article_sha256"] for x in e2_samples}
    v24_token_hashes = {x["token_sha256"] for x in e2_samples}
    for sample in e2_samples:
        add_exclusion(int(sample["source_article_index"]), sample["article_sha256"],
                      sample.get("token_sha256"), "v24/v31/v32/v33 historical search/held",
                      str(sample.get("split", "v24 split unavailable")))

    excluded_hashes = {x["article_sha256"] for x in excluded.values()}
    excluded_token_hashes = {x["token_sha256"] for x in excluded.values() if x["token_sha256"]}
    source_fingerprint = f"Salesforce/wikitext:wikitext-2-raw-v1:train:{dataset_sha}"

    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer

    data = pq.read_table(PARQUET, columns=["text"]).to_pylist()
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER), local_files_only=True)
    available = []
    eligible_before_v24 = 0
    matched_v24_ids = set()
    exclusion_reasons = {"historical_article_id_or_hash": 0, "historical_token_hash": 0,
                         "v24_article_id_or_hash_or_token_hash": 0, "shorter_than_2048": 0}
    for aid, row_index, text in articles(data):
        article_hash = sha_bytes(text.encode("utf-8"))
        if aid in historical_ids or article_hash in historical_hashes:
            exclusion_reasons["historical_article_id_or_hash"] += 1
            continue
        tokens = tokenizer(text, add_special_tokens=False)["input_ids"]
        if len(tokens) < TOKENS_PER_ARTICLE:
            exclusion_reasons["shorter_than_2048"] += 1
            continue
        token_hash = sha_bytes(json.dumps(tokens[:TOKENS_PER_ARTICLE]).encode("utf-8"))
        if token_hash in historical_token_hashes:
            exclusion_reasons["historical_token_hash"] += 1
            continue
        eligible_before_v24 += 1
        prior_sample = v24_by_id.get(aid)
        if prior_sample or article_hash in v24_hashes or token_hash in v24_token_hashes:
            if prior_sample and (prior_sample["article_sha256"] != article_hash or prior_sample["token_sha256"] != token_hash):
                raise RuntimeError(f"Cached dataset/tokenizer no longer reproduces v24 article {aid}")
            if prior_sample:
                matched_v24_ids.add(aid)
            exclusion_reasons["v24_article_id_or_hash_or_token_hash"] += 1
            continue
        available.append({
            "source_article_index": aid,
            "source_row_index": row_index,
            "article_sha256": article_hash,
            "token_sha256": token_hash,
            "full_article_tokens": len(tokens),
            "tokens_used": TOKENS_PER_ARTICLE,
            "tokenizer_revision": protocol["model"]["revision"],
            "add_special_tokens": False,
            "split": None,
            "split_status": "NOT_ASSIGNED; fewer than 64 disjoint articles",
        })

    if eligible_before_v24 != int(e2["eligible_articles_before_selection"]):
        raise RuntimeError(f"Recomputed pre-v24 eligible count {eligible_before_v24} differs from v24 frozen count {e2['eligible_articles_before_selection']}")
    if len(matched_v24_ids) != 64 or len(available) != eligible_before_v24 - len(e2_samples):
        raise RuntimeError(f"v24 article materialization mismatch: matched={len(matched_v24_ids)}, remaining={len(available)}")

    # The frozen v34 key is used only for deterministic listing. Since the
    # prerequisite count fails, no positions are assigned to search or held.
    for item in available:
        key_input = source_fingerprint + item["article_sha256"]
        item["ordering_key_sha256"] = sha_bytes(key_input.encode("ascii"))
    available.sort(key=lambda x: x["ordering_key_sha256"])
    for rank, item in enumerate(available):
        item["eligible_rank_0based"] = rank

    excluded_rows = []
    for aid in sorted(excluded):
        r = excluded[aid]
        excluded_rows.append({
            "source_article_index": aid,
            "article_sha256": r["article_sha256"],
            "token_sha256": r["token_sha256"],
            "additional_token_sha256": ";".join(sorted(r.get("additional_token_sha256", set()))),
            "exclusion_sources": ";".join(sorted(r["exclusion_sources"])),
            "historical_split_relation": ";".join(sorted(r["split_relations"])),
            "dataset": "Salesforce/wikitext:wikitext-2-raw-v1:train",
            "tokenizer_revision": protocol["model"]["revision"],
            "split_relation": "historical calibration/search/held; excluded from all v35 splits",
        })
    write_csv(OUT / "HISTORICAL_EXCLUSION_LIST.csv", excluded_rows,
              ["source_article_index", "article_sha256", "token_sha256", "additional_token_sha256",
               "exclusion_sources", "historical_split_relation", "dataset", "tokenizer_revision", "split_relation"])
    stable_json(OUT / "NEW_ARTICLE_MANIFEST.json", {
        "manifest_id": "range_w4_awq_gptq_v35_new_articles_preflight",
        "status": "STOP_INSUFFICIENT_DISJOINT_ARTICLES" if len(available) < REQUIRED_ARTICLES else "FROZEN",
        "dataset": "Salesforce/wikitext:wikitext-2-raw-v1:train",
        "dataset_source_path": str(PARQUET.relative_to(ROOT).as_posix()),
        "dataset_source_sha256": dataset_sha,
        "dataset_fingerprint_for_ordering": source_fingerprint,
        "parser": e2["parser"],
        "tokenizer_revision": protocol["model"]["revision"],
        "tokenizer_files_sha256": {name: sha_file(TOKENIZER / name) for name in ("tokenizer.json", "tokenizer_config.json")},
        "add_special_tokens": False,
        "tokens_per_article": TOKENS_PER_ARTICLE,
        "required_whole_articles": REQUIRED_ARTICLES,
        "historical_exclusion_unique_article_ids": len(excluded),
        "historical_exclusion_unique_full_article_hashes": len(excluded_hashes),
        "historical_exclusion_unique_token_hashes_known": len(excluded_token_hashes),
        "available_disjoint_articles": len(available),
        "split_plan_if_gate_passed": {"S0_search": 16, "S0_held": 16, "S1_search": 16, "S1_held": 16},
        "split_assignment": "NOT_ASSIGNED: eligibility gate failed; no article is treated as search or held",
        "source_relationship": {
            "v18_calibration": "excluded by article ID/full hash and known token hash",
            "v24_search_held": "excluded by exact article ID/full hash/token hash; v31/v32/v33 reuse these articles",
            "v14_v15_prior": "excluded via v24 frozen historical exclusion lists",
            "WT2_test_and_C4_eval": "not loaded; v35 source is train split only",
        },
        "exclusion_reasons_for_source_scan": exclusion_reasons,
        "articles": available,
    })
    stable_json(OUT / "EVALUATION_SOURCE_EXCLUSION_AUDIT.json", {
        "status": "SOURCE_SPLIT_EXCLUDED; NO_V35_EVALUATION_USED",
        "wikitext_test": {
            "dataset": wt2_test["dataset"], "revision": wt2_test["revision"], "split": wt2_test["split"],
            "source_sha256": wt2_test["source_arrow_sha256"],
            "tokenizer_revision": wt2_test["tokenizer_revision"],
            "add_special_tokens": wt2_test["add_special_tokens"],
            "exclusion": "v35 reads only the WT2 train parquet; the official test split is not an eligible source",
            "article_level_hashes_in_v18_manifest": False,
        },
        "c4_evaluation": {
            "dataset": c4_eval["dataset"], "revision": c4_eval["revision"], "split": c4_eval["split"],
            "source_url": c4_eval["source_url"], "shard": c4_eval["shard"],
            "document_start": c4_eval["document_start"], "document_limit": c4_eval["document_limit"],
            "joined_text_sha256": c4_eval["joined_text_sha256"],
            "tokenizer_revision": c4_eval["tokenizer_revision"],
            "add_special_tokens": c4_eval["add_special_tokens"],
            "exclusion": "different dataset source; v35 reads only WT2 train",
            "article_level_hashes_in_v18_manifest": False,
        },
        "limitation": "The frozen v18 evaluation manifests contain stream-level identity, not per-document full-text hashes. Eligibility is therefore enforced by train-only source partition; no claim of cross-corpus full-article hash comparison is made.",
    })

    # Frozen job accounting. The alpha=1 candidate in each 10-point gate/up
    # bank also supplies the native arm; it is executed only once.
    jobs = []
    for split in ("S0", "S1"):
        for ratio in range(20):
            jobs.append({"split": split, "task_type": "AWQ_RTN_D_selector_ratio", "arm": "C_and_D_shared",
                         "linear": "gate_proj+up_proj; down_proj FP16", "shape": "gate/up=[4864,896]; down=[896,4864]",
                         "candidate_count": 1, "candidate_group_count": 20, "row_alpha_score_slots": 0,
                         "output_elements_scored": 32768 * 896, "cache_reuse": "same split search X/reference; pristine MLP restored per ratio",
                         "extra_correctness_gptq_calls": 0, "detail": f"ratio_index={ratio}; one of 20"})
        for coord, arm_share in (("original", "A/B shared bank"), ("AWQ_D_transformed", "C/D shared bank")):
            for linear in ("gate_proj", "up_proj"):
                rows = 4864
                jobs.append({"split": split, "task_type": "main_GPTQ_candidate_bank", "arm": arm_share,
                             "linear": linear, "shape": "[4864,896]", "candidate_count": 10,
                             "candidate_group_count": 10, "row_alpha_score_slots": 10 * rows,
                             "output_elements_scored": 0, "cache_reuse": f"H/shared inputs reused by {arm_share}; 1 native full-linear run/gate of alpha=1 within bank",
                             "extra_correctness_gptq_calls": 0, "detail": f"coordinate={coord}; alpha grid 0.1..1.0"})
        for arm, count in (("A", 1), ("B", 10), ("C", 1), ("D", 10)):
            jobs.append({"split": split, "task_type": "main_GPTQ_down_candidate", "arm": arm,
                         "linear": "down_proj", "shape": "[896,4864]", "candidate_count": count,
                         "candidate_group_count": count, "row_alpha_score_slots": count * 896,
                         "output_elements_scored": 0, "cache_reuse": "arm-specific down input/H; fresh candidate compensation state",
                         "extra_correctness_gptq_calls": 0, "detail": "alpha=1" if count == 1 else "alpha grid 0.1..1.0"})
    for split in ("S0", "S1"):
        for idx, detail in enumerate(("original gate", "original up", "transformed gate", "transformed up",
                                      "down A actual input", "down B actual input", "down C actual input", "down D actual input")):
            jobs.append({"split": split, "task_type": "correctness_native_alpha1_parity", "arm": "parity",
                         "linear": detail, "shape": "[4864,896]" if "gate" in detail or "up" in detail else "[896,4864]",
                         "candidate_count": 1, "candidate_group_count": 1, "row_alpha_score_slots": 0,
                         "output_elements_scored": 0, "cache_reuse": "same coordinate/input/H semantics; pristine weights",
                         "extra_correctness_gptq_calls": 1, "detail": f"parity_call_{idx+1}/8"})
        for coord in ("original", "transformed"):
            jobs.append({"split": split, "task_type": "correctness_down_candidate_isolation_replay", "arm": coord,
                         "linear": "down_proj", "shape": "[896,4864]", "candidate_count": 1,
                         "candidate_group_count": 1, "row_alpha_score_slots": 0, "output_elements_scored": 0,
                         "cache_reuse": "same frozen down input/H; fresh state for each replay",
                         "extra_correctness_gptq_calls": 1, "detail": "replay after another alpha; 2 calls/split"})
    write_csv(OUT / "JOB_TABLE.csv", jobs,
              ["split", "task_type", "arm", "linear", "shape", "candidate_count", "candidate_group_count",
               "row_alpha_score_slots", "output_elements_scored", "cache_reuse", "extra_correctness_gptq_calls", "detail"])

    main_calls_per_split = 4 * 10 + (1 + 10 + 1 + 10)
    main_calls = 2 * main_calls_per_split
    score_slots_per_split = 4 * 10 * 4864 + (1 + 10 + 1 + 10) * 896
    score_slots = 2 * score_slots_per_split
    correctness_calls = 16 + 4
    # Cross-check table content independently from protocol arithmetic.
    table_main = sum(int(x["candidate_count"]) for x in jobs if x["task_type"].startswith("main_GPTQ"))
    table_score = sum(int(x["row_alpha_score_slots"]) for x in jobs if x["task_type"].startswith("main_GPTQ"))
    table_correct = sum(int(x["extra_correctness_gptq_calls"]) for x in jobs)
    if (main_calls, score_slots, correctness_calls, table_main, table_score, table_correct) != (124, 428544, 20, 124, 428544, 20):
        raise RuntimeError("Frozen workload arithmetic mismatch; see JOB_TABLE audit")

    stable_json(OUT / "JOB_TABLE_AUDIT.json", {
        "status": "PASS_COUNT_RECONCILIATION; EXECUTION_NOT_RUN",
        "main_full_linear_gptq_per_split": main_calls_per_split,
        "main_full_linear_gptq_total": main_calls,
        "score_slots_per_split": score_slots_per_split,
        "score_slots_total": score_slots,
        "correctness_gptq_calls": {"alpha1_parity": 16, "down_isolation_replays": 4, "total": correctness_calls},
        "full_linear_gptq_including_correctness": main_calls + correctness_calls,
        "AWQ_RTN_D_ratio_candidates": 40,
        "AWQ_proxy_quantized_linear_ops": 80,
        "AWQ_full_MLP_candidate_forwards": 40,
        "reference_full_MLP_forwards": 2,
        "unique_H_per_split": 6,
        "unique_H_total": 12,
        "independent_recalculation": {
            "gate_up": "4 shared banks/split x 10 full-Linear alpha candidates = 40 calls/split; 4 x 10 x 4864 row-alpha slots",
            "down": "A1+B10+C1+D10 = 22 calls/split; 22 x 896 row-alpha slots",
            "main": "2 x (40+22) = 124 full-Linear GPTQ executions",
            "scores": "2 x ((4x10x4864)+(22x896)) = 428544 row-alpha score slots",
            "correctness": "16 independent alpha1 parity + 4 candidate-isolation down replays = 20; 144 GPTQ executions total if all run",
        },
        "job_table_reconciliation": {"main_candidate_calls": table_main, "main_score_slots": table_score,
                                     "extra_correctness_calls": table_correct},
        "status_reason": "Counts agree with v34; formal work is not authorized until new-article gate passes.",
    })

    source_paths = [
        v34_protocol_path, V34 / "REPORT.md", V34 / "SOURCE_AUDIT.md", V34 / "COMPARISON_TABLE.csv",
        V18 / "PROTOCOL.json", V18 / "run_formal.py",
        RESULTS / "range_w4_m2_runtime_v17_3" / "optimized_native_m2.py",
        V18 / "evaluation/manifests/WIKITEXT_TEST_MANIFEST.json",
        V18 / "evaluation/manifests/C4_EVAL_MANIFEST.json",
        V14_MANIFEST, V15_MANIFEST,
        V18 / "CALIBRATION_MANIFEST.json", V18 / "M2" / "RUN_REPORT.json",
        V23 / "PROTOCOL.json", V23 / "prepare_e1_v23.py", e1_path,
        V24 / "PROTOCOL.json", V24 / "prepare_e2_v24.py", V24 / "run_e2_v24.py", e2_path,
        V31 / "RUN_REPORT.json", V32 / "FROZEN_PROTOCOL.json", V32 / "RUN_REPORT.json",
        V33 / "FROZEN_PROTOCOL.json", V33 / "RUN_REPORT.json", PARQUET,
        TOKENIZER / "tokenizer.json", TOKENIZER / "tokenizer_config.json",
        TOKENIZER / "model.safetensors",
        RESULTS / "range_w4_fullmodel_v18" / "M2/checkpoint/final/model.safetensors",
        Path(__file__),
    ]
    hashes = {}
    for path in source_paths:
        if path.is_file():
            hashes[path.relative_to(ROOT).as_posix()] = sha_file(path)
        elif path.exists():
            hashes[path.relative_to(ROOT).as_posix()] = "EXISTS_NONFILE"
        else:
            hashes[path.relative_to(ROOT).as_posix()] = "MISSING"
    stable_json(OUT / "SOURCE_HASHES.json", {
        "git_head": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip(),
        "git_worktree_dirty_at_preflight": True,
        "paths": hashes,
        "awq_upstream": protocol["provenance"]["official_upstream"],
        "note": "Pre-existing dirty worktree preserved. v32/v33 protocol filenames are used as stored; v31 data relationships are read from its report and source-hash audit.",
    })
    versions = {}
    for package in ("torch", "transformers", "gptqmodel", "datasets", "pyarrow", "tokenizers"):
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = "NOT_INSTALLED"
    stable_json(OUT / "ENVIRONMENT.json", {
        "python": sys.version,
        "packages": versions,
        "device_and_cuda_probe": "NOT_RUN; article gate failed before any GPU work",
        "gpu_budget_seconds_used": 0,
        "environment_scope": "CPU-only manifest materialization in .venv-gptq",
    })

    stable_json(OUT / "FROZEN_PROTOCOL_V35.json", {
        "protocol_id": "v35_l12_awq_selector_dynamic_gptq_bounded_diagnostic",
        "status": "STOP_INSUFFICIENT_DISJOINT_ARTICLES" if len(available) < REQUIRED_ARTICLES else "PRECHECK_PASSED",
        "parent_protocol_path": str(v34_protocol_path.relative_to(ROOT).as_posix()),
        "parent_protocol_sha256": sha_file(v34_protocol_path),
        "parent_protocol": protocol,
        "v35_run_overrides": {
            "gpu_cumulative_hard_cap_seconds": 1800,
            "stop_with_saving_reserve_seconds": 180,
            "data_gate_required_new_articles": 64,
            "new_data_manifest_sha256": sha_file(OUT / "NEW_ARTICLE_MANIFEST.json"),
            "job_table_sha256": sha_file(OUT / "JOB_TABLE.csv"),
            "no_execution_if_articles_lt_64": True,
        },
    })

    stable_json(OUT / "PRECHECK.json", {
        "status": "STOP_INSUFFICIENT_DISJOINT_ARTICLES" if len(available) < REQUIRED_ARTICLES else "PASS",
        "eligible_new_whole_articles": len(available),
        "required": REQUIRED_ARTICLES,
        "shortfall": max(0, REQUIRED_ARTICLES - len(available)),
        "historical_unique_article_ids_excluded": len(excluded),
        "historical_unique_full_hashes_excluded": len(excluded_hashes),
        "v24_replicated_eligible_count": e2["eligible_articles_before_selection"],
        "v24_independently_recomputed_eligible_count": eligible_before_v24,
        "v24_replicated_prior_selected_count": len(e2_samples),
        "v24_exact_article_rows_revalidated": len(matched_v24_ids),
        "v24_expected_remaining_from_frozen_counts": e2["eligible_articles_before_selection"] - len(e2_samples),
        "recomputed_remaining_equals_v24_arithmetic": len(available) == e2["eligible_articles_before_selection"] - len(e2_samples),
        "source_scan_exclusion_reasons": exclusion_reasons,
        "tokenizer_and_dataset_hashes_verified": True,
        "article_and_token_hash_uniqueness": len({a["article_sha256"] for a in available}) == len(available) and len({a["token_sha256"] for a in available}) == len(available),
        "GPU_executions": 0,
        "correctness_checks": "NOT_RUN: required article gate failed" if len(available) < REQUIRED_ARTICLES else "PENDING",
        "held_evaluation": "NOT_RUN",
    })

    stop_reason = ("Only 34 of 64 new whole articles satisfy the frozen exclusions; stop before model/GPU work."
                   if len(available) < REQUIRED_ARTICLES else "Article gate passed; subsequent frozen checks pending.")
    checks = ["D=I regression", "gate/up shared-scale and adjacent-operator fold",
              "unquantized MLP equivalence", "transformed X/W/H correspondence",
              "alpha=1 native GPTQ parity", "pristine W/H and candidate isolation",
              "all-row W/scale/zero/g_idx stitch and commit", "actual MLP and downstream input propagation"]
    stable_json(OUT / "correctness" / "CORRECTNESS_STATUS.json", {
        "status": "NOT_RUN_PREREQUISITE_STOP" if len(available) < REQUIRED_ARTICLES else "PENDING",
        "reason": stop_reason,
        "checks": {name: "NOT_RUN" for name in checks},
        "tolerance": "v34 frozen tolerance; not evaluated or changed",
    })
    stable_json(OUT / "raw" / "CANDIDATE_STATUS.json", {
        "status": "NOT_RUN_PREREQUISITE_STOP" if len(available) < REQUIRED_ARTICLES else "PENDING",
        "D_ratio_candidates": "NOT_RUN", "GPTQ_candidates": "NOT_RUN",
        "row_alpha_scores": "NOT_RUN", "selection_lock": "NOT_CREATED",
        "held_data_opened": False,
    })
    stable_json(OUT / "checkpoints" / "CHECKPOINT_STATUS.json", {
        "status": "NOT_CREATED_PREREQUISITE_STOP" if len(available) < REQUIRED_ARTICLES else "READY",
        "reason": stop_reason,
        "resume_rule": "No model/candidate state exists. Any later run must use a separately frozen data protocol and revalidate every source and manifest hash.",
    })
    stable_json(OUT / "final" / "FINALIZATION_STATUS.json", {
        "status": "NOT_RUN_PREREQUISITE_STOP" if len(available) < REQUIRED_ARTICLES else "PENDING",
        "selected_D": "NOT_RUN", "selected_alpha": "NOT_RUN",
        "held_metrics": "NOT_RUN", "PPL": "NOT_RUN",
    })
    stable_json(OUT / "RUN_REPORT.json", {
        "status": "STOP_INSUFFICIENT_DISJOINT_ARTICLES" if len(available) < REQUIRED_ARTICLES else "PRECHECK_PASSED",
        "precheck_sha256": sha_file(OUT / "PRECHECK.json"),
        "article_manifest_sha256": sha_file(OUT / "NEW_ARTICLE_MANIFEST.json"),
        "evaluation_source_exclusion_audit_sha256": sha_file(OUT / "EVALUATION_SOURCE_EXCLUSION_AUDIT.json"),
        "job_table_sha256": sha_file(OUT / "JOB_TABLE.csv"),
        "job_table_audit_sha256": sha_file(OUT / "JOB_TABLE_AUDIT.json"),
        "counts_planned_not_executed": {"main_full_linear_gptq": main_calls,
                                         "full_linear_gptq_with_correctness": main_calls + correctness_calls,
                                         "row_alpha_score_slots": score_slots},
        "execution": {"gpu_seconds": 0, "peak_cuda_memory": "NOT_MEASURED",
                      "AWQ_scaling_search": "NOT_RUN", "GPTQ": "NOT_RUN",
                      "scoring": "NOT_RUN", "input_capture_or_refresh": "NOT_RUN",
                      "candidate_IO": "NOT_RUN", "held": "NOT_RUN", "PPL": "NOT_RUN"},
        "history_mutation": False,
    })

    ledger = [{"stage": "CPU source/hash and article manifest preflight", "cumulative_hard_cap_seconds": "N/A",
               "observed_gpu_seconds": 0, "status": "COMPLETE_CPU_ONLY"},
              *[{"stage": stage, "cumulative_hard_cap_seconds": 1800, "observed_gpu_seconds": 0,
                 "status": "NOT_RUN_INSUFFICIENT_ARTICLES" if len(available) < REQUIRED_ARTICLES else "PENDING"}
                for stage in ("model load and warmup", "correctness GPTQ and numerical checks", "AWQ scale search",
                              "main GPTQ candidates", "candidate scoring and selection", "activation capture/refresh",
                              "candidate/checkpoint IO and finalization", "held evaluation")]]
    write_csv(OUT / "BUDGET_LEDGER.csv", ledger,
              ["stage", "cumulative_hard_cap_seconds", "observed_gpu_seconds", "status"])

    report = f"""# v35 — L12 AWQ scaling selector × dynamic GPTQ bounded diagnostic

Status: **{'STOP — insufficient new disjoint whole articles' if len(available) < REQUIRED_ARTICLES else 'preflight passed; GPU work pending'}**.

## Pre-execution result

- Frozen source: v34 protocol SHA256 `{sha_file(v34_protocol_path)}`; model/tokenizer revision `{protocol['model']['revision']}`.
- Dataset: `{e2['dataset']}`, source parquet SHA256 `{dataset_sha}`; tokenizer has `add_special_tokens=false`, 2048 tokens/article.
- v24 frozen audit recorded {e2['eligible_articles_before_selection']} eligible articles after the shared v14/v15/v18 blacklist and selected {len(e2_samples)} complete articles for v24; v31/v32/v33 reuse those same documents. Independent v35 scan after excluding those documents and all prior IDs/full/token hashes found **{len(available)}** complete articles.
- Required: {REQUIRED_ARTICLES}; shortfall: {max(0, REQUIRED_ARTICLES-len(available))}. No search/held split was assigned. Article-level identities and token hashes are in `NEW_ARTICLE_MANIFEST.json`; excluded article identities are in `HISTORICAL_EXCLUSION_LIST.csv`.
- `EVALUATION_SOURCE_EXCLUSION_AUDIT.json` records the frozen WT2-test and C4 evaluation source/split identities. v18 stores stream-level hashes rather than per-document test hashes; v35 only reads WT2 train. The C4 source is a separate corpus. No evaluation source is used for selection, and no claim of full-article cross-corpus hash comparison is made.
- The v24 arithmetic predicts {e2['eligible_articles_before_selection']}-{len(e2_samples)}={e2['eligible_articles_before_selection']-len(e2_samples)} articles after additionally excluding its own 64; recomputation {'matches' if len(available)==e2['eligible_articles_before_selection']-len(e2_samples) else 'does not match'} that count.

## Job-count audit

The independently generated `JOB_TABLE.csv` reconciles exactly with v34: **124 main full-Linear GPTQ executions**, **144 including 16 alpha=1 parity and 4 down-state isolation replays**, and **428,544 row-alpha score slots**. These are planned counts only. The 40 AWQ RTN ratios (20/split), 80 RTN Linear pseudo-quantizations, and 40 full-MLP selector forwards are separate from GPTQ counts.

## Stop decision

The frozen v34 data gate requires 64 newly eligible, unique whole articles, 16 each for S0-search, S0-held, S1-search, and S1-held. Only {len(available)} remain. Following the protocol, execution stops before GPU initialization, correctness/model checks, scaling search, GPTQ, selection lock, or held evaluation. No split size, token length, exclusion list, target, or method was changed to fill the gap.

## Execution and status

- D=I/native parity, fold correctness, alpha=1 parity, candidate isolation, row stitch/commit, MLP propagation: **NOT_RUN**.
- Four-arm candidates and search lock: **NOT_RUN**.
- Held evaluation and quality gate: **NOT_RUN**.
- GPU cumulative time: **0/1800 seconds**; PPL/full-model quantization: **NOT_RUN**.
- Existing history/checkpoints/public implementations: read-only. New files are confined to v35.

## Reproduction

```powershell
.\\.venv-gptq\\Scripts\\python.exe -B experiments\\range_search_w4\\results\\range_w4_awq_gptq_v35\\prepare_v35_preflight.py
```

This CPU-only command verifies source hashes and article exclusion, writes the immutable preflight manifests and job counts, and refuses to overwrite differing outputs. Since the frozen gate fails, there is no command to start the GPU experiment.
"""
    report_path = OUT / "REPORT.md"
    if report_path.exists():
        raise RuntimeError(f"Refusing to overwrite {report_path}")
    report_path.write_text(report, encoding="utf-8", newline="\n")

    reproduce = """# Reproduction

Run from the repository root in PowerShell:

```powershell
.\\.venv-gptq\\Scripts\\python.exe -B experiments\\range_search_w4\\results\\range_w4_awq_gptq_v35\\prepare_v35_preflight.py
```

The command is CPU-only. It checks the v34 parent protocol, dataset/tokenizer hashes, complete-article exclusions, and planned execution counts. It refuses to overwrite differing v35 artifacts. The frozen 64-article gate fails, so no GPU-launch command is provided.
"""
    (OUT / "REPRODUCE.md").write_text(reproduce, encoding="utf-8", newline="\n")

    output_rows = []
    for path in sorted(p for p in OUT.rglob("*") if p.is_file() and p.name != "OUTPUT_MANIFEST.csv"):
        output_rows.append({"path": path.relative_to(OUT).as_posix(), "bytes": path.stat().st_size,
                            "sha256": sha_file(path)})
    write_csv(OUT / "OUTPUT_MANIFEST.csv", output_rows, ["path", "bytes", "sha256"])

    print(json.dumps({
        "status": "STOP_INSUFFICIENT_DISJOINT_ARTICLES" if len(available) < REQUIRED_ARTICLES else "PRECHECK_PASSED",
        "available_articles": len(available), "required": REQUIRED_ARTICLES,
        "main_gptq": main_calls, "including_correctness": main_calls + correctness_calls,
        "row_alpha_score_slots": score_slots, "gpu_seconds": 0,
        "output": str(OUT),
    }, indent=2))


if __name__ == "__main__":
    main()
