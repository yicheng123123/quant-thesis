"""Materialize the preregistered E1 complete-document split once from local WT2 train."""
from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
RESULTS = ROOT / "experiments/range_search_w4/results"
V14 = RESULTS / "range_w4_calibration_coverage_v14/article_manifest.json"
V15 = RESULTS / "range_w4_quantized_prefix_v15/article_manifest.json"
V18 = RESULTS / "range_w4_fullmodel_v18/CALIBRATION_MANIFEST.json"
MANIFEST = OUT / "E1_DOCUMENT_MANIFEST.json"
PARQUET = ROOT / ".hf_cache/hub/datasets--Salesforce--wikitext/snapshots/b08601e04326c79dfdd32d625aee71d232d685c3/wikitext-2-raw-v1/train-00000-of-00001.parquet"
TOKENIZER = ROOT / ".hf_cache/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775"
SELECTION_DOMAIN = "v22-next-diagnostic-20261022:"
N_TOKENS = 2048


def sha_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()


def records(obj):
    found = []
    def visit(x):
        if isinstance(x, dict):
            if "source_article_index" in x and x.get("article_sha256"):
                found.append({"source_article_index": int(x["source_article_index"]),
                              "article_sha256": x["article_sha256"],
                              "token_sha256": x.get("token_sha256")})
            for v in x.values(): visit(v)
        elif isinstance(x, list):
            for v in x: visit(v)
    visit(obj)
    return found


def articles(ds):
    current, start, idx = [], None, 0
    for row_index, record in enumerate(ds):
        line = record["text"]
        if re.match(r"^= [^=].* =$", line.strip()) and ";" not in line:
            if current:
                yield idx, start, "\n".join(current)
                idx += 1
            current, start = [line], row_index
        elif current:
            current.append(line)
    if current:
        yield idx, start, "\n".join(current)


def main():
    if MANIFEST.exists():
        raise RuntimeError(f"Frozen manifest already exists; refusing to resample/overwrite: {MANIFEST}")
    if not all(p.is_file() for p in (V14, V15, V18, PARQUET)) or not TOKENIZER.is_dir():
        raise FileNotFoundError("A frozen source manifest, cached train parquet, or local tokenizer is missing")

    v14 = json.loads(V14.read_text(encoding="utf-8"))
    v15 = json.loads(V15.read_text(encoding="utf-8"))
    v18 = json.loads(V18.read_text(encoding="utf-8"))
    prior = (records(v14.get("historical_blacklist", [])) +
             records(v14.get("calibration", [])) + records(v14.get("heldout", [])) +
             records(v15.get("calibration", [])) + records(v15.get("heldout", [])) +
             records(v18.get("samples", [])))
    by_id = {}
    by_hash = set()
    for r in prior:
        aid = r["source_article_index"]
        old = by_id.get(aid)
        if old and old["article_sha256"] != r["article_sha256"]:
            raise RuntimeError(f"Historical ID/hash conflict at article {aid}")
        by_id[aid] = r
        by_hash.add(r["article_sha256"])
    excluded_ids, excluded_hashes = set(by_id), by_hash

    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer
    # Read the exact cached train parquet directly. This avoids datasets trying
    # to create its default cache under the user's read-only profile.
    ds = pq.read_table(PARQUET, columns=["text"]).to_pylist()
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER), local_files_only=True)

    eligible = []
    for article_id, source_row, text in articles(ds):
        if article_id in excluded_ids:
            continue
        article_hash = sha_bytes(text.encode("utf-8"))
        if article_hash in excluded_hashes:
            continue
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        if len(ids) < N_TOKENS:
            continue
        ids = ids[:N_TOKENS]
        token_hash = sha_bytes(json.dumps(ids).encode("utf-8"))
        eligible.append({
            "source_article_index": article_id,
            "source_row_index": source_row,
            "article_sha256": article_hash,
            "full_article_tokens": len(tokenizer(text, add_special_tokens=False)["input_ids"]),
            "token_ids": ids,
            "token_sha256": token_hash,
            "selection_key_sha256": sha_bytes((SELECTION_DOMAIN + article_hash).encode("ascii")),
        })

    eligible.sort(key=lambda a: a["selection_key_sha256"])
    if len(eligible) < 64:
        raise RuntimeError(f"Frozen eligibility has {len(eligible)} whole articles; 64 required. STOP; do not resample.")
    chosen = eligible[:64]
    assert len({x["source_article_index"] for x in chosen}) == 64
    assert len({x["article_sha256"] for x in chosen}) == 64
    assert len({x["token_sha256"] for x in chosen}) == 64
    assert not ({x["source_article_index"] for x in chosen} & excluded_ids)
    assert not ({x["article_sha256"] for x in chosen} & excluded_hashes)
    split_membership = {
        **{i: ("S0_search" if i < 16 else "S0_held") for i in range(32)},
        **{i: ("S1_search" if i < 48 else "S1_held") for i in range(32, 64)},
    }
    for order, item in enumerate(chosen):
        item["sorted_eligible_order"] = order
        item["split"] = split_membership[order]
        del item["selection_key_sha256"]
    ordered = [tok for item in chosen for tok in item["token_ids"]]
    digest_i64le = hashlib.sha256(b"".join(int(t).to_bytes(8, "little", signed=True) for t in ordered)).hexdigest()
    payload = {
        "manifest_id": "range_w4_block_objective_e1_v23_documents",
        "selection_rule_source": "v22 MINIMAL_DISCRIMINATING_EXPERIMENTS.md",
        "selection_domain": SELECTION_DOMAIN,
        "dataset": "Salesforce/wikitext:wikitext-2-raw-v1:train",
        "dataset_split": "train only; WT2 test and C4 evaluation splits were not loaded",
        "dataset_source": PARQUET.relative_to(ROOT).as_posix(),
        "dataset_source_sha256": sha_file(PARQUET),
        "parser": "v14/v15 full article parser; heading regex ^=[space][^=].*[space]=$ and semicolon headings excluded",
        "tokenizer_revision": "7ae557604adf67be50417f59c2c2f167def9a775",
        "tokenizer_path_sha256": {p.name: sha_file(p) for p in (TOKENIZER / "tokenizer.json", TOKENIZER / "tokenizer_config.json") if p.is_file()},
        "tokenizer_add_special_tokens": False,
        "tokens_per_document": N_TOKENS,
        "eligible_articles_before_selection": len(eligible),
        "historical_exclusions": {
            "source_manifests": {str(p.relative_to(ROOT).as_posix()): sha_file(p) for p in (V14, V15, V18)},
            "unique_article_ids": len(excluded_ids),
            "unique_full_article_sha256": len(excluded_hashes),
            "article_ids": sorted(excluded_ids),
            "full_article_sha256": sorted(excluded_hashes),
        },
        "split_rule": {
            "A": {"search_sorted_positions_0based": [0, 15], "held_sorted_positions_0based": [16, 31]},
            "B": {"search_sorted_positions_0based": [32, 47], "held_sorted_positions_0based": [48, 63]},
            "semantics": "whole documents; no document or token fragments shared between search and held",
        },
        "ordered_token_ids_sha256_i64le": digest_i64le,
        "sample_count": len(chosen),
        "unique_full_articles": len({x["article_sha256"] for x in chosen}),
        "unique_token_sequences": len({x["token_sha256"] for x in chosen}),
        "token_budget": len(ordered),
        "samples": chosen,
    }
    text = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    with MANIFEST.open("x", encoding="utf-8", newline="\n") as f:
        f.write(text)
    print(json.dumps({"status": "FROZEN", "manifest_sha256": sha_file(MANIFEST),
                      "eligible": len(eligible), "selected": len(chosen),
                      "excluded_historical_articles": len(excluded_ids),
                      "search_docs": 32, "held_docs": 32,
                      "ordered_token_ids_sha256_i64le": digest_i64le}, indent=2))


if __name__ == "__main__":
    main()
