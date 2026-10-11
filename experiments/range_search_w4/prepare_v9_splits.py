"""Freeze ten disjoint article-level splits before v9 quantization runs."""
from __future__ import annotations

import hashlib
import json
import random
import re
from pathlib import Path

from experiments.range_search_w4.run import ROOT, SNAPSHOT
from datasets import load_dataset
from transformers import AutoTokenizer

HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "results" / "range_w4_l23r448_v9"
V8 = HERE / "results" / "range_w4_dynamic_v8"
SEED = 42
N_SPLITS = 10
ARTICLES_PER_SPLIT = 8
N_POOL = N_SPLITS * ARTICLES_PER_SPLIT


def article_candidates():
    """Yield the exact v7 top-level article rule, in WikiText train order."""
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train",
                      cache_dir=str(ROOT / ".hf_cache" / "datasets"))
    current, start, source_index = [], None, 0
    for row_index, record in enumerate(ds):
        line = record["text"]
        if re.match(r"^= [^=].* =$", line.strip()) and ";" not in line:
            if current:
                yield source_index, start, "\n".join(current)
                source_index += 1
            current, start = [line], row_index
        elif current:
            current.append(line)
    if current:
        yield source_index, start, "\n".join(current)


def prepare() -> tuple[dict, dict]:
    tokenizer = AutoTokenizer.from_pretrained(SNAPSHOT, local_files_only=True)
    eligible = []
    for source_index, source_row_index, article in article_candidates():
        ids = tokenizer(article, add_special_tokens=False)["input_ids"]
        if len(ids) >= 128:
            ids = ids[:128]
            eligible.append({"eligible_index": len(eligible),
                             "source_article_index": source_index,
                             "source_row_index": source_row_index,
                             "article_sha256": hashlib.sha256(article.encode()).hexdigest(),
                             "token_sha256": hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
                             "token_ids": ids})
        if len(eligible) == 8 + N_POOL:
            break
    if len(eligible) != 8 + N_POOL:
        raise RuntimeError(f"need {8 + N_POOL} eligible articles; found {len(eligible)}")
    v8 = json.loads((V8 / "manifest.json").read_text(encoding="utf-8"))
    if any(a["article_sha256"] != b["article_sha256"] or
           a["token_sha256"] != b["sha256"] for a, b in zip(eligible[:8], v8["documents"])):
        raise RuntimeError("article parser does not reproduce the frozen v8 first eight")
    pool = eligible[8:]
    random.Random(SEED).shuffle(pool)
    splits = []
    for split_id in range(N_SPLITS):
        members = pool[split_id * 8:(split_id + 1) * 8].copy()
        sub_seed = int.from_bytes(hashlib.sha256(f"{SEED}:split:{split_id}".encode()).digest()[:8], "big")
        random.Random(sub_seed).shuffle(members)
        splits.append({"split_id": f"S{split_id:02d}", "sub_seed": sub_seed,
                       "search": members[:4], "heldout": members[4:]})
    all_docs = [doc for split in splits for key in ("search", "heldout") for doc in split[key]]
    if len({doc["source_article_index"] for doc in all_docs}) != N_POOL:
        raise RuntimeError("split articles are not globally disjoint")
    if len({doc["token_sha256"] for doc in all_docs}) != N_POOL:
        raise RuntimeError("token sequences repeat")
    manifest = {"id": "range_w4_l23r448_v9_splits", "seed": SEED,
        "selection_rule": "same v7 full-article parser; first 88 eligible >=128-token train articles; exclude first eight v8 articles; shuffle next 80 with Random(42), partition sequentially into ten groups of eight; shuffle each group with stored SHA256-derived sub_seed; first four search, last four heldout",
        "dataset": "Salesforce/wikitext:wikitext-2-raw-v1:train",
        "tokens_per_article": 128, "n_splits": N_SPLITS,
        "v8_reference_token_hashes": [d["sha256"] for d in v8["documents"]],
        "splits": splits}
    protocol = {"id": "range_w4_l23r448_v9", "model": "Qwen/Qwen2.5-0.5B-Instruct",
        "revision": v8["source_protocol"]["model_revision"], "model_dtype": "float16",
        "math_dtype": "float32", "device": "cuda", "layer": 23,
        "projection": "self_attn.q_proj", "row": 448,
        "bits": 4, "activation_bits": 16, "symmetric": True,
        "integer_range": [0, 15], "zero": 8, "rounding": "torch.round ties to even",
        "group_size": 128, "block_size": 128, "damp_percent": 0.1,
        "desc_act": False, "static_groups": False, "true_sequential": True,
        "alpha_grid": [round(i / 10, 1) for i in range(1, 11)],
        "D": "alpha=1 with native dynamic G128 find_params timing",
        "C": "search-output-MSE-selected shared alpha, each candidate full pristine-W dynamic GPTQ rerun",
        "loss": "FP32 output SSE divided by token-output element count; FP16 dequantized weight",
        "input_semantics": "FP model q_proj inputs, no quantized prefix",
        "dataset": manifest["dataset"], "n_splits": 10,
        "articles_per_split": {"search": 4, "heldout": 4},
        "tokens_per_article": 128, "seed": SEED,
        "regression_tolerance_relative": 1e-4}
    return manifest, protocol


def main() -> None:
    manifest, protocol = prepare()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for name, value in (("splits_manifest.json", manifest), ("protocol.json", protocol)):
        path = OUTPUT / name
        content = json.dumps(value, ensure_ascii=False, indent=2) + "\n"
        if path.exists():
            if path.read_text(encoding="utf-8") != content:
                raise RuntimeError(f"frozen file differs, refusing overwrite: {path}")
        else:
            path.write_text(content, encoding="utf-8")
    print(f"FROZEN {len(manifest['splits'])} splits, {N_POOL} globally distinct v9 articles, "
          f"manifest={OUTPUT / 'splits_manifest.json'}")
    for split in manifest["splits"]:
        print(split["split_id"], "search", [d["source_article_index"] for d in split["search"]],
              "heldout", [d["source_article_index"] for d in split["heldout"]])


if __name__ == "__main__":
    main()
