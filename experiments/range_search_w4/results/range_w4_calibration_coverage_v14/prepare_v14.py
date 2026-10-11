"""Freeze new v14 rows and WikiText articles before any quantization run."""
from __future__ import annotations

import hashlib
import json
import os
import random
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
os.environ.update({"HF_HOME": str(ROOT / ".hf_cache"),
                   "HF_HUB_CACHE": str(ROOT / ".hf_cache" / "hub"),
                   "HF_DATASETS_OFFLINE": "1", "HF_HUB_OFFLINE": "1"})

import torch
from datasets import load_dataset
from safetensors import safe_open
from transformers import AutoTokenizer

OUT = Path(__file__).resolve().parent
RESULTS = OUT.parent
REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"
LAYERS = (2, 6, 14, 18)
PROJECTIONS = ("q_proj", "o_proj", "up_proj", "down_proj")
BUCKETS = ("low", "mid", "high")
ROW_SEED = 20261002
ARTICLE_SEED = 20261003
K_VALUES = (4, 8, 16, 32)
SNAPSHOT = ROOT / ".hf_cache" / "hub" / "models--Qwen--Qwen2.5-0.5B-Instruct" / "snapshots" / REVISION
HIST = {
    "v8": RESULTS / "range_w4_dynamic_v8" / "manifest.json",
    "v9": RESULTS / "range_w4_l23r448_v9" / "splits_manifest.json",
    "v10": RESULTS / "range_w4_sixrow_v10" / "manifest_reference.json",
    "v11": RESULTS / "range_w4_96row_v11" / "manifest_reference.json",
    "v12": RESULTS / "range_w4_gate_v12" / "validation_split_manifest.json",
}
TREE_DIRS = {f"v{i}": RESULTS / f"range_w4_{name}" for i, name in [
    (8, "dynamic_v8"), (9, "l23r448_v9"), (10, "sixrow_v10"),
    (11, "96row_v11"), (12, "gate_v12"), (13, "failure_mechanism_v13")]}


def sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def tensor_hash(t: torch.Tensor) -> str:
    return sha_bytes(t.detach().contiguous().cpu().numpy().tobytes())


def frozen_write(path: Path, obj: dict) -> str:
    payload = json.dumps(obj, ensure_ascii=False, indent=2) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != payload:
            raise RuntimeError(f"frozen artifact already exists with different content: {path}")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(payload, encoding="utf-8", newline="\n")
    return sha_file(path)


def row_tensor_key(layer: int, projection: str) -> str:
    path = "self_attn" if projection in ("q_proj", "o_proj") else "mlp"
    return f"model.layers.{layer}.{path}.{projection}.weight"


def row_manifest() -> dict:
    config = json.loads((SNAPSHOT / "config.json").read_text(encoding="utf-8"))
    hidden, inter = int(config["hidden_size"]), int(config["intermediate_size"])
    dims = {"q_proj": (hidden, hidden), "o_proj": (hidden, hidden),
            "up_proj": (inter, hidden), "down_proj": (hidden, inter)}
    rng = random.Random(ROW_SEED)
    rows = []
    for layer in LAYERS:
        for projection in PROJECTIONS:
            out_dim, in_dim = dims[projection]
            bounds = ((0, out_dim // 3), (out_dim // 3, (2 * out_dim) // 3), ((2 * out_dim) // 3, out_dim))
            for bucket, (start, stop) in zip(BUCKETS, bounds):
                index = rng.randrange(start, stop)
                rows.append({"layer": layer, "projection": projection,
                             "projection_name": (f"self_attn.{projection}" if projection in ("q_proj", "o_proj")
                                                 else f"mlp.{projection}"),
                             "row_index": index, "input_dim": in_dim, "output_dim": out_dim,
                             "bucket": bucket, "bucket_start_inclusive": start,
                             "bucket_stop_exclusive": stop, "seed": ROW_SEED})
    model_file = SNAPSHOT / "model.safetensors"
    with safe_open(str(model_file), framework="pt", device="cpu") as sf:
        for item in rows:
            x = sf.get_slice(row_tensor_key(item["layer"], item["projection"]))[item["row_index"], :]
            item["weight_hash"] = tensor_hash(x.to(torch.float16).float().contiguous())
            item["checkpoint_tensor_dtype"] = str(x.dtype)
            item["effective_model_dtype"] = "float16"
    return {"manifest_id": "range_w4_calibration_coverage_v14_rows",
            "model": "Qwen/Qwen2.5-0.5B-Instruct", "model_revision": REVISION,
            "checkpoint_sha256": sha_file(model_file),
            "config_sha256": sha_file(SNAPSHOT / "config.json"),
            "layers": list(LAYERS), "projections": list(PROJECTIONS),
            "rows_per_layer_projection": 3, "bucket_definition": "three contiguous near-equal output-index intervals; low=[0,floor(d/3)), mid=[floor(d/3),floor(2d/3)), high=[floor(2d/3),d)",
            "sampling_seed": ROW_SEED,
            "sampling_order": {"layers": list(LAYERS), "projections": list(PROJECTIONS), "buckets": list(BUCKETS)},
            "sampling_rule": "one row per bucket using one Python random.Random(20261002) stream in manifest order",
            "row_count": len(rows), "rows": rows}


def walk_docs(value):
    found = []
    def visit(x):
        if isinstance(x, dict):
            if "source_article_index" in x and ("article_sha256" in x or "token_sha256" in x):
                found.append({"source_article_index": int(x["source_article_index"]),
                              "article_sha256": x.get("article_sha256"),
                              "token_sha256": x.get("token_sha256", x.get("sha256"))})
            for v in x.values():
                visit(v)
        elif isinstance(x, list):
            for v in x:
                visit(v)
    visit(value)
    unique = {}
    for d in found:
        unique[(d["source_article_index"], d["article_sha256"], d["token_sha256"])] = d
    return list(unique.values())


def historical_blacklist() -> tuple[list[dict], dict]:
    loaded = {name: json.loads(path.read_text(encoding="utf-8")) for name, path in HIST.items()}
    docs = {name: walk_docs(obj) for name, obj in loaded.items()}
    assert len(docs["v8"]) == 8 and len(docs["v9"]) == 80
    assert len(docs["v10"]) == 80 and len(docs["v11"]) == 24 and len(docs["v12"]) == 80
    def tup(x): return (x["source_article_index"], x["article_sha256"], x["token_sha256"])
    assert set(map(tup, docs["v10"])) == set(map(tup, docs["v9"])), "v10 references differ from v9"
    assert set(map(tup, docs["v11"])).issubset(set(map(tup, docs["v9"]))), "v11 article reference not contained in v9"
    assert set(map(tup, docs["v8"])).isdisjoint(set(map(tup, docs["v9"])))
    assert set(map(tup, docs["v12"])).isdisjoint(set(map(tup, docs["v8"] + docs["v9"])))
    # v13 has no new article sample: its protocol points to the v12 frozen pool.
    v13_protocol_path = RESULTS / "range_w4_failure_mechanism_v13" / "PROTOCOL.json"
    v13_protocol = json.loads(v13_protocol_path.read_text(encoding="utf-8"))
    assert v13_protocol["source_file_sha256"]["validation_split_manifest.json"] == sha_file(HIST["v12"])
    merged = {}
    for name in ("v8", "v9", "v12"):
        for d in docs[name]: merged[tup(d)] = d
    old = sorted(merged.values(), key=lambda d: (d["source_article_index"], d["article_sha256"] or ""))
    ids = {d["source_article_index"] for d in old}
    hashes = {h for d in old for h in (d["article_sha256"], d["token_sha256"]) if h}
    assert len(ids) == len(old) == 168
    refs = {name: {"path": str(path.relative_to(ROOT)).replace("\\", "/"), "sha256": sha_file(path),
                   "unique_article_records": len(docs[name])} for name, path in HIST.items()}
    refs["v13"] = {"path": str(v13_protocol_path.relative_to(ROOT)).replace("\\", "/"),
                   "sha256": sha_file(v13_protocol_path), "new_article_records": 0,
                   "uses_v12_split_manifest_sha256": sha_file(HIST["v12"])}
    return old, {"article_ids": sorted(ids), "article_hashes": sorted(hashes)}, refs


def article_iter(ds):
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


def article_manifest(old_docs: list[dict], blacklist: dict, refs: dict) -> dict:
    tokenizer = AutoTokenizer.from_pretrained(SNAPSHOT, local_files_only=True)
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train",
                      cache_dir=str(ROOT / ".hf_cache" / "datasets"), download_mode="reuse_dataset_if_exists")
    old_ids, old_hashes = set(blacklist["article_ids"]), set(blacklist["article_hashes"])
    eligible = []
    for source_index, source_row, article in article_iter(ds):
        if source_index in old_ids:
            continue
        ids = tokenizer(article, add_special_tokens=False)["input_ids"]
        if len(ids) < 128:
            continue
        token_ids = ids[:128]
        article_hash = sha_bytes(article.encode("utf-8"))
        token_hash = sha_bytes(json.dumps(token_ids).encode("utf-8"))
        if article_hash in old_hashes or token_hash in old_hashes:
            continue
        eligible.append({"source_article_index": source_index, "source_row_index": source_row,
                         "article_sha256": article_hash, "token_sha256": token_hash,
                         "token_ids": token_ids})
    rng = random.Random(ARTICLE_SEED)
    permutation = eligible.copy()
    rng.shuffle(permutation)
    chosen = permutation[:64]
    if len(chosen) != 64:
        raise RuntimeError(f"only {len(chosen)} eligible unseen articles; need 64")
    cal = chosen[:32]
    held = chosen[32:]
    assert len({x["source_article_index"] for x in chosen}) == 64
    assert len({x["article_sha256"] for x in chosen}) == 64
    assert len({x["token_sha256"] for x in chosen}) == 64
    assert not ({x["source_article_index"] for x in chosen} & old_ids)
    assert not ({h for x in chosen for h in (x["article_sha256"], x["token_sha256"])} & old_hashes)
    calibration = [{"article_id": f"C{i+1:02d}", **x} for i, x in enumerate(cal)]
    heldout = [{"article_id": f"H{i+1:02d}", **x} for i, x in enumerate(held)]
    membership = {f"K{k}": [x["article_id"] for x in calibration[:k]] for k in K_VALUES}
    return {"manifest_id": "range_w4_calibration_coverage_v14_articles",
            "dataset": "Salesforce/wikitext:wikitext-2-raw-v1:train", "split_used": "train",
            "parser": "same full-article parser as v7/v9/v12; top-level headings ^= [^=].* =$ excluding semicolon table headings",
            "tokenizer_model_revision": REVISION, "tokens_per_article": 128,
            "article_selection_seed": ARTICLE_SEED,
            "selection_rule": "enumerate eligible full train articles excluding v8-v13 IDs and hashes; shuffle once with random.Random(20261003); first 32 are calibration C01-C32, next 32 held-out H01-H32",
            "eligible_unseen_article_count_before_shuffle": len(eligible),
            "calibration_permutation": [x["source_article_index"] for x in cal],
            "calibration": calibration, "heldout": heldout,
            "K_membership": membership,
            "history_blacklist_sources": refs,
            "historical_unique_article_count": len(old_docs),
            "historical_blacklist": old_docs,
            "global_unique_new_article_count": len(chosen), "test_split_used": False}


def tree_fingerprint(path: Path) -> dict:
    return {str(p.relative_to(path)).replace("\\", "/"): {"sha256": sha_file(p), "size_bytes": p.stat().st_size}
            for p in sorted(path.rglob("*")) if p.is_file()}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    row = row_manifest()
    row_sha = frozen_write(OUT / "row_manifest.json", row)
    old_docs, blacklist, refs = historical_blacklist()
    articles = article_manifest(old_docs, blacklist, refs)
    article_sha = frozen_write(OUT / "article_manifest.json", articles)
    # Verify exact nested prefixes and persisted manifest bytes before any GPTQ call.
    checks = {
        "row_count_48": len(row["rows"]) == 48,
        "three_buckets_each_layer_projection": all(sum(r["layer"] == l and r["projection"] == p for r in row["rows"]) == 3 for l in LAYERS for p in PROJECTIONS),
        "new_layers_unstudied": set(LAYERS).isdisjoint({0, 4, 8, 12, 20, 23}),
        "article_count_64": len(articles["calibration"] + articles["heldout"]) == 64,
        "articles_unique_by_id": len({d["source_article_index"] for d in articles["calibration"] + articles["heldout"]}) == 64,
        "articles_unique_by_hash": len({d["article_sha256"] for d in articles["calibration"] + articles["heldout"]}) == 64,
        "articles_unique_by_token_hash": len({d["token_sha256"] for d in articles["calibration"] + articles["heldout"]}) == 64,
        "calibration_heldout_disjoint": not ({d["source_article_index"] for d in articles["calibration"]} & {d["source_article_index"] for d in articles["heldout"]}),
        "historical_id_disjoint": not ({d["source_article_index"] for d in articles["calibration"] + articles["heldout"]} & set(blacklist["article_ids"])),
        "historical_hash_disjoint": not ({h for d in articles["calibration"] + articles["heldout"] for h in (d["article_sha256"], d["token_sha256"])} & set(blacklist["article_hashes"])),
        "nested_K_exact": articles["K_membership"] == {f"K{k}": [f"C{i:02d}" for i in range(1, k+1)] for k in K_VALUES},
        "row_manifest_frozen": sha_file(OUT / "row_manifest.json") == row_sha,
        "article_manifest_frozen": sha_file(OUT / "article_manifest.json") == article_sha,
    }
    if not all(checks.values()):
        raise RuntimeError(f"manifest freeze check failed: {checks}")
    historical = {name: tree_fingerprint(path) for name, path in TREE_DIRS.items()}
    frozen_write(OUT / "historical_fingerprints_before.json", historical)
    protocol = {"protocol_id": "range_w4_calibration_coverage_v14", "status": "FROZEN_BEFORE_PREFLIGHT_AND_CANDIDATES",
                "model": "Qwen/Qwen2.5-0.5B-Instruct", "revision": REVISION,
                "checkpoint_sha256": row["checkpoint_sha256"], "model_dtype": "float16", "math_dtype": "float32",
                "layers": list(LAYERS), "projections": list(PROJECTIONS), "rows": 48,
                "row_manifest_sha256": row_sha, "article_manifest_sha256": article_sha,
                "quantization": {"format": "uniform W4A16", "symmetric": True, "integer_range": [0, 15],
                                  "zero_point": 8, "rounding": "torch.round ties to even", "group_size": 128,
                                  "block_size": 128, "damp_percent": 0.1, "desc_act": False,
                                  "static_groups": False, "true_sequential": True,
                                  "native_dynamic_range_at_each_group": True,
                                  "shared_alpha_applied_after_native_dynamic_minmax": True},
                "alpha_grid": [round(i / 10, 1) for i in range(1, 11)],
                "coverage_levels": list(K_VALUES),
                "D_K": "alpha=1.0 dynamic GPTQ using H_K built from the first K calibration articles",
                "C_K": "select argmin mean projection output MSE over the same K calibration articles; each candidate starts from pristine W and fresh compensation state",
                "candidate_slots": {"nominal": 1920, "reusable": 0, "new": 1920},
                "heldout": "the same H01-H32 pool for every K; evaluated only after search alpha is fixed",
                "input_semantics": "FP model projection activations, no quantized prefix",
                "dataset": "Salesforce/wikitext:wikitext-2-raw-v1:train; first 128 tokens/article; no test split",
                "historical_tree_fingerprints_file_sha256": sha_file(OUT / "historical_fingerprints_before.json"),
                "historical_tree_file_counts": {k: len(v) for k, v in historical.items()},
                "manifest_correctness_checks": checks}
    frozen_write(OUT / "PROTOCOL.json", protocol)
    print(json.dumps({"status": "MANIFESTS_FROZEN", "row_manifest_sha256": row_sha,
                      "article_manifest_sha256": article_sha, "historical_unique_article_count": len(old_docs),
                      "eligible_unseen_articles": len(articles["calibration"]) + len(articles["heldout"]),
                      "historical_tree_file_counts": protocol["historical_tree_file_counts"],
                      "checks": checks}, indent=2))


if __name__ == "__main__":
    main()
