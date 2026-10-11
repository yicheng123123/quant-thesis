"""Freeze validation row/article manifests for the already frozen Gate v1.

This preparation stage does not run GPTQ or evaluate candidate outcomes.
"""
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
V8 = RESULTS / "range_w4_dynamic_v8"
V9 = RESULTS / "range_w4_l23r448_v9"
V10 = RESULTS / "range_w4_sixrow_v10"
V11 = RESULTS / "range_w4_96row_v11"
GATE_SHA = "7943b8f26e5f94456b7b00144f6d47fc775d0224ccca4a6a1eff38e5740999c9"
REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"
LAYERS = (4, 8, 20)
PROJECTIONS = ("q_proj", "o_proj", "up_proj", "down_proj")
SEED = 20261001
SNAPSHOT = ROOT / ".hf_cache" / "hub" / "models--Qwen--Qwen2.5-0.5B-Instruct" / "snapshots" / REVISION


def digest_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def digest_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def tensor_hash(tensor: torch.Tensor) -> str:
    return digest_bytes(tensor.detach().contiguous().cpu().numpy().tobytes())


def write_frozen(path: Path, value: dict) -> str:
    payload = json.dumps(value, ensure_ascii=False, indent=2) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != payload:
            raise RuntimeError(f"Frozen manifest exists with different contents: {path}")
    else:
        path.write_text(payload, encoding="utf-8", newline="\n")
    return digest_file(path)


def key_for(layer: int, projection: str) -> str:
    if projection in ("q_proj", "o_proj"):
        return f"model.layers.{layer}.self_attn.{projection}.weight"
    return f"model.layers.{layer}.mlp.{projection}.weight"


def row_manifest() -> dict:
    gate_path = OUT / "GATE_V1.json"
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    gate_sha = digest_file(gate_path)
    if gate_sha != GATE_SHA or gate["tau_gain"] != 0.10 or gate["tau_margin"] != 0.06:
        raise RuntimeError("Frozen Gate identity changed; stopping before manifest creation")
    plan = (OUT / "VALIDATION_PLAN_FROZEN.md").read_text(encoding="utf-8")
    if GATE_SHA not in plan or "20261001" not in plan:
        raise RuntimeError("Frozen validation plan identity/seed mismatch")
    dims_cfg = json.loads((SNAPSHOT / "config.json").read_text(encoding="utf-8"))
    hidden = int(dims_cfg["hidden_size"])
    inter = int(dims_cfg["intermediate_size"])
    dimensions = {"q_proj": (hidden, hidden), "o_proj": (hidden, hidden),
                  "up_proj": (inter, hidden), "down_proj": (hidden, inter)}
    rng = random.Random(SEED)
    entries = []
    for layer in LAYERS:
        for projection in PROJECTIONS:
            output_dim, input_dim = dimensions[projection]
            for bucket_index, bucket_name in enumerate(("low", "low-mid", "high-mid", "high")):
                start = bucket_index * output_dim // 4
                stop = (bucket_index + 1) * output_dim // 4
                row_index = rng.randrange(start, stop)
                entries.append({"layer": layer, "projection": projection,
                                "projection_name": (f"self_attn.{projection}" if projection in ("q_proj", "o_proj")
                                                    else f"mlp.{projection}"),
                                "row_index": row_index, "input_dim": input_dim,
                                "output_dim": output_dim, "bucket": bucket_name,
                                "bucket_start_inclusive": start, "bucket_stop_exclusive": stop,
                                "seed": SEED})
    # Cross-check all 48 indices against the explicit list in the frozen plan.
    expected = {}
    for line in plan.splitlines():
        m = re.fullmatch(r"L(\d+) (q_proj|o_proj|up_proj|down_proj): ([0-9, ]+)", line.strip())
        if m:
            expected[(int(m.group(1)), m.group(2))] = [int(x.strip()) for x in m.group(3).split(",")]
    actual = {}
    for item in entries:
        actual.setdefault((item["layer"], item["projection"]), []).append(item["row_index"])
    if actual != expected or len(entries) != 48:
        raise RuntimeError("Fixed-seed row sample does not match frozen plan; stopping")

    safetensor = SNAPSHOT / "model.safetensors"
    v11_manifest = json.loads((V11 / "row_manifest.json").read_text(encoding="utf-8"))
    if digest_file(safetensor) != v11_manifest["source_checkpoint_sha256"]:
        raise RuntimeError("Checkpoint hash differs from v11; stopping")
    cfg_hash = digest_file(SNAPSHOT / "config.json")
    if cfg_hash != v11_manifest["source_config_sha256"]:
        raise RuntimeError("Model config hash differs from v11; stopping")
    with safe_open(str(safetensor), framework="pt", device="cpu") as sf:
        for item in entries:
            raw_row = sf.get_slice(key_for(item["layer"], item["projection"]))[item["row_index"], :]
            # Model is loaded in FP16; v11 hashes the FP16 row represented as FP32.
            effective = raw_row.to(torch.float16).float().contiguous()
            item["weight_hash"] = tensor_hash(effective)
            item["checkpoint_tensor_dtype"] = str(raw_row.dtype)
            item["effective_model_dtype"] = "float16"
    return {"manifest_id": "range_w4_gate_v12_validation_rows",
            "gate_v1_sha256": gate_sha, "model": "Qwen/Qwen2.5-0.5B-Instruct",
            "model_revision": REVISION, "checkpoint_sha256": digest_file(safetensor),
            "config_sha256": cfg_hash, "sampling_seed": SEED,
            "sampling_rule": "fixed Python random.Random seed, one row per equal-width output-index bucket; iteration order L4/L8/L20, q/o/up/down, low/low-mid/high-mid/high",
            "layers": list(LAYERS), "projections": list(PROJECTIONS),
            "row_count": len(entries), "rows": entries}


def collect_reference_docs(value: dict) -> dict[str, dict]:
    found = {}
    def add(doc):
        if "source_article_index" not in doc:
            return
        article = doc.get("article_sha256")
        token = doc.get("token_sha256", doc.get("sha256"))
        found[f"{doc['source_article_index']}|{article}|{token}"] = {
            "source_article_index": int(doc["source_article_index"]),
            "article_sha256": article, "token_sha256": token}
    for doc in value.get("documents", []):
        add(doc)
    for split in value.get("splits", []):
        for side in ("search", "heldout"):
            for doc in split.get(side, []):
                add(doc)
    for split in value.get("split_article_token_hashes", []):
        for side in ("search", "heldout"):
            for doc in split.get(side, []):
                add(doc)
    return found


def old_article_blacklist() -> tuple[set[int], set[str], dict]:
    paths = {"v8": V8 / "manifest.json", "v9": V9 / "splits_manifest.json",
             "v10": V10 / "manifest_reference.json", "v11": V11 / "manifest_reference.json"}
    loaded = {name: json.loads(path.read_text(encoding="utf-8")) for name, path in paths.items()}
    v8_docs = collect_reference_docs(loaded["v8"])
    v9_docs = collect_reference_docs(loaded["v9"])
    v10_docs = collect_reference_docs(loaded["v10"])
    v11_docs = collect_reference_docs(loaded["v11"])
    if len(v8_docs) != 8 or len(v9_docs) != 80:
        raise RuntimeError(f"Unexpected historical article sets: v8={len(v8_docs)}, v9={len(v9_docs)}")
    if v10_docs != v9_docs:
        raise RuntimeError("v10 article references do not exactly match v9 split articles")
    v9_first_three = {(d["source_article_index"], d["article_sha256"], d["token_sha256"])
                      for s in loaded["v9"]["splits"][:3]
                      for side in ("search", "heldout") for d in s[side]}
    v11_tuples = {(d["source_article_index"], d["article_sha256"], d["token_sha256"])
                  for d in v11_docs.values()}
    if v11_tuples != v9_first_three:
        raise RuntimeError("v11 article references do not exactly match its reused v9 S00-S02 splits")
    all_docs = list(v8_docs.values()) + list(v9_docs.values())
    old_ids = {d["source_article_index"] for d in all_docs}
    old_hashes = {h for d in all_docs for h in (d["article_sha256"], d["token_sha256"]) if h}
    refs = {name: {"path": str(path.relative_to(ROOT)).replace("\\", "/"),
                   "sha256": digest_file(path), "article_records": len(collect_reference_docs(loaded[name]))}
            for name, path in paths.items()}
    return old_ids, old_hashes, {"references": refs, "v8_article_count": len(v8_docs),
                                 "v9_article_count": len(v9_docs),
                                 "unique_old_source_ids": len(old_ids),
                                 "unique_old_article_and_token_hashes": len(old_hashes)}


def article_candidates():
    import re as regex
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train",
                      cache_dir=str(ROOT / ".hf_cache" / "datasets"), download_mode="reuse_dataset_if_exists")
    current, start, source_index = [], None, 0
    for row_index, record in enumerate(ds):
        line = record["text"]
        if regex.match(r"^= [^=].* =$", line.strip()) and ";" not in line:
            if current:
                yield source_index, start, "\n".join(current)
                source_index += 1
            current, start = [line], row_index
        elif current:
            current.append(line)
    if current:
        yield source_index, start, "\n".join(current)


def split_manifest() -> dict:
    old_ids, old_hashes, old_audit = old_article_blacklist()
    tokenizer = AutoTokenizer.from_pretrained(SNAPSHOT, local_files_only=True)
    chosen, seen_ids, seen_hashes = [], set(), set()
    for source_index, source_row, article in article_candidates():
        ids = tokenizer(article, add_special_tokens=False)["input_ids"]
        if len(ids) < 128 or source_index in old_ids:
            continue
        article_hash = digest_bytes(article.encode("utf-8"))
        token_ids = ids[:128]
        token_hash = digest_bytes(json.dumps(token_ids).encode("utf-8"))
        if article_hash in old_hashes or token_hash in old_hashes:
            continue
        if source_index in seen_ids or article_hash in seen_hashes or token_hash in seen_hashes:
            continue
        chosen.append({"source_article_index": source_index, "source_row_index": source_row,
                       "article_sha256": article_hash, "token_sha256": token_hash,
                       "token_ids": token_ids})
        seen_ids.add(source_index)
        seen_hashes.update((article_hash, token_hash))
        if len(chosen) == 80:
            break
    if len(chosen) != 80:
        raise RuntimeError(f"Need 80 unseen complete train articles; found {len(chosen)}")
    rng = random.Random(SEED)
    shuffled = chosen.copy()
    rng.shuffle(shuffled)
    splits = []
    for i in range(10):
        group = shuffled[i * 8:(i + 1) * 8]
        splits.append({"split_id": f"S{i:02d}", "search": group[:4], "heldout": group[4:]})
    all_docs = [d for s in splits for side in ("search", "heldout") for d in s[side]]
    if len({d["source_article_index"] for d in all_docs}) != 80:
        raise RuntimeError("Validation articles are not globally unique")
    if len({d["article_sha256"] for d in all_docs}) != 80 or len({d["token_sha256"] for d in all_docs}) != 80:
        raise RuntimeError("Validation article or token hashes repeat")
    if ({d["source_article_index"] for d in all_docs} & old_ids or
        {h for d in all_docs for h in (d["article_sha256"], d["token_sha256"])} & old_hashes):
        raise RuntimeError("Validation article set overlaps v8-v11 blacklist")
    return {"manifest_id": "range_w4_gate_v12_validation_splits",
            "dataset": "Salesforce/wikitext:wikitext-2-raw-v1:train",
            "parser": "same v7/v9 full-article parser: top-level ^= [^=].* =$ headings excluding semicolon table headings",
            "tokenizer_model_revision": REVISION, "tokens_per_article": 128,
            "seed": SEED, "selection_rule": "first 80 eligible full train articles in source order after excluding v8-v11 source IDs and article/token hashes; shuffle once with Random(20261001); sequentially partition into ten groups of eight, first four search and final four held-out",
            "n_splits": 10, "articles_per_split": {"search": 4, "heldout": 4},
            "global_unique_article_count": len(all_docs), "old_article_blacklist_audit": old_audit,
            "splits": splits}


def main() -> None:
    if digest_file(OUT / "GATE_V1.json") != GATE_SHA:
        raise RuntimeError("Gate SHA mismatch; stopping")
    rows = row_manifest()
    rows_sha = write_frozen(OUT / "validation_row_manifest.json", rows)
    splits = split_manifest()
    split_sha = write_frozen(OUT / "validation_split_manifest.json", splits)
    v11 = json.loads((V11 / "row_manifest.json").read_text(encoding="utf-8"))
    protocol = {
        "protocol_id": "range_w4_gate_v12_independent_validation",
        "status": "FROZEN_BEFORE_PREFLIGHT_AND_CANDIDATE_RUNS",
        "created_at_utc": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "gate_v1_sha256": GATE_SHA, "tau_gain": 0.10, "tau_margin": 0.06,
        "gate_decision": "row ACCEPT iff search_gain_median >= 0.10 AND search_margin_median >= 0.06; otherwise FALLBACK alpha=1.0",
        "search_gain_per_split": "1 - min candidate search_SSE / alpha1 search_SSE",
        "search_margin_per_split": "(second-smallest search_SSE - smallest search_SSE) / second-smallest search_SSE; rank candidate entries by search_SSE then alpha ascending",
        "gate_aggregation": "median across all ten validation search splits per row; accepted row uses each split's own search-best alpha; fallback row uses alpha=1 for every split",
        "model": "Qwen/Qwen2.5-0.5B-Instruct", "revision": REVISION,
        "checkpoint_sha256": v11["source_checkpoint_sha256"], "model_dtype": "float16", "math_dtype": "float32",
        "quantization": "symmetric uniform W4A16", "integer_range": [0, 15], "zero_point": 8,
        "rounding": "round to nearest, ties to even", "group_size": 128,
        "block_size": 128, "damp_percent": 0.1, "desc_act": False,
        "static_groups": False, "true_sequential": True,
        "alpha_grid": [round(i / 10, 1) for i in range(1, 11)],
        "input_semantics": "FP model projection inputs; no quantized prefix",
        "candidate_evaluation": "every candidate starts from pristine W and fresh GPTQ compensation state; native dynamic MinMax is estimated at each group boundary from currently compensated W, then shared alpha scales the range",
        "layers": [4, 8, 20], "projections": list(PROJECTIONS), "rows": 48, "splits": 10,
        "nominal_candidate_slots": 4800, "reusable_candidate_slots": 0,
        "row_manifest_sha256": rows_sha, "split_manifest_sha256": split_sha,
        "pass_criteria": {
            "A": "2 * count(G held_gain < -0.10) <= count(C held_gain < -0.10)",
            "B": "Gate pooled SSE gain versus D > 0",
            "C": "when C pooled gain > 0, Gate pooled gain / C pooled gain >= 0.50",
            "final": "GATE_V1_VALIDATION_PASS iff all applicable A/B/C pass, else GATE_V1_VALIDATION_FAIL"},
        "article_protocol": {"dataset": splits["dataset"], "manifest_sha256": split_sha,
                             "independent_documents": 80, "tokens_per_article": 128,
                             "test_split_used": False},
        "source_manifest_sha256s": {
            "v8_manifest": digest_file(V8 / "manifest.json"),
            "v9_split_manifest": digest_file(V9 / "splits_manifest.json"),
            "v10_manifest_reference": digest_file(V10 / "manifest_reference.json"),
            "v11_manifest_reference": digest_file(V11 / "manifest_reference.json")}}
    write_frozen(OUT / "VALIDATION_PROTOCOL.json", protocol)
    print(json.dumps({"row_manifest_sha256": rows_sha, "split_manifest_sha256": split_sha,
                      "rows": rows["row_count"], "splits": splits["n_splits"],
                      "unique_articles": splits["global_unique_article_count"],
                      "old_ids_excluded": splits["old_article_blacklist_audit"]["unique_old_source_ids"],
                      "candidate_slots": 4800}, indent=2))


if __name__ == "__main__":
    main()
