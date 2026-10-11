"""Freeze v15 paired articles, reused rows, native order, and protocol."""
from __future__ import annotations

import hashlib
import json
import os
import random
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
RESULTS = OUT.parent
os.environ.update({
    "HF_HOME": str(ROOT / ".hf_cache"),
    "HF_HUB_CACHE": str(ROOT / ".hf_cache" / "hub"),
    "HF_HUB_OFFLINE": "1",
    "HF_DATASETS_OFFLINE": "1",
})

from datasets import load_dataset
from transformers import AutoTokenizer
from gptqmodel.models.definitions.qwen2 import Qwen2GPTQ

REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"
SEED = 20261004
CAL_COUNT = 16
HELD_COUNT = 32
ROW_REF = RESULTS / "range_w4_calibration_coverage_v14" / "row_manifest.json"
V14_ARTICLES = RESULTS / "range_w4_calibration_coverage_v14" / "article_manifest.json"
SNAPSHOT = ROOT / ".hf_cache" / "hub" / "models--Qwen--Qwen2.5-0.5B-Instruct" / "snapshots" / REVISION
TASK_SOURCE = Path(r"E:\CodexHome\attachments\05d9e0a5-dda3-484e-9d6b-59e282e6a309\已粘贴的文本.txt")
TREE_DIRS = {
    "v8": RESULTS / "range_w4_dynamic_v8",
    "v9": RESULTS / "range_w4_l23r448_v9",
    "v10": RESULTS / "range_w4_sixrow_v10",
    "v11": RESULTS / "range_w4_96row_v11",
    "v12": RESULTS / "range_w4_gate_v12",
    "v13": RESULTS / "range_w4_failure_mechanism_v13",
    "v14": RESULTS / "range_w4_calibration_coverage_v14",
}
ARTICLE_SOURCES = {
    "v8": RESULTS / "range_w4_dynamic_v8" / "manifest.json",
    "v9": RESULTS / "range_w4_l23r448_v9" / "splits_manifest.json",
    "v10": RESULTS / "range_w4_sixrow_v10" / "manifest_reference.json",
    "v11": RESULTS / "range_w4_96row_v11" / "manifest_reference.json",
    "v12": RESULTS / "range_w4_gate_v12" / "validation_split_manifest.json",
    "v13": RESULTS / "range_w4_failure_mechanism_v13" / "PROTOCOL.json",
    "v14": V14_ARTICLES,
}
QWEN2_DEF = ROOT / ".venv-gptq" / "Lib" / "site-packages" / "gptqmodel" / "models" / "definitions" / "qwen2.py"
GPTQMODEL_BASE = ROOT / ".venv-gptq" / "Lib" / "site-packages" / "gptqmodel" / "models" / "base.py"


def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def save_frozen(path: Path, obj: dict) -> str:
    payload = json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if path.exists() and path.read_text(encoding="utf-8") != payload:
        raise RuntimeError(f"refusing to alter frozen artifact: {path}")
    if not path.exists():
        path.write_text(payload, encoding="utf-8", newline="\n")
    return sha_file(path)


def article_records(obj) -> list[dict]:
    found: list[dict] = []

    def visit(x):
        if isinstance(x, dict):
            if "source_article_index" in x and ("article_sha256" in x or "token_sha256" in x):
                found.append({
                    "source_article_index": int(x["source_article_index"]),
                    "article_sha256": x.get("article_sha256"),
                    "token_sha256": x.get("token_sha256", x.get("sha256")),
                })
            for value in x.values():
                visit(value)
        elif isinstance(x, list):
            for value in x:
                visit(value)

    visit(obj)
    unique = {(d["source_article_index"], d["article_sha256"], d["token_sha256"]): d for d in found}
    return list(unique.values())


def make_blacklist() -> tuple[set[int], set[str], dict]:
    source_docs = {}
    source_meta = {}
    for name, path in ARTICLE_SOURCES.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        obj = json.loads(path.read_text(encoding="utf-8"))
        source_docs[name] = article_records(obj)
        source_meta[name] = {
            "path": path.relative_to(ROOT).as_posix(),
            "sha256": sha_file(path),
            "article_records_found": len(source_docs[name]),
        }

    # v13 is a cached audit of v12 and adds no article pool of its own.
    v13_protocol = json.loads(ARTICLE_SOURCES["v13"].read_text(encoding="utf-8"))
    v12_sha = sha_file(ARTICLE_SOURCES["v12"])
    v13_refs = v13_protocol.get("source_file_sha256", {})
    v13_ref_matches = v13_refs.get("validation_split_manifest.json") == v12_sha
    if not v13_ref_matches:
        raise RuntimeError("v13 does not reference the current frozen v12 validation manifest")

    all_records = []
    for name in ("v8", "v9", "v10", "v11", "v12", "v14"):
        all_records.extend(source_docs[name])
    # v13 reuses v12 data; record that provenance without double-counting it.
    by_id: dict[int, dict] = {}
    for item in all_records:
        article_id = item["source_article_index"]
        prior = by_id.get(article_id)
        if prior is not None:
            for key in ("article_sha256", "token_sha256"):
                if prior.get(key) and item.get(key) and prior[key] != item[key]:
                    raise RuntimeError(f"historical article {article_id} has conflicting {key}")
            prior["article_sha256"] = prior.get("article_sha256") or item.get("article_sha256")
            prior["token_sha256"] = prior.get("token_sha256") or item.get("token_sha256")
        else:
            by_id[article_id] = dict(item)

    v14 = json.loads(V14_ARTICLES.read_text(encoding="utf-8"))
    embedded = {d["source_article_index"]: d for d in v14.get("historical_blacklist", [])}
    v14_new = v14.get("calibration", []) + v14.get("heldout", [])
    embedded.update({d["source_article_index"]: d for d in v14_new})
    if set(by_id) != set(embedded):
        raise RuntimeError(f"independent history ID inventory disagrees with v14: {len(by_id)} vs {len(embedded)}")
    for article_id, item in by_id.items():
        ref = embedded[article_id]
        for key in ("article_sha256", "token_sha256"):
            if item.get(key) and ref.get(key) and item[key] != ref[key]:
                raise RuntimeError(f"historical hash mismatch for article {article_id}: {key}")

    ids = set(by_id)
    hashes = {value for d in by_id.values() for value in (d.get("article_sha256"), d.get("token_sha256")) if value}
    source_meta["v13_reference_check"] = {"v13_reuses_v12_manifest": v13_ref_matches, "v12_manifest_sha256": v12_sha}
    source_meta["deduplicated_historical_article_count"] = len(ids)
    source_meta["deduplicated_historical_hash_count"] = len(hashes)
    return ids, hashes, source_meta


def iter_full_articles(ds):
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


def module_order(config: dict) -> dict:
    layers = int(config["num_hidden_layers"])
    groups = [list(group) for group in Qwen2GPTQ.layer_modules]
    expected = [
        ["self_attn.k_proj", "self_attn.v_proj", "self_attn.q_proj"],
        ["self_attn.o_proj"],
        ["mlp.up_proj", "mlp.gate_proj"],
        ["mlp.down_proj"],
    ]
    if groups != expected:
        raise RuntimeError(f"unexpected GPTQModel Qwen2 layer groups: {groups}")
    module_group = {
        "q_proj": 0,
        "o_proj": 1,
        "up_proj": 2,
        "down_proj": 3,
    }
    targets = []
    projections = ("q_proj", "o_proj", "up_proj", "down_proj")
    for layer in (2, 6, 14, 18):
        for projection in projections:
            group_idx = module_group[projection]
            prior = []
            for prior_layer in range(layer):
                for group in groups:
                    prior.extend(f"model.layers.{prior_layer}.{name}" for name in group)
            for group in groups[:group_idx]:
                prior.extend(f"model.layers.{layer}.{name}" for name in group)
            current_group = [f"model.layers.{layer}.{name}" for name in groups[group_idx]]
            target_name = f"model.layers.{layer}.{'self_attn' if projection in ('q_proj', 'o_proj') else 'mlp'}.{projection}"
            targets.append({
                "layer": layer,
                "projection": projection,
                "target_module": target_name,
                "native_group_index_zero_based": group_idx,
                "native_group_members": current_group,
                "same_group_siblings": [name for name in current_group if name != target_name],
                "quantized_prefix_modules_at_capture": prior,
                "capture_rule": "all earlier native groups/layers use alpha=1 GPTQ; target dense FP16 weight is temporarily installed; same-group siblings share block/MLP input and do not precede the target input",
            })
    return {
        "model_architecture": "Qwen2ForCausalLM",
        "model_type": config.get("model_type"),
        "num_hidden_layers": layers,
        "gptqmodel_version_expected": "2.2.0",
        "resolved_gptq_model_class": "gptqmodel.models.definitions.qwen2.Qwen2GPTQ",
        "true_sequential": True,
        "native_layer_group_order": groups,
        "native_semantics_source": "GPTQModel BaseGPTQModel.quantize: one hook-capture forward per group, then quantize each member; propagate quantized layer output to next layer",
        "source_files": {
            "qwen2_definition": {"path": QWEN2_DEF.relative_to(ROOT).as_posix(), "sha256": sha_file(QWEN2_DEF)},
            "base_quantize_loop": {"path": GPTQMODEL_BASE.relative_to(ROOT).as_posix(), "sha256": sha_file(GPTQMODEL_BASE)},
        },
        "target_capture_order": targets,
        "unquantized_non_linear_or_outside_modules": ["model.embed_tokens", "model.norm", "lm_head"],
    }


def tree_fingerprint(path: Path) -> dict:
    return {
        p.relative_to(path).as_posix(): {"sha256": sha_file(p), "size_bytes": p.stat().st_size}
        for p in sorted(path.rglob("*")) if p.is_file()
    }


def main():
    frozen_outputs = (
        "PROTOCOL.json", "row_manifest_reference.json", "article_manifest.json",
        "module_execution_order.json", "historical_fingerprints_before.json",
    )
    if any((OUT / name).exists() for name in frozen_outputs):
        raise RuntimeError(f"v15 frozen outputs already exist; refusing to overwrite: {OUT}")
    OUT.mkdir(parents=True, exist_ok=True)
    v14_rows = json.loads(ROW_REF.read_text(encoding="utf-8"))
    row_sha = sha_file(ROW_REF)
    if len(v14_rows["rows"]) != 48 or v14_rows["model_revision"] != REVISION:
        raise RuntimeError("v14 row manifest does not match frozen v15 target shape/revision")
    row_reference = {
        "reference_id": "v15_reuses_v14_rows_exactly",
        "source_path": ROW_REF.relative_to(ROOT).as_posix(),
        "source_sha256": row_sha,
        "row_count": len(v14_rows["rows"]),
        "layers": v14_rows["layers"],
        "projections": v14_rows["projections"],
        "rows": v14_rows["rows"],
    }
    row_ref_sha = save_frozen(OUT / "row_manifest_reference.json", row_reference)

    historical_ids, historical_hashes, source_meta = make_blacklist()
    tokenizer = AutoTokenizer.from_pretrained(SNAPSHOT, local_files_only=True)
    ds = load_dataset(
        "Salesforce/wikitext", "wikitext-2-raw-v1", split="train",
        cache_dir=str(ROOT / ".hf_cache" / "datasets"), download_mode="reuse_dataset_if_exists",
    )
    eligible = []
    for source_index, source_row, text in iter_full_articles(ds):
        if source_index in historical_ids:
            continue
        token_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        if len(token_ids) < 128:
            continue
        token_ids = token_ids[:128]
        article_hash = sha_bytes(text.encode("utf-8"))
        token_hash = sha_bytes(json.dumps(token_ids).encode("utf-8"))
        if article_hash in historical_hashes or token_hash in historical_hashes:
            continue
        eligible.append({
            "source_article_index": source_index,
            "source_row_index": source_row,
            "article_sha256": article_hash,
            "token_sha256": token_hash,
            "token_ids": token_ids,
        })
    permutation = eligible.copy()
    random.Random(SEED).shuffle(permutation)
    chosen = permutation[:CAL_COUNT + HELD_COUNT]
    if len(chosen) != CAL_COUNT + HELD_COUNT:
        raise RuntimeError(f"only {len(chosen)} eligible new articles; need {CAL_COUNT + HELD_COUNT}")
    if len({x["source_article_index"] for x in chosen}) != 48 or len({x["article_sha256"] for x in chosen}) != 48 or len({x["token_sha256"] for x in chosen}) != 48:
        raise RuntimeError("new article sample is not globally unique")
    if ({x["source_article_index"] for x in chosen} & historical_ids) or ({h for x in chosen for h in (x["article_sha256"], x["token_sha256"])} & historical_hashes):
        raise RuntimeError("new article sample overlaps v8-v14 history")
    cal = [{"article_id": f"C{i+1:02d}", "role": "calibration", **x} for i, x in enumerate(chosen[:CAL_COUNT])]
    held = [{"article_id": f"H{i+1:02d}", "role": "heldout", **x} for i, x in enumerate(chosen[CAL_COUNT:])]
    article_manifest = {
        "manifest_id": "range_w4_quantized_prefix_v15_articles",
        "dataset": "Salesforce/wikitext:wikitext-2-raw-v1:train",
        "split_used": "train",
        "parser": "same full-article heading parser used by v14",
        "tokenizer_model_revision": REVISION,
        "tokens_per_article": 128,
        "seed": SEED,
        "selection_rule": "enumerate eligible full train articles excluding v8-v14 source article IDs/article hashes/token hashes; shuffle once with random.Random(20261004); first 16 calibration C01-C16, next 32 heldout H01-H32",
        "eligible_unseen_article_count_before_shuffle": len(eligible),
        "permutation_source_article_indices": [x["source_article_index"] for x in chosen],
        "calibration": cal,
        "heldout": held,
        "historical_blacklist_source_inventory": source_meta,
        "historical_unique_article_ids_excluded": len(historical_ids),
        "historical_unique_article_and_token_hashes_excluded": len(historical_hashes),
        "global_unique_new_article_count": len(chosen),
        "calibration_heldout_disjoint": True,
        "test_split_used": False,
    }
    article_sha = save_frozen(OUT / "article_manifest.json", article_manifest)

    config = json.loads((SNAPSHOT / "config.json").read_text(encoding="utf-8"))
    execution = module_order(config)
    exec_sha = save_frozen(OUT / "module_execution_order.json", execution)

    history_fingerprints = {name: tree_fingerprint(path) for name, path in TREE_DIRS.items()}
    history_sha = save_frozen(OUT / "historical_fingerprints_before.json", history_fingerprints)
    core_path = ROOT / "experiments" / "range_search_w4" / "dynamic_core.py"
    protocol = {
        "protocol_id": "range_w4_quantized_prefix_v15",
        "status": "FROZEN_BEFORE_PREFLIGHT_AND_CANDIDATES",
        "research_question": "Does FP-input shared-alpha range-search gain transfer to real native-sequential GPTQ quantized-prefix inputs?",
        "model": "Qwen/Qwen2.5-0.5B-Instruct",
        "revision": REVISION,
        "checkpoint_sha256": v14_rows["checkpoint_sha256"],
        "dtype": {"weights": "float16", "GPTQ_Hessian_output_loss": "float32"},
        "targets": {"row_count": 48, "layers": [2, 6, 14, 18], "projections": ["q_proj", "o_proj", "up_proj", "down_proj"], "row_source": "exact v14 frozen row manifest"},
        "row_manifest_reference_sha256": row_ref_sha,
        "v14_row_manifest_sha256": row_sha,
        "article_manifest_sha256": article_sha,
        "module_execution_order_sha256": exec_sha,
        "dataset": {"name": "Salesforce/wikitext:wikitext-2-raw-v1", "split": "train", "articles": 48, "calibration_articles": 16, "heldout_articles": 32, "tokens_per_article": 128, "seed": SEED, "article_manifest_contains_exact_token_ids": True, "same_documents_for_FP_and_QPREFIX": True, "disjoint_from_v8_v14": True},
        "quantization": {"format": "uniform W4A16", "bits": 4, "symmetric": True, "group_size": 128, "block_size": 128, "damp_percent": 0.1, "desc_act": False, "static_groups": False, "true_sequential": True, "alpha_grid": [i / 10 for i in range(1, 11)]},
        "gptqmodel": {"version": "2.2.0", "native_model_definition": "Qwen2GPTQ", "native_layer_groups": execution["native_layer_group_order"], "qprefix_uses_alpha": 1.0, "qprefix_native_full_model_pass": True, "lm_head_quantized": False},
        "input_regimes": {
            "FP_INPUT": "FP16 model full forward; capture pristine target module inputs on all 48 frozen documents.",
            "QUANTIZED_PREFIX_INPUT": "GPTQModel native full sequential W4A16 G128 alpha=1 baseline using C01-C16; capture hidden states at target block inputs from that quantized model, then replay the target block with each target projection temporarily restored to its pristine FP16 weight before capturing X_Q.",
            "target_weight": "same pristine FP16 target row in both regimes; its input is captured with dense target module while all required earlier native module groups remain quantized.",
        },
        "candidate_semantics": {"D": "alpha=1 dynamic GPTQ with regime-specific calibration input", "C": "ten shared alpha candidates; select by mean calibration output MSE using that regime's 16 calibration articles", "heldout": "H01-H32 in both regimes; used only after alpha selection"},
        "candidate_workload": {"nominal_row_level_candidates": 960, "historical_candidates_reusable": 0, "new_candidates": 960},
        "preflight_requirements": ["four FP alpha=1 parity checks", "four QPREFIX alpha=1 parity checks", "same pristine target W hash", "target module restored dense during Q input capture", "all required prefix modules are native W4A16 G128 alpha=1", "finite outputs and activations"],
        "primary_metrics": ["exact and within-one-step alpha agreement", "gain sign agreement", "positive-to-positive and positive-to-negative counts", "Spearman(gain_FP,gain_Q)", "median and pooled gain by regime", "tail regression counts and worst row", "D_Q/D_FP heldout MSE ratio", "activation drift", "calibration Gram/Hessian drift", "cross-alpha evaluation"],
        "transfer_support_gate": {
            "verdict_if_all_pass": "TRANSFER_SIGNAL_PRESERVED",
            "A": "QPREFIX pooled C_Q vs D_Q gain > 0",
            "B": "QPREFIX median row gain >= 0",
            "C": "at least two thirds of FP-positive rows stay positive under QPREFIX",
            "D": "QPREFIX severe count gain < -0.10 <= max(FP severe count * 2, FP severe count + 3)",
            "otherwise": "TRANSFER_SIGNAL_NOT_PRESERVED",
        },
        "source_hashes": {
            "dynamic_core.py": sha_file(core_path),
            "qwen2_definition.py": execution["source_files"]["qwen2_definition"]["sha256"],
            "gptqmodel_base.py": execution["source_files"]["base_quantize_loop"]["sha256"],
            "task_instruction_attachment": sha_file(TASK_SOURCE),
        },
        "historical_tree_fingerprints_file_sha256": history_sha,
        "historical_tree_names": list(TREE_DIRS),
    }
    protocol_sha = save_frozen(OUT / "PROTOCOL.json", protocol)
    print(json.dumps({
        "status": "FROZEN",
        "row_manifest_v14_sha256": row_sha,
        "row_reference_sha256": row_ref_sha,
        "article_manifest_sha256": article_sha,
        "module_execution_order_sha256": exec_sha,
        "protocol_sha256": protocol_sha,
        "new_articles": len(chosen),
        "historical_article_ids_excluded": len(historical_ids),
        "historical_trees_fingerprinted": list(TREE_DIRS),
        "nominal_target_candidates": 960,
        "native_layer_groups": groups if (groups := execution["native_layer_group_order"]) else [],
    }, indent=2))


if __name__ == "__main__":
    main()
