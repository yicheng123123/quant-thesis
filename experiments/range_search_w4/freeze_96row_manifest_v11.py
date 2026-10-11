"""Freeze the outcome-independent row sample for W4 GPTQ v11 Stage 1."""
from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

import torch
from safetensors import safe_open

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
OUT = RESULTS / "range_w4_96row_v11"
V8 = RESULTS / "range_w4_dynamic_v8"
V9 = RESULTS / "range_w4_l23r448_v9"
V10 = RESULTS / "range_w4_sixrow_v10"
REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"
SNAPSHOT = ROOT / ".hf_cache" / "hub" / "models--Qwen--Qwen2.5-0.5B-Instruct" / "snapshots" / REVISION
EXPECTED_V9_MANIFEST = "34a24ef8602b8ce1e38e7a284f112acf7f766962d87d737f69148b78bcfaba78"
LAYERS = (0, 12, 23)
PROJECTIONS = ("q_proj", "o_proj", "up_proj", "down_proj")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def tensor_hash(value: torch.Tensor) -> str:
    return sha256_bytes(value.detach().contiguous().cpu().numpy().tobytes())


def model_module_key(layer: int, projection: str) -> str:
    if projection in ("q_proj", "o_proj"):
        return f"model.layers.{layer}.self_attn.{projection}.weight"
    return f"model.layers.{layer}.mlp.{projection}.weight"


def resolved_dims(config: dict, projection: str) -> tuple[int, int]:
    hidden = int(config["hidden_size"])
    intermediate = int(config["intermediate_size"])
    if projection in ("q_proj", "o_proj"):
        return hidden, hidden
    if projection == "up_proj":
        return intermediate, hidden
    if projection == "down_proj":
        return hidden, intermediate
    raise KeyError(projection)


def tree_fingerprint(path: Path) -> dict:
    result = {}
    for file in sorted(x for x in path.rglob("*") if x.is_file()):
        result[str(file.relative_to(path))] = {
            "sha256": sha256_bytes(file.read_bytes()),
            "mtime_ns": file.stat().st_mtime_ns,
            "size_bytes": file.stat().st_size,
        }
    return result


def frozen_write(path: Path, value: dict) -> None:
    content = json.dumps(value, ensure_ascii=False, indent=2) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != content:
            raise RuntimeError(f"frozen artifact differs; refusing overwrite: {path}")
        return
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(content, encoding="utf-8")
    temp.replace(path)


def main() -> None:
    if not V8.is_dir() or not V9.is_dir() or not V10.is_dir():
        raise FileNotFoundError("v8, v9 and v10 result trees are required")
    snapshot_file = SNAPSHOT / "model.safetensors"
    config_path = SNAPSHOT / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    v9_manifest_path = V9 / "splits_manifest.json"
    v9_sha = sha256_bytes(v9_manifest_path.read_bytes())
    if v9_sha != EXPECTED_V9_MANIFEST:
        raise RuntimeError(f"v9 frozen manifest mismatch: {v9_sha}")

    # Freeze row IDs before any v11 model inference or candidate results.
    rng = random.Random(42)
    pending = []
    for layer in LAYERS:
        for projection in PROJECTIONS:
            output_dim, input_dim = resolved_dims(config, projection)
            key = model_module_key(layer, projection)
            for stratum in range(8):
                lo = (stratum * output_dim) // 8
                hi = ((stratum + 1) * output_dim) // 8
                if hi <= lo:
                    raise RuntimeError(f"empty output-row stratum: {layer}/{projection}/{stratum}")
                row_index = rng.randrange(lo, hi)
                pending.append({
                    "layer": layer,
                    "projection": projection,
                    "projection_name": ("self_attn." + projection) if projection in ("q_proj", "o_proj") else ("mlp." + projection),
                    "row_index": row_index,
                    "output_dim": output_dim,
                    "input_dim": input_dim,
                    "selection_seed": 42,
                    "selection_method": "one row sampled with random.Random(42) from each of eight equal-width contiguous output-index strata; fixed iteration order layer=(0,12,23), projection=(q_proj,o_proj,up_proj,down_proj), stratum=0..7",
                    "output_index_stratum": stratum,
                    "stratum_start_inclusive": lo,
                    "stratum_stop_exclusive": hi,
                    "safetensors_key": key,
                })

    if len(pending) != 96 or len({(r["layer"], r["projection"]) for r in pending}) != 12:
        raise AssertionError("row sample geometry is not 3 x 4 x 8")
    with safe_open(str(snapshot_file), framework="pt", device="cpu") as reader:
        for row in pending:
            sl = reader.get_slice(row["safetensors_key"])
            shape = sl.get_shape()
            if shape != [row["output_dim"], row["input_dim"]]:
                raise RuntimeError(f"checkpoint geometry mismatch for {row['safetensors_key']}: {shape}")
            source = sl[row["row_index"]:row["row_index"] + 1, :]
            # The published checkpoint stores BF16 tensors. The frozen experiment
            # loads the model in FP16; hash the effective FP16 row as FP32 bytes.
            effective_fp16_as_fp32 = source.to(torch.float16).to(torch.float32).reshape(-1).contiguous()
            row["weight_hash"] = tensor_hash(effective_fp16_as_fp32)
            row["checkpoint_tensor_dtype"] = sl.get_dtype()
            row["effective_model_dtype"] = "float16"

    manifest = {
        "id": "range_w4_96row_v11",
        "model": "Qwen/Qwen2.5-0.5B-Instruct",
        "model_revision": REVISION,
        "source_checkpoint": str(snapshot_file.relative_to(ROOT)),
        "source_config_sha256": sha256_bytes(config_path.read_bytes()),
        "source_checkpoint_sha256": sha256_bytes(snapshot_file.read_bytes()),
        "selection_seed": 42,
        "selection_method": pending[0]["selection_method"],
        "selection_order": {"layers": list(LAYERS), "projections": list(PROJECTIONS), "strata": list(range(8))},
        "rows_per_layer_projection": 8,
        "layers": list(LAYERS),
        "projections": [
            "self_attn.q_proj", "self_attn.o_proj", "mlp.up_proj", "mlp.down_proj"
        ],
        "n_rows": len(pending),
        "rows": pending,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    frozen_write(OUT / "row_manifest.json", manifest)
    row_manifest_sha = sha256_bytes((OUT / "row_manifest.json").read_bytes())

    v9_manifest = json.loads(v9_manifest_path.read_text(encoding="utf-8"))
    frozen_splits = []
    for split in v9_manifest["splits"][:3]:
        frozen_splits.append({
            "split_id": split["split_id"],
            "search": [{k: d[k] for k in ("source_article_index", "source_row_index", "article_sha256", "token_sha256")} for d in split["search"]],
            "heldout": [{k: d[k] for k in ("source_article_index", "source_row_index", "article_sha256", "token_sha256")} for d in split["heldout"]],
        })
    reference = {
        "v9_splits_manifest_sha256": v9_sha,
        "row_manifest_sha256": row_manifest_sha,
        "reused_split_ids": [s["split_id"] for s in v9_manifest["splits"][:3]],
        "split_article_token_hashes": frozen_splits,
        "historical_fingerprints_before_v11": {
            "v8": tree_fingerprint(V8),
            "v9": tree_fingerprint(V9),
            "v10": tree_fingerprint(V10),
        },
    }
    frozen_write(OUT / "manifest_reference.json", reference)
    print(json.dumps({
        "row_manifest_sha256": sha256_bytes((OUT / "row_manifest.json").read_bytes()),
        "rows": len(pending),
        "dimensions": {p: resolved_dims(config, p) for p in PROJECTIONS},
        "q_proj_overlap_with_v10_six_rows": [
            {"layer": r["layer"], "row_index": r["row_index"]}
            for r in pending if r["projection"] == "q_proj" and r["row_index"] in (0, 448)
        ],
        "v9_manifest_sha256": v9_sha,
        "v8_v9_v10_fingerprints_recorded": True,
    }, indent=2))


if __name__ == "__main__":
    main()
