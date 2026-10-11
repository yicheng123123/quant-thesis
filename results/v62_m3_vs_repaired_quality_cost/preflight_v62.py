from __future__ import annotations

import csv, hashlib, json, math, platform, sys, time
from pathlib import Path

import numpy as np
import torch
import transformers
from safetensors import safe_open
from transformers import AutoConfig, AutoTokenizer

ROOT = Path(r"E:\quant-thesis")
OUT = ROOT / "results/v62_m3_vs_repaired_quality_cost"
V59 = ROOT / "experiments/range_search_w4/results/v59_same_process_four_checkpoint_eval"
V61 = ROOT / "results/v61_m3_calibration_cost_memory"
V55 = ROOT / "experiments/range_search_w4/results/v55_m2_decoder_fixed_integer_repair"
V18 = ROOT / "experiments/range_search_w4/results/range_w4_fullmodel_v18"
M3 = V61 / "M3/dense_export"
R2 = V55 / "R1/checkpoint/final"
MODEL_REV = "7ae557604adf67be50417f59c2c2f167def9a775"
EFFECTIVE_CONFIG_EXPECTED = "a33901fd787a1385efbdd5574d88012774c97110064a3f6a2cb05d070cba9dd2"

def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()

def sha_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

def tensor_sha(t: torch.Tensor) -> str:
    return sha_bytes(t.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())

def canonical_config(d: dict) -> dict:
    return {k: v for k, v in d.items() if k not in {"_name_or_path", "name_or_path", "transformers_version", "_commit_hash"}}

def config_effective(path: Path) -> tuple[str, dict]:
    cfg = AutoConfig.from_pretrained(str(path), local_files_only=True)
    effective = canonical_config(cfg.to_dict())
    digest = sha_bytes(json.dumps(effective, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8"))
    return digest, effective

def tensor_manifest(path: Path) -> list[dict]:
    rows = []
    with safe_open(str(path), framework="pt", device="cpu") as f:
        for name in sorted(f.keys()):
            t = f.get_tensor(name)
            rows.append({"tensor": name, "shape": json.dumps(list(t.shape), separators=(",", ":")), "dtype": str(t.dtype), "sha256": tensor_sha(t)})
    return rows

def main() -> None:
    start = time.perf_counter()
    out = OUT
    out.mkdir(parents=True, exist_ok=True)
    sources = [M3 / "model.safetensors", M3 / "config.json", R2 / "model.safetensors", R2 / "config.json"]
    missing = [str(p) for p in sources if not p.is_file()]
    if missing:
        raise FileNotFoundError(missing)
    if OUT.joinpath("RUN_LOCK.json").exists():
        raise RuntimeError("v62 output already contains a frozen run lock; use a fresh output directory")
    v61_complete = json.loads((V61 / "COMPLETENESS.json").read_text(encoding="utf-8"))
    v61_runtime = json.loads((V61 / "ACTUAL_RUNTIME.json").read_text(encoding="utf-8"))
    v61_token = json.loads((V61 / "TOKENIZER_CHECK.json").read_text(encoding="utf-8"))
    v61_training = json.loads((V61 / "M3/M3_TRAINING_SUMMARY.json").read_text(encoding="utf-8"))
    v55_lock = json.loads((V55 / "FINAL_CHECKPOINT_LOCK.json").read_text(encoding="utf-8"))
    v55_proto = json.loads((V55 / "PROTOCOL.json").read_text(encoding="utf-8"))
    v59_env = json.loads((V59 / "PROCESS_ENVIRONMENT.json").read_text(encoding="utf-8"))
    v59_proto = json.loads((V59 / "PROTOCOL.json").read_text(encoding="utf-8"))
    v59_cost = json.loads((V59 / "COST_ACCOUNTING.json").read_text(encoding="utf-8"))
    v59_recalc = list(csv.DictReader((V59 / "INDEPENDENT_WINDOW_RECALC.csv").open(encoding="utf-8-sig", newline="")))
    v59_stages = list(csv.DictReader((V59 / "STAGE_TIMINGS.csv").open(encoding="utf-8-sig", newline="")))
    v55_named_rows = list(csv.DictReader((V55 / "NAMED_TENSOR_HASHES.csv").open(encoding="utf-8-sig", newline="")))
    v55_named = {r["tensor"]: r for r in v55_named_rows}
    if len(v55_named) != 290 or len(v55_named_rows) != 290:
        raise RuntimeError(f"v55 named tensor source count is not 290: {len(v55_named_rows)}")

    result = {}
    for arm, path, expected_sha, ref_named in [
        ("v61_M3", M3, v61_complete["final_dense_weight_sha256"], None),
        ("v55_M2_plus_repair", R2, v55_lock["model_sha256"], v55_named),
    ]:
        model_path = path / "model.safetensors"
        model_sha = sha_file(model_path)
        cfg_sha = sha_file(path / "config.json")
        eff_sha, eff_config = config_effective(path)
        named = tensor_manifest(model_path)
        mismatches = []
        if ref_named is not None:
            for r in named:
                ref = ref_named.get(r["tensor"])
                if not ref or r["sha256"] != ref["sha256"] or json.loads(r["shape"]) != json.loads(ref["shape"]) or r["dtype"].replace("torch.", "") != ref["dtype"].replace("torch.", ""):
                    mismatches.append(r["tensor"])
        if model_sha != expected_sha:
            raise RuntimeError(f"model SHA mismatch {arm}: {model_sha} != {expected_sha}")
        if len(named) != 290 or mismatches:
            raise RuntimeError(f"named tensor manifest mismatch {arm}: count={len(named)} mismatch={mismatches[:4]}")
        (OUT / f"{arm}_NAMED_TENSORS.csv").write_text("tensor,shape,dtype,sha256\n" + "".join(f'{r["tensor"]},{json.dumps(r["shape"])},{r["dtype"]},{r["sha256"]}\n' for r in named), encoding="utf-8")
        result[arm] = {"checkpoint_path": str(model_path), "checkpoint_sha256": model_sha, "bytes": model_path.stat().st_size,
                       "config_path": str(path / "config.json"), "config_sha256": cfg_sha,
                       "effective_config_sha256": eff_sha, "effective_config": eff_config,
                       "named_tensor_count": len(named), "named_tensor_manifest_csv": str(OUT / f"{arm}_NAMED_TENSORS.csv"),
                       "named_tensor_manifest_sha256": sha_file(OUT / f"{arm}_NAMED_TENSORS.csv"),
                       "named_tensor_manifest_matches_checkpoint_lock": not mismatches if ref_named is not None else "computed_from_locked_safetensors"}

    # Both exported model configs must resolve to v59's same effective Qwen2 configuration.
    for arm, rec in result.items():
        if rec["effective_config_sha256"] != EFFECTIVE_CONFIG_EXPECTED:
            raise RuntimeError(f"effective config mismatch for {arm}: {rec['effective_config_sha256']}")
        if rec["effective_config"].get("model_type") != "qwen2" or rec["effective_config"].get("num_hidden_layers") != 24:
            raise RuntimeError(f"unexpected architecture/config for {arm}")

    # Compare tokenizer semantics on fixed probes; the evaluator itself uses frozen token IDs.
    base = ROOT / "artifacts/models/qwen2.5-0.5b-instruct-7ae5576" / MODEL_REV
    tok_results = {}
    probes = ["Quantization uses a fixed grid.", "The quick brown fox jumps over 13 lazy dogs.", "量化校准样本文本。"]
    toks = {}
    for name, path in [("M3_export", M3), ("v18_revision", base)]:
        tok = AutoTokenizer.from_pretrained(str(path), local_files_only=True, use_fast=True)
        enc = [tok.encode(s, add_special_tokens=False) for s in probes]
        toks[name] = enc
        tok_files = [p for p in path.iterdir() if p.is_file() and p.name in {"tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "vocab.json", "merges.txt", "added_tokens.json"}]
        tok_results[name] = {"source": str(path), "class": type(tok).__name__, "is_fast": bool(tok.is_fast), "vocab_size": tok.vocab_size,
                             "len_tokenizer": len(tok), "bos_token_id": tok.bos_token_id, "eos_token_id": tok.eos_token_id,
                             "pad_token_id": tok.pad_token_id, "files": [{"name": p.name, "bytes": p.stat().st_size, "sha256": sha_file(p)} for p in sorted(tok_files)],
                             "probe_token_ids": enc}
    if toks["M3_export"] != toks["v18_revision"]:
        raise RuntimeError("M3 exported tokenizer differs on frozen deterministic probe strings")
    if tok_results["M3_export"]["vocab_size"] != tok_results["v18_revision"]["vocab_size"] or tok_results["M3_export"]["eos_token_id"] != tok_results["v18_revision"]["eos_token_id"]:
        raise RuntimeError("M3 exported tokenizer metadata differs from model revision")

    if v61_complete["status"] != "COMPLETE" or v61_complete["new_ppl"]:
        raise RuntimeError("v61 M3 completion state not suitable")
    if v55_proto["quantization"]["symmetric"] is not True or v55_proto["quantization"]["weight_bits"] != 4:
        raise RuntimeError("v55 protocol is not expected fixed W4 symmetric format")
    if v59_cost["cap_seconds"] != 3600.0 or v59_proto["evaluation"]["wikitext"]["windows"] != 582 or v59_proto["evaluation"]["c4_fixed_budget"]["windows"] != 253:
        raise RuntimeError("v59 evaluation source/protocol unexpected")
    pair_rows = {}
    for arm in ["M1", "M2", "M1_plus_repair", "M2_plus_repair"]:
        selected = [r for r in v59_stages if r["arm"] == arm and r["stage"] == "evaluation"]
        pair_rows[arm] = {r["dataset"]: float(r["seconds"]) for r in selected}
    pair_rows["M1_pair_eval_seconds"] = sum(pair_rows["M1"].values())
    pair_rows["M2_pair_eval_seconds"] = sum(pair_rows["M2"].values())
    pair_rows["M1_plus_repair_pair_eval_seconds"] = sum(pair_rows["M1_plus_repair"].values())
    pair_rows["M2_plus_repair_pair_eval_seconds"] = sum(pair_rows["M2_plus_repair"].values())
    pair_rows["one_checkpoint_pair_process_wall_from_v58_seconds"] = float(json.loads((ROOT / "experiments/range_search_w4/results/v58_m2_repaired_unified_eval/COST_ACCOUNTING.json").read_text(encoding="utf-8"))["process_wall_including_loader_and_evaluation"])
    pair_rows["v56_one_checkpoint_pair_continuation_seconds"] = 2283.576
    max_pair = max(pair_rows[k] for k in ["M1_pair_eval_seconds", "M2_pair_eval_seconds", "M1_plus_repair_pair_eval_seconds", "M2_plus_repair_pair_eval_seconds"])
    forecast = {"cap_seconds": 5400.0, "source_v59_max_measured_pair_eval_seconds": max_pair,
                "two_checkpoint_pairs_at_2x_slowest_v59_pair_seconds": 4 * max_pair,
                "source_precheck_load_finalization_allowance_seconds": 600.0,
                "projected_active_seconds": 4 * max_pair + 600.0,
                "remaining_margin_seconds": 5400.0 - (4 * max_pair + 600.0),
                "v56_2283_576s_scope": "one checkpoint WT2+C4 eval-only continuation wall including load; seconds, not minutes; historical slower execution, not the current forecast",
                "v59_scope": "eight complete evaluation stages across four checkpoints, total 2383.0349 s; max measured per-checkpoint WT2+C4 evaluation pair used conservatively above",
                "decision": "WITHIN_BUDGET" if 4 * max_pair + 600.0 < 5400.0 else "STOP_BEFORE_EVALUATION"}
    recalculated = []
    for r in v59_recalc:
        recalculated.append({"dataset": r["dataset"], "arms": r["arm"], "windows": int(r["windows"]), "valid_tokens": int(r["valid_tokens"]),
                             "nll_sum": float(r["nll_sum"]), "weighted_nll": float(r["weighted_nll"]), "ppl": float(r["ppl"])})
    cpu_summary = {"status": "PASS", "mode": "CPU-only preflight; no CUDA initialization or model forward",
                   "python": platform.python_version(), "python_executable": sys.executable, "pytorch": torch.__version__,
                   "transformers": transformers.__version__, "checkpoints": result, "tokenizers": tok_results,
                   "v61_calibration": {"scheduled_steps": v61_complete["scheduled_iterations"], "successful_optimizer_steps": v61_complete["successful_optimizer_steps"],
                   "skips": v61_complete["gradscaler_skips"], "model_revision": v61_training["model_revision"],
                                       "calibration_manifest_sha256": v61_complete["token_manifest_sha256"], "ordered_token_sha256": v61_complete["ordered_token_sha256"]},
                   "v55_quantization": v55_proto["quantization"], "v55_calibration": v55_proto["calibration"],
                   "v59_runtime": v59_env, "v59_evaluator_and_manifests": v59_proto["evaluation"],
                   "v59_cost_reference": v59_cost, "v59_pair_times": pair_rows, "forecast": forecast,
                   "v59_independent_quality_recalc": recalculated, "preflight_elapsed_seconds": time.perf_counter() - start}
    (OUT / "CPU_PRECHECK.json").write_text(json.dumps(cpu_summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "v61_m3_sha256": result["v61_M3"]["checkpoint_sha256"],
                      "v55_r2_sha256": result["v55_M2_plus_repair"]["checkpoint_sha256"],
                      "effective_configs_match": True, "tokenizer_probe_match": True,
                      "v59_pair_eval_seconds": pair_rows, "forecast": forecast,
                      "preflight_elapsed_seconds": cpu_summary["preflight_elapsed_seconds"]}, indent=2))

if __name__ == "__main__":
    main()
