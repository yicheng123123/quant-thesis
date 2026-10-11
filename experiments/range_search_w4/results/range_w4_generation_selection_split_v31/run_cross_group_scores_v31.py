"""Generate only the frozen missing cross-group scores; never runs GPTQ."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
RESULTS = ROOT / "experiments/range_search_w4/results"
V23 = RESULTS / "range_w4_block_objective_e1_v23"
V24 = RESULTS / "range_w4_structured_group_e2_v24"
V29 = RESULTS / "range_w4_quadratic_hybrid_m2_v29"
MODEL = ROOT / ".hf_cache/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775/model.safetensors"
PROTOCOL = json.loads((HERE / "FROZEN_PROTOCOL.json").read_text("utf-8"))
AUDIT = json.loads((HERE / "DATA_RELATION_AUDIT.json").read_text("utf-8"))
LIMIT = float(PROTOCOL["budget"]["new_gpu_wall_seconds_max"])
DEVICE = torch.device("cuda")
ALPHAS = [float(a) for a in PROTOCOL["inheritance"]["v24_scalar_grid"]]
ROWS = {k: [int(x) for x in v] for k, v in PROTOCOL["inheritance"]["v24_fixed_rows"].items()}
GROUPS = {k: [int(x) for x in v] for k, v in PROTOCOL["documents"]["groups"].items()}
SCORE_PLAN = [("S0", "C"), ("S0", "D"), ("S1", "A"), ("S1", "B")]


def sync() -> None:
    torch.cuda.synchronize(DEVICE)


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def sha_tensor(t: torch.Tensor) -> str:
    return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def atomic_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n", "utf-8")
    temp.replace(path)


def append_event(path: Path, event: dict) -> None:
    with path.open("a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(event, ensure_ascii=False, allow_nan=False) + "\n")


def target_key(layer: int, projection: str) -> str:
    return f"model.layers.{layer}.mlp.{projection}.weight"


def input_path(layer: int, projection: str) -> Path:
    if projection == "up_proj":
        return V23 / "cache" / f"input_L{layer}_up_proj.f16"
    return V24 / "cache" / f"input_L{layer}_down_proj.f16"


def input_shape(projection: str) -> tuple[int, int, int]:
    return (64, 2048, 896 if projection == "up_proj" else 4864)


def candidate_metadata(layer: int, projection: str, split: str, alpha: float) -> tuple[Path, dict]:
    candidate_root = V24 / "raw/full_candidates"
    for meta_path in candidate_root.glob(f"L{layer}_{projection}_{split}_a*.json"):
        meta = json.loads(meta_path.read_text("utf-8"))
        if math.isclose(float(meta["alpha"]), alpha, rel_tol=0.0, abs_tol=1e-12):
            return meta_path, meta
    raise FileNotFoundError(f"frozen candidate missing: L{layer}/{projection}/{split}/alpha={alpha}")


def score_group(x_mm: np.memmap, doc_positions: list[int], original_rows: torch.Tensor,
                quantized_rows: torch.Tensor) -> dict[int, float]:
    """Exact v24 direct scorer path: two docs per chunk, FP32 GEMM, FP64 sum."""
    delta = quantized_rows.to(DEVICE).float() - original_rows.to(DEVICE).float()
    totals = torch.zeros((delta.shape[0],), dtype=torch.float64, device=DEVICE)
    for offset in range(0, len(doc_positions), 2):
        docs = doc_positions[offset:offset + 2]
        x = torch.from_numpy(np.array(x_mm[docs], copy=True)).to(DEVICE).float()
        output_delta = torch.matmul(x.reshape(-1, x.shape[-1]), delta.T)
        totals += output_delta.to(torch.float64).square().sum(0)
        del x, output_delta
    values = (totals / float(len(doc_positions) * 2048)).detach().cpu().tolist()
    return {int(i): float(v) for i, v in enumerate(values)}


def main() -> None:
    if AUDIT.get("status") != "PASS" or not (HERE / "FROZEN_PROTOCOL.json").is_file():
        raise RuntimeError("v31 data relation audit and frozen protocol must pass before scoring")
    if PROTOCOL.get("status") != "FROZEN_BEFORE_CROSS_GROUP_SCORING":
        raise RuntimeError("v31 protocol status is not frozen")
    cuda_started = time.perf_counter()
    budget_path = HERE / "BUDGET.json"
    budget = json.loads(budget_path.read_text("utf-8")) if budget_path.is_file() else {
        "max_seconds": LIMIT, "gpu_wall_elapsed_seconds": 0.0, "last_recorded_stage": "before CUDA initialization"
    }
    if float(budget.get("max_seconds", LIMIT)) != LIMIT:
        raise RuntimeError("budget does not match the frozen v31 limit")
    previous = float(budget.get("gpu_wall_elapsed_seconds", 0.0))
    if previous >= LIMIT:
        raise RuntimeError("v31 GPU scoring budget exhausted; existing results remain resumable")

    torch.cuda.init()
    if torch.cuda.get_device_name(0) != PROTOCOL["inheritance"]["v24_model"]["device"]:
        raise RuntimeError(f"unexpected CUDA device: {torch.cuda.get_device_name(0)}")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if torch.backends.cuda.matmul.allow_tf32 or torch.backends.cudnn.allow_tf32:
        raise RuntimeError("TF32 must stay disabled")
    sync()

    target_root = HERE / "raw/cross_group_scores"
    events_path = HERE / "RUN_EVENTS.jsonl"
    out_manifest = HERE / "RUN_STATUS.json"
    record_path = HERE / "FORMAL_TIMING.csv"
    if not record_path.exists():
        with record_path.open("w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(["bank", "layer", "projection", "eval_group", "alpha", "status", "wall_seconds", "gpu_peak_allocated_bytes", "gpu_peak_reserved_bytes", "record_path"])

    sync(); warmup_start = time.perf_counter()
    first_layer, first_projection = 0, "up_proj"
    first_input = np.memmap(input_path(first_layer, first_projection), dtype="<f2", mode="r", shape=input_shape(first_projection))
    first_weight = None
    with safe_open(str(MODEL), framework="pt", device="cpu") as f:
        first_weight = f.get_tensor(target_key(first_layer, first_projection)).to(torch.float16).contiguous()[ROWS[first_projection]]
    first_meta_path, first_meta = candidate_metadata(first_layer, first_projection, "S0", ALPHAS[0])
    first_candidate = torch.load(first_meta_path.with_suffix(".pt"), map_location="cpu", weights_only=True)["weight_fp16"]
    x_warm = torch.from_numpy(np.array(first_input[:2], copy=True)).to(DEVICE).float()
    delta_warm = first_candidate.to(DEVICE).float() - first_weight.to(DEVICE).float()
    warm_out = torch.matmul(x_warm.reshape(-1, x_warm.shape[-1]), delta_warm.T)
    warm_sum = warm_out.double().square().sum(0)
    sync()
    warmup_seconds = time.perf_counter() - warmup_start
    del x_warm, delta_warm, warm_out, warm_sum, first_input, first_weight, first_candidate, first_meta
    sync()

    # Cache CPU pristine rows and read-only input maps one target at a time.
    total_calls_expected = 240
    score_calls_new = 0
    score_rows_new = 0
    completed_keys = set()
    if record_path.exists():
        for saved in csv.DictReader(record_path.open(newline="", encoding="utf-8")):
            completed_keys.add((saved["bank"], int(saved["layer"]), saved["projection"], saved["eval_group"], float(saved["alpha"])))
    input_manifest_rows = list(csv.DictReader((HERE / "ACTIVATION_CACHE_AUDIT.csv").open(newline="", encoding="utf-8")))
    input_hash_by_target = {(int(r["layer"]), r["projection"]): r["current_sha256"] for r in input_manifest_rows}
    source_rows = list(csv.DictReader((HERE / "CANDIDATE_BANK_AUDIT.csv").open(newline="", encoding="utf-8")))
    index = {(int(r["layer"]), r["projection"], r["split"], float(r["alpha"])): r for r in source_rows}
    for bank, eval_group in SCORE_PLAN:
        generated_split = bank
        docs = GROUPS[eval_group]
        for layer, projection in [(0, "up_proj"), (0, "down_proj"), (12, "up_proj"), (12, "down_proj"), (23, "up_proj"), (23, "down_proj")]:
            x = np.memmap(input_path(layer, projection), dtype="<f2", mode="r", shape=input_shape(projection))
            with safe_open(str(MODEL), framework="pt", device="cpu") as f:
                original = f.get_tensor(target_key(layer, projection)).to(torch.float16).contiguous()
            rows = ROWS[projection]
            pristine_rows = original[rows].clone()
            del original
            for alpha in ALPHAS:
                meta_path, meta = candidate_metadata(layer, projection, generated_split, alpha)
                pt_path = meta_path.with_suffix(".pt")
                meta_record = index[(layer, projection, generated_split, alpha)]
                for row in rows:
                    if str(row) not in meta["search_mse_by_row"]:
                        raise RuntimeError(f"candidate row absent in sidecar: {meta_path.name}/{row}")
                output_path = target_root / bank / eval_group / f"L{layer}_{projection}_a{alpha:.1f}.json"
                expected = {
                    "protocol_sha256": sha_file(HERE / "FROZEN_PROTOCOL.json"),
                    "candidate_weight_sha256": meta["weight_sha256"],
                    "candidate_hessian_sha256": meta["hessian_sha256"],
                    "candidate_generation_documents": meta["search_doc_indices"],
                    "evaluation_group": eval_group, "evaluation_doc_indices": docs,
                    "input_cache_sha256": input_hash_by_target[(layer, projection)],
                }
                timing_key = (bank, layer, projection, eval_group, alpha)
                if output_path.is_file():
                    existing = json.loads(output_path.read_text("utf-8"))
                    if any(existing.get(k) != v for k, v in expected.items()):
                        raise RuntimeError(f"resume score checkpoint protocol mismatch: {output_path}")
                    if set(map(str, existing.get("score_mse_by_row", {}))) != set(map(str, rows)):
                        raise RuntimeError(f"resume score checkpoint row set mismatch: {output_path}")
                    if timing_key not in completed_keys:
                        with record_path.open("a", newline="", encoding="utf-8") as f:
                            csv.writer(f).writerow([bank, layer, projection, eval_group, f"{alpha:.1f}", "RESUMED_CHECKPOINT",
                                                    f"{existing['scoring_wall_seconds']:.9f}", existing["gpu_peak_allocated_bytes"],
                                                    existing["gpu_peak_reserved_bytes"], str(output_path.relative_to(HERE)).replace("\\", "/")])
                        completed_keys.add(timing_key)
                    score_calls_new += 0
                    continue
                elapsed_total = previous + (time.perf_counter() - cuda_started)
                if elapsed_total + 0.25 >= LIMIT:
                    status = {"status": "BUDGET_STOP_RESUMABLE", "gpu_wall_elapsed_seconds": elapsed_total,
                              "max_seconds": LIMIT, "score_calls_completed": score_calls_new,
                              "score_calls_expected": total_calls_expected, "last_stage": f"{bank}/{eval_group}/L{layer}/{projection}/a{alpha:.1f}",
                              "warmup_seconds": warmup_seconds, "timestamp_utc": datetime.now(timezone.utc).isoformat()}
                    atomic_json(budget_path, {**status, "last_recorded_stage": status["last_stage"]})
                    atomic_json(out_manifest, status)
                    raise RuntimeError("v31 10-minute GPU scoring budget reached; rerun to resume exact frozen scope")
                candidate_load_start = time.perf_counter()
                candidate = torch.load(pt_path, map_location="cpu", weights_only=True)["weight_fp16"]
                if sha_tensor(candidate) != meta["weight_sha256"] or meta_record["weight_sha256"] != meta["weight_sha256"]:
                    raise RuntimeError(f"candidate checkpoint hash changed: {pt_path}")
                candidate_load_seconds = time.perf_counter() - candidate_load_start
                torch.cuda.reset_peak_memory_stats(DEVICE)
                sync(); start = time.perf_counter()
                scores = score_group(x, docs, pristine_rows, candidate)
                sync(); wall = time.perf_counter() - start
                score_calls_new += 1
                score_rows_new += len(rows)
                allocated = int(torch.cuda.max_memory_allocated(DEVICE))
                reserved = int(torch.cuda.max_memory_reserved(DEVICE))
                record = {
                    **expected,
                    "bank": bank, "layer": layer, "projection": projection,
                    "rows": rows, "alpha": alpha, "score_mse_by_row": {str(k): v for k, v in scores.items()},
                    "candidate_load_seconds": candidate_load_seconds,
                    "scoring_wall_seconds": wall,
                    "candidate_file": str(pt_path.relative_to(ROOT)).replace("\\", "/"),
                    "candidate_file_sha256": sha_file(pt_path),
                    "gpu_peak_allocated_bytes": allocated, "gpu_peak_reserved_bytes": reserved,
                    "scorer": "v24 direct score_selected_rows semantics; 2 docs/chunk, FP32 GEMM, FP64 square reduction",
                }
                atomic_json(output_path, record)
                with record_path.open("a", newline="", encoding="utf-8") as f:
                    csv.writer(f).writerow([bank, layer, projection, eval_group, f"{alpha:.1f}", "SCORED", f"{wall:.9f}", allocated, reserved, str(output_path.relative_to(HERE)).replace("\\", "/")])
                completed_keys.add(timing_key)
                elapsed_total = previous + (time.perf_counter() - cuda_started)
                budget_record = {
                    "status": "RUNNING", "max_seconds": LIMIT, "gpu_wall_elapsed_seconds": elapsed_total,
                    "score_calls_completed": len(completed_keys), "score_calls_expected": total_calls_expected,
                    "new_row_alpha_scores_completed": len(completed_keys) * 4, "warmup_seconds": warmup_seconds,
                    "last_recorded_stage": f"{bank}/{eval_group}/L{layer}/{projection}/a{alpha:.1f}",
                    "gpu_peak_allocated_bytes": allocated, "gpu_peak_reserved_bytes": reserved,
                    "updated_utc": datetime.now(timezone.utc).isoformat(),
                }
                atomic_json(budget_path, budget_record)
                atomic_json(out_manifest, budget_record)
                append_event(events_path, {"event": "candidate_group_scored", "at_utc": datetime.now(timezone.utc).isoformat(), **budget_record})
            del pristine_rows, x
    sync()
    elapsed_total = previous + (time.perf_counter() - cuda_started)
    existing_records = list(csv.DictReader(record_path.open(newline="", encoding="utf-8")))
    scored_unique = {(r["bank"], int(r["layer"]), r["projection"], r["eval_group"], float(r["alpha"])) for r in existing_records}
    expected_unique = 240
    complete = len(scored_unique) == expected_unique
    status = {
        "status": "COMPLETE" if complete else "INCOMPLETE_RESUMABLE",
        "max_seconds": LIMIT, "gpu_wall_elapsed_seconds": elapsed_total,
        "score_calls_completed": len(scored_unique), "score_calls_expected": expected_unique,
        "new_row_alpha_scores_completed": len(scored_unique) * 4,
        "new_model_forwards": 0, "GPTQ_candidate_executions": 0,
        "warmup_seconds_this_process": warmup_seconds,
        "peak_allocated_bytes": max((int(r["gpu_peak_allocated_bytes"]) for r in existing_records), default=0),
        "peak_reserved_bytes": max((int(r["gpu_peak_reserved_bytes"]) for r in existing_records), default=0),
        "completed_utc": datetime.now(timezone.utc).isoformat(),
    }
    atomic_json(budget_path, {**status, "last_recorded_stage": "all frozen cross-group score checkpoints"})
    atomic_json(out_manifest, status)
    if not complete:
        raise RuntimeError(f"v31 incomplete: {len(scored_unique)}/{expected_unique} unique scores; resumable")
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
