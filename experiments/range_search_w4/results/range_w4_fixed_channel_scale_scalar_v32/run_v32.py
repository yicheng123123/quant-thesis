"""Frozen v32 scaled-coordinate GPTQ local diagnostic; no model forward or PPL."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
RESULTS = ROOT / "experiments/range_search_w4/results"
V24 = RESULTS / "range_w4_structured_group_e2_v24"
MODEL = ROOT / ".hf_cache/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775/model.safetensors"
sys.path.insert(0, str(RESULTS / "range_w4_m2_runtime_v17_3"))
sys.path.insert(0, str(RESULTS / "range_w4_m2_parity_v17_1"))
from optimized_native_m2 import optimized_dynamic_gptq, prepare_native_state  # noqa: E402
from native_m2 import native_dynamic_gptq  # noqa: E402
from gptqmodel.quantization import QuantizeConfig  # noqa: E402
from gptqmodel.quantization.gptq import GPTQ  # noqa: E402


PROTOCOL_PATH = HERE / "FROZEN_PROTOCOL.json"
P = json.loads(PROTOCOL_PATH.read_text("utf-8"))
DEVICE = torch.device("cuda:0")
ALPHAS = [float(x) for x in P["alpha_grid"]]
ROWS = {k: [int(v) for v in xs] for k, xs in P["targets"]["rows"].items()}
DOCS = P["split_documents"]
SEQ = 2048
LIMIT = float(P["budget"]["new_gpu_wall_seconds_max"])


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def sha_tensor(t: torch.Tensor) -> str:
    return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def sync() -> None:
    torch.cuda.synchronize(DEVICE)


def dump(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n", "utf-8")
    tmp.replace(path)


def save_pt(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    tmp.replace(path)


def append_event(event: dict) -> None:
    with (HERE / "RUN_EVENTS.jsonl").open("a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(event, ensure_ascii=False, allow_nan=False) + "\n")


def timed(fn):
    sync(); started = time.perf_counter()
    result = fn()
    sync(); return result, time.perf_counter() - started


class Budget:
    def __init__(self, mode: str):
        self.mode = mode
        p = HERE / "BUDGET.json"
        prior = json.loads(p.read_text("utf-8")) if p.is_file() else {}
        if prior and float(prior["max_seconds"]) != LIMIT:
            raise RuntimeError("v32 cumulative GPU budget changed")
        self.previous = float(prior.get("elapsed_seconds", 0.0))
        self.started = time.perf_counter()
        self.status = "RUNNING"
        if self.previous >= LIMIT:
            raise RuntimeError("v32 cumulative GPU budget already exhausted")
        self.record("process_start")

    def elapsed(self) -> float:
        return self.previous + time.perf_counter() - self.started

    def check(self, stage: str, margin: float = 25.0) -> None:
        if self.elapsed() + margin >= LIMIT:
            self.finish("BUDGET_STOP_RESUMABLE", stage=stage)
            raise RuntimeError(f"v32 cumulative 20-minute GPU budget reached before {stage}")

    def record(self, stage: str, **fields) -> None:
        obj = {"at_utc": datetime.now(timezone.utc).isoformat(), "mode": self.mode,
               "stage": stage, "cumulative_wall_seconds": self.elapsed(), **fields}
        append_event(obj)
        dump(HERE / "BUDGET.json", {"status": self.status, "max_seconds": LIMIT,
                                    "elapsed_seconds": obj["cumulative_wall_seconds"],
                                    "last_stage": stage, "mode": self.mode,
                                    "updated_utc": obj["at_utc"]})

    def finish(self, status: str, **fields) -> None:
        self.status = status
        self.record("process_finish", **fields)


def target_stem(layer: int, projection: str, split: str) -> str:
    return f"L{layer}_{projection}_{split}"


def input_mm(layer: int, projection: str) -> np.memmap:
    p = ROOT / P["input_paths"][f"L{layer}_{projection}"]
    d = 896 if projection == "up_proj" else 4864
    if p.stat().st_size != 64 * SEQ * d * 2:
        raise RuntimeError(f"input cache shape/byte mismatch: {p}")
    return np.memmap(p, dtype="<f2", mode="r", shape=(64, SEQ, d))


def v24_alpha1_meta(layer: int, projection: str, split: str) -> tuple[Path, dict]:
    for p in (V24 / "raw/full_candidates").glob(f"L{layer}_{projection}_{split}_a*.json"):
        m = json.loads(p.read_text("utf-8"))
        if float(m["alpha"]) == 1.0:
            return p, m
    raise FileNotFoundError(f"v24 alpha1 candidate missing: {target_stem(layer, projection, split)}")


def pristine_weight(layer: int, projection: str, split: str) -> torch.Tensor:
    key = f"model.layers.{layer}.mlp.{projection}.weight"
    with safe_open(str(MODEL), framework="pt", device="cpu") as f:
        w = f.get_tensor(key).to(torch.float16).contiguous().to(DEVICE)
    _, meta = v24_alpha1_meta(layer, projection, split)
    if sha_tensor(w) != meta["pristine_weight_sha256"]:
        raise RuntimeError(f"pristine FP16 weight hash mismatch: {target_stem(layer, projection, split)}")
    return w


def config() -> QuantizeConfig:
    return QuantizeConfig(bits=4, group_size=128, sym=True, desc_act=False,
                          static_groups=False, true_sequential=True,
                          damp_percent=0.1, damp_auto_increment=0, mse=0)


def native_hessian(weight: torch.Tensor, x_mm: np.memmap, indices: list[int],
                   d: torch.Tensor | None) -> tuple[torch.Tensor, int, float]:
    module = torch.nn.Linear(weight.shape[1], weight.shape[0], bias=False,
                             device="meta", dtype=torch.float16)
    module.weight = torch.nn.Parameter(weight.detach(), requires_grad=False)
    core = GPTQ(module, qcfg=config())
    core.quantizer.configure(perchannel=True, bits=4, sym=True)
    core.H = torch.zeros((weight.shape[1], weight.shape[1]), dtype=torch.float32, device=DEVICE)
    core.nsamples = 0
    sync(); started = time.perf_counter()
    for idx in indices:
        x = torch.from_numpy(np.array(x_mm[idx], copy=True)).unsqueeze(0).to(DEVICE)
        if d is not None:
            x = (x.float() / d).to(torch.float16)
        core.process_batch(x)
        del x
    sync(); wall = time.perf_counter() - started
    return core.H.detach().clone(), int(core.nsamples), wall


def load_or_build_scale(layer: int, projection: str, split: str, x_mm: np.memmap,
                        budget: Budget) -> tuple[torch.Tensor, dict]:
    stem = target_stem(layer, projection, split)
    path = HERE / "raw/scales" / f"{stem}.pt"
    meta_path = path.with_suffix(".json")
    expected = {"protocol_sha256": sha_file(PROTOCOL_PATH), "stem": stem,
                "search_doc_indices": DOCS[split]["search"],
                "input_sha256": P["input_sha256"][f"L{layer}_{projection}"]}
    if path.is_file() or meta_path.is_file():
        if not path.is_file() or not meta_path.is_file():
            raise RuntimeError(f"incomplete D checkpoint pair: {stem}")
        m = json.loads(meta_path.read_text("utf-8"))
        data = torch.load(path, map_location="cpu", weights_only=True)
        if any(m.get(k) != v for k, v in expected.items()):
            raise RuntimeError(f"D checkpoint protocol mismatch: {stem}")
        if sha_tensor(data["D"]) != m["D_sha256"] or sha_tensor(data["a_fp64"]) != m["a_sha256"]:
            raise RuntimeError(f"D checkpoint tensor hash mismatch: {stem}")
        return data["D"].to(DEVICE), m

    budget.check(f"D statistic {stem}")
    started = time.perf_counter()
    totals = np.zeros((x_mm.shape[-1],), dtype=np.float64)
    for i in DOCS[split]["search"]:
        raw = np.array(x_mm[i], copy=True).astype(np.float32)
        totals += np.abs(raw).sum(axis=0, dtype=np.float64)
    a = totals / float(16 * SEQ)
    raw_d = np.sqrt(np.maximum(a, 1e-8))
    norm = math.sqrt(float(raw_d.min()) * float(raw_d.max()))
    d = (raw_d / norm).astype(np.float32)
    if not np.all(np.isfinite(d)) or np.any(d <= 0):
        raise RuntimeError(f"nonfinite/nonpositive D: {stem}")
    stats_seconds = time.perf_counter() - started
    data = {"D": torch.from_numpy(d.copy()), "a_fp64": torch.from_numpy(a.copy())}
    m = {**expected, "D_sha256": sha_tensor(data["D"]), "a_sha256": sha_tensor(data["a_fp64"]),
         "D_dtype": str(data["D"].dtype), "a_dtype": str(data["a_fp64"].dtype),
         "D_min": float(d.min()), "D_max": float(d.max()),
         "a_min": float(a.min()), "a_max": float(a.max()),
         "normalization_fp64": norm, "statistic_seconds": stats_seconds}
    save_pt(path, data); dump(meta_path, m)
    budget.record("D_statistic_saved", stem=stem, seconds=stats_seconds, D_sha256=m["D_sha256"])
    return data["D"].to(DEVICE), m


def transformed_weight(w: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
    return (w.float() * d.unsqueeze(0)).to(torch.float16).contiguous()


def load_or_build_scaled_h(layer: int, projection: str, split: str, wprime: torch.Tensor,
                           d: torch.Tensor, d_meta: dict, x_mm: np.memmap,
                           budget: Budget) -> tuple[torch.Tensor, int, dict]:
    stem = target_stem(layer, projection, split)
    path = HERE / "raw/hessians" / f"{stem}.pt"
    meta_path = path.with_suffix(".json")
    expected = {"protocol_sha256": sha_file(PROTOCOL_PATH), "stem": stem,
                "D_sha256": d_meta["D_sha256"], "Wprime_sha256": sha_tensor(wprime),
                "search_doc_indices": DOCS[split]["search"],
                "input_sha256": P["input_sha256"][f"L{layer}_{projection}"]}
    if path.is_file() or meta_path.is_file():
        if not path.is_file() or not meta_path.is_file():
            raise RuntimeError(f"incomplete scaled H checkpoint pair: {stem}")
        m = json.loads(meta_path.read_text("utf-8"))
        obj = torch.load(path, map_location="cpu", weights_only=True)
        if any(m.get(k) != v for k, v in expected.items()):
            raise RuntimeError(f"scaled H checkpoint protocol mismatch: {stem}")
        if sha_tensor(obj["H"]) != m["H_sha256"] or int(obj["nsamples"]) != 16:
            raise RuntimeError(f"scaled H checkpoint tensor mismatch: {stem}")
        return obj["H"].to(DEVICE), int(obj["nsamples"]), m
    budget.check(f"scaled H build {stem}")
    h, nsamples, wall = native_hessian(wprime, x_mm, DOCS[split]["search"], d)
    if nsamples != 16:
        raise RuntimeError(f"scaled H nsamples !=16: {stem}")
    m = {**expected, "H_sha256": sha_tensor(h), "H_dtype": str(h.dtype),
         "nsamples": nsamples, "hessian_build_seconds": wall,
         "same_as_unscaled_H": False}
    save_pt(path, {"H": h.detach().cpu(), "nsamples": nsamples})
    dump(meta_path, m)
    budget.record("scaled_H_saved", stem=stem, seconds=wall, H_sha256=m["H_sha256"])
    return h, nsamples, m


def equivalence_spot(w: torch.Tensor, d: torch.Tensor, x_mm: np.memmap,
                     rows: list[int], split: str) -> dict:
    x16 = torch.from_numpy(np.array(x_mm[DOCS[split]["search"][0], :128], copy=True)).to(DEVICE)
    w16 = w[rows]
    x64, w64, d64 = x16.double(), w16.double(), d.double()
    fp64_base = x64 @ w64.T
    fp64_scaled = (x64 / d64) @ (w64 * d64).T
    x32, w32, d32 = x16.float(), w16.float(), d.float()
    fp32_base = x32 @ w32.T
    fp32_scaled = (x32 / d32) @ (w32 * d32).T
    xprime16 = (x32 / d32).half()
    wprime16 = (w32 * d32).half()
    fp16_scaled = xprime16.float() @ wprime16.float().T
    return {
        "spot_tokens": 128, "spot_rows": rows, "real_identity": "(X D^-1)(W D)^T = X W^T by diagonal cancellation",
        "fp64_max_abs": float((fp64_scaled - fp64_base).abs().max().item()),
        "fp64_mse": float((fp64_scaled - fp64_base).square().mean().item()),
        "fp32_max_abs": float((fp32_scaled - fp32_base).abs().max().item()),
        "fp32_mse": float((fp32_scaled - fp32_base).square().mean().item()),
        "fp16_cast_max_abs": float((fp16_scaled - fp32_base).abs().max().item()),
        "fp16_cast_mse": float((fp16_scaled - fp32_base).square().mean().item()),
    }


def offset_chunks(x_mm: np.memmap, indices: list[int], w: torch.Tensor,
                  wprime: torch.Tensor, d: torch.Tensor, rows: list[int]) -> tuple[list[torch.Tensor], dict]:
    offsets = []
    squared = torch.zeros((len(rows),), device=DEVICE, dtype=torch.float64)
    max_abs = 0.0
    for start in range(0, len(indices), 2):
        x = torch.from_numpy(np.array(x_mm[indices[start:start + 2]], copy=True)).to(DEVICE)
        xp = (x.float() / d).half().float().reshape(-1, x.shape[-1])
        xf = x.float().reshape(-1, x.shape[-1])
        ref = xf @ w[rows].float().T
        transformed_ref = xp @ wprime[rows].float().T
        offset = transformed_ref - ref
        offsets.append(offset.detach())
        squared += offset.double().square().sum(0)
        max_abs = max(max_abs, float(offset.abs().max().item()))
        del x, xp, xf, ref, transformed_ref
    drift = {str(row): float(squared[j].item() / (len(indices) * SEQ)) for j, row in enumerate(rows)}
    return offsets, {"unquantized_fp16_coordinate_drift_mse_by_row": drift,
                     "unquantized_fp16_coordinate_drift_max_abs": max_abs,
                     "reference": "original X_FP16.float @ W_FP16.float.T"}


def score_scaled(x_mm: np.memmap, indices: list[int], wprime: torch.Tensor,
                 q_rows: torch.Tensor, d: torch.Tensor, rows: list[int],
                 offsets: list[torch.Tensor]) -> dict[int, float]:
    delta = q_rows.to(DEVICE).float() - wprime[rows].float()
    totals = torch.zeros((len(rows),), dtype=torch.float64, device=DEVICE)
    for chunk_i, start in enumerate(range(0, len(indices), 2)):
        x = torch.from_numpy(np.array(x_mm[indices[start:start + 2]], copy=True)).to(DEVICE)
        xp = (x.float() / d).half().float()
        err = xp.reshape(-1, xp.shape[-1]) @ delta.T
        err = err + offsets[chunk_i]
        totals += err.double().square().sum(0)
        del x, xp, err
    return {row: float(totals[j].item() / (len(indices) * SEQ)) for j, row in enumerate(rows)}


def score_identity(x_mm: np.memmap, indices: list[int], w: torch.Tensor,
                   q_rows: torch.Tensor, rows: list[int]) -> dict[int, float]:
    delta = q_rows.float() - w[rows].float()
    totals = torch.zeros((len(rows),), dtype=torch.float64, device=DEVICE)
    for start in range(0, len(indices), 2):
        x = torch.from_numpy(np.array(x_mm[indices[start:start + 2]], copy=True)).to(DEVICE).float()
        err = x.reshape(-1, x.shape[-1]) @ delta.T
        totals += err.double().square().sum(0)
    return {row: float(totals[j].item() / (len(indices) * SEQ)) for j, row in enumerate(rows)}


def baseline_map() -> dict[tuple, dict]:
    with (HERE / "ORIGINAL_BANK_REUSED.csv").open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return {(r["bank"], int(r["layer"]), r["projection"], int(r["row"]), float(r["alpha"])): r for r in rows}


def identity_preflight(layer: int, projection: str, budget: Budget) -> dict:
    split = "S0"
    stem = target_stem(layer, projection, split)
    x_mm = input_mm(layer, projection)
    w = pristine_weight(layer, projection, split)
    w_hash = sha_tensor(w)
    budget.check(f"identity H parity {stem}")
    h, nsamples, h_seconds = native_hessian(w, x_mm, DOCS[split]["search"], None)
    old_h = torch.load(V24 / "cache/hessians" / f"{stem}.pt", map_location="cpu", weights_only=True)
    h_hash = sha_tensor(h)
    if nsamples != old_h["nsamples"] or h_hash != old_h["H_sha256"] or not torch.equal(h.cpu(), old_h["H"]):
        raise RuntimeError(f"D=I native Hessian exact parity FAIL: {stem}")
    prepared, factor_seconds = timed(lambda: prepare_native_state(h, nsamples=nsamples))
    result, gptq_seconds = timed(lambda: optimized_dynamic_gptq(w, prepared, alpha=1.0, check_cholesky=True))
    v24_path, meta = v24_alpha1_meta(layer, projection, split)
    saved = torch.load(v24_path.with_suffix(".pt"), map_location="cpu", weights_only=True)
    rows = ROWS[projection]
    compare = {"weight_fp16": result.weight_fp16[rows].cpu(),
               "scales": result.scales[rows].cpu(), "zeros": result.zeros[rows].cpu(),
               "g_idx": result.g_idx.cpu()}
    equality = {k: torch.equal(value, saved[k]) for k, value in compare.items()}
    if not all(equality.values()):
        raise RuntimeError(f"D=I GPTQ tensor parity FAIL: {stem}: {equality}")
    scores, score_seconds = timed(lambda: score_identity(x_mm, DOCS[split]["search"], w,
                                                          result.weight_fp16[rows], rows))
    ones = torch.ones((w.shape[1],), dtype=torch.float32, device=DEVICE)
    identity_offsets, identity_drift = offset_chunks(x_mm, DOCS[split]["search"], w, w, ones, rows)
    scaled_path_scores, scaled_path_seconds = timed(lambda: score_scaled(
        x_mm, DOCS[split]["search"], w, result.weight_fp16[rows], ones, rows, identity_offsets))
    baseline = baseline_map()
    score_parity = {str(row): scores[row] == float(baseline[(split, layer, projection, row, 1.0)]["search_mse"])
                    for row in rows}
    scaled_score_parity = {str(row): scaled_path_scores[row] == scores[row] for row in rows}
    if not all(score_parity.values()) or not all(scaled_score_parity.values()) or any(
            float(v) != 0.0 for v in identity_drift["unquantized_fp16_coordinate_drift_mse_by_row"].values()):
        raise RuntimeError(f"D=I direct/scaled score parity FAIL: {stem}: {score_parity}/{scaled_score_parity}")
    if sha_tensor(w) != w_hash or sha_tensor(h) != h_hash:
        raise RuntimeError(f"D=I W/H mutated by GPTQ: {stem}")
    budget.record("identity_preflight_pass", stem=stem, h_seconds=h_seconds,
                  factor_seconds=factor_seconds, gptq_seconds=gptq_seconds,
                  score_seconds=score_seconds, scaled_path_score_seconds=scaled_path_seconds)
    return {"stem": stem, "H_sha256": h_hash, "H_exact_vs_v24": True, "nsamples": nsamples,
            "candidate_tensor_exact_vs_v24": equality, "search_alpha1_score_exact_vs_v24": score_parity,
            "D_identity_scaled_path_score_exact_vs_direct": scaled_score_parity,
            "original_W_unchanged": True, "H_unchanged": True,
            "hessian_seconds": h_seconds, "factor_seconds": factor_seconds,
            "additional_full_linear_GPTQ_calls": 1, "candidate_score_seconds": score_seconds,
            "scaled_path_score_seconds": scaled_path_seconds}


def candidate_paths(stem: str, alpha: float) -> tuple[Path, Path]:
    pt = HERE / "raw/candidates" / f"{stem}_a{alpha:.1f}.pt"
    return pt, pt.with_suffix(".json")


def load_candidate(stem: str, alpha: float, common: dict) -> tuple[dict, dict] | None:
    pt, jp = candidate_paths(stem, alpha)
    if not pt.is_file() and not jp.is_file():
        return None
    if not pt.is_file() or not jp.is_file():
        raise RuntimeError(f"incomplete candidate checkpoint: {stem}/alpha={alpha}")
    m = json.loads(jp.read_text("utf-8"))
    if any(m.get(k) != v for k, v in common.items()) or float(m["alpha"]) != alpha:
        raise RuntimeError(f"candidate checkpoint protocol mismatch: {stem}/alpha={alpha}")
    obj = torch.load(pt, map_location="cpu", weights_only=True)
    for key, hash_key in (("weight_fp16", "weight_sha256"), ("scales", "scales_sha256"),
                          ("zeros", "zeros_sha256"), ("g_idx", "g_idx_sha256")):
        if sha_tensor(obj[key]) != m[hash_key]:
            raise RuntimeError(f"candidate checkpoint tensor hash mismatch: {stem}/{alpha}/{key}")
    return obj, m


def equal_result_fields(a, b) -> dict[str, bool]:
    return {key: torch.equal(getattr(a, key), getattr(b, key))
            for key in ("weight_fp16", "scales", "zeros", "g_idx")}


def get_candidate(layer: int, projection: str, split: str, alpha: float,
                  wprime: torch.Tensor, w: torch.Tensor, d: torch.Tensor,
                  prepared, h: torch.Tensor, h_meta: dict, d_meta: dict,
                  x_mm: np.memmap, search_offsets: list[torch.Tensor],
                  budget: Budget) -> tuple[dict, dict]:
    stem = target_stem(layer, projection, split)
    rows = ROWS[projection]
    common = {"protocol_sha256": sha_file(PROTOCOL_PATH), "stem": stem, "rows": rows,
              "D_sha256": d_meta["D_sha256"], "H_sha256": h_meta["H_sha256"],
              "Wprime_sha256": h_meta["Wprime_sha256"],
              "search_doc_indices": DOCS[split]["search"],
              "input_sha256": P["input_sha256"][f"L{layer}_{projection}"]}
    cached = load_candidate(stem, alpha, common)
    if cached is not None:
        if alpha == 1.0:
            parity_path = HERE / "raw/candidates" / f"{stem}_native_alpha1_parity.json"
            if not parity_path.is_file():
                raise RuntimeError(f"cached alpha1 candidate lacks native parity record: {stem}")
            parity = json.loads(parity_path.read_text("utf-8"))
            if parity.get("status") != "PASS" or any(parity.get(k) != v for k, v in common.items()):
                raise RuntimeError(f"cached alpha1 native parity record mismatch: {stem}")
        return cached
    budget.check(f"full Linear GPTQ {stem}/alpha={alpha:.1f}")
    result, gptq_seconds = timed(lambda: optimized_dynamic_gptq(
        wprime, prepared, alpha=alpha, check_cholesky=(alpha == 1.0)))
    if alpha == 1.0:
        budget.check(f"native scaled alpha1 parity {stem}")
        native, parity_seconds = timed(lambda: native_dynamic_gptq(
            wprime, h, nsamples=int(h_meta["nsamples"]), alpha=1.0))
        parity = equal_result_fields(result, native)
        parity_record = {**common, "status": "PASS" if all(parity.values()) else "FAIL",
                         "fields_exact": parity, "native_parity_seconds": parity_seconds,
                         "optimized_alpha1_full_weight_sha256": sha_tensor(result.weight_fp16),
                         "native_alpha1_full_weight_sha256": sha_tensor(native.weight_fp16)}
        dump(HERE / "raw/candidates" / f"{stem}_native_alpha1_parity.json", parity_record)
        budget.record("native_scaled_alpha1_parity", stem=stem, seconds=parity_seconds,
                      status=parity_record["status"])
        if not all(parity.values()):
            raise RuntimeError(f"scaled alpha1 native parity FAIL: {stem}: {parity}")
        del native
    qrows = result.weight_fp16[rows].detach()
    scores, score_seconds = timed(lambda: score_scaled(x_mm, DOCS[split]["search"],
                                                        wprime, qrows, d, rows, search_offsets))
    if not torch.isfinite(qrows).all().item() or not all(math.isfinite(x) for x in scores.values()):
        raise RuntimeError(f"nonfinite scaled candidate/score: {stem}/{alpha}")
    obj = {"weight_fp16": qrows.cpu().contiguous(),
           "scales": result.scales[rows].detach().cpu().contiguous(),
           "zeros": result.zeros[rows].detach().cpu().contiguous(),
           "g_idx": result.g_idx.detach().cpu().contiguous()}
    m = {**common, "alpha": alpha, "search_mse_by_row": {str(r): scores[r] for r in rows},
         "weight_sha256": sha_tensor(obj["weight_fp16"]),
         "scales_sha256": sha_tensor(obj["scales"]),
         "zeros_sha256": sha_tensor(obj["zeros"]), "g_idx_sha256": sha_tensor(obj["g_idx"]),
         "gptq_seconds": gptq_seconds, "search_score_seconds": score_seconds,
         "candidate_status": "COMPLETE", "candidate_weight_dtype": "torch.float16",
         "range_and_compensation": "native dynamic G128 with alpha hook; original GPTQ compensation"}
    pt, jp = candidate_paths(stem, alpha)
    io_started = time.perf_counter()
    save_pt(pt, obj); dump(jp, m)
    m["checkpoint_io_seconds"] = time.perf_counter() - io_started
    dump(jp, m)
    budget.record("scaled_candidate_saved", stem=stem, alpha=alpha,
                  GPTQ_seconds=gptq_seconds, search_score_seconds=score_seconds,
                  io_seconds=m["checkpoint_io_seconds"], Q_sha256=m["weight_sha256"])
    return obj, m


def search_lock(layer: int, projection: str, split: str, candidates: dict[float, tuple[dict, dict]],
                d_meta: dict, h_meta: dict, budget: Budget) -> dict:
    stem = target_stem(layer, projection, split)
    alphas = {}
    scores = {}
    for row in ROWS[projection]:
        values = {a: float(candidates[a][1]["search_mse_by_row"][str(row)]) for a in ALPHAS}
        lowest = min(values.values())
        selected = max(a for a, value in values.items() if value == lowest)
        alphas[str(row)] = selected
        scores[str(row)] = lowest
    lock = {"status": "LOCKED_BEFORE_HELD_SCORING", "protocol_sha256": sha_file(PROTOCOL_PATH),
            "stem": stem, "search_doc_indices": DOCS[split]["search"],
            "D_sha256": d_meta["D_sha256"], "H_sha256": h_meta["H_sha256"],
            "selected_alpha_by_row": alphas, "selected_search_mse_by_row": scores,
            "candidate_weight_sha256_by_alpha": {f"{a:.1f}": candidates[a][1]["weight_sha256"] for a in ALPHAS}}
    path = HERE / "raw" / f"{stem}_SEARCH_LOCK.json"
    if path.is_file():
        prior = json.loads(path.read_text("utf-8"))
        if prior != lock:
            raise RuntimeError(f"search lock changed on resume: {stem}")
    else:
        dump(path, lock)
        budget.record("scaled_search_lock_saved", stem=stem, selected_alphas=alphas)
    return lock


def held_result(layer: int, projection: str, split: str, w: torch.Tensor,
                wprime: torch.Tensor, d: torch.Tensor, x_mm: np.memmap,
                d_meta: dict, h_meta: dict, candidates: dict[float, tuple[dict, dict]],
                lock: dict, budget: Budget) -> dict:
    stem = target_stem(layer, projection, split)
    path = HERE / "raw" / f"{stem}_HELD_RESULTS.json"
    if path.is_file():
        m = json.loads(path.read_text("utf-8"))
        if m["protocol_sha256"] != sha_file(PROTOCOL_PATH) or m["search_lock_sha256"] != sha_file(HERE / "raw" / f"{stem}_SEARCH_LOCK.json"):
            raise RuntimeError(f"held result protocol/search lock changed: {stem}")
        return m
    budget.check(f"held score {stem}")
    held_indices = DOCS[split]["held"]
    rows = ROWS[projection]
    (offset_data, drift), offset_seconds = timed(lambda: offset_chunks(
        x_mm, held_indices, w, wprime, d, rows))
    q10 = candidates[1.0][0]["weight_fp16"].to(DEVICE)
    q11 = torch.stack([candidates[float(lock["selected_alpha_by_row"][str(r)])][0]["weight_fp16"][j]
                       for j, r in enumerate(rows)]).to(DEVICE)
    g10, g10_seconds = timed(lambda: score_scaled(x_mm, held_indices, wprime, q10, d, rows, offset_data))
    g11, g11_seconds = timed(lambda: score_scaled(x_mm, held_indices, wprime, q11, d, rows, offset_data))
    baseline = baseline_map()
    records = []
    for r in rows:
        alpha01 = float(baseline[(split, layer, projection, r, 1.0)]["original_selected_alpha"])
        g00 = float(baseline[(split, layer, projection, r, 1.0)]["held_mse"])
        g01 = float(baseline[(split, layer, projection, r, alpha01)]["held_mse"])
        records.append({"row": r, "G00_alpha": 1.0, "G00_held_mse": g00,
                        "G01_alpha": alpha01, "G01_held_mse": g01,
                        "G10_alpha": 1.0, "G10_held_mse": g10[r],
                        "G11_alpha": float(lock["selected_alpha_by_row"][str(r)]), "G11_held_mse": g11[r],
                        "G10_search_mse": float(candidates[1.0][1]["search_mse_by_row"][str(r)]),
                        "G11_search_mse": float(lock["selected_search_mse_by_row"][str(r)])})
    m = {"status": "COMPLETE", "protocol_sha256": sha_file(PROTOCOL_PATH), "stem": stem,
         "D_sha256": d_meta["D_sha256"], "H_sha256": h_meta["H_sha256"],
         "search_lock_sha256": sha_file(HERE / "raw" / f"{stem}_SEARCH_LOCK.json"),
         "held_doc_indices": held_indices, "rows": records,
         "held_offset_build_seconds": offset_seconds, "G10_score_seconds": g10_seconds,
         "G11_score_seconds": g11_seconds, "held_coordinate_drift": drift,
         "G10_weight_rows_sha256": sha_tensor(q10), "G11_stitched_weight_rows_sha256": sha_tensor(q11)}
    dump(path, m)
    budget.record("held_result_saved", stem=stem, offset_seconds=offset_seconds,
                  G10_score_seconds=g10_seconds, G11_score_seconds=g11_seconds)
    return m


def run_preflight(budget: Budget) -> None:
    if (HERE / "PREFLIGHT.json").is_file():
        prior = json.loads((HERE / "PREFLIGHT.json").read_text("utf-8"))
        if prior.get("status") != "PASS" or prior.get("protocol_sha256") != sha_file(PROTOCOL_PATH):
            raise RuntimeError("existing preflight is not PASS for frozen protocol")
        return
    identity = [identity_preflight(0, projection, budget) for projection in ("up_proj", "down_proj")]
    equivalence = []
    for projection in ("up_proj", "down_proj"):
        x_mm = input_mm(0, projection)
        w = pristine_weight(0, projection, "S0")
        d, d_meta = load_or_build_scale(0, projection, "S0", x_mm, budget)
        spot, seconds = timed(lambda: equivalence_spot(w, d, x_mm, ROWS[projection], "S0"))
        spot.update({"stem": target_stem(0, projection, "S0"), "D_sha256": d_meta["D_sha256"],
                     "spot_seconds": seconds, "D_min": d_meta["D_min"], "D_max": d_meta["D_max"]})
        # FP64 cancellation is a numerical spot, not an exact-real proof; guard against gross axis mistakes.
        if spot["fp64_max_abs"] > 1e-9 or spot["fp32_max_abs"] > 1e-3 or not math.isfinite(spot["fp16_cast_mse"]):
            raise RuntimeError(f"unquantized coordinate equivalence FAIL: {spot['stem']}: {spot}")
        equivalence.append(spot)
        budget.record("coordinate_equivalence_spot_pass", stem=spot["stem"], seconds=seconds,
                      fp64_max_abs=spot["fp64_max_abs"], fp32_max_abs=spot["fp32_max_abs"],
                      fp16_cast_mse=spot["fp16_cast_mse"])
    # One real scaled-coordinate full Linear checks native alpha=1 and candidate-order state independence.
    layer, projection, split = 0, "up_proj", "S0"
    stem = target_stem(layer, projection, split)
    x_mm = input_mm(layer, projection)
    w = pristine_weight(layer, projection, split)
    d, d_meta = load_or_build_scale(layer, projection, split, x_mm, budget)
    wprime = transformed_weight(w, d)
    wprime_sha = sha_tensor(wprime)
    h, nsamples, h_meta = load_or_build_scaled_h(layer, projection, split, wprime, d, d_meta, x_mm, budget)
    h_sha = sha_tensor(h)
    prepared, factor_seconds = timed(lambda: prepare_native_state(h, nsamples=nsamples))
    budget.record("scaled_preflight_factor", stem=stem, seconds=factor_seconds)
    offsets, drift = offset_chunks(x_mm, DOCS[split]["search"], w, wprime, d, ROWS[projection])
    first = get_candidate(layer, projection, split, 1.0, wprime, w, d, prepared, h, h_meta,
                          d_meta, x_mm, offsets, budget)
    second = get_candidate(layer, projection, split, 0.8, wprime, w, d, prepared, h, h_meta,
                           d_meta, x_mm, offsets, budget)
    repeated, repeat_seconds = timed(lambda: optimized_dynamic_gptq(wprime, prepared, alpha=1.0, check_cholesky=True))
    repeat_equal = {"weight_fp16": torch.equal(repeated.weight_fp16[ROWS[projection]].cpu(), first[0]["weight_fp16"]),
                    "scales": torch.equal(repeated.scales[ROWS[projection]].cpu(), first[0]["scales"]),
                    "zeros": torch.equal(repeated.zeros[ROWS[projection]].cpu(), first[0]["zeros"]),
                    "g_idx": torch.equal(repeated.g_idx.cpu(), first[0]["g_idx"])}
    if not all(repeat_equal.values()) or sha_tensor(wprime) != wprime_sha or sha_tensor(h) != h_sha:
        raise RuntimeError(f"candidate order/state independence FAIL: {stem}: {repeat_equal}")
    budget.record("candidate_order_state_pass", stem=stem, seconds=repeat_seconds)
    preflight = {"status": "PASS", "protocol_sha256": sha_file(PROTOCOL_PATH),
                 "D_identity": identity, "unquantized_coordinate_equivalence": equivalence,
                 "scaled_alpha1_native_parity": json.loads((HERE / "raw/candidates" / f"{stem}_native_alpha1_parity.json").read_text("utf-8"))["status"],
                 "state_independence_spot": {"stem": stem, "order": [1.0, 0.8, 1.0],
                                             "repeated_alpha1_equal": repeat_equal,
                                             "pristine_Wprime_unchanged": True, "scaled_H_unchanged": True,
                                             "additional_full_linear_GPTQ_calls": 1},
                 "preflight_candidate_alphas_banked": [1.0, 0.8],
                 "search_full_16_document_fp16_coordinate_drift": drift,
                 "no_held_data_used": True, "new_model_forwards": 0}
    dump(HERE / "PREFLIGHT.json", preflight)
    budget.record("preflight_pass", D_identity_targets=[v["stem"] for v in identity])


def run_formal(budget: Budget) -> None:
    preflight = json.loads((HERE / "PREFLIGHT.json").read_text("utf-8"))
    if preflight.get("status") != "PASS" or preflight["protocol_sha256"] != sha_file(PROTOCOL_PATH):
        raise RuntimeError("v32 D=I and coordinate preflight must PASS before formal")
    completed = []
    for layer in (0, 12, 23):
        for projection in ("up_proj", "down_proj"):
            for split in ("S0", "S1"):
                stem = target_stem(layer, projection, split)
                budget.check(f"formal target {stem}", margin=35.0)
                existing_held = HERE / "raw" / f"{stem}_HELD_RESULTS.json"
                if existing_held.is_file():
                    rec = json.loads(existing_held.read_text("utf-8"))
                    if rec.get("status") != "COMPLETE" or rec.get("protocol_sha256") != sha_file(PROTOCOL_PATH):
                        raise RuntimeError(f"held checkpoint mismatch: {stem}")
                    completed.append(stem)
                    continue
                x_mm = input_mm(layer, projection)
                w = pristine_weight(layer, projection, split)
                original_sha = sha_tensor(w)
                d, d_meta = load_or_build_scale(layer, projection, split, x_mm, budget)
                wprime = transformed_weight(w, d)
                wprime_sha = sha_tensor(wprime)
                h, nsamples, h_meta = load_or_build_scaled_h(layer, projection, split, wprime, d,
                                                              d_meta, x_mm, budget)
                h_sha = sha_tensor(h)
                prepared, factor_seconds = timed(lambda: prepare_native_state(h, nsamples=nsamples))
                budget.record("scaled_factor_prepared", stem=stem, seconds=factor_seconds,
                              prepared_H_sha256=prepared.h_sha256)
                (offsets, search_drift), offset_seconds = timed(lambda: offset_chunks(
                    x_mm, DOCS[split]["search"], w, wprime, d, ROWS[projection]))
                budget.record("search_reference_offset", stem=stem, seconds=offset_seconds,
                              fp16_coordinate_drift=search_drift)
                candidates = {}
                # Alpha=1 first to establish exact native parity before changed-range candidates.
                for alpha in [1.0] + [a for a in ALPHAS if a != 1.0]:
                    candidates[alpha] = get_candidate(layer, projection, split, alpha, wprime, w, d,
                                                       prepared, h, h_meta, d_meta, x_mm, offsets, budget)
                if sha_tensor(w) != original_sha or sha_tensor(wprime) != wprime_sha or sha_tensor(h) != h_sha:
                    raise RuntimeError(f"candidate bank mutated pristine state: {stem}")
                lock = search_lock(layer, projection, split, candidates, d_meta, h_meta, budget)
                held_result(layer, projection, split, w, wprime, d, x_mm, d_meta, h_meta,
                            candidates, lock, budget)
                completed.append(stem)
                budget.record("formal_target_complete", stem=stem, completed_targets=len(completed),
                              GPU_peak_allocated_bytes=int(torch.cuda.max_memory_allocated(DEVICE)),
                              GPU_peak_reserved_bytes=int(torch.cuda.max_memory_reserved(DEVICE)))
                del candidates, offsets, prepared, h, wprime, w, x_mm
    dump(HERE / "RUN_STATUS.json", {"status": "COMPLETE", "protocol_sha256": sha_file(PROTOCOL_PATH),
                                    "targets_completed": completed, "target_count": len(completed),
                                    "candidate_GPTQ_expected": 120,
                                    "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(DEVICE)),
                                    "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(DEVICE)),
                                    "new_model_forwards": 0, "PPL_runs": 0})
    budget.record("all_formal_targets_complete", target_count=len(completed))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("preflight", "formal"))
    args = ap.parse_args()
    if P["status"] != "FROZEN_BEFORE_GPU" or json.loads((HERE / "SOURCE_AUDIT.json").read_text("utf-8"))["status"] != "PASS":
        raise RuntimeError("v32 must have a PASS source audit and frozen protocol")
    budget = Budget(args.mode)
    try:
        torch.cuda.init()
        if torch.cuda.get_device_name(0) != P["model"]["device"]:
            raise RuntimeError("CUDA device differs from frozen protocol")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        if torch.backends.cuda.matmul.allow_tf32 or torch.backends.cudnn.allow_tf32:
            raise RuntimeError("TF32 must remain disabled")
        sync()
        if args.mode == "preflight":
            run_preflight(budget)
        else:
            run_formal(budget)
        budget.finish("COMPLETE" if args.mode == "formal" else "PREFLIGHT_PASS")
        print(json.dumps({"mode": args.mode, "status": budget.status,
                          "cumulative_wall_seconds": budget.elapsed()}, indent=2))
    except Exception as exc:
        if budget.status != "BUDGET_STOP_RESUMABLE":
            budget.finish("FAIL", error=repr(exc))
        raise


if __name__ == "__main__":
    main()
