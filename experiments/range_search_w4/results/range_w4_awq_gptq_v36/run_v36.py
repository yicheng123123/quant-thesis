"""Bounded v36 L12 AWQ-scaling-selector x dynamic GPTQ experiment.

Imports/copies no historical runner with side effects. Reads held token group
files only after that split's atomic selection lock exists. All writes stay in
this v36 directory. The GPTQ algorithm is the pinned v17.3 dynamic wrapper.
"""
from __future__ import annotations

import csv
import gc
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open
from transformers import AutoModelForCausalLM

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
RESULTS = ROOT / "experiments/range_search_w4/results"
V18 = RESULTS / "range_w4_fullmodel_v18"
V171 = RESULTS / "range_w4_m2_parity_v17_1"
V173 = RESULTS / "range_w4_m2_runtime_v17_3"
MODEL_DIR = ROOT / ".hf_cache/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775"
PREFIX_DIR = V18 / "M2/checkpoint/final"
DEVICE = torch.device("cuda")
SEQ = 2048
N_DOCS = 8
ALPHAS = [round(i / 10, 1) for i in range(1, 11)]
RATIOS = [round(i * 0.05, 2) for i in range(20)]
LAYER_INDEX = 12
GROUP_SIZE = 128
SPLITS = {"S0": {"search": "P0", "held": "P1"}, "S1": {"search": "P2", "held": "P3"}}
sys.path.insert(0, str(V173))
sys.path.insert(0, str(V171))
from optimized_native_m2 import optimized_dynamic_gptq, prepare_native_state  # noqa: E402
from native_m2 import native_dynamic_gptq  # noqa: E402
from gptqmodel.quantization import QuantizeConfig  # noqa: E402
from gptqmodel.quantization.gptq import GPTQ  # noqa: E402

PROTOCOL_PATH = HERE / "manifests/FROZEN_PROTOCOL.json"
ARTICLE_PATH = HERE / "manifests/ARTICLE_MANIFEST.json"
TOKEN_DIR = HERE / "cache"
RAW = HERE / "raw"
CACHE = HERE / "cache"
LOGS = HERE / "logs"
FINAL = HERE / "final"
CORRECT = HERE / "correctness"
for d in (RAW, CACHE, LOGS, FINAL, CORRECT, HERE / "checkpoints"):
    d.mkdir(parents=True, exist_ok=True)


def sha_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def sha_tensor(t: torch.Tensor) -> str:
    return sha_bytes(t.detach().cpu().contiguous().numpy().tobytes())


def sha_memmap(mm: np.memmap) -> str:
    h = hashlib.sha256()
    for lo in range(0, mm.shape[0], 2):
        h.update(np.asarray(mm[lo:lo + 2]).tobytes(order="C"))
    return h.hexdigest()


def stable_json(path: Path, obj: dict, *, replace: bool = False) -> None:
    text = json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    if path.exists() and not replace:
        if path.read_text(encoding="utf-8") != text:
            raise RuntimeError(f"immutable v36 artifact mismatch: {path}")
        return
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8", newline="\n")
    os.replace(tmp, path)


def sync() -> None:
    torch.cuda.synchronize(DEVICE)


def peak_reset() -> None:
    sync()
    torch.cuda.reset_peak_memory_stats(DEVICE)


def peak_read() -> tuple[int, int]:
    sync()
    return int(torch.cuda.max_memory_allocated(DEVICE)), int(torch.cuda.max_memory_reserved(DEVICE))


class Budget:
    def __init__(self):
        self.path = RAW / "BUDGET_LEDGER.csv"
        self.state_path = RAW / "RUN_STATE.json"
        self.start = time.perf_counter()
        state = json.loads(self.state_path.read_text(encoding="utf-8"))
        hashes = (sha_file(PROTOCOL_PATH), sha_file(ARTICLE_PATH), sha_file(HERE / "JOB_TABLE.csv"),
                  sha_file(HERE/"manifests/GROUP_TOKEN_CACHE_MANIFEST.json"))
        expected = (state["protocol_sha256"], state["article_manifest_sha256"], state["job_table_sha256"],
                    state["group_token_cache_manifest_sha256"])
        if hashes != expected:
            raise RuntimeError("frozen artifact hash mismatch; refusing GPU run/resume")
        # RUN_STATE.gpu_seconds_used is already cumulative and includes the
        # frozen external probe row; do not charge that probe again on resume.
        self.previous = float(state.get("gpu_seconds_used", state.get("external_cuda_probe_wall_seconds", 0.0)))
        self.limit = 1800.0
        self.stop_at = 1620.0
        self.last_record_used = self.previous
        self.stage = []
        if not self.path.exists():
            with self.path.open("w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(["split", "stage", "task", "wall_seconds", "cumulative_gpu_wall_seconds",
                                        "peak_allocated_bytes", "peak_reserved_bytes", "status", "detail"])

    @property
    def used(self) -> float:
        return self.previous + (time.perf_counter() - self.start)

    def record(self, split: str, stage: str, task: str, seconds: float, status="PASS", detail=""):
        alloc, reserved = peak_read()
        with self.path.open("a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([split, stage, task, f"{seconds:.6f}", f"{self.used:.6f}", alloc, reserved, status, detail])
        self.stage.append({"split": split, "stage": stage, "task": task, "seconds": seconds,
                           "peak_allocated_bytes": alloc, "peak_reserved_bytes": reserved,
                           "status": status, "detail": detail})
        state = json.loads(self.state_path.read_text(encoding="utf-8"))
        state.update({"gpu_seconds_used": round(self.used, 6), "last_stage": stage, "last_task": task,
                      "updated_utc_epoch": time.time()})
        stable_json(self.state_path, state, replace=True)
        self.last_record_used = self.used
        print(f"[v36] {split} {stage} {task} {seconds:.3f}s cumulative={self.used:.1f}/1800s", flush=True)
    def check(self):
        if self.used >= self.stop_at:
            raise BudgetStop(f"compute reserve reached at {self.used:.1f}s; stop before 180s finalization reserve")


class BudgetStop(RuntimeError):
    pass


class CorrectnessFail(RuntimeError):
    pass


def timed(budget: Budget, split: str, stage: str, task: str, fn):
    peak_reset()
    start = time.perf_counter()
    value = fn()
    sync()
    elapsed = time.perf_counter() - start
    budget.record(split, stage, task, elapsed)
    return value, elapsed


def new_mm(path: Path, shape: tuple[int, ...], dtype=np.float16, *, resume=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and resume and path.stat().st_size == int(np.prod(shape) * np.dtype(dtype).itemsize):
        return np.memmap(path, dtype=dtype, mode="r+", shape=shape)
    if path.exists():
        path.unlink()
    return np.memmap(path, dtype=dtype, mode="w+", shape=shape)


def json_tokens(group: str) -> list[dict]:
    path = TOKEN_DIR / f"{group}_tokens.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    group_manifest=json.loads((HERE/"manifests/GROUP_TOKEN_CACHE_MANIFEST.json").read_text(encoding="utf-8"))
    if sha_file(path)!=group_manifest["groups"][group]["sha256"]:
        raise CorrectnessFail(f"frozen per-group token cache hash mismatch: {group}")
    rows = json.loads(path.read_text(encoding="utf-8"))
    for row in rows:
        if len(row["token_ids"]) != SEQ or sha_bytes(json.dumps(row["token_ids"]).encode("utf-8")) != row["token_sha256"]:
            raise CorrectnessFail(f"token cache integrity failed: {group}/{row.get('source_article_index')}")
    return rows


def qcfg():
    return QuantizeConfig(bits=4, group_size=GROUP_SIZE, sym=True, desc_act=False,
                          static_groups=False, true_sequential=True,
                          damp_percent=0.1, damp_auto_increment=0, mse=0)


def gptq_geometry(weight: torch.Tensor):
    mod = torch.nn.Linear(weight.shape[1], weight.shape[0], bias=False,
                          device="meta", dtype=weight.dtype)
    mod.weight = torch.nn.Parameter(weight.detach(), requires_grad=False)
    core = GPTQ(mod, qcfg=qcfg())
    core.quantizer.configure(perchannel=True, bits=4, sym=True)
    return mod, core


@torch.inference_mode()
def build_hessian(weight: torch.Tensor, x_mm: np.memmap) -> tuple[torch.Tensor, int]:
    _, core = gptq_geometry(weight)
    core.H = torch.zeros((weight.shape[1], weight.shape[1]), dtype=torch.float32, device=DEVICE)
    core.nsamples = 0
    for i in range(N_DOCS):
        x = torch.from_numpy(np.array(x_mm[i], copy=True)).unsqueeze(0).to(DEVICE)
        core.process_batch(x)
        del x
    sync()
    h = core.H.detach().clone()
    nsamples = int(core.nsamples)
    if nsamples != N_DOCS or h.dtype != torch.float32:
        raise CorrectnessFail(f"native GPTQ H aggregation mismatch: nsamples={nsamples}")
    return h, nsamples


@torch.inference_mode()
def traced_dynamic_gptq(weight: torch.Tensor, prepared, alpha: float, *, trace=False):
    """v17.3 native path, with passive group-entry snapshots for parity."""
    if weight.dtype != torch.float16 or weight.ndim != 2:
        raise ValueError("expected frozen FP16 [out,in] W")
    module = torch.nn.Linear(weight.shape[1], weight.shape[0], bias=False,
                             device="meta", dtype=weight.dtype)
    module.weight = torch.nn.Parameter(weight.detach(), requires_grad=False)
    core = GPTQ(module, qcfg=qcfg())
    core.quantizer.configure(perchannel=True, bits=4, sym=True)
    core.H = prepared.hessian.clone()
    core.nsamples = prepared.nsamples

    def cached_inverse(this, lower, upper=False, block_size=512):
        if alpha == 1.0 and not torch.equal(lower, prepared.cholesky):
            raise CorrectnessFail("alpha=1 candidate Cholesky differs from frozen prepared H")
        return prepared.inverse_before_final_cholesky
    from types import MethodType
    core.block_cholesky_inverse = MethodType(cached_inverse, core)
    entries = []
    if alpha != 1.0 or trace:
        original_find_params = core.quantizer.find_params
        calls = 0
        def range_hook(current, weight=False):
            nonlocal calls
            original_find_params(current, weight=weight)
            if calls > 0:
                if trace:
                    entries.append(current.detach().cpu().contiguous().clone())
                if alpha != 1.0:
                    core.quantizer.scale = core.quantizer.scale * alpha
            calls += 1
        core.quantizer.find_params = range_hook
    q, scale, zero, g_idx, *_ = core.quantize(blocksize=128)
    return {"weight": q, "scale": scale, "zero": zero, "g_idx": g_idx, "group_entries": entries}


@torch.inference_mode()
def prepare_prefix_inputs(model, group: str, budget: Budget, split: str):
    rows = json_tokens(group)  # called only with search group until lock.
    stem = f"{split}_{group}"
    hidden_path = CACHE / f"{stem}_hidden_l12.f16"
    prenorm_path = CACHE / f"{stem}_prenorm.f16"
    x_path = CACHE / f"{stem}_postnorm.f16"
    shape = (N_DOCS, SEQ, int(model.config.hidden_size))
    paths = (hidden_path, prenorm_path, x_path)
    side = RAW / f"{stem}_capture.json"
    if side.exists():
        meta = json.loads(side.read_text(encoding="utf-8"))
        if meta.get("token_group_sha256") != sha_file(TOKEN_DIR / f"{group}_tokens.json"):
            raise CorrectnessFail("prefix capture resume token manifest mismatch")
        if all(p.is_file() and p.stat().st_size == np.prod(shape)*2 for p in paths):
            h = [np.memmap(p, dtype=np.float16, mode="r", shape=shape) for p in paths]
            hashes = [sha_memmap(x) for x in h]
            if hashes == [meta["hidden_sha256"], meta["prenorm_sha256"], meta["postnorm_sha256"]]:
                budget.record(split, "prefix_capture_resume_cache_hit", group, 0.0)
                return tuple(h)
            for mm in h:
                mm._mmap.close()
    hidden_mm, pre_mm, x_mm = [new_mm(p, shape) for p in paths]
    layer12 = model.model.layers[LAYER_INDEX]
    active = {"pre": None, "post": None}
    def pre_hook(_m, args):
        active["pre"] = args[0].detach()
    def post_hook(_m, _args, out):
        active["post"] = out.detach()
    hpre = layer12.post_attention_layernorm.register_forward_pre_hook(pre_hook)
    hpost = layer12.post_attention_layernorm.register_forward_hook(post_hook)
    pos = torch.arange(SEQ, device=DEVICE)[None]
    start = time.perf_counter(); peak_reset()
    try:
        for i, row in enumerate(rows):
            ids = torch.tensor(row["token_ids"], dtype=torch.long, device=DEVICE)[None]
            with torch.inference_mode():
                h = model.model.embed_tokens(ids)
                for li in range(LAYER_INDEX + 1):
                    layer = model.model.layers[li]
                    if li == LAYER_INDEX:
                        hidden_mm[i] = h[0].to("cpu").numpy()
                    pos_emb = model.model.rotary_emb(h, pos)
                    h = layer(h, attention_mask=None, position_ids=pos,
                              cache_position=pos[0], position_embeddings=pos_emb,
                              use_cache=False)[0]
                    if li == LAYER_INDEX:
                        if active["pre"] is None or active["post"] is None:
                            raise CorrectnessFail("L12 post-attention norm capture hook did not run")
                        pre_mm[i] = active["pre"][0].to("cpu").numpy()
                        x_mm[i] = active["post"][0].to("cpu").numpy()
                        active.update(pre=None, post=None)
            del ids, h
    finally:
        hpre.remove(); hpost.remove()
    for mm in (hidden_mm, pre_mm, x_mm):
        mm.flush()
    sync(); elapsed = time.perf_counter()-start
    budget.record(split, "prefix_activation_capture", group, elapsed,
                  detail=f"8 complete articles x 2048; prefix v18 M2 through L12 attention; held group not read")
    meta = {"split": split, "group": group, "token_group_sha256": sha_file(TOKEN_DIR / f"{group}_tokens.json"),
            "hidden_sha256": sha_memmap(hidden_mm), "prenorm_sha256": sha_memmap(pre_mm),
            "postnorm_sha256": sha_memmap(x_mm), "shape": list(shape), "dtype": "float16"}
    stable_json(side, meta)
    budget.check()
    return hidden_mm, pre_mm, x_mm


def load_base_target() -> dict[str, torch.Tensor]:
    keys = {"norm": "model.layers.12.post_attention_layernorm.weight",
            "gate": "model.layers.12.mlp.gate_proj.weight",
            "up": "model.layers.12.mlp.up_proj.weight",
            "down": "model.layers.12.mlp.down_proj.weight"}
    out = {}
    with safe_open(str(MODEL_DIR / "model.safetensors"), framework="pt", device="cpu") as f:
        for name, key in keys.items():
            out[name] = f.get_tensor(key).to(dtype=torch.float16, device=DEVICE).contiguous()
    expected = {"norm": (896,), "gate": (4864,896), "up": (4864,896), "down": (896,4864)}
    if {k: tuple(v.shape) for k,v in out.items()} != expected:
        raise CorrectnessFail("pristine target tensor shapes differ from v34 protocol")
    return out


def set_weight(module, weight: torch.Tensor):
    with torch.no_grad():
        module.weight.copy_(weight)


def rms_norm_fold_input(layer, pre_mm: np.memmap, d: torch.Tensor,
                        base_norm: torch.Tensor, *, docs: range | list[int] | None = None) -> np.memmap:
    if docs is None:
        docs = range(N_DOCS)
    shape = (len(docs), SEQ, pre_mm.shape[-1])
    out = np.empty(shape, dtype=np.float16)
    norm = layer.post_attention_layernorm
    old = norm.weight.detach().clone()
    folded = (base_norm.float() / d.float()).to(torch.float16)
    set_weight(norm, folded)
    try:
        for oi, i in enumerate(docs):
            raw = torch.from_numpy(np.array(pre_mm[i], copy=True)).unsqueeze(0).to(DEVICE)
            with torch.inference_mode():
                val = norm(raw)
            out[oi] = val[0].cpu().numpy()
            del raw, val
    finally:
        set_weight(norm, old)
    return out


def weight_prime(w: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
    return (w.float() * d.float().view(1, -1)).to(torch.float16).contiguous()


def awq_rtn_asym(weight: torch.Tensor, group_size=128) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    rows, cols = weight.shape
    if cols % group_size:
        raise ValueError("AWQ proxy group does not divide the input width")
    view = weight.float().reshape(rows, cols // group_size, group_size)
    wmin = view.amin(dim=-1, keepdim=True)
    wmax = view.amax(dim=-1, keepdim=True)
    scale = (wmax - wmin).clamp(min=1e-5) / 15.0
    zero = (-torch.round(wmin / scale)).clamp(0, 15)
    q = (torch.round(view / scale) + zero).clamp(0, 15)
    deq = ((q - zero) * scale).reshape(rows, cols).to(torch.float16).contiguous()
    return deq, scale.squeeze(-1).contiguous(), zero.squeeze(-1).contiguous()


def collect_flat(mm: np.memmap, device=DEVICE) -> torch.Tensor:
    return torch.from_numpy(np.array(mm, copy=True)).to(device).reshape(-1, mm.shape[-1])


@torch.inference_mode()
def awq_select(layer, x_mm, pre_mm, base, split: str, budget: Budget):
    candidate_path = RAW / f"{split}_awq_ratio_scores.csv"
    best_path = RAW / f"{split}_D_selection.json"
    d_bank_path = CACHE / f"{split}_D_ratio_bank.pt"
    if best_path.exists() and d_bank_path.exists():
        info = json.loads(best_path.read_text(encoding="utf-8"))
        bank = torch.load(d_bank_path, map_location=DEVICE, weights_only=True)
        if not candidate_path.is_file():
            raise CorrectnessFail("AWQ selector resume has selection/bank but no frozen ratio scores")
        with candidate_path.open(newline="",encoding="utf-8") as f:
            cached_ratios=list(csv.DictReader(f))
        cached_d=bank["selected_D"].to(DEVICE)
        if (info["article_group"] != SPLITS[split]["search"] or info["x_sha256"] != sha_memmap(x_mm)
                or info.get("pre_norm_sha256") != sha_memmap(pre_mm)
                or bank.get("source_x_sha256") != sha_memmap(x_mm)
                or bank.get("ratios") != RATIOS or len(cached_ratios) != len(RATIOS)
                or info.get("selected_D_sha256") != sha_tensor(cached_d)):
            raise CorrectnessFail("AWQ selector cache/search-input provenance mismatch")
        if [float(r["ratio"]) for r in cached_ratios] != RATIOS:
            raise CorrectnessFail("AWQ selector cached ratio grid/order mismatch")
        return cached_d, info

    # AWQ get_act_scale: flatten token dimensions, reduce in captured activation dtype.
    x16 = collect_flat(x_mm)
    act = x16.abs().mean(dim=0)
    if act.dtype != x16.dtype or not torch.isfinite(act).all():
        raise CorrectnessFail("AWQ activation statistic dtype or finite check failed")
    layernorm = layer.post_attention_layernorm
    mlp = layer.mlp
    original_module_weights = {n: layer.mlp.get_submodule(n).weight.detach().clone()
                               for n in ("gate_proj", "up_proj", "down_proj")}
    original_norm = layernorm.weight.detach().clone()
    # Original target MLP result, used as the single search objective reference.
    set_weight(layernorm, base["norm"])
    for n, k in (("gate_proj", "gate"), ("up_proj", "up"), ("down_proj", "down")):
        set_weight(mlp.get_submodule(n), base[k])
    ref = mlp(x16)
    if ref.shape != (N_DOCS*SEQ, 896):
        raise CorrectnessFail("AWQ selector reference full-MLP shape mismatch")
    score_rows = []
    d_values = []
    best_loss = float("inf"); best_i = None; best_d = None; best_candidate_hash = None
    start_total = time.perf_counter()
    try:
        for ri, ratio in enumerate(RATIOS):
            ratio_start=time.perf_counter()
            # Keep every scale operation in the captured FP16 activation dtype.
            d = act.pow(ratio).clamp(min=1e-4)
            d = d / torch.sqrt(d.max() * d.min())
            d = d.to(dtype=x16.dtype).contiguous()
            d_values.append(d.detach().cpu())
            # Source-compatible fold: norm weight and gate/up columns scale in FP16.
            folded_norm = (base["norm"].float() / d.float()).to(torch.float16)
            set_weight(layernorm, folded_norm)
            pre = collect_flat(pre_mm)
            xprime = layernorm(pre)
            wgate = weight_prime(base["gate"], d)
            wup = weight_prime(base["up"], d)
            qgate, sg, zg = awq_rtn_asym(wgate)
            qup, su, zu = awq_rtn_asym(wup)
            set_weight(mlp.gate_proj, qgate); set_weight(mlp.up_proj, qup)
            set_weight(mlp.down_proj, base["down"])
            candidate = mlp(xprime)
            loss = float((candidate.float() - ref.float()).square().mean().item())
            if not np.isfinite(loss):
                raise CorrectnessFail(f"nonfinite AWQ ratio loss split={split} ratio={ratio}")
            candidate_hash = sha_tensor(qgate) + sha_tensor(qup)
            score_rows.append({"split": split, "search_group": SPLITS[split]["search"], "ratio": ratio,
                               "loss_fp32_mse": loss, "D_sha256": sha_tensor(d),
                               "gate_rtn_weight_sha256": sha_tensor(qgate), "up_rtn_weight_sha256": sha_tensor(qup),
                               "gate_scale_sha256": sha_tensor(sg), "gate_zero_sha256": sha_tensor(zg),
                               "up_scale_sha256": sha_tensor(su), "up_zero_sha256": sha_tensor(zu),
                               "candidate_output_shape": list(candidate.shape), "evaluated_tokens": int(candidate.shape[0]),
                               "ratio_wall_seconds":time.perf_counter()-ratio_start})
            if loss < best_loss:  # upstream strict comparison; ascending grid keeps smaller ratio on exact tie.
                best_loss, best_i, best_d, best_candidate_hash = loss, ri, d.detach().clone(), candidate_hash
            del pre, xprime, wgate, wup, qgate, qup, candidate, sg, zg, su, zu
    finally:
        set_weight(layernorm, original_norm)
        for n, value in original_module_weights.items():
            set_weight(mlp.get_submodule(n), value)
    if best_d is None or best_i is None:
        raise CorrectnessFail("AWQ ratio selector did not produce a choice")
    elapsed = time.perf_counter()-start_total
    budget.record(split, "AWQ_RTN_selector_20_ratio_full_MLP", "gate_up_shared_D", elapsed,
                  detail="20 fixed ratios; 8 search whole articles; 16,384 tokens; no held opened")
    if candidate_path.exists():
        raise RuntimeError(f"refusing to overwrite AWQ ratio scores: {candidate_path}")
    with candidate_path.open("w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=list(score_rows[0]))
        wr.writeheader(); wr.writerows(score_rows)
    torch.save({"ratios": RATIOS, "D_bank": torch.stack(d_values), "selected_index": best_i,
                "selected_D": best_d.detach().cpu(), "act_mean": act.detach().cpu(),
                "source_x_sha256": sha_memmap(x_mm)}, d_bank_path)
    info = {"split": split, "article_group": SPLITS[split]["search"], "selected_ratio": RATIOS[best_i],
            "selected_index": best_i, "selected_loss_fp32_mse": best_loss, "selected_D_sha256": sha_tensor(best_d),
            "selected_candidate_gate_up_hash": best_candidate_hash, "activation_stat_dtype": str(act.dtype),
            "D_dtype": str(best_d.dtype), "x_sha256": sha_memmap(x_mm), "pre_norm_sha256": sha_memmap(pre_mm),
            "tie_rule": "strict less-than, ascending ratio; exact tie retains smaller ratio",
            "scored_tokens": int(x16.shape[0]), "ratio_candidates": 20,
            "wall_seconds": elapsed}
    stable_json(best_path, info)
    budget.check()
    del ref, x16, act
    return best_d, info


@torch.inference_mode()
def make_xprime(layer, pre_mm: np.memmap, base_norm: torch.Tensor, d: torch.Tensor, path: Path):
    shape = (N_DOCS, SEQ, pre_mm.shape[-1])
    xprime = new_mm(path, shape, resume=True)
    old = layer.post_attention_layernorm.weight.detach().clone()
    folded = (base_norm.float() / d.float()).to(torch.float16)
    set_weight(layer.post_attention_layernorm, folded)
    try:
        for i in range(N_DOCS):
            pre = torch.from_numpy(np.array(pre_mm[i], copy=True)).unsqueeze(0).to(DEVICE)
            with torch.inference_mode():
                xp = layer.post_attention_layernorm(pre)
            xprime[i] = xp[0].cpu().numpy()
            del pre, xp
    finally:
        set_weight(layer.post_attention_layernorm, old)
    xprime.flush()
    return xprime


@torch.inference_mode()
def score_rows_direct(x_orig_mm, x_coord_mm, w_orig, w_coord, q_coord, *, transformed=False):
    totals = torch.zeros((w_orig.shape[0],), dtype=torch.float64, device=DEVICE)
    for i in range(N_DOCS):
        x = torch.from_numpy(np.array(x_orig_mm[i], copy=True)).to(DEVICE).float().reshape(-1, w_orig.shape[1])
        xc = torch.from_numpy(np.array(x_coord_mm[i], copy=True)).to(DEVICE).float().reshape(-1, w_coord.shape[1])
        if transformed:
            delta = xc @ (q_coord.float() - w_coord.float()).T
            offset = xc @ w_coord.float().T - x @ w_orig.float().T
            err = delta + offset
        else:
            err = xc @ (q_coord.float() - w_coord.float()).T
        totals += err.double().square().sum(dim=0)
        del x, xc, err
        if transformed:
            del delta, offset
    return (totals / float(N_DOCS*SEQ)).detach()


def atomic_torch_save(path: Path, obj) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


@torch.inference_mode()
def run_candidate_bank(split: str, coord: str, linear_name: str, weight_orig: torch.Tensor,
                       weight_coord: torch.Tensor, x_orig_mm: np.memmap, x_coord_mm: np.memmap,
                       hessian: torch.Tensor, nsamples: int, *, transformed: bool, budget: Budget):
    key = f"{split}_{coord}_{linear_name}"
    outpath = RAW / f"bank_{key}.pt"
    progpath = HERE / "checkpoints" / f"{key}_progress.pt"
    common = {"split": split, "coord": coord, "linear": linear_name,
              "weight_orig_sha256": sha_tensor(weight_orig), "weight_coord_sha256": sha_tensor(weight_coord),
              "x_orig_sha256": sha_memmap(x_orig_mm), "x_coord_sha256": sha_memmap(x_coord_mm),
              "H_sha256": sha_tensor(hessian), "nsamples": nsamples, "transformed": transformed}
    if outpath.exists():
        bank = torch.load(outpath, map_location=DEVICE, weights_only=True)
        if bank["common"] != common:
            raise CorrectnessFail(f"completed candidate bank resume provenance mismatch: {key}")
        return bank
    if weight_coord.shape[0] != weight_orig.shape[0]:
        raise CorrectnessFail("coordinate target matrix output dimension changed")
    prep_start=time.perf_counter();peak_reset()
    prepared = prepare_native_state(hessian, nsamples=nsamples)
    sync();prep_seconds=time.perf_counter()-prep_start
    budget.record(split,"GPTQ_damping_factor_preparation",key,prep_seconds,
                  detail=f"H_sha256={common['H_sha256']}; nsamples={nsamples}; cached across alpha bank")
    budget.check()
    rows = int(weight_orig.shape[0])
    best_loss = torch.full((rows,), float("inf"), dtype=torch.float64, device=DEVICE)
    best_alpha = torch.full((rows,), -1, dtype=torch.float64, device=DEVICE)
    best_q = torch.empty_like(weight_coord)
    best_s = None; best_z = None; best_g = None; alpha1 = None
    scores = []; times = []
    next_index = 0
    if progpath.exists():
        p = torch.load(progpath, map_location=DEVICE, weights_only=True)
        if p["common"] != common:
            raise CorrectnessFail(f"partial candidate checkpoint provenance mismatch: {key}")
        next_index = int(p["next_index"])
        best_loss = p["best_loss"].to(DEVICE); best_alpha = p["best_alpha"].to(DEVICE)
        best_q = p["best_q"].to(DEVICE); best_s = p["best_s"].to(DEVICE); best_z = p["best_z"].to(DEVICE)
        best_g = p["best_g"].to(DEVICE); alpha1 = p["alpha1"]
        if alpha1 is not None:
            alpha1 = {k: v.to(DEVICE) for k,v in alpha1.items()}
        scores = p["scores"]; times = p["times"]
    run_order = [1.0] + [a for a in ALPHAS if a != 1.0]
    for ai in range(next_index, len(run_order)):
        alpha = run_order[ai]
        peak_reset(); start = time.perf_counter()
        result = traced_dynamic_gptq(weight_coord, prepared, alpha, trace=(alpha == 1.0))
        sync(); gptq_elapsed = time.perf_counter()-start
        q, scale, zero, gidx = result["weight"], result["scale"], result["zero"], result["g_idx"]
        if q.dtype != torch.float16 or not all(torch.isfinite(v).all() for v in (q,scale,zero)):
            raise CorrectnessFail(f"nonfinite or wrong-dtype GPTQ candidate {key} alpha={alpha}")
        expected_g = torch.arange(weight_coord.shape[1], device=DEVICE, dtype=gidx.dtype)//GROUP_SIZE
        if not torch.equal(gidx, expected_g):
            raise CorrectnessFail(f"unexpected dynamic group IDs {key} alpha={alpha}")
        pristine_hashes = (common["weight_orig_sha256"], common["weight_coord_sha256"], common["H_sha256"])
        if (sha_tensor(weight_orig), sha_tensor(weight_coord), sha_tensor(hessian)) != pristine_hashes:
            raise CorrectnessFail(f"GPTQ candidate mutated pristine state {key} alpha={alpha}")
        score_start = time.perf_counter()
        losses = score_rows_direct(x_orig_mm, x_coord_mm, weight_orig, weight_coord, q,
                                   transformed=transformed)
        sync(); score_elapsed = time.perf_counter()-score_start
        weight_mse = (q.float()-weight_coord.float()).square().mean(dim=1)
        rec = {"alpha": alpha, "losses": losses.detach().cpu(), "weight_mse": weight_mse.detach().cpu(),
               "q_sha256": sha_tensor(q), "scale_sha256": sha_tensor(scale), "zero_sha256": sha_tensor(zero),
               "g_idx_sha256": sha_tensor(gidx), "group_entry_sha256": [sha_tensor(t) for t in result["group_entries"]],
               "gptq_seconds": gptq_elapsed, "score_seconds": score_elapsed}
        scores.append(rec); times.append({"alpha": alpha, "gptq_seconds": gptq_elapsed, "score_seconds": score_elapsed})
        if best_g is None:
            best_g = gidx.detach().clone()
            best_s = torch.zeros_like(scale); best_z = torch.zeros_like(zero)
        elif not torch.equal(best_g, gidx):
            raise CorrectnessFail(f"g_idx differs between candidates {key} alpha={alpha}")
        better = (losses < best_loss) | ((losses == best_loss) & (alpha > best_alpha))
        best_loss[better] = losses[better]
        best_alpha[better] = alpha
        best_q[better] = q[better]
        best_s[better] = scale[better]
        best_z[better] = zero[better]
        if alpha == 1.0:
            alpha1 = {"weight": q.detach().clone(), "scale": scale.detach().clone(),
                      "zero": zero.detach().clone(), "g_idx": gidx.detach().clone(),
                      "losses": losses.detach().clone(), "group_entries": result["group_entries"]}
        budget.record(split, "full_Linear_dynamic_GPTQ_candidate", key, gptq_elapsed,
                      detail=f"alpha={alpha:.1f}; shape={list(weight_coord.shape)}; score_seconds={score_elapsed:.6f}; all rows")
        budget.record(split, "all_row_direct_output_score", key, score_elapsed,
                      detail=f"alpha={alpha:.1f}; rows={rows}; input_tokens={N_DOCS*SEQ}; FP32 GEMM/FP64 SSE")
        # Complete candidate is checkpointed before the next one can start.
        atomic_torch_save(progpath, {"common": common, "next_index": ai+1,
                                    "best_loss": best_loss.detach().cpu(), "best_alpha": best_alpha.detach().cpu(),
                                    "best_q": best_q.detach().cpu(), "best_s": best_s.detach().cpu(),
                                    "best_z": best_z.detach().cpu(), "best_g": best_g.detach().cpu(),
                                    "alpha1": None if alpha1 is None else {k: v for k,v in alpha1.items() if k!="group_entries"},
                                    "alpha1_group_entries": [] if alpha1 is None else alpha1["group_entries"],
                                    "scores": scores, "times": times})
        budget.check()
        if alpha == 1.0:
            # Alpha=1 candidate is checked against an independent native call before
            # any non-native range candidate in this bank is allowed to run.
            native_start = time.perf_counter(); peak_reset()
            native = native_dynamic_gptq(weight_coord, hessian, nsamples=nsamples, alpha=1.0, trace=True)
            sync(); native_seconds = time.perf_counter()-native_start
            native_cmp = compare_parity(alpha1, native)
            score_native_start=time.perf_counter()
            native_loss=score_rows_direct(x_orig_mm,x_coord_mm,weight_orig,weight_coord,native.weight_fp16,
                                          transformed=transformed)
            sync();native_score_seconds=time.perf_counter()-score_native_start
            budget.record(split,"alpha1_parity_direct_row_score",key,native_score_seconds,
                          detail=f"all {rows} rows; separate correctness score, not a main candidate slot")
            native_cmp["direct_row_output_scores"] = torch.equal(native_loss,alpha1["losses"])
            if not native_cmp["direct_row_output_scores"]:
                native_cmp["direct_row_output_max_abs"] = float((native_loss-alpha1["losses"]).abs().max().item())
            budget.record(split, "alpha1_independent_native_parity", key,
                          native_seconds,
                          status="PASS" if all(v for k,v in native_cmp.items() if isinstance(v,bool)) else "FAIL",
                          detail=json.dumps({k:v for k,v in native_cmp.items() if k!="native_seconds"}, sort_keys=True))
            if not all(v for k,v in native_cmp.items() if isinstance(v,bool)):
                raise CorrectnessFail(f"alpha=1 native GPTQ parity failure {key}: {native_cmp}")
            correctness_path = CORRECT/f"{key}_alpha1_native_parity.json"
            stable_json(correctness_path,{"status":"PASS","fields":{k:v for k,v in native_cmp.items() if k!="native_seconds"},
                         "native_reference_full_linear_GPTQ":True,"native_reference_seconds":native_seconds})
            budget.check()
        del result, q, scale, zero, gidx, losses, weight_mse, better
    if alpha1 is None:
        raise CorrectnessFail(f"alpha=1 candidate missing from bank {key}")
    # Reload trace entries for alpha1 from the completed candidate's progress.
    p = torch.load(progpath, map_location=DEVICE, weights_only=True)
    alpha1["group_entries"] = p.get("alpha1_group_entries", [])
    bank = {"common": common, "best_weight": best_q.detach().cpu(), "best_scale": best_s.detach().cpu(),
            "best_zero": best_z.detach().cpu(), "best_g_idx": best_g.detach().cpu(),
            "best_alpha": best_alpha.detach().cpu(), "best_loss": best_loss.detach().cpu(),
            "alpha1": {k: v.detach().cpu() if isinstance(v, torch.Tensor) else v for k,v in alpha1.items()},
            "scores": scores, "times": times}
    atomic_torch_save(outpath, bank)
    progpath.unlink(missing_ok=True)
    score_csv = RAW / f"row_scores_{key}.csv"
    with score_csv.open("w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f); wr.writerow(["split","coordinate","linear","alpha","row","output_mse","weight_mse"])
        for entry in scores:
            for row, (loss, wmse) in enumerate(zip(entry["losses"].tolist(), entry["weight_mse"].tolist())):
                wr.writerow([split, coord, linear_name, entry["alpha"], row, loss, wmse])
    return {k: (v.to(DEVICE) if isinstance(v, torch.Tensor) else v) for k,v in bank.items()}


def candidate_state(bank, *, native: bool):
    if native:
        a = bank["alpha1"]
        return {"weight": a["weight"].to(DEVICE), "scale": a["scale"].to(DEVICE),
                "zero": a["zero"].to(DEVICE), "g_idx": a["g_idx"].to(DEVICE),
                "selected_alpha": torch.ones(a["weight"].shape[0], device=DEVICE),
                "losses": a["losses"].to(DEVICE)}
    return {"weight": bank["best_weight"].to(DEVICE), "scale": bank["best_scale"].to(DEVICE),
            "zero": bank["best_zero"].to(DEVICE), "g_idx": bank["best_g_idx"].to(DEVICE),
            "selected_alpha": bank["best_alpha"].to(DEVICE), "losses": bank["best_loss"].to(DEVICE)}


def compare_parity(candidate, native) -> dict:
    def exact(a, b):
        return torch.equal(a.detach().cpu(), b.detach().cpu())
    return {"weight": exact(candidate["weight"], native.weight_fp16),
            "scale": exact(candidate["scale"], native.scales),
            "zero": exact(candidate["zero"], native.zeros),
            "g_idx": exact(candidate["g_idx"], native.g_idx),
            "group_entry_count": len(candidate.get("group_entries", [])) == len(native.group_inputs),
            "group_entries": (len(candidate.get("group_entries", [])) == len(native.group_inputs) and
                              all(exact(a,b) for a,b in zip(candidate.get("group_entries", []), native.group_inputs)))}


@torch.inference_mode()
def make_down_input(layer, xcoord_mm, gate_state, up_state, base, d: torch.Tensor | None,
                    out_path: Path):
    mlp = layer.mlp
    old = {n: mlp.get_submodule(n).weight.detach().clone() for n in ("gate_proj","up_proj","down_proj")}
    norm_old = layer.post_attention_layernorm.weight.detach().clone()
    if d is None:
        folded_norm = base["norm"]
    else:
        folded_norm = (base["norm"].float()/d.float()).half()
    set_weight(layer.post_attention_layernorm, folded_norm)
    set_weight(mlp.gate_proj, gate_state["weight"])
    set_weight(mlp.up_proj, up_state["weight"])
    out_mm = new_mm(out_path, (N_DOCS, SEQ, 4864), resume=False)
    try:
        for i in range(N_DOCS):
            x = torch.from_numpy(np.array(xcoord_mm[i], copy=True)).unsqueeze(0).to(DEVICE)
            with torch.inference_mode():
                gate = mlp.gate_proj(x)
                up = mlp.up_proj(x)
                y = mlp.act_fn(gate) * up
            out_mm[i] = y[0].cpu().numpy()
            del x, gate, up, y
        out_mm.flush()
    finally:
        for n, value in old.items(): set_weight(mlp.get_submodule(n), value)
        set_weight(layer.post_attention_layernorm, norm_old)
    return out_mm


@torch.inference_mode()
def candidate_and_parity_down(split, arm, coord, xdown, base, state, budget, *, search=True):
    w = base["down"]
    down_path = RAW / f"downbank_{split}_{arm}.pt"
    input_hash = sha_memmap(xdown)
    if down_path.exists():
        cached = torch.load(down_path, map_location=DEVICE, weights_only=True)
        if cached["x_sha256"] != input_hash or cached["weight_sha256"] != sha_tensor(w):
            raise CorrectnessFail(f"down candidate bank cache provenance mismatch {split}/{arm}")
        return cached["state"], cached["H"].to(DEVICE), int(cached["nsamples"]), cached["parity"]
    h_path = CACHE / f"{split}_H_down_{arm}.pt"
    if h_path.exists():
        obj = torch.load(h_path, map_location=DEVICE, weights_only=True)
        if obj["x_sha256"] != input_hash: raise CorrectnessFail("down H cache input mismatch")
        h = obj["H"].to(DEVICE); ns = int(obj["nsamples"])
    else:
        h, ns = build_hessian(w, xdown)
        atomic_torch_save(h_path, {"H":h.cpu(),"nsamples":ns,"x_sha256":input_hash})
    hsha = sha_tensor(h)
    prep_start=time.perf_counter();peak_reset();prepared = prepare_native_state(h, nsamples=ns);sync()
    budget.record(split,"GPTQ_damping_factor_preparation",f"down_{arm}",time.perf_counter()-prep_start,
                  detail=f"H_sha256={hsha}; arm-specific actual down input")
    budget.check()
    order = [1.0] + [a for a in ALPHAS if a != 1.0] if search else [1.0]
    progress_path=HERE/"checkpoints"/f"down_{split}_{arm}_progress.pt"
    common={"split":split,"arm":arm,"coord":coord,"x_sha256":input_hash,"W_sha256":sha_tensor(w),
            "H_sha256":hsha,"nsamples":ns,"search":search}
    best_loss=torch.full((w.shape[0],),float("inf"),device=DEVICE,dtype=torch.float64)
    best_alpha=torch.full((w.shape[0],),-1.,device=DEVICE)
    best_q=torch.empty_like(w); best_s=None;best_z=None;best_g=None;native_state=None
    bank_records=[]; next_idx=0; parity=None
    if progress_path.exists():
        p=torch.load(progress_path,map_location=DEVICE,weights_only=True)
        if p["common"]!=common: raise CorrectnessFail(f"down progress provenance mismatch {split}/{arm}")
        next_idx=int(p["next_index"]);best_loss=p["best_loss"].to(DEVICE);best_alpha=p["best_alpha"].to(DEVICE)
        best_q=p["best_q"].to(DEVICE);best_s=p["best_s"].to(DEVICE);best_z=p["best_z"].to(DEVICE);best_g=p["best_g"].to(DEVICE)
        native_state=p["alpha1"]; bank_records=p["records"]; parity=p["parity"]
        if native_state is not None: native_state={k:(v.to(DEVICE) if isinstance(v,torch.Tensor) else v) for k,v in native_state.items()}
    if native_state is not None and parity is None:
        ref_start=time.perf_counter();peak_reset()
        native=native_dynamic_gptq(w,h,nsamples=ns,alpha=1.0,trace=True);sync(); parity_seconds=time.perf_counter()-ref_start
        parity=compare_parity(native_state,native)
        budget.record(split,"alpha1_independent_native_parity",f"down_{arm}",parity_seconds,
                      status="PASS" if all(v for k,v in parity.items() if isinstance(v,bool)) else "FAIL",detail=json.dumps(parity,sort_keys=True))
        if not all(v for k,v in parity.items() if isinstance(v,bool)): raise CorrectnessFail(f"down alpha1 parity failed {split}/{arm}: {parity}")
        atomic_torch_save(progress_path,{"common":common,"next_index":next_idx,"best_loss":best_loss.cpu(),"best_alpha":best_alpha.cpu(),
            "best_q":best_q.cpu(),"best_s":best_s.cpu(),"best_z":best_z.cpu(),"best_g":best_g.cpu(),"alpha1":native_state,"records":bank_records,"parity":parity})
        budget.check()
    for ai in range(next_idx,len(order)):
        alpha=order[ai]
        start=time.perf_counter();peak_reset();r=traced_dynamic_gptq(w,prepared,alpha,trace=(alpha==1.0));sync();t=time.perf_counter()-start
        score_start=time.perf_counter();loss=score_rows_direct(xdown,xdown,w,w,r["weight"],transformed=False);sync();st=time.perf_counter()-score_start
        if best_g is None:best_g=r["g_idx"].clone();best_s=torch.zeros_like(r["scale"]);best_z=torch.zeros_like(r["zero"])
        elif not torch.equal(best_g,r["g_idx"]):raise CorrectnessFail(f"down g_idx changed {split}/{arm}")
        better=(loss<best_loss)|((loss==best_loss)&(alpha>best_alpha))
        best_loss[better]=loss[better];best_alpha[better]=alpha;best_q[better]=r["weight"][better];best_s[better]=r["scale"][better];best_z[better]=r["zero"][better]
        rec={"alpha":alpha,"losses":loss.detach().cpu(),"weight_mse":(r["weight"].float()-w.float()).square().mean(1).cpu(),
             "q_sha256":sha_tensor(r["weight"]),"scale_sha256":sha_tensor(r["scale"]),"zero_sha256":sha_tensor(r["zero"]),
             "g_idx_sha256":sha_tensor(r["g_idx"]),"gptq_seconds":t,"score_seconds":st}
        bank_records.append(rec)
        if alpha==1.0:native_state={"weight":r["weight"].detach().cpu(),"scale":r["scale"].detach().cpu(),
            "zero":r["zero"].detach().cpu(),"g_idx":r["g_idx"].detach().cpu(),"group_entries":r["group_entries"],"losses":loss.detach().cpu()}
        atomic_torch_save(progress_path,{"common":common,"next_index":ai+1,"best_loss":best_loss.cpu(),"best_alpha":best_alpha.cpu(),
            "best_q":best_q.cpu(),"best_s":best_s.cpu(),"best_z":best_z.cpu(),"best_g":best_g.cpu(),
            "alpha1":native_state,"records":bank_records,"parity":parity})
        budget.record(split,"full_Linear_dynamic_GPTQ_candidate",f"down_{arm}",t,detail=f"alpha={alpha}; shape={list(w.shape)}")
        budget.record(split,"all_row_direct_output_score",f"down_{arm}",st,detail=f"alpha={alpha}; rows=896; tokens={N_DOCS*SEQ}")
        del r,loss,better
        if alpha==1.0 and parity is None:
            ref_start=time.perf_counter();peak_reset()
            native=native_dynamic_gptq(w,h,nsamples=ns,alpha=1.0,trace=True);sync();parity_seconds=time.perf_counter()-ref_start
            parity=compare_parity(native_state,native)
            score_native_start=time.perf_counter()
            native_loss=score_rows_direct(xdown,xdown,w,w,native.weight_fp16,transformed=False)
            sync();native_score_seconds=time.perf_counter()-score_native_start
            budget.record(split,"alpha1_parity_direct_row_score",f"down_{arm}",native_score_seconds,
                          detail="all 896 rows; separate correctness score, not a main candidate slot")
            alpha1_loss=next(rec["losses"] for rec in bank_records if float(rec["alpha"])==1.0).to(DEVICE)
            parity["direct_row_output_scores"]=torch.equal(native_loss,alpha1_loss)
            if not parity["direct_row_output_scores"]:parity["direct_row_output_max_abs"]=float((native_loss-alpha1_loss).abs().max().item())
            budget.record(split,"alpha1_independent_native_parity",f"down_{arm}",parity_seconds,
                          status="PASS" if all(v for k,v in parity.items() if isinstance(v,bool)) else "FAIL",detail=json.dumps(parity,sort_keys=True))
            if not all(v for k,v in parity.items() if isinstance(v,bool)):raise CorrectnessFail(f"down alpha1 parity failed {split}/{arm}: {parity}")
            atomic_torch_save(progress_path,{"common":common,"next_index":ai+1,"best_loss":best_loss.cpu(),"best_alpha":best_alpha.cpu(),
                "best_q":best_q.cpu(),"best_s":best_s.cpu(),"best_z":best_z.cpu(),"best_g":best_g.cpu(),"alpha1":native_state,"records":bank_records,"parity":parity})
        budget.check()
    if native_state is None or parity is None: raise CorrectnessFail("down alpha1 candidate/parity missing")
    selected={"weight":best_q.detach().cpu(),"scale":best_s.detach().cpu(),"zero":best_z.detach().cpu(),
              "g_idx":best_g.detach().cpu(),"selected_alpha":best_alpha.detach().cpu(),"losses":best_loss.detach().cpu(),
              "alpha1":native_state,"H_sha256":hsha,"H_nsamples":ns,"x_sha256":input_hash,
              "scores":bank_records}
    atomic_torch_save(down_path,{"x_sha256":input_hash,"weight_sha256":sha_tensor(w),"H":h.cpu(),
                                "nsamples":ns,"state":selected,"parity":parity})
    with (RAW/f"row_scores_{split}_down_{arm}.csv").open("w",newline="",encoding="utf-8") as f:
        wr=csv.writer(f);wr.writerow(["split","arm","alpha","row","output_mse","weight_mse"])
        for rec in bank_records:
            for row,(lossv,wm) in enumerate(zip(rec["losses"].tolist(),rec["weight_mse"].tolist())):
                wr.writerow([split,arm,rec["alpha"],row,lossv,wm])
    progress_path.unlink(missing_ok=True)
    return selected,h,ns,parity


def project_output(layer, x, weight_name: str):
    return layer.mlp.get_submodule(weight_name)(x)


@torch.inference_mode()
def run_full_mlp_and_boundary(model, hidden_mm, pre_mm, base, states, d, split, budget,
                              *, held: bool, reference: torch.Tensor | None = None):
    """Commit one full stitched arm, score complete MLP, and capture L13 input."""
    layer=model.model.layers[LAYER_INDEX]; mlp=layer.mlp
    original={n:mlp.get_submodule(n).weight.detach().clone() for n in ("gate_proj","up_proj","down_proj")}
    orig_norm=layer.post_attention_layernorm.weight.detach().clone()
    next_norm=model.model.layers[LAYER_INDEX+1].input_layernorm
    arm=states["arm"]
    transformed=d is not None
    ddev=None if d is None else d.to(DEVICE)
    norm_new=base["norm"] if ddev is None else (base["norm"].float()/ddev.float()).half()
    set_weight(layer.post_attention_layernorm,norm_new)
    for name,key in (("gate_proj","gate"),("up_proj","up"),("down_proj","down")):
        set_weight(mlp.get_submodule(name),states[key]["weight"].to(DEVICE))
    committed={n:sha_tensor(mlp.get_submodule(n).weight) for n in ("gate_proj","up_proj","down_proj")}
    if any(committed[n] != sha_tensor(states[key]["weight"]) for n,key in (("gate_proj","gate"),("up_proj","up"),("down_proj","down"))):
        raise CorrectnessFail(f"selected full Linear commit hash mismatch {split}/{arm}")
    shape=(N_DOCS,SEQ,896)
    mlp_out=np.empty(shape,dtype=np.float16)
    block_out=np.empty(shape,dtype=np.float16)
    l13=np.empty(shape,dtype=np.float16)
    local={"mlp":None,"pre":None,"post":None,"down_input":None,"down_output":None}
    def hook(_m,_args,out): local["mlp"]=out.detach()
    hh=mlp.register_forward_hook(hook)
    def norm_pre(_m,args):local["pre"]=args[0].detach()
    def norm_post(_m,_args,out):local["post"]=out.detach()
    def down_pre(_m,args):local["down_input"]=args[0].detach()
    def down_hook(_m,_args,out):local["down_output"]=out.detach()
    hpre=layer.post_attention_layernorm.register_forward_pre_hook(norm_pre)
    hpost=layer.post_attention_layernorm.register_forward_hook(norm_post)
    hdown=mlp.down_proj.register_forward_pre_hook(down_pre)
    hdown_out=mlp.down_proj.register_forward_hook(down_hook)
    pos=torch.arange(SEQ,device=DEVICE)[None]
    start=time.perf_counter(); peak_reset()
    try:
        for i in range(N_DOCS):
            hidden=torch.from_numpy(np.array(hidden_mm[i],copy=True)).unsqueeze(0).to(DEVICE)
            expected_pre=torch.from_numpy(np.array(pre_mm[i],copy=True)).unsqueeze(0).to(DEVICE)
            with torch.inference_mode():
                expected_post=layer.post_attention_layernorm(expected_pre)
                position_embeddings=model.model.rotary_emb(hidden,pos)
                out=layer(hidden,attention_mask=None,position_ids=pos,cache_position=pos[0],
                          position_embeddings=position_embeddings,use_cache=False)[0]
                if local["mlp"] is None: raise CorrectnessFail("full L12 MLP output hook did not run")
                if not torch.equal(local["pre"],expected_pre):raise CorrectnessFail(f"actual L12 pre-norm input changed from prefix capture {split}/{arm}/doc{i}")
                if not torch.equal(local["post"],expected_post):raise CorrectnessFail(f"actual folded post-norm input differs from expected {split}/{arm}/doc{i}")
                if local["down_input"] is None:raise CorrectnessFail("actual down_proj input hook did not run")
                if tuple(local["down_input"].shape)!=(1,SEQ,4864) or not torch.isfinite(local["down_input"]).all():
                    raise CorrectnessFail(f"actual down_proj activation invalid {split}/{arm}/doc{i}")
                if local["down_output"] is None or not torch.equal(local["mlp"],local["down_output"]):
                    raise CorrectnessFail(f"L12 MLP output is not the hooked down_proj output {split}/{arm}/doc{i}")
                mlp_out[i]=local["mlp"][0].cpu().numpy()
                block_out[i]=out[0].cpu().numpy()
                l13[i]=next_norm(out)[0].cpu().numpy()
                local.update(mlp=None,pre=None,post=None,down_input=None,down_output=None)
            del hidden,out,position_embeddings,expected_pre,expected_post
    finally:
        hh.remove();hpre.remove();hpost.remove();hdown.remove();hdown_out.remove()
        for n,value in original.items(): set_weight(mlp.get_submodule(n),value)
        set_weight(layer.post_attention_layernorm,orig_norm)
    elapsed=time.perf_counter()-start; sync()
    budget.record(split,"held_full_mlp_arm_and_L13_boundary" if held else "search_full_mlp_arm_refresh",
                  arm,elapsed,detail="whole L12 block; one commit for entire Linear; no row-batch refresh")
    if reference is not None:
        score_start=time.perf_counter();peak_reset()
        err=torch.from_numpy(mlp_out).to(DEVICE).float()-reference.to(DEVICE).float()
        article_mse=err.square().mean(dim=(1,2)).detach().cpu().tolist()
        perrow_sum=torch.zeros(896,dtype=torch.float64,device=DEVICE)
        err2=err.double().square()
        perrow_sum+=err2.sum(dim=(0,1))
        full_loss=float(err2.mean().item())
        l13_hash=sha_bytes(l13.tobytes())
        state_hashes={"mlp_out_sha256":sha_bytes(mlp_out.tobytes()),"l13_input_sha256":l13_hash,
                      "block_output_sha256":sha_bytes(block_out.tobytes()),"committed_weights":committed}
        sync();budget.record(split,"held_full_mlp_output_scoring",arm,time.perf_counter()-score_start,
                             detail="full-output MSE, article MSE, row MSE; all 8 held articles")
        return {"arm":arm,"full_mlp_mse":full_loss,"article_mse":article_mse,
                "mlp_row_mse":(perrow_sum/(N_DOCS*SEQ)).detach().cpu().numpy().tolist(),
                "boundary":state_hashes,"output":mlp_out,"l13":l13}
    return {"arm":arm,"mlp_out":mlp_out,"block_out":block_out,"l13":l13,
            "boundary":{"mlp_out_sha256":sha_bytes(mlp_out.tobytes()),"l13_input_sha256":sha_bytes(l13.tobytes()),
                        "block_output_sha256":sha_bytes(block_out.tobytes()),"committed_weights":committed}}


def get_layer_module_weight(layer, name: str):
    return layer.mlp.get_submodule(name).weight


def save_lock(split: str, d, d_info, arms, protocol_hash: str):
    lock_path=FINAL/f"{split}_SELECTION_LOCK.json"
    tensors_path=FINAL/f"{split}_selected_tensors.pt"
    if lock_path.exists():
        info=json.loads(lock_path.read_text(encoding="utf-8"))
        if info.get("protocol_sha256")!=protocol_hash: raise CorrectnessFail("selection lock protocol hash mismatch")
        if not tensors_path.is_file() or sha_file(tensors_path)!=info.get("selected_tensor_file_sha256"):
            raise CorrectnessFail("selection lock tensor payload missing or hash mismatch")
        return info
    payload={}
    summary={}
    for arm,state in arms.items():
        payload[arm]={}
        for key in ("gate","up","down"):
            st=state[key]
            payload[arm][key]={k:st[k].detach().cpu() for k in ("weight","scale","zero","g_idx","selected_alpha")}
            summary[f"{arm}.{key}"]={"weight_sha256":sha_tensor(st["weight"]),"scale_sha256":sha_tensor(st["scale"]),
                                     "zero_sha256":sha_tensor(st["zero"]),"g_idx_sha256":sha_tensor(st["g_idx"]),
                                     "alpha_histogram":{str(a):int((st["selected_alpha"]==a).sum().item()) for a in ALPHAS}}
    payload["D"] = None if d is None else d.detach().cpu()
    atomic_torch_save(tensors_path,payload)
    lock={"split":split,"protocol_sha256":protocol_hash,"search_group":SPLITS[split]["search"],
          "held_group_not_opened_before_lock":SPLITS[split]["held"],"D_selection":d_info,
          "selected_tensor_file":tensors_path.name,"selected_tensor_file_sha256":sha_file(tensors_path),
          "selected_tensors":summary,"selection_locked_before_held":True,
          "selection_rule":"D by AWQ search RTN MLP loss; alpha independently per output row by direct local M2 output MSE; exact ties larger alpha"}
    stable_json(lock_path,lock)
    return lock


def protocol_hashes_ok():
    protocol=json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    manifest=json.loads(ARTICLE_PATH.read_text(encoding="utf-8"))
    if protocol["protocol_id"]!="v36_l12_awq_selector_dynamic_gptq_32_new_articles": raise RuntimeError("unexpected frozen protocol")
    if len(manifest["selected_articles"])!=32 or len(manifest["not_used"])!=2: raise RuntimeError("article selection size mismatch")
    for group,ids in (("P0",range(0,8)),("P1",range(8,16)),("P2",range(16,24)),("P3",range(24,32))):
        rows=manifest["groups"][group]
        if [x["eligible_rank_0based"] for x in rows]!=list(ids): raise RuntimeError("frozen split order mismatch")
    return protocol,manifest


def load_prefix_model(budget: Budget):
    start=time.perf_counter(); peak_reset()
    model=AutoModelForCausalLM.from_pretrained(str(PREFIX_DIR),torch_dtype=torch.float16,
            low_cpu_mem_usage=True,local_files_only=True,attn_implementation="sdpa").eval().to(DEVICE)
    sync(); elapsed=time.perf_counter()-start
    budget.record("","model_load","v18_M2_prefix_FP16",elapsed,detail="local-only frozen v18 M2 dense-dequant prefix")
    return model


def validate_d_identity(layer, pre_mm, x_mm, base, budget, split):
    one=torch.ones((896,),dtype=torch.float16,device=DEVICE)
    # D=I reproduces input and all fold parameters exactly.
    folded=(base["norm"].float()/one.float()).half()
    if not torch.equal(folded,base["norm"]): raise CorrectnessFail("D=I norm fold is not exact")
    for name,key in (("gate_proj","gate"),("up_proj","up")):
        if not torch.equal(weight_prime(base[key],one),base[key]): raise CorrectnessFail(f"D=I W fold not exact: {key}")
    norm_old=layer.post_attention_layernorm.weight.detach().clone()
    linear_old={n:layer.mlp.get_submodule(n).weight.detach().clone() for n in ("gate_proj","up_proj","down_proj")}
    layer.post_attention_layernorm.weight.data.copy_(base["norm"])
    for n,k in (("gate_proj","gate"),("up_proj","up"),("down_proj","down")):
        set_weight(layer.mlp.get_submodule(n),base[k])
    for i in (0,):
        pre=torch.from_numpy(np.array(pre_mm[i],copy=True)).unsqueeze(0).to(DEVICE)
        with torch.inference_mode(): x=layer.post_attention_layernorm(pre)
        expected=torch.from_numpy(np.array(x_mm[i],copy=True)).unsqueeze(0).to(DEVICE)
        if not torch.equal(x,expected): raise CorrectnessFail("D=I norm output differs from frozen prefix capture")
        with torch.inference_mode(): y=layer.mlp(x); yref=layer.mlp(expected)
        if not torch.equal(y,yref): raise CorrectnessFail("D=I full target MLP output differs")
    for n,v in linear_old.items():set_weight(layer.mlp.get_submodule(n),v)
    set_weight(layer.post_attention_layernorm,norm_old)
    info={"status":"PASS","D_sha256":sha_tensor(one),"norm_fold_exact":True,
          "gate_W_fold_exact":True,"up_W_fold_exact":True,"postnorm_capture_exact":True,
          "full_mlp_fp16_output_exact":True}
    stable_json(CORRECT/f"{split}_D_identity.json",info)
    budget.record(split,"D_identity_regression","norm+gate_up",0.0,detail="exact FP16 parameter and captured-input identity")


def equivalence_spot(base, x_mm, d, wname, split):
    w=base[wname]
    x=torch.from_numpy(np.array(x_mm[0,:128],copy=True)).to(DEVICE)
    if wname == "down":
        # D folds the postnorm input coordinates of gate/up.  It does not act
        # on down_proj's 4864-wide intermediate coordinates, so use its real
        # pristine MLP input and an explicit identity transform here.
        gate=F.linear(x.float(),base["gate"].float())
        up=F.linear(x.float(),base["up"].float())
        x=F.silu(gate)*up
        d=torch.ones((w.shape[1],),device=DEVICE,dtype=torch.float16)
    d64=d.double(); x64=x.double(); w64=w.double()
    real_a=x64@w64.T; real_b=(x64/d64)@(w64*d64).T
    fp32_a=x.float()@w.float().T; fp32_b=(x.float()/d.float())@(w.float()*d.float()).T
    wp=weight_prime(w,d)
    xp=(x.float()/d.float()).half().float()
    cast=xp@wp.float().T
    return {"split":split,"linear":wname,"tokens":128,"coordinate_scope":"identity_down_input" if wname=="down" else "shared_postnorm_gate_up_input",
            "fp64_max_abs":float((real_a-real_b).abs().max().item()),
            "fp32_max_abs":float((fp32_a-fp32_b).abs().max().item()),
            "fp16_fold_max_abs":float((cast-fp32_a).abs().max().item()),
            "fp16_fold_mse":float((cast-fp32_a).square().mean().item()),
            "Wprime_sha256":sha_tensor(wp),"D_sha256":sha_tensor(d)}


@torch.inference_mode()
def selected_fold_correctness(layer, pre_mm, x_mm, base, d, split, budget):
    mlp=layer.mlp; norm=layer.post_attention_layernorm
    old_norm=norm.weight.detach().clone()
    old={n:mlp.get_submodule(n).weight.detach().clone() for n in ("gate_proj","up_proj","down_proj")}
    for n,k in (("gate_proj","gate"),("up_proj","up"),("down_proj","down")):set_weight(mlp.get_submodule(n),base[k])
    set_weight(norm,base["norm"])
    x=collect_flat(x_mm)
    with torch.inference_mode(): ref=mlp(x)
    set_weight(norm,(base["norm"].float()/d.float()).half())
    xp=norm(collect_flat(pre_mm))
    with torch.inference_mode(): folded=mlp(xp)
    diff=folded.float()-ref.float()
    values={"status":"PASS","split":split,"D_sha256":sha_tensor(d),
            "selected_D_unquantized_full_MLP_fold_mse":float(diff.double().square().mean().item()),
            "selected_D_unquantized_full_MLP_fold_max_abs":float(diff.abs().max().item()),
            "finite":bool(torch.isfinite(diff).all()),"input_tokens":int(x.shape[0]),
            "fold_rule":"norm weight / D and gate/up input columns * D in FP16; actual folded norm re-evaluated"}
    for n,v in old.items():set_weight(mlp.get_submodule(n),v)
    set_weight(norm,old_norm)
    if not values["finite"]:raise CorrectnessFail(f"selected-D FP16 full MLP fold produced nonfinite values {split}")
    stable_json(CORRECT/f"{split}_selected_D_full_mlp_fold.json",values)
    budget.record(split,"correctness_selected_D_unquantized_MLP_fold","gate_up_shared_D",0.0,
                  detail=json.dumps(values,sort_keys=True))
    budget.check()
    return values


def make_arm_state(split, linear_banks, down_states):
    arms={}
    for arm, coord, native in (("G00","original",True),("G01","original",False),
                               ("G10","scaled",True),("G11","scaled",False)):
        gate=candidate_state(linear_banks[(coord,"gate_proj")],native=native)
        up=candidate_state(linear_banks[(coord,"up_proj")],native=native)
        # During the first pass only gate/up outputs are committed in order to
        # capture the arm-specific down input; the full arm state is assembled
        # again with its completed down candidate bank afterwards.
        down=down_states.get(arm)
        arms[arm]={"arm":arm,"gate":gate,"up":up,"down":down}
    return arms


@torch.inference_mode()
def verify_row_stitch_commit(layer, arms, base, d, split):
    mlp=layer.mlp; norm=layer.post_attention_layernorm
    old_norm=norm.weight.detach().clone()
    old={n:mlp.get_submodule(n).weight.detach().clone() for n in ("gate_proj","up_proj","down_proj")}
    record={"split":split,"status":"PASS","arms":{}}
    for arm,state in arms.items():
        norm_w=base["norm"] if arm in ("G00","G01") else (base["norm"].float()/d.float()).half()
        set_weight(norm,norm_w)
        arm_result={}
        for name,key in (("gate_proj","gate"),("up_proj","up"),("down_proj","down")):
            q=state[key]["weight"].to(DEVICE)
            set_weight(mlp.get_submodule(name),q)
            expected=sha_tensor(q);actual=sha_tensor(mlp.get_submodule(name).weight)
            if actual!=expected:raise CorrectnessFail(f"actual selected W commit hash mismatch {split}/{arm}/{name}")
            arm_result[name]={"W_sha256":actual,"scale_sha256":sha_tensor(state[key]["scale"].to(DEVICE)),
                              "zero_sha256":sha_tensor(state[key]["zero"].to(DEVICE)),
                              "g_idx_sha256":sha_tensor(state[key]["g_idx"].to(DEVICE)),
                              "row_count":int(q.shape[0])}
        record["arms"][arm]=arm_result
    for n,v in old.items():set_weight(mlp.get_submodule(n),v)
    set_weight(norm,old_norm)
    return record


def read_actual_group(group: str):
    # Deliberately invoked for held only after FINAL/<split>_SELECTION_LOCK.json exists.
    return json_tokens(group)


@torch.inference_mode()
def eval_held(model, split, group, held_rows, base, arms, d, pre_mm, hidden_mm, x_mm, budget):
    layer=model.model.layers[LAYER_INDEX]; mlp=layer.mlp
    # Reference target uses the pristine FP16 target MLP on the exact frozen prefix input.
    originals={n:mlp.get_submodule(n).weight.detach().clone() for n in ("gate_proj","up_proj","down_proj")}
    norm0=layer.post_attention_layernorm.weight.detach().clone()
    ref_start=time.perf_counter();peak_reset()
    set_weight(layer.post_attention_layernorm,base["norm"])
    for n,k in (("gate_proj","gate"),("up_proj","up"),("down_proj","down")): set_weight(mlp.get_submodule(n),base[k])
    xref=collect_flat(x_mm)
    with torch.inference_mode(): ref=mlp(xref).detach().clone()
    sync();budget.record(split,"held_reference_MLP_forward","pristine_target",time.perf_counter()-ref_start,
                         detail="unquantized target reference on all held L12 postnorm inputs")
    # FP16 unquantized fold error is measured on this held split after the search lock.
    fold_start=time.perf_counter();peak_reset()
    ddev=d.to(DEVICE)
    set_weight(layer.post_attention_layernorm,(base["norm"].float()/ddev.float()).half())
    for n,k in (("gate_proj","gate"),("up_proj","up"),("down_proj","down")): set_weight(mlp.get_submodule(n),base[k])
    xfold=layer.post_attention_layernorm(collect_flat(pre_mm))
    with torch.inference_mode(): folded_ref=mlp(xfold).detach().clone()
    fold_error=float((folded_ref.float()-ref.float()).square().mean().item())
    sync();budget.record(split,"held_selected_D_fold_reference","unquantized_fold",time.perf_counter()-fold_start,
                         detail="FP16-folded norm and original MLP; full held L12 MLP")
    set_weight(layer.post_attention_layernorm,norm0)
    for n,val in originals.items(): set_weight(mlp.get_submodule(n),val)

    article_rows=[]; all_row_metrics=[]; arm_outputs={}; boundary={}
    for arm,state in arms.items():
        result=run_full_mlp_and_boundary(model,hidden_mm,pre_mm,base,state,None if arm in ("G00","G01") else d,
                                         split,budget,held=True,reference=ref.reshape(N_DOCS,SEQ,896))
        arm_outputs[arm]=result["output"]
        boundary[arm]=result["boundary"]
        for i,entry in enumerate(held_rows):
            article_rows.append({"split":split,"group":group,"arm":arm,"article_id":entry["source_article_index"],
                                 "article_sha256":entry["token_sha256"],"full_mlp_mse":result["article_mse"][i]})
        for row,val in enumerate(result["mlp_row_mse"]):
            all_row_metrics.append({"split":split,"arm":arm,"projection":"full_mlp_output","row":row,"mse":val})
    # Complete projection-row tails: gate/up against common original target; down uses each arm's actual MLP input.
    projection_start=time.perf_counter();peak_reset()
    for arm,state in arms.items():
        coord_x=xfold if arm in ("G10","G11") else xref
        g=state["gate"]["weight"].to(DEVICE); u=state["up"]["weight"].to(DEVICE); qd=state["down"]["weight"].to(DEVICE)
        set_weight(layer.post_attention_layernorm, (base["norm"].float()/ddev.float()).half() if arm in ("G10","G11") else base["norm"])
        set_weight(mlp.gate_proj,g); set_weight(mlp.up_proj,u); set_weight(mlp.down_proj,qd)
        with torch.inference_mode():
            qg=mlp.gate_proj(coord_x); qu=mlp.up_proj(coord_x)
            refg=F.linear(xref,base["gate"]); refu=F.linear(xref,base["up"])
            down_in=mlp.act_fn(qg)*qu
            qdown=mlp.down_proj(down_in); refdown=F.linear(down_in,base["down"])
        for projection,a,b in (("gate_proj",qg,refg),("up_proj",qu,refu),("down_proj",qdown,refdown)):
            err=(a.float()-b.float()).reshape(N_DOCS,SEQ,-1).double().square()
            row=err.sum(dim=(0,1))/(N_DOCS*SEQ)
            all_row_metrics.extend({"split":split,"arm":arm,"projection":projection,"row":r,"mse":float(v)}
                                   for r,v in enumerate(row.cpu().tolist()))
        del qg,qu,refg,refu,down_in,qdown,refdown
    for n,val in originals.items(): set_weight(mlp.get_submodule(n),val)
    set_weight(layer.post_attention_layernorm,norm0)
    sync();budget.record(split,"held_projection_row_diagnostics","all_arms",time.perf_counter()-projection_start,
                         detail="diagnostic projection row losses; not used for full MLP gate")
    io_start=time.perf_counter()
    np.savez_compressed(RAW/f"{split}_held_mlp_outputs.npz",**{a:o for a,o in arm_outputs.items()},reference=ref.reshape(N_DOCS,SEQ,896).cpu().numpy())
    budget.record(split,"held_result_io","compressed_full_mlp_outputs",time.perf_counter()-io_start,
                  detail="compressed outputs for four arms plus FP16 reference")
    return article_rows,all_row_metrics,{"fold_error_mse":fold_error,"arm_boundaries":boundary,
        "reference_output_sha256":sha_tensor(ref),"folded_reference_sha256":sha_tensor(folded_ref),
        "held_group":group,"held_articles":len(held_rows)}


def main():
    protocol, manifest = protocol_hashes_ok()
    budget=Budget()
    overall_start=time.perf_counter()
    start_state=json.loads((RAW/"RUN_STATE.json").read_text(encoding="utf-8"))
    start_state.update({"runtime_status":"RUNNING_CORRECTNESS_FIRST","gpu_budget_cap_seconds":1800,
                        "stop_compute_by_seconds":1620,"held_access":"locked_per_split"})
    stable_json(RAW/"RUN_STATE.json",start_state,replace=True)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.cuda.init()
    torch.cuda.synchronize()
    model=load_prefix_model(budget)
    budget.check()
    env={"python":sys.version,"torch":torch.__version__,"transformers":__import__("transformers").__version__,
         "gptqmodel":__import__("importlib.metadata",fromlist=["version"]).version("gptqmodel"),
         "device":torch.cuda.get_device_name(0),"cuda_runtime":torch.version.cuda,
         "model_revision":json.loads((HERE/"manifests/FROZEN_PROTOCOL.json").read_text())["model"]["revision"],
         "model_dtype":"float16","tf32":False,"prefix_checkpoint_sha256":sha_file(PREFIX_DIR/"model.safetensors")}
    stable_json(HERE/"manifests/ENVIRONMENT.json",env)
    # Explicit short CUDA warmup is counted; it does not use held data or model outputs.
    def warmup():
        a=torch.randn((512,512),device=DEVICE,dtype=torch.float16)
        _=a@a.T
        sync()
        del a
    _,_=timed(budget,"","cuda_kernel_warmup","512x512_fp16_matmul",warmup)
    load_start=time.perf_counter();peak_reset();base=load_base_target();sync()
    budget.record("","target_pristine_weight_io","L12_gate_up_down_norm",time.perf_counter()-load_start,
                  detail="selective safetensors reads; pristine tensors from frozen base revision")
    budget.check()
    layer=model.model.layers[LAYER_INDEX]
    for name,key in (("gate_proj","gate"),("up_proj","up"),("down_proj","down")):
        if tuple(get_layer_module_weight(layer,name).shape)!=tuple(base[key].shape): raise CorrectnessFail(f"target geometry mismatch {name}")
        if layer.mlp.get_submodule(name).bias is not None:
            raise CorrectnessFail(f"v34 frozen no-bias L12 MLP assumption failed: {name}")
    all_results={}; all_article=[]; all_rows=[]; correctness=[]
    protocol_sha=sha_file(PROTOCOL_PATH)
    try:
        for split,groups in SPLITS.items():
            search_group=groups["search"]
            hidden_mm,pre_mm,x_mm=prepare_prefix_inputs(model,search_group,budget,split)
            validate_d_identity(layer,pre_mm,x_mm,base,budget,split)
            # D-selector uses only this split's search documents.
            d,d_info=awq_select(layer,x_mm,pre_mm,base,split,budget)
            # Source-compatible numerical relation checks, before any quality comparison.
            spots=[equivalence_spot(base,x_mm,d,k,split) for k in ("gate","up","down")]
            if any(not np.isfinite(v["fp16_fold_mse"]) for v in spots): raise CorrectnessFail("nonfinite FP16 folding drift")
            if any(v["fp64_max_abs"]>1e-9 or v["fp32_max_abs"]>1e-3 for v in spots):
                raise CorrectnessFail(f"real-coordinate equivalence failed split={split}: {spots}")
            stable_json(CORRECT/f"{split}_coordinate_equivalence.json",{"status":"PASS","tolerances":{"fp64_max_abs":1e-9,"fp32_max_abs":1e-3},"spots":spots})
            fold_check=selected_fold_correctness(layer,pre_mm,x_mm,base,d,split,budget)
            # Keep selected D and ratio bank immutable before candidate creation.
            d_path=FINAL/f"{split}_selected_D.pt"
            atomic_torch_save(d_path,{"D":d.detach().cpu(),"D_fp32":d.float().detach().cpu(),"selector":d_info})
            xprime_path=CACHE/f"{split}_search_Xprime.f16"
            xprime=make_xprime(layer,pre_mm,base["norm"],d,xprime_path)
            wcoord={"original":{"gate":base["gate"],"up":base["up"]},
                    "scaled":{"gate":weight_prime(base["gate"],d),"up":weight_prime(base["up"],d)}}
            input_coord={"original":x_mm,"scaled":xprime}
            hessians={}
            # Recompute H for each actual coordinate using the native process_batch path.
            for coord in ("original","scaled"):
                hfile=CACHE/f"{split}_H_gateup_{coord}.pt"
                if hfile.exists():
                    ho=torch.load(hfile,map_location=DEVICE,weights_only=True)
                    if ho["x_sha256"]!=sha_memmap(input_coord[coord]): raise CorrectnessFail("gate/up H cache input mismatch")
                    h=ho["H"].to(DEVICE); ns=int(ho["nsamples"])
                else:
                    start=time.perf_counter(); peak_reset()
                    h,ns=build_hessian(wcoord[coord]["gate"],input_coord[coord])
                    budget.record(split,"native_GPTQ_Hessian_build",f"gate_up_{coord}",time.perf_counter()-start,
                                  detail=f"shared sibling input; nsamples={ns}; shape={list(h.shape)}")
                    atomic_torch_save(hfile,{"H":h.cpu(),"nsamples":ns,"x_sha256":sha_memmap(input_coord[coord])})
                    budget.check()
                hessians[coord]=(h,ns)
            # H relationship is diagnostic: transformed H is rebuilt from actual rounded X', not reused.
            h0=hessians["original"][0]; h1=hessians["scaled"][0]
            h_expected=d.float().reciprocal()[:,None]*h0*d.float().reciprocal()[None,:]
            h_error=(h1-h_expected).abs()
            h_rel=float(torch.linalg.norm(h1-h_expected)/torch.linalg.norm(h_expected).clamp_min(1e-30))
            stable_json(CORRECT/f"{split}_H_coordinate_relation.json",{"status":"DIAGNOSTIC_REBUILT_FROM_XPRIME",
                         "original_H_sha256":sha_tensor(h0),"transformed_H_sha256":sha_tensor(h1),
                         "expected_Dinv_H_Dinv_max_abs":float(h_error.max().item()),"relative_frobenius_error":h_rel,
                         "transformed_H_was_recomputed":True,"damping_reapplied_after_build":True})
            # Full candidate banks: alpha=1 is native D/GPTQ, and all candidates start pristine.
            banks={}
            for coord in ("original","scaled"):
                h,ns=hessians[coord]
                for linear in ("gate","up"):
                    bank=run_candidate_bank(split,coord,f"{linear}_proj",base[linear],wcoord[coord][linear],
                        x_mm,input_coord[coord],h,ns,transformed=(coord=="scaled"),budget=budget)
                    banks[(coord,f"{linear}_proj")]=bank
                    parity=json.loads((CORRECT/f"{split}_{coord}_{linear}_proj_alpha1_native_parity.json").read_text(encoding="utf-8"))
                    correctness.append({"split":split,"check":"alpha1_native_gateup","target":f"{coord}_{linear}","fields":parity["fields"]})
            # Gate/up complete Linear commits, then actual MLP propagation gives arm-specific down inputs.
            temp_arms=make_arm_state(split,banks,{})
            down_states={}; down_hessians={}
            for arm in ("G00","G01","G10","G11"):
                d_arm=None if arm in ("G00","G01") else d
                gx=temp_arms[arm]["gate"]; ux=temp_arms[arm]["up"]
                # Single whole-MLP gate/up commit; no row-batch prefix refresh.
                set_weight(layer.post_attention_layernorm,base["norm"] if d_arm is None else (base["norm"].float()/d_arm.float()).half())
                set_weight(layer.mlp.gate_proj,gx["weight"]); set_weight(layer.mlp.up_proj,ux["weight"])
                ypath=CACHE/f"{split}_{arm}_down_input.f16"
                ymeta=RAW/f"{split}_{arm}_down_input.json"
                down_input_source=input_coord["original" if d_arm is None else "scaled"]
                down_input_common={"split":split,"arm":arm,"x_sha256":sha_memmap(down_input_source),
                                   "gate_sha256":sha_tensor(gx["weight"]),"up_sha256":sha_tensor(ux["weight"]),
                                   "D_sha256":None if d_arm is None else sha_tensor(d_arm)}
                if ypath.exists() and ypath.stat().st_size==(N_DOCS*SEQ*4864*2) and ymeta.exists():
                    ymeta_obj=json.loads(ymeta.read_text(encoding="utf-8"))
                    if any(ymeta_obj.get(k)!=v for k,v in down_input_common.items()):
                        raise CorrectnessFail(f"down activation cache provenance mismatch {split}/{arm}")
                    y=np.memmap(ypath,dtype=np.float16,mode="r+",shape=(N_DOCS,SEQ,4864))
                else:
                    refresh_start=time.perf_counter();peak_reset()
                    y=make_down_input(layer,down_input_source,gx,ux,base,d_arm,ypath)
                    stable_json(ymeta,{**down_input_common,"y_sha256":sha_memmap(y),"shape":[N_DOCS,SEQ,4864]})
                    sync();budget.record(split,"search_gate_up_to_down_activation_refresh",arm,time.perf_counter()-refresh_start,
                                          detail="committed full gate/up Linear; actual arm input for down H/GPTQ")
                    budget.check()
                # Compare actual postnorm+MLP gate/up first batch to stitched input produced by model modules.
                # The input has the exact fold and activation dtype used by the model.
                down_native_count=1 if arm in ("G00","G10") else 10
                ds,h,ns,parity=candidate_and_parity_down(split,arm,"original" if d_arm is None else "scaled",y,base,temp_arms[arm],budget,
                                                         search=(down_native_count==10))
                down_states[arm]=ds; down_hessians[arm]=h
                correctness.append({"split":split,"check":"alpha1_native_down","target":arm,"fields":parity})
                y.flush(); y._mmap.close()
            arms=make_arm_state(split,banks,down_states)
            # Down candidate isolation replays: original/scaled selected .8 after the full bank.
            for arm in ("G01","G11"):
                ds=down_states[arm]; h=down_hessians[arm]
                prep=prepare_native_state(h,nsamples=int(ds["H_nsamples"]))
                w_hash_before=sha_tensor(base["down"]);h_hash_before=sha_tensor(h)
                replay_start=time.perf_counter();peak_reset()
                replay=traced_dynamic_gptq(base["down"],prep,0.8,trace=False)
                sync();replay_seconds=time.perf_counter()-replay_start
                saved=next((r for r in ds["scores"] if float(r["alpha"])==0.8),None)
                if saved is None: raise CorrectnessFail(f"missing alpha0.8 down score replay {split}/{arm}")
                replay_score=score_rows_direct(
                    replay_mm:=np.memmap(CACHE/f"{split}_{arm}_down_input.f16",dtype=np.float16,mode="r",shape=(N_DOCS,SEQ,4864)),
                    replay_mm,base["down"],base["down"],replay["weight"],transformed=False)
                replay_mm._mmap.close()
                replay_cmp={"weight":sha_tensor(replay["weight"])==saved["q_sha256"],
                            "scale":sha_tensor(replay["scale"])==saved["scale_sha256"],
                            "zero":sha_tensor(replay["zero"])==saved["zero_sha256"],
                            "g_idx":sha_tensor(replay["g_idx"])==saved["g_idx_sha256"],
                            "score_exact":torch.equal(replay_score,saved["losses"].to(DEVICE)),
                            "pristine_W_unchanged":sha_tensor(base["down"])==w_hash_before,
                            "H_unchanged":sha_tensor(h)==h_hash_before}
                budget.record(split,"down_candidate_isolation_replay",arm,replay_seconds,
                              status="PASS" if all(replay_cmp.values()) else "FAIL",detail=json.dumps(replay_cmp,sort_keys=True))
                if not all(replay_cmp.values()):raise CorrectnessFail(f"down candidate order/state isolation failed {split}/{arm}: {replay_cmp}")
            # Row-wise W/scale/zero/g_idx stitch checks and actual commit occur before selection lock.
            for arm,state in arms.items():
                for proj in ("gate","up","down"):
                    st=state[proj]
                    if st["weight"].shape != base[proj].shape or st["scale"].shape[0] != base[proj].shape[0] or st["zero"].shape != st["scale"].shape:
                        raise CorrectnessFail(f"row stitch/qparam shape mismatch {split}/{arm}/{proj}")
                    expected=torch.arange(base[proj].shape[1],device=DEVICE,dtype=st["g_idx"].dtype)//GROUP_SIZE
                    if not torch.equal(st["g_idx"].to(DEVICE),expected): raise CorrectnessFail(f"stitched g_idx mismatch {split}/{arm}/{proj}")
            commit_check=verify_row_stitch_commit(layer,arms,base,d,split)
            stable_json(CORRECT/f"{split}_row_stitch_and_commit.json",commit_check)
            correctness.append({"split":split,"check":"row_stitch_and_actual_commit","status":"PASS",
                                "arms":commit_check["arms"]})
            lock=save_lock(split,d,d_info,arms,protocol_sha)
            correctness.append({"split":split,"check":"selection_lock_and_row_stitch","status":"PASS",
                                "lock_sha256":sha_file(FINAL/f"{split}_SELECTION_LOCK.json")})
            stable_json(CORRECT/f"{split}_PRE_HELD_CORRECTNESS.json",{"status":"PASS","checks":correctness,"held_opened":False})
            # Only now are held tokens read/materialized and evaluation results produced.
            held_group=groups["held"]
            held_rows=read_actual_group(held_group)
            hh,pp,xx=prepare_prefix_inputs(model,held_group,budget,split)
            article_rows,row_metrics,held_meta=eval_held(model,split,held_group,held_rows,base,arms,d,pp,hh,xx,budget)
            all_article.extend(article_rows); all_rows.extend(row_metrics)
            all_results[split]={"D_selection":d_info,"selection_lock_sha256":sha_file(FINAL/f"{split}_SELECTION_LOCK.json"),
                                "held":held_meta,"arm_full_mlp_mse":{},"article_metrics":article_rows}
            for a in arms:
                vals=[r["full_mlp_mse"] for r in article_rows if r["arm"]==a]
                # pooled over equal article token counts
                all_results[split]["arm_full_mlp_mse"][a]=float(np.mean(vals))
            stable_json(RAW/f"{split}_held_result.json",all_results[split])
            stable_json(CORRECT/f"{split}_POST_HELD_CORRECTNESS.json",{"status":"PASS","checks":correctness,
                "full_mlp_arm_boundary_hashes":held_meta["arm_boundaries"],"held_group":held_group})
            # Release split data; preserve on-disk caches/checkpoints for resume.
            for mm in (hidden_mm,pre_mm,x_mm,xprime):
                mm.flush(); mm._mmap.close()
            del hidden_mm,pre_mm,x_mm,xprime,arms,banks,hessians,down_hessians,down_states,temp_arms
            gc.collect(); torch.cuda.empty_cache()
        # Consolidated raw result tables. No article/token treated as independent model replicate.
        for name,rows,fields in (("PER_ARTICLE_HELD.csv",all_article,["split","group","arm","article_id","article_sha256","full_mlp_mse"]),
                                ("PROJECTION_AND_ROW_HELD.csv",all_rows,["split","arm","projection","row","mse"])):
            path=RAW/name
            if path.exists():
                with path.open("w",newline="",encoding="utf-8") as f:
                    wr=csv.DictWriter(f,fieldnames=fields);wr.writeheader();wr.writerows(rows)
            else:
                with path.open("x",newline="",encoding="utf-8") as f:
                    wr=csv.DictWriter(f,fieldnames=fields);wr.writeheader();wr.writerows(rows)
        status={"status":"COMPLETE_HELD_EVALUATION","correctness":"PASS","splits":all_results,
                "gpu_seconds_used":budget.used,"gpu_budget_cap_seconds":1800,"compute_stopped_before":1620,
                "wall_seconds_this_process":time.perf_counter()-overall_start,
                "ppl":"NOT_RUN","full_model_quantization":"NOT_RUN"}
        stable_json(RAW/"FINAL_RUN_REPORT.json",status)
    except (BudgetStop,CorrectnessFail) as e:
        label="BUDGET_STOP" if isinstance(e,BudgetStop) else "CORRECTNESS_FAIL"
        unlogged=max(0.0,budget.used-budget.last_record_used)
        budget.record("",label,str(e),unlogged,status="FAIL",detail="failure tail charged to cumulative runtime wall")
        stable_json(RAW/"FINAL_RUN_REPORT.json",{"status":label,"reason":str(e),"correctness_checks":correctness,
                    "gpu_seconds_used":budget.used,"gpu_budget_cap_seconds":1800,"held_results_written":list(all_results),
                    "ppl":"NOT_RUN","full_model_quantization":"NOT_RUN"},replace=True)
        raise
    except Exception as e:
        unlogged=max(0.0,budget.used-budget.last_record_used)
        budget.record("","RUNTIME_ERROR",repr(e),unlogged,status="FAIL",detail="failure tail charged to cumulative runtime wall")
        stable_json(RAW/"FINAL_RUN_REPORT.json",{"status":"RUNTIME_ERROR","reason":repr(e),
                    "correctness_checks":correctness,"gpu_seconds_used":budget.used,
                    "gpu_budget_cap_seconds":1800,"held_results_written":list(all_results),
                    "ppl":"NOT_RUN","full_model_quantization":"NOT_RUN"},replace=True)
        raise
    finally:
        if "model" in locals(): del model
        gc.collect(); torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
