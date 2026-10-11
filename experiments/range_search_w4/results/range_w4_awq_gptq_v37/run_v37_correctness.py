"""Bounded v37 correctness repair/re-audit. All writes stay in this v37 folder."""
from __future__ import annotations

import csv
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open
from transformers import AutoModelForCausalLM

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
V36 = ROOT / "experiments/range_search_w4/results/range_w4_awq_gptq_v36"
V18 = ROOT / "experiments/range_search_w4/results/range_w4_fullmodel_v18"
V171 = ROOT / "experiments/range_search_w4/results/range_w4_m2_parity_v17_1"
V173 = ROOT / "experiments/range_search_w4/results/range_w4_m2_runtime_v17_3"
BASE_DIR = ROOT / ".hf_cache/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775"
PREFIX_DIR = V18 / "M2/checkpoint/final"
DEVICE = torch.device("cuda")
LAYER = 12
SEQ = 2048
NDOC = 8
ROUTE_PRIOR = 180.335929
SUBBUDGET = 300.0
STOP_AT = 270.0

RAW = HERE / "raw"
CORRECT = HERE / "correctness"
STATE_PATH = RAW / "V37_STATE.json"
LEDGER_PATH = RAW / "BUDGET_LEDGER.csv"
PROTOCOL_PATH = HERE / "FROZEN_DIAGNOSTIC_PROTOCOL.json"

sys.path.insert(0, str(V173))
sys.path.insert(0, str(V171))
from native_m2 import native_dynamic_gptq  # noqa: E402


def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha_tensor(t: torch.Tensor) -> str:
    return sha_bytes(t.detach().cpu().contiguous().numpy().tobytes())


def sha_array(a: np.ndarray) -> str:
    return sha_bytes(np.ascontiguousarray(a).tobytes())


def sha_mm(mm: np.memmap) -> str:
    h = hashlib.sha256()
    for i in range(mm.shape[0]):
        h.update(np.asarray(mm[i]).tobytes(order="C"))
    return h.hexdigest()


def sync() -> None:
    torch.cuda.synchronize(DEVICE)


def set_weight(module, value: torch.Tensor) -> None:
    with torch.no_grad():
        module.weight.copy_(value.to(device=module.weight.device, dtype=module.weight.dtype))


def verify_frozen_sources() -> dict:
    p=HERE/"manifests/SOURCE_ARTIFACT_HASHES.json"
    frozen=json.loads(p.read_text(encoding="utf-8"))
    checks={}
    checks["protocol"]=sha_file(PROTOCOL_PATH)==frozen["frozen_protocol_sha256"]
    checks["runner"]=sha_file(Path(__file__))==frozen["correctness_script_sha256"]
    checks["freeze_script"]=sha_file(HERE/"freeze_v37_inputs.py")==frozen["freeze_script_sha256"]
    for path,expected in frozen["v36_sources"].items():
        checks[path]=sha_file(ROOT/path)==expected
    for obj in frozen["model_files"].values():
        checks[obj["path"]]=sha_file(Path(obj["path"]))==obj["sha256"]
    for path,expected in frozen["dependency_sources"].items():
        checks[path]=sha_file(ROOT/path)==expected
    result={"status":"PASS" if all(checks.values()) else "FAIL","checks":checks,
            "checked_count":len(checks),"manifest_sha256":sha_file(p)}
    (RAW/"INPUT_HASH_VERIFICATION.json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    if result["status"]!="PASS":raise RuntimeError("v37 frozen-input hash mismatch")
    return result


def open_mm(group: str, suffix: str, shape: tuple[int, ...]) -> np.memmap:
    return np.memmap(V36 / "cache" / f"{group}_{suffix}.f16", dtype=np.float16, mode="r", shape=shape)


def load_base() -> dict[str, torch.Tensor]:
    keys = {"norm": "model.layers.12.post_attention_layernorm.weight",
            "gate": "model.layers.12.mlp.gate_proj.weight",
            "up": "model.layers.12.mlp.up_proj.weight",
            "down": "model.layers.12.mlp.down_proj.weight"}
    with safe_open(str(BASE_DIR / "model.safetensors"), framework="pt", device="cpu") as f:
        return {k: f.get_tensor(v).to(device=DEVICE, dtype=torch.float16).contiguous() for k, v in keys.items()}


class Ledger:
    def __init__(self):
        self.start = time.perf_counter()
        self.stages: list[dict] = []
        if not LEDGER_PATH.exists():
            with LEDGER_PATH.open("w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(["stage", "task", "gpu_seconds", "cumulative_v37_gpu_seconds",
                                        "cumulative_route_gpu_seconds", "peak_allocated_bytes",
                                        "peak_reserved_bytes", "status", "detail"])
        with LEDGER_PATH.open(newline="",encoding="utf-8") as f:
            existing=list(csv.DictReader(f))
        self.previous=float(existing[-1]["cumulative_v37_gpu_seconds"]) if existing else 0.0

    @property
    def used(self) -> float:
        return self.previous + time.perf_counter() - self.start

    def record(self, name: str, task: str, seconds: float, status="PASS", detail="") -> None:
        alloc = int(torch.cuda.max_memory_allocated(DEVICE)) if torch.cuda.is_initialized() else 0
        reserved = int(torch.cuda.max_memory_reserved(DEVICE)) if torch.cuda.is_initialized() else 0
        row = [name, task, f"{seconds:.6f}", f"{self.used:.6f}", f"{ROUTE_PRIOR+self.used:.6f}",
               alloc, reserved, status, detail]
        with LEDGER_PATH.open("a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(row)
        self.stages.append({"stage": name, "task": task, "seconds": seconds, "status": status, "detail": detail,
                            "peak_allocated_bytes": alloc, "peak_reserved_bytes": reserved})
        state = {"protocol_sha256": sha_file(PROTOCOL_PATH), "status": "RUNNING",
                 "v36_gpu_seconds_already_charged": ROUTE_PRIOR, "v37_gpu_seconds_used": self.used,
                 "route_gpu_seconds_used": ROUTE_PRIOR+self.used, "v37_subbudget_seconds": SUBBUDGET,
                 "route_cap_seconds": 1800, "last_stage": name, "last_task": task}
        STATE_PATH.write_text(json.dumps(state, indent=2)+"\n", encoding="utf-8")
        print(f"[v37] {name}/{task}: {seconds:.3f}s; v37={self.used:.1f}/{SUBBUDGET:.0f}s", flush=True)

    def measure(self, name: str, task: str, fn):
        sync()
        torch.cuda.reset_peak_memory_stats(DEVICE)
        start = time.perf_counter()
        try:
            result = fn()
            sync()
            elapsed = time.perf_counter()-start
            self.record(name, task, elapsed)
            if self.used >= STOP_AT:
                raise RuntimeError(f"v37 stop reserve reached: {self.used:.3f}s >= {STOP_AT}s")
            return result
        except Exception as e:
            try:
                sync()
                elapsed = time.perf_counter()-start
                self.record(name, task, elapsed, "FAIL", repr(e))
            except Exception:
                pass
            raise


def read_group_rows(group: str) -> list[dict]:
    p = V36 / "cache" / f"{group}_tokens.json"
    rows = json.loads(p.read_text(encoding="utf-8"))
    gm = json.loads((V36 / "manifests/GROUP_TOKEN_CACHE_MANIFEST.json").read_text(encoding="utf-8"))
    if sha_file(p) != gm["groups"][group]["sha256"]:
        raise RuntimeError(f"frozen token group hash mismatch: {group}")
    for row in rows:
        token_hash = sha_bytes(json.dumps(row["token_ids"]).encode("utf-8"))
        if len(row["token_ids"]) != SEQ or token_hash != row["token_sha256"]:
            raise RuntimeError(f"article token hash mismatch: {group}/{row.get('source_article_index')}")
    return rows


def capture_fresh_p2(model, rows: list[dict]) -> dict:
    layer = model.model.layers[LAYER]
    norm = layer.post_attention_layernorm
    cfg = {"model_type": model.config.model_type, "hidden_size": model.config.hidden_size,
           "rms_norm_eps": model.config.rms_norm_eps, "norm_variance_epsilon": getattr(norm, "variance_epsilon", None),
           "norm_dtype": str(norm.weight.dtype), "norm_device": str(norm.weight.device),
           "prefix_norm_sha256": sha_tensor(norm.weight)}
    shape = (NDOC, SEQ, int(model.config.hidden_size))
    hidden = np.empty(shape, dtype=np.float16)
    pre = np.empty(shape, dtype=np.float16)
    post = np.empty(shape, dtype=np.float16)
    active: dict[str, torch.Tensor | None] = {"pre_alias": None, "pre_snap": None, "post_alias": None, "post_snap": None}
    alias_checks = []

    def pre_hook(_m, args):
        active["pre_alias"] = args[0].detach()
        active["pre_snap"] = args[0].detach().clone()

    def post_hook(_m, _args, out):
        active["post_alias"] = out.detach()
        active["post_snap"] = out.detach().clone()

    hpre = norm.register_forward_pre_hook(pre_hook)
    hpost = norm.register_forward_hook(post_hook)
    pos = torch.arange(SEQ, device=DEVICE)[None]
    try:
        for i, row in enumerate(rows):
            ids = torch.tensor(row["token_ids"], dtype=torch.long, device=DEVICE)[None]
            with torch.inference_mode():
                h = model.model.embed_tokens(ids)
                for li in range(LAYER+1):
                    block = model.model.layers[li]
                    if li == LAYER:
                        hidden[i] = h[0].detach().cpu().numpy()
                    pos_emb = model.model.rotary_emb(h, pos)
                    h = block(h, attention_mask=None, position_ids=pos,
                              cache_position=pos[0], position_embeddings=pos_emb,
                              use_cache=False)[0]
                    if li == LAYER:
                        if active["pre_snap"] is None or active["post_snap"] is None:
                            raise RuntimeError("fresh P2 norm hook did not fire")
                        alias_checks.append({
                            "doc": i,
                            "pre_alias_unchanged_after_block": torch.equal(active["pre_alias"], active["pre_snap"]),
                            "post_alias_unchanged_after_block": torch.equal(active["post_alias"], active["post_snap"]),
                            "pre_storage_ptr": int(active["pre_alias"].untyped_storage().data_ptr()),
                            "pre_clone_storage_ptr": int(active["pre_snap"].untyped_storage().data_ptr()),
                            "post_storage_ptr": int(active["post_alias"].untyped_storage().data_ptr()),
                            "post_clone_storage_ptr": int(active["post_snap"].untyped_storage().data_ptr())})
                        pre[i] = active["pre_snap"][0].cpu().numpy()
                        post[i] = active["post_snap"][0].cpu().numpy()
                        active.update(pre_alias=None, pre_snap=None, post_alias=None, post_snap=None)
            del ids, h
    finally:
        hpre.remove(); hpost.remove()

    # Re-evaluate the norm over both saved and fresh pre-norm tensors, without changing parameters.
    replay_old = []
    replay_fresh = []
    old_pre = open_mm("S1_P2", "prenorm", shape)
    old_post = open_mm("S1_P2", "postnorm", shape)
    for i in range(NDOC):
        p_old = torch.from_numpy(np.array(old_pre[i], copy=True)).unsqueeze(0).to(DEVICE)
        y_old = norm(p_old)
        p_new = torch.from_numpy(pre[i].copy()).unsqueeze(0).to(DEVICE)
        y_new = norm(p_new)
        replay_old.append(y_old[0].detach().half().cpu().numpy())
        replay_fresh.append(y_new[0].detach().half().cpu().numpy())
        del p_old, y_old, p_new, y_new
    old_post_np = np.asarray(old_post)
    replay_old_np = np.stack(replay_old)
    replay_fresh_np = np.stack(replay_fresh)
    details = {
        "fresh_capture": {"hidden_sha256": sha_array(hidden), "prenorm_sha256": sha_array(pre),
                          "postnorm_sha256": sha_array(post), "shape": list(shape), "dtype": "float16"},
        "v36_capture": json.loads((V36/"raw/S1_P2_capture.json").read_text(encoding="utf-8")),
        "fresh_norm_config": cfg,
        "hook_alias_checks": alias_checks,
        "clean_capture_self_replay_exact": bool(np.array_equal(replay_fresh_np, post)),
        "old_capture_replay_exact": bool(np.array_equal(replay_old_np, old_post_np)),
        "old_post_vs_norm_replay": summarize_array_error(old_post_np, replay_old_np),
        "fresh_post_vs_norm_replay": summarize_array_error(post, replay_fresh_np),
        "v36_vs_fresh": {name: compare_arrays(np.asarray(a), np.asarray(b)) for name,a,b in
                          (("hidden", open_mm("S1_P2","hidden_l12",shape), hidden),
                           ("prenorm", old_pre, pre), ("postnorm", old_post, post))},
        "per_doc": []}
    for i in range(NDOC):
        details["per_doc"].append({"doc": i,
            "cached_post_replay": summarize_array_error(old_post_np[i], replay_old_np[i]),
            "fresh_post_replay": summarize_array_error(post[i], replay_fresh_np[i]),
            "cached_vs_fresh_pre": summarize_array_error(np.asarray(old_pre[i]), pre[i]),
            "cached_vs_fresh_post": summarize_array_error(old_post_np[i], post[i])})
    # Keep all independent recaptures in v37; never replace v36 caches.
    for name, arr in (("hidden_l12", hidden), ("prenorm", pre), ("postnorm", post)):
        arr.tofile(RAW/f"S1_P2_clean_{name}.f16")
    return details


def summarize_array_error(a: np.ndarray, b: np.ndarray) -> dict:
    diff = np.asarray(a).astype(np.float32)-np.asarray(b).astype(np.float32)
    return {"exact": bool(np.array_equal(a,b)), "max_abs": float(np.max(np.abs(diff))),
            "mean_abs": float(np.mean(np.abs(diff))), "equal_elements": int(np.count_nonzero(np.asarray(a)==np.asarray(b))),
            "elements": int(np.asarray(a).size)}


def compare_arrays(a: np.ndarray, b: np.ndarray) -> dict:
    return {"same_sha256": sha_array(a)==sha_array(b), **summarize_array_error(a,b)}


def parity_and_missing_score(base: dict[str, torch.Tensor], ledger: Ledger) -> dict:
    down = torch.load(V36/"raw/downbank_S0_G00.pt", map_location=DEVICE, weights_only=True)
    state = down["state"]
    candidate = state["alpha1"]
    w = base["down"]
    h = down["H"].to(DEVICE)
    ns = int(down["nsamples"])
    x = np.memmap(V36/"cache/S0_G00_down_input.f16", dtype=np.float16, mode="r", shape=(NDOC,SEQ,4864))
    input_hash = sha_mm(x)
    checks = {"saved_input_hash_matches_downbank": input_hash == down["x_sha256"],
              "saved_pristine_weight_hash_matches_downbank": sha_tensor(w) == down["weight_sha256"],
              "saved_H_hash_matches_state": sha_tensor(h) == state["H_sha256"],
              "saved_nsamples_matches_state": ns == int(state["H_nsamples"]),
              "saved_alpha1_score_equals_record": False}
    if any(not checks[k] for k in list(checks)[:4]):
        raise RuntimeError(f"G00 down parity input provenance failed: {checks}")
    rec = next(r for r in state["scores"] if float(r["alpha"]) == 1.0)
    saved_loss = rec["losses"].to(DEVICE)
    checks["saved_alpha1_score_equals_record"] = torch.equal(saved_loss, candidate["losses"].to(DEVICE))
    if not checks["saved_alpha1_score_equals_record"]:
        raise RuntimeError("G00 down saved candidate alpha1 score differs from row-score record")

    # The frozen v36 native GPTQ comparison already established exact alpha=1
    # equality for W/qparams/group entries. Only its direct output-score field
    # was absent. Re-score the saved candidate and use the recorded exact-W
    # parity to infer the native score; do not spend another GPTQ execution.
    fields={k:bool(down["parity"].get(k,False)) for k in
            ("weight","scale","zero","g_idx","group_entry_count","group_entries")}

    def direct_score(q: torch.Tensor) -> torch.Tensor:
        total=torch.zeros((w.shape[0],),device=DEVICE,dtype=torch.float64)
        for i in range(NDOC):
            xi=torch.from_numpy(np.array(x[i],copy=True)).to(DEVICE).float().reshape(-1,w.shape[1])
            err=xi@(q.float()-w.float()).T
            total += err.double().square().sum(dim=0)
            del xi,err
        return total/float(NDOC*SEQ)

    candidate_loss = ledger.measure("direct_alpha1_candidate_score", "S0_G00_down_all_896_rows", lambda:direct_score(candidate["weight"]))
    fields["saved_search_score_matches_candidate_direct_score"] = torch.equal(candidate_loss,saved_loss)
    fields["direct_row_output_scores"] = fields["weight"] and fields["saved_search_score_matches_candidate_direct_score"]
    delta=(candidate_loss-saved_loss).abs()
    checks.update({"candidate_vs_native_weight_exact":fields["weight"],
                   "direct_score_candidate_vs_saved_exact":fields["saved_search_score_matches_candidate_direct_score"],
                   "direct_score_native_equivalent_by_exact_weight_parity":fields["direct_row_output_scores"],
                   "max_abs_score_difference":float(delta.max().item()),
                   "mismatched_rows":int((candidate_loss!=saved_loss).sum().item()),
                   "rows":int(w.shape[0]), "input_tokens":NDOC*SEQ,
                   "old_parity_record_missing_direct_score": "direct_row_output_scores" not in down["parity"],
                   "native_score_inference_basis":"v36 saved alpha1 native parity has exact quantized W; direct score is deterministic in saved X, pristine W, and W_hat; v37 recomputed the same direct score on that saved W_hat."})
    x._mmap.close()
    return {"status":"PASS" if all(v for k,v in fields.items() if isinstance(v,bool)) and all(checks[k] for k in
             ("saved_input_hash_matches_downbank","saved_pristine_weight_hash_matches_downbank","saved_H_hash_matches_state",
              "saved_nsamples_matches_state","saved_alpha1_score_equals_record","direct_score_candidate_vs_saved_exact",
              "direct_score_native_equivalent_by_exact_weight_parity")) else "FAIL",
            "provenance":checks,"parity_fields":fields,
            "native_weight_sha256_from_exact_v36_parity":sha_tensor(candidate["weight"]),
            "candidate_weight_sha256":sha_tensor(candidate["weight"]),
            "native_scale_sha256_from_exact_v36_parity":sha_tensor(candidate["scale"]),"candidate_scale_sha256":sha_tensor(candidate["scale"]),
            "native_zero_sha256_from_exact_v36_parity":sha_tensor(candidate["zero"]),"candidate_zero_sha256":sha_tensor(candidate["zero"]),
            "native_g_idx_sha256_from_exact_v36_parity":sha_tensor(candidate["g_idx"]),"candidate_g_idx_sha256":sha_tensor(candidate["g_idx"]),
            "native_group_entry_count_from_exact_v36_parity":len(candidate["group_entries"]),
            "score_max_abs_diff":float(delta.max().item()),"score_mismatch_rows":int((candidate_loss!=saved_loss).sum().item()),
            "additional_native_gptq_execution_v37":False}


def recompute_s0(base: dict[str, torch.Tensor], model, ledger: Ledger) -> dict:
    layer=model.model.layers[LAYER]
    norm=layer.post_attention_layernorm
    mlp=layer.mlp
    old_norm=norm.weight.detach().clone()
    old={n:mlp.get_submodule(n).weight.detach().clone() for n in ("gate_proj","up_proj","down_proj")}
    selected=torch.load(V36/"final/S0_selected_tensors.pt",map_location="cpu",weights_only=True)
    d=selected["D"].to(DEVICE)
    lock=json.loads((V36/"final/S0_SELECTION_LOCK.json").read_text(encoding="utf-8"))
    selected_lock=lock["selected_tensors"]
    selection_checks={"D_matches_selection_lock":sha_tensor(d)==lock["D_selection"]["selected_D_sha256"]}
    for arm in ("G00","G01","G10","G11"):
        for proj in ("gate","up","down"):
            st=selected[arm][proj]
            key=f"{arm}.{proj}"
            for field in ("weight","scale","zero","g_idx"):
                s=sha_tensor(st[field])
                selection_checks[f"{key}.{field}_matches_lock"] = s == selected_lock[key][f"{field}_sha256"]
    if not all(selection_checks.values()):
        raise RuntimeError("selected S0 checkpoint does not match frozen selection lock")

    def row_error(a,b):
        diff=(a.float()-b.float()).double().square()
        return diff.mean(dim=(0,1)).detach().cpu().numpy()

    fold_stats={}
    try:
        for group,label in (("P0","search"),("P1","held")):
            xmm=open_mm(f"S0_{group}","postnorm",(NDOC,SEQ,896))
            pmm=open_mm(f"S0_{group}","prenorm",(NDOC,SEQ,896))
            x_orig=torch.from_numpy(np.array(xmm,copy=True)).to(DEVICE).reshape(-1,896)
            p_pre=torch.from_numpy(np.array(pmm,copy=True)).to(DEVICE).reshape(-1,896)
            set_weight(norm,base["norm"])
            set_weight(mlp.gate_proj,base["gate"]);set_weight(mlp.up_proj,base["up"]);set_weight(mlp.down_proj,base["down"])
            with torch.inference_mode():
                ref=mlp(x_orig).detach()
            set_weight(norm,(base["norm"].float()/d.float()).half())
            set_weight(mlp.gate_proj,(base["gate"].float()*d.float()[None,:]).half())
            set_weight(mlp.up_proj,(base["up"].float()*d.float()[None,:]).half())
            with torch.inference_mode():
                xp=norm(p_pre)
                folded=mlp(xp).detach()
            diff=folded.float()-ref.float()
            fold_stats[label]={"source_group":group,"D_sha256":sha_tensor(d),"reference_sha256":sha_tensor(ref),
                "folded_reference_sha256":sha_tensor(folded),"fold_error_mse":float(diff.double().square().mean().item()),
                "fold_error_max_abs":float(diff.abs().max().item()),"finite":bool(torch.isfinite(diff).all()),
                "correct_fold":"norm weight / D; pristine gate/up weights multiplied by D and cast to FP16; down unchanged",
                "token_positions":NDOC*SEQ}
            del x_orig,p_pre,ref,folded,diff,xp
            xmm._mmap.close();pmm._mmap.close()

        # Independently recreate held reference, every selected gate/up -> down input,
        # and each saved full-MLP output from the locked tensors and frozen P1 inputs.
        held_group="P1"
        xorig_mm=open_mm("S0_P1","postnorm",(NDOC,SEQ,896))
        pre_mm=open_mm("S0_P1","prenorm",(NDOC,SEQ,896))
        x_orig=torch.from_numpy(np.array(xorig_mm,copy=True)).to(DEVICE)
        pre=torch.from_numpy(np.array(pre_mm,copy=True)).to(DEVICE)
        x_flat=x_orig.reshape(-1,896); pre_flat=pre.reshape(-1,896)
        saved_npz=np.load(V36/"raw/S0_held_mlp_outputs.npz")
        ref_saved=torch.from_numpy(saved_npz["reference"].copy()).to(DEVICE)
        set_weight(norm,base["norm"])
        for n,k in (("gate_proj","gate"),("up_proj","up"),("down_proj","down")):set_weight(mlp.get_submodule(n),base[k])
        with torch.inference_mode():ref=mlp(x_flat).detach()
        reference_exact=torch.equal(ref.reshape(NDOC,SEQ,896),ref_saved)
        reference_err=summarize_tensor_err(ref.reshape(NDOC,SEQ,896),ref_saved)
        if not reference_exact:
            # Keep exact comparison as frozen; a mismatch means the saved reference is not accepted.
            pass

        # First reproduce the four search-side down inputs that generated the saved
        # down H/GPTQ banks. The v36 down-input cache is P0 search, never P1 held.
        search_xmm=open_mm("S0_P0","postnorm",(NDOC,SEQ,896))
        search_pmm=open_mm("S0_P0","prenorm",(NDOC,SEQ,896))
        search_x=torch.from_numpy(np.array(search_xmm,copy=True)).to(DEVICE)
        search_pre=torch.from_numpy(np.array(search_pmm,copy=True)).to(DEVICE)
        down_input_hashes={}
        for arm in ("G00","G01","G10","G11"):
            st=selected[arm]; transformed=arm in ("G10","G11")
            if transformed:
                set_weight(norm,(base["norm"].float()/d.float()).half())
                search_coord=None
            else:
                set_weight(norm,base["norm"]); search_coord=search_x
            set_weight(mlp.gate_proj,st["gate"]["weight"]);set_weight(mlp.up_proj,st["up"]["weight"])
            with torch.inference_mode():
                search_parts=[]
                for doc in range(NDOC):
                    if transformed:
                        search_doc=norm(search_pre[doc:doc+1])
                    else:
                        search_doc=search_coord[doc:doc+1]
                    sg=mlp.gate_proj(search_doc);su=mlp.up_proj(search_doc)
                    search_parts.append((mlp.act_fn(sg)*su).detach())
                    del sg,su,search_doc
                search_down=torch.cat(search_parts,dim=0)
            cached_search=np.memmap(V36/"cache"/f"S0_{arm}_down_input.f16",dtype=np.float16,mode="r",shape=(NDOC,SEQ,4864))
            down_saved=torch.from_numpy(np.array(cached_search,copy=True)).to(DEVICE)
            down_input_hashes[arm]={"search_group":"P0","exact":torch.equal(search_down,down_saved),
                "computed_sha256":sha_tensor(search_down),"saved_sha256":sha_tensor(down_saved),
                "max_abs":summarize_tensor_err(search_down,down_saved)["max_abs"]}
            cached_search._mmap.close()
            del search_down,down_saved,search_coord
        search_xmm._mmap.close();search_pmm._mmap.close()

        held_arrays={}; projection_rows={}; arm_losses={}
        for arm in ("G00","G01","G10","G11"):
            st=selected[arm]
            transformed=arm in ("G10","G11")
            if transformed:
                set_weight(norm,(base["norm"].float()/d.float()).half())
                coord_x=norm(pre_flat)
            else:
                set_weight(norm,base["norm"])
                coord_x=x_flat
            set_weight(mlp.gate_proj,st["gate"]["weight"])
            set_weight(mlp.up_proj,st["up"]["weight"])
            set_weight(mlp.down_proj,st["down"]["weight"])
            with torch.inference_mode():
                qg=mlp.gate_proj(coord_x);qu=mlp.up_proj(coord_x)
                down_input=mlp.act_fn(qg)*qu
                qdown=mlp.down_proj(down_input)
                refg=F.linear(x_flat,base["gate"]);refu=F.linear(x_flat,base["up"])
                refdown=F.linear(down_input,base["down"])
                out_docs=[]
                for i in range(NDOC):
                    if transformed:
                        block_x=norm(pre[i:i+1])
                    else:
                        block_x=x_orig[i:i+1]
                    out_docs.append(mlp(block_x).detach())
                    del block_x
                out=torch.cat(out_docs,dim=0)
            saved_arm=torch.from_numpy(saved_npz[arm].copy()).to(DEVICE)
            held_arrays[arm]=out.detach().clone()
            err=(out.float()-ref.reshape(NDOC,SEQ,896).float()).double().square()
            arm_losses[arm]=float(err.mean().item())
            projection_rows[arm]={
                "gate":((qg.float()-refg.float()).double().square().mean(dim=0)).detach().cpu().tolist(),
                "up":((qu.float()-refu.float()).double().square().mean(dim=0)).detach().cpu().tolist(),
                "down":((qdown.float()-refdown.float()).double().square().mean(dim=0)).detach().cpu().tolist(),
                "full_mlp":row_error(out,ref.reshape(NDOC,SEQ,896)).tolist()}
            projection_rows[arm]["saved_full_mlp_output"]={"exact":torch.equal(out,saved_arm),**summarize_tensor_err(out,saved_arm)}
            del qg,qu,down_input,qdown,out,refg,refu,refdown,saved_arm,coord_x

        # Verify selection scores and candidate-state provenance from the already-saved banks.
        bank_checks={}
        for coord,projs in (("original",("gate_proj","up_proj")),("scaled",("gate_proj","up_proj"))):
            xorigmm=open_mm("S0_P0","postnorm",(NDOC,SEQ,896))
            xcoordmm=(xorigmm if coord=="original" else np.memmap(V36/"cache/S0_search_Xprime.f16",
                       dtype=np.float16,mode="r",shape=(NDOC,SEQ,896)))
            hname="S0_H_gateup_original.pt" if coord=="original" else "S0_H_gateup_scaled.pt"
            hobj=torch.load(V36/"cache"/hname,map_location="cpu",weights_only=True)
            coord_d=d.detach().cpu()
            for name in projs:
                bank=torch.load(V36/f"raw/bank_S0_{coord}_{name}.pt",map_location="cpu",weights_only=True)
                basekey="gate" if name=="gate_proj" else "up"
                expected_coord=base[basekey].detach().cpu() if coord=="original" else (base[basekey].detach().cpu().float()*coord_d.float()[None,:]).half()
                bank_checks[f"{coord}.{name}.input_orig_hash_matches_P0"]=bank["common"]["x_orig_sha256"]==sha_mm(xorigmm)
                bank_checks[f"{coord}.{name}.input_coord_hash_matches_P0"]=bank["common"]["x_coord_sha256"]==sha_mm(xcoordmm)
                bank_checks[f"{coord}.{name}.pristine_weight_hash_matches"]=bank["common"]["weight_orig_sha256"]==sha_tensor(base[basekey])
                bank_checks[f"{coord}.{name}.coordinate_weight_hash_matches"]=bank["common"]["weight_coord_sha256"]==sha_tensor(expected_coord)
                bank_checks[f"{coord}.{name}.H_hash_matches_cache"]=bank["common"]["H_sha256"]==sha_tensor(hobj["H"])
                bank_checks[f"{coord}.{name}.H_nsamples_matches_cache"]=int(bank["common"]["nsamples"])==int(hobj["nsamples"])
            if coord!="original":xcoordmm._mmap.close()
            xorigmm._mmap.close()
            for name in projs:
                bank=torch.load(V36/f"raw/bank_S0_{coord}_{name}.pt",map_location="cpu",weights_only=True)
                key=("G00" if coord=="original" else "G10")+(".gate" if name=="gate_proj" else ".up")
                searched=key.replace("G00","G01").replace("G10","G11")
                for arm_key,native in ((key,True),(searched,False)):
                    actual=selected[arm_key.split(".")[0]]["gate" if name=="gate_proj" else "up"]
                    if native:
                        candidate=bank["alpha1"]
                        alpha_expected=torch.ones_like(actual["selected_alpha"]).cpu()
                        score_expected=bank["alpha1"]["losses"]
                    else:
                        candidate=bank
                        alpha_expected=bank["best_alpha"]
                        score_expected=bank["best_loss"]
                    bank_checks[f"{arm_key}.weight_is_saved_bank_result"]=sha_tensor(actual["weight"]) == sha_tensor(candidate["weight"] if "weight" in candidate else bank["best_weight"])
                    if not native:
                        bank_checks[f"{arm_key}.alpha_equals_recorded_row_argmin"]=torch.equal(actual["selected_alpha"].cpu(),alpha_expected.cpu())
                        score_matrix=torch.stack([r["losses"].double() for r in bank["scores"]],dim=0)
                        alphas=torch.tensor([float(r["alpha"]) for r in bank["scores"]],dtype=torch.float64)
                        order=torch.argsort(alphas,descending=True,stable=True)
                        ordered=score_matrix[order]
                        arg=ordered.argmin(dim=0)
                        replay_alpha=alphas[order][arg]
                        bank_checks[f"{arm_key}.alpha_recomputed_from_saved_scores"]=torch.equal(replay_alpha,actual["selected_alpha"].cpu().double())
                        bank_checks[f"{arm_key}.best_loss_recomputed_from_saved_scores"]=torch.equal(ordered.min(dim=0).values,bank["best_loss"].cpu().double())
        for arm in ("G00","G01","G10","G11"):
            downbank=torch.load(V36/f"raw/downbank_S0_{arm}.pt",map_location="cpu",weights_only=True)
            st=downbank["state"]
            actual=selected[arm]["down"]
            native=arm in ("G00","G10")
            source=st["alpha1"] if native else st
            for field in ("weight","scale","zero","g_idx"):
                bank_checks[f"{arm}.down.{field}_matches_saved_bank"]=sha_tensor(actual[field])==sha_tensor(source[field])
            if not native:
                bank_checks[f"{arm}.down.alpha_matches_saved_row_argmin"]=torch.equal(actual["selected_alpha"].cpu(),st["selected_alpha"].cpu())
                score_matrix=torch.stack([r["losses"].double() for r in st["scores"]],dim=0)
                alphas=torch.tensor([float(r["alpha"]) for r in st["scores"]],dtype=torch.float64)
                order=torch.argsort(alphas,descending=True,stable=True)
                ordered=score_matrix[order]; arg=ordered.argmin(dim=0)
                replay_alpha=alphas[order][arg]
                bank_checks[f"{arm}.down.alpha_recomputed_from_saved_scores"]=torch.equal(replay_alpha,actual["selected_alpha"].cpu().double())
                bank_checks[f"{arm}.down.best_loss_recomputed_from_saved_scores"]=torch.equal(ordered.min(dim=0).values,st["losses"].cpu().double())
            down_mm=np.memmap(V36/"cache"/f"S0_{arm}_down_input.f16",dtype=np.float16,mode="r",shape=(NDOC,SEQ,4864))
            bank_checks[f"{arm}.down.input_hash_matches_saved_downbank"]=sha_mm(down_mm)==downbank["x_sha256"]
            down_mm._mmap.close()

        # Original P1 output archive is the primary source for paired full-MLP metrics.
        saved_metrics={arm:float(((torch.from_numpy(saved_npz[arm].copy()).to(DEVICE).float()-ref_saved.float()).double().square()).mean().item())
                       for arm in ("G00","G01","G10","G11")}
        max_saved_metric_diff=max(abs(saved_metrics[a]-arm_losses[a]) for a in arm_losses)
        s0_summary={"arm_full_mlp_mse_recomputed_from_saved_outputs":saved_metrics,
                    "arm_full_mlp_mse_independent_gpu_recompute":arm_losses,
                    "independent_reference_exact":reference_exact,"reference_error":reference_err,
                    "max_metric_difference_vs_saved":max_saved_metric_diff,
                    "contrasts_fractional_loss_change":{
                        "G10_vs_G00":saved_metrics["G10"]/saved_metrics["G00"]-1.0,
                        "G11_vs_G10":saved_metrics["G11"]/saved_metrics["G10"]-1.0,
                        "G11_vs_G01":saved_metrics["G11"]/saved_metrics["G01"]-1.0},
                    "row_tail":{},"projection_quality":{},"selection_checks":selection_checks,
                    "candidate_bank_checks":bank_checks,"down_input_checks":down_input_hashes,
                    "held_output_checks":{a:projection_rows[a]["saved_full_mlp_output"] for a in projection_rows}}
        # Paired row loss: positive is degradation relative to G01.
        for proj in ("gate","up","down","full_mlp"):
            b=np.asarray(projection_rows["G01"][proj],dtype=np.float64)
            for arm in ("G00","G10","G11"):
                a=np.asarray(projection_rows[arm][proj],dtype=np.float64)
                rel=np.divide(a-b,np.maximum(b,1e-30))
                s0_summary["row_tail"][f"{arm}_vs_G01_{proj}"]={"wins":int((a<b).sum()),"ties":int((a==b).sum()),
                    "losses":int((a>b).sum()),"median_relative_improvement":float(np.median(-rel)),
                    "worst_relative_degradation":float(np.max(rel)),"rows_over_10pct_degradation":int((rel>0.10).sum()),
                    "rows_over_50pct_degradation":int((rel>0.50).sum()),"row_count":int(a.size)}
                if proj in ("gate","up","down"):
                    pooled_a=float(np.mean(a)); pooled_b=float(np.mean(b))
                    s0_summary["projection_quality"][f"{arm}_vs_G01_{proj}"]={"pooled_mse_change":pooled_a/pooled_b-1.0,
                        "no_more_than_5pct_regression":pooled_a <= pooled_b*1.05}
        fold_held=fold_stats["held"]["fold_error_mse"]
        g11=saved_metrics["G11"]
        gate_improvement=1.0-saved_metrics["G11"]/saved_metrics["G01"]
        s0_summary["fold_error_held_corrected"] = fold_stats["held"]
        s0_summary["fold_error_search_corrected"] = fold_stats["search"]
        s0_summary["fold_signal_floor"]={"G11_mse":g11,"five_x_fold_error":5.0*fold_held,
            "passes":bool(g11>=5.0*fold_held),"ratio":g11/max(fold_held,1e-30)}
        s0_summary["primary_gate"]={"frozen_minimum_improvement":0.01,"observed_G11_vs_G01_improvement":gate_improvement,
            "passes":bool(gate_improvement>=0.01)}
        s0_summary["all_projection_gates_pass"] = all(v["no_more_than_5pct_regression"] for v in s0_summary["projection_quality"].values())
        projection_tail={k:v for k,v in s0_summary["row_tail"].items() if k.endswith(("_gate","_up","_down"))}
        s0_summary["all_row_tail_gates_pass"] = all(v["rows_over_10pct_degradation"] <= 0.05*v["row_count"] and
                                                       v["rows_over_50pct_degradation"] == 0 for v in projection_tail.values())
        s0_summary["protocol_projection_row_tail_scope"] = list(projection_tail)
        s0_summary["full_mlp_row_tails_descriptive_only"] = {k:v for k,v in s0_summary["row_tail"].items() if k.endswith("_full_mlp")}
        s0_summary["candidate_selection_and_held_path_valid"] = bool(
            all(selection_checks.values()) and all(bank_checks.values()) and
            all(v["exact"] for v in down_input_hashes.values()) and reference_exact and
            all(v["exact"] for v in s0_summary["held_output_checks"].values()) and
            max_saved_metric_diff <= 1e-12 and all(v["finite"] for v in fold_stats.values()))
        s0_summary["quality_gates"]={
            "primary_G11_vs_G01_ge_1pct":s0_summary["primary_gate"]["passes"],
            "corrected_fold_signal_floor":s0_summary["fold_signal_floor"]["passes"],
            "per_projection_no_regression_over_5pct":s0_summary["all_projection_gates_pass"],
            "projection_row_tail_at_most_5pct_over_10pct_and_none_over_50pct":s0_summary["all_row_tail_gates_pass"]}
        s0_summary["quality_gate_passes"] = all(s0_summary["quality_gates"].values())
        xorig_mm._mmap.close();pre_mm._mmap.close()
        return {"s0":s0_summary,"fold":fold_stats}
    finally:
        for n,v in old.items():set_weight(mlp.get_submodule(n),v)
        set_weight(norm,old_norm)


def summarize_tensor_err(a: torch.Tensor, b: torch.Tensor) -> dict:
    diff=(a.float()-b.float()).abs()
    return {"exact":bool(torch.equal(a,b)),"max_abs":float(diff.max().item()),"mean_abs":float(diff.mean().item()),
            "equal_elements":int((a==b).sum().item()),"elements":int(a.numel())}


def main() -> None:
    if STATE_PATH.exists():
        prior=json.loads(STATE_PATH.read_text(encoding="utf-8"))
        with LEDGER_PATH.open(newline="",encoding="utf-8") as f:
            rows=list(csv.DictReader(f))
        can_retry=(prior.get("status")=="RUNNING" and rows and rows[-1]["status"]=="FAIL"
                   and not (CORRECT/"v37_correctness_results.json").exists())
        if not can_retry:
            raise RuntimeError("v37 execution is complete or has no retryable failed stage; preserve it and do not rerun")
    input_hash_audit=verify_frozen_sources()
    if ROUTE_PRIOR + SUBBUDGET > 1800:
        raise RuntimeError("v37 subbudget violates original v36 cumulative route hard cap")
    ledger=Ledger()
    started=time.perf_counter()
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    init_start=time.perf_counter()
    torch.cuda.init();sync()
    ledger.record("cuda_init", "fresh_diagnostic_process", time.perf_counter()-init_start,
                  detail="CUDA context and initial synchronize; charged against v37 subbudget")
    def warmup():
        z=torch.ones((512,512),device=DEVICE,dtype=torch.float16)
        _=z@z.T
        sync()
        del z
    ledger.measure("cuda_warmup", "512x512_fp16_matmul", warmup)
    device_info={"name":torch.cuda.get_device_name(0),"torch":torch.__version__,"cuda":torch.version.cuda,
                 "tf32":torch.backends.cuda.matmul.allow_tf32}
    model=ledger.measure("model_load", "fresh_v18_M2_prefix_process", lambda:AutoModelForCausalLM.from_pretrained(
        str(PREFIX_DIR),torch_dtype=torch.float16,low_cpu_mem_usage=True,local_files_only=True,
        attn_implementation="sdpa").eval().to(DEVICE))
    model.eval()
    base=ledger.measure("pristine_target_weight_load", "L12_norm_gate_up_down", load_base)
    prefix_norm=model.model.layers[LAYER].post_attention_layernorm
    target_norm_hash=sha_tensor(base["norm"])
    norm_compare={"base_norm_sha256":target_norm_hash,"prefix_norm_sha256":sha_tensor(prefix_norm.weight),
                 "base_equals_prefix":torch.equal(base["norm"],prefix_norm.weight),
                 "rms_norm_eps_config":model.config.rms_norm_eps,
                 "variance_epsilon_module":getattr(prefix_norm,"variance_epsilon",None),
                 "prefix_checkpoint_sha256":sha_file(PREFIX_DIR/"model.safetensors"),
                 "base_model_shard_sha256":sha_file(BASE_DIR/"model.safetensors"),
                 "prefix_config_sha256":sha_file(PREFIX_DIR/"config.json"),
                 "base_config_sha256":sha_file(BASE_DIR/"config.json")}
    if norm_compare["prefix_checkpoint_sha256"] != "810d6329cdc51d995543dbb701e2368568b19459a44f340df1a8a0019d447eed":
        raise RuntimeError("prefix checkpoint hash differs from v36 frozen environment")

    # First GPU data operation is S1/P2 search-only clean-process capture reproduction.
    p2_path=CORRECT/"S1_P2_CLEAN_RECAPTURE.json"
    if p2_path.exists():
        p2_report=json.loads(p2_path.read_text(encoding="utf-8"))
        if not p2_report.get("clean_capture_self_replay_exact",False):
            raise RuntimeError("saved v37 clean P2 recapture did not pass its own norm replay")
    else:
        p2_rows=read_group_rows("P2")
        p2_report=ledger.measure("clean_process_prefix_capture", "S1_P2_search_only_no_P3", lambda:capture_fresh_p2(model,p2_rows))
        p2_report["norm_source_comparison"]=norm_compare
        p2_path.write_text(json.dumps(p2_report,indent=2,allow_nan=False)+"\n",encoding="utf-8")

    # Correct the only missing parity field on the already saved S0/G00/down alpha1 candidate.
    parity_path=CORRECT/"G00_DOWN_ALPHA1_PARITY_REPAIRED.json"
    if parity_path.exists():
        parity=json.loads(parity_path.read_text(encoding="utf-8"))
    else:
        parity=parity_and_missing_score(base,ledger)
        parity_path.write_text(json.dumps(parity,indent=2,allow_nan=False)+"\n",encoding="utf-8")

    # Correct selected-D FP16 fold references and independently validate S0 held arms.
    s0_path=RAW/"S0_METRICS_RECOMPUTED.json"
    if s0_path.exists():
        s0={"s0":json.loads(s0_path.read_text(encoding="utf-8"))}
        s0["fold"]={"search":s0["s0"].get("fold_error_search_corrected"),
                    "held":s0["s0"].get("fold_error_held_corrected")}
    else:
        s0=ledger.measure("S0_fold_held_recompute_and_candidate_validation", "P0_search_P1_held_no_new_candidates",
                          lambda:recompute_s0(base,model,ledger))
        s0_path.write_text(json.dumps(s0["s0"],indent=2,allow_nan=False)+"\n",encoding="utf-8")
    v36_metrics={a:float(((np.load(V36/"raw/S0_held_mlp_outputs.npz")[a].astype(np.float32)-
                            np.load(V36/"raw/S0_held_mlp_outputs.npz")["reference"].astype(np.float32))**2).mean())
                 for a in ("G00","G01","G10","G11")}
    out={"status":"COMPLETE_DIAGNOSTIC_PASS" if parity["status"]=="PASS" and
         p2_report["clean_capture_self_replay_exact"] and s0["s0"]["candidate_selection_and_held_path_valid"] else "PARTIAL_OR_FAIL",
         "device":device_info,"v36_gpu_seconds_before_v37":ROUTE_PRIOR,"v37_gpu_seconds_used":ledger.used,
         "input_hash_audit":input_hash_audit,
         "route_gpu_seconds_used":ROUTE_PRIOR+ledger.used,"v37_subbudget_seconds":SUBBUDGET,
         "remaining_original_route_budget_seconds":1800-(ROUTE_PRIOR+ledger.used),
         "s1_p2_search_capture_diagnosis":p2_report,"s0_g00_down_alpha1_parity_repair":parity,
         "s0_independent_revalidation":s0,
         "s0_metrics_numpy_recomputed":v36_metrics,
         "held_data_access":{"S0_P1":"already-opened in v36; read only for frozen revalidation",
                              "S1_P3":"NOT_READ"},
         "new_quality_candidates":"NOT_RUN","S1_quality_candidates":"NOT_RUN","PPL":"NOT_RUN",
         "full_model_quantization":"NOT_RUN","process_wall_seconds":time.perf_counter()-started,
         "stage_records_this_process":ledger.stages,
         "budget_ledger_all_attempts":list(csv.DictReader(LEDGER_PATH.open(newline="",encoding="utf-8")))}
    (CORRECT/"v37_correctness_results.json").write_text(json.dumps(out,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    (RAW/"S0_METRICS_RECOMPUTED.json").write_text(json.dumps(s0["s0"],indent=2,allow_nan=False)+"\n",encoding="utf-8")
    (HERE/"S0_VALIDITY.json").write_text(json.dumps({"status":"VALID" if s0["s0"]["candidate_selection_and_held_path_valid"] else "INVALID_OR_INCOMPLETE",
      "candidate_selection_and_held_path_valid":s0["s0"]["candidate_selection_and_held_path_valid"],
      "quality_gates":s0["s0"]["quality_gates"],"quality_gate_passes":s0["s0"]["quality_gate_passes"],
      "selection_checks":s0["s0"]["selection_checks"],"candidate_bank_checks":s0["s0"]["candidate_bank_checks"],
      "down_input_checks":s0["s0"]["down_input_checks"],"held_output_checks":s0["s0"]["held_output_checks"],
      "reference_exact":s0["s0"]["independent_reference_exact"],"max_metric_difference":s0["s0"]["max_metric_difference_vs_saved"]},indent=2,allow_nan=False)+"\n",encoding="utf-8")
    (CORRECT/"S1_P2_CLEAN_RECAPTURE.json").write_text(json.dumps(p2_report,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    (CORRECT/"G00_DOWN_ALPHA1_PARITY_REPAIRED.json").write_text(json.dumps(parity,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    state={"status":out["status"],"protocol_sha256":sha_file(PROTOCOL_PATH),"v37_gpu_seconds_used":ledger.used,
           "route_gpu_seconds_used":ROUTE_PRIOR+ledger.used,"last_stage":"COMPLETE"}
    STATE_PATH.write_text(json.dumps(state,indent=2)+"\n",encoding="utf-8")
    model.cpu();del model;torch.cuda.empty_cache()
    print(json.dumps({"status":out["status"],"v37_gpu_seconds_used":ledger.used,
                      "p2_fresh_self_replay":p2_report["clean_capture_self_replay_exact"],
                      "p2_old_capture_replay":p2_report["old_capture_replay_exact"],
                      "g00_down_parity":parity["status"],"s0_primary_gate":s0["s0"]["primary_gate"]},indent=2),flush=True)


if __name__ == "__main__":
    main()
