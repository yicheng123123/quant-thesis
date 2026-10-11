"""Re-evaluate the already locked v36 S0 arms against a pristine P1 reference.

No GPTQ/candidate generation, selector search, or alpha recomputation is done here.
All corrected outputs are new v37 files; v36 artifacts remain unchanged.
"""
from __future__ import annotations

import csv
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_v37_correctness as v37  # noqa: E402


def sha_np(x: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()


@torch.inference_mode()
def replay_full_block(model, hidden_mm, pre_mm, base, state, d, reference):
    layer = model.model.layers[v37.LAYER]
    mlp = layer.mlp
    norm = layer.post_attention_layernorm
    old_norm = norm.weight.detach().clone()
    old_weights = {n: mlp.get_submodule(n).weight.detach().clone()
                   for n in ("gate_proj", "up_proj", "down_proj")}
    arm = state["arm"]
    norm_value = base["norm"] if d is None else (base["norm"].float()/d.float()).half()
    v37.set_weight(norm, norm_value)
    for name, key in (("gate_proj", "gate"), ("up_proj", "up"), ("down_proj", "down")):
        v37.set_weight(mlp.get_submodule(name), state[key]["weight"].to(v37.DEVICE))
    committed = {name: v37.sha_tensor(mlp.get_submodule(name).weight)
                 for name in ("gate_proj", "up_proj", "down_proj")}
    local = {"mlp": None, "pre": None, "post": None, "down_in": None, "down_out": None}
    def mlp_hook(_m, _args, out): local["mlp"] = out.detach().clone()
    def pre_hook(_m, args): local["pre"] = args[0].detach().clone()
    def post_hook(_m, _args, out): local["post"] = out.detach().clone()
    def down_pre(_m, args): local["down_in"] = args[0].detach().clone()
    def down_hook(_m, _args, out): local["down_out"] = out.detach().clone()
    hooks = [mlp.register_forward_hook(mlp_hook), norm.register_forward_pre_hook(pre_hook),
             norm.register_forward_hook(post_hook), mlp.down_proj.register_forward_pre_hook(down_pre),
             mlp.down_proj.register_forward_hook(down_hook)]
    pos = torch.arange(v37.SEQ, device=v37.DEVICE)[None]
    outputs = np.empty((v37.NDOC, v37.SEQ, 896), dtype=np.float16)
    block_outputs = np.empty_like(outputs)
    l13_inputs = np.empty_like(outputs)
    try:
        for i in range(v37.NDOC):
            h = torch.from_numpy(np.array(hidden_mm[i], copy=True)).unsqueeze(0).to(v37.DEVICE)
            expected_pre = torch.from_numpy(np.array(pre_mm[i], copy=True)).unsqueeze(0).to(v37.DEVICE)
            expected_post = norm(expected_pre)
            pos_emb = model.model.rotary_emb(h, pos)
            out = layer(h, attention_mask=None, position_ids=pos, cache_position=pos[0],
                        position_embeddings=pos_emb, use_cache=False)[0]
            if not torch.equal(local["pre"], expected_pre):
                raise RuntimeError(f"held P1 actual norm input differs from frozen capture for {arm}, article {i}")
            if not torch.equal(local["post"], expected_post):
                raise RuntimeError(f"held P1 actual norm output differs from arm norm for {arm}, article {i}")
            if local["down_in"] is None or tuple(local["down_in"].shape) != (1,v37.SEQ,4864):
                raise RuntimeError(f"held P1 actual down input shape failed for {arm}, article {i}")
            if local["mlp"] is None or local["down_out"] is None or not torch.equal(local["mlp"],local["down_out"]):
                raise RuntimeError(f"held P1 MLP/down output hook mismatch for {arm}, article {i}")
            outputs[i] = local["mlp"][0].cpu().numpy()
            block_outputs[i] = out[0].cpu().numpy()
            l13_inputs[i] = model.model.layers[v37.LAYER+1].input_layernorm(out)[0].cpu().numpy()
            local.update(mlp=None,pre=None,post=None,down_in=None,down_out=None)
            del h, expected_pre, expected_post, pos_emb, out
    finally:
        for hook in hooks: hook.remove()
        with torch.no_grad():
            norm.weight.copy_(old_norm)
            for name, weight in old_weights.items(): mlp.get_submodule(name).weight.copy_(weight)
    err=torch.from_numpy(outputs).to(v37.DEVICE).float()-reference.float()
    article_mse=err.double().square().mean(dim=(1,2)).cpu().tolist()
    full_mse=float(err.double().square().mean().item())
    return {"output":outputs,"block_output":block_outputs,"l13":l13_inputs,
            "full_mlp_mse":full_mse,"article_mse":article_mse,
            "boundary":{"mlp_out_sha256":sha_np(outputs),"block_output_sha256":sha_np(block_outputs),
                "l13_input_sha256":sha_np(l13_inputs),"committed_weights":committed}}


def main() -> None:
    ledger = v37.Ledger()
    if ledger.used >= 240:
        raise RuntimeError(f"v37 pre-reserved stop threshold reached before S0 held repair: {ledger.used:.3f}")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    t0 = time.perf_counter()
    torch.cuda.init(); v37.sync()
    ledger.record("cuda_init", "S0_P1_repair", time.perf_counter() - t0)
    model = ledger.measure("model_load", "frozen_v18_M2_prefix", lambda: AutoModelForCausalLM.from_pretrained(
        str(v37.PREFIX_DIR), torch_dtype=torch.float16, low_cpu_mem_usage=True,
        local_files_only=True, attn_implementation="sdpa").eval().to(v37.DEVICE))
    model.eval()
    base = ledger.measure("pristine_weight_load", "L12_target", v37.load_base)
    layer = model.model.layers[v37.LAYER]
    norm = layer.post_attention_layernorm
    selected = torch.load(v37.V36 / "final/S0_selected_tensors.pt", map_location="cpu", weights_only=True)
    lock = json.loads((v37.V36 / "final/S0_SELECTION_LOCK.json").read_text(encoding="utf-8"))
    lock_tensors = lock["selected_tensors"]
    lock_checks = {"D": v37.sha_tensor(selected["D"]) == lock["D_selection"]["selected_D_sha256"]}
    for arm in ("G00", "G01", "G10", "G11"):
        for proj in ("gate", "up", "down"):
            for fld in ("weight", "scale", "zero", "g_idx"):
                lock_checks[f"{arm}.{proj}.{fld}"] = v37.sha_tensor(selected[arm][proj][fld]) == lock_tensors[f"{arm}.{proj}"][f"{fld}_sha256"]
    if not all(lock_checks.values()):
        raise RuntimeError("S0 selection lock does not match stored selected tensors")
    d = selected["D"].to(device=v37.DEVICE, dtype=torch.float16)

    # P1 hidden and prenorm captures were produced before any target-arm commit.
    # Recreate pristine norm output directly from frozen prenorm, not stale postnorm.
    pre = np.memmap(v37.V36 / "cache/S0_P1_prenorm.f16", dtype=np.float16, mode="r",
                    shape=(v37.NDOC, v37.SEQ, 896))
    hidden = np.memmap(v37.V36 / "cache/S0_P1_hidden_l12.f16", dtype=np.float16, mode="r",
                       shape=(v37.NDOC, v37.SEQ, 896))
    pre_t = torch.from_numpy(np.array(pre, copy=True)).to(v37.DEVICE)
    with torch.no_grad():
        norm.weight.copy_(base["norm"])
    def make_reference_input():
        with torch.inference_mode():
            return torch.cat([norm(pre_t[i:i+1]) for i in range(v37.NDOC)], dim=0)
    x_ref = ledger.measure("pristine_P1_norm_replay", "all_8_articles", make_reference_input)
    x_ref_flat = x_ref.reshape(-1, 896)
    originals = {n: layer.mlp.get_submodule(n).weight.detach().clone() for n in ("gate_proj", "up_proj", "down_proj")}
    original_norm = norm.weight.detach().clone()
    def reference_forward():
        with torch.inference_mode():
            for n, k in (("gate_proj", "gate"), ("up_proj", "up"), ("down_proj", "down")):
                v37.set_weight(layer.mlp.get_submodule(n), base[k])
            return layer.mlp(x_ref_flat).detach().clone()
    reference = ledger.measure("pristine_P1_reference_forward", "all_16384_tokens", reference_forward)
    reference_3d = reference.reshape(v37.NDOC, v37.SEQ, 896)

    old_npz = np.load(v37.V36 / "raw/S0_held_mlp_outputs.npz")
    arm_outputs = {}
    arm_replays = {}
    for arm in ("G00", "G01", "G10", "G11"):
        state = {k: selected[arm][k] for k in ("gate", "up", "down")}
        state["arm"] = arm
        arm_d = None if arm in ("G00", "G01") else d
        result = ledger.measure("full_block_arm_replay", f"S0_P1_{arm}", lambda state=state, arm_d=arm_d: \
            replay_full_block(model, hidden, pre, base, state, arm_d, reference_3d))
        output = np.asarray(result["output"])
        old = np.asarray(old_npz[arm])
        arm_outputs[arm] = output
        arm_replays[arm] = {"exact_vs_v36_saved_full_block_output": bool(np.array_equal(output, old)),
            "old_output_sha256": sha_np(old), "replayed_output_sha256": sha_np(output),
            "max_abs": float(np.max(np.abs(output.astype(np.float32)-old.astype(np.float32)))),
            "full_mlp_mse_vs_correct_reference": result["full_mlp_mse"],
            "l13_boundary": result["boundary"], "replayed_l13_sha256": result["boundary"]["l13_input_sha256"]}

    # Recompute original v36 per-projection row diagnostics using the corrected P1 input.
    projection_rows: dict[str, dict[str, np.ndarray]] = {}
    def projection_scoring():
        for arm in ("G00", "G01", "G10", "G11"):
            scaled = arm in ("G10", "G11")
            if scaled:
                v37.set_weight(norm, (base["norm"].float()/d.float()).half())
                coord_x = norm(pre_t).reshape(-1,896)
            else:
                v37.set_weight(norm, base["norm"])
                coord_x = x_ref_flat
            st = selected[arm]
            qg = F.linear(coord_x, st["gate"]["weight"].to(v37.DEVICE))
            qu = F.linear(coord_x, st["up"]["weight"].to(v37.DEVICE))
            refg = F.linear(x_ref_flat, base["gate"])
            refu = F.linear(x_ref_flat, base["up"])
            down_in = layer.mlp.act_fn(qg) * qu
            qdown = F.linear(down_in, st["down"]["weight"].to(v37.DEVICE))
            refdown = F.linear(down_in, base["down"])
            values = {}
            for key, a, b in (("gate",qg,refg),("up",qu,refu),("down",qdown,refdown)):
                err=(a.float()-b.float()).reshape(v37.NDOC,v37.SEQ,-1).double().square()
                values[key]=err.sum(dim=(0,1))/(v37.NDOC*v37.SEQ)
            err=(torch.from_numpy(arm_outputs[arm]).to(v37.DEVICE).float()-reference_3d.float()).double().square()
            values["full_mlp"]=err.sum(dim=(0,1))/(v37.NDOC*v37.SEQ)
            projection_rows[arm]={k:v.detach().cpu().numpy() for k,v in values.items()}
            del coord_x,qg,qu,refg,refu,down_in,qdown,refdown,err
        return True
    ledger.measure("projection_and_row_scoring", "all_rows_all_arms_correct_P1", projection_scoring)

    # Correct full-fold diagnostic: compare pristine MLP on pristine norm(pre)
    # with the FP16 folded norm and W*D projections.
    def fold_metrics(group: str) -> dict:
        pre_mm = np.memmap(v37.V36 / f"cache/S0_{group}_prenorm.f16", dtype=np.float16, mode="r",
                           shape=(v37.NDOC,v37.SEQ,896))
        post_mm = np.memmap(v37.V36 / f"cache/S0_{group}_postnorm.f16", dtype=np.float16, mode="r",
                            shape=(v37.NDOC,v37.SEQ,896))
        pre_local=torch.from_numpy(np.array(pre_mm,copy=True)).to(v37.DEVICE)
        post=torch.from_numpy(np.array(post_mm,copy=True)).to(v37.DEVICE)
        if group=="P1": post=x_ref
        v37.set_weight(norm,base["norm"])
        for n,k in (("gate_proj","gate"),("up_proj","up"),("down_proj","down")):
            v37.set_weight(layer.mlp.get_submodule(n),base[k])
        with torch.inference_mode(): ref=layer.mlp(post.reshape(-1,896)).detach()
        v37.set_weight(norm,(base["norm"].float()/d.float()).half())
        v37.set_weight(layer.mlp.gate_proj,(base["gate"].float()*d.float()[None,:]).half())
        v37.set_weight(layer.mlp.up_proj,(base["up"].float()*d.float()[None,:]).half())
        v37.set_weight(layer.mlp.down_proj,base["down"])
        with torch.inference_mode():
            xfold=torch.cat([norm(pre_local[i:i+1]) for i in range(v37.NDOC)],dim=0)
            folded=layer.mlp(xfold.reshape(-1,896)).detach()
        delta=folded.float()-ref.float()
        report={"group":group,"D_sha256":v37.sha_tensor(d),"reference_input_sha256":v37.sha_tensor(post),
            "folded_input_sha256":v37.sha_tensor(xfold),"fold_error_mse":float(delta.double().square().mean().item()),
            "fold_error_max_abs":float(delta.abs().max().item()),"finite":bool(torch.isfinite(delta).all()),
            "correct_fold":"norm weight / D; pristine gate/up W*D in FP16; down unchanged"}
        pre_mm._mmap.close();post_mm._mmap.close()
        return report
    folds={"search_P0":ledger.measure("correct_full_fold_reference", "S0_P0_search", lambda:fold_metrics("P0")),
           "held_P1":ledger.measure("correct_full_fold_reference", "S0_P1_held", lambda:fold_metrics("P1"))}

    # Restore modules before finalizing and write only new v37 artifacts.
    with torch.no_grad():
        norm.weight.copy_(original_norm)
        for n,w in originals.items(): layer.mlp.get_submodule(n).weight.copy_(w)
    v37.sync()
    metrics={arm:float(np.mean([r["full_mlp_mse_vs_correct_reference"] for r in [arm_replays[arm]]]))
             for arm in arm_replays}
    metric_recalc={arm:float(((arm_outputs[arm].astype(np.float32)-reference_3d.detach().cpu().numpy().astype(np.float32))**2).mean())
                   for arm in arm_outputs}
    tails={}; projection_quality={}
    for proj in ("gate","up","down","full_mlp"):
        baseline=projection_rows["G01"][proj]
        for arm in ("G00","G10","G11"):
            candidate=projection_rows[arm][proj]
            rel=(candidate-baseline)/np.maximum(baseline,1e-30)
            tails[f"{arm}_vs_G01_{proj}"]={"wins":int(np.sum(candidate<baseline)),"ties":int(np.sum(candidate==baseline)),
                "losses":int(np.sum(candidate>baseline)),"median_relative_improvement":float(np.median(-rel)),
                "worst_relative_degradation":float(np.max(rel)),"rows_over_10pct_degradation":int(np.sum(rel>0.10)),
                "rows_over_50pct_degradation":int(np.sum(rel>0.50)),"row_count":int(candidate.size)}
            if proj in ("gate","up","down"):
                projection_quality[f"{arm}_vs_G01_{proj}"]={"pooled_mse_change":float(candidate.mean()/baseline.mean()-1),
                    "no_more_than_5pct_regression":bool(candidate.mean()<=baseline.mean()*1.05)}
    improvement=1.0-metrics["G11"]/metrics["G01"]
    fold_held=folds["held_P1"]["fold_error_mse"]
    projection_scope={k:v for k,v in tails.items() if k.endswith(("_gate","_up","_down"))}
    summary={"status":"S0_REEVALUATED_FROM_LOCKED_CANDIDATES",
        "candidate_generation":"NOT_RUN; exact v36 frozen selected tensors used",
        "P1_reference":"replayed from frozen P1 prenorm with pristine L12 norm and original FP16 MLP",
        "selection_lock_checks":lock_checks,
        "pristine_P1_postnorm_sha256":v37.sha_tensor(x_ref),
        "old_P1_postnorm_sha256":v37.sha_file(v37.V36/"cache/S0_P1_postnorm.f16"),
        "old_P1_postnorm_reused":"NO; proven to equal norm(base/D, prenorm)",
        "saved_full_block_arm_replay":arm_replays,
        "arm_full_mlp_mse":metrics,"independent_metric_recalc_from_np_arrays":metric_recalc,
        "max_mse_difference_between_recalc_paths":max(abs(metrics[a]-metric_recalc[a]) for a in metrics),
        "contrasts_fractional_loss_change":{"G10_vs_G00":metrics["G10"]/metrics["G00"]-1,
            "G11_vs_G10":metrics["G11"]/metrics["G10"]-1,"G11_vs_G01":metrics["G11"]/metrics["G01"]-1},
        "primary_G11_vs_G01_improvement":improvement,"primary_gate_minimum":0.01,
        "primary_gate_passes":bool(improvement>=0.01),"correct_fold_errors":folds,
        "fold_signal_floor":{"G11_mse":metrics["G11"],"five_x_fold_error":5*fold_held,
            "ratio":metrics["G11"]/max(fold_held,1e-30),"passes":bool(metrics["G11"]>=5*fold_held)},
        "projection_row_tail":tails,"projection_quality":projection_quality,
        "all_projection_gates_pass":all(v["no_more_than_5pct_regression"] for v in projection_quality.values()),
        "all_projection_row_tail_gates_pass":all(v["rows_over_10pct_degradation"]<=0.05*v["row_count"] and
                                                    v["rows_over_50pct_degradation"]==0 for v in projection_scope.values()),
        "full_mlp_row_tails_descriptive_only":{k:v for k,v in tails.items() if k.endswith("_full_mlp")},
        "GPU_seconds_used":ledger.used,"route_gpu_seconds_used":v37.ROUTE_PRIOR+ledger.used,
        "source_sha256":v37.sha_file(Path(__file__))}
    np.savez_compressed(v37.HERE/"raw/S0_HELD_REPAIRED_OUTPUTS.npz", **arm_outputs,
                        reference=reference_3d.detach().cpu().numpy(), pristine_postnorm=x_ref.detach().cpu().numpy())
    (v37.HERE/"raw/S0_HELD_REPAIRED_RESULT.json").write_text(json.dumps(summary,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    with (v37.HERE/"raw/S0_HELD_REPAIRED_PROJECTION_ROWS.csv").open("w",newline="",encoding="utf-8") as f:
        wr=csv.writer(f); wr.writerow(["arm","projection","row","mse"])
        for arm,projs in projection_rows.items():
            for proj,vals in projs.items():
                wr.writerows((arm,proj,int(i),float(x)) for i,x in enumerate(vals))
    with (v37.HERE/"raw/S0_HELD_REPAIRED_ARTICLES.csv").open("w",newline="",encoding="utf-8") as f:
        wr=csv.writer(f); wr.writerow(["group","arm","article_index","token_sha256","full_mlp_mse","score_dtype"])
        articles=json.loads((v37.V36/"cache/P1_tokens.json").read_text(encoding="utf-8"))
        for arm in arm_outputs:
            for i,row in enumerate(articles):
                diff=(torch.from_numpy(arm_outputs[arm][i]).float()-
                      torch.from_numpy(reference_3d[i].detach().cpu().numpy()).float())
                wr.writerow(["P1",arm,row.get("source_article_index",i),row.get("token_sha256"),
                             float(diff.square().mean().item()),"float32 per-article M2 scoring rule"])
    pre._mmap.close(); hidden._mmap.close()
    old_npz.close()
    model.cpu(); del model; torch.cuda.empty_cache()
    print(json.dumps({"status":summary["status"],"GPU_seconds_used":summary["GPU_seconds_used"],
        "arm_full_mlp_mse":metrics,"primary_improvement":improvement,"primary_gate_passes":summary["primary_gate_passes"],
        "all_projection_gates_pass":summary["all_projection_gates_pass"],
        "all_projection_row_tail_gates_pass":summary["all_projection_row_tail_gates_pass"]},indent=2))


if __name__ == "__main__":
    main()
