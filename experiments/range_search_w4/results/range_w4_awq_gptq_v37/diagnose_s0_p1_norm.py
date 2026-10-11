"""Small v37 diagnostic: replay only frozen S0 P0/P1/P2 L12 RMSNorm inputs."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_v37_correctness as v37  # noqa: E402


def main() -> None:
    ledger = v37.Ledger()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    start = time.perf_counter()
    torch.cuda.init()
    v37.sync()
    ledger.record("cuda_init", "S0_P1_norm_replay", time.perf_counter() - start)
    model = ledger.measure(
        "model_load", "S0_P1_norm_replay",
        lambda: AutoModelForCausalLM.from_pretrained(
            str(v37.PREFIX_DIR), torch_dtype=torch.float16, low_cpu_mem_usage=True,
            local_files_only=True, attn_implementation="sdpa").eval().to(v37.DEVICE))
    norm = model.model.layers[v37.LAYER].post_attention_layernorm
    base_norm = ledger.measure("pristine_norm_load", "L12_base_norm", lambda: v37.load_base()["norm"])
    d_obj = ledger.measure("locked_D_load", "S0_search_selected_D", lambda: torch.load(
        v37.V36 / "final/S0_selected_D.pt", map_location="cpu", weights_only=True))
    d = d_obj["D"].to(device=v37.DEVICE, dtype=torch.float16)
    scaled_norm = (base_norm.float() / d.float()).half()
    d_sha256 = v37.sha_tensor(d)
    reports = {}
    for split, group in (("S0", "P0"), ("S0", "P1"), ("S1", "P2")):
        stem = f"{split}_{group}"
        pre = np.memmap(v37.V36 / "cache" / f"{stem}_prenorm.f16", dtype=np.float16,
                        mode="r", shape=(v37.NDOC, v37.SEQ, 896))
        old_post = np.memmap(v37.V36 / "cache" / f"{stem}_postnorm.f16", dtype=np.float16,
                             mode="r", shape=(v37.NDOC, v37.SEQ, 896))
        pre_t = torch.from_numpy(np.array(pre, copy=True)).to(v37.DEVICE)
        old_t = torch.from_numpy(np.array(old_post, copy=True)).to(v37.DEVICE)

        def replay():
            with torch.inference_mode():
                batched = norm(pre_t)
                per_doc = torch.cat([norm(pre_t[i:i + 1]) for i in range(v37.NDOC)], dim=0)
            return batched, per_doc

        batched, per_doc = ledger.measure("norm_replay", f"{stem}_batched_and_per_doc", replay)
        norm_before = norm.weight.detach().clone()
        with torch.no_grad():
            norm.weight.copy_(scaled_norm)
        scaled_replay = ledger.measure("locked_S0_D_norm_replay", f"{stem}_using_S0_selected_D", replay)[0]
        with torch.no_grad():
            norm.weight.copy_(norm_before)

        def summary(a: torch.Tensor, b: torch.Tensor) -> dict:
            delta = (a.float() - b.float()).abs()
            return {"exact": bool(torch.equal(a, b)), "max_abs": float(delta.max().item()),
                    "mean_abs": float(delta.mean().item()),
                    "equal_elements": int((a == b).sum().item()), "elements": int(a.numel())}

        reports[stem] = {
            "pre_sha256": v37.sha_tensor(pre_t), "old_post_sha256": v37.sha_tensor(old_t),
            "norm_weight_sha256": v37.sha_tensor(norm.weight),
            "norm_weight_matches_pristine": torch.equal(norm.weight, base_norm),
            "locked_S0_D_sha256": d_sha256,
            "locked_S0_scaled_norm_sha256": v37.sha_tensor(scaled_norm),
            "locked_S0_D_replay_vs_old_post": summary(scaled_replay, old_t),
            "batched_replay_vs_old_post": summary(batched, old_t),
            "per_doc_replay_vs_old_post": summary(per_doc, old_t),
            "batched_vs_per_doc": summary(batched, per_doc),
        }
        pre._mmap.close()
        old_post._mmap.close()
        del pre_t, old_t, batched, per_doc, scaled_replay, norm_before
    out = {"status": "COMPLETE", "scope": "norm-only replay of existing cache; no candidate/GPTQ/quality scoring",
           "device": torch.cuda.get_device_name(0), "reports": reports,
           "v37_gpu_seconds_used": ledger.used, "route_gpu_seconds_used": v37.ROUTE_PRIOR + ledger.used,
           "source_sha256": v37.sha_file(Path(__file__))}
    (HERE / "raw" / "S0_P1_NORM_REPLAY_WITH_D.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    model.cpu()
    del model
    torch.cuda.empty_cache()
    print(json.dumps({"status": out["status"], "v37_gpu_seconds_used": out["v37_gpu_seconds_used"],
                      "reports": reports}, indent=2))


if __name__ == "__main__":
    main()
