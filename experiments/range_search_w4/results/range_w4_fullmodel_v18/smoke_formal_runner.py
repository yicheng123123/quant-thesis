"""Orchestration smoke for the exact v18 M2 candidate/capture functions."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM

import run_formal as f


def main():
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(20261005)
    f.FORMAL_TIMING = f.OUT / "logs" / "formal_runner_smoke_timing.csv"
    rec = f.Recorder()
    try:
        model = AutoModelForCausalLM.from_pretrained(
            str(f.MODEL_DIR), torch_dtype=torch.float16, low_cpu_mem_usage=True,
            local_files_only=True, attn_implementation="sdpa",
        ).eval().to(f.DEVICE)
        layer = model.model.layers[0]
        hidden = np.memmap(f.OUT / "raw/precheck_activations/L0_hidden_in.f16",
                           dtype=np.float16, mode="r", shape=(128, 2048, 896))
        x = np.memmap(f.OUT / "raw/precheck_activations/L0_qkv_input.f16",
                      dtype=np.float16, mode="r", shape=(128, 2048, 896))
        timing_rows = []
        h, ns, hs = f.rp.build_hessian(layer.self_attn.k_proj, x, timing_rows, "M2 smoke kv H")
        prepared = f.prepare_native_state(h, nsamples=ns)
        counts = {"full_linear_alpha_candidate_executions": 0, "row_alpha_score_slots": 0,
                  "alpha1_native_parity_reference_executions": 0}
        k = layer.self_attn.k_proj.weight.detach().clone()
        state = f.quantize_search_module("M2_SMOKE", 0, "self_attn.k_proj",
            layer.self_attn.k_proj, h, prepared, ns, x, f.rp.sha_memmap(x), hs["sha256"], rec, counts)
        if counts["full_linear_alpha_candidate_executions"] != 10 or counts["row_alpha_score_slots"] != 1280:
            raise RuntimeError(f"formal M2 work-unit mismatch: {counts}")
        if state["candidate_losses"].shape != (10, 128) or state["weight"].shape != (128, 896):
            raise RuntimeError("formal M2 candidate/state geometry mismatch")
        if not np.isfinite(state["candidate_losses"].numpy()).all():
            raise RuntimeError("formal M2 candidate landscape is nonfinite")
        # Exercise the production capture hook after a whole Linear commit.
        x_capture, x_hash = f.run_block_capture(
            model, layer, hidden, ["self_attn.o_proj"],
            f.OUT / "raw/activations/smoke_o_input.f16", "M2_SMOKE", 0,
            "smoke_qkv_commit_to_o_input", rec)
        if x_capture.shape != (128, 2048, 896) or len(x_hash) != 64:
            raise RuntimeError("formal boundary capture output mismatch")
        f.close_mm(x_capture)
        hidden._mmap.close(); x._mmap.close()
        del model, prepared, h, state, k
        torch.cuda.empty_cache()
        result = {"status": "PASS", "method": "v18 formal runner functions",
                  "candidate_executions": counts["full_linear_alpha_candidate_executions"],
                  "row_alpha_score_slots": counts["row_alpha_score_slots"],
                  "alpha1_native_parity_reference_executions": counts["alpha1_native_parity_reference_executions"],
                  "selected_weight_shape": [128, 896], "capture_shape": [128, 2048, 896],
                  "post_commit_o_input_sha256": x_hash,
                  "checks": {"all_candidate_losses_finite": True, "candidate_count_exact": True,
                             "per_row_stitch_state_shapes": True, "post_whole_linear_capture": True}}
        (f.OUT / "FORMAL_RUNNER_SMOKE.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(result, indent=2))
    finally:
        rec.close()


if __name__ == "__main__":
    main()
