"""Locate FP32 scale ULP differences between full-row and one-row GPTQ shapes."""
import hashlib
import json
import sys
from pathlib import Path

import torch
from gptqmodel.quantization.quantizer import Quantizer

OUT = Path(__file__).resolve().parent
V172 = OUT.parent / "range_w4_fullmodel_smoke_v17_2"
V171 = OUT.parent / "range_w4_m2_parity_v17_1"
sys.path.insert(0, str(V171))
from native_m2 import native_dynamic_gptq


def sha(t):
    return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def main():
    state = torch.load(V172 / "smoke_state.pt", map_location="cuda", weights_only=False)
    w, h = state["W_k"], state["H_k"]
    original = Quantizer.find_params
    records = []

    def capture_find_params(self, current, weight=False):
        result = original(self, current, weight=weight)
        records.append({
            "input": current.detach().cpu().clone(),
            "scale": self.scale.detach().cpu().clone(),
            "zero": self.zero.detach().cpu().clone(),
        })
        return result

    Quantizer.find_params = capture_find_params
    try:
        full = native_dynamic_gptq(w, h, nsamples=state["nsamples_k"], alpha=0.1)
        full_records = records[:]
        records.clear()
        row = native_dynamic_gptq(w[0:1].contiguous(), h, nsamples=state["nsamples_k"], alpha=0.1)
        row_records = records[:]
    finally:
        Quantizer.find_params = original

    groups = []
    for idx, (a, b) in enumerate(zip(full_records, row_records)):
        full_x = a["input"][0:1] if a["input"].ndim == 2 and a["input"].shape[0] > 1 else a["input"]
        row_x = b["input"]
        full_s = a["scale"][0:1] if a["scale"].ndim >= 2 and a["scale"].shape[0] > 1 else a["scale"]
        row_s = b["scale"]
        full_z = a["zero"][0:1] if a["zero"].ndim >= 2 and a["zero"].shape[0] > 1 else a["zero"]
        row_z = b["zero"]
        groups.append({
            "find_params_call": idx,
            "kind": "initial whole-weight estimate" if idx == 0 else f"dynamic group {idx - 1}",
            "input_shape_full": list(a["input"].shape),
            "input_shape_single_row": list(b["input"].shape),
            "current_weight_row_exact": torch.equal(full_x, row_x),
            "current_weight_row_max_abs_diff": float((full_x.float() - row_x.float()).abs().max()),
            "pre_alpha_scale_exact": torch.equal(full_s, row_s),
            "pre_alpha_scale_max_abs_diff": float((full_s.float() - row_s.float()).abs().max()),
            "pre_alpha_scale_max_rel_diff": float(((full_s.float() - row_s.float()).abs() / full_s.float().abs().clamp_min(1e-30)).max()),
            "zero_exact": torch.equal(full_z, row_z),
        })
    result = {
        "scope": "one preselected row 0; alpha=0.1; compare full [128,896] GPTQ to isolated [1,896] GPTQ",
        "full_calls": 1,
        "single_row_calls": 1,
        "full_output_row_exact": torch.equal(full.weight_fp16[0].cpu(), row.weight_fp16[0].cpu()),
        "groups": groups,
        "interpretation": "If compensated group-entry weight rows differ, the first differing group locates shape-dependent arithmetic; if entry rows match but find_params scales differ, the ULP difference is in the per-channel FP32 reduction/arithmetic. The full-linear candidate/stitch path itself always retains the original [out,in] geometry.",
    }
    (OUT / "SCALE_ROUNDING_DIAGNOSTIC.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"full_output_row_exact": result["full_output_row_exact"], "groups": groups}, indent=2))


if __name__ == "__main__":
    main()
