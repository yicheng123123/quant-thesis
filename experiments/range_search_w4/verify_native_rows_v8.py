"""Validate all six custom-alpha=1 rows against native full-layer GPTQModel."""
from __future__ import annotations

import json
from pathlib import Path

import torch

from experiments.range_search_w4.dynamic_core import dynamic_gptq, native_hinv_from_documents, output_sse
from experiments.range_search_w4.run import capture
from experiments.range_search_w4.run_task0_parity import OUTPUT


def main() -> None:
    from gptqmodel.quantization import QuantizeConfig
    from gptqmodel.quantization.gptq import GPTQ

    docs, captured, weights, _ = capture()
    manifest = json.loads((OUTPUT / "manifest.json").read_text(encoding="utf-8"))
    if [d["sha256"] for d in docs] != [d["sha256"] for d in manifest["documents"]]:
        raise RuntimeError("article hashes changed")
    entries = []
    for layer in (0, 12, 23):
        xs_docs = [x.cuda() for x in captured[str(layer)][:4]]
        xs, xh = torch.cat(xs_docs), torch.cat(captured[str(layer)][4:]).cuda()
        w = weights[str(layer)].cuda()
        module = torch.nn.Linear(w.shape[1], w.shape[0], bias=False,
                                 device="cuda", dtype=torch.float16)
        module.weight.data.copy_(w.half())
        cfg = QuantizeConfig(bits=4, group_size=128, sym=True, desc_act=False,
                             static_groups=False, true_sequential=True,
                             damp_percent=0.1, mse=0.0)
        native = GPTQ(module, qcfg=cfg)
        native.quantizer.configure(perchannel=True, bits=4, sym=True)
        for x in xs_docs:
            native.process_batch(x[None])
        native_q, *_ = native.quantize(blocksize=128)
        _, hinv, dead = native_hinv_from_documents(xs_docs, module, 0.1)
        if dead:
            raise RuntimeError("unexpected dead columns")
        for row in (0, 448):
            own = dynamic_gptq(w[row:row+1], hinv, alpha=1.0).weight_fp32[0].half()
            raw = json.loads((OUTPUT / f"layer{layer}_row{row}.json").read_text(encoding="utf-8"))
            d = raw["methods"]["D_native_dynamic"]
            native_s = output_sse(xs, w[row], native_q[row])
            native_h = output_sse(xh, w[row], native_q[row])
            diff = (native_q[row].float() - own.float()).abs()
            entries.append({"layer": layer, "row": row,
                "fp16_weight_equal": bool(torch.equal(native_q[row], own)),
                "weight_max_abs": float(diff.max()), "weight_mismatched_elements": int((diff != 0).sum()),
                "native_search_sse": native_s, "stored_D_search_sse": d["search_sse"],
                "native_heldout_sse": native_h, "stored_D_heldout_sse": d["heldout_sse"],
                "search_sse_relative_diff": abs(native_s - d["search_sse"]) / max(native_s, 1e-30),
                "heldout_sse_relative_diff": abs(native_h - d["heldout_sse"]) / max(native_h, 1e-30)})
    ok = all(v["fp16_weight_equal"] and v["search_sse_relative_diff"] < 1e-6 and
             v["heldout_sse_relative_diff"] < 1e-6 for v in entries)
    (OUTPUT / "native_six_row_check.json").write_text(json.dumps({"status": "PASS" if ok else "FAIL",
        "rows": entries}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("NATIVE_SIX_ROW", "PASS" if ok else "FAIL")
    if not ok:
        raise RuntimeError("native row parity failed")


if __name__ == "__main__":
    main()
