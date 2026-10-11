"""Read-only FP activation shift diagnostic for the frozen search/held split."""
from __future__ import annotations

import json
from pathlib import Path

import torch

from experiments.range_search_w4.run import capture
from experiments.range_search_w4.run_task0_parity import OUTPUT


def main() -> None:
    manifest = json.loads((OUTPUT / "manifest.json").read_text(encoding="utf-8"))
    docs, captured, _, _ = capture()
    if [d["sha256"] for d in docs] != [d["sha256"] for d in manifest["documents"]]:
        raise RuntimeError("article hashes changed")
    result = {"scope": "FP projection input distribution only; descriptive, not a causal test",
              "search_article_hashes": [d["sha256"] for d in docs[:4]],
              "heldout_article_hashes": [d["sha256"] for d in docs[4:]], "layers": {}}
    for layer in (0, 12, 23):
        docs_x = captured[str(layer)]
        xs = torch.cat(docs_x[:4]).float()
        xh = torch.cat(docs_x[4:]).float()
        rms_s = xs.square().mean(0).sqrt()
        rms_h = xh.square().mean(0).sqrt()
        rel_rms = (rms_h - rms_s).abs() / rms_s.clamp_min(1e-12)
        gram_s = xs.T @ xs / xs.shape[0]
        gram_h = xh.T @ xh / xh.shape[0]
        result["layers"][str(layer)] = {
            "search_rms": float(xs.square().mean().sqrt()),
            "heldout_rms": float(xh.square().mean().sqrt()),
            "global_rms_ratio": float(xh.square().mean().sqrt() / xs.square().mean().sqrt()),
            "feature_rms_relative_change_median": float(rel_rms.median()),
            "feature_rms_relative_change_p90": float(torch.quantile(rel_rms, .9)),
            "gram_relative_frobenius_change": float(torch.linalg.vector_norm(gram_h - gram_s) /
                                                      torch.linalg.vector_norm(gram_s)),
            "group_rms_ratio": [float(xh[:, i:i+128].square().mean().sqrt() /
                                      xs[:, i:i+128].square().mean().sqrt())
                                for i in range(0, xs.shape[1], 128)]}
    (OUTPUT / "activation_shift_diagnostics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["layers"]["23"], ensure_ascii=False))


if __name__ == "__main__":
    main()
