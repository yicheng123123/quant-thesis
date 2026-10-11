"""Real layer-0 native-vs-custom dynamic G128 parity gate.

This command creates one isolated v8 directory and never writes v1-v7 files.
It must pass before any v8 candidate-search command is used.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
from pathlib import Path
import time

from experiments.range_search_w4.run import PROTOCOL as V7, ROOT, capture, sync

import torch

from experiments.range_search_w4.core import prepare_hinv
from experiments.range_search_w4.dynamic_core import (
    dynamic_gptq, native_hinv_from_documents, output_sse, weight_mse,
)

HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "results" / "range_w4_dynamic_v8"
HISTORY = HERE / "results" / "range_w4_local_v7"


def historical_fingerprint() -> dict[str, dict]:
    result = {}
    for path in sorted(HISTORY.iterdir()):
        if path.is_file():
            result[path.name] = {
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "mtime_ns": path.stat().st_mtime_ns,
                "size_bytes": path.stat().st_size,
            }
    return result


def save_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def tensor_hash(value: torch.Tensor) -> str:
    return hashlib.sha256(value.detach().contiguous().cpu().numpy().tobytes()).hexdigest()


def main() -> None:
    if OUTPUT.exists() and any(path.is_file() for path in OUTPUT.iterdir()):
        raise FileExistsError(f"v8 parity root already contains results: {OUTPUT}")
    before = historical_fingerprint()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    docs, captured, weights, capture_peak = capture()
    xdocs = [item.cuda() for item in captured["0"][:4]]
    xsearch = torch.cat(xdocs)
    xheld = torch.cat(captured["0"][4:]).cuda()
    original = weights["0"].cuda()
    module = torch.nn.Linear(original.shape[1], original.shape[0], bias=False,
                             device="cuda", dtype=torch.float16)
    module.weight.data.copy_(original.half())

    from gptqmodel.quantization import QuantizeConfig
    from gptqmodel.quantization.gptq import GPTQ

    cfg = QuantizeConfig(bits=4, group_size=128, sym=True, desc_act=False,
                         static_groups=False, true_sequential=True,
                         damp_percent=0.1, mse=0.0)
    native = GPTQ(module, qcfg=cfg)
    native.quantizer.configure(perchannel=True, bits=4, sym=True)
    for x in xdocs:
        native.process_batch(x[None])
    h_native = native.H.detach().clone()
    native_entries = []
    native_scales = []
    native_zeros = []
    original_find = native.quantizer.find_params

    def traced_find(x, weight=False):
        if x.shape[-1] == 128:
            native_entries.append(x.detach().float().cpu().clone())
        original_find(x, weight=weight)
        if x.shape[-1] == 128:
            native_scales.append(native.quantizer.scale.detach().float().cpu().clone())
            native_zeros.append(native.quantizer.zero.detach().float().cpu().clone())

    native.quantizer.find_params = traced_find
    original_inverse = native.block_cholesky_inverse
    inverse_results = []

    def traced_inverse(chol, *args, **kwargs):
        inverse = original_inverse(chol, *args, **kwargs)
        inverse_results.append(inverse.detach().clone())
        return inverse

    native.block_cholesky_inverse = traced_inverse
    sync()
    native_start = time.perf_counter()
    native_q, returned_scale, returned_zero, group_idx, *_ = native.quantize(blocksize=128)
    sync()
    native_seconds = time.perf_counter() - native_start
    if len(inverse_results) != 1 or len(native_entries) != 7:
        raise RuntimeError("native trace did not observe one inverse and seven groups")
    hinv_native = torch.linalg.cholesky(inverse_results[0], upper=True)
    sync()
    custom_start = time.perf_counter()
    custom = dynamic_gptq(original, hinv_native, alpha=1.0,
                          group_size=128, block_size=128, trace=True)
    custom_q = custom.weight_fp32.half()
    sync()
    custom_seconds = time.perf_counter() - custom_start
    h_helper, hinv_helper, dead_count = native_hinv_from_documents(xdocs, module, 0.1)
    h_v7 = prepare_hinv(xsearch, document_count=4, damp_percent=0.1)
    h_native_damped = h_native.clone()
    h_native_damped.diagonal().add_(0.1 * h_native_damped.diagonal().mean())

    input_diffs = [float((a - b).abs().max()) for a, b in zip(native_entries, custom.group_inputs)]
    scale_diff = float((returned_scale.float() - custom.scales).abs().max())
    zero_diff = float((returned_zero.float() - custom.zeros).abs().max())
    weight_abs = (native_q.float() - custom_q.float()).abs()
    native_trace_scale_diff = float((returned_scale.cpu() - torch.cat(native_scales, 1)).abs().max())
    native_trace_zero_diff = float((returned_zero.cpu() - torch.cat(native_zeros, 1)).abs().max())
    native_sse_search = output_sse(xsearch, original, native_q)
    custom_sse_search = output_sse(xsearch, original, custom_q)
    native_sse_held = output_sse(xheld, original, native_q)
    custom_sse_held = output_sse(xheld, original, custom_q)
    sse_den_search = xsearch.shape[0] * original.shape[0]
    sse_den_held = xheld.shape[0] * original.shape[0]
    checks = {
        "seven_group_boundaries": torch.equal(group_idx.cpu(), torch.arange(original.shape[1], dtype=torch.int32) // 128),
        "group_entry_weights": len(input_diffs) == 7 and max(input_diffs) <= 1e-5,
        "scale": torch.allclose(returned_scale.float(), custom.scales, atol=1e-6, rtol=1e-5),
        "zero": torch.equal(returned_zero.float(), custom.zeros),
        "final_fp16_weight": torch.equal(native_q, custom_q),
        "search_output_mse": abs(native_sse_search - custom_sse_search) <= 1e-6 * max(native_sse_search, 1),
        "heldout_output_mse": abs(native_sse_held - custom_sse_held) <= 1e-6 * max(native_sse_held, 1),
        "no_dead_columns": dead_count == 0,
        "source_inverse_helper": torch.allclose(h_native_damped, h_helper, atol=1e-5, rtol=1e-5) and torch.allclose(hinv_native, hinv_helper, atol=1e-4, rtol=1e-5),
        "historical_v7_unchanged": before == historical_fingerprint(),
    }
    report = {
        "status": "PASS" if all(checks.values()) else "STOP_PARITY_MISMATCH",
        "checks": checks,
        "configuration": {"model_revision": V7["model_revision"], "layer": 0,
                          "projection": "self_attn.q_proj", "bits": 4, "sym": True,
                          "group_size": 128, "block_size": 128, "damp_percent": 0.1,
                          "desc_act": False, "static_groups": False,
                          "true_sequential": True, "alpha": 1.0,
                          "gptqmodel_version": importlib.metadata.version("gptqmodel")},
        "tensor_shapes": {"weight": list(original.shape), "search_x": list(xsearch.shape),
                          "heldout_x": list(xheld.shape), "h": list(h_native.shape)},
        "differences": {"group_entry_max_abs_by_group": input_diffs,
                        "scale_max_abs": scale_diff, "zero_max_abs": zero_diff,
                        "final_weight_max_abs": float(weight_abs.max()),
                        "final_weight_mean_abs": float(weight_abs.mean()),
                        "final_weight_mismatched_elements": int((weight_abs != 0).sum()),
                        "native_trace_scale_max_abs": native_trace_scale_diff,
                        "native_trace_zero_max_abs": native_trace_zero_diff,
                        "h_native_damped_vs_helper_max_abs": float((h_native_damped - h_helper).abs().max()),
                        "hinv_native_vs_helper_max_abs": float((hinv_native - hinv_helper).abs().max()),
                        "hinv_native_vs_v7_max_abs": float((hinv_native - h_v7).abs().max())},
        "metrics": {"native_search_sse": native_sse_search, "custom_search_sse": custom_sse_search,
                    "native_heldout_sse": native_sse_held, "custom_heldout_sse": custom_sse_held,
                    "native_search_output_mse": native_sse_search / sse_den_search,
                    "custom_search_output_mse": custom_sse_search / sse_den_search,
                    "native_heldout_output_mse": native_sse_held / sse_den_held,
                    "custom_heldout_output_mse": custom_sse_held / sse_den_held,
                    "native_weight_mse": weight_mse(original, native_q),
                    "custom_weight_mse": weight_mse(original, custom_q)},
        "timing_seconds": {"capture": time.perf_counter() - started,
                           "native_quantize": native_seconds, "custom_quantize": custom_seconds},
        "files": {"group_entry_weights": "parity_group_entries.pt",
                  "final_weights": "parity_weights.pt"},
    }
    torch.save({"native": native_entries, "custom": custom.group_inputs,
                "native_scales": native_scales, "custom_scales": custom.scales.cpu(),
                "native_zeros": native_zeros, "custom_zeros": custom.zeros.cpu()},
               OUTPUT / "parity_group_entries.pt")
    torch.save({"original": original.cpu(), "native": native_q.cpu(),
                "custom": custom_q.cpu(), "h_native": h_native.cpu(),
                "hinv_native": hinv_native.cpu()}, OUTPUT / "parity_weights.pt")
    manifest = {"source_protocol": V7, "documents": [{k: v for k, v in d.items() if k != "ids"} for d in docs],
                "historical_v7_before": before, "historical_v7_after": historical_fingerprint(),
                "input_sha256": {"search": tensor_hash(xsearch), "heldout": tensor_hash(xheld)},
                "weight_sha256": tensor_hash(original), "capture_peak_cuda": capture_peak}
    save_json(OUTPUT / "manifest.json", manifest)
    save_json(OUTPUT / "dynamic_parity_report.json", report)
    (OUTPUT / "dynamic_parity_report.md").write_text(
        "# Dynamic G128 parity\n\n"
        f"Status: **{report['status']}**. Native GPTQModel 2.2.0 vs custom dynamic alpha=1.\n\n"
        f"Max FP16 weight difference: {report['differences']['final_weight_max_abs']:.9g}; "
        f"mean difference: {report['differences']['final_weight_mean_abs']:.9g}; "
        f"mismatched elements: {report['differences']['final_weight_mismatched_elements']}.\n\n"
        f"Max group-entry difference: {max(input_diffs):.9g}; "
        f"scale difference: {scale_diff:.9g}; zero difference: {zero_diff:.9g}.\n\n"
        f"Native/custom search MSE: {report['metrics']['native_search_output_mse']:.9g} / "
        f"{report['metrics']['custom_search_output_mse']:.9g}.\n\n"
        f"Native/custom held-out MSE: {report['metrics']['native_heldout_output_mse']:.9g} / "
        f"{report['metrics']['custom_heldout_output_mse']:.9g}.\n\n"
        "The native call uses GPTQModel's real `GPTQ.quantize` with a read-only "
        "find_params trace. `parity_group_entries.pt` saves each compensated "
        "group-entry weight and scale/zero. `parity_weights.pt` saves the final "
        "weights and H/Hinv. Search continues only if all checks pass.\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": report["status"], "checks": checks,
                      "differences": report["differences"]}, indent=2))
    if report["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
