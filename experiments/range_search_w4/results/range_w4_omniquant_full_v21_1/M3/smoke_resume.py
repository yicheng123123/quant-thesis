from __future__ import annotations

import importlib
import json
import os
import shutil
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import torch

import run_formal_m3 as runner


OUT = runner.HERE / "RESUME_SMOKE"
SMOKE_SAMPLES = 2
SMOKE_TOKENS = 64
SMOKE_LAYERS = 2
SMOKE_EPOCHS = 1


def make_model():
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        str(runner.FROZEN_MODEL), torch_dtype=torch.float16, low_cpu_mem_usage=True,
        attn_implementation="sdpa", local_files_only=True,
    ).eval().to(runner.DEVICE)
    model.model.layers = torch.nn.ModuleList(list(model.model.layers[:SMOKE_LAYERS]))
    return model


def make_args(out: Path, resume: Path | None = None):
    config = runner.load_json(runner.BUNDLE / "configs" / "m3_preflight.json")
    args = runner.make_args(config, resume, 1 if resume else 0, out)
    args.nsamples = SMOKE_SAMPLES
    args.seqlen = SMOKE_TOKENS
    args.epochs = SMOKE_EPOCHS
    args.smoke_layer_limit = SMOKE_LAYERS
    args.smoke_step_limit = None
    args.v21_resume_completed_layers = 1 if resume else 0
    return args


def canonical_layer_state(model):
    from quantize.int_linear import QuantLinear
    rows = {}
    for layer_index, wrapper in enumerate(model.model.layers):
        native = getattr(wrapper, "native_layer", wrapper)
        for name, module in native.named_modules():
            if isinstance(module, QuantLinear):
                rows[f"{layer_index}.{name}.weight"] = module.weight.detach().cpu().clone()
                rows[f"{layer_index}.{name}.scales"] = module.weight_quantizer.scales.detach().cpu().clone()
                rows[f"{layer_index}.{name}.zeros"] = module.weight_quantizer.zeros.detach().cpu().clone()
    return rows


def logits(model, ids):
    model.to(runner.DEVICE).eval()
    with torch.inference_mode():
        out = model(torch.tensor(ids, dtype=torch.long, device=runner.DEVICE).view(1, -1), use_cache=False).logits.float().cpu()
    torch.cuda.synchronize(runner.DEVICE)
    return out


def invoke(model, args, token_ids):
    omni = importlib.import_module("quantize.omniquant")
    dataloader = [(torch.tensor(ids, dtype=torch.long).view(1, -1), index) for index, ids in enumerate(token_ids)]
    lm = SimpleNamespace(model=model, device=runner.DEVICE, _device=runner.DEVICE, seqlen=SMOKE_TOKENS)
    return omni.omniquant(lm, args, dataloader, None, None, runner.NullLogger())


def main():
    if OUT.exists():
        raise RuntimeError(f"resume smoke output already exists: {OUT}")
    OUT.mkdir(parents=True)
    runner.write_runtime_environment()
    config, calibration, token_ids_all, _ = runner.frozen_context()
    token_ids = [ids[:SMOKE_TOKENS] for ids in token_ids_all[:SMOKE_SAMPLES]]
    probe = token_ids[0]
    sys.path.insert(0, str(runner.HERE / "official_port"))
    if importlib.util.find_spec("termcolor") is None:
        termcolor = types.ModuleType("termcolor")
        termcolor.colored = lambda text, *args, **kwargs: text
        sys.modules["termcolor"] = termcolor

    # Reference: uninterrupted two-block run from the exact frozen base weights.
    runner.write_runtime_environment()
    baseline_model = make_model()
    baseline_args = make_args(OUT / "baseline")
    Path(baseline_args.output_dir).mkdir(parents=True)
    baseline_states = {}
    save_fn = torch.save

    def collect_baseline(value, path, *args, **kwargs):
        if Path(os.fspath(path)).name == "omni_parameters.pth":
            baseline_states.update(runner.to_cpu_state(value))
        return save_fn(value, path, *args, **kwargs)

    torch.save = collect_baseline
    try:
        baseline_model = invoke(baseline_model, baseline_args, token_ids)
    finally:
        torch.save = save_fn
    baseline_parameters = {key: {name: tensor.detach().cpu().clone() for name, tensor in value.items()}
                           for key, value in baseline_states.items()}
    baseline_quant = canonical_layer_state(baseline_model)
    baseline_logits = logits(baseline_model, probe)
    del baseline_model
    torch.cuda.empty_cache()

    # Interrupted run: commit block 0 and stop immediately at its official save boundary.
    runner.write_runtime_environment()
    interrupted_model = make_model()
    interrupted_args = make_args(OUT / "interrupted")
    Path(interrupted_args.output_dir).mkdir(parents=True)
    partial_dir = OUT / "checkpoint"
    partial_dir.mkdir()
    partial_state_path = partial_dir / "omni_parameters.pth"
    partial_rng_path = partial_dir / "rng_state.pth"
    stopped = {"value": False}

    class StopAfterFirstBlock(RuntimeError):
        pass

    def interrupt_after_first(value, path, *args, **kwargs):
        if Path(os.fspath(path)).name == "omni_parameters.pth" and not stopped["value"]:
            save_fn(runner.to_cpu_state(value), partial_state_path)
            save_fn(runner.current_rng_state(), partial_rng_path)
            stopped["value"] = True
            raise StopAfterFirstBlock("intentional smoke interruption after completed block 0")
        return save_fn(value, path, *args, **kwargs)

    torch.save = interrupt_after_first
    try:
        try:
            invoke(interrupted_model, interrupted_args, token_ids)
        except StopAfterFirstBlock:
            pass
        else:
            raise RuntimeError("resume smoke did not interrupt after completed block 0")
    finally:
        torch.save = save_fn
    del interrupted_model
    torch.cuda.empty_cache()
    partial_state = torch.load(partial_state_path, map_location="cpu", weights_only=False)
    rng_state = torch.load(partial_rng_path, map_location="cpu", weights_only=False)

    # Resume: replay block 0, restore RNG at the first incomplete block, and train block 1.
    runner.write_runtime_environment()
    resumed_model = make_model()
    resumed_args = make_args(OUT / "resumed", partial_state_path)
    Path(resumed_args.output_dir).mkdir(parents=True)
    resumed_args.v21_restore_rng = lambda: runner.restore_rng_state(rng_state)
    resumed_states = {}

    def collect_resumed(value, path, *args, **kwargs):
        if Path(os.fspath(path)).name == "omni_parameters.pth":
            resumed_states.update(runner.to_cpu_state(value))
        return save_fn(value, path, *args, **kwargs)

    torch.save = collect_resumed
    try:
        resumed_model = invoke(resumed_model, resumed_args, token_ids)
    finally:
        torch.save = save_fn
    resumed_parameters = {key: {name: tensor.detach().cpu().clone() for name, tensor in value.items()}
                          for key, value in resumed_states.items()}
    resumed_quant = canonical_layer_state(resumed_model)
    resumed_logits = logits(resumed_model, probe)

    state_diffs = []
    for layer_index in range(SMOKE_LAYERS):
        left, right = baseline_parameters[layer_index], resumed_parameters[layer_index]
        if set(left) != set(right):
            raise RuntimeError(f"L{layer_index} saved LWC state keys differ across resume")
        state_diffs.extend((left[name] == right[name]).all().item() for name in left)
    quant_diffs = []
    if set(baseline_quant) != set(resumed_quant):
        raise RuntimeError("quantized weight/scale/zero inventory differs across resume")
    for name in baseline_quant:
        quant_diffs.append(bool(torch.equal(baseline_quant[name], resumed_quant[name])))
    logit_diff = (baseline_logits - resumed_logits).abs()
    result = {
        "status": "PASS" if all(state_diffs) and all(quant_diffs) and torch.allclose(baseline_logits, resumed_logits, rtol=1e-3, atol=1e-3) else "FAIL",
        "smoke_only": True, "formal_quality_evidence": False,
        "layers": SMOKE_LAYERS, "samples": SMOKE_SAMPLES, "sequence_length": SMOKE_TOKENS, "epochs": SMOKE_EPOCHS,
        "interruption_boundary": "after block 0 propagation, scales/zeros registration and checkpoint save",
        "resume_completed_layers": 1,
        "lwc_state_tensor_count": len(state_diffs), "lwc_state_tensors_equal": all(state_diffs),
        "quantized_weight_scale_zero_tensor_count": len(quant_diffs), "quantized_tensors_equal": all(quant_diffs),
        "logits_shape": list(baseline_logits.shape), "logits_bitwise_equal": bool(torch.equal(baseline_logits, resumed_logits)),
        "logits_allclose": bool(torch.allclose(baseline_logits, resumed_logits, rtol=1e-3, atol=1e-3)),
        "logits_max_abs_diff": float(logit_diff.max().item()), "logits_mean_abs_diff": float(logit_diff.mean().item()),
        "calibration_token_sample_hashes": [calibration["samples"][i]["token_sha256"] for i in range(SMOKE_SAMPLES)],
    }
    (OUT / "RESUME_SMOKE.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if result["status"] != "PASS":
        raise RuntimeError("block-boundary resume smoke failed")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
