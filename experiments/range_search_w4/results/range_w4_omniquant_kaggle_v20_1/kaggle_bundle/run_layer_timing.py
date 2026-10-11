#!/usr/bin/env python3
"""Run a bounded 2048-token update sequence through the frozen v20 OmniQuant loop."""
from __future__ import annotations

import csv
import gc
import shutil
import tempfile
import time
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
import torch.nn as nn


TARGETS = (
    "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
    "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
)


def move_tree(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, tuple):
        return tuple(move_tree(item, device) for item in value)
    if isinstance(value, list):
        return [move_tree(item, device) for item in value]
    if isinstance(value, dict):
        return {key: move_tree(item, device) for key, item in value.items()}
    return value


class ReplayEmbedding(nn.Module):
    """Feed a real frozen sample's saved FP prefix activation to the v20 catcher."""

    def __init__(self, activations: list[torch.Tensor]):
        super().__init__()
        stacked = torch.cat([item.detach().cpu() for item in activations], dim=0)
        self.register_buffer("activations", stacked, persistent=False)
        self.cursor = 0
        self.last_index = -1

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        index = self.cursor
        if index >= self.activations.shape[0]:
            raise RuntimeError("ReplayEmbedding received more samples than the frozen input list")
        self.cursor += 1
        self.last_index = index
        return self.activations[index:index + 1].to(device=input_ids.device)


class ReplayBody(nn.Module):
    def __init__(self, original_layer: nn.Module, activations: list[torch.Tensor]):
        super().__init__()
        self.layers = nn.ModuleList([original_layer])
        self.embed_tokens = ReplayEmbedding(activations)
        self.norm = nn.Identity()
        self.rotary_emb = nn.Identity()


class ReplayQwenLM(nn.Module):
    """Minimal façade for the v20 official loop; architecture remains the native Qwen block."""

    def __init__(self, config, original_layer: nn.Module,
                 activations: list[torch.Tensor], layer_kwargs: list[dict[str, Any]]):
        super().__init__()
        self.config = config
        self.model = ReplayBody(original_layer, activations)
        self._layer_kwargs = layer_kwargs

    def forward(self, input_ids: torch.Tensor):
        hidden = self.model.embed_tokens(input_ids)
        index = self.model.embed_tokens.last_index
        kwargs = move_tree(self._layer_kwargs[index], input_ids.device)
        return self.model.layers[0](hidden, **kwargs)


class NullLogger:
    def info(self, *args, **kwargs):
        pass


def _sync(device: torch.device) -> None:
    torch.cuda.synchronize(device)


def _install_timing_observers(omni_module, step_count: int, device: torch.device):
    """Wrap the v20 methods to time final quantization, transition, scale registration and save."""
    control: dict[str, Any] = {
        "transition_remaining": 0,
        "transition_step_seconds": [],
        "final_quantization_seconds": [],
        "scale_registration_seconds": [],
        "checkpoint_io_seconds": [],
    }
    base_adapter = omni_module.QuantQwen2DecoderLayer

    class TimingAdapter(base_adapter):
        def forward(self, *args, **kwargs):
            if control["transition_remaining"] <= 0:
                return super().forward(*args, **kwargs)
            start = torch.cuda.Event(enable_timing=True)
            stop = torch.cuda.Event(enable_timing=True)
            _sync(device)
            start.record()
            out = super().forward(*args, **kwargs)
            stop.record()
            _sync(device)
            control["transition_step_seconds"].append(start.elapsed_time(stop) / 1000.0)
            control["transition_remaining"] -= 1
            return out

    omni_module.QuantQwen2DecoderLayer = TimingAdapter
    original_quantize = omni_module.smooth_and_quant_inplace
    original_register = omni_module.register_scales_and_zeros
    original_save = torch.save

    def timed_quantize(*args, **kwargs):
        _sync(device)
        started = time.perf_counter()
        value = original_quantize(*args, **kwargs)
        _sync(device)
        control["final_quantization_seconds"].append(time.perf_counter() - started)
        control["transition_remaining"] = step_count
        return value

    def timed_register(*args, **kwargs):
        _sync(device)
        started = time.perf_counter()
        value = original_register(*args, **kwargs)
        _sync(device)
        control["scale_registration_seconds"].append(time.perf_counter() - started)
        return value

    def timed_save(*args, **kwargs):
        _sync(device)
        started = time.perf_counter()
        value = original_save(*args, **kwargs)
        _sync(device)
        control["checkpoint_io_seconds"].append(time.perf_counter() - started)
        return value

    omni_module.smooth_and_quant_inplace = timed_quantize
    omni_module.register_scales_and_zeros = timed_register
    torch.save = timed_save

    def restore():
        omni_module.QuantQwen2DecoderLayer = base_adapter
        omni_module.smooth_and_quant_inplace = original_quantize
        omni_module.register_scales_and_zeros = original_register
        torch.save = original_save

    return control, restore


def _make_args(config: dict[str, Any], nsamples: int, step_limit: int, output_dir: Path):
    qcfg = config["quantization"]
    return SimpleNamespace(
        net="qwen2", model=config["model_id"], nsamples=nsamples,
        seqlen=config["formal_training"]["tokens_per_sample"], batch_size=1,
        deactive_amp=False, epochs=config["formal_training"]["epochs"],
        smoke_step_limit=step_limit, smoke_layer_limit=1, aug_loss=False,
        let=False, lwc=True, wbits=qcfg["weight_bits"], abits=qcfg["activation_bits"],
        group_size=qcfg["group_size"], symmetric=qcfg["symmetric"],
        disable_zero_point=False, w_dynamic_method="per_channel", a_dynamic_method="per_token",
        alpha=0.5, let_lr=config["formal_training"]["let_learning_rate_recorded_but_inactive"],
        lwc_lr=config["formal_training"]["lwc_learning_rate"],
        wd=config["formal_training"]["weight_decay"], resume=None,
        output_dir=str(output_dir), real_quant=False,
        weight_quant_params={
            "n_bits": qcfg["weight_bits"], "per_channel_axes": [0], "symmetric": qcfg["symmetric"],
            "dynamic_method": "per_channel", "group_size": qcfg["group_size"],
            "lwc": qcfg["lwc"], "disable_zero_point": False,
        },
        act_quant_params={"n_bits": qcfg["activation_bits"], "per_channel_axes": [],
                          "symmetric": False, "dynamic_method": "per_token"},
    )


def run_layer_sequence(*, omni_module, config: dict[str, Any], full_model,
                       layer_index: int, activations: list[torch.Tensor],
                       layer_kwargs: list[dict[str, Any]], token_ids: list[list[int]],
                       warmup_steps: int, measured_steps: int, output_csv: Path,
                       sequence_json: Path | None = None) -> dict[str, Any]:
    """Run one continuous optimizer sequence on a fresh v20 target-block instance."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; refusing a local CPU substitute")
    device = torch.device("cuda:0")
    step_count = warmup_steps + measured_steps
    if len(activations) < step_count or len(layer_kwargs) < step_count or len(token_ids) < step_count:
        raise ValueError("not enough captured frozen samples for the requested optimizer updates")
    if any(item.shape[1] != 2048 for item in activations[:step_count]):
        raise ValueError("preflight activations must have exactly 2048 tokens")

    # The source model stays resident, as in the v20 smoke path. The selected
    # native layer is deep-copied by the existing v20 adapter before optimization.
    original_layer = deepcopy(full_model.model.layers[layer_index])
    replay = ReplayQwenLM(full_model.config, original_layer,
                          activations[:step_count], layer_kwargs[:step_count])
    replay.eval()
    input_rows = [(torch.tensor(ids, dtype=torch.long).view(1, -1), i)
                  for i, ids in enumerate(token_ids[:step_count])]
    callback_rows: list[dict[str, Any]] = []
    teacher_seconds: list[float] = []
    args = _make_args(config, step_count, step_count, output_csv.parent / f"state_l{layer_index}_{step_count}")
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    def step_callback(row):
        row = dict(row)
        row["allocated_after_step_bytes"] = int(torch.cuda.memory_allocated(device))
        row["reserved_after_step_bytes"] = int(torch.cuda.memory_reserved(device))
        callback_rows.append(row)

    args.v20_step_callback = step_callback
    args.v20_timing_callback = lambda stage, seconds: teacher_seconds.append(float(seconds)) \
        if stage == "teacher_forward" else None
    lm = SimpleNamespace(model=replay, device=device, _device=device,
                         seqlen=config["formal_training"]["tokens_per_sample"])
    baseline_alloc = int(torch.cuda.memory_allocated(device))
    baseline_reserved = int(torch.cuda.memory_reserved(device))
    free_before, total_vram = torch.cuda.mem_get_info(device)
    torch.cuda.reset_peak_memory_stats(device)
    control, restore = _install_timing_observers(omni_module, step_count, device)
    _sync(device)
    call_started = time.perf_counter()
    try:
        omni_module.omniquant(lm, args, input_rows, None, None, NullLogger())
        _sync(device)
    finally:
        omniquant_wall = time.perf_counter() - call_started
        restore()

    peak_alloc = int(torch.cuda.max_memory_allocated(device))
    peak_reserved = int(torch.cuda.max_memory_reserved(device))
    free_after, _ = torch.cuda.mem_get_info(device)
    if len(callback_rows) != step_count:
        raise RuntimeError(f"expected {step_count} optimizer updates, observed {len(callback_rows)}")
    if len(control["transition_step_seconds"]) != step_count:
        raise RuntimeError("did not observe exactly one post-quantization transition forward per sample")

    teacher_per_sample = (teacher_seconds[-1] / step_count) if teacher_seconds else float("nan")
    rows = []
    for index, item in enumerate(callback_rows):
        rows.append({
            "status": "MEASURED",
            "layer": layer_index,
            "repeat": index,
            "warmup_excluded": index < warmup_steps,
            "teacher_s": teacher_per_sample,
            "student_s": float(item["student_quant_forward_seconds"]),
            "loss_s": float(item["loss_seconds"]),
            "backward_s": float(item["backward_seconds_excluding_optimizer"]),
            "optimizer_s": float(item["optimizer_step_seconds"]),
            "total_s": float(item["step_total_seconds"]),
            "peak_allocated_gb": peak_alloc / (1024 ** 3),
            "peak_reserved_gb": peak_reserved / (1024 ** 3),
            "allocated_after_step_gb": item["allocated_after_step_bytes"] / (1024 ** 3),
            "reserved_after_step_gb": item["reserved_after_step_bytes"] / (1024 ** 3),
            "loss": float(item["loss_before_update"]),
            "lwc_parameter_count": int(item["lwc_parameter_count"]),
            "lwc_gradient_finite": bool(item["gradients_finite"]),
            "lwc_nonzero_grad_tensor_count": int(item["nonzero_grad_tensor_count"]),
            "lwc_update_l1": float(item["lwc_update_l1"]),
            "lwc_changed": bool(item["lwc_changed"]),
        })
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with output_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    train_rows = rows[warmup_steps:]
    def sum_or_nan(values):
        return float(sum(values)) if values else 0.0
    transition_s = sum(control["transition_step_seconds"])
    final_quant_s = sum(control["final_quantization_seconds"])
    register_s = sum(control["scale_registration_seconds"])
    checkpoint_s = sum(control["checkpoint_io_seconds"])
    step_s = sum(float(item["total_s"]) for item in rows)
    teacher_total = sum(teacher_seconds)
    other_s = max(0.0, omniquant_wall - step_s - teacher_total - transition_s - final_quant_s - register_s - checkpoint_s)
    result = {
        "layer": layer_index,
        "sequence_length": 2048,
        "warmup_steps": warmup_steps,
        "measured_steps": measured_steps,
        "updates_observed": len(rows),
        "all_losses_finite": all(torch.isfinite(torch.tensor(r["loss"])).item() for r in rows),
        "all_lwc_gradients_finite": all(r["lwc_gradient_finite"] for r in rows),
        "all_steps_have_nonzero_lwc_gradient": all(r["lwc_nonzero_grad_tensor_count"] > 0 for r in rows),
        "any_lwc_parameter_changed": any(r["lwc_changed"] and r["lwc_update_l1"] > 0 for r in rows),
        "lwc_parameters_present": all(r["lwc_parameter_count"] > 0 for r in rows),
        "training_total_seconds_including_warmup": step_s,
        "teacher_forward_total_seconds": teacher_total,
        "teacher_forward_per_sample_seconds": teacher_per_sample,
        "layer_transition_total_seconds": transition_s,
        "layer_transition_per_sample_seconds": transition_s / step_count,
        "final_quantization_seconds": final_quant_s,
        "scale_registration_seconds": register_s,
        "checkpoint_io_seconds": checkpoint_s,
        "checkpoint_bytes": sum(p.stat().st_size for p in Path(args.output_dir).glob("*.pth")),
        "omniquant_call_wall_seconds": omniquant_wall,
        "other_block_overhead_seconds": other_s,
        "memory": {
            "baseline_allocated_bytes": baseline_alloc,
            "baseline_reserved_bytes": baseline_reserved,
            "peak_allocated_bytes": peak_alloc,
            "peak_reserved_bytes": peak_reserved,
            "free_vram_before_bytes": int(free_before),
            "free_vram_after_bytes": int(free_after),
            "total_vram_bytes": int(total_vram),
        },
        "rows": rows,
        "measured_summary": {
            key: {
                "mean": sum(float(row[key]) for row in train_rows) / len(train_rows),
                "median": float(torch.tensor([float(row[key]) for row in train_rows]).median().item()),
                "p95": float(torch.quantile(torch.tensor([float(row[key]) for row in train_rows]), 0.95).item()),
            }
            for key in ("student_s", "loss_s", "backward_s", "optimizer_s", "total_s")
        },
        "semantic_note": "Calls the v20 omniquant() loop and v20 QuantQwen2DecoderLayer. Smoke step limit is the only loop limit; no optimizer/loss/quantizer changes. Prefix activations are native FP16 full-model activations from frozen real tokens.",
    }
    if sequence_json is not None:
        sequence_json.parent.mkdir(parents=True, exist_ok=True)
        sequence_json.write_text(__import__("json").dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    state_dir = Path(args.output_dir).resolve()
    if state_dir.parent == output_csv.parent.resolve() and state_dir.name.startswith("state_l") and state_dir.is_dir():
        shutil.rmtree(state_dir)
    gc.collect()
    torch.cuda.empty_cache()
    return result
