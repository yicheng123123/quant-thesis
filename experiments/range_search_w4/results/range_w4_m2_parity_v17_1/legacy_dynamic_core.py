"""Dynamic symmetric G128 GPTQ with a shared clipping ratio.

The alpha=1 path mirrors GPTQModel 2.2.0's dynamic find_params timing.
All candidate calls start from a clone of the original FP weight.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from collections.abc import Sequence

import torch


@dataclass
class DynamicResult:
    weight_fp32: torch.Tensor
    scales: torch.Tensor
    zeros: torch.Tensor
    group_inputs: list[torch.Tensor]


@torch.no_grad()
def dynamic_params(group_weight: torch.Tensor, alpha: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Replicate ordinary Quantizer.find_params(..., weight=True, sym=True).

    The native full-zero fallback and its all-positive-row behavior are kept.
    Alpha changes the grid width only after dynamic MinMax is estimated.
    """
    if not (0 < alpha <= 1):
        raise ValueError("alpha must be in (0, 1]")
    w = group_weight.float()
    low = torch.minimum(w.min(dim=1).values, torch.zeros(w.shape[0], device=w.device))
    high = torch.maximum(w.max(dim=1).values, torch.zeros(w.shape[0], device=w.device))
    high = torch.maximum(low.abs(), high)
    low = torch.where(low < 0, -high, low)
    empty = (low == 0) & (high == 0)
    low = torch.where(empty, -torch.ones_like(low), low)
    high = torch.where(empty, torch.ones_like(high), high)
    # GPTQModel divides by its int64 scalar maxq tensor. On CUDA, division by
    # a Python float uses a slightly different kernel/rounding; a one-ULP
    # scale change can flip later round-to-even decisions and propagate.
    maxq = torch.tensor(15, device=w.device, dtype=torch.int64)
    scale = (high - low) / maxq
    if alpha != 1.0:
        scale = scale * alpha
    scale = scale.unsqueeze(1)
    zero = torch.full_like(scale, 8.0)
    return scale, zero


@torch.no_grad()
def dynamic_gptq(
    original_weight: torch.Tensor,
    hinv: torch.Tensor,
    *,
    alpha: float | Sequence[float] = 1.0,
    group_size: int = 128,
    block_size: int = 128,
    trace: bool = False,
) -> DynamicResult:
    """Use the same block updates and group-entry W as GPTQModel.

    Native find_params reads global W at the group boundary, while the current
    block uses W1 for column compensation. Keep those states distinct.
    """
    if original_weight.ndim != 2 or hinv.ndim != 2:
        raise ValueError("expected W [rows, columns] and square Hinv")
    rows, cols = original_weight.shape
    if hinv.shape != (cols, cols) or group_size <= 0 or block_size <= 0:
        raise ValueError("incompatible GPTQ geometry")
    ratios = [float(alpha)] * math.ceil(cols / group_size) if isinstance(alpha, (int, float)) else list(alpha)
    if len(ratios) != math.ceil(cols / group_size) or any(not (0 < a <= 1) for a in ratios):
        raise ValueError("one valid alpha per group is required")
    work = original_weight.float().clone()
    qweight = torch.zeros_like(work)
    scales = []
    zeros = []
    entries = []
    current_scale = None
    current_zero = None
    for i1 in range(0, cols, block_size):
        i2 = min(i1 + block_size, cols)
        local = work[:, i1:i2].clone()
        errors = torch.zeros_like(local)
        local_inv = hinv[i1:i2, i1:i2]
        for j in range(i2 - i1):
            col = i1 + j
            if col % group_size == 0:
                entering = work[:, col:min(col + group_size, cols)]
                if trace:
                    entries.append(entering.detach().cpu().clone())
                current_scale, current_zero = dynamic_params(entering, ratios[col // group_size])
                scales.append(current_scale.clone())
                zeros.append(current_zero.clone())
            assert current_scale is not None and current_zero is not None
            current = local[:, j:j + 1]
            q = (torch.round(current / current_scale) + current_zero).clamp(0, 15)
            q = (q - current_zero) * current_scale
            qweight[:, col] = q.flatten()
            error = (current.flatten() - q.flatten()) / local_inv[j, j]
            local[:, j:] -= error[:, None] @ local_inv[j, j:].unsqueeze(0)
            errors[:, j] = error
        if i2 < cols:
            work[:, i2:] -= errors @ hinv[i1:i2, i2:]
    return DynamicResult(qweight, torch.cat(scales, dim=1), torch.cat(zeros, dim=1), entries)


@torch.no_grad()
def dynamic_gptq_batch(
    original_weight: torch.Tensor,
    hinv: torch.Tensor,
    alphas: Sequence[float],
    *,
    group_size: int = 128,
    block_size: int = 128,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Vectorize independent shared-alpha candidates without sharing state."""
    if original_weight.ndim != 2 or not alphas:
        raise ValueError("expected W [rows, columns] and nonempty alphas")
    if any(not (0 < a <= 1) for a in alphas):
        raise ValueError("invalid alpha")
    rows, cols = original_weight.shape
    batch = len(alphas)
    ratios = torch.tensor(alphas, device=original_weight.device, dtype=torch.float32)[:, None, None]
    work = original_weight.float()[None].expand(batch, -1, -1).clone()
    result = torch.zeros_like(work)
    scales = []
    zeros = []
    current_scale = None
    current_zero = None
    for i1 in range(0, cols, block_size):
        i2 = min(i1 + block_size, cols)
        local = work[:, :, i1:i2].clone()
        errors = torch.zeros_like(local)
        local_inv = hinv[i1:i2, i1:i2]
        for j in range(i2 - i1):
            col = i1 + j
            if col % group_size == 0:
                entering = work[:, :, col:min(col + group_size, cols)]
                base_scale, base_zero = dynamic_params(entering.reshape(batch * rows, -1), 1.0)
                current_scale = base_scale.reshape(batch, rows, 1) * ratios
                current_zero = base_zero.reshape(batch, rows, 1)
                scales.append(current_scale.clone())
                zeros.append(current_zero.clone())
            current = local[:, :, j:j + 1]
            q = (torch.round(current / current_scale) + current_zero).clamp(0, 15)
            q = (q - current_zero) * current_scale
            result[:, :, col] = q.squeeze(-1)
            error = (current - q).squeeze(-1) / local_inv[j, j]
            local[:, :, j:] -= error[:, :, None] * local_inv[j, j:][None, None, :]
            errors[:, :, j] = error
        if i2 < cols:
            work[:, :, i2:] -= errors @ hinv[i1:i2, i2:]
    return result, torch.cat(scales, dim=2), torch.cat(zeros, dim=2)


@torch.no_grad()
def fixed_minmax_gptq(original_weight: torch.Tensor, hinv: torch.Tensor,
                      *, group_size: int = 128, block_size: int = 128) -> DynamicResult:
    """Diagnostic A: freeze native MinMax grids from pristine groups."""
    rows, cols = original_weight.shape
    grids = [dynamic_params(original_weight[:, s:min(s + group_size, cols)], 1.0)
             for s in range(0, cols, group_size)]
    work = original_weight.float().clone()
    result = torch.zeros_like(work)
    for i1 in range(0, cols, block_size):
        i2 = min(i1 + block_size, cols)
        local = work[:, i1:i2].clone()
        errors = torch.zeros_like(local)
        local_inv = hinv[i1:i2, i1:i2]
        for j in range(i2 - i1):
            col = i1 + j
            scale, zero = grids[col // group_size]
            current = local[:, j:j + 1]
            q = ((current / scale).round() + zero).clamp(0, 15)
            q = (q - zero) * scale
            result[:, col] = q.flatten()
            error = (current.flatten() - q.flatten()) / local_inv[j, j]
            local[:, j:] -= error[:, None] @ local_inv[j, j:].unsqueeze(0)
            errors[:, j] = error
        if i2 < cols:
            work[:, i2:] -= errors @ hinv[i1:i2, i2:]
    return DynamicResult(result, torch.cat([g[0] for g in grids], 1),
                         torch.cat([g[1] for g in grids], 1), [])


@torch.no_grad()
def native_hinv_from_documents(docs: list[torch.Tensor], module: torch.nn.Linear, damp_percent: float) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Build H and Hinv using GPTQModel's process_batch and inversion code."""
    from gptqmodel.quantization import QuantizeConfig
    from gptqmodel.quantization.gptq import GPTQ

    qcfg = QuantizeConfig(bits=4, group_size=128, sym=True, desc_act=False,
                          static_groups=False, true_sequential=True,
                          damp_percent=damp_percent, mse=0.0)
    reference = GPTQ(module, qcfg=qcfg)
    reference.quantizer.configure(perchannel=True, bits=4, sym=True)
    for doc in docs:
        reference.process_batch(doc[None])
    h = reference.H.clone()
    dead = torch.diag(h) == 0
    dead_count = int(dead.sum())
    if dead_count:
        raise ValueError("dead columns require native weight-zeroing path")
    h.diagonal().add_(damp_percent * h.diagonal().mean())
    chol = torch.linalg.cholesky(h)
    inv = reference.block_cholesky_inverse(chol, block_size=h.shape[0])
    return h, torch.linalg.cholesky(inv, upper=True), dead_count


def output_sse(x: torch.Tensor, original: torch.Tensor, quantized: torch.Tensor) -> float:
    delta = quantized.float() - original.float()
    error = x.float() @ (delta if delta.ndim == 1 else delta.T)
    return float(error.square().sum())


def weight_mse(original: torch.Tensor, quantized: torch.Tensor) -> float:
    return float((quantized.float() - original.float()).square().mean())
