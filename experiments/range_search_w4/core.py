"""Fixed-grid W4 search and row-separable GPTQ diagnostic.

The affine grid matches GPTQModel 2.2.0's ordinary Quantizer: round-to-even,
q in [0, 15], and symmetric zero=8. This module never mutates input weights.
"""

from __future__ import annotations

from dataclasses import dataclass
import itertools
import math

import torch


QMAX = 15
GRID = tuple(round(i / 10, 1) for i in range(1, 11))


@dataclass(frozen=True)
class Grid:
    scale: float
    zero: float
    alpha: float
    beta: float
    low: float
    high: float


def make_grid(w: torch.Tensor, alpha: float = 1, beta: float = 1, *, sym: bool = True) -> Grid:
    if not (0 < alpha <= 1 and 0 < beta <= 1):
        raise ValueError("alpha and beta must be in (0, 1]")
    w = w.float()
    low = min(float(w.min()), 0.0) * alpha
    high = max(float(w.max()), 0.0) * beta
    if sym:
        if alpha != beta:
            raise ValueError("symmetric range has only one shrink parameter")
        radius = max(abs(low), abs(high))
        if radius == 0:
            radius = 1.0  # GPTQModel's all-zero fallback [-1, 1]
            low = -radius
        elif low < 0:
            low = -radius
        # GPTQModel leaves xmin=0 for an all-positive row, despite sym=True.
        high = radius
        zero = 8.0
    else:
        if low == high:
            low, high = -1.0, 1.0
        zero = float(torch.round(torch.tensor(-low / ((high - low) / QMAX))))
    return Grid((high - low) / QMAX, zero, alpha, beta, low, high)


@torch.no_grad()
def quantize(w: torch.Tensor, grid: Grid) -> torch.Tensor:
    return (torch.round(w.float() / grid.scale) + grid.zero).clamp(0, QMAX).sub(grid.zero).mul(grid.scale)


def candidate_grids(w: torch.Tensor, *, sym: bool, dense_single: bool = False) -> tuple[list[Grid], int]:
    values = tuple(round(i / 100, 2) for i in range(1, 101)) if dense_single else GRID
    pairs = ((a, a) for a in values) if sym or dense_single else itertools.product(GRID, repeat=2)
    unique: dict[tuple[float, float], Grid] = {}
    count = 0
    for a, b in pairs:
        count += 1
        g = make_grid(w, a, b, sym=sym)
        unique.setdefault((g.scale, g.zero), g)
    return list(unique.values()), count


def output_mse(x: torch.Tensor, w: torch.Tensor, q: torch.Tensor) -> float:
    # A single row: denominator is number of output elements.
    return float((x.float() @ (q.float() - w.float())).square().mean())


def weight_mse(w: torch.Tensor, q: torch.Tensor) -> float:
    return float((q.float() - w.float()).square().mean())


@torch.no_grad()
def prepare_hinv(x: torch.Tensor, *, document_count: int, damp_percent: float = 0.1) -> torch.Tensor:
    """Same H normalization/damping and upper inverse factor as GPTQModel.

    GPTQModel counts input batches (documents), not flattened tokens, in nsamples.
    """
    if document_count < 1 or not (0 < damp_percent < 1):
        raise ValueError("invalid document count or damping")
    x = x.float()
    h = (2.0 / document_count) * (x.T @ x)
    dead = torch.diag(h) == 0
    if dead.any():
        raise ValueError("dead input columns require GPTQModel's weight-zeroing path; this local diagnostic does not support them")
    h.diagonal().add_(damp_percent * h.diagonal().mean())
    return torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(h)), upper=True)


@torch.no_grad()
def gptq_row(w: torch.Tensor, grids: list[Grid], hinv: torch.Tensor, *, group_size: int = -1, block_size: int = 128) -> torch.Tensor:
    """GPTQModel's column-order error compensation with externally fixed grids.

    At desc_act=False each output row is independent once Hinv and grid are fixed.
    Groups only select grid parameters; their errors still propagate to later groups.
    """
    original = w.float()
    cols = original.numel()
    gsize = cols if group_size == -1 else group_size
    if len(grids) != math.ceil(cols / gsize):
        raise ValueError("one grid per input group required")
    work = original.clone()
    result = torch.empty_like(work)
    for i1 in range(0, cols, block_size):
        i2 = min(i1 + block_size, cols)
        local = work[i1:i2].clone()
        errors = torch.zeros_like(local)
        local_inv = hinv[i1:i2, i1:i2]
        for j in range(i2 - i1):
            i = i1 + j
            d = local_inv[j, j]
            q = quantize(local[j], grids[i // gsize])
            result[i] = q
            error = (local[j] - q) / d
            local[j:] -= error * local_inv[j, j:]
            errors[j] = error
        if i2 < cols:
            work[i2:] -= errors @ hinv[i1:i2, i2:]
    return result


@torch.no_grad()
def gptq_candidates(w: torch.Tensor, candidates: list[list[Grid]], hinv: torch.Tensor, *, group_size: int = -1, block_size: int = 128, chunk_size: int = 16) -> torch.Tensor:
    """Batch independent candidates; each starts from an identical weight copy."""
    if not candidates:
        raise ValueError("no candidates")
    cols = w.numel()
    gsize = cols if group_size == -1 else group_size
    groups = math.ceil(cols / gsize)
    if any(len(item) != groups for item in candidates):
        raise ValueError("candidate/group shape mismatch")
    outputs = []
    for start in range(0, len(candidates), chunk_size):
        part = candidates[start:start + chunk_size]
        scales = torch.tensor([[g.scale for g in gs] for gs in part], device=w.device, dtype=torch.float32)
        zeros = torch.tensor([[g.zero for g in gs] for gs in part], device=w.device, dtype=torch.float32)
        work = w.float().unsqueeze(0).expand(len(part), -1).clone()
        result = torch.empty_like(work)
        for i1 in range(0, cols, block_size):
            i2 = min(i1 + block_size, cols)
            local = work[:, i1:i2].clone()
            errors = torch.zeros_like(local)
            local_inv = hinv[i1:i2, i1:i2]
            for j in range(i2 - i1):
                i = i1 + j
                scale = scales[:, i // gsize]
                zero = zeros[:, i // gsize]
                q = (torch.round(local[:, j] / scale) + zero).clamp(0, QMAX).sub(zero).mul(scale)
                result[:, i] = q
                error = (local[:, j] - q) / local_inv[j, j]
                local[:, j:] -= error[:, None] * local_inv[j, j:][None, :]
                errors[:, j] = error
            if i2 < cols:
                work[:, i2:] -= errors @ hinv[i1:i2, i2:]
        outputs.append(result)
    return torch.cat(outputs)


def group_errors(x: torch.Tensor, w: torch.Tensor, q: torch.Tensor, group_size: int) -> dict[str, float]:
    components = [x[:, i:i + group_size].float() @ (q[i:i + group_size].float() - w[i:i + group_size].float()) for i in range(0, w.numel(), group_size)]
    independent = sum(float(e.square().sum()) for e in components)
    full = float(torch.stack(components).sum(0).square().sum())
    return {"independent_sum_squares": independent, "full_sum_squares": full, "cross_term": full - independent, "full_mse": full / x.shape[0]}
