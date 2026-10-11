"""Shared PPL tokenization and token-weighted sliding-window evaluation."""

from __future__ import annotations

import math
from typing import Any

import torch
from transformers import AutoTokenizer

from benchmark.eval.ppl_datasets import load_ppl_text


def prepare_ppl_input_ids(
    model_id: str,
    ppl_config: dict[str, Any],
    cache_dir: str,
) -> torch.Tensor:
    """Build one deterministic contiguous token stream from a dataset adapter."""
    text = load_ppl_text(ppl_config, cache_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    return tokenizer(
        text,
        return_tensors="pt",
        add_special_tokens=False,
    )["input_ids"]


def select_scope(input_ids: torch.Tensor, ppl_config: dict[str, Any]) -> torch.Tensor:
    """Select the full stream or its fixed smoke prefix."""
    if input_ids.ndim != 2 or input_ids.size(0) != 1 or input_ids.size(1) < 2:
        raise ValueError(f"Expected input_ids with shape [1, N>=2], got {input_ids.shape}")
    scope = ppl_config["scope"]
    if scope == "full":
        return input_ids
    if scope == "smoke":
        smoke_tokens = int(ppl_config["smoke_tokens"])
        if input_ids.size(1) < smoke_tokens:
            raise ValueError("The corpus is shorter than the requested smoke prefix.")
        return input_ids[:, :smoke_tokens]
    raise ValueError(f"Unsupported PPL scope: {scope}")


@torch.inference_mode()
def evaluate_ppl(
    model,
    input_ids: torch.Tensor,
    *,
    max_length: int,
    stride: int,
    progress_every: int = 25,
) -> tuple[float, float, int]:
    """Compute token-weighted causal-LM perplexity without overlap bias."""
    if input_ids.ndim != 2 or input_ids.size(0) != 1:
        raise ValueError("input_ids must have shape [1, N].")
    if not 0 < stride <= max_length:
        raise ValueError("PPL requires 0 < stride <= max_length.")
    if input_ids.size(1) < 2:
        raise ValueError("At least two input tokens are required.")

    device = next(model.parameters()).device
    sequence_length = input_ids.size(1)
    total_nll = 0.0
    total_tokens = 0
    previous_end = 0

    for window_index, begin in enumerate(range(0, sequence_length, stride)):
        end = min(begin + max_length, sequence_length)
        target_length = end - previous_end
        window = input_ids[:, begin:end].to(device)
        labels = window.clone()
        labels[:, :-target_length] = -100
        outputs = model(input_ids=window, labels=labels, use_cache=False)
        valid_tokens = int((labels[:, 1:] != -100).sum().item())
        total_nll += float(outputs.loss.float().item()) * valid_tokens
        total_tokens += valid_tokens

        if progress_every > 0 and (
            window_index % progress_every == 0 or end == sequence_length
        ):
            print(
                f"window={window_index:03d} range=[{begin}:{end}] "
                f"new={target_length} valid={valid_tokens} "
                f"loss={outputs.loss.float().item():.6f}",
                flush=True,
            )
        previous_end = end
        if end == sequence_length:
            break

    if total_tokens == 0:
        raise RuntimeError("PPL evaluation produced zero target tokens.")
    average_nll = total_nll / total_tokens
    return math.exp(average_nll), average_nll, total_tokens
