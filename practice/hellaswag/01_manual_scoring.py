"""Score one HellaSwag example by hand for learning and inspection.

This is deliberately separate from the formal benchmark. Formal downstream
results are produced only by lm-evaluation-harness through
``benchmark.runners.eval_downstream``.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("HF_HOME", str(REPOSITORY_ROOT / ".hf_cache"))
os.environ.setdefault("HF_DATASETS_CACHE", str(REPOSITORY_ROOT / ".hf_cache" / "datasets"))

import torch
from datasets import load_dataset

from benchmark.config import PROJECT_ROOT, load_config
from benchmark.eval.model import load_model_and_tokenizer


def preprocess(text: str) -> str:
    """Match lm-eval 0.4.13's HellaSwag preprocessing."""
    text = text.strip()
    text = text.replace(" [title]", ". ")
    text = re.sub(r"\[.*?\]", "", text)
    return text.replace("  ", " ")


def encode_pair(tokenizer, context: str, continuation: str) -> tuple[list[int], list[int]]:
    """Match the causal HFLM context/continuation token boundary."""
    trailing_spaces = len(context) - len(context.rstrip())
    if trailing_spaces:
        continuation = context[-trailing_spaces:] + continuation
        context = context[:-trailing_spaces]
    context_ids = tokenizer.encode(context, add_special_tokens=True)
    whole_ids = tokenizer.encode(context + continuation, add_special_tokens=True)
    return context_ids, whole_ids[len(context_ids) :]


@torch.inference_mode()
def continuation_loglikelihood(model, context_ids: list[int], continuation_ids: list[int]) -> float:
    """Return log P(continuation | context) with causal next-token alignment."""
    if not continuation_ids:
        raise ValueError("Continuation encoded to zero tokens.")
    device = next(model.parameters()).device
    combined = torch.tensor([context_ids + continuation_ids], device=device)
    logits = model(combined).logits[:, :-1, :].float()
    targets = combined[:, 1:]
    first_continuation_target = len(context_ids) - 1
    token_logprobs = torch.log_softmax(logits, dim=-1).gather(
        dim=-1,
        index=targets.unsqueeze(-1),
    ).squeeze(-1)
    return float(token_logprobs[0, first_continuation_target:].sum().item())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "benchmark" / "configs" / "fp16.yaml",
    )
    parser.add_argument("--sample-index", type=int, default=0)
    args = parser.parse_args()

    config = load_config(args.config.resolve())
    dataset = load_dataset(
        "Rowan/hellaswag",
        cache_dir=str(PROJECT_ROOT / ".hf_cache" / "datasets"),
    )["validation"]
    if not 0 <= args.sample_index < len(dataset):
        raise IndexError(f"sample-index must be in [0, {len(dataset) - 1}].")

    raw = dataset[args.sample_index]
    context = preprocess(
        raw["activity_label"] + ": " + raw["ctx_a"] + " " + raw["ctx_b"].capitalize()
    )
    choices = [preprocess(ending) for ending in raw["endings"]]
    gold = int(raw["label"])
    model, tokenizer = load_model_and_tokenizer(config)

    rows: list[dict[str, float | int | str]] = []
    for label, choice in zip("ABCD", choices, strict=True):
        continuation = " " + choice  # lm-eval's default target_delimiter
        context_ids, continuation_ids = encode_pair(tokenizer, context, continuation)
        total_ll = continuation_loglikelihood(model, context_ids, continuation_ids)
        rows.append(
            {
                "label": label,
                "choice": choice,
                "tokens": len(continuation_ids),
                "chars": len(choice),
                "total_ll": total_ll,
                "token_avg_ll": total_ll / len(continuation_ids),
                # lm-eval 0.4.13 defines HellaSwag acc_norm with choice
                # character length, not continuation-token length.
                "harness_norm_ll": total_ll / len(choice),
            }
        )

    raw_prediction = max(range(4), key=lambda index: rows[index]["total_ll"])
    token_norm_prediction = max(range(4), key=lambda index: rows[index]["token_avg_ll"])
    harness_norm_prediction = max(range(4), key=lambda index: rows[index]["harness_norm_ll"])

    print(f"experiment   = {config.experiment_id}")
    print(f"sample_index = {args.sample_index}")
    print(f"context      = {context}")
    print()
    print("choice  tokens  chars     total_LL  token_avg_LL  harness_norm_LL  ending")
    for row in rows:
        print(
            f"{row['label']:>6}  {row['tokens']:>6}  {row['chars']:>5}  "
            f"{row['total_ll']:>11.4f}  {row['token_avg_ll']:>12.4f}  "
            f"{row['harness_norm_ll']:>15.4f}  {row['choice']}"
        )
    print()
    print(f"raw prediction          = {'ABCD'[raw_prediction]}")
    print(f"token-average prediction= {'ABCD'[token_norm_prediction]}")
    print(f"harness acc_norm pred   = {'ABCD'[harness_norm_prediction]}")
    print(f"gold                    = {'ABCD'[gold]}")


if __name__ == "__main__":
    main()
