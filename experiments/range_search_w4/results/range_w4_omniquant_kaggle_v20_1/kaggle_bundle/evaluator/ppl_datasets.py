"""Dataset-text adapters shared by the token-level PPL evaluator."""

from __future__ import annotations

from itertools import islice
from typing import Any

from datasets import load_dataset


SUPPORTED_DATASET_ADAPTERS = frozenset({"wikitext2", "c4"})


def _load_wikitext2_text(ppl_config: dict[str, Any], cache_dir: str) -> str:
    """Preserve the original WikiText-2 loading and text ordering exactly."""
    dataset = load_dataset(
        ppl_config["dataset_id"],
        ppl_config["dataset_config"],
        split=ppl_config["split"],
        cache_dir=cache_dir,
    )
    return "\n\n".join(dataset["text"])


def _load_c4_text(ppl_config: dict[str, Any], cache_dir: str) -> str:
    """Read a bounded, deterministic prefix of C4 through streaming iteration."""
    document_limit = ppl_config.get("document_limit")
    if document_limit is None:
        raise ValueError(
            "C4 PPL requires ppl.document_limit > 0; refusing to scan the full validation split."
        )
    if isinstance(document_limit, bool) or int(document_limit) <= 0:
        raise ValueError("C4 ppl.document_limit must be a positive integer.")
    if ppl_config.get("streaming", True) is not True:
        raise ValueError("C4 PPL requires ppl.streaming=true to avoid loading the full split.")

    dataset = load_dataset(
        ppl_config["dataset_id"],
        ppl_config["dataset_config"],
        split=ppl_config["split"],
        streaming=True,
        cache_dir=cache_dir,
    )
    texts: list[str] = []
    for sample in islice(dataset, int(document_limit)):
        if not isinstance(sample, dict) or "text" not in sample:
            raise ValueError("C4 samples must contain a text field.")
        text = sample["text"]
        if not isinstance(text, str):
            raise ValueError("C4 sample text must be a string.")
        texts.append(text)
    if len(texts) != int(document_limit):
        raise ValueError(
            f"C4 split ended after {len(texts)} documents; "
            f"requested {int(document_limit)}."
        )
    return "\n\n".join(texts)


def load_ppl_text(ppl_config: dict[str, Any], cache_dir: str) -> str:
    """Load one deterministic text stream for the shared PPL tokenizer."""
    adapter = ppl_config.get("dataset_adapter", "wikitext2")
    if adapter == "wikitext2":
        return _load_wikitext2_text(ppl_config, cache_dir)
    if adapter == "c4":
        return _load_c4_text(ppl_config, cache_dir)
    supported = ", ".join(sorted(SUPPORTED_DATASET_ADAPTERS))
    raise ValueError(f"Unsupported ppl.dataset_adapter={adapter!r}; expected one of: {supported}.")
