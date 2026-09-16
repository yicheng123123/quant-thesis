"""Stable interface shared by every benchmark model loader."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from benchmark.config import ExperimentConfig


@dataclass(frozen=True)
class AdapterCapabilities:
    """Operations supported by a loader, independent of result comparability."""

    quality: bool = True
    downstream: bool = True
    system: bool = True
    checkpoint_metadata: bool = True

    def to_dict(self) -> dict[str, bool]:
        return asdict(self)


@dataclass(frozen=True)
class LoadedModel:
    """Algorithm-neutral value passed from an adapter to evaluators."""

    model: Any
    tokenizer: Any
    detected_metadata: dict[str, Any]
    capabilities: AdapterCapabilities


class QuantizationAdapter(ABC):
    """Load one checkpoint without exposing its quantization method to evaluators."""

    name = ""
    capabilities = AdapterCapabilities()

    @classmethod
    def supports_method(cls, method: str) -> bool:
        """Return whether this loader accepts the declared algorithm identity."""
        del method
        return True

    @classmethod
    @abstractmethod
    def inspect_checkpoint(cls, checkpoint: Path | None) -> dict[str, Any]:
        """Return only identity fields that are actually detectable on disk."""

    @abstractmethod
    def load(self, config: "ExperimentConfig") -> LoadedModel:
        """Load a model/tokenizer pair and return normalized adapter output."""

    @staticmethod
    def finalize(model: Any, tokenizer: Any) -> tuple[Any, Any]:
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model.eval()
        return model, tokenizer
