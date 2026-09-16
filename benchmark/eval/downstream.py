"""lm-evaluation-harness integration and downstream provenance capture."""

from __future__ import annotations

import hashlib
import importlib.metadata as package_metadata
import json
from pathlib import Path
from typing import Any

from benchmark.config import ExperimentConfig, PROJECT_ROOT
from benchmark.eval.model import load_model


HARNESS_NAME = "lm-evaluation-harness"
DATASET_NAME = "Rowan/hellaswag"
DATASET_SPLIT = "validation"


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _task_config_path(task_name: str) -> Path:
    import lm_eval

    path = Path(lm_eval.__file__).resolve().parent / "tasks" / task_name / f"{task_name}.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"lm-eval task config not found: {path}")
    return path


def tokenizer_fingerprint(tokenizer: Any) -> str:
    """Hash stable tokenizer behavior without library-private serialization.

    The same Qwen tokenizer can be loaded by different Transformers releases
    across runtime environments. Hashing ``backend_tokenizer.to_str`` makes
    that implementation detail look like a protocol difference. The
    canonical fingerprint below instead covers the complete token-to-id map,
    special-token IDs, and representative encoding behavior.
    """
    probes = (
        "A person walks into a room.",
        " leading and trailing spaces ",
        "punctuation: (a), [b], can't; 3.14!",
        "Unicode: 量化模型 café naïve",
        "line one\nline two\tindented",
    )
    payload = {
        "algorithm": "v1.2-vocab-special-probes",
        "vocabulary": sorted(tokenizer.get_vocab().items()),
        "special_token_ids": {
            name: getattr(tokenizer, f"{name}_token_id", None)
            for name in ("bos", "eos", "pad", "unk", "sep", "cls", "mask")
        },
        "padding_side": tokenizer.padding_side,
        "truncation_side": tokenizer.truncation_side,
        "probe_encodings": [
            {
                "text": probe,
                "with_special_tokens": tokenizer.encode(probe, add_special_tokens=True),
                "without_special_tokens": tokenizer.encode(probe, add_special_tokens=False),
            }
            for probe in probes
        ],
    }
    serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return _sha256_bytes(serialized)


def task_protocol(config: ExperimentConfig, task_name: str, scope: str) -> dict[str, Any]:
    """Resolve one configured task and smoke/full sample limit."""
    downstream = config.data.get("downstream")
    if not isinstance(downstream, dict) or downstream.get("enabled") is not True:
        raise ValueError(f"Downstream evaluation is not enabled for {config.experiment_id}.")
    tasks = downstream.get("tasks")
    task = next(
        (item for item in tasks if isinstance(item, dict) and item.get("name") == task_name),
        None,
    )
    if task is None:
        raise ValueError(f"Task {task_name!r} is not configured for {config.experiment_id}.")
    if scope not in {"smoke", "full"}:
        raise ValueError("Downstream scope must be smoke or full.")
    scope_config = downstream["scope"]
    limit = scope_config["smoke_limit"] if scope == "smoke" else scope_config["full_limit"]
    return {
        "task": task,
        "batch_size": int(downstream["batch_size"]),
        "apply_chat_template": bool(downstream["apply_chat_template"]),
        "scope": scope,
        "limit": None if limit is None else int(limit),
    }


def evaluate_task(
    config: ExperimentConfig,
    *,
    task_name: str,
    scope: str,
) -> dict[str, Any]:
    """Evaluate one downstream task through lm-evaluation-harness."""
    from datasets import load_dataset
    from lm_eval import evaluator
    from lm_eval.models.huggingface import HFLM

    protocol = task_protocol(config, task_name, scope)
    task = protocol["task"]
    if task_name != "hellaswag":
        raise ValueError("benchmark v1.2 enables only HellaSwag.")

    dataset = load_dataset(
        DATASET_NAME,
        cache_dir=str(PROJECT_ROOT / ".hf_cache" / "datasets"),
    )
    dataset_split = dataset[DATASET_SPLIT]
    dataset_fingerprint = str(dataset_split._fingerprint)
    original_samples = len(dataset_split)

    loaded = load_model(config)
    model, tokenizer = loaded.model, loaded.tokenizer
    tokenizer_digest = tokenizer_fingerprint(tokenizer)
    harness_model = HFLM(
        pretrained=model,
        tokenizer=tokenizer,
        batch_size=protocol["batch_size"],
    )
    results = evaluator.simple_evaluate(
        model=harness_model,
        tasks=[task_name],
        num_fewshot=int(task["num_fewshot"]),
        batch_size=protocol["batch_size"],
        limit=protocol["limit"],
        apply_chat_template=protocol["apply_chat_template"],
        log_samples=False,
        random_seed=0,
        numpy_random_seed=1234,
        torch_random_seed=1234,
        fewshot_random_seed=1234,
    )
    if results is None:
        raise RuntimeError("lm-evaluation-harness returned no result on the main process.")

    metrics = results["results"][task_name]
    n_samples = int(results["n-samples"][task_name]["effective"])
    task_config_path = _task_config_path(task_name)
    return {
        "acc": float(metrics["acc,none"]),
        "acc_stderr": float(metrics["acc_stderr,none"]),
        "acc_norm": float(metrics["acc_norm,none"]),
        "acc_norm_stderr": float(metrics["acc_norm_stderr,none"]),
        "num_fewshot": int(task["num_fewshot"]),
        "n_samples": n_samples,
        "original_samples": original_samples,
        "scope": protocol["scope"],
        "limit": protocol["limit"],
        "primary_metric": str(task["primary_metric"]),
        "harness": HARNESS_NAME,
        "harness_version": package_metadata.version("lm_eval"),
        "task_version": str(results["versions"][task_name]),
        "task_name": task_name,
        "task_config_sha256": _sha256_bytes(task_config_path.read_bytes()),
        "dataset_name": DATASET_NAME,
        "dataset_split": DATASET_SPLIT,
        "dataset_fingerprint": dataset_fingerprint,
        "apply_chat_template": protocol["apply_chat_template"],
        "batch_size": protocol["batch_size"],
        "tokenizer_class": type(tokenizer).__name__,
        "tokenizer_vocab_size": int(len(tokenizer)),
        "tokenizer_fingerprint": tokenizer_digest,
        "tokenizer_fingerprint_algorithm": "v1.2-vocab-special-probes",
        "offline": True,
        "quality_comparable": protocol["scope"] == "full",
    }
