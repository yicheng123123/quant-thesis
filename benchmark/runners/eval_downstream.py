"""Run one lm-evaluation-harness task in a config-selected process."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path
from typing import Any

from benchmark.config import (
    PROJECT_ROOT,
    apply_runtime_environment,
    load_config,
    utc_now,
    write_result,
)


def _existing_downstream(config_path: Path) -> dict[str, Any]:
    config = load_config(config_path, check_paths=False)
    if not config.raw_result_path.is_file():
        return {}
    payload = json.loads(config.raw_result_path.read_text(encoding="utf-8"))
    quality = payload.get("quality")
    if not isinstance(quality, dict):
        return {}
    downstream = quality.get("downstream")
    return dict(downstream) if isinstance(downstream, dict) else {}


def run_worker(config_path: Path, task_name: str, scope: str) -> None:
    """Load one model, run one downstream task, and write a unified result."""
    config = load_config(config_path, check_paths=True)
    apply_runtime_environment()
    os.environ["HF_DATASETS_CACHE"] = str(PROJECT_ROOT / ".hf_cache" / "datasets")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    from benchmark.eval.downstream import evaluate_task

    task_result = evaluate_task(config, task_name=task_name, scope=scope)
    downstream = _existing_downstream(config_path)
    downstream[task_name] = task_result
    result_path = write_result(
        config,
        {
            "quality": {"downstream": downstream},
            "provenance": {
                "downstream": "lm-evaluation-harness",
                "downstream_run_at_utc": utc_now(),
                "downstream_scope": scope,
                "downstream_task": task_name,
                "downstream_offline": True,
            },
        },
    )
    print(f"downstream_result={result_path}", flush=True)
    print(
        f"task={task_name} scope={scope} n_samples={task_result['n_samples']} "
        f"acc={task_result['acc']:.12f} acc_norm={task_result['acc_norm']:.12f}",
        flush=True,
    )


def launch_isolated(config_path: Path, task_name: str, scope: str) -> None:
    """Spawn the worker under the Python interpreter declared by the config."""
    config = load_config(config_path, check_paths=True)
    command = [
        str(config.python_executable),
        "-X",
        "utf8",
        "-m",
        "benchmark.runners.eval_downstream",
        "--config",
        str(config.path),
        "--task",
        task_name,
        "--scope",
        scope,
        "--worker",
    ]
    environment = os.environ.copy()
    environment["PYTHONUTF8"] = "1"
    subprocess.run(command, cwd=PROJECT_ROOT, env=environment, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified isolated downstream runner")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--task", default="hellaswag")
    parser.add_argument("--scope", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.worker:
        run_worker(args.config, args.task, args.scope)
    else:
        launch_isolated(args.config, args.task, args.scope)


if __name__ == "__main__":
    main()
