#!/usr/bin/env python3
"""Record the actual Kaggle runtime without changing it."""
from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("KAGGLE_ENVIRONMENT.json"))
    parser.add_argument("--disk-path", type=Path, default=Path.cwd())
    args = parser.parse_args()

    import torch
    import tokenizers
    import transformers
    try:
        import psutil
        ram = {"total_bytes": int(psutil.virtual_memory().total),
               "available_bytes": int(psutil.virtual_memory().available),
               "process_rss_bytes": int(psutil.Process().memory_info().rss)}
    except Exception as exc:
        ram = {"available": False, "reason": f"psutil unavailable: {type(exc).__name__}: {exc}"}

    devices = []
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(index)
            devices.append({"index": index, "name": props.name,
                            "total_vram_bytes": int(props.total_memory),
                            "total_vram_gib": props.total_memory / (1024 ** 3),
                            "capability": [int(props.major), int(props.minor)]})
        torch.cuda.synchronize(0)
        free_bytes, total_bytes = torch.cuda.mem_get_info(0)
        cuda0_memory = {"free_bytes": int(free_bytes), "total_bytes": int(total_bytes)}
    else:
        cuda0_memory = None
    try:
        disk = shutil.disk_usage(args.disk_path)
        disk_info = {"path": str(args.disk_path.resolve()), "total_bytes": int(disk.total),
                     "free_bytes": int(disk.free)}
    except OSError as exc:
        disk_info = {"path": str(args.disk_path), "available": False, "reason": str(exc)}

    required_transformers = "4.49.0"
    required_tokenizers = "0.21.4"
    checks = {
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda0_available": bool(torch.cuda.is_available() and torch.cuda.device_count() >= 1),
        "transformers_pin": transformers.__version__ == required_transformers,
        "tokenizers_pin": tokenizers.__version__ == required_tokenizers,
    }
    result = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "python_version": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "tokenizers_version": tokenizers.__version__,
        "cuda_runtime": torch.version.cuda,
        "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "gpus": devices,
        "selected_device": "cuda:0" if checks["cuda0_available"] else None,
        "cuda0_free_vram": cuda0_memory,
        "tf32": {"matmul": bool(torch.backends.cuda.matmul.allow_tf32),
                 "cudnn": bool(torch.backends.cudnn.allow_tf32)},
        "deterministic": {"algorithms": bool(torch.are_deterministic_algorithms_enabled()),
                           "cudnn": bool(torch.backends.cudnn.deterministic)},
        "disk": disk_info,
        "cpu_ram": ram,
        "checks": checks,
        "device_policy": "Only cuda:0 is selected. Additional visible GPUs are recorded and not used.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if checks["cuda0_available"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
