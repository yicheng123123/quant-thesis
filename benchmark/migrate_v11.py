"""Backfill v1.1 metadata onto existing validated v1 latest-result files."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from benchmark.config import (
    BENCHMARK_ROOT,
    RAW_RESULTS_DIR,
    SCHEMA_VERSION,
    load_config,
)


def _legacy_run_id(payload: dict[str, Any]) -> str:
    """Create a stable migration ID from the old result timestamp."""
    experiment_id = str(payload["experiment_id"])
    updated_at = str(payload.get("updated_at_utc", "unknown"))
    try:
        stamp = datetime.fromisoformat(updated_at.replace("Z", "+00:00")).strftime(
            "%Y%m%d_%H%M%S"
        )
    except ValueError:
        stamp = "unknown"
    return f"migrated_{stamp}_{experiment_id}"


def _probe_environment(python_executable: Path) -> dict[str, Any]:
    """Probe the configured environment instead of the migration process env."""
    code = (
        "import json; "
        "from benchmark.config import collect_environment_fingerprint; "
        "print(json.dumps(collect_environment_fingerprint(), ensure_ascii=False))"
    )
    try:
        completed = subprocess.run(
            [str(python_executable), "-X", "utf8", "-c", code],
            cwd=python_executable.parents[2],
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        )
        lines = [line for line in completed.stdout.splitlines() if line.strip()]
        fingerprint = json.loads(lines[-1])
        if isinstance(fingerprint, dict):
            return fingerprint
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError, IndexError):
        pass
    return {
        "status": "probe_failed",
        "python_executable": str(python_executable),
    }


def migrate_file(path: Path, *, force: bool) -> Path | None:
    """Add v1.1 fields and create one immutable migrated snapshot."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object: {path}")
    if payload.get("schema_version") == SCHEMA_VERSION and payload.get("run_id"):
        return None

    config = load_config(BENCHMARK_ROOT / "configs" / f"{payload['experiment_id']}.yaml")
    run_id = _legacy_run_id(payload)
    if force:
        run_id = f"{run_id}_forced"
    payload["schema_version"] = SCHEMA_VERSION
    payload["run_id"] = run_id
    payload["status"] = "passed"
    environment = _probe_environment(config.python_executable)
    environment["capture_status"] = "backfilled_during_v1_migration"
    environment["original_run_environment_recorded"] = False
    payload["environment"] = environment
    provenance = payload.get("provenance")
    if not isinstance(provenance, dict):
        provenance = {}
    provenance["metadata_migration"] = "v1_to_v1.1"
    payload["provenance"] = provenance

    run_dir = RAW_RESULTS_DIR / "runs"
    run_dir.mkdir(parents=True, exist_ok=True)
    snapshot_path = run_dir / f"{run_id}.json"
    if snapshot_path.exists() and not force:
        raise FileExistsError(snapshot_path)
    serialized = json.dumps(payload, ensure_ascii=False, indent=2)
    snapshot_path.write_text(serialized, encoding="utf-8")
    temp_path = path.with_suffix(".json.tmp")
    temp_path.write_text(serialized, encoding="utf-8")
    temp_path.replace(path)
    return snapshot_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill benchmark v1.1 result metadata")
    parser.add_argument("--force", action="store_true", help="Replace existing migrated snapshots.")
    args = parser.parse_args()

    outputs = []
    for path in sorted(RAW_RESULTS_DIR.glob("*.json")):
        output = migrate_file(path, force=args.force)
        if output is not None:
            outputs.append(output)
    if not outputs:
        print("migration=already_current")
    for output in outputs:
        print(f"migrated={output}")


if __name__ == "__main__":
    main()
