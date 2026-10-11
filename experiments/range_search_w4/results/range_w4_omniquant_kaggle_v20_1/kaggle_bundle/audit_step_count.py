#!/usr/bin/env python3
"""Recompute formal OmniQuant optimizer updates from the shipped loop source."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle-root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--output", type=Path, default=Path("FORMAL_STEP_COUNT.json"))
    args = parser.parse_args()
    root = args.bundle_root.resolve()
    config = json.loads((root / "configs/m3_preflight.json").read_text(encoding="utf-8"))
    source_path = root / "omniquant_port/upstream/quantize/omniquant.py"
    lines = source_path.read_text(encoding="utf-8").splitlines()
    snippets = {
        "layer_bound": "smoke_layer_limit = getattr(args, \"smoke_layer_limit\", len(layers))",
        "layer_loop": "for i in range(smoke_layer_limit):",
        "epoch_loop": "for epochs in range(args.epochs):",
        "sample_loop": "for j in range(args.nsamples//args.batch_size):",
        "optimizer_update": "norm = loss_scaler(loss, optimizer,parameters= get_omni_parameters(qlayer, use_shift)).cpu()",
        "batch_guard": "if args.batch_size != 1:",
    }
    locations = {}
    for name, snippet in snippets.items():
        found = [index + 1 for index, line in enumerate(lines) if snippet in line]
        if len(found) != 1:
            raise SystemExit(f"FAIL: expected one {name} source line, found {found}")
        locations[name] = {"line": found[0], "snippet": snippet}
    manifest = json.loads((root / "manifests/v20_evidence/QWEN25_MODULE_MAP.json").read_text(encoding="utf-8"))
    formal = config["formal_training"]
    layers = int(manifest["layers"])
    if manifest.get("target_projection_count") != 168 or layers != 24:
        raise SystemExit("FAIL: frozen Qwen module map is not 24 layers × 7 target projections")
    samples = int(formal["samples"])
    batch = int(formal["batch_size"])
    epochs = int(formal["epochs"])
    if batch != 1 or samples % batch:
        raise SystemExit("FAIL: frozen Qwen batch semantics changed")
    per_block_epoch = samples // batch
    total = layers * epochs * per_block_epoch
    result = {
        "status": "PASS",
        "source_path": "omniquant_port/upstream/quantize/omniquant.py",
        "source_sha256": sha(source_path),
        "module_map_path": "manifests/v20_evidence/QWEN25_MODULE_MAP.json",
        "module_map_sha256": sha(root / "manifests/v20_evidence/QWEN25_MODULE_MAP.json"),
        "decoder_layers": layers,
        "target_projections": int(manifest["target_projection_count"]),
        "samples_per_epoch": samples,
        "batch_size": batch,
        "epochs": epochs,
        "steps_per_sample_per_epoch": 1,
        "optimizer_updates_per_block_per_epoch": per_block_epoch,
        "optimizer_updates_per_block": epochs * per_block_epoch,
        "optimizer_updates_per_epoch_all_blocks": layers * per_block_epoch,
        "total_optimizer_updates": total,
        "formula": f"{layers} decoder blocks × {epochs} epochs × ({samples} calibration samples / batch size {batch}) = {total:,} optimizer updates",
        "source_locations": locations,
        "formal_epochs_unchanged": True,
        "preflight_smoke_limits_excluded": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
