from __future__ import annotations

import csv
import hashlib
import importlib.machinery
import importlib.util
import json
import math
import sys
import time
import types
from pathlib import Path

import torch


sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
REPO_RESULTS = ROOT.parent
AUDIT12 = REPO_RESULTS / "range_w4_omniquant_nan_audit_v21_2"
AUDIT13 = REPO_RESULTS / "range_w4_omniquant_amp_audit_v21_3"
SMOKE_DIR = HERE / "diagnostics" / "gate_smoke_07"
FIXTURE = AUDIT12 / "fixtures" / "actual_l21.pt"
FIXTURE_MANIFEST = AUDIT12 / "FIXTURE_MANIFEST.json"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def import_file(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    if SMOKE_DIR.exists():
        raise RuntimeError(f"smoke output already exists; refusing overwrite: {SMOKE_DIR}")
    SMOKE_DIR.mkdir(parents=True)
    source_manifest = read_json(FIXTURE_MANIFEST)
    expected_fixture_sha = next(
        item["sha256"] for item in source_manifest["files"]
        if item["path"].replace("\\", "/") == "fixtures/actual_l21.pt"
    )
    if sha(FIXTURE) != expected_fixture_sha:
        raise RuntimeError("v21.2 actual_l21 fixture hash mismatch")

    official_port = HERE / "official_port"
    preflight = ROOT / "LOCAL_PREFLIGHT"
    sys.path.insert(0, str(AUDIT12))
    sys.path.insert(0, str(official_port))
    sys.path.insert(0, str(preflight))
    if importlib.util.find_spec("termcolor") is None:
        termcolor = types.ModuleType("termcolor")
        termcolor.colored = lambda text, *args, **kwargs: text
        termcolor.__spec__ = importlib.machinery.ModuleSpec("termcolor", loader=None)
        sys.modules["termcolor"] = termcolor

    runner = import_file("v21_4_runner_for_gate_smoke", HERE / "run_formal_m3.py")
    runner.write_runtime_environment()
    lock = read_json(HERE / "FORMAL_PROTOCOL_LOCK.json")
    audit = read_json(HERE / "OMNIQUANT_SOURCE_AUDIT.json")
    for relative, info in audit["files"].items():
        expected = info.get("patched_sha256") or info["official_sha256"]
        if runner.sha_file(official_port / relative) != expected:
            raise RuntimeError(f"v21.4 source hash mismatch: {relative}")

    import audit_common as ac
    from quantize.utils import get_omni_parameters, let_parameters, lwc_parameters, smooth_and_quant_temporary

    fixture = torch.load(FIXTURE, map_location="cpu", weights_only=False)
    if fixture.get("layer") != 21:
        raise RuntimeError("frozen v21.2 fixture is not decoder layer 21")
    layer, args = ac.new_layer(fixture)
    args.v21_resume_completed_layers = 21
    args.v21_prefix_successful_optimizer_steps = 53760
    args.v21_prefix_gradscaler_skipped_steps = 0
    args.output_dir = str(SMOKE_DIR)
    args.smoke_layer_limit = 22

    runner.STEP_LOG = SMOKE_DIR / "training_steps.csv"
    runner.EPOCH_LOG = SMOKE_DIR / "training_epochs.csv"
    runner.BLOCK_LOG = SMOKE_DIR / "block_health.csv"
    runner.AMP_EVENT_LOG = SMOKE_DIR / "AMP_EVENT_LOG.csv"
    counts, logs = runner.install_source_callbacks(args, lock, None, "v21_4_gate_smoke_01")

    # Load the isolated v21.4 patched native scaler explicitly. Its calls retain
    # upstream scale/backward/unscale/step/update order and only expose metadata.
    patched_utils = import_file("v21_4_patched_native_utils", official_port / "utils.py")
    scaler = patched_utils.NativeScalerWithGradNormCount()
    optimizer = torch.optim.AdamW(
        [
            {"params": let_parameters(layer, False), "lr": args.let_lr},
            {"params": lwc_parameters(layer), "lr": args.lwc_lr},
        ],
        weight_decay=args.wd,
    )
    owned = list(lwc_parameters(layer))
    hook_counts = {"pre": 0, "post": 0}

    def pre_hook(*_):
        hook_counts["pre"] += 1

    def post_hook(*_):
        hook_counts["post"] += 1

    pre_handle = optimizer.register_step_pre_hook(pre_hook)
    post_handle = optimizer.register_step_post_hook(post_hook)
    sample_records = []
    device = torch.device("cuda:0")
    try:
        for sample in range(4):
            hook_counts["pre"] = hook_counts["post"] = 0
            inputs = ac.tree(fixture["inputs"][sample:sample + 1], device)
            targets = ac.tree(fixture["targets"][sample:sample + 1], device)
            kwargs = ac.tree(fixture["kwargs"][sample], device)
            with torch.cuda.amp.autocast():
                smooth_and_quant_temporary(layer, args, False)
                output = layer(inputs, **kwargs)[0]
                if not bool(torch.isfinite(output).all().item()):
                    raise FloatingPointError(f"smoke forward output nonfinite at sample={sample}")
                loss = torch.nn.MSELoss()(targets, output)
            if not math.isfinite(float(loss.detach())):
                raise FloatingPointError(f"smoke loss nonfinite at sample={sample}")
            optimizer.zero_grad()
            before = [param.detach().clone() for param in owned]
            optimizer_state_before = runner.optimizer_state_signature(optimizer)
            norm = scaler(loss, optimizer, parameters=get_omni_parameters(layer, False)).cpu()
            args.v21_step_check_callback(
                layer, 21, 0, sample, loss, norm, optimizer,
                scaler.last_step_diagnostics, dict(hook_counts), before, optimizer_state_before,
            )
            skipped = scaler.last_step_diagnostics["found_inf"] > 0
            record = {
                "layer": 21,
                "epoch": 0,
                "sample": sample,
                "loss": float(loss.detach().float().item()),
                "scale_before": scaler.last_step_diagnostics["scale_before"],
                "found_inf": scaler.last_step_diagnostics["found_inf"],
                "step_hook_counts": dict(hook_counts),
                "step_skipped": skipped,
                "scale_after": scaler.last_step_diagnostics["scale_after"],
                "lwc_finite": all(bool(torch.isfinite(param).all()) for param in owned),
                "optimizer_state_finite": all(
                    bool(torch.isfinite(value).all())
                    for state in optimizer.state.values()
                    for value in state.values()
                    if isinstance(value, torch.Tensor)
                ),
                "optimizer_state_unchanged_on_skip": (
                    skipped and runner.optimizer_state_signature(optimizer)["sha256"] == optimizer_state_before["sha256"]
                ),
                "norm_finite": bool(torch.isfinite(norm).all()),
                "formal_checkpoint_committed": False,
            }
            sample_records.append(record)
    finally:
        pre_handle.remove()
        post_handle.remove()
        for handle in logs:
            handle.close()

    first = sample_records[0]
    later_success = [record for record in sample_records[1:] if not record["step_skipped"]]
    passed = (
        first["step_skipped"]
        and first["found_inf"] > 0
        and first["scale_after"] < first["scale_before"]
        and first["step_hook_counts"] == {"pre": 0, "post": 0}
        and first["lwc_finite"]
        and first["optimizer_state_finite"]
        and first["optimizer_state_unchanged_on_skip"]
        and len(later_success) == 3
        and all(record["step_hook_counts"] == {"pre": 1, "post": 1} for record in later_success)
        and all(record["lwc_finite"] and record["optimizer_state_finite"] for record in sample_records)
    )
    result = {
        "status": "PASS" if passed else "FAIL",
        "fixture": str(FIXTURE.relative_to(ROOT.parent.parent)),
        "fixture_sha256": expected_fixture_sha,
        "fixture_provenance": "v21.2 actual L21 block_20-prefix optimizer-init capture, reused byte-identically; no prefix recapture",
        "v21_4_runner_sha256": runner.sha_file(HERE / "run_formal_m3.py"),
        "v21_4_native_scaler_source_sha256": runner.sha_file(official_port / "utils.py"),
        "sample_count": len(sample_records),
        "samples": sample_records,
        "formal_committed_updates": 0,
        "formal_checkpoint_committed": False,
        "dynamic_scaler_initial_scale_left_to_default": True,
        "sample0_loss_finite": math.isfinite(first["loss"]),
        "sample0_native_skip_and_backoff": bool(first["step_skipped"] and first["scale_after"] < first["scale_before"]),
        "samples1_to3_successful_updates": len(later_success),
        "nonfinite_forward_or_loss_fatal_gate_exercised_on_finite_outputs": True,
        "optimizer_and_scaler_calls_per_sample": "exact native wrapper; no retries; each scheduled sample executed once",
        "runtime_seconds": time.strftime("%Y-%m-%d %H:%M:%S%z"),
    }
    (SMOKE_DIR / "M3_PATCHED_GATE_SMOKE.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    if not passed:
        raise RuntimeError("v21.4 patched gate smoke failed")
    print(json.dumps({"status": result["status"], "sample0": first,
                      "successful_steps_after_backoff": len(later_success)}, indent=2))


if __name__ == "__main__":
    main()
