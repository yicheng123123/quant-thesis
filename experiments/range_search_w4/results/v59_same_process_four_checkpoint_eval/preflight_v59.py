"""CPU/source and runtime preflight for v59; no model forward is performed."""
from __future__ import annotations

import importlib.util
import json
import time
from pathlib import Path

ROOT = Path(r"E:\quant-thesis")
OUT = ROOT / "experiments/range_search_w4/results/v59_same_process_four_checkpoint_eval"
RUNNER = OUT / "run_v59_eval.py"
spec = importlib.util.spec_from_file_location("v59_runner_preflight", RUNNER)
runner = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(runner)

runner.torch.set_num_threads(4)
runner.torch.backends.cuda.matmul.allow_tf32 = False
runner.torch.backends.cudnn.allow_tf32 = False
runner.torch.set_float32_matmul_precision("highest")
started = time.perf_counter()
sources, manifests = runner.verify_source_records()
runtime = runner.stable_runtime_snapshot()
v58 = runner.read_json(runner.V58 / "ENVIRONMENT_LOCK.json")
for actual, recorded in [
    (runtime["python"], v58["python"]),
    (runtime["pytorch"], v58["pytorch"]),
    (runtime["cuda_runtime"], v58["cuda_build"]),
    (runtime["transformers"], v58["transformers"]),
    (runtime["tokenizers"], v58["tokenizers"]),
    (runtime["gpu"], v58["device"]),
]:
    if actual != recorded:
        raise RuntimeError(f"current runtime differs from v58: {actual!r} vs {recorded!r}")
if runner.sha_file(runner.V58 / "run_v58_eval.py") != runner.V58_RUNNER_EXPECTED_SHA:
    raise RuntimeError("v58 runner source hash mismatch")
if runner.sha_file(ROOT / "benchmark/eval/ppl.py") != runner.EXPECTED_CORE_EVALUATOR_SHA:
    raise RuntimeError("frozen core evaluator source hash mismatch")
test_file = OUT / "_write_probe.tmp"
test_file.write_text("v59 writable\n", encoding="utf8")
test_file.unlink()
elapsed = time.perf_counter() - started
creation = (OUT.stat().st_ctime_ns / 1e9)
charged = max(0.0, time.time() - creation)
pair_wall = float(runner.read_json(runner.V58 / "COST_ACCOUNTING.json")["process_wall_including_loader_and_evaluation"])
projection = charged + 4 * pair_wall + 4 * 15 + 90 + 60 + runner.FINISH_RESERVE_SECONDS
if projection >= runner.CAP_SECONDS:
    raise TimeoutError(f"v59 preflight projection {projection:.1f}s exceeds {runner.CAP_SECONDS:.1f}s cap")
print(json.dumps({
    "status": "PASS",
    "preflight_seconds": elapsed,
    "charged_elapsed_since_output_creation_seconds": charged,
    "projected_total_with_reserve_seconds": projection,
    "v58_pair_wall_seconds": pair_wall,
    "runtime": runtime,
    "arms": {k: {x: v for x, v in rec.items() if x != "named_expected_map"} for k, rec in sources.items()},
    "manifests": manifests,
    "gpu_model_forward_or_evaluation": False,
}, ensure_ascii=False), flush=True)
