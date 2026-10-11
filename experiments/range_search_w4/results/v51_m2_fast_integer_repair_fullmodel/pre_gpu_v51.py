"""Final CPU-only gate before the single v51 GPU job."""
import hashlib, json, runpy, sys, time
from pathlib import Path
import torch, transformers, safetensors

ROOT = Path(r"E:\quant-thesis")
BASE = ROOT / "experiments/range_search_w4/results"
OUT = BASE / "v51_m2_fast_integer_repair_fullmodel"

def sha(p):
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()

def main():
    p = json.loads((OUT / "PROTOCOL.json").read_text(encoding="utf-8"))
    s = json.loads((OUT / "SOURCE_AUDIT.json").read_text(encoding="utf-8"))
    assert p["status"] == "FROZEN_BEFORE_GPU" and s["pass"]
    assert torch.__version__ == p["execution_runtime"]["torch_version_required"] == "2.6.0+cu124"
    assert str(Path(sys.executable).resolve()) == str((ROOT / ".venv/Scripts/python.exe").resolve())
    assert (OUT / "run_v51.py").is_file() and not (OUT / "GPU_START.json").exists()
    for d in ["original_codes", "formal/groups", "formal/rows", "R1/checkpoint/final", "evaluation", "reports", "diagnostics", "recovery"]:
        assert (OUT / d).is_dir()
    assert not list((OUT / "formal/groups").glob("layer_*.pt")), "new run must start from original M2"
    v48 = BASE / "v48_fast_numerical_acceptance"
    assert sha(v48 / "fast_accept.py") == p["algorithm_source"]["fast_accept_sha256"]
    assert sha(v48 / "ACCEPTANCE_RULES.md") == p["algorithm_source"]["acceptance_rules_sha256"]
    assert p["algorithm_source"]["v50_gram_path_used"] is False
    assert "mixed_gram" not in (OUT / "run_v51.py").read_text(encoding="utf-8")
    v45 = BASE / "v45_m2_single_integer_repair_fullmodel"
    assert len(list((v45 / "formal/groups").glob("layer_*.pt"))) == 96
    v46_audit = json.loads((BASE / "v46_m1_single_integer_repair_fullmodel/PPL_SOURCE_AUDIT.json").read_text(encoding="utf-8"))
    assert v46_audit["pass"] and not v46_audit["P3_read"]

    # Import the actual derived runner on CPU to verify paths and function binding without launching main().
    ns = runpy.run_path(str(OUT / "run_v51.py"), run_name="v51_cpu_import_preflight")
    rt, fast = ns["rt"], ns["fast"]
    assert Path(rt.OUT).resolve() == OUT.resolve()
    assert Path(rt.FP).resolve() == (ROOT / ".hf_cache/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775").resolve()
    assert fast.batch_repair.__module__ == "v48_fast_accept_v49"
    assert Path(fast.__file__).resolve() == (v48 / "fast_accept.py").resolve()
    hashes = {"runner": sha(OUT / "run_v51.py"), "protocol": sha(OUT / "PROTOCOL.json"), "source_audit": sha(OUT / "SOURCE_AUDIT.json"), "fast_accept": sha(v48 / "fast_accept.py"), "rules": sha(v48 / "ACCEPTANCE_RULES.md"), "runtime": sha(BASE / "v46_m1_single_integer_repair_fullmodel/runtime_v46.py"), "original_m2_model": sha(BASE / "range_w4_fullmodel_v18/M2/checkpoint/final/model.safetensors")}
    lock = {"pass": True, "time_epoch": time.time(), "gpu_started": False, "python": sys.executable, "torch": torch.__version__, "transformers": transformers.__version__, "safetensors": safetensors.__version__, "fast_function": "fast_accept.batch_repair", "fast_module_file": str(Path(fast.__file__).resolve()), "no_m1_or_m2_activations_loaded": True, "gpu_scope": "one formal run only; CAP=10800 seconds from GPU_START; 600 seconds final reserve", "hashes": hashes}
    (OUT / "RUNTIME_PREFLIGHT.json").write_text(json.dumps(lock, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("V51_PRE_GPU_CPU_GATE_PASS", json.dumps(lock, ensure_ascii=False), flush=True)

if __name__ == "__main__":
    main()
