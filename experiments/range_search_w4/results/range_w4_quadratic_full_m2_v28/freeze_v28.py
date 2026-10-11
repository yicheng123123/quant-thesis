"""Freeze v28 protocol and create isolated run folders without touching v18/v27."""
import hashlib
import json
from pathlib import Path

O = Path(__file__).resolve().parent
R = O.parents[3]
V = O.parent
V18 = V / "range_w4_fullmodel_v18"
V26 = V / "range_w4_factor_cache_v26"
V27 = V / "range_w4_quadratic_score_v27"


def sha(p):
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    v18 = json.loads((V18 / "PROTOCOL.json").read_text("utf-8"))
    v27 = json.loads((V27 / "SUMMARY.json").read_text("utf-8"))
    v26 = json.loads((V26 / "ACTIVATION_CACHE.json").read_text("utf-8"))
    sources = [
        V18 / "PROTOCOL.json", V18 / "CALIBRATION_MANIFEST.json", V18 / "CALIBRATION_VALIDATION.json",
        V18 / "run_formal.py", V18 / "run_precheck.py", V18 / "M2/RUN_REPORT.json",
        V18 / "M2/checkpoint/final/model.safetensors", V27 / "FORMULA.md", V27 / "run_v27.py",
        V27 / "SUMMARY.json", V27 / "RAW_SCORES_up_proj.npz", V27 / "RAW_SCORES_down_proj.npz",
        V26 / "ACTIVATION_CACHE.json", V26 / "cache/L0_up_proj_128x2048.f16",
        V26 / "cache/L0_down_proj_128x2048.f16",
        V / "range_w4_m2_runtime_v17_3/optimized_native_m2.py",
        R / ".venv-gptq/Lib/site-packages/gptqmodel/quantization/gptq.py",
    ]
    protocol = {
        "name": "v28 complete M2 dual-score validation and conditional quadratic-only run",
        "status": "FROZEN_BEFORE_FULLMODEL_GPU_RUN",
        "v18_protocol": v18,
        "scope": "same Qwen2.5-0.5B-Instruct revision, FP16 W4A16 symmetric dynamic GPTQ G128, original native compensation, alpha grid, 128x2048 WikiText-2 train documents, own-prefix input propagation and exact direct score selection",
        "modules": 168,
        "rows": 304128,
        "candidate_executions_each_full_run": 1680,
        "row_alpha_scores_each_full_run": 3041280,
        "scoring": {
            "direct": "v18 run_precheck.score_rows unchanged; X@(Q.float-W.float).T; FP32 matmul FP64 square accumulation; equal 262144 tokens",
            "quadratic": "C=X.T@X/262144 from exact current group X, no mask/weights; all entries kept; e=Q.float-W.float; e@C FP32 then FP64 row dot",
            "bias": "actual bias state audited module by module; unchanged common bias cancels; reference remains exact v18 weight-only target",
            "input_sharing": "one C per true v18 dependency/H input group; reuse for qkv siblings and up/gate siblings only after exact shared-input check and equal input hash",
            "hessian": "never substitute damped H, inverse, GPTQ loss, or compensation accumulation for C",
            "tie_break": "strict lower score; exact equality chooses larger alpha; no epsilon threshold",
        },
        "phase1": "run complete v18 M2 dynamics and prefix propagation; each candidate is generated once, direct and quadratic scores are both computed from that same tensor; commit and propagate the direct score choice; retain both full score vectors and selections per Linear; compare selected W/scale/zero/g_idx and all row-alpha scores",
        "phase2_gate": "only if every module has identical selected alpha and committed tensors, all phase1 correctness passes, and remaining 90-minute budget supports projected phase2 cost plus 20% reserve, run complete M2 again using quadratic-only selection and own-prefix propagation",
        "phase2": "same M2 implementation/settings and candidate grid; direct scorer disabled; C-only scorer selects and commits; save independent checkpoint chain; compare each input/H hash and normalized final tensors against v18 M2",
        "recoverability": "atomic checkpoints after each full dependency group (qkv, o, up_gate, down) and after each full layer; resume only from verified contiguous group checkpoints",
        "budget": {"gpu_wall_seconds_total_max": 5400, "soft_stop_reserve_seconds": 600, "include": ["source/hash setup", "model loads", "capture/refresh", "H and C builds", "warmups", "candidates", "both scorers", "retry/resume", "stitch/commit", "checkpoint and final IO"]},
        "v27_local_results": v27,
        "v26_activation_sha256": v26["input_sha256"],
        "ppl": "NOT_RUN; reuse v18 PPL interpretation only if final named tensors/config are exactly equal",
        "source_sha256": {str(p.relative_to(R)): sha(p) for p in sources},
        "historical_read_only": True,
        "new_artifacts": "v28 directory only",
    }
    p = O / "PROTOCOL.json"
    if p.exists():
        assert json.loads(p.read_text("utf-8")) == protocol
    else:
        p.write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + "\n", "utf-8")
    for label in ("validation", "quadratic_only"):
        out = O / label
        out.mkdir(exist_ok=True)
        (out / "raw").mkdir(exist_ok=True)
        manifest = out / "RUN_MANIFEST.json"
        if not manifest.exists():
            manifest.write_text(json.dumps({"status": "FROZEN_READY", "run": label,
                "protocol_sha256": sha(p), "calibration_manifest_sha256": sha(V18 / "CALIBRATION_MANIFEST.json"),
                "last_complete_group": None}, indent=2) + "\n", "utf-8")
    print("FROZEN", p)


if __name__ == "__main__":
    main()
