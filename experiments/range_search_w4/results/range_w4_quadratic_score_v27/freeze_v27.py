"""Freeze v27 comparison before any new GPU candidate scoring."""
import hashlib
import json
from pathlib import Path

O = Path(__file__).resolve().parent
R = O.parents[3]
V = O.parent
V18 = V / "range_w4_fullmodel_v18"
V26 = V / "range_w4_factor_cache_v26"


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    source = [
        V18 / "PROTOCOL.json", V18 / "CALIBRATION_MANIFEST.json",
        V18 / "run_precheck.py", V18 / "run_formal.py",
        V18 / "M2/checkpoint/final/model.safetensors",
        V / "range_w4_m2_runtime_v17_3/optimized_native_m2.py",
        V26 / "PROTOCOL.json", V26 / "ACTIVATION_CACHE.json",
        V26 / "PROFILE_ORIGINAL.json", V26 / "PROFILE_DECISION.json",
        R / ".venv-gptq/Lib/site-packages/gptqmodel/quantization/gptq.py",
    ]
    v18 = json.loads((V18 / "PROTOCOL.json").read_text("utf-8"))
    v26 = json.loads((V26 / "PROTOCOL.json").read_text("utf-8"))
    act = json.loads((V26 / "ACTIVATION_CACHE.json").read_text("utf-8"))
    assert v18["m2"]["alpha_grid"] == [i / 10 for i in range(1, 11)]
    assert act["documents"] == 128 and act["token_count"] == 262144
    assert act["input_sha256"]["up_proj"] == "b164fb179e772e548542ed89b6cf82adee601b545e020ea619799dc66531e856"
    assert act["input_sha256"]["down_proj"] == "9a159128abbd83130edf5a5741be93808c6e05a92840434f386aed9f1e778eb5"
    protocol = {
        "name": "v27 frozen M2 mathematical-equivalent candidate score check",
        "frozen_before_gpu": True,
        "model": v18["model"], "calibration": v18["calibration"],
        "quantization": v18["quantization"], "m2": v18["m2"],
        "targets": {"up_proj": [4864, 896], "down_proj": [896, 4864]},
        "input": "Read-only v26 captured formal 128x2048 own-prefix input, verified by frozen SHA256",
        "reference": "same X times pristine FP16 W cast to FP32; Q cast FP32; target is X@(Q.float-W.float).T; bias cancels / absent in targets",
        "direct_score": "v18 run_precheck.score_rows, 4-document chunks, FP32 matmul, FP64 output-square accumulation, divide by 262144",
        "quadratic_score": "independently build C=sum(X_fp32.T@X_fp32)/262144 in FP32 from 4-doc chunks; e=Q_fp16.float-W_fp16.float; row score=sum_fp64(e*(e@C))",
        "hessian_policy": "raw GPTQ H is mathematically 4096*C for 2048-token docs but uses different FP32 accumulation; not reused as C, never damped H/inverse",
        "candidate_grid": [i / 10 for i in range(1, 11)],
        "candidate_generation": "once per alpha per full Linear using existing dynamic G128 native compensation; both scorers share the exact tensor",
        "selection": "strictly smaller FP64 score; exact ties prefer larger alpha, as v18; any different selected alpha is mismatch, no tolerance relaxation",
        "validation": "all rows/all ten alpha; raw score abs/relative errors; best-second gap; selected W/scale/zero/g_idx; original direct full-row rescore differing candidates",
        "timing": "after validation, two warmed scorer repetitions; direct full10 versus C construction plus quadratic full10; CUDA sync at stage boundaries only; record peak allocated/reserved and reassessment cost",
        "gpu_budget_seconds": 600,
        "no_new_heldout_ppl_fullmodel": True,
        "source_sha256": {str(path.relative_to(R)): sha(path) for path in source},
        "activation_sha256": act["input_sha256"],
        "historical_read_only": True,
    }
    path = O / "PROTOCOL.json"
    if path.exists():
        assert json.loads(path.read_text("utf-8")) == protocol
    else:
        path.write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("FROZEN", path)


if __name__ == "__main__":
    main()
