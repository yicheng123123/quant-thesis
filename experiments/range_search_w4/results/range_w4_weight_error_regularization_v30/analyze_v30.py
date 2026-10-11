"""Artifact-only v30: fixed-candidate weight-error regularization diagnosis.

Reads frozen v24/v25 artifacts, never calls CUDA, GPTQ, model forwards, held
scoring, PPL, or quantization. New files are written only beside this script.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import statistics
import time
from collections import Counter, defaultdict
from decimal import Decimal, localcontext
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import numpy as np
import torch
from safetensors import safe_open

torch.set_num_threads(2)

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[3]
RESULTS = OUT.parent
V24 = RESULTS / "range_w4_structured_group_e2_v24"
V25 = RESULTS / "range_w4_dense_scalar_audit_v25"
V23 = RESULTS / "range_w4_block_objective_e1_v23"
REV = "7ae557604adf67be50417f59c2c2f167def9a775"
MODEL = ROOT / ".hf_cache/hub/models--Qwen--Qwen2.5-0.5B-Instruct" / f"snapshots/{REV}/model.safetensors"
LAYERS = [0, 12, 23]
PROJECTIONS = ["up_proj", "down_proj"]
ROW_IDS = {"up_proj": [0, 224, 448, 672], "down_proj": [0, 224, 448, 672]}
SEQ = 2048
N_DOCS = 64
TAUS = [0.0, 0.01, 0.1, 1.0]
CPU_BUDGET_SECONDS = 900.0
START_WALL = time.perf_counter()
START_CPU = time.process_time()
CHECKS: dict[str, bool] = {}
SOURCES: dict[str, dict] = {}
PHASES: list[dict] = []


def rel(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def sha_tensor(t: torch.Tensor) -> str:
    return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def sha_json(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def add_source(path: Path, role: str, expected: str | None = None, actual: str | None = None) -> str:
    path = Path(path).resolve()
    digest = actual if actual is not None else sha_file(path)
    expected_match = None if expected is None else digest == expected
    SOURCES[str(path)] = {"path": rel(path), "role": role, "sha256": digest,
                          "expected_sha256": expected, "expected_match": expected_match}
    if expected is not None:
        check("source_hash:" + rel(path), expected_match)
    return digest


def check(name: str, condition: bool) -> None:
    CHECKS[name] = bool(condition)
    if not condition:
        raise AssertionError(name)


def read_json(path: Path, role: str, expected: str | None = None) -> dict:
    add_source(path, role, expected)
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_csv(path: Path, role: str) -> list[dict]:
    add_source(path, role)
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_json(name: str, obj) -> None:
    p = OUT / name
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(p)


def write_csv(name: str, rows: list[dict]) -> None:
    p = OUT / name
    tmp = p.with_suffix(p.suffix + ".tmp")
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with tmp.open("w", encoding="utf-8", newline="") as f:
        if not fields:
            f.write("\n")
        else:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            for row in rows:
                w.writerow({k: json.dumps(v, ensure_ascii=False, allow_nan=False) if isinstance(v, (dict, list, tuple)) else v
                            for k, v in row.items()})
    tmp.replace(p)


def alpha_key(x) -> str:
    return f"{float(x):.12g}"


def split_positions(protocol: dict, split: str, kind: str) -> list[int]:
    lo, hi = protocol["documents"]["split_ids"][split][f"{kind}_positions"]
    return list(range(int(lo), int(hi) + 1))


def phase(name: str) -> None:
    PHASES.append({"phase": name, "wall_seconds_since_start": time.perf_counter() - START_WALL,
                   "cpu_seconds_since_start": time.process_time() - START_CPU})
    if time.process_time() - START_CPU > CPU_BUDGET_SECONDS:
        raise TimeoutError(f"CPU analysis exceeded {CPU_BUDGET_SECONDS:.0f} seconds")


def check_close(a: float, b: float, *, rel_tol=1e-11, abs_tol=1e-15) -> bool:
    return math.isclose(float(a), float(b), rel_tol=rel_tol, abs_tol=abs_tol)


def input_source_path(li: int, projection: str) -> tuple[Path, dict]:
    name = f"input_L{li}_{projection}.f16"
    if projection == "up_proj":
        manifest = PREFIX_MANIFEST
        path = V23 / "cache" / name
    else:
        manifest = DOWN_MANIFEST
        path = V24 / "cache" / name
    return path, manifest["files"][name]


def read_and_hash_inputs(protocol: dict) -> dict:
    """One sequential pass per frozen input: whole-file hash, search-slice hash, trace."""
    out = {}
    for li in LAYERS:
        for projection in PROJECTIONS:
            path, entry = input_source_path(li, projection)
            input_dim = 896 if projection == "up_proj" else 4864
            expected_shape = [N_DOCS, SEQ, input_dim]
            if "shape" in entry:
                check(f"input_shape_manifest:{li}:{projection}", entry["shape"] == expected_shape)
            if "dtype" in entry:
                check(f"input_dtype_manifest:{li}:{projection}", entry["dtype"] == "float16")
            docs_by_split = {s: split_positions(protocol, s, "search") for s in ("S0", "S1")}
            docsets = {s: set(v) for s, v in docs_by_split.items()}
            split_hashers = {s: hashlib.sha256() for s in docs_by_split}
            sqsum = {s: 0.0 for s in docs_by_split}
            whole = hashlib.sha256()
            doc_bytes = SEQ * input_dim * 2
            check(f"input_expected_size:{li}:{projection}", path.stat().st_size == doc_bytes * N_DOCS == entry["size_bytes"])
            with path.open("rb") as f:
                for di in range(N_DOCS):
                    block = f.read(doc_bytes)
                    check(f"input_doc_read:{li}:{projection}:{di}", len(block) == doc_bytes)
                    whole.update(block)
                    for split, indices in docsets.items():
                        if di in indices:
                            split_hashers[split].update(block)
                            values = np.frombuffer(block, dtype="<f2").astype(np.float64)
                            sqsum[split] += float(np.dot(values, values))
                    if time.process_time() - START_CPU > CPU_BUDGET_SECONDS:
                        raise TimeoutError(f"CPU analysis exceeded {CPU_BUDGET_SECONDS:.0f} seconds while hashing inputs")
                check(f"input_no_trailing_bytes:{li}:{projection}", f.read(1) == b"")
            whole_hex = whole.hexdigest()
            add_source(path, f"frozen {projection} activations L{li}", entry["sha256"], whole_hex)
            for split in ("S0", "S1"):
                n_docs = len(docs_by_split[split])
                n_tokens = n_docs * SEQ
                c_trace = sqsum[split] / n_tokens
                kappa = c_trace / input_dim
                out[(li, projection, split)] = {
                    "layer": li, "projection": projection, "split_id": split,
                    "input_dim": input_dim, "input_file": rel(path), "input_file_sha256": whole_hex,
                    "input_expected_sha256": entry["sha256"],
                    "input_shape_assurance": "manifest shape" if "shape" in entry else "v24 protocol dimension and exact manifest byte size",
                    "input_dtype_assurance": entry.get("dtype", "float16 from .f16 cache format"),
                    "search_input_bytes_sha256": split_hashers[split].hexdigest(),
                    "search_doc_indices": docs_by_split[split],
                    "search_doc_indices_sha256": sha_json(docs_by_split[split]),
                    "search_documents": n_docs, "search_tokens": n_tokens,
                    "search_input_square_sum": sqsum[split], "C_search_trace": c_trace,
                    "kappa_trace_over_input_dim": kappa,
                    "C_normalization": "sum over FP16 search-token input vectors x x^T divided by N_tokens; kappa=trace(C)/input_dim",
                }
    return out


def load_hessian_stats(protocol: dict, inputs: dict) -> dict:
    stats = {}
    for li in LAYERS:
        for projection in PROJECTIONS:
            dim = inputs[(li, projection, "S0")]["input_dim"]
            for split in ("S0", "S1"):
                path = V24 / "cache" / "hessians" / f"L{li}_{projection}_{split}.pt"
                meta_path = path.with_suffix(".json")
                obj = torch.load(path, map_location="cpu", weights_only=True)
                add_source(path, "v24 undamped search Hessian cache")
                meta = read_json(meta_path, "v24 Hessian sidecar")
                ids = split_positions(protocol, split, "search")
                check(f"hessian_tensor_hash:{li}:{projection}:{split}", sha_tensor(obj["H"]) == obj["H_sha256"] == meta["H_sha256"])
                check(f"hessian_docs:{li}:{projection}:{split}", obj["doc_indices"] == ids == meta["doc_indices"])
                check(f"hessian_nsamples_docs:{li}:{projection}:{split}", int(obj["nsamples"]) == len(ids))
                check(f"hessian_shape_dtype:{li}:{projection}:{split}", tuple(obj["H"].shape) == (dim, dim) and obj["H"].dtype == torch.float32)
                h_trace = float(torch.diagonal(obj["H"]).to(torch.float64).sum().item())
                n_tokens = len(ids) * SEQ
                kappa_h = h_trace / (2.0 * SEQ * dim)
                kappa_x = inputs[(li, projection, split)]["kappa_trace_over_input_dim"]
                rel_gap = (kappa_h - kappa_x) / kappa_x if kappa_x else math.nan
                stats[(li, projection, split)] = {
                    "hessian_path": rel(path), "hessian_file_sha256": SOURCES[str(path.resolve())]["sha256"],
                    "hessian_meta_path": rel(meta_path), "hessian_meta_sha256": SOURCES[str(meta_path.resolve())]["sha256"],
                    "H_tensor_sha256": obj["H_sha256"], "H_trace": h_trace,
                    "GPTQ_nsamples_documents": int(obj["nsamples"]), "search_tokens": n_tokens,
                    "kappa_from_raw_undamped_H_with_doc_and_token_normalization": kappa_h,
                    "kappa_from_original_search_inputs": kappa_x, "relative_gap_H_vs_input": rel_gap,
                }
                # The cached raw H is 2/n_documents * sum_doc X_doc^T X_doc.
                # H already averages over documents; dividing by 2*SEQ converts
                # its per-document token sum to per-token covariance.
                check(f"hessian_normalization_matches_input:{li}:{projection}:{split}", abs(rel_gap) <= 2e-4)
                del obj
    return stats


def score_and_weight_candidates(protocol: dict, lock: dict, docs: dict, inputs: dict, hstats: dict,
                                search_rows: list[dict]) -> tuple[list[dict], dict, dict]:
    alpha_grid = [float(x) for x in protocol["methods"]["M2_original_scalar"]["alpha_grid"]]
    # v24's dense grid definition is reconstructed only for stable artifact indices;
    # candidate metrics remain restricted to the exact original 10 alphas.
    dense_grid = sorted(set(alpha_grid + [0.1 + 0.9 * j / 37 for j in range(1, 37)]))
    search_map = {}
    selected_by_unit = defaultdict(list)
    for r in search_rows:
        if r.get("candidate_method") != "scalar_dense_bank" or r.get("is_original_10", "").lower() != "true":
            continue
        unit = r["unit_id"]
        a = float(r["alpha"])
        search_map[(unit, alpha_key(a))] = r
        if r.get("selected_by_original_10", "").lower() == "true":
            selected_by_unit[unit].append(a)

    model_sha_expected = protocol["frozen_source_sha256"][f".hf_cache/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/{REV}/model.safetensors"]
    model_sha = add_source(MODEL, "pristine Qwen checkpoint tensor source", model_sha_expected)
    pristine = {}
    checkpoint_dtypes = {}
    with safe_open(str(MODEL), framework="pt", device="cpu") as f:
        for li in LAYERS:
            for projection in PROJECTIONS:
                key = f"model.layers.{li}.mlp.{projection}.weight"
                source_w = f.get_tensor(key)
                checkpoint_dtypes[(li, projection)] = str(source_w.dtype)
                # v24 loads the checkpoint in the frozen FP16 model dtype before GPTQ.
                w = source_w.half().contiguous()
                check(f"pristine_dtype:{li}:{projection}", w.dtype == torch.float16)
                pristine[(li, projection)] = w
    candidate_rows = []
    unit_info = {}
    expected_count = 0
    for li in LAYERS:
        for projection in PROJECTIONS:
            w = pristine[(li, projection)]
            w_hash = sha_tensor(w)
            check(f"pristine_shape:{li}:{projection}", w.shape[0] > max(ROW_IDS[projection]) and w.shape[1] == inputs[(li, projection, "S0")]["input_dim"])
            for split in ("S0", "S1"):
                unit_ids = [f"L{li}_{projection}_R{row}_{split}" for row in ROW_IDS[projection]]
                for unit in unit_ids:
                    unit_info[unit] = {"layer": li, "projection": projection,
                                       "row": int(unit.split("_R")[1].split("_")[0]), "split_id": split}
                for alpha in alpha_grid:
                    expected_count += len(ROW_IDS[projection])
                    gidx = min(range(len(dense_grid)), key=lambda i: abs(dense_grid[i] - alpha))
                    check(f"dense_index:{li}:{projection}:{split}:{alpha}", abs(dense_grid[gidx] - alpha) <= 1e-12)
                    stem = f"L{li}_{projection}_{split}_a{gidx:02d}"
                    meta_path = V24 / "raw" / "full_candidates" / f"{stem}.json"
                    tensor_path = meta_path.with_suffix(".pt")
                    check(f"candidate_pair_exists:{stem}", meta_path.exists() and tensor_path.exists())
                    meta = read_json(meta_path, "v24 scalar10 candidate metadata")
                    obj = torch.load(tensor_path, map_location="cpu", weights_only=True)
                    tensor_file_sha = add_source(tensor_path, "v24 scalar10 candidate tensors")
                    check(f"candidate_identity:{stem}", meta["layer"] == li and meta["projection"] == projection and meta["split"] == split and float(meta["alpha"]) == alpha)
                    check(f"candidate_rows:{stem}", meta["rows"] == ROW_IDS[projection])
                    check(f"candidate_search_docs:{stem}", meta["search_doc_indices"] == split_positions(protocol, split, "search"))
                    check(f"candidate_manifest:{stem}", meta["e2_manifest_sha256"] == lock["e2_document_manifest_sha256"])
                    check(f"candidate_pristine_hash:{stem}", meta["pristine_weight_sha256"] == w_hash)
                    check(f"candidate_hessian_hash:{stem}", meta["hessian_sha256"] == hstats[(li, projection, split)]["H_tensor_sha256"])
                    expected_shapes = {"weight_fp16": (len(ROW_IDS[projection]), w.shape[1]),
                                       "scales": (len(ROW_IDS[projection]), w.shape[1] // 128),
                                       "zeros": (len(ROW_IDS[projection]), w.shape[1] // 128),
                                       "g_idx": (w.shape[1],)}
                    expected_dtypes = {"weight_fp16": torch.float16, "scales": torch.float32,
                                       "zeros": torch.float32, "g_idx": torch.int32}
                    for field in expected_shapes:
                        check(f"candidate_shape:{stem}:{field}", tuple(obj[field].shape) == expected_shapes[field])
                        check(f"candidate_dtype:{stem}:{field}", obj[field].dtype == expected_dtypes[field])
                        tensor_digest = sha_tensor(obj[field])
                        check(f"candidate_tensor_hash:{stem}:{field}", tensor_digest == meta[{"weight_fp16": "weight_sha256", "scales": "scales_sha256", "zeros": "zeros_sha256", "g_idx": "g_idx_sha256"}[field]])
                    check(f"candidate_finite:{stem}", all(bool(torch.isfinite(obj[k]).all().item()) for k in ("weight_fp16", "scales", "zeros")))
                    check(f"candidate_scales_positive:{stem}", bool((obj["scales"] > 0).all().item()))
                    check(f"candidate_zero8:{stem}", bool((obj["zeros"] == 8).all().item()))
                    expected_gidx = torch.arange(w.shape[1], dtype=torch.int32) // 128
                    check(f"candidate_gidx_order:{stem}", torch.equal(obj["g_idx"], expected_gidx))
                    for j, row in enumerate(ROW_IDS[projection]):
                        unit = f"L{li}_{projection}_R{row}_{split}"
                        sr = search_map.get((unit, alpha_key(alpha)))
                        check(f"search_score_row_exists:{unit}:{alpha}", sr is not None)
                        search_mse = float(sr["search_mse"])
                        metadata_mse = float(meta["search_mse_by_row"][str(row)])
                        check(f"search_score_matches_candidate_meta:{unit}:{alpha}", check_close(search_mse, metadata_mse))
                        err = obj["weight_fp16"][j].float() - w[row].float()
                        norm2 = float(err.double().square().sum().item())
                        candidate_rows.append({
                            "unit_id": unit, "layer": li, "projection": projection, "row": row, "split_id": split,
                            "alpha": alpha, "search_output_mse": search_mse,
                            "search_output_sse": search_mse * len(split_positions(protocol, split, "search")) * SEQ,
                            "search_output_mse_unit": "mean squared output residual per search token for this row",
                            "search_output_sse_unit": "sum squared output residual across search tokens for this row",
                            "weight_squared_norm_e2": norm2,
                            "weight_mse_e2_over_input_dim": norm2 / int(w.shape[1]),
                            "weight_error_unit": "sum of squared FP32(Q_FP16-W_FP16) entries; weight units squared",
                            "pristine_weight_dtype_used_for_error": str(w.dtype),
                            "pristine_checkpoint_storage_dtype": checkpoint_dtypes[(li, projection)],
                            "input_dim": int(w.shape[1]), "candidate_weight_tensor_dtype": str(obj["weight_fp16"].dtype),
                            "scale_dtype": str(obj["scales"].dtype), "zero_dtype": str(obj["zeros"].dtype),
                            "g_idx_dtype": str(obj["g_idx"].dtype),
                            "candidate_q_four_rows_tensor_sha256": meta["weight_sha256"],
                            "candidate_q_row_tensor_sha256": sha_tensor(obj["weight_fp16"][j]),
                            "candidate_scales_tensor_sha256": meta["scales_sha256"],
                            "candidate_zeros_tensor_sha256": meta["zeros_sha256"],
                            "candidate_g_idx_tensor_sha256": meta["g_idx_sha256"],
                            "candidate_tensor_file_sha256": tensor_file_sha,
                            "candidate_metadata_file_sha256": SOURCES[str(meta_path.resolve())]["sha256"],
                            "pristine_weight_tensor_sha256": w_hash,
                            "pristine_checkpoint_file_sha256": model_sha,
                            "input_file_sha256": inputs[(li, projection, split)]["input_file_sha256"],
                            "search_input_bytes_sha256": inputs[(li, projection, split)]["search_input_bytes_sha256"],
                            "search_doc_indices_sha256": inputs[(li, projection, split)]["search_doc_indices_sha256"],
                            "hessian_tensor_sha256": hstats[(li, projection, split)]["H_tensor_sha256"],
                            "protocol_sha256": SOURCES[str((V24 / "PROTOCOL.json").resolve())]["sha256"],
                            "selection_lock_sha256": lock["search_selection_sha256"],
                            "search_results_file_sha256": SOURCES[str((V24 / "SEARCH_RESULTS.csv").resolve())]["sha256"],
                            "source_candidate_stem": stem,
                        })
                    del obj
    check("complete_scalar10_row_candidate_count", expected_count == len(candidate_rows) == 480)
    check("complete_scalar10_search_map", len(search_map) == 480)
    direct = {}
    for unit, vals in selected_by_unit.items():
        check(f"one_original_selection:{unit}", len(vals) == 1)
        direct[unit] = vals[0]
    check("48_original_search_selections", len(direct) == 48)
    by_unit = defaultdict(list)
    for row in candidate_rows:
        by_unit[row["unit_id"]].append(row)
    for unit, bank in by_unit.items():
        selected = min(bank, key=lambda r: (r["search_output_mse"], -float(r["alpha"])))
        check(f"direct_choice_reproduced:{unit}", float(selected["alpha"]) == direct[unit])
        for row in bank:
            row["alpha1_search_output_mse"] = next(r["search_output_mse"] for r in bank if float(r["alpha"]) == 1.0)
            row["alpha1_weight_squared_norm_e2"] = next(r["weight_squared_norm_e2"] for r in bank if float(r["alpha"]) == 1.0)
            row["original_search_selected_alpha"] = direct[unit]
            row["is_original_search_selected"] = float(row["alpha"]) == direct[unit]
    return candidate_rows, by_unit, unit_info


def decimal_value(x) -> Decimal:
    return Decimal(str(float(x)))


def choose_at_rho(bank: list[dict], rho: Decimal, direct_alpha: float) -> dict:
    if rho == 0:
        return next(r for r in bank if float(r["alpha"]) == direct_alpha)
    scores = [(decimal_value(r["search_output_mse"]) + rho * decimal_value(r["weight_squared_norm_e2"]), r) for r in bank]
    best = min(s for s, _ in scores)
    tied = [r for s, r in scores if s == best]
    return max(tied, key=lambda r: float(r["alpha"]))


def lower_path(bank: list[dict], direct_alpha: float) -> tuple[list[dict], list[dict]]:
    with localcontext() as ctx:
        ctx.prec = 60
        roots = set()
        for i in range(len(bank)):
            for j in range(i + 1, len(bank)):
                bi, bj = decimal_value(bank[i]["search_output_mse"]), decimal_value(bank[j]["search_output_mse"])
                mi, mj = decimal_value(bank[i]["weight_squared_norm_e2"]), decimal_value(bank[j]["weight_squared_norm_e2"])
                if mi == mj:
                    continue
                x = (bj - bi) / (mi - mj)
                if x > 0:
                    roots.add(x)
        roots = sorted(roots)
        raw_segments = []
        for i in range(len(roots) + 1):
            lo = roots[i - 1] if i else Decimal(0)
            hi = roots[i] if i < len(roots) else None
            if hi is None:
                test = lo + max(Decimal(1), abs(lo) + Decimal(1))
            else:
                test = (lo + hi) / 2
            chosen = choose_at_rho(bank, test, direct_alpha)
            raw_segments.append({"lo": lo, "hi": hi, "test": test, "chosen": chosen})
        segments = []
        boundaries = []
        for s in raw_segments:
            if segments and float(segments[-1]["alpha"]) == float(s["chosen"]["alpha"]):
                segments[-1]["rho_upper"] = None if s["hi"] is None else str(s["hi"])
                continue
            if segments:
                x = s["lo"]
                left = segments[-1]["alpha"]
                right = float(s["chosen"]["alpha"])
                at = choose_at_rho(bank, x, direct_alpha)
                tied = []
                target_score = decimal_value(at["search_output_mse"]) + x * decimal_value(at["weight_squared_norm_e2"])
                for r in bank:
                    score = decimal_value(r["search_output_mse"]) + x * decimal_value(r["weight_squared_norm_e2"])
                    if score == target_score:
                        tied.append(float(r["alpha"]))
                boundaries.append({"rho": str(x), "left_alpha": left, "right_alpha": right,
                                   "boundary_alpha_by_larger_alpha_tie_rule": max(tied) if tied else float(at["alpha"]),
                                   "tied_alphas": sorted(set(tied))})
            segments.append({"rho_lower": str(s["lo"]), "rho_upper": None if s["hi"] is None else str(s["hi"]),
                             "rho_test": str(s["test"]), "alpha": float(s["chosen"]["alpha"])})
        return segments, boundaries


def create_search_only_paths(by_unit: dict, direct: dict, inputs: dict) -> tuple[list[dict], list[dict], list[dict], list[dict], list[dict]]:
    segments_all, boundaries_all, zero_rows = [], [], []
    all_bounds = set()
    for unit, bank in sorted(by_unit.items()):
        info = unit_info_global[unit]
        segs, bounds = lower_path(bank, direct[unit])
        first_positive = next((s["alpha"] for s in segs if Decimal(s["rho_lower"]) == 0), direct[unit])
        zero_rows.append({"unit_id": unit, "original_search_alpha_at_rho0": direct[unit],
                          "right_limit_alpha_for_rho_positive": first_positive,
                          "differs_at_zero_tie": first_positive != direct[unit]})
        for idx, seg in enumerate(segs):
            row = {"unit_id": unit, "layer": info["layer"], "projection": info["projection"], "row": info["row"],
                   "split_id": info["split_id"], "segment_index": idx, **seg,
                   "selection_rule": "min(search_output_mse + rho*weight_squared_norm); equal positive-rho scores choose larger alpha"}
            segments_all.append(row)
        for b in bounds:
            all_bounds.add(Decimal(b["rho"]))
            boundaries_all.append({"unit_id": unit, "layer": info["layer"], "projection": info["projection"],
                                   "row": info["row"], "split_id": info["split_id"], **b})

    global_bounds = sorted(all_bounds)
    intervals = []
    for i in range(len(global_bounds) + 1):
        lo = global_bounds[i - 1] if i else Decimal(0)
        hi = global_bounds[i] if i < len(global_bounds) else None
        test = lo + max(Decimal(1), abs(lo) + Decimal(1)) if hi is None else (lo + hi) / 2
        intervals.append({"path_interval_id": f"P{i:04d}", "rho_lower_exclusive": str(lo),
                          "rho_upper_exclusive": None if hi is None else str(hi), "rho_test": str(test)})

    tau_choices = []
    for tau in TAUS:
        for unit, bank in sorted(by_unit.items()):
            info = unit_info_global[unit]
            kappa = inputs[(info["layer"], info["projection"], info["split_id"])]["kappa_trace_over_input_dim"]
            rho = Decimal(0) if tau == 0 else Decimal(str(tau)) * decimal_value(kappa)
            chosen = choose_at_rho(bank, rho, direct[unit])
            tau_choices.append({"selector": f"tau={tau:g}", "tau": tau, "unit_id": unit,
                                "layer": info["layer"], "projection": info["projection"], "row": info["row"],
                                "split_id": info["split_id"], "kappa": kappa, "rho": str(rho),
                                "alpha": float(chosen["alpha"]), "search_output_mse": chosen["search_output_mse"],
                                "weight_squared_norm_e2": chosen["weight_squared_norm_e2"],
                                "selection_stage": "search_only_before_held_read"})

    path_choices = []
    for interval in intervals:
        rho = Decimal(interval["rho_test"])
        for unit, bank in sorted(by_unit.items()):
            info = unit_info_global[unit]
            chosen = choose_at_rho(bank, rho, direct[unit])
            path_choices.append({"selector": interval["path_interval_id"], **interval,
                                 "unit_id": unit, "layer": info["layer"], "projection": info["projection"],
                                 "row": info["row"], "split_id": info["split_id"], "alpha": float(chosen["alpha"]),
                                 "search_output_mse": chosen["search_output_mse"],
                                 "weight_squared_norm_e2": chosen["weight_squared_norm_e2"],
                                 "selection_stage": "search_only_before_held_read"})
    boundary_choices = []
    for bi, bound in enumerate(global_bounds):
        for unit, bank in sorted(by_unit.items()):
            info = unit_info_global[unit]
            chosen = choose_at_rho(bank, bound, direct[unit])
            boundary_choices.append({"selector": f"B{bi:04d}", "rho": str(bound), "unit_id": unit,
                                     "layer": info["layer"], "projection": info["projection"], "row": info["row"],
                                     "split_id": info["split_id"], "alpha": float(chosen["alpha"]),
                                     "search_output_mse": chosen["search_output_mse"],
                                     "weight_squared_norm_e2": chosen["weight_squared_norm_e2"],
                                     "selection_stage": "search_only_before_held_read"})
    return segments_all, boundaries_all, zero_rows, tau_choices, path_choices, boundary_choices, intervals


def read_held_after_path(protocol: dict, lock: dict, by_unit: dict, direct: dict, search_rows: list[dict]) -> tuple[list[dict], dict]:
    held_csv = read_csv(V24 / "HELD_RESULTS.csv", "v24 held summary and historical epsilon")
    held_method = {(r["unit_id"], r["method"]): r for r in held_csv}
    held_rows = []
    for unit, bank in sorted(by_unit.items()):
        info = unit_info_global[unit]
        for alpha in [float(r["alpha"]) for r in bank]:
            cand = next(r for r in bank if float(r["alpha"]) == alpha)
            stem = cand["source_candidate_stem"]
            score_path = V24 / "raw" / "held_full_scores" / f"{stem}.json"
            score = read_json(score_path, "v24 pre-existing held score sidecar")
            candidate_meta_path = V24 / "raw" / "full_candidates" / f"{stem}.json"
            candidate_meta = json.loads(candidate_meta_path.read_text(encoding="utf-8-sig"))
            add_source(candidate_meta_path, "v24 scalar10 candidate metadata (held linkage)")
            # The score sidecar hashes the exact dequantized FP16 candidate tensor supplied to the held scorer.
            check(f"held_lock:{unit}:{alpha}", score["search_selection_sha256"] == lock["search_selection_sha256"])
            check(f"held_docs:{unit}:{alpha}", score["held_doc_indices"] == split_positions(protocol, info["split_id"], "held"))
            check(f"held_candidate_q_identity:{unit}:{alpha}", score["candidate_weight_sha256"] == candidate_meta["weight_sha256"] == cand["candidate_q_four_rows_tensor_sha256"])
            held_mse = float(score["held_mse_by_row"][str(info["row"])])
            srow = next(r for r in search_rows if r["unit_id"] == unit and alpha_key(r["alpha"]) == alpha_key(alpha)
                        and r.get("candidate_method") == "scalar_dense_bank" and r.get("is_original_10", "").lower() == "true")
            search_mse = float(srow["search_mse"])
            nheld_docs = len(split_positions(protocol, info["split_id"], "held"))
            held_rows.append({**cand, "held_output_mse": held_mse,
                              "held_output_sse": held_mse * nheld_docs * SEQ,
                              "held_output_mse_unit": "mean squared output residual per held token for this row",
                              "held_output_sse_unit": "sum squared output residual across held tokens for this row",
                              "held_score_candidate_q_sha256": score["candidate_weight_sha256"],
                              "held_score_sidecar_sha256": SOURCES[str(score_path.resolve())]["sha256"],
                              "held_doc_indices_sha256": sha_json(score["held_doc_indices"]),
                              "search_mse_sidecar_csv_match": check_close(search_mse, cand["search_output_mse"])})
        epsrow = held_method[(unit, "M2_original_scalar10")]
        nativerow = held_method[(unit, "M1_native_alpha1")]
        direct_candidate = next(r for r in held_rows if r["unit_id"] == unit and float(r["alpha"]) == direct[unit])
        alpha1_candidate = next(r for r in held_rows if r["unit_id"] == unit and float(r["alpha"]) == 1.0)
        check(f"held_selected_alpha_matches_csv:{unit}", float(epsrow["alpha_A"]) == direct[unit])
        check(f"held_selected_mse_matches_summary:{unit}", check_close(float(epsrow["held_local_mse"]), direct_candidate["held_output_mse"]))
        check(f"held_alpha1_mse_matches_native:{unit}", check_close(float(nativerow["held_local_mse"]), alpha1_candidate["held_output_mse"]))
        eps = float(epsrow["unit_tie_epsilon_mse"])
        check(f"historical_held_doc_count:{unit}", int(epsrow["held_doc_count"]) == nheld_docs == 16)
        for row in held_rows:
            if row["unit_id"] == unit:
                row["alpha1_held_output_mse"] = alpha1_candidate["held_output_mse"]
                row["alpha1_held_output_sse"] = alpha1_candidate["held_output_sse"]
                row["original_search_selected_alpha"] = direct[unit]
                row["original_search_selected_held_output_mse"] = direct_candidate["held_output_mse"]
                row["historical_epsilon_mse"] = eps
                row["historical_epsilon_sse"] = eps * nheld_docs * SEQ
                row["historical_epsilon_source"] = "v24 HELD_RESULTS.csv:M2_original_scalar10.unit_tie_epsilon_mse"
    # v25 independently audited the same v24 scalar10 bank. Read this held-bearing
    # cross-check only after the v30 search-only paths have already been written.
    v25_units = read_csv(V25 / "DIRECT_UNIT_RESULTS.csv", "v25 scalar10 direct-audit cross-check")
    v25_map = {r["unit_id"]: r for r in v25_units}
    check("v25_direct_unit_count", len(v25_map) == 48)
    for unit in sorted(by_unit):
        v25 = v25_map[unit]
        selected = next(r for r in held_rows if r["unit_id"] == unit and float(r["alpha"]) == direct[unit])
        check(f"v25_scalar10_alpha:{unit}", float(v25["alpha10"]) == direct[unit])
        check(f"v25_scalar10_search:{unit}", check_close(float(v25["scalar10_search_mse"]), selected["search_output_mse"]))
        check(f"v25_scalar10_held:{unit}", check_close(float(v25["scalar10_held_mse"]), selected["held_output_mse"]))
        check(f"v25_scalar10_weight_mse:{unit}", check_close(float(v25["scalar10_weight_mse"]), selected["weight_mse_e2_over_input_dim"], rel_tol=1e-5, abs_tol=1e-15))
    algorithm_audit = read_json(V25 / "ALGORITHM_AUDIT.json", "v25 candidate-generation audit")
    check("v25_candidate_semantics_audit", algorithm_audit.get("candidate_semantics", "").startswith("Pristine W and fresh GPTQ state"))
    return held_rows, held_method


def pareto_tables(held_rows: list[dict]) -> tuple[list[dict], list[dict]]:
    groups = defaultdict(list)
    for r in held_rows:
        groups[r["unit_id"]].append(r)
    table, summaries = [], []
    for unit, bank in sorted(groups.items()):
        dominated = {}
        for a in bank:
            dom = [b for b in bank if b is not a and
                   b["search_output_mse"] <= a["search_output_mse"] and
                   b["weight_squared_norm_e2"] <= a["weight_squared_norm_e2"] and
                   (b["search_output_mse"] < a["search_output_mse"] or b["weight_squared_norm_e2"] < a["weight_squared_norm_e2"])]
            dominated[float(a["alpha"])] = sorted(float(x["alpha"]) for x in dom)
        info = unit_info_global[unit]
        for r in bank:
            ds = dominated[float(r["alpha"])]
            table.append({"unit_id": unit, "layer": info["layer"], "projection": info["projection"], "row": info["row"],
                          "split_id": info["split_id"], "alpha": r["alpha"],
                          "search_output_mse": r["search_output_mse"], "weight_squared_norm_e2": r["weight_squared_norm_e2"],
                          "is_dominated": bool(ds), "dominator_alphas": ds,
                          "dominance_definition": "another original-10 candidate has <= search MSE and <= weight squared norm, with at least one strict"})
        summaries.append({"unit_id": unit, "layer": info["layer"], "projection": info["projection"], "row": info["row"],
                          "split_id": info["split_id"], "candidate_count": len(bank),
                          "dominated_candidate_count": sum(bool(v) for v in dominated.values()),
                          "pareto_candidate_count": sum(not bool(v) for v in dominated.values()),
                          "pareto_alphas": [a for a, ds in sorted(dominated.items()) if not ds]})
    return table, summaries


def mechanism(held_rows: list[dict], direct: dict) -> tuple[list[dict], list[dict]]:
    byunit = defaultdict(list)
    for r in held_rows:
        byunit[r["unit_id"]].append(r)
    rows, summaries = [], []
    for unit, bank in sorted(byunit.items()):
        alpha_s = direct[unit]
        selected = next(r for r in bank if float(r["alpha"]) == alpha_s)
        native = next(r for r in bank if float(r["alpha"]) == 1.0)
        search_improved = selected["search_output_mse"] < native["search_output_mse"]
        weight_increased = selected["weight_squared_norm_e2"] > native["weight_squared_norm_e2"]
        held_diff = selected["held_output_mse"] - native["held_output_mse"]
        eps = native["historical_epsilon_mse"]
        held_outcome = "loss" if held_diff > eps else "win" if held_diff < -eps else "tie"
        if search_improved and weight_increased and held_outcome == "loss":
            cls = 1
        elif search_improved and weight_increased and held_outcome == "win":
            cls = 2
        elif search_improved and not weight_increased and held_outcome == "loss":
            cls = 3
        else:
            cls = 4
        info = unit_info_global[unit]
        rows.append({"unit_id": unit, "layer": info["layer"], "projection": info["projection"], "row": info["row"],
                     "split_id": info["split_id"], "alpha1": 1.0, "original_search_selected_alpha": alpha_s,
                     "search_output_mse_alpha1": native["search_output_mse"],
                     "search_output_mse_selected": selected["search_output_mse"],
                     "search_output_mse_change_selected_minus_alpha1": selected["search_output_mse"] - native["search_output_mse"],
                     "search_strictly_improved": search_improved,
                     "weight_squared_norm_alpha1": native["weight_squared_norm_e2"],
                     "weight_squared_norm_selected": selected["weight_squared_norm_e2"],
                     "weight_squared_norm_change_selected_minus_alpha1": selected["weight_squared_norm_e2"] - native["weight_squared_norm_e2"],
                     "weight_error_increased": weight_increased,
                     "held_output_mse_alpha1": native["held_output_mse"],
                     "held_output_mse_selected": selected["held_output_mse"],
                     "held_output_mse_change_selected_minus_alpha1": held_diff,
                     "historical_epsilon_mse": eps, "held_outcome_vs_alpha1": held_outcome,
                     "mechanism_class": cls})
    scopes = make_scopes(rows)
    for scope, subset in scopes:
        count = Counter(r["mechanism_class"] for r in subset)
        held_counts = Counter(r["held_outcome_vs_alpha1"] for r in subset)
        search_improved = [r for r in subset if r["search_strictly_improved"]]
        summaries.append({"scope": scope, "row_split_units": len(subset),
                          "class1_search_improve_weight_increase_held_loss": count[1],
                          "class2_search_improve_weight_increase_held_win": count[2],
                          "class3_search_improve_weight_not_increase_held_loss": count[3],
                          "class4_other": count[4],
                          "search_strictly_improved_count": len(search_improved),
                          "search_improved_weight_norm_increased_count": sum(r["weight_error_increased"] for r in search_improved),
                          "search_improved_norm_not_increased_held_win_count": sum(not r["weight_error_increased"] and r["held_outcome_vs_alpha1"] == "win" for r in search_improved),
                          "search_improved_norm_not_increased_held_tie_count": sum(not r["weight_error_increased"] and r["held_outcome_vs_alpha1"] == "tie" for r in search_improved),
                          "search_improved_norm_not_increased_held_loss_count": sum(not r["weight_error_increased"] and r["held_outcome_vs_alpha1"] == "loss" for r in search_improved),
                          "held_win_tie_loss_vs_alpha1": {k: held_counts[k] for k in ("win", "tie", "loss")},
                          "unit_ids_by_class": {str(k): [r["unit_id"] for r in subset if r["mechanism_class"] == k] for k in range(1, 5)}})
    return rows, summaries


def make_scopes(rows: list[dict]) -> list[tuple[str, list[dict]]]:
    scopes = [("ALL", list(rows))]
    for split in ("S0", "S1"):
        scopes.append((split, [r for r in rows if r["split_id"] == split]))
    for li in LAYERS:
        lr = [r for r in rows if r["layer"] == li]
        scopes.append((f"L{li}", lr))
        for proj in PROJECTIONS:
            pr = [r for r in lr if r["projection"] == proj]
            scopes.append((f"L{li}/{proj}", pr))
            for split in ("S0", "S1"):
                scopes.append((f"L{li}/{proj}/{split}", [r for r in pr if r["split_id"] == split]))
    return scopes


def selection_summary(selector: str, choices: dict[str, dict], held_by_unit: dict[str, list[dict]], direct: dict,
                      scopes) -> list[dict]:
    rows = []
    candidates = {u: {float(r["alpha"]): r for r in b} for u, b in held_by_unit.items()}
    unit_base = {}
    for unit, bank in candidates.items():
        direct_row = bank[direct[unit]]
        alpha1_row = bank[1.0]
        unit_base[unit] = {"vs_original_search": direct_row, "vs_alpha1": alpha1_row}
    for scope, units in scopes:
        for comparison in ("vs_original_search", "vs_alpha1"):
            info = []
            for unit in units:
                chosen_alpha = float(choices[unit]["alpha"])
                new = candidates[unit][chosen_alpha]
                base = unit_base[unit][comparison]
                info.append((unit, base, new))
            if not info:
                continue
            held_base_sse = math.fsum(b["held_output_sse"] for _, b, _ in info)
            held_new_sse = math.fsum(n["held_output_sse"] for _, _, n in info)
            eps_sse = math.fsum(n["historical_epsilon_sse"] for _, _, n in info)
            wins = ties = losses = 0
            gains = []
            per_unit_savings = []
            severe = 0
            for unit, base, new in info:
                d_mse = base["held_output_mse"] - new["held_output_mse"]
                eps = new["historical_epsilon_mse"]
                if d_mse > eps:
                    wins += 1
                elif d_mse < -eps:
                    losses += 1
                else:
                    ties += 1
                gains.append(d_mse / base["held_output_mse"] if base["held_output_mse"] else 0.0)
                per_unit_savings.append((base["held_output_sse"] - new["held_output_sse"], unit))
                alpha1 = candidates[unit][1.0]
                if (alpha1["held_output_mse"] - new["held_output_mse"]) / alpha1["held_output_mse"] < -0.10:
                    severe += 1
            worst_gain, worst_unit = min((g, u) for (u, b, n), g in zip(info, gains))
            gross_positive = math.fsum(max(0.0, x) for x, _ in per_unit_savings)
            ranked = sorted(((x, u) for x, u in per_unit_savings if x > 0), reverse=True)
            top1 = math.fsum(x for x, _ in ranked[:1])
            top3 = math.fsum(x for x, _ in ranked[:3])
            search_base_sse = math.fsum(unit_base[u][comparison]["search_output_sse"] for u, _, _ in info)
            search_new_sse = math.fsum(candidates[u][float(choices[u]["alpha"])]["search_output_sse"] for u, _, _ in info)
            norm_base = math.fsum(unit_base[u][comparison]["weight_squared_norm_e2"] for u, _, _ in info)
            norm_new = math.fsum(candidates[u][float(choices[u]["alpha"])]["weight_squared_norm_e2"] for u, _, _ in info)
            total_dims = sum(int(candidates[u][float(choices[u]["alpha"])]["input_dim"]) for u, _, _ in info)
            rows.append({"selector": selector, "scope": scope, "comparison": comparison, "row_split_units": len(info),
                         "held_baseline_sse": held_base_sse, "held_selected_sse": held_new_sse,
                         "held_sse_improvement_baseline_minus_selected": held_base_sse - held_new_sse,
                         "held_pooled_relative_improvement": (held_base_sse - held_new_sse) / held_base_sse if held_base_sse else 0.0,
                         "held_baseline_pooled_mse": held_base_sse / (len(info) * 16 * SEQ),
                         "held_selected_pooled_mse": held_new_sse / (len(info) * 16 * SEQ),
                         "historical_epsilon_sse_sum": eps_sse,
                         "pooled_improvement_beyond_historical_epsilon": held_base_sse - held_new_sse > eps_sse,
                         "wins": wins, "ties": ties, "losses": losses,
                         "median_unit_relative_held_improvement": statistics.median(gains),
                         "worst_unit": worst_unit, "worst_unit_relative_held_improvement": worst_gain,
                         "severe_held_degradations_vs_alpha1_below_minus_10pct": severe,
                         "search_baseline_sse": search_base_sse, "search_selected_sse": search_new_sse,
                         "search_sse_change_selected_minus_baseline": search_new_sse - search_base_sse,
                         "search_sse_change_fraction": (search_new_sse - search_base_sse) / search_base_sse if search_base_sse else 0.0,
                         "weight_squared_norm_baseline_sum": norm_base, "weight_squared_norm_selected_sum": norm_new,
                         "weight_squared_norm_reduction_baseline_minus_selected": norm_base - norm_new,
                         "weight_squared_norm_reduction_fraction": (norm_base - norm_new) / norm_base if norm_base else 0.0,
                         "pooled_weight_mse_baseline_norm2_over_total_weights": norm_base / total_dims,
                         "pooled_weight_mse_selected_norm2_over_total_weights": norm_new / total_dims,
                         "gross_positive_held_savings_sse": gross_positive,
                         "top1_share_of_positive_held_savings": top1 / gross_positive if gross_positive else 0.0,
                         "top3_share_of_positive_held_savings": top3 / gross_positive if gross_positive else 0.0,
                         "top1_share_of_net_held_improvement": top1 / (held_base_sse - held_new_sse) if held_base_sse != held_new_sse else None,
                         "top3_share_of_net_held_improvement": top3 / (held_base_sse - held_new_sse) if held_base_sse != held_new_sse else None,
                         "top_positive_unit_ids": [u for _, u in ranked[:3]],
                         "interpretation": "Held is retrospective only; no rho is selected from this table."})
    return rows


def choices_map(rows: list[dict]) -> dict[str, dict]:
    return {r["unit_id"]: r for r in rows}


def main() -> None:
    global PREFIX_MANIFEST, DOWN_MANIFEST, unit_info_global
    owned = {"analyze_v30.py", "README.md", "SEARCH_CANDIDATE_TABLE.csv", "SELECTION_PATH.csv",
             "SELECTION_PATH_BOUNDARIES.csv", "RHO_ZERO_TIE_AUDIT.csv", "TAU_SELECTIONS_SEARCH_ONLY.csv",
             "FULL_PATH_INTERVAL_CHOICES_SEARCH_ONLY.csv", "FULL_PATH_BOUNDARY_CHOICES_SEARCH_ONLY.csv",
             "FULL_PATH_INTERVAL_INDEX.csv", "CANDIDATE_ROW_TABLE.csv", "PARETO_DOMINANCE.csv",
             "PARETO_UNIT_SUMMARY.csv", "MECHANISM_RESULTS.csv", "MECHANISM_SUMMARY.csv",
             "TAU_SELECTIONS.csv", "REGULARIZATION_SUMMARY.csv", "FULL_PATH_INTERVAL_SUMMARY.csv",
             "FULL_PATH_BOUNDARY_SUMMARY.csv", "NEGATIVE_RESULTS.csv", "KAPPA_NORMALIZATION.csv",
             "PAIRED_SPLIT_OUTCOMES.csv", "SOURCE_MANIFEST.csv", "RUN_SUMMARY.json", "AUDIT_CHECKS.json", "PHASE_TIMING.json",
             "REPORT.md", "RUN_FAILURE.json"}
    if OUT.exists() and any(p.name not in owned for p in OUT.iterdir()):
        raise RuntimeError(f"v30 directory contains files not owned by this analysis: {OUT}")
    (OUT / "README.md").write_text(
        "# v30 weight-error regularization diagnosis\n\n"
        "Standalone artifact-only analysis of the original v24 scalar10 candidates. "
        "The script reads frozen v23/v24/v25 files and writes only to this directory.\n",
        encoding="utf-8")

    protocol = read_json(V24 / "PROTOCOL.json", "v24 frozen protocol")
    lock = read_json(V24 / "SEARCH_SELECTION_LOCK.json", "v24 search selection lock")
    docs = read_json(V24 / "E2_DOCUMENT_MANIFEST.json", "v24 frozen document/input manifest")
    PREFIX_MANIFEST = read_json(V23 / "cache" / "PREFIX_CACHE_MANIFEST.json", "v23 frozen up-input cache manifest")
    DOWN_MANIFEST = read_json(V24 / "cache" / "DOWN_INPUT_CACHE_MANIFEST.json", "v24 frozen down-input cache manifest")
    check("v24_frozen_protocol", protocol["status"] == "FROZEN_BEFORE_E2_SEARCH")
    check("v24_selection_locked_before_held", lock["status"] == "LOCKED_BEFORE_HELD_SCORING" and lock["held_indices_not_read_for_selection"] is True)
    check("v24_document_manifest_hash", sha_file(V24 / "E2_DOCUMENT_MANIFEST.json") == lock["e2_document_manifest_sha256"])
    check("v24_protocol_lock_hash", sha_file(V24 / "PROTOCOL.json") == lock["protocol_sha256"])
    check("v24_search_lock_hash", sha_file(V24 / "SEARCH_RESULTS.csv") == lock["search_results_sha256"])
    check("protocol_original_alpha10", protocol["methods"]["M2_original_scalar"]["alpha_grid"] == [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0])
    check("protocol_fixed_rows", all(protocol["targets"]["rows"][p] == ROW_IDS[p] for p in PROJECTIONS))
    check("protocol_targets", protocol["targets"]["layers"] == LAYERS and protocol["targets"]["projections"] == PROJECTIONS)
    phase("protocol_and_lock_read")

    search_rows = read_csv(V24 / "SEARCH_RESULTS.csv", "v24 original search scores")
    input_stats = read_and_hash_inputs(protocol)
    hstats = load_hessian_stats(protocol, input_stats)
    phase("search_inputs_and_undamped_statistics_verified")

    # The historical source and checkpoint are frozen read-only. Hash their bytes once.
    add_source(V24 / "run_e2_v24.py", "v24 original candidate generation and score source")
    gptq_py = ROOT / ".venv-gptq/Lib/site-packages/gptqmodel/quantization/gptq.py"
    add_source(gptq_py, "GPTQ process_batch normalization source")
    candidate_rows, by_unit, unit_info_global = score_and_weight_candidates(protocol, lock, docs, input_stats, hstats, search_rows)
    write_csv("SEARCH_CANDIDATE_TABLE.csv", candidate_rows)
    direct = {u: next(float(r["alpha"]) for r in bank if r.get("is_original_search_selected")) for u, bank in by_unit.items()}
    segments, boundaries, zero_rows, tau_choices, path_choices, boundary_choices, intervals = create_search_only_paths(
        by_unit, direct, input_stats)
    write_csv("SELECTION_PATH.csv", segments)
    write_csv("SELECTION_PATH_BOUNDARIES.csv", boundaries)
    write_csv("RHO_ZERO_TIE_AUDIT.csv", zero_rows)
    write_csv("TAU_SELECTIONS_SEARCH_ONLY.csv", tau_choices)
    write_csv("FULL_PATH_INTERVAL_CHOICES_SEARCH_ONLY.csv", path_choices)
    write_csv("FULL_PATH_BOUNDARY_CHOICES_SEARCH_ONLY.csv", boundary_choices)
    write_csv("FULL_PATH_INTERVAL_INDEX.csv", intervals)
    phase("search_only_path_built_and_written_before_held_read")

    held_rows, held_method = read_held_after_path(protocol, lock, by_unit, direct, search_rows)
    phase("historical_held_scores_read_after_search_path")
    # Attach search-stage selection values to the already frozen held outcomes.
    held_by_unit = defaultdict(list)
    for r in held_rows:
        held_by_unit[r["unit_id"]].append(r)
    enriched = []
    for r in held_rows:
        enriched.append(r)
    write_csv("CANDIDATE_ROW_TABLE.csv", enriched)

    pareto, pareto_unit = pareto_tables(held_rows)
    write_csv("PARETO_DOMINANCE.csv", pareto)
    write_csv("PARETO_UNIT_SUMMARY.csv", pareto_unit)
    mech_rows, mech_summary = mechanism(held_rows, direct)
    write_csv("MECHANISM_RESULTS.csv", mech_rows)
    write_csv("MECHANISM_SUMMARY.csv", mech_summary)

    scopes = [(name, [r["unit_id"] for r in subset]) for name, subset in make_scopes(mech_rows)]
    tau_choice_map = {}
    tau_choice_rows = []
    tau_by_unit = defaultdict(dict)
    for r in tau_choices:
        tau_by_unit[r["selector"]][r["unit_id"]] = r
    selection_summaries = []
    all_choices = {}
    for selector, m in tau_by_unit.items():
        all_choices[selector] = m
        selection_summaries.extend(selection_summary(selector, m, held_by_unit, direct, scopes))
        for unit, x in m.items():
            row = dict(x)
            candidate = next(r for r in held_by_unit[unit] if float(r["alpha"]) == float(x["alpha"]))
            row.update({"held_output_mse": candidate["held_output_mse"], "held_output_sse": candidate["held_output_sse"],
                        "historical_epsilon_mse": candidate["historical_epsilon_mse"],
                        "held_delta_vs_original_search_mse": candidate["held_output_mse"] - candidate["original_search_selected_held_output_mse"],
                        "held_delta_vs_alpha1_mse": candidate["held_output_mse"] - candidate["alpha1_held_output_mse"]})
            tau_choice_rows.append(row)
    write_csv("TAU_SELECTIONS.csv", tau_choice_rows)

    path_choice_by_interval = defaultdict(dict)
    for r in path_choices:
        path_choice_by_interval[r["selector"]][r["unit_id"]] = r
    path_interval_summaries = []
    for selector, m in path_choice_by_interval.items():
        path_interval_summaries.extend(selection_summary(selector, m, held_by_unit, direct, scopes))
    boundary_choice_by_id = defaultdict(dict)
    for r in boundary_choices:
        boundary_choice_by_id[r["selector"]][r["unit_id"]] = r
    path_boundary_summaries = []
    for selector, m in boundary_choice_by_id.items():
        path_boundary_summaries.extend(selection_summary(selector, m, held_by_unit, direct, scopes))
    write_csv("REGULARIZATION_SUMMARY.csv", selection_summaries)
    write_csv("FULL_PATH_INTERVAL_SUMMARY.csv", path_interval_summaries)
    write_csv("FULL_PATH_BOUNDARY_SUMMARY.csv", path_boundary_summaries)

    # Complete negative row outcomes for all fixed tau values and each interval/boundary.
    negative_rows = []
    candidate_lookup = {u: {float(r["alpha"]): r for r in b} for u, b in held_by_unit.items()}
    selector_maps = {**tau_by_unit, **path_choice_by_interval, **boundary_choice_by_id}
    for selector, selected in selector_maps.items():
        for unit, ch in selected.items():
            cand = candidate_lookup[unit][float(ch["alpha"])]
            for comparison, base_alpha in (("vs_original_search", direct[unit]), ("vs_alpha1", 1.0)):
                base = candidate_lookup[unit][base_alpha]
                diff = base["held_output_mse"] - cand["held_output_mse"]
                eps = cand["historical_epsilon_mse"]
                outcome = "win" if diff > eps else "loss" if diff < -eps else "tie"
                if outcome == "loss":
                    negative_rows.append({"selector": selector, "unit_id": unit, "layer": cand["layer"],
                                          "projection": cand["projection"], "row": cand["row"], "split_id": cand["split_id"],
                                          "comparison": comparison, "selected_alpha": cand["alpha"],
                                          "baseline_alpha": base_alpha, "held_baseline_mse": base["held_output_mse"],
                                          "held_selected_mse": cand["held_output_mse"],
                                          "held_relative_improvement": diff / base["held_output_mse"],
                                          "held_delta_sse_baseline_minus_selected": diff * 16 * SEQ,
                                          "historical_epsilon_mse": eps, "outcome": outcome})
    write_csv("NEGATIVE_RESULTS.csv", negative_rows)

    paired_rows = []
    pair_ids = sorted({u.rsplit("_", 1)[0] for u in candidate_lookup})
    for selector, selected in sorted(tau_by_unit.items()):
        for pair_id in pair_ids:
            chosen = {}
            outcomes = {}
            for split in ("S0", "S1"):
                unit = f"{pair_id}_{split}"
                cand = candidate_lookup[unit][float(selected[unit]["alpha"])]
                chosen[split] = cand
                for comparison, base_alpha in (("vs_original_search", direct[unit]), ("vs_alpha1", 1.0)):
                    base = candidate_lookup[unit][base_alpha]
                    diff = base["held_output_mse"] - cand["held_output_mse"]
                    eps = cand["historical_epsilon_mse"]
                    outcomes[(split, comparison)] = "win" if diff > eps else "loss" if diff < -eps else "tie"
            paired_rows.append({"selector": selector, "paired_layer_projection_row": pair_id,
                                "layer": chosen["S0"]["layer"], "projection": chosen["S0"]["projection"], "row": chosen["S0"]["row"],
                                "S0_alpha": chosen["S0"]["alpha"], "S1_alpha": chosen["S1"]["alpha"],
                                "S0_outcome_vs_original_search": outcomes[("S0", "vs_original_search")],
                                "S1_outcome_vs_original_search": outcomes[("S1", "vs_original_search")],
                                "paired_outcome_vs_original_search": outcomes[("S0", "vs_original_search")] + "/" + outcomes[("S1", "vs_original_search")],
                                "S0_outcome_vs_alpha1": outcomes[("S0", "vs_alpha1")],
                                "S1_outcome_vs_alpha1": outcomes[("S1", "vs_alpha1")],
                                "paired_outcome_vs_alpha1": outcomes[("S0", "vs_alpha1")] + "/" + outcomes[("S1", "vs_alpha1")],
                                "interpretation": "same row across S0/S1 is a paired observation, not two independent samples"})
    write_csv("PAIRED_SPLIT_OUTCOMES.csv", paired_rows)

    write_csv("KAPPA_NORMALIZATION.csv", [{**input_stats[k], **hstats[k]} for k in sorted(input_stats)])
    write_csv("PARETO_DOMINANCE.csv", pareto)
    source_rows = sorted(SOURCES.values(), key=lambda r: r["path"])
    write_csv("SOURCE_MANIFEST.csv", source_rows)

    direct_by_unit = {u: {"unit_id": u, "alpha": a} for u, a in direct.items()}
    all_class_counts = Counter(r["mechanism_class"] for r in mech_rows)
    pareto_counts = {"candidate_count": len(pareto), "dominated": sum(r["is_dominated"] for r in pareto),
                     "nondominated": sum(not r["is_dominated"] for r in pareto)}
    tau_all = {(r["selector"], r["comparison"]): r for r in selection_summaries if r["scope"] == "ALL"}
    tau_split = {(r["selector"], r["scope"], r["comparison"]): r for r in selection_summaries if r["scope"] in ("S0", "S1")}
    nonzero_direct_gains = [tau_all[(f"tau={t:g}", "vs_original_search")]["held_pooled_relative_improvement"] for t in TAUS if t > 0]
    robust = []
    for t in TAUS:
        if t == 0:
            continue
        selector = f"tau={t:g}"
        all_vs = tau_all[(selector, "vs_original_search")]
        s0 = tau_split[(selector, "S0", "vs_original_search")]
        s1 = tau_split[(selector, "S1", "vs_original_search")]
        target_splits = [r for r in selection_summaries if r["selector"] == selector and r["comparison"] == "vs_original_search"
                         and r["scope"].startswith("L") and r["scope"].count("/") == 2]
        both_split_positive_targets = sum(
            next(x for x in target_splits if x["scope"] == f"{target}/S0")["held_sse_improvement_baseline_minus_selected"] > 0 and
            next(x for x in target_splits if x["scope"] == f"{target}/S1")["held_sse_improvement_baseline_minus_selected"] > 0
            for target in [f"L{li}/{pr}" for li in LAYERS for pr in PROJECTIONS]
        )
        severe = all_vs["severe_held_degradations_vs_alpha1_below_minus_10pct"]
        robust.append(bool(all_vs["pooled_improvement_beyond_historical_epsilon"] and
                           s0["pooled_improvement_beyond_historical_epsilon"] and
                           s1["pooled_improvement_beyond_historical_epsilon"] and
                           both_split_positive_targets >= 4 and severe == 0))
    if any(robust):
        conclusion = "A"
        conclusion_text = "预固定tau中至少一个同时改善两split pooled held并超过历史epsilon、至少4/6 target两split均为正且无新增<-10% alpha1严重退化；观察支持在新文章上验证。"
        next_step = "只提出tau=0.1的冻结验证规则：每个row-split按原始search选择min(L_search + 0.1*kappa*||Q-W||^2)，并在全新文章按原split单位评估；不在本轮执行。"
    elif any(g > 0 for g in nonzero_direct_gains):
        conclusion = "B"
        conclusion_text = "预固定tau或完整有限候选路径存在旧样例正收益，但没有通过预设跨split一致性门槛；目前只支持少数旧例探索信号，证据不足。"
        next_step = "暂不提出新的验证规则；固定候选的旧数据探索尚未提供足够信号。"
    else:
        conclusion = "C"
        conclusion_text = "所有预固定正tau pooled held相对原search选择均无正收益，权重误差下降不能支持所检验机制，或伴随明显search/held损失。"
        next_step = "不进入新文章验证；先停止该正则化方向。"

    run_info = {
        "status": "COMPLETE_ARTIFACT_ONLY_DIAGNOSTIC",
        "scope": "original v24 scalar10 only; L0/L12/L23 up/down; fixed rows [0,224,448,672]; S0/S1",
        "new_held_forwards": 0, "new_search_forwards": 0, "cuda_tasks": 0, "ppl_runs": 0, "quantization_runs": 0,
        "candidate_row_count": len(held_rows), "row_split_units": len(by_unit), "alpha_values_per_unit": 10,
        "full_linear_candidate_tensor_states": 120,
        "held_read_order": "all rho paths and tau selections were generated and written from search data and ||Q-W||^2 before HELD_RESULTS.csv or held score sidecars were opened",
        "output_mse_definition": "mean squared local projection output residual per input token for one selected row",
        "output_sse_definition": "output MSE multiplied by number of search/held tokens (16*2048 per row)",
        "weight_mse_definition": "||Q_FP16-W_FP16||_2^2 divided by input dimension; FP32 difference and square accumulation",
        "regularizer_definition": "search output MSE + rho * weight squared norm; no mean/sum substitution",
        "hessian_C_normalization": "GPTQ raw H=2/n_documents sum_doc X_doc^T X_doc; H already averages over documents, and C_search per token is H/(2*SEQ), so kappa=trace(H)/(2*SEQ*input_dim), independently recomputed from source FP16 search inputs",
        "mechanism_class_counts": {str(k): all_class_counts[k] for k in range(1,5)},
        "pareto_counts": pareto_counts,
        "conclusion_class": conclusion,
        "conclusion": conclusion_text,
        "next_step": next_step,
        "cross_split_pairing": "24 same layer/projection/row pairs recur in S0 and S1; descriptive paired observations, no independence or inferential testing claimed",
        "checks_passed": sum(CHECKS.values()), "checks_total": len(CHECKS),
        "cpu_seconds": time.process_time() - START_CPU, "wall_seconds": time.perf_counter() - START_WALL,
    }
    write_json("RUN_SUMMARY.json", run_info)
    write_json("AUDIT_CHECKS.json", {"checks": CHECKS, "all_passed": all(CHECKS.values()), "passed": sum(CHECKS.values()), "total": len(CHECKS)})
    write_json("PHASE_TIMING.json", {"cpu_budget_seconds": CPU_BUDGET_SECONDS, "phases": PHASES,
                                     "cpu_seconds_total": time.process_time() - START_CPU,
                                     "wall_seconds_total": time.perf_counter() - START_WALL})
    write_report(run_info, protocol, lock, input_stats, hstats, mech_rows, mech_summary, pareto, pareto_unit,
                 selection_summaries, path_interval_summaries, path_boundary_summaries, intervals, robust)
    # Re-write after report source hashes are complete.
    write_csv("SOURCE_MANIFEST.csv", sorted(SOURCES.values(), key=lambda r: r["path"]))
    phase("reports_and_provenance_written")
    stale_failure = OUT / "RUN_FAILURE.json"
    if stale_failure.exists():
        stale_failure.unlink()


def write_report(info: dict, protocol: dict, lock: dict, inputs: dict, hstats: dict,
                 mech_rows: list[dict], mech_summary: list[dict], pareto: list[dict], pareto_unit: list[dict],
                 tau_summaries: list[dict], path_summaries: list[dict], boundary_summaries: list[dict],
                 path_intervals: list[dict], robust: list[bool]) -> None:
    classes = Counter(r["mechanism_class"] for r in mech_rows)
    search_improved = [r for r in mech_rows if r["search_strictly_improved"]]
    search_improved_norm_up = sum(r["weight_error_increased"] for r in search_improved)
    search_improved_norm_down_held_win = sum(not r["weight_error_increased"] and r["held_outcome_vs_alpha1"] == "win" for r in search_improved)
    search_improved_norm_down_held_tie = sum(not r["weight_error_increased"] and r["held_outcome_vs_alpha1"] == "tie" for r in search_improved)
    search_improved_norm_down_held_loss = sum(not r["weight_error_increased"] and r["held_outcome_vs_alpha1"] == "loss" for r in search_improved)
    dominated = sum(bool(r["is_dominated"]) for r in pareto)
    nonzero_taus = [r for r in tau_summaries if r["scope"] == "ALL" and r["comparison"] == "vs_original_search" and r["selector"] != "tau=0"]
    lines = [
        "# v30：固定候选上的权重误差正则化诊断", "",
        f"状态：{info['status']}。全程 CPU {info['cpu_seconds']:.1f}s / 900s 预算，wall {info['wall_seconds']:.1f}s。",
        "仅读取冻结 v23/v24/v25 产物；没有新前向、CUDA、GPTQ、held、PPL 或整模型量化。", "",
        "## 对象与口径", "",
        "使用 Qwen2.5-0.5B-Instruct revision `7ae557604adf67be50417f59c2c2f167def9a775` 的 v24 冻结实验：原 scalar10 alpha=0.1…1.0；L0/L12/L23 的 up/down；每个线性层固定行 [0,224,448,672]；S0/S1。共 48 个 row-split 单元、120 个四行 candidate tensor states、480 个 row-alpha 观察。v11 早期结果未合并；v28 浮点选择差异未作为分布退化证据。",
        "直接 search/held 分数来自 v24 原文件，candidate `weight_fp16` 是保存的 FP16 反量化权重。checkpoint safetensors 内目标 W 存为 BF16；v24 的冻结模型 dtype 是 FP16，因此 v30 将来源 W cast 成 FP16 后计算 e=Q−W，并核对转换后 tensor hash。每个候选的四行 Q 张量、scale、zero、g_idx dtype/shape/hash均逐项核验；held sidecar 的 `candidate_weight_sha256` 与同一 Q 张量 hash 一致，search 原始 CSV 与候选 sidecar 的 row 分数一致。",
        "误差口径：output MSE 是每个 row、每个 input token 的局部投影输出残差平方均值；output SSE=MSE×该 split 的 16×2048 tokens。权重 MSE=||Q−W||²/input_dim；正则化只使用平方范数 ||Q−W||²。详见 `CANDIDATE_ROW_TABLE.csv`。", "",
        "## 统计量和归一化", "",
        "原始 search 输入及未阻尼 Hessian 均可读。GPTQ `process_batch` 在每篇文章 batch 上令 `H = (2/n_documents) Σ_doc X_docᵀX_doc`，这里 nsamples=16、每篇2048 tokens；缓存 H 在阻尼/求逆前保存。H 已按文章数取平均，但每篇仍累加2048个 token，因此 v30 用 `C_search=H/(2×2048)=Σ_token x xᵀ/N_tokens`，`kappa=trace(C_search)/input_dim`，并由原始 FP16 search 输入独立重算 trace。核对关系是 `trace(H)/(2×2048×input_dim)`；不把 raw H/2 直接当逐 token 协方差，也不使用 damped H 或 inverse。详见 `KAPPA_NORMALIZATION.csv`。",
        "rho=tau*kappa，tau 预固定为 {0, 0.01, 0.1, 1}。rho=0 按 v24 原 `selected_by_original_10` 结果直接复现；正 rho 的并列按更大 alpha 处理。完整有限候选下包络的全部区间、交点和选择都在 held 数据打开前由 search MSE 与权重平方范数生成，详见 `SELECTION_PATH*.csv` 与 `FULL_PATH_*_SEARCH_ONLY.csv`。", "",
        "## search 选择与 alpha=1 的机制对照", "",
        f"四类数量（48 row-split observations）：1) search 改善、权重误差增大、held 退化 {classes[1]}；2) search 改善、权重误差增大、held 改善 {classes[2]}；3) search 改善、权重误差未增大、held 退化 {classes[3]}；4) 其他 {classes[4]}。第4类中，search 严格改善 {len(search_improved)} 个；其权重平方误差增大的有 {search_improved_norm_up} 个。search 改善且权重误差未增大的单元，held 相对 alpha=1 有 {search_improved_norm_down_held_win} 胜、{search_improved_norm_down_held_tie} 平、{search_improved_norm_down_held_loss} 负。held win/tie/loss 沿用 v24 每个单元的 frozen epsilon。所有单元的实际 alpha 和误差变化见 `MECHANISM_RESULTS.csv`，target/split 数量见 `MECHANISM_SUMMARY.csv`。",
        f"10 个候选的 Pareto 审计共 {len(pareto)} 个点，其中被另一个候选同时在 search output MSE 与 ||Q−W||² 支配 {dominated} 个，未支配 {len(pareto)-dominated} 个。逐候选支配者见 `PARETO_DOMINANCE.csv`；单位与 split 计数见 `PARETO_UNIT_SUMMARY.csv`。", "",
        "## 预固定 tau 和完整路径的 held 事后诊断", "",
        "下表使用 pooled held SSE，列出所有范围；行级负结果全部保存在 `NEGATIVE_RESULTS.csv`。S0/S1 和六个 target 的完整比较在 `REGULARIZATION_SUMMARY.csv`，逐行选择和 held 分数在 `TAU_SELECTIONS.csv`。完整路径区间及交点另见 `FULL_PATH_INTERVAL_SUMMARY.csv` 和 `FULL_PATH_BOUNDARY_SUMMARY.csv`。胜/平/负阈值沿用 v24 epsilon；严重退化计数沿用原协议的相对 alpha=1 held MSE <−10% 口径。正收益集中度按 top1/top3 row 对 gross positive held SSE saving 的占比，净收益占比同时列出。",
        "", "正的 held change 表示 pooled held SSE 降低。", "",
        "| selector | vs search ALL/S0/S1 (%) | search SSE change (+ = sacrifice) | norm² reduction (+ = decrease) | W/T/L | median / worst row (%) | top1 / top3 gross positive share | severe vs alpha1 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    slookup = {(r["selector"], r["scope"], r["comparison"]): r for r in tau_summaries}
    for selector in ["tau=0", "tau=0.01", "tau=0.1", "tau=1"]:
        allr = slookup[(selector, "ALL", "vs_original_search")]
        s0 = slookup[(selector, "S0", "vs_original_search")]
        s1 = slookup[(selector, "S1", "vs_original_search")]
        lines.append(f"| {selector} | {allr['held_pooled_relative_improvement']*100:.4f} / {s0['held_pooled_relative_improvement']*100:.4f} / {s1['held_pooled_relative_improvement']*100:.4f} | {allr['search_sse_change_selected_minus_baseline']:.6g} | {allr['weight_squared_norm_reduction_baseline_minus_selected']:.6g} | {allr['wins']}/{allr['ties']}/{allr['losses']} | {allr['median_unit_relative_held_improvement']*100:.4f} / {allr['worst_unit_relative_held_improvement']*100:.4f} | {allr['top1_share_of_positive_held_savings']*100:.1f} / {allr['top3_share_of_positive_held_savings']*100:.1f}% | {allr['severe_held_degradations_vs_alpha1_below_minus_10pct']} |")
    lines += ["", "对 alpha=1 的 pooled held change、权重范数下降与行级统计：", "",
              "| selector | held change vs alpha1 ALL/S0/S1 (%) | norm² reduction vs alpha1 (%) | W/T/L | median / worst row (%) | top1 / top3 gross positive share |",
              "|---|---:|---:|---:|---:|---:|"]
    for selector in ["tau=0", "tau=0.01", "tau=0.1", "tau=1"]:
        allr = slookup[(selector, "ALL", "vs_alpha1")]
        s0 = slookup[(selector, "S0", "vs_alpha1")]
        s1 = slookup[(selector, "S1", "vs_alpha1")]
        lines.append(f"| {selector} | {allr['held_pooled_relative_improvement']*100:.4f} / {s0['held_pooled_relative_improvement']*100:.4f} / {s1['held_pooled_relative_improvement']*100:.4f} | {allr['weight_squared_norm_reduction_fraction']*100:.4f} | {allr['wins']}/{allr['ties']}/{allr['losses']} | {allr['median_unit_relative_held_improvement']*100:.4f} / {allr['worst_unit_relative_held_improvement']*100:.4f} | {allr['top1_share_of_positive_held_savings']*100:.1f} / {allr['top3_share_of_positive_held_savings']*100:.1f}% |")
    target_scope_names = list(dict.fromkeys(r["scope"] for r in tau_summaries if r["selector"] == "tau=0.1" and r["comparison"] == "vs_original_search" and r["scope"].startswith("L") and r["scope"].count("/") in (1, 2)))
    lines += ["", "预固定 tau 的六个 target pooled 与 S0/S1 held 结果（相对原 search；% 和 W/T/L）：", "",
              "| scope | tau=0.01 | tau=0.1 | tau=1 |", "|---|---:|---:|---:|"]
    for scope in target_scope_names:
        cells = []
        for selector in ["tau=0.01", "tau=0.1", "tau=1"]:
            r = slookup[(selector, scope, "vs_original_search")]
            cells.append(f"{r['held_pooled_relative_improvement']*100:.4f}% ({r['wins']}/{r['ties']}/{r['losses']})")
        lines.append(f"| {scope} | " + " | ".join(cells) + " |")
    positive_path = [r for r in path_summaries if r["scope"] == "ALL" and r["comparison"] == "vs_original_search" and r["held_pooled_relative_improvement"] > 0]
    negative_path = [r for r in path_summaries if r["scope"] == "ALL" and r["comparison"] == "vs_original_search" and r["held_pooled_relative_improvement"] < 0]
    interval_by_id = {r["path_interval_id"]: r for r in path_intervals}
    positive_details = [f"{r['selector']} rho∈({interval_by_id[r['selector']]['rho_lower_exclusive']}, {interval_by_id[r['selector']]['rho_upper_exclusive']}) pooled={r['held_pooled_relative_improvement']*100:.4f}%，top1正收益占比={r['top1_share_of_positive_held_savings']*100:.1f}%" for r in positive_path]
    lines += ["", f"完整正 rho 下包络包含 {len({r['selector'] for r in path_summaries})} 个 rho 区间，另有 {len({r['selector'] for r in boundary_summaries})} 个全局交点。相对原 search 选择 pooled held 为正的区间 {len(positive_path)} 个、为负的区间 {len(negative_path)} 个。正区间：{'；'.join(positive_details) if positive_details else '无'}。这些事后区间不用于挑选最终 rho，只描述固定 10 候选集的有限路径。",
              "跨 split 的相同行是配对观测：共24个 layer/projection/row 身份各自出现在 S0、S1；这里不将48个单元解释为独立样本，也不作显著性或泛化推断。", "",
              "## 结论边界", "",
              info["conclusion"], info["next_step"],
              "这个诊断只是在固定 GPTQ 候选上重选 alpha；没有改候选生成 Hessian 或 GPTQ 补偿。权重误差下降不等同于鲁棒性提升；没有 held/PPL 保证，也没有证明既往退化由 Hessian 不确定性造成，更没有提出新的权重偏移正则。v28 的浮点选择差异未计入机制证据。", "",
              "## 文件与复现", "",
              "`SOURCE_MANIFEST.csv` 列出实际读取来源及其 SHA-256；`AUDIT_CHECKS.json` 汇总候选张量、分数、Hessian normalization 和 provenance 检查；`RUN_SUMMARY.json` 记录范围与执行边界。完整负项见 `NEGATIVE_RESULTS.csv`，跨 split 配对行见 `PAIRED_SPLIT_OUTCOMES.csv`，全部候选和 target/split 结果分别见候选表与 summary CSV。", "",
              "在 PowerShell 中从仓库根目录复现：", "", "```powershell",
              "& 'E:/quant-thesis/.venv-gptq/Scripts/python.exe' -B -X utf8 'E:/quant-thesis/experiments/range_search_w4/results/range_w4_weight_error_regularization_v30/analyze_v30.py'",
              "```", ""]
    (OUT / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        try:
            write_json("RUN_FAILURE.json", {"status": "FAILED", "exception": repr(exc),
                                             "checks": CHECKS, "sources": sorted(SOURCES.values(), key=lambda r: r["path"]),
                                             "cpu_seconds": time.process_time() - START_CPU,
                                             "wall_seconds": time.perf_counter() - START_WALL,
                                             "phases": PHASES})
        finally:
            raise
