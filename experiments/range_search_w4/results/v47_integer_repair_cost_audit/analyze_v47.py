"""Read-only v46 single-integer-repair cost and trajectory audit.

Reads v46 formal group result checkpoints and timing summaries only. Does not
load a model, touch a GPU, replay repair, or write into the v46 directory.
"""
from __future__ import annotations

import csv
import hashlib
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch


ROOT = Path(r"E:\quant-thesis\experiments\range_search_w4\results")
V46 = ROOT / "v46_m1_single_integer_repair_fullmodel"
OUT = ROOT / "v47_integer_repair_cost_audit"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def csv_rows(path: Path):
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows, fields):
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    timings = csv_rows(V46 / "STAGE_TIMINGS.csv")
    formal_t = [r for r in timings if r["mode"] == "formal"]
    block_rows = csv_rows(V46 / "BLOCK_TIMINGS.csv")
    module_rows = [r for r in csv_rows(V46 / "MODULE_RESULTS.csv") if r["mode"] == "formal"]
    groups_dir = V46 / "formal" / "groups"
    group_files = sorted(groups_dir.glob("layer_*.pt"))
    protocol_sha = sha256(V46 / "PROTOCOL.json")

    # Aggregate per-call synchronized stage timing. A timer can be nested; the
    # independent FP64 check is nested in accepted_update_and_rejection_restore.
    stage = defaultdict(lambda: {"calls": 0, "seconds": 0.0})
    by_module_direct = defaultdict(list)
    for r in formal_t:
        k = r["stage"]
        stage[k]["calls"] += 1
        stage[k]["seconds"] += float(r["seconds"])
        if k == "direct_recompute":
            by_module_direct[(int(r["layer"]), r["module"])].append(float(r["seconds"]))

    # `batch_repair` has a fixed call order: initial direct score, candidate
    # direct scores in the loop, final direct score. Attribute the first and
    # last timer entry per Linear accordingly; middle entries are proposals.
    direct_initial = direct_proposals = direct_final = 0.0
    direct_calls_initial = direct_calls_proposals = direct_calls_final = 0
    direct_call_detail = []
    for (layer, module), vals in sorted(by_module_direct.items()):
        if len(vals) < 2:
            raise RuntimeError(f"Expected initial and final direct scores for {(layer, module)}")
        direct_initial += vals[0]
        direct_final += vals[-1]
        direct_proposals += sum(vals[1:-1])
        direct_calls_initial += 1
        direct_calls_final += 1
        direct_calls_proposals += len(vals) - 2
        direct_call_detail.append({"layer": layer, "module": module,
                                   "direct_calls": len(vals),
                                   "initial_loss_seconds": vals[0],
                                   "proposal_direct_seconds": sum(vals[1:-1]),
                                   "final_loss_seconds": vals[-1]})
    write_csv(OUT / "DIRECT_CALL_ATTRIBUTION.csv", direct_call_detail,
              ["layer", "module", "direct_calls", "initial_loss_seconds",
               "proposal_direct_seconds", "final_loss_seconds"])

    # Full block interval is the user-requested repair-wall denominator.
    block_seconds = sum(float(r["seconds"]) for r in block_rows)
    stage_exclusive = sum(v["seconds"] for k, v in stage.items()
                          if k != "independent_FP64_increment_recheck")
    unattributed = block_seconds - stage_exclusive
    if unattributed < -1e-3:
        raise RuntimeError(f"Exclusive stage timers exceed block wall by {-unattributed:.3f}s")

    # One row per request category. Combined timers are deliberately preserved
    # as combined, never apportioned by FLOPs or event count.
    labels = {
        "dependency_input_capture": ("1 input capture", "Formal dependency input capture"),
        "C_construction": ("2 FP64 C construction", "Full undamped FP64 C, including chunk transfer, symmetrization and finite check"),
        "repair_initialization": ("3 initial e/p", "Code/grid materialization, Q/e and p; initial loss is attributed separately below"),
        "candidate_scoring": ("4-5 candidate score and minimum", "Combined eligibility, delta score, budget, minimum selection and row status; no internal split"),
        "accepted_update_and_rejection_restore": ("6-8 accepted update / reject handling", "Merged code/Q/p update, rejection restore/blacklist and assertions; nested FP64 rechecks excluded from exclusive sum"),
        "direct_recompute": ("7 direct FP32 path", "Split by fixed call order into initial loss, proposal checks, final loss"),
        "independent_FP64_increment_recheck": ("8 independent FP64 recheck", "Nested in accepted-update timer; report separately, not additive"),
        "p_drift_check": ("8 cache drift check", "Periodic p-cache refresh/check; separate timer"),
        "input_hash": ("1 input hash", "SHA-256 of captured input; measured separately"),
        "down_commit_prefix_refresh": ("1 dependency refresh", "Full block forward used to refresh next-block input"),
        "packing_checkpoint_IO": ("9 packing/checkpoint IO", "Group checkpoint and commit metadata write"),
        "whole_group_commit": ("9 group commit", "Copy repaired sibling weights into the working model"),
        "final_correctness": ("9 final correctness/packing", "Final code/grid/pack correctness checks; not checkpoint write"),
    }
    stage_rows = []
    for key, (name, definition) in labels.items():
        if key == "direct_recompute":
            for sub, calls, seconds, note in [
                ("initial direct loss", direct_calls_initial, direct_initial,
                 "First direct_recompute call per Linear; ordering fixed by batch_repair"),
                ("proposal direct checks", direct_calls_proposals, direct_proposals,
                 "Middle direct_recompute calls; each proposal call scores every output row, only proposed rows are consumed"),
                ("final direct loss", direct_calls_final, direct_final,
                 "Last direct_recompute call per Linear; ordering fixed by batch_repair"),
            ]:
                stage_rows.append({"category": name, "stage": sub, "calls": calls,
                                   "seconds": seconds, "pct_of_block_wall": 100*seconds/block_seconds,
                                   "accounting": "exclusive measured synchronized wall",
                                   "coverage": note})
        else:
            v = stage.get(key)
            if not v:
                stage_rows.append({"category": name, "stage": key, "calls": 0,
                                   "seconds": "", "pct_of_block_wall": "",
                                   "accounting": "NOT_MEASURED", "coverage": definition})
                continue
            nested = key == "independent_FP64_increment_recheck"
            stage_rows.append({"category": name, "stage": key, "calls": v["calls"],
                               "seconds": v["seconds"], "pct_of_block_wall": 100*v["seconds"]/block_seconds,
                               "accounting": "nested, not additive" if nested else "exclusive measured synchronized wall",
                               "coverage": definition})
    stage_rows.append({"category": "10 unattributed", "stage": "block wall minus exclusive timers",
                       "calls": "", "seconds": unattributed, "pct_of_block_wall": 100*unattributed/block_seconds,
                       "accounting": "derived residual", "coverage": "Python gaps, loop/status handling, notes/logging, timer gaps and other untimed block work"})
    write_csv(OUT / "STAGE_COSTS.csv", stage_rows,
              ["category", "stage", "calls", "seconds", "pct_of_block_wall", "accounting", "coverage"])

    timer_coverage = [
        {"requested_component": "Input capture and dependency refresh", "observed_timer": "dependency_input_capture; down_commit_prefix_refresh; input_hash", "coverage": "separate synchronized timers; hash is a distinct CPU stage", "limit": "capture timer includes full prefix execution/hooks and host/device copies"},
        {"requested_component": "FP64 C", "observed_timer": "C_construction", "coverage": "separate full C build timer", "limit": "includes X chunk copies, GEMMs, normalization/symmetrization, finite check"},
        {"requested_component": "Initial e/p/loss", "observed_timer": "repair_initialization + first direct_recompute per Linear", "coverage": "e/p init separate; initial loss attributed by fixed call order", "limit": "Q/E/p init remains one merged timer"},
        {"requested_component": "Candidate score and min selection", "observed_timer": "candidate_scoring", "coverage": "combined timer", "limit": "delta scoring, eligible mask, budget, min reduction/status not individually timed"},
        {"requested_component": "Code and p update", "observed_timer": "accepted_update_and_rejection_restore", "coverage": "combined timer", "limit": "cannot isolate p vector update; timer also includes code/Q update, reject restore, mask updates and checks"},
        {"requested_component": "Direct output scoring", "observed_timer": "direct_recompute", "coverage": "per-call synchronized timer, call-order split", "limit": "FP32 matmul + FP64 square/sum and X host-to-device chunk copies remain merged"},
        {"requested_component": "Reject handling and FP64 recheck", "observed_timer": "accepted_update_and_rejection_restore + independent_FP64_increment_recheck", "coverage": "outer reject/update timer plus nested FP64 timer", "limit": "no standalone reject-only time; FP64 timer nested and not additive"},
        {"requested_component": "Checkpoint and IO", "observed_timer": "packing_checkpoint_IO; whole_group_commit", "coverage": "separate group write/commit timers", "limit": "final model save/reload and PPL occur outside the 24 block repair-wall sum"},
        {"requested_component": "Minimum selection only", "observed_timer": "candidate_scoring", "coverage": "NOT_MEASURED separately", "limit": "no safe measured split from candidate scoring aggregate"},
        {"requested_component": "p update only", "observed_timer": "V46 accepted_update_and_rejection_restore; V47 profile_p_vector_update", "coverage": "V46 full-model timer is merged; V47 adds an isolated synchronized p-update timer on both complete L0 matrices", "limit": "profile scope is only L0 up_proj/down_proj; child timer includes synchronization overhead and is nested in outer update timer"},
    ]
    write_csv(OUT / "TIMER_COVERAGE.csv", timer_coverage,
              ["requested_component", "observed_timer", "coverage", "limit"])

    # Stream committed group result checkpoints, aggregating row update counts,
    # stored per-step direct loss deltas, and exact full-row direct-call waste.
    expected_hashes = set()
    protocol_values = set()
    row_hist = np.zeros(33, dtype=np.int64)
    layer_mod = defaultdict(lambda: {"rows": 0, "accepted": 0, "evals": 0, "cap_rows": 0,
                                    "rejects": 0, "linears": 0})
    gains = {n: 0.0 for n in (1, 4, 8, 16, 32)}
    initial_total = final_total = 0.0
    direct_candidate_full_outputs = direct_candidate_needed_outputs = 0
    proposal_batches = proposal_rows = rejected_proposals = 0
    update_sum_check = 0
    cp_hash_metadata = []
    if len(group_files) != 96:
        raise RuntimeError(f"Expected 96 formal group checkpoints, found {len(group_files)}")
    for path in group_files:
        ck = torch.load(path, map_location="cpu", weights_only=False)
        if not ck.get("committed"):
            raise RuntimeError(f"Uncommitted group in formal directory: {path.name}")
        protocol_values.add(ck.get("protocol_sha256"))
        if ck.get("protocol_sha256") != protocol_sha:
            raise RuntimeError(f"Protocol hash mismatch: {path.name}")
        layer = int(ck["layer"])
        cp_hash_metadata.append({"checkpoint": path.name, "layer": layer, "group": ck["group"],
                                 "input_sha256": ck["input_sha256"], "C_sha256": ck["C_sha256"],
                                 "protocol_sha256": ck["protocol_sha256"],
                                 "bytes": path.stat().st_size})
        for module, result in ck["modules"].items():
            updates = result["updates"].numpy().astype(np.int64, copy=False)
            evaluations = result["evaluations"].numpy().astype(np.int64, copy=False)
            nrows = len(updates)
            if np.any(updates > 32) or np.any(updates < 0):
                raise RuntimeError(f"Invalid update count in {path.name}:{module}")
            row_hist += np.bincount(updates, minlength=33)
            g = layer_mod[(layer, module)]
            g["rows"] += nrows
            g["accepted"] += int(updates.sum())
            g["evals"] += int(evaluations.sum())
            g["cap_rows"] += int((updates == 32).sum())
            g["rejects"] += int(result["rejections"].sum())
            g["linears"] += 1
            update_sum_check += int(updates.sum())
            initial_total += float(result["initial_mse"].sum())
            final_total += float(result["final_mse"].sum())
            moves = result["moves"]
            reject_events = result["rejection_events"]
            rounds = set()
            for mv in moves:
                rounds.add(int(mv["round"]))
                ua = mv["updates_after"].numpy().astype(np.int64, copy=False)
                dl = mv["direct_delta"].numpy().astype(np.float64, copy=False)
                for cutoff in gains:
                    gains[cutoff] += float(dl[ua <= cutoff].sum())
                n = len(mv["rows"])
                proposal_rows += n
            for ev in reject_events:
                rounds.add(int(ev["attempt"]))
                n = len(ev["rows"])
                proposal_rows += n
                rejected_proposals += n
            proposal_batches += len(rounds)
            direct_candidate_full_outputs += nrows * len(rounds)
            direct_candidate_needed_outputs += sum(len(mv["rows"]) for mv in moves)
            direct_candidate_needed_outputs += sum(len(ev["rows"]) for ev in reject_events)

    if protocol_values != {protocol_sha}:
        raise RuntimeError("Unexpected protocol hash set")
    if update_sum_check != sum(int(r["accepted_updates"]) for r in module_rows):
        raise RuntimeError("Checkpoint update count total disagrees with MODULE_RESULTS.csv")
    if proposal_batches != stage["direct_recompute"]["calls"] - 2*len(module_rows):
        raise RuntimeError(f"Proposal direct call count mismatch: {proposal_batches}")
    if proposal_rows != update_sum_check + rejected_proposals:
        raise RuntimeError("Proposal row count did not reconcile")

    update_dist = []
    for n, count in enumerate(row_hist.tolist()):
        update_dist.append({"accepted_updates_per_row": n, "row_count": count,
                            "share_of_304128_rows": count/304128,
                            "at_32_update_cap": n == 32})
    write_csv(OUT / "UPDATE_DISTRIBUTION.csv", update_dist,
              ["accepted_updates_per_row", "row_count", "share_of_304128_rows", "at_32_update_cap"])

    layer_rows = []
    for (layer, module), g in sorted(layer_mod.items()):
        layer_rows.append({"layer": layer, "module": module, **g,
                           "mean_updates_per_row": g["accepted"]/g["rows"],
                           "cap_row_share": g["cap_rows"]/g["rows"]})
    write_csv(OUT / "LAYER_PROJECTION_UPDATES.csv", layer_rows,
              ["layer", "module", "linears", "rows", "accepted", "evals", "cap_rows", "rejects",
               "mean_updates_per_row", "cap_row_share"])

    projection_rows = []
    by_projection = defaultdict(lambda: {"linears": 0, "rows": 0, "accepted": 0, "evals": 0,
                                         "cap_rows": 0, "rejects": 0})
    by_layer = defaultdict(lambda: {"linears": 0, "rows": 0, "accepted": 0, "evals": 0,
                                    "cap_rows": 0, "rejects": 0})
    for (layer, module), g in layer_mod.items():
        for dest in (by_projection[module], by_layer[layer]):
            for k in dest:
                dest[k] += g[k]
    for module, g in sorted(by_projection.items()):
        projection_rows.append({"scope": "projection", "key": module, **g,
                                "mean_updates_per_row": g["accepted"]/g["rows"],
                                "cap_row_share": g["cap_rows"]/g["rows"]})
    for layer, g in sorted(by_layer.items()):
        projection_rows.append({"scope": "layer", "key": f"{layer:02d}", **g,
                                "mean_updates_per_row": g["accepted"]/g["rows"],
                                "cap_row_share": g["cap_rows"]/g["rows"]})
    write_csv(OUT / "PROJECTION_AND_BLOCK_SUMMARY.csv", projection_rows,
              ["scope", "key", "linears", "rows", "accepted", "evals", "cap_rows", "rejects",
               "mean_updates_per_row", "cap_row_share"])

    gain_rows = []
    for n in gains:
        change = gains[n]
        gain_rows.append({"per_row_update_cutoff": n,
                          "cumulative_search_mse_change_sum": change,
                          "cumulative_search_mse_reduction": -change,
                          "relative_to_unrepaired_initial_sum_pct": 100*(-change)/initial_total,
                          "observed_method": "stored direct deltas filtered by updates_after <= cutoff",
                          "interpretation": "fixed-input trajectory summary only; not a model PPL or new budget result"})
    write_csv(OUT / "ACCEPTANCE_BUDGET_GAINS.csv", gain_rows,
              ["per_row_update_cutoff", "cumulative_search_mse_change_sum", "cumulative_search_mse_reduction",
               "relative_to_unrepaired_initial_sum_pct", "observed_method", "interpretation"])

    waste = [{"proposal_direct_calls": proposal_batches,
              "proposal_candidates_scored_by_direct": proposal_rows,
              "full_linear_row_outputs_computed": direct_candidate_full_outputs,
              "proposed_row_outputs_needed": direct_candidate_needed_outputs,
              "unchanged_row_outputs_recomputed": direct_candidate_full_outputs-direct_candidate_needed_outputs,
              "computed_to_needed_ratio": direct_candidate_full_outputs/direct_candidate_needed_outputs,
              "fraction_full_outputs_not_consumed": 1-direct_candidate_needed_outputs/direct_candidate_full_outputs,
              "caveat": "Counting output rows only, not a time estimate; changed row-batch GEMM shapes may alter FP32 rounding and need frozen-path numerical validation"}]
    write_csv(OUT / "REPEATED_DIRECT_WORK.csv", waste, list(waste[0]))

    local_score_seconds = direct_proposals + stage["candidate_scoring"]["seconds"] + stage["accepted_update_and_rejection_restore"]["seconds"]
    bounds = []
    for label, seconds, scope_note, local in [
        ("proposal direct FP32 checks", direct_proposals, "Measured proposal direct path; strict direct check is required by frozen acceptance flow", True),
        ("all direct FP32 score calls", stage["direct_recompute"]["seconds"], "Initial, proposal and final direct scores", False),
        ("FP64 C construction", stage["C_construction"]["seconds"], "Exact C path; sibling sharing already occurs within dependency groups", False),
        ("candidate scoring plus minimum", stage["candidate_scoring"]["seconds"], "Minimum is not separately timed; corresponds to whole combined timer", True),
        ("p update upper bound", stage["accepted_update_and_rejection_restore"]["seconds"], "p-only time is not measured; entire combined update/reject timer is a conservative bound", True),
        ("input capture plus prefix refresh and hash", stage["dependency_input_capture"]["seconds"]+stage["down_commit_prefix_refresh"]["seconds"]+stage["input_hash"]["seconds"], "Capture, SHA-256 and dependent next-block refresh", False),
    ]:
        f = seconds / block_seconds
        lf = seconds / local_score_seconds if local else ""
        bounds.append({"component": label, "measured_seconds_or_upper_bound": seconds,
                       "fraction_of_full_formal_repair": f,
                       "ideal_infinite_speedup_upper_bound_full_repair": 1/(1-f),
                       "local_scoring_fraction_where_applicable": lf,
                       "ideal_infinite_speedup_upper_bound_local_scoring_where_applicable": 1/(1-lf) if local else "",
                       "scope_note": scope_note,
                       "warning": "Amdahl ideal only; not measured savings or a runtime prediction"})
    write_csv(OUT / "OPTIMIZATION_BOUNDS.csv", bounds,
              ["component", "measured_seconds_or_upper_bound", "fraction_of_full_formal_repair",
               "ideal_infinite_speedup_upper_bound_full_repair", "local_scoring_fraction_where_applicable",
               "ideal_infinite_speedup_upper_bound_local_scoring_where_applicable", "scope_note", "warning"])

    source_list = []
    for rel in ["PROTOCOL.json", "repair_v45.py", "precision_check.py", "runtime_v46.py",
                "run_v46_retry.py", "STAGE_TIMINGS.csv", "BLOCK_TIMINGS.csv", "MODULE_RESULTS.csv",
                "formal/COMMIT_STATUS.json", "formal/PREFIX_STATUS.json", "FINAL_STATUS.json",
                "REPORT.md", "FINAL_DECISION.json", "COST_COMPARISON.csv", "SOURCE_AUDIT.json",
                "CHECKPOINT_LOCK.json", "DEPLOYMENT_LOCK.json",
                "FORMAL_REPAIR_COMPLETE.json", "EXECUTION_AMENDMENT_001.json",
                "EXECUTION_AMENDMENT_002.json", "FAILED_ATTEMPTS.jsonl"]:
        p = V46 / rel
        source_list.append({"source": str(p), "sha256": sha256(p), "bytes": p.stat().st_size,
                            "role": "v46 frozen protocol/source/timing/summary/commit evidence; read-only"})
    for row in cp_hash_metadata:
        source_list.append({"source": str(groups_dir / row["checkpoint"]), "sha256": "recorded inside v46 checkpoint chain; not rehashed",
                            "bytes": row["bytes"], "role": f"committed {row['group']} result; input={row['input_sha256']}; C={row['C_sha256']}"})
    write_csv(OUT / "SOURCE_AUDIT.csv", source_list, ["source", "sha256", "bytes", "role"])

    # A compact machine-readable reconciliation for report generation.
    block_total = sum(float(r["seconds"]) for r in block_rows)
    v = {
        "status": "COMPLETE_READ_ONLY_AUDIT",
        "gpu_used": False,
        "formal_blocks": len(block_rows),
        "formal_linears": len(module_rows),
        "formal_rows": int(row_hist.sum()),
        "accepted_updates": update_sum_check,
        "candidate_evaluations": sum(int(r["candidate_evaluations"]) for r in module_rows),
        "rejected_candidates": int(row_hist.sum()*0 + sum(g["rejects"] for g in layer_mod.values())),
        "block_wall_seconds": block_total,
        "exclusive_stage_timer_seconds": stage_exclusive,
        "unattributed_block_seconds": unattributed,
        "nested_fp64_recheck_seconds": stage["independent_FP64_increment_recheck"]["seconds"],
        "direct_recompute_calls": stage["direct_recompute"]["calls"],
        "direct_initial_loss_seconds": direct_initial,
        "direct_proposal_seconds": direct_proposals,
        "direct_final_loss_seconds": direct_final,
        "candidate_score_min_seconds": stage["candidate_scoring"]["seconds"],
        "accepted_update_restore_seconds": stage["accepted_update_and_rejection_restore"]["seconds"],
        "C_construction_seconds": stage["C_construction"]["seconds"],
        "input_capture_seconds": stage["dependency_input_capture"]["seconds"],
        "prefix_refresh_seconds": stage["down_commit_prefix_refresh"]["seconds"],
        "candidate_direct_calls": proposal_batches,
        "candidate_proposed_rows": proposal_rows,
        "initial_search_mse_sum": initial_total,
        "final_search_mse_sum": final_total,
        "search_change_pct": 100*(final_total/initial_total-1),
        "row_updates_32": int(row_hist[32]),
        "max_updates_observed": max(i for i, c in enumerate(row_hist) if c),
        "rows": int(row_hist.sum()),
        "analysis_seconds": time.perf_counter()-started,
        "formal_protocol_sha256": protocol_sha,
    }
    (OUT / "SUMMARY.json").write_text(json.dumps(v, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(v, indent=2))


if __name__ == "__main__":
    main()
