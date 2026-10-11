"""Artifact-only summary. Never reruns model inference or edits raw row files."""

from __future__ import annotations

import csv
import argparse
import json
from pathlib import Path
import statistics as stats

HERE = Path(__file__).resolve().parent
PROTOCOL = json.loads((HERE / "protocol.json").read_text(encoding="utf-8"))


def write_new(path: Path, content: str) -> None:
    if path.exists():
        if path.read_text(encoding="utf-8") != content:
            raise FileExistsError(f"refusing to overwrite an existing summary: {path}")
        return
    path.write_text(content, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", default=PROTOCOL["id"])
    args = parser.parse_args()
    if not args.run_name.replace("_", "").replace("-", "").isalnum():
        raise ValueError("run-name must be a simple directory name")
    RESULTS = HERE / "results" / args.run_name
    manifest = json.loads((RESULTS / "manifest.json").read_text(encoding="utf-8"))
    rows = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(RESULTS.glob("layer*_row*.json"))]
    expected = {(layer, row) for layer in PROTOCOL["layers"] for row in PROTOCOL["row_indices"]}
    if {(r["layer"], r["row"]) for r in rows} != expected:
        raise ValueError("representative rows incomplete")
    hashes = [d["sha256"] for d in manifest["documents"]]
    if len(set(hashes)) != 8 or len({d["article_sha256"] for d in manifest["documents"]}) != 8:
        raise ValueError("document leakage in manifest")
    if any(r["document_hashes"] != hashes for r in rows):
        raise ValueError("row input hashes differ")

    summary_rows = []
    methods = ("A_minmax_fixed_gptq", "B_rtn_weight_fixed_gptq", "B_rtn_output_fixed_gptq", "C_gptq_output_fixed_gptq")
    for fmt, family in (("symmetric", "single_10"), ("asymmetric_format_control", "single_10"), ("asymmetric_format_control", "single_100"), ("asymmetric_format_control", "independent_100")):
        for method in methods:
            selected = [next(x for x in r[fmt] if x["method"] == family + "/" + method) for r in rows]
            baseline = [next(x for x in r[fmt] if x["method"] == family + "/A_minmax_fixed_gptq") for r in rows]
            search = [1 - a["search_output_mse"] / b["search_output_mse"] for a, b in zip(selected, baseline)]
            held = [1 - a["heldout_output_mse"] / b["heldout_output_mse"] for a, b in zip(selected, baseline)]
            summary_rows.append({
                "format": "symmetric" if fmt == "symmetric" else "asymmetric",
                "family": family, "method": method, "rows": len(rows),
                "candidate_evaluations_per_row": selected[0]["candidate_evaluations"],
                "search_median_improvement_percent": stats.median(search) * 100,
                "heldout_median_improvement_percent": stats.median(held) * 100,
                "heldout_wins": sum(v > 1e-12 for v in held),
                "heldout_losses": sum(v < -1e-12 for v in held),
                "heldout_ties": sum(abs(v) <= 1e-12 for v in held),
                "heldout_min_improvement_percent": min(held) * 100,
                "heldout_max_improvement_percent": max(held) * 100,
            })
    paired = []
    for method in methods[1:]:
        for comparator in ("single_10", "single_100"):
            a = [next(x for x in r["asymmetric_format_control"] if x["method"] == "independent_100/" + method) for r in rows]
            b = [next(x for x in r["asymmetric_format_control"] if x["method"] == comparator + "/" + method) for r in rows]
            gains = [1 - x["heldout_output_mse"] / y["heldout_output_mse"] for x, y in zip(a, b)]
            paired.append({"method": method, "comparison": "independent_100_vs_" + comparator, "heldout_wins": sum(v > 1e-12 for v in gains), "heldout_median_gain_percent": stats.median(gains) * 100, "heldout_losses": sum(v < -1e-12 for v in gains)})
    group = {
        "search_ranking_disagreements": sum(x["search"]["ranking_disagrees"] for r in rows for x in r["group_diagnostic"]["ranking_diagnostic"]),
        "heldout_ranking_disagreements": sum(x["heldout"]["ranking_disagrees"] for r in rows for x in r["group_diagnostic"]["ranking_diagnostic"]),
        "groups_compared": sum(len(r["group_diagnostic"]["ranking_diagnostic"]) for r in rows),
        "rtn_coordinate_accepted_updates": sum(len(r["group_diagnostic"]["coordinate_replace_one_sweep"]["accepted"]) for r in rows),
        "rtn_coordinate_heldout_wins_vs_independent": sum(r["group_diagnostic"]["coordinate_replace_one_sweep"]["heldout"]["full_mse"] < r["group_diagnostic"]["independent_group_search"]["heldout"]["full_mse"] for r in rows),
        "gptq_independent_heldout_wins_vs_minmax": sum(r["group_diagnostic"]["gptq_independent_external"]["heldout"]["full_mse"] < r["group_diagnostic"]["gptq_minmax_external"]["heldout"]["full_mse"] for r in rows),
    }
    symmetric_seconds = stats.median(r["symmetric"][0]["evaluation_seconds"] for r in rows)
    cost = {
        "total_wall_seconds": manifest.get("total_seconds_all_invocations", manifest["total_seconds_this_invocation"]),
        "capture_seconds": manifest["capture_seconds"],
        "hessian_seconds_by_layer": manifest["hessian_seconds_by_layer"],
        "capture_peak_allocated_bytes": manifest["capture_peak_cuda"]["allocated_bytes"],
        "capture_peak_reserved_bytes": manifest["capture_peak_cuda"]["reserved_bytes"],
        "search_peak_allocated_bytes": max(x["peak_allocated_bytes"] for r in rows for x in r["symmetric"] + r["asymmetric_format_control"]),
        "median_symmetric_10_candidate_seconds_per_q_proj_row": symmetric_seconds,
        "rough_full_transformer_linear_output_rows": 304128,
        "naive_rowwise_full_symmetric_10_candidate_hours": symmetric_seconds * 304128 / 3600,
        "estimate_caveat": "not a strict lower bound: batching across rows may reduce time, while 4864-input projections and capture/integration/packing may increase it",
    }
    result = {"protocol_id": PROTOCOL["id"], "document_hashes": hashes, "method_summary": summary_rows, "paired_asymmetric": paired, "group_diagnostic": group, "cost": cost, "row_failures": [
        {"layer": r["layer"], "row": r["row"], "finding": "RTN coordinate replacement worsens heldout full output MSE"}
        for r in rows if r["group_diagnostic"]["coordinate_replace_one_sweep"]["heldout"]["full_mse"] > r["group_diagnostic"]["independent_group_search"]["heldout"]["full_mse"]
    ]}
    write_new(RESULTS / "summary.json", json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    path = RESULTS / "summary.csv"
    if not path.exists():
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary_rows[0]))
            writer.writeheader()
            writer.writerows(summary_rows)
    print(json.dumps({"methods": len(summary_rows), "rows": len(rows), "group": group, "cost": cost}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
