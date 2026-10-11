#!/usr/bin/env python3
"""Build archive indexes without reading local-only artifact contents.

Run from the repository root. Existing artifact hashes are reused from the
archive manifest. New local-only artifacts receive NOT_COMPUTED; this script
does not hash checkpoint/cache bytes or reopen full token-ID manifests.
"""
from __future__ import annotations

import csv
import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs" / "experiment_archive"
RESULT_ROOT = ROOT / "experiments" / "range_search_w4" / "results"
EXTRA_RESULTS = [
    ROOT / "results" / "v61_m3_calibration_cost_memory",
    ROOT / "results" / "v62_m3_vs_repaired_quality_cost",
]
SKIP_DIRS = {
    "__pycache__", "site-packages", "upstream", "official_source",
    "omniquant_source", ".git", ".pytest_cache", ".cache",
}
ARTIFACT_SUFFIXES = {
    ".safetensors", ".pt", ".pth", ".bin", ".f16", ".npy", ".npz", ".mat",
}
VERSION_RE = re.compile(r"(?:^|_)v(\d+)(?:_(\d+))?(?:_|$)", re.I)


def walk_files(root: Path):
    if not root.exists():
        return
    for current, dirs, names in os.walk(root):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS and not d.startswith(".venv")]
        base = Path(current)
        for name in names:
            p = base / name
            if p.is_file():
                yield p


def short_text(path: Path, limit: int = 360) -> str:
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    kept = []
    for line in lines:
        s = line.strip()
        if not s or s.startswith(("#", "|", "```", "---")):
            continue
        kept.append(s)
        if len(" ".join(kept)) >= limit:
            break
    return re.sub(r"\s+", " ", " ".join(kept))[:limit]


def first_title(path: Path) -> str:
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.lstrip().startswith("#"):
            return line.lstrip("# ").strip()
    return ""


def find_report(directory: Path) -> Path | None:
    candidates = [p for p in directory.rglob("*.md")
                  if not any(x in SKIP_DIRS or x.startswith(".venv") for x in p.parts)
                  and any(k in p.name.lower() for k in ("report", "decision", "summary"))]
    if not candidates:
        return None
    preference = {"report.md": 0, "run_report.md": 1, "decision.md": 2,
                  "full10_run_report.md": 3}
    candidates.sort(key=lambda p: (len(p.relative_to(directory).parts),
                                   preference.get(p.name.lower(), 5), str(p).lower()))
    return candidates[0]


def version_id(name: str) -> str:
    m = VERSION_RE.search(name)
    if not m:
        return "unversioned-precheck" if "candidate_budget" in name else "unversioned"
    return f"v{m.group(1)}" + (f".{m.group(2)}" if m.group(2) else "")


def structured_status(directory: Path, report: Path | None) -> str:
    # Prefer a root run ledger when it has an explicit status field.
    for name in ("RUN_STATUS.json", "FINAL_STATUS.json", "DECISION.json"):
        p = directory / name
        if p.exists():
            try:
                obj = json.loads(p.read_text(encoding="utf-8"))
                for key in ("status", "verdict", "state", "completion_status"):
                    if key in obj and isinstance(obj[key], (str, int, float, bool)):
                        return f"{key}={obj[key]} ({p.relative_to(ROOT).as_posix()})"
            except Exception:
                pass
    if report:
        for line in report.read_text(encoding="utf-8", errors="replace").splitlines()[:100]:
            m = re.search(r"(?i)\b(status|verdict|decision|result)\b\s*[:：]\s*([^|]{1,180})", line)
            if m:
                value = re.sub(r"[*`]+", "", m.group(2)).strip()
                return f"{m.group(1).lower()}={value} ({report.relative_to(ROOT).as_posix()})"
    return "未统一编码；按链接原报告及停止记录核对"


def index_row(directory: Path, label: str | None = None):
    report = find_report(directory)
    if report:
        title = first_title(report)
        summary = short_text(report)
        report_ref = report.relative_to(ROOT).as_posix()
        status = structured_status(directory, report)
    else:
        title = directory.name.replace("_", " ")
        summary = "该目录没有独立 Markdown 报告；请以归档的原始结果和协议文件为准。"
        report_ref = "NOT_FOUND"
        status = "无独立报告；原始结果保留"
    reproduce = next((p for p in sorted(directory.rglob("REPRODUCE.md"))
                      if not any(x in SKIP_DIRS or x.startswith(".venv") for x in p.parts)), None)
    if reproduce:
        reproduce_ref = reproduce.relative_to(ROOT).as_posix()
    else:
        scripts = [p for p in sorted(directory.glob("*.py"))]
        reproduce_ref = scripts[0].relative_to(ROOT).as_posix() if scripts else "见报告/协议；无独立复现入口"
    return {
        "version_or_id": label or version_id(directory.name),
        "directory": directory.relative_to(ROOT).as_posix(),
        "research_question": title,
        "execution_status": status,
        "primary_result_from_report": summary,
        "report": report_ref,
        "reproduction_entry": reproduce_ref,
    }


def write_csv(path: Path, rows, fields):
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def build_index():
    rows = []
    if RESULT_ROOT.exists():
        dirs = sorted((p for p in RESULT_ROOT.iterdir() if p.is_dir()),
                      key=lambda p: (int(VERSION_RE.search(p.name).group(1))
                                     if VERSION_RE.search(p.name) else 999,
                                     VERSION_RE.search(p.name).group(2) or ""
                                     if VERSION_RE.search(p.name) else "", p.name))
        rows.extend(index_row(d) for d in dirs)
    rows.append(index_row(EXTRA_RESULTS[0], "v61"))
    rows.append(index_row(EXTRA_RESULTS[1], "v62"))
    rows[-2]["execution_status"] = "COMPLETE: one full repository M3/LWC calibration measurement; see v61 report"
    fields = ["version_or_id", "directory", "research_question", "execution_status",
              "primary_result_from_report", "report", "reproduction_entry"]
    write_csv(OUT / "EXPERIMENT_INDEX.csv", rows, fields)


def artifact_reason(path: Path) -> tuple[str, str]:
    low = path.as_posix().lower()
    name = path.name.lower()
    if name in {"wikitext_test_manifest.json", "c4_eval_manifest.json"}:
        return ("tokenized evaluation input manifest", "Excluded to avoid uploading a duplicate tokenized corpus; frozen source revision and hashes are recorded in reports. Rebuild with the referenced dataset revision and tokenizer.")
    if name in {"p0_tokens.json", "p1_tokens.json", "p2_tokens.json", "p3_tokens.json", "tokens_by_group.json"}:
        return ("split token cache", "Dataset-derived token cache excluded; reconstruct from the referenced split manifest and tokenizer. No external download link is asserted here.")
    if path.suffix.lower() == ".zip" and "kaggle_bundle_v20_1" in name:
        return ("duplicate Kaggle source/data bundle", "Excluded as a duplicate archive that may package local source/data; rebuild from the v20.1 packaging instructions. No public link is asserted.")
    if "dense_export" in low and name.startswith("tokenizer"):
        return ("model-export tokenizer duplicate", "Regenerable from the recorded tokenizer revision; kept local to avoid duplicating model download assets.")
    if path.suffix.lower() == ".safetensors" or "checkpoint" in low or "dense_export" in low:
        return ("model checkpoint or dense export", "仅本地保存；由对应冻结协议、模型 revision、校准 manifest 及量化/修复源码重建；没有核实的外链可用。")
    if "hessian" in low or "hinv" in name or "gram" in name or "factor_cache" in low:
        return ("Hessian/Gram/factor cache", "仅本地保存；按该实验冻结输入及实现重算；不作为 Git 输入。")
    if any(k in low for k in ("activation", "_x.", "_input", "inputs/", "cache/")):
        return ("activation/input tensor cache", "仅本地保存；按对应冻结校准数据和前缀传播流程重捕获；不作为 Git 输入。")
    return ("binary tensor or optimizer artifact", "仅本地保存；按对应实验协议重新生成，或从本机已有 checkpoint/cache 路径恢复；无已核实公开下载链接。")


def artifact_experiment(path: Path) -> str:
    parts = path.relative_to(ROOT).parts
    if len(parts) >= 4 and parts[0] == "experiments":
        return parts[3]
    if len(parts) >= 2 and parts[0] == "results":
        return parts[1]
    return "unclassified"


def build_artifacts_and_manifest_index():
    roots = [RESULT_ROOT, *EXTRA_RESULTS]
    artifact_rows = []
    manifest_path = OUT / "DATA_INPUT_MANIFESTS.csv"
    # Preserve the previously prepared metadata-only summary. Rebuilding it
    # from its source JSON would reopen token-ID arrays and is unnecessary.
    if not manifest_path.exists():
        write_csv(manifest_path, [], [
            "source_manifest", "source_manifest_bytes", "source_manifest_sha256",
            "experiment", "dataset", "revision", "config", "split",
            "source_data_sha256", "tokenizer_revision", "token_sha256",
            "input_tokens", "evaluated_tokens", "max_length", "stride",
            "documents_count", "raw_token_ids_uploaded",
        ])
    previous_path = OUT / "LARGE_ARTIFACTS_MANIFEST.csv"
    previous = {}
    if previous_path.exists():
        with previous_path.open(newline="", encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                previous[row.get("relative_path", "")] = row
    seen = set()
    for root in roots:
        for path in walk_files(root) or ():
            rel = path.relative_to(ROOT)
            rel_posix = rel.as_posix()
            if "p3" in path.name.lower() or "p3" in rel_posix.lower():
                # Keep P3-derived inputs outside this archive walk.
                continue
            name_lower = path.name.lower()
            is_eval_manifest = name_lower in {"wikitext_test_manifest.json", "c4_eval_manifest.json"}
            is_token_cache = name_lower in {"p0_tokens.json", "p1_tokens.json", "p2_tokens.json", "tokens_by_group.json"}
            is_export_tokenizer = "dense_export" in rel_posix.lower() and name_lower.startswith("tokenizer")
            is_tensor = path.suffix.lower() in ARTIFACT_SUFFIXES
            is_bundle_zip = path.suffix.lower() == ".zip" and "kaggle_bundle_v20_1" in name_lower
            if is_tensor or is_eval_manifest or is_token_cache or is_export_tokenizer or is_bundle_zip:
                if rel_posix not in seen:
                    role, rebuild = artifact_reason(path)
                    old = previous.get(rel_posix, {})
                    artifact_rows.append({
                        "relative_path": rel_posix,
                        "bytes": path.stat().st_size,
                        "sha256": old.get("sha256") or "NOT_COMPUTED",
                        "hash_source": old.get("hash_source") or (
                            "legacy archive manifest; provenance not recorded"
                            if old.get("sha256") else "NOT_COMPUTED"
                        ),
                        "hash_reverified_this_round": old.get(
                            "hash_reverified_this_round", "NO"
                        ),
                        "experiment": artifact_experiment(path),
                        "purpose": role,
                        "availability_or_rebuild": rebuild,
                        "git_upload_status": old.get("git_upload_status", "NOT_UPLOADED"),
                        "compressed_git_copy": old.get("compressed_git_copy", "NONE"),
                    })
                    seen.add(rel_posix)
    artifact_rows.sort(key=lambda r: r["relative_path"].lower())
    write_csv(OUT / "LARGE_ARTIFACTS_MANIFEST.csv", artifact_rows,
              ["relative_path", "bytes", "sha256", "hash_source",
               "hash_reverified_this_round", "experiment", "purpose",
               "availability_or_rebuild", "git_upload_status", "compressed_git_copy"])
    print(f"index_rows={sum(1 for _ in csv.DictReader((OUT / 'EXPERIMENT_INDEX.csv').open(encoding='utf-8-sig')))}")
    print(f"excluded_artifacts={len(artifact_rows)} excluded_bytes={sum(r['bytes'] for r in artifact_rows)}")
    print(f"evaluation_manifest_summaries_preserved={manifest_path.exists()}")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    build_index()
    if "--refresh-artifact-inventory" in sys.argv[1:]:
        build_artifacts_and_manifest_index()
    else:
        print("large artifact inventory preserved; pass --refresh-artifact-inventory to enumerate paths/sizes (no content hashes are computed)")
