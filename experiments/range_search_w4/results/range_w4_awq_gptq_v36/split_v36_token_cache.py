"""Materialize per-group token cache files so runtime opens held only after lock."""
import hashlib, json
from pathlib import Path

HERE = Path(__file__).resolve().parent
CACHE = HERE / "cache"
source = json.loads((CACHE / "tokens_by_group.json").read_text(encoding="utf-8"))
manifest = json.loads((HERE / "manifests/ARTICLE_MANIFEST.json").read_text(encoding="utf-8"))
expected = {g: [int(x["source_article_index"]) for x in manifest["groups"][g]] for g in ("P0", "P1", "P2", "P3")}
out = {}
for group in expected:
    rows = source[group]
    if [int(x["source_article_index"]) for x in rows] != expected[group]:
        raise RuntimeError(f"article order mismatch for {group}")
    for row in rows:
        token_sha = hashlib.sha256(json.dumps(row["token_ids"]).encode("utf-8")).hexdigest()
        if token_sha != row["token_sha256"]:
            raise RuntimeError(f"token id hash mismatch for article {row['source_article_index']}")
    path = CACHE / f"{group}_tokens.json"
    payload = json.dumps(rows, separators=(",", ":"), ensure_ascii=False) + "\n"
    if path.exists() and path.read_text(encoding="utf-8") != payload:
        raise RuntimeError(f"refusing to overwrite differing group cache {group}")
    if not path.exists():
        path.write_text(payload, encoding="utf-8", newline="\n")
    out[group] = {"path": path.relative_to(HERE).as_posix(),
                  "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                  "articles": len(rows), "opened_by_runtime_before_selection_lock": group in ("P0", "P2")}
manifest_path=HERE/"manifests/GROUP_TOKEN_CACHE_MANIFEST.json"
payload={"status":"FROZEN_BEFORE_MODEL_OR_GPU","groups":out,
         "held_groups_opened_by_runtime_only_after_split_selection_lock":["P1","P3"]}
encoded=json.dumps(payload,indent=2,ensure_ascii=False)+"\n"
if manifest_path.exists() and manifest_path.read_text(encoding="utf-8")!=encoded:
    raise RuntimeError("refusing to overwrite differing group-cache manifest")
if not manifest_path.exists(): manifest_path.write_text(encoded,encoding="utf-8",newline="\n")
print(json.dumps(payload, indent=2))
