# v17 run report

Run ended at the M2 alpha=1 native parity gate. Strict parity failed before any candidate alpha evaluation. Per the frozen stop rule, no later experiments or cost-based GO decision were performed.

The only supported verdict is **FORMAL_EXECUTION_NOT_READY**. The detailed comparison and diagnostics are in `M2_PARITY.json`, `M2_PARITY_DIAGNOSTIC.json`, and `M2_PARITY_TRACE.json`. The remainder of the checklist is explicitly marked not run in `correctness_checks.json`.

The v17 source copy and all newly written records are isolated under this result directory. v8-v16 files were not intentionally modified; the v16 delivered artifact/source hash checks are summarized in `HISTORICAL_INTEGRITY_CHECK.json` and `OFFICIAL_SOURCE_PROVENANCE.json`.
