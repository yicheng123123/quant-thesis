# v21.4 gate smoke setup attempts

All attempts below used only the v21.2 frozen layer-21 fixture and disposable optimizer/scaler objects. No formal checkpoint was committed.

1. `gate_smoke_01`: stopped before fixture import because the v21.2 audit helper directory was not on `sys.path`.
2. `gate_smoke_02`: stopped during helper import because a local `termcolor` stub lacked `__spec__`.
3. `gate_smoke_03`: reached the v21.2 L21 sample-0 forward, finite loss, and native GradScaler step/update; the v21.4 post-step callback then raised `NameError` because `math` was not imported. The callback fix is recorded in the v21.4 runner patch manifest. The disposable sample did not commit a checkpoint.

These were harness/setup failures, not model quality results. The next attempt uses a fresh diagnostic directory after the source and lock hashes are refreshed.
