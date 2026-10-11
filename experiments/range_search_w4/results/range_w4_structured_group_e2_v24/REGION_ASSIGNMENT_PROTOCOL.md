# E2 Region Assignment Protocol

**Frozen before E2 candidate scoring.** Use the exact v22 E2 group-index partition. v22 froze a concrete contiguous assignment, so the v24 fallback to select a statistic/threshold is not invoked. No M3 quality result, search loss, held loss, activation label, or threshold is used to assign groups.

- For input width 896, G128 creates 7 groups: Region A = groups 0–2; Region B = groups 3–6.
- For input width 4864, G128 creates 38 groups: Region A = groups 0–18; Region B = groups 19–37.
- The assignment is identical for every output row at the same target shape. Each region receives its own shared alpha for the row.
- The group order is native input-column order under `desc_act=False` and `static_groups=False`. A candidate changes the alpha hook at native dynamic group entry and reruns GPTQ compensation from the pristine row.

Pristine group max-absolute weight/range, alpha=1 group reconstruction error, and search-H diagonal summaries may be reported descriptively after the assignment is locked. They do not change A/B membership. Held data is excluded from region definition and all search decisions.
