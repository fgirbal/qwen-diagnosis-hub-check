# Judge report

Generator: Claude Opus (separate agent, fixed spec). Judge: Claude Sonnet (fresh contexts, blind to intended labels and to
any Qwen output). J3/J4 are scripted. Pass rules are in PREREGISTRATION.md §5.

| Set | Round 0 pass | Round 1 pass (final) |
|---|---|---|
| S0 | 10/10 | 10/10 |
| S1 | 40/40 | 40/40 |
| S2 | 40/40 | 40/40 |
| S3 | 15/30 | 30/30 |

**Round 0 failures:** S3-01 to S3-15 (the original's wording minus pregnancy) failed J1 on confidence only. The judge named
a biliary cause (not a pregnancy condition) with confidence 2, below the required 3. The generator was told only that, and
added one finding to each (7 cholecystitis, 4 biliary colic, 4 hepatitis A). A fresh judge rated all 15 at confidence >= 3.

**J2 (S1, same clinical picture as the original):** 40/40 yes. Judge's view of the
original: "Preeclampsia with severe features (evolving HELLP syndrome)". Caveats the judge raised (all still "yes"):
- Swelling of the face or hands as the only supporting finding: S1-12, S1-15, S1-21, S1-24, S1-28, S1-35.
- Early gestation (21-22 weeks): S1-22, S1-33.

**Per-item verdicts:** `report_round1.json` (final) and `report_round0.json`. Inputs, keys and raw judge outputs are in this folder.
