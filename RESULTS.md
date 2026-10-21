# Results: is Qwen3-32B's "preeclampsia hub" a diagnosis hub?

Run 2026-10-20, 00:55–07:27 local time, on `mlx-community/Qwen3-32B-8bit`, streamed from disk on an M1 Max (deviation D1).
Pre-registration: `PREREGISTRATION.md` (deviations D1 and D2 were logged before the main results existed).
Raw results: `results/8bit/` (`stage0.json`, `run_main.jsonl`, `decisions_main.json`).

## Pre-registered verdict: **mixed**

| Test | Result | Outcome |
|---|---|---|
| **P1** diagnosis or wording (18 twin pairs) | Hub activity in the twins (same words, normal BP, a different diagnosis) is **0.18** of their preeclampsia partners' [95% CI 0.14, 0.23]. The diagnosis account needed ≤ 0.5. | **diagnosis** |
| **P2** generalization (32 reworded preeclampsia cases) | *Ask frame:* removing the hub lowers P(visual) by **61%** on average (C-act 95th percentile: 0.6%). *Dx frame:* the stated diagnosis leaves preeclampsia in only **9%** of cases (3/32; C-act 95th percentile 3%). The hub account needed both ≥ 50%. | **intermediate** |
| **P3** specificity (55 non-preeclampsia cases) | Removing the hub shifts the answer distribution by TV **0.18**, above the 95th percentile of activation-matched removals (0.15; their median is 0.08). It never changes the stated diagnosis. | **not specific** |
| **P4** label check (40 sentences) | Hub activity on sentences naming preeclampsia or eclampsia is **4.7×** that on other pregnancy conditions. | **pass** |

The rules give "supported" only if P1 = diagnosis, P2 = hub and P3 = specific, and "refuted" only if P2 = prompt-specific
and P1 = wording. Neither holds.

## Stage 0: the 8-bit model reproduces the essay

| | Essay (bf16) | 4-bit (failed gate) | 8-bit (passed) |
|---|---|---|---|
| Intact P(visual) | 0.903 | 0.860 | 0.912 |
| Hub removed, P(visual) | 0.0005 | 0.123 | 0.00048 |
| Hub removed, continuation | "pruritus (itching)…" | "dark urine…" | "pruritus (itching)**, particularly **without a rash**" |
| 2 microangiopathic neurons removed | 0.181 | 0.809 | passed (in [0.05, 0.45]) |
| Hub activations vs published (Spearman) | — | 0.97 | 0.999 |

## Descriptive (not pre-registered)

- **Original wording (S0, 7 cases):** removing the hub takes P(visual) from 0.895 to 0.022 on average, and the
  model asks about itching (5) or jaundice (2). The essay's demonstration holds across near-copies of its prompt.
- **Reworded preeclampsia (S1):** P(visual) falls from 0.739 to 0.293 on average. The greedy answer is still often
  visual disturbances, or switches to proteinuria or edema. The effect generalizes, but it is far weaker than on
  the original wording.
- **Diagnosis:** with the hub removed, the model still names preeclampsia in 29 of 32 reworded cases. The
  "diagnosis hub" is not necessary for the model to state the diagnosis. What it changes is which follow-up
  question the model picks.
- **Specificity:** on lexical twins the hub shifts answers about 3× as much as matched random neurons (mean TV
  0.24 vs 0.07). On non-pregnant cases the gap is smaller (0.13 vs 0.09). Example: S3-21's question changes from
  hematuria to oral ulcers.
- **Timing:** the hub neurons reach about 80% of their peak before the blood pressure appears in the original
  wording (S0), and about 40% in reworded cases (S1). In the twins, activity then falls once the normal BP and
  the other finding arrive.

## Reading

The four neurons carry a real preeclampsia signal. It tracks the clinical picture rather than the wording (P1), it
fires on text about preeclampsia (P4), and removing it matters far more than removing comparable neurons. That much
supports the essay's labels.

Calling it the diagnosis "hub" from one prompt overstates what it does, for three reasons:
- **Weaker away from the prompt.** The essay's single-prompt effect (0.90 to 0.0005) shrinks to an average 61% drop
  on reworded cases, which leaves P(visual) at about 0.29.
- **Not needed to diagnose.** Removing the hub almost never changes the model's stated diagnosis.
- **Not specific.** Its removal also perturbs answers on cases that aren't preeclampsia, more than matched neurons do.

This is the same pattern as the V-neuron study: a label that fits the examples chosen for it, and a mechanism story
that holds only partly once cases are varied independently.

## Exclusions and caveats

- **Manipulation check** (intact model's diagnosis must match the intended label): 25 of 120 cases excluded.
  - S0: 3, where the model diagnosed acute fatty liver of pregnancy.
  - S1: 8, mostly gestational hypertension.
  - S2: 15, where the model's diagnosis didn't match the twin's intended alternative.
  - S3: 0.
  - Full list in `decisions_main.json`.
- **8-bit, not bf16.** It reproduces the essay's published numbers almost exactly (Stage 0), but it isn't the
  identical model.
- **Batch-shape noise.** Batch shape and padding move P(visual) by up to about 0.04 (D1 parity test). Every
  condition of a case shares one batch.
- **P3 is close.** 0.18 against a cut-off of 0.15.
