# Pre-registration: is Qwen3-32B's "preeclampsia hub" a diagnosis hub?

Status: final. Hashed into `results/prereg.sha256` before any run of Qwen3-32B. It is followed as written;
changes are logged under Deviations, with reasons, before the affected step runs.

## 1. Claim under test

*Neural Networks Are Already Clean* (Goodfire, 2026), "Exploring Language Models": "The diagnosis hub is just
four neurons, and if you remove them the model asks about itching instead." The four neurons are positive-pole
MLP neurons in Qwen3-32B: **L31:5482, L35:3336, L35:20744, L7:2026** (group "preeclampsia" in the essay's
published graph data).

The hub was identified by attribution on a single prompt (§3). Removing neurons picked because they matter on
a prompt is expected to change that prompt's answer. The test asks whether the four neurons behave like a
*preeclampsia diagnosis* hub across cases, or like a response to this prompt's wording and context.

## 2. Hypotheses

- **Hub (diagnosis) account:** the neurons fire when a case points to preeclampsia, whatever the wording, and
  not when the same words point elsewhere. Removing them shifts preeclampsia-specific behaviour across
  differently worded preeclampsia cases, beyond what removing comparable neurons does, and doesn't
  specifically disturb non-preeclampsia cases.
- **Prompt-specific (wording) account:** their activity tracks the original's wording and the pregnancy
  context more than the diagnosis, and removing them has little effect on preeclampsia cases written
  differently.

## 3. Known before this pre-registration

These are facts from the essay's published data, saved in `data/published/`:

- **Prompt (79 tokens):** `Human: A 32-year-old female at 30 weeks gestation presents with severe right upper
  quadrant pain, mild headache, and nausea. BP is 162/98 mmHg, and labs show mildly elevated liver
  enzymes.\n\nIf we can only ask about one other symptom, we should ask whether she's experiencing...\n\nAssistant:
  We should ask whether she's experiencing **`.
- **Published results:**
  - Intact: P(visual) 0.903, P(protein) 0.058.
  - Hub removed: P(visual) 0.0005, with the continuation "pruritus (itching)**…".
  - Their random control (as many neurons, same layers): P(visual) 0.900.
  - Removing the 2 "microangiopathic syndromes" neurons: P(visual) 0.181.
  - Removing the 14 "pregnancy" neurons: P(visual) 0.187.
- **Hub firing positions:** peak at the assistant's " she" (token 75; L31:5482 = 46.5). L31:5482 already
  reaches 45.0 at the comma after "headache" (token 26), before "BP" (token 30) appears.
- **Smoke test on Qwen3-0.6B (same tokenizer, pipeline test only):** the prompt tokenizes exactly to the
  published 79 tokens. All scripts ran end to end. Its numbers are not data about Qwen3-32B and are not reported
  as such.
- **The stimuli and their judge reports (§5) exist before this document is hashed.** No Qwen3-32B output on
  any of them has been seen.

## 4. Model, neurons and removal

- **Model:** `mlx-community/Qwen3-32B-4bit` on an Apple M1 Max (32 GB) with MLX. Neither the bf16 model
  (65 GB) nor the 8-bit model fits in memory, so 4-bit fidelity is checked in Stage 0 before anything else.
- **Neuron:** index *i* of the SwiGLU product `silu(gate_proj(x)) * up_proj(x)` in layer *L*'s MLP, before
  `down_proj`.
- **Removal:** that product is set to 0 at every position (mode "zero"). If Stage 0's V0c fails under "zero",
  the fallback is "zero_pos" (only positive values set to 0). The first mode that passes is frozen. A
  positions-only removal is not considered, because it can't be applied to new cases.
- **Controls,** with all 54 neurons in the published preeclampsia graph excluded:
  - **C-layer** (the essay's own control): 4 random neurons from the same layers (1 × L7, 1 × L31, 2 × L35).
  - **C-act** (primary control): for each hub neuron, a random neuron from the same layer whose maximum
    activation over the original prompt is within ±30% of that hub neuron's. If there are fewer than 5
    candidates, the window widens to ±50%, and then to the adjacent layers.
  - K draws of each, with seed 20261019. K = 20, or 10 if Stage 0's timing projects the main run at more than
    6 hours.

## 5. Stimuli

The generator (Claude Opus, a separate agent working to a fixed spec) writes the stimuli. Checks:

- Scripted: J3 (twin minimality) and J4 (banned words and diagnosis names, substrings).
- An independent judge (Claude Sonnet, fresh contexts, never shown intended labels or Qwen outputs):
  - **J1:** blind most-likely diagnosis with confidence, on shuffled cases with neutral IDs.
  - **J2:** for S1 only, whether each case gives the same clinical picture as the original.
- **J1 pass rules (confidence ≥ 3):**
  - S0/S1: preeclampsia or eclampsia, with HELLP accepted.
  - S2: matches the twin's intended alternative (its keywords) and isn't in the preeclampsia family.
  - S3: not a pregnancy condition.
- **J2 pass rule:** "yes".
- **Rewrites:** failing items are rewritten by the generator, which is told only which check failed, for at most
  two rounds; then they are dropped. S0 items are exempt (they are the original and near-copies of it): their
  verdicts are reported and they are kept.
- **Reporting:** every verdict is reported in `stimuli/judge/`.
- **Outcome (before hashing):** all 120 cases pass after one rewrite round (S3-01–15). J2 was 40/40 "yes",
  with the judge's caveats recorded in `stimuli/judge/JUDGE_REPORT.md`.

| Set | n | Content |
|---|---|---|
| S0 | 10 | The original (S0-01) and near-copies that change only age, week and BP (still in the preeclampsia range) |
| S1 | 40 | Preeclampsia (over 20 weeks, BP ≥ 140/90, a supporting finding), with none of the original's distinctive words and no visual symptoms or abdominal pain |
| S2 | 40 | Lexical twins: S2-01–10 of S0-01–10, S2-11–40 of S1-01–30. Each is its partner with a normal BP and one finding pointing to a non-preeclampsia diagnosis (cholecystitis, biliary colic, pyelonephritis, hepatitis A, appendicitis, UTI) |
| S3 | 30 | Non-pregnant women with high BP and overlapping symptoms. S3-01–15 reuse the original's wording minus pregnancy, plus one finding added in judge round 1 (cholecystitis, biliary colic or hepatitis A) because the judge's confidence without it was 2/5. |
| S4 | 40 | Plain sentences: 15 naming preeclampsia or eclampsia, 15 other pregnancy conditions, 10 non-pregnancy hypertension |

Every case is used in two frames. The question and answer stem are fixed, and only the case changes:
- **Ask frame** (the original's): `Human: {case}\n\nIf we can only ask about one other symptom, we should ask
  whether she's experiencing...\n\nAssistant: We should ask whether she's experiencing **`
- **Dx frame:** `Human: {case}\n\nWhat is the most likely diagnosis?\n\nAssistant: The most likely diagnosis is **`

**Manipulation check (inclusion):** a case is analysed only if the intact model's greedy dx answer matches its
intended label:
- S0/S1: contains "eclampsia".
- S2: not preeclampsia-family or HELLP, and contains one of its keywords.
- S3: not preeclampsia-family or HELLP.

Exclusions are reported per set, along with how often the judge and the intact model agree.

## 6. Measurements

- **Ask frame:** P(visual), the probability of the first token of "visual" at the answer position, plus the
  first-token mass of {visual, vision, blurred, protein, swelling, edema, seizures} and the total variation
  distance (TV) of the first-token distribution from the intact model's.
- **Dx frame:** the greedy 12-token answer, labelled preeclampsia-family if it contains "eclampsia", hellp if it
  contains "hellp", and other otherwise.
- **Hub activation:** all positions, intact model, ask frame. The **post-case window** runs from the first
  token after the case text to the end. A neuron enters P1 only if its post-case maximum on the original prompt
  is ≥ 20% of its maximum over the whole prompt (fixed in Stage 0 and reported).
- **Conditions per case and frame:** intact; hub removed; K × C-layer; K × C-act.

## 7. Stage 0 gate (runs first; stop if it fails)

| Check | Pass criterion |
|---|---|
| V0a | Prompt text and tokens equal the published 79 |
| V0b | Intact P(visual) in [0.75, 0.98], and the greedy continuation starts "visual disturbances" |
| V0c | Hub removed: P(visual) ≤ 0.05, and the continuation names itching or pruritus (mode fallback as in §4) |
| V0d | ≥ 18 of 20 C-layer draws keep P(visual) ≥ 0.75 |
| V0e | Spearman ≥ 0.8 between our and the published hub activations at the published positions; L31:5482 at token 75 within ±30% of 46.5 |
| V0f | Reported only: removing the 2 microangiopathic neurons gives P(visual) in [0.05, 0.45] |

V0a–V0e must all pass. Otherwise nothing further runs, and the result is reported as "4-bit model not a faithful
stand-in".

## 8. Predictions and decision rules

**P1, diagnosis or wording.** For each included twin whose partner is also included: ratio = twin hub level /
partner hub level. The hub level is the mean, over included neurons, of the post-case-window maximum divided
by the same on the original prompt.
- Diagnosis: mean ratio ≤ 0.5, with the 95% bootstrap CI's upper bound < 0.75.
- Wording: mean ratio ≥ 0.8.
- Otherwise intermediate.
- Also reported per set: the fraction of each neuron's peak reached before the BP appears.

**P2, generalization (S1, included).**
- *Ask frame:* the relative drop is (P_intact(visual) − P_removed(visual)) / P_intact(visual), over cases with
  intact P(visual) ≥ 0.2.
- *Dx frame:* the switch rate is the fraction of cases whose greedy answer leaves the preeclampsia family when
  the hub is removed.
- Hub: mean drop ≥ 0.5 and above the 95th percentile of the K C-act draw means; and switch rate ≥ 0.5 and
  above the 95th percentile of the C-act switch rates.
- Prompt-specific: mean drop < 0.25, or ≤ the C-act 95th percentile.
- Otherwise intermediate.

**P3, specificity (S2 and S3, included).** Specific if the mean TV under hub removal is ≤ the 95th percentile
of the C-act draw means. The change in dx label is also reported.

**P4, label check (S4).** The mean hub level (each neuron's maximum divided by its whole-prompt maximum on the
original) on preeclampsia or eclampsia sentences, divided by the same on other-pregnancy sentences, is ≥ 3.

Any P1–P3 test with fewer than 10 included cases (or pairs) is reported as underpowered.

**Decision.**
- Hub account *supported*: P1 = diagnosis, P2 = hub and P3 = specific.
- Hub account *refuted* (prompt-specific): P2 = prompt-specific and P1 = wording.
- Otherwise *mixed*, reported test by test.

C-layer results are reported alongside but don't enter the decision.

## 9. Procedure

1. **Hash:** this document, `stimuli/stimuli.json`, the judge reports and every file in `src/` are hashed
   (SHA-256, UTC timestamp) into `results/prereg.sha256` before Stage 0.
2. **Stage 0** (§7).
3. **Debug run:** the stand-in group (the 2 microangiopathic neurons) in place of the hub, on 10 random cases
   per set (seed 20261019), with K = 5 reusing the first 5 draws of each control. Debug results are kept and
   labelled.
4. **Main run:** the hub, all cases and K from Stage 0. This is the first and only run of the hub on the
   stimuli.
5. **Analysis:** `src/s2_analysis.py` exactly as hashed.

Runs: `caffeinate -is zsh run_all.sh` (Stage 0 → debug → main → analysis; it stops at a failed gate).

## 10. Limitations known in advance

- A 4-bit model is a stand-in for the published bf16 model. Stage 0 checks fidelity on the published prompt
  only.
- The generator and judge are from the same model family (Claude).
- The diagnosis readout is a greedy string label, not a probability.
- The twins change both the BP and one finding, so P1 can't separate the two.
- Activity before the BP appears in a case can only reflect what precedes it.

## Deviations

**D1 (logged 2026-10-19, about 23:00 UTC, after the 4-bit Stage 0 and before any 8-bit or bf16 run).**

*What happened.* Stage 0 ran on the 4-bit model as pre-registered (`results/stage0.json`, `results/run_all.log`).
The gate failed at V0c:
- Removing the hub gave P(visual) 0.123 ("zero") and 0.139 ("zero_pos"), not ≤ 0.05.
- The continuation was "dark urine" instead of itching or pruritus.

The other gate checks passed:
- V0a: identical 79 tokens.
- V0b: P(visual) 0.860, continuing "visual disturbances".
- V0d: 20/20 same-layer controls ≥ 0.75 (min 0.81).
- V0e: Spearman 0.97, and L31:5482 = 45.0 against the published 46.5.

V0f failed: removing the 2 microangiopathic neurons gave 0.809 against the published 0.181. As pre-registered, nothing further ran on the 4-bit model.

*Change.* Stage 0 is re-run, with identical checks, thresholds and fallback order:
1. First on `mlx-community/Qwen3-32B-8bit`.
2. If 8-bit fails the gate, on `mlx-community/Qwen3-32B-bf16`, the precision the essay used. The 8-bit weights are
   deleted first to free disk, and the 4-bit weights too if there still isn't room.

The first model that passes the gate is used for the debug run, the main run and the analysis exactly as
pre-registered. If neither passes, the study stops. It is then reported as: the essay's knockout is not reproduced
by zero-ablation of the four neurons at 4-bit, 8-bit or bf16. Results go to `results/8bit/` and `results/bf16/`.

*Code changes (needed because neither model fits in the 32 GB of memory).*
- `src/hubcommon.py` can stream the model: weights are read lazily from the safetensors files one layer at a
  time and dropped after use (HUB_STREAM=1).
- `Probe.run` accepts prompts of different lengths. They are left-padded, and the pads are masked from
  attention.
- `src/s1_run.py` packs several cases into one batch, about 252 sequences, so each pass over the weights serves
  many prompts.
- `src/s0_validity.py` projects the main-run time for that procedure when applying the K rule (K = 20 unless the
  projection exceeds 6 hours). It also uses the new return signature of `run`.
- `run_stage.sh` and `overnight.sh` run the sequence.
- No change to the stimuli, measurements, predictions, thresholds or analysis code.

*Parity test (`src/parity_test.py`, `results/parity/`), run on the 4-bit model, which fits in memory.*
- Streaming is bit-identical to holding the model in memory: probabilities, activations and greedy tokens differ
  by 0.0, including in padded batches.
- At batch size 1, the new code reproduces the 4-bit Stage 0 numbers exactly (P(visual) 0.860; L31:5482 = 45.0).
- Changing the pad token's identity changes nothing (difference 0.0), so pads are fully masked.
- Batch shape and padding move P(visual) by up to about 0.04, through bf16 rounding of shifted positions.
  Examples: intact 0.845–0.874 across batch sizes; 0.829 with 9 pads; hub removed 0.110–0.123.
- All conditions of a case share one batch and one padding, so within-case comparisons (P2, P3) share the same
  shape. Cross-case comparisons (P1) carry this small noise, which is reported.

*Outcome of D1 so far.* The 8-bit Stage 0 gate passed (`results/8bit/stage0.json`):
- V0b: P(visual) 0.912, continuing "visual disturbances** (e.g., blurred vision, photophobia".
- V0c: hub removed gives P(visual) 0.00048 and "pruritus (itching)**, particularly **without a rash**".
- V0d: 20/20.
- V0e: Spearman 0.999; L31:5482 = 47.25.
- V0f passed as well.
- K = 20.

The study therefore runs on the 8-bit model, and bf16 is not needed.

**D2 (logged 2026-10-20, about 00:45 UTC, after the debug run on the stand-in group and before any main-run
result was inspected; the main run started at 00:40 UTC).**

*Problem.* `diagnosis_label` checked "hellp" before "eclampsia". An answer such as "preeclampsia** with possible
**HELLP syndrome" was labelled hellp instead of preeclampsia-family. That contradicts §6, which defines the family
as answers containing "eclampsia" and doesn't give HELLP precedence. It also contradicts §5's manipulation check
for S0/S1 ("contains 'eclampsia'"). The debug run exposed it: 6 of its 10 S0 cases, including the original prompt
S0-01, were excluded for this reason alone.

*Fix.*
- "eclampsia" is checked first.
- `src/s2_analysis.py` recomputes every dx label from the stored answer text, so the main run (already started
  with the old function) needs no rerun.

No other change. The decision rules, thresholds and the remaining manipulation-check rules are as registered.
HELLP-only answers still count as leaving the family for P2, as §8 states.

**Post-run note (2026-10-20, after the main run; not hashed).** The main run (8-bit, K = 20, 120 cases and 40
sentences) and the analysis finished at 06:27 UTC.

| Test | Result | Outcome |
|---|---|---|
| P1 | twin/partner ratio 0.18 [0.14, 0.23] | diagnosis |
| P2 | ask-frame drop 0.61 (C-act 95th percentile 0.006); dx switch 0.094 (C-act 95th percentile 0.031) | intermediate |
| P3 | TV 0.180 vs C-act 95th percentile 0.148 | not specific |
| P4 | ratio 4.7 | pass |

Decision: **mixed**. Details are in `RESULTS.md`.
