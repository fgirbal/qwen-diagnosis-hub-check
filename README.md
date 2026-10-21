# How Targeted Is the Preeclampsia Diagnosis Hub?

A pre-registered test of whether four MLP neurons in Qwen3-32B, which the essay
[*Neural Networks Are Already Clean*](https://already-clean.goodfire.com/) calls a preeclampsia "diagnosis hub",
behave like a diagnosis hub across cases.

**Blog post:** [How targeted is the preeclampsia diagnosis hub?](https://fgirbal.github.io/blog/preeclampsia-hub/)

The essay rebuilds a medical-diagnosis attribution graph out of Qwen3-32B's own neurons. Its prompt describes a
woman 30 weeks pregnant with high blood pressure, pain under her right ribs and raised liver enzymes, and asks
which one symptom to check next. The model answers "visual disturbances". At the centre of the graph are four
positive-pole MLP neurons labelled preeclampsia: **L31:5482, L35:3336, L35:20744 and L7:2026**. In the essay's
words, "the diagnosis hub is just four neurons, and if you remove them the model asks about itching instead."

That claim rests on one prompt, and neurons picked because they matter on a prompt are expected to change that
prompt's answer. This repository tests the four neurons on 120 clinical cases and 40 sentences, against
activation-matched random neurons, with every prediction and decision rule fixed in advance. It contains the
code, the stimuli with every judge verdict, the raw results and the pre-registration.

- Full results and reading: [RESULTS.md](RESULTS.md)
- Pre-registration, with deviations D1–D2 and a post-run note: [PREREGISTRATION.md](PREREGISTRATION.md)

This is the second study in a series testing the essay's experiments. The first, on InceptionV1's "V neuron",
has its own repository: [fgirbal/v-neuron-test](https://github.com/fgirbal/v-neuron-test).

## Results

Run on 2026-10-20 on `mlx-community/Qwen3-32B-8bit`, streamed from disk on an Apple M1 Max (deviation D1).
The verdicts are in `results/8bit/decisions_main.json`.

| Test | Result | Outcome |
|---|---|---|
| **P1** diagnosis or wording (18 twin pairs) | Hub activity in the twins (same words, normal BP, a different diagnosis) is **0.18** of their preeclampsia partners' [95% CI 0.14, 0.23]. The diagnosis account needed ≤ 0.5. | **diagnosis** |
| **P2** generalization (32 reworded preeclampsia cases) | *Ask frame:* removing the hub lowers P(visual) by **61%** on average (C-act 95th percentile: 0.6%). *Dx frame:* the stated diagnosis leaves preeclampsia in only **9%** of cases (3/32; C-act 95th percentile 3%). The hub account needed both ≥ 50%. | **intermediate** |
| **P3** specificity (55 non-preeclampsia cases) | Removing the hub shifts the answer distribution by TV **0.18**, above the 95th percentile of activation-matched removals (0.15). | **not specific** |
| **P4** label check (40 sentences) | Hub activity on sentences naming preeclampsia or eclampsia is **4.7×** that on other pregnancy conditions. | **pass** |

**Pre-registered verdict: mixed.** The rules give "supported" only if P1 = diagnosis, P2 = hub and P3 = specific,
and "refuted" only if P2 = prompt-specific and P1 = wording. Neither holds.

In short, the four neurons carry a real preeclampsia signal that tracks the clinical picture rather than the
wording. Removing them changes which follow-up question the model asks far more than comparable neurons do. But
they are not needed for the model to state the diagnosis (it still names preeclampsia in 29 of 32 reworded
cases), the effect is much weaker away from the essay's prompt, and removing them also perturbs answers on cases
that aren't preeclampsia.

**Stage 0 (the reproduction gate).** The 4-bit model, the pre-registered stand-in, failed the gate: removing the
hub left P(visual) at 0.123 and the model asked about dark urine (`results/stage0.json`). As deviation D1
prescribes, the study moved to the 8-bit model, which reproduces the essay's published numbers:

| | Essay (bf16) | 4-bit (failed the gate) | 8-bit (passed, used) |
|---|---|---|---|
| Intact P(visual) | 0.903 | 0.860 | 0.912 |
| Hub removed, P(visual) | 0.0005 | 0.123 | 0.00048 |
| Hub removed, continuation | "pruritus (itching)…" | "dark urine…" | "pruritus (itching)**, particularly **without a rash**" |
| Hub activations vs published (Spearman) | | 0.97 | 0.999 |

**Exclusions.** A case is analysed only if the intact model's diagnosis matches its intended label: 25 of 120
cases were excluded (S0: 3, S1: 8, S2: 15, S3: 0). The full list is in `decisions_main.json` under `inclusion`.

**Descriptive analyses** (not pre-registered) are reported in RESULTS.md, separately from the verdict.

## Repository layout

```
README.md
LICENSE                     MIT
CITATION.cff                citation metadata
PREREGISTRATION.md          the pre-registration; deviations D1–D2; post-run note
RESULTS.md                  results, descriptive analyses, reading and caveats
requirements.txt
hash_prereg.sh              hashes the pre-registration, stimuli, code and results so far into results/prereg.sha256
run_all.sh                  pre-registered runner (4-bit model, held in memory)
run_stage.sh                runner for one model streamed from disk (deviation D1)
overnight.sh                deviation D1 sequence: 8-bit, then bf16 only if 8-bit failed the gate
src/
  hubcommon.py              model loading (in memory or streamed), neuron removal, readouts, prompts, labels
  s0_validity.py            Stage 0 gate
  s1_run.py                 debug and main runs
  s2_analysis.py            pre-registered analysis: P1–P4 and the decision
  parity_test.py            parity test for the streaming engine and padded batches (deviation D1)
stimuli/
  stimuli.json              120 cases (S0–S3) and 40 sentences (S4)
  judge/                    generator outputs, judge inputs and outputs, scripted checks, JUDGE_REPORT.md
data/published/
  SOURCE.md                 where the essay's published data comes from (the data files are not included)
results/
  prereg.sha256             hash log, three entries; prereg_check.sha256 is the latest entry, checked by the runners
  stage0.json, run_all.*    4-bit Stage 0, which failed the gate (deviation D1)
  parity/                   parity test summaries (deviation D1)
  overnight.*               log of the D1 sequence
  8bit/                     the reported study: stage0.json, run_debug.jsonl, decisions_debug*.json,
                            run_main.jsonl, decisions_main.json, run.log
```

- `run_main.jsonl` holds one line per case with both frames under every condition (intact, hub removed and
  K = 20 draws of each control), plus one line for the S4 sentences.
- The debug run used a stand-in group (the 2 "microangiopathic syndromes" neurons) on 10 cases per set, before the
  single main run. `decisions_debug_preD2.json` is its analysis before the labelling fix in deviation D2. Both
  are kept for transparency and are not part of the reported results.
- Logs use the local time of the machine that ran the study (UTC+1); hash entries use UTC. Some log lines show
  that machine's absolute paths. `run_all.stdout` and `overnight.stdout` are terminal captures of the same runs.
- The raw activation arrays from the parity test (`results/parity/parity_*.npz`, about 7 MB each) are not
  included. Their summaries are in `parity.json` and `parity_notes.json`.
- The scripts that build the blog post are not part of this repository.

## Pre-registration and hashing

The protocol:

1. The pre-registration, the stimuli, every judge record, the code and the essay's data were hashed (SHA-256,
   UTC timestamp) into `results/prereg.sha256` before Stage 0 ran, using `hash_prereg.sh`.
2. Deviations were appended to the pre-registration, with reasons, before the step they affect, and everything
   was hashed again, together with the results known so far.
3. Each runner first checks `results/prereg_check.sha256` (the latest entry) and stops if any file differs.
4. The pipeline was debugged on a stand-in group first. The reported main run of the hub is the first and only
   one.
5. After the analysis, a short "Post-run note" was appended to the pre-registration. It was not hashed, so the
   current `PREREGISTRATION.md` does not match its last recorded hash.

### Verification (checked against the files in this repository)

| `prereg.sha256` entry | Recorded (UTC) | What was hashed | Check | Result |
|---|---|---|---|---|
| "before Stage 0" | 2026-10-19 18:33:27 | `PREREGISTRATION.md`, first version (`b2e76e3a…`); stimuli, judge records, code, `run_all.sh`, the essay's data | Pre-registration: the current file cut after its "## Deviations" heading, with "(None yet.)" as the section body. Stimuli and judge records: current files | Match |
| "D1: before any 8-bit or bf16 run" | 2026-10-19 22:58:54 | `PREREGISTRATION.md` with D1 (`19a27180…`); the D1 code changes and runners; the 4-bit Stage 0 and parity results | Pre-registration: the current file cut just before "*Outcome of D1 so far.*" | Match |
| "D2: before any main-run result is inspected" | 2026-10-20 00:41:59 | `PREREGISTRATION.md` with D2 (`e39df37b…`); the D2 code fix; the 8-bit Stage 0 and debug results | Pre-registration: the current file with the post-run note removed. Every other file: current file | Match |

In summary:
- Every recorded version of the pre-registration is reproduced exactly from the current file.
- Every other file in the last entry matches the published file, except the two essay data files, which are
  not included (see Reproducing).
- Code changed between entries only as described in deviations D1 (streaming, padded batches, batch packing)
  and D2 (the diagnosis-label order).
- The D2 entry was recorded after the main run had started (00:40 UTC) but before any of its results were
  inspected, as D2 states. The fix only changes how stored answers are labelled, and `s2_analysis.py` recomputes
  every label from the stored text.

Two caveats apply. The hash logs were written on the same machine by the same analyst who ran the tests. They
show that the files are unchanged since the logged points (apart from the noted append), but they cannot prove
when those points were. And git does not preserve file modification times.

To check the hashes yourself, run this from the repository root (`sha256sum` on Linux):

```bash
shasum -a 256 -c results/prereg_check.sha256   # PREREGISTRATION.md differs only by its post-run note
python3 - <<'EOF'
import hashlib
b = open("PREREGISTRATION.md", "rb").read()
pre_run = b[:b.index(b"\n**Post-run note")].rstrip(b"\n") + b"\n"   # drop the appended post-run note
d1 = b[:b.index(b"\n*Outcome of D1 so far.*")].rstrip(b"\n") + b"\n"
first = b[:b.index(b"## Deviations\n") + len(b"## Deviations\n")] + b"\n(None yet.)\n"
for label, v in [("before Stage 0", first), ("D1", d1), ("D2, pre-run", pre_run)]:
    print(hashlib.sha256(v).hexdigest(), label)
EOF
grep PREREGISTRATION.md results/prereg.sha256
```

## Reproducing

**Hardware.** The study ran on an Apple M1 Max with 32 GB of memory, using [MLX](https://github.com/ml-explore/mlx),
which needs Apple Silicon. The 8-bit model (about 35 GB) does not fit in memory, so it is streamed from disk one
layer at a time (`HUB_STREAM=1`; see `src/hubcommon.py`).

**Environment.** Python 3.12.11; `requirements.txt` pins the versions used. The shell runners call
`~/venvs/preeclampsia-hub/bin/python`. They are kept exactly as hashed, so either create the virtual
environment at that path or change `PY` in your copy:

```bash
python3.12 -m venv ~/venvs/preeclampsia-hub
~/venvs/preeclampsia-hub/bin/pip install -r requirements.txt
```

**Model weights.** The scripts run with `HF_HUB_OFFLINE=1`, so download the weights first (into
`~/.cache/huggingface`):

```bash
~/venvs/preeclampsia-hub/bin/python -c "from huggingface_hub import snapshot_download; snapshot_download('mlx-community/Qwen3-32B-8bit')"
```

The 4-bit model, `mlx-community/Qwen3-32B-4bit` (about 18 GB), is only needed to repeat the failed 4-bit Stage 0
and the parity test.

**The essay's data** (not included). Stage 0 compares against the essay's published numbers, which are read
from two files in `data/published/`. `SOURCE.md` describes them:

| File | Source | SHA-256 used in the study |
|---|---|---|
| `graph-carousel-hover.json` | `https://already-clean.goodfire.com/data/graph-carousel-hover.json` | `367bdaa9fa89002f…` |
| `preeclampsia_knockouts.json` | the `const D = {...}` object embedded in the essay's HTML, saved as JSON | `2e4d9d1874a60c5c…` |

Both were downloaded on 2026-10-19. Their full hashes are in `results/prereg_check.sha256`. If the essay's page
has changed since, the files will not match.

**Running.** `run_stage.sh` runs the whole sequence (hash check, Stage 0, debug run, main run, analysis). Its
hash check now stops at the published `PREREGISTRATION.md` because of the post-run note, so to rerun, call the
stages directly. Use a new results tag: `s1_run.py` appends to `run_<tag>.jsonl` and skips cases that are
already there.

```bash
cd src
export HF_HUB_OFFLINE=1 HUB_STREAM=1 HUB_MODEL=mlx-community/Qwen3-32B-8bit HUB_RESULTS_TAG=rerun
PY=~/venvs/preeclampsia-hub/bin/python
$PY -I s0_validity.py                                        # Stage 0 gate -> results/rerun/stage0.json
$PY -I s1_run.py --tag debug && $PY -I s2_analysis.py --tag debug   # stand-in group (optional)
$PY -I s1_run.py --tag main                                  # the hub, all cases, K from Stage 0
$PY -I s2_analysis.py --tag main                             # P1–P4 and the decision
```

The parity test runs on the 4-bit model, held in memory:

```bash
cd src
export HF_HUB_OFFLINE=1 HUB_MODEL=mlx-community/Qwen3-32B-4bit HUB_RESULTS_TAG=parity
python -I parity_test.py dump resident
HUB_STREAM=1 python -I parity_test.py dump stream
python -I parity_test.py compare
```

`HUB_SMOKE=1` switches to `mlx-community/Qwen3-0.6B-4bit` (same tokenizer) for a quick end-to-end check of the
pipeline. Its numbers say nothing about Qwen3-32B.

On the M1 Max, streaming the 8-bit model, Stage 0 took about 10 minutes, the debug run about 35 minutes and the
main run about 5 hours 45 minutes. Expect small numerical differences on other hardware or library versions: batch
shape and padding alone move P(visual) by up to about 0.04 (deviation D1).

## Citation

```bibtex
@misc{eiras2026preeclampsia,
  author = {Eiras, Francisco},
  title  = {How targeted is the preeclampsia diagnosis hub?},
  year   = {2026},
  url    = {https://fgirbal.github.io/blog/preeclampsia-hub/}
}
```

GitHub's "Cite this repository" button uses [CITATION.cff](CITATION.cff).

## Authorship

The study design and the experiments were orchestrated by a human (Francisco Girbal Eiras). The implementation
(measurements and analysis) was written with Claude Opus 5.5 and checked by a human. The stimuli were written by
Claude Opus and checked by an independent Claude Sonnet judge, as described in PREREGISTRATION.md §5.

## License

The code and written material are released under the [MIT License](LICENSE). The clinical cases are synthetic.
The essay's published data, which this code reads from `data/published/`, is not included and remains under its
owner's terms.
