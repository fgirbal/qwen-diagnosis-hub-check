#!/bin/zsh
# Hash the pre-registration, stimuli, judge records, code, runners and results known so far, before a run.
#   zsh hash_prereg.sh "<label for the log entry>"
# Appends a timestamped entry to results/prereg.sha256 and rewrites results/prereg_check.sha256 (checked by the runners).
set -e
cd "$(dirname "$0")"
mkdir -p results
LABEL=${1:-"before Stage 0"}
FILES=(PREREGISTRATION.md stimuli/stimuli.json stimuli/judge/*(.) src/*.py run_all.sh run_stage.sh(N) overnight.sh(N)
       data/published/*(.) results/stage0.json(N) results/parity/*.json(N)
       results/*/stage0.json(N) results/*/run_debug.jsonl(N) results/*/decisions_debug*.json(N))
shasum -a 256 $FILES > results/prereg_check.sha256
{ echo "--- $(date -u '+%Y-%m-%dT%H:%M:%SZ') $LABEL:"; cat results/prereg_check.sha256; } >> results/prereg.sha256
echo "hashed ${#FILES} files"
