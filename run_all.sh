#!/bin/zsh
# Overnight run: Stage 0 gate -> debug run (stand-in group) -> main run -> analysis.
# Start it with:  caffeinate -is zsh run_all.sh
# (keeps the Mac awake while on power; keep the lid open). Stops at the first failing step.
set -e
setopt pipefail
cd "$(dirname "$0")/src"
PY=~/venvs/preeclampsia-hub/bin/python
export HF_HUB_OFFLINE=1
LOG=../results/run_all.log
mkdir -p ../results
step() { echo "=== $(date '+%F %T') $*" | tee -a "$LOG"; }

step "hash check"
(cd .. && shasum -a 256 -c results/prereg_check.sha256) 2>&1 | tee -a "$LOG"
step "stage 0"
if ! $PY -I s0_validity.py 2>&1 | tee -a "$LOG"; then step "stage 0 gate FAILED: stopping"; exit 1; fi
step "debug run (stand-in group)"
$PY -I s1_run.py --tag debug 2>&1 | tee -a "$LOG"
$PY -I s2_analysis.py --tag debug 2>&1 | tee -a "$LOG"
step "main run"
$PY -I s1_run.py --tag main 2>&1 | tee -a "$LOG"
step "analysis"
$PY -I s2_analysis.py --tag main 2>&1 | tee -a "$LOG"
step "done"
