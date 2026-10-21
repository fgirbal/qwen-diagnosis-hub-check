#!/bin/zsh
# Full pipeline for one model, streamed from disk (deviation D1):
#   zsh run_stage.sh <hf repo id> <results tag>      e.g. zsh run_stage.sh mlx-community/Qwen3-32B-8bit 8bit
# hash check -> Stage 0 gate -> debug run (stand-in) -> main run -> analysis.
# Exit codes: 0 done, 2 Stage 0 gate failed, 3 hash check failed, other = error.
set -e
setopt pipefail
MODEL=$1; TAG=$2
cd "$(dirname "$0")/src"
PY=~/venvs/preeclampsia-hub/bin/python
export HF_HUB_OFFLINE=1 HUB_STREAM=1 HUB_MODEL=$MODEL HUB_RESULTS_TAG=$TAG
mkdir -p ../results/$TAG
LOG=../results/$TAG/run.log
step() { echo "=== $(date '+%F %T') $*" | tee -a "$LOG"; }

step "hash check"
if ! (cd .. && shasum -a 256 -c results/prereg_check.sha256 > /dev/null); then step "hash check FAILED"; exit 3; fi
step "stage 0 ($MODEL, streamed)"
if ! $PY -I s0_validity.py 2>&1 | tee -a "$LOG"; then step "stage 0 gate FAILED"; exit 2; fi
step "debug run (stand-in group)"
$PY -I s1_run.py --tag debug 2>&1 | tee -a "$LOG"
$PY -I s2_analysis.py --tag debug 2>&1 | tee -a "$LOG"
step "main run"
$PY -I s1_run.py --tag main 2>&1 | tee -a "$LOG"
step "analysis"
$PY -I s2_analysis.py --tag main 2>&1 | tee -a "$LOG"
step "done"
