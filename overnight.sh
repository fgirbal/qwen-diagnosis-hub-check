#!/bin/zsh
# Deviation D1 sequence. Start with:  caffeinate -is zsh overnight.sh
# 1. wait for the 8-bit download, verify it, run the 8-bit pipeline;
# 2. if 8-bit fails the Stage 0 gate: delete the 8-bit weights (and the 4-bit weights if disk is still short),
#    download bf16, run the bf16 pipeline.
cd "$(dirname "$0")"
LOG=results/overnight.log
PY=~/venvs/preeclampsia-hub/bin/python
HUB=~/.cache/huggingface/hub
log() { echo "=== $(date '+%F %T') $*" | tee -a "$LOG"; }
dl() { HF_HUB_DISABLE_XET=1 $PY -c "from huggingface_hub import snapshot_download as s; print('downloaded', s('$1', max_workers=4))" >> "$LOG" 2>&1; }
free_gib() { df -g ~ | tail -1 | awk '{print $4}'; }

log "waiting for the 8-bit download (free disk $(free_gib) GiB)"
while pgrep -f "Qwen3-32B-8bit', max_workers" > /dev/null; do sleep 60; done
log "8-bit download process finished; verifying / resuming"
dl mlx-community/Qwen3-32B-8bit || { log "8-bit download failed"; exit 1; }
zsh run_stage.sh mlx-community/Qwen3-32B-8bit 8bit; rc=$?
log "8-bit pipeline exit code $rc"
if [ $rc -ne 2 ]; then log "finished (no bf16 needed)"; exit $rc; fi

log "8-bit failed the Stage 0 gate: deleting the 8-bit weights and trying bf16"
rm -rf "$HUB/models--mlx-community--Qwen3-32B-8bit"
log "free disk $(free_gib) GiB"
if [ $(free_gib) -lt 66 ]; then
  log "less than 66 GiB free: deleting the 4-bit weights as well (re-downloadable)"
  rm -rf "$HUB/models--mlx-community--Qwen3-32B-4bit"
  log "free disk $(free_gib) GiB"
fi
if [ $(free_gib) -lt 66 ]; then log "still not enough disk for bf16 (61 GiB + margin): stopping"; exit 1; fi
dl mlx-community/Qwen3-32B-bf16 || { log "bf16 download failed"; exit 1; }
zsh run_stage.sh mlx-community/Qwen3-32B-bf16 bf16; rc=$?
log "bf16 pipeline exit code $rc"
log "finished"
exit $rc
