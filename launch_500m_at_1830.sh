#!/bin/bash
# Waits until 18:30 today (UTC, matches server clock), then launches the
# real 500M pretraining run across all 8 GPUs. Meant to be run inside a
# `screen` session so it survives independently of any interactive shell.
#
# EPOCHS=1 is passed EXPLICITLY here -- train_500m.py's underlying default
# is "2" (inherited from train.py's original design), which would silently
# double both wall-clock time and token exposure if omitted. This was the
# single most important thing caught during verification before scheduling.
set -euo pipefail

cd /raid/llm_sec/legal-slm-125M

TARGET_EPOCH=$(date -d "18:30 today" +%s)
NOW=$(date +%s)
WAIT=$(( TARGET_EPOCH - NOW ))

if [ "$WAIT" -gt 0 ]; then
    echo "$(date): waiting ${WAIT}s until 18:30 to start training..."
    sleep "$WAIT"
else
    echo "$(date): 18:30 already passed today, starting immediately."
fi

echo "$(date): starting 500M pretraining run."

export SLM_DATA_ROOT_500M=/raid/llm_sec/legal-slm-125M/data_500m
export SLM_DATA_ROOT=/raid/llm_sec/legal-slm-125M/data
export EPOCHS=1
export COMPILE=1

LOG_PATH="$SLM_DATA_ROOT_500M/checkpoints/train_$(date +%Y%m%d_%H%M).log"
mkdir -p "$SLM_DATA_ROOT_500M/checkpoints"

.venv/bin/torchrun --standalone --nproc_per_node=8 train_500m.py \
    2>&1 | tee -a "$LOG_PATH"

echo "$(date): training run finished (see log above for final val_loss/ppl)."
