#!/bin/bash
# Reproduce the selected T2L LoRA configuration for one seed.
#
# Trains the strip-def hypernetwork AND the same-shape static reference (multi-task LoRA) on the
# 479-task decontaminated SNI split (gemma-2-2b, plain SFT cross-entropy, definition stripped from the
# input so the description is the only route to the task), then scores matched - static on the 21
# held-out SNI validation tasks: teacher-forced CE (primary) + greedy-generation accuracy (corroborating).
#
# The selected free HPs are scale=22.627417, lr=1e-4, warmup=0.1, and 8,000 steps.
# Usage: scripts/reproduce/task_t2l_lora.sh [SEED] [GPUS_HYPER] [GPUS_STATIC]
#   e.g. scripts/reproduce/task_t2l_lora.sh 1801 0,1,2,3 4,5,6,7
set -euo pipefail
cd "$(dirname "$0")/../.."
SEED="${1:-1801}"; GH="${2:-0,1,2,3}"; GS="${3:-4,5,6,7}"
ROOT="${ROOT:-results/repro/t2l_lora_scale22.627417_lr1e-4}"
STEPS=8000 LR=1e-4 LIMIT=40 PGB=16 CE_LIMIT=64 ACC_LIMIT=48 \
  bash scripts/t2l_codec_trial.sh lora 22.627417 "$SEED" "$GH" "$GS" "$ROOT"
