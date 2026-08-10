#!/bin/bash
# Full three-seed T2A LoRA confirmation reproduction. Wrapper around t2a_reproduce_all.sh.
# Usage: scripts/reproduce/task_t2a_lora_all.sh [GPUS_HYPER] [GPUS_STATIC]
exec "$(dirname "$0")/t2a_reproduce_all.sh" lora "$@"
