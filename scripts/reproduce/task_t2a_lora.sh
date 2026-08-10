#!/bin/bash
# Reproduce one T2A LoRA confirmation seed. Wrapper around t2a_reproduce_seed.sh.
# Usage: scripts/reproduce/task_t2a_lora.sh [SEED] [GPUS_HYPER] [GPUS_STATIC]
exec "$(dirname "$0")/t2a_reproduce_seed.sh" lora "$@"
