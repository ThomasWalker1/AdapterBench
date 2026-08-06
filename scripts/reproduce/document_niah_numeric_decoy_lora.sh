#!/bin/bash
# Reproduce the five-seed final LoRA D2A numeric-decoy result.
set -euo pipefail
cd "$(dirname "$0")/../.."
exec bash scripts/reproduce/document_niah_numeric_decoy.sh lora "${1:-cuda:0}"
