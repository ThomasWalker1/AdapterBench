#!/bin/bash
# Reproduce one T2A LoKr confirmation seed. Wrapper around t2a_reproduce_seed.sh.
exec "$(dirname "$0")/t2a_reproduce_seed.sh" lokr "$@"
