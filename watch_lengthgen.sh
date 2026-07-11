#!/usr/bin/env bash
# Stateful oversight of the 9 length-gen runs. Stays quiet until a run does something
# actionable, then emits ONE line: a phase-transition (niah_256 accuracy crosses 0.5 the
# first time), completion (results.jsonl written), or a crash (process gone before it
# finished, or a Traceback in the log). Exits once all 9 logs have terminated (finished
# or crashed). Poll every 4 min - transitions/finishes are rare, so volume stays low.
set -u
cd /home/tw78/AdapterBench
declare -A state   # per-run: pending | transitioned | done | crashed

runs="lora_s777 lora_s778 lora_s779 freeze_a_lora_s777 freeze_a_lora_s778 freeze_a_lora_s779 lokr_s777 lokr_s778 lokr_s779"
for r in $runs; do state[$r]=pending; done

while true; do
  live=0
  for r in $runs; do
    log="results/d2p_lengthgen_${r}.log"
    [ -f "$log" ] || { live=1; continue; }
    st=${state[$r]}
    [ "$st" = done ] || [ "$st" = crashed ] && continue

    # completion
    if grep -q "wrote results/" "$log"; then
      last=$(grep -oE "'niah_256': \{'accuracy': [0-9.]+" "$log" | tail -1 | grep -oE "[0-9.]+$")
      echo "DONE  ${r}  (final niah_256 acc=${last:-?})"; state[$r]=done; continue
    fi
    # crash: a python traceback in the log
    if grep -qiE "Traceback|CUDA out of memory|RuntimeError|Killed" "$log"; then
      echo "CRASH ${r}  -> $(grep -iE 'Error|Killed|Traceback' "$log" | tail -1 | cut -c1-100)"
      state[$r]=crashed; continue
    fi
    live=1
    # first phase transition: niah_256 accuracy >= 0.5
    if [ "$st" = pending ]; then
      hit=$(grep -oE "'niah_256': \{'accuracy': [0-9.]+" "$log" | grep -oE "[0-9.]+$" \
            | awk '$1>=0.5{print; exit}')
      if [ -n "$hit" ]; then
        step=$(grep -E "\[step " "$log" | tail -1 | grep -oE "step [0-9]+" | grep -oE "[0-9]+")
        echo "TRANSITION ${r}  niah_256 acc>=0.5 around step ${step}"
        state[$r]=transitioned
      fi
    fi
  done
  [ "$live" -eq 0 ] && { echo "ALL RUNS TERMINATED"; break; }
  sleep 240
done
