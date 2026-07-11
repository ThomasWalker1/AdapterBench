#!/usr/bin/env bash
# Length-generalization grid, low-rank family, shape comparison (PROJECT_PLAN step 1).
# Train short (128,256), eval a sweep (256..8192), 3 seeds, scale-matched: 45.25 applied
# to every LoRA-family weight codec via --scale-weight-codecs, so freeze_a_lora/lokr are
# compared to LoRA at the SAME effective scale => a pure shape comparison, not scale-
# confounded. LoRA's 3 seeds are launched separately (already running); this grid is the
# freeze_a_lora + lokr half. Generous 8000-step budget because the retrieval phase
# transition is init/seed-sensitive and lands late on some seeds (gotcha #18) - LoRA s778
# transitioned ~2500 while s777/s779 were still climbing at 3500. Restart-safe (re-run to
# resume from the last checkpoint).
set -u
cd /home/tw78/AdapterBench

COMMON=(--needle-style generic
        --context-lengths 128,256
        --eval-context-lengths 256,512,1024,2048,4096,8192
        --num-train-documents 512 --steps 8000 --eval-every 500
        --learning-rate 4e-5 --n-latents 208 --num-blocks 8
        --eval-limit 32 --batch-size 8 --scale-weight-codecs)

launch() {  # adapter seed gpu
  local adapter=$1 seed=$2 gpu=$3
  local out=results/d2p_lengthgen_${adapter}_s${seed}
  echo "launch ${adapter} s${seed} -> cuda:${gpu} ($out)"
  nohup .venv/bin/adapterbench d2p-niah \
    --adapters "$adapter" --seed "$seed" --device "cuda:${gpu}" \
    "${COMMON[@]}" --output "$out" \
    > "${out}.log" 2>&1 &
}

# LoRA 777/778/779 already running on cuda:1/5/6 (launched at 6000 steps).
# freeze_a_lora + lokr pairs share GPUs 0/3/4 (2 small runs per 80GB card).
launch freeze_a_lora  777 0
launch freeze_a_lora  778 3
launch freeze_a_lora  779 4
launch lokr           777 0
launch lokr           778 3
launch lokr           779 4
sleep 3  # let the last nohup detach before the launcher returns
echo "grid launched"
