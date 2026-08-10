# Shared T2A confirmation operating points (AUTORESEARCH.md).
# Source this file, then call: t2a_load_codec_config <codec>
#
# Each codec trains hyper and static* at independently selected free hyperparameters.
# Confirmation seeds are disjoint from scout and selection seeds.

t2a_load_codec_config() {
  local codec="${1:?codec required (lora|ia3|lokr|fourierft|steering)}"
  CODEC="$codec"
  case "$codec" in
    lora)
      HYPER_SCALE=1.414214; HYPER_LR=5e-5
      STATIC_SCALE=22.627417; STATIC_LR=5e-5
      STEPS=8000
      CONFIRMATION_SEEDS=(1741 1742 1743)
      ;;
    ia3)
      HYPER_SCALE=16; HYPER_LR=5e-5
      STATIC_SCALE=16; STATIC_LR=8e-4
      STEPS=8000
      CONFIRMATION_SEEDS=(1751 1752 1753)
      ;;
    lokr)
      HYPER_SCALE=0.25; HYPER_LR=2e-4
      STATIC_SCALE=0.25; STATIC_LR=2e-4
      STEPS=6000
      CONFIRMATION_SEEDS=(2741 2742 2743)
      ;;
    fourierft)
      HYPER_SCALE=0.25; HYPER_LR=1e-4
      STATIC_SCALE=4; STATIC_LR=1e-4
      STEPS=6000
      CONFIRMATION_SEEDS=(5041 5042 5043)
      ;;
    steering)
      HYPER_SCALE=1; HYPER_LR=1e-4
      STATIC_SCALE=64; STATIC_LR=2e-4
      STEPS=8000
      CONFIRMATION_SEEDS=(4741 4742 4743)
      ;;
    *)
      echo "unsupported codec: $codec" >&2
      return 2
      ;;
  esac
  HYPER_TAG="${codec}_scale${HYPER_SCALE}_lr${HYPER_LR}_hyper"
  STATIC_TAG="${codec}_scale${STATIC_SCALE}_lr${STATIC_LR}_static"
  BASE_ROOT="results/autoresearch/t2a/${codec}/confirmation_v2"
}

t2a_assert_confirmation_seed() {
  local seed="$1"
  local found=0
  for s in "${CONFIRMATION_SEEDS[@]}"; do
    [[ "$s" == "$seed" ]] && found=1 && break
  done
  if [[ "$found" != "1" ]]; then
    echo "seed $seed is not a confirmation seed for $CODEC (expected: ${CONFIRMATION_SEEDS[*]})" >&2
    return 2
  fi
}
