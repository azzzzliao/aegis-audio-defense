#!/usr/bin/env bash
# Ablations (paper Table 2, LOBO): the LoRA is applied unconditionally -- no gate, no
# closed loop -- with the same LOBO training files, targets and judge as AEGIS.
#   VARIANT=full       LoRA on all decoder layers    (--no-gate --late-from 0)
#   VARIANT=always_on  LoRA on the AEGIS late layers (--no-gate --late-from $LATE_FROM)
#
#   MODEL=qwen2_audio VARIANT=full bash scripts/run_ablation.sh
#
# Env as in run_aegis.sh; EPOCHS defaults to 3 (the setting used for the ablations);
# BATCH_SIZE > 1 batches generation (valid here because nothing is gated).
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$REPO/scripts/models.sh"
VARIANT="${VARIANT:?set VARIANT=full or VARIANT=always_on}"
case "$VARIANT" in
  full) FROM=0 ;;
  always_on) FROM="$LATE_FROM" ;;
  *) echo "VARIANT must be full or always_on" >&2; exit 1 ;;
esac
PY="${PY:-python}"; JUDGE_PY="${JUDGE_PY:-$PY}"
DEVICE="${DEVICE:-cuda:0}"; TRAIN_DEVICE="${TRAIN_DEVICE:-$DEVICE}"
JUDGE_DEVICE="${JUDGE_DEVICE:-$DEVICE}"; [ "$JUDGE_DEVICE" = auto ] && JUDGE_DEVICE=cuda:0
GUARD="${GUARD:-meta-llama/Llama-Guard-3-8B}"
MAN="${MANIFESTS:-$REPO/data/manifests}"
OUT="${RUNS:-$REPO/runs}/$MODEL"; V="$OUT/ablation_$VARIANT"
BENCHES="${BENCHES:-audiojailbreak jalm sacred}"
EPOCHS="${EPOCHS:-3}"; SEED="${SEED:-42}"; BATCH_SIZE="${BATCH_SIZE:-1}"
declare -A FULL=( [audiojailbreak]=audiojailbreak.jsonl [jalm]=jalm.jsonl [sacred]=sacred_msd.jsonl )

for H in $BENCHES; do
  D="$V/$H"; mkdir -p "$D"
  echo "==== [$MODEL | $VARIANT | held-out $H] $(date '+%F %T')"
  if [ ! -s "$D/adapter/config.json" ]; then
    "$PY" "$REPO/aegis/train_gate_lora.py" --model "$MODEL" --device "$TRAIN_DEVICE" \
      --data "$OUT/train/train_lobo_$H.jsonl" --out-dir "$D/adapter" --prompt-mode "$PROMPT_MODE" \
      --gate-layer "$GATE_LAYER" --late-from "$FROM" --epochs "$EPOCHS" --rank 16 \
      --seed "$SEED" --no-gate
  fi
  "$PY" "$REPO/aegis/run_defense.py" --model "$MODEL" --device "$DEVICE" --adapter "$D/adapter" \
    --prompt-mode "$PROMPT_MODE" --jobs "$H:$MAN/${FULL[$H]},xstest:$MAN/xstest.jsonl" \
    --out-dir "$D/defended" --unconditional --batch-size "$BATCH_SIZE" --resume
  "$JUDGE_PY" "$REPO/aegis/judge.py" --input "$D/defended/$H.jsonl" \
    --out "$D/defended/${H}_judged.jsonl" --response-key "$RESP_KEY" \
    --model "$GUARD" --device "$JUDGE_DEVICE"
done

"$PY" "$REPO/scripts/summarize.py" --run-dir "$OUT" --variant-dir "$V" --response-key "$RESP_KEY"
