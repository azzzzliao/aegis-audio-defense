#!/usr/bin/env bash
# Steps 1-4 -- AEGIS for one model and one protocol, one benchmark at a time:
#   1. train the risk gate + late-layer LoRA
#        PROTOCOL=indomain  on the benchmark's train split (+ Benign_train)
#        PROTOCOL=lobo      on the other two benchmarks (+ Benign_train)
#   2. gate scores on XSTest and the evaluation set (prefill only, no generation)
#        indomain: the benchmark's test split; lobo: the whole held-out benchmark
#   3. threshold on the XSTest validation half: added over-refusal <= BUDGET
#   4. defended generation on the evaluation set + XSTest, Llama-Guard judge
# then print the summary table (scripts/summarize.py). Run scripts/run_baseline.sh first.
#
#   MODEL=gemma4_e4b_it PROTOCOL=indomain DEVICE=cuda:0 bash scripts/run_aegis.sh
#
# Env as in run_baseline.sh, plus TRAIN_DEVICE, BENCHES (default: all three),
# BUDGET (0.10), TAU (0.5), ALPHAS, EPOCHS (4), SEED (42), and REUSE_BASELINE (1):
# with greedy decoding a non-fired input is identical to the undefended run, so its
# baseline response is reused instead of regenerated. Resumable.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$REPO/scripts/models.sh"
PROTOCOL="${PROTOCOL:?set PROTOCOL=indomain or PROTOCOL=lobo}"
PY="${PY:-python}"; JUDGE_PY="${JUDGE_PY:-$PY}"
DEVICE="${DEVICE:-cuda:0}"; TRAIN_DEVICE="${TRAIN_DEVICE:-$DEVICE}"
JUDGE_DEVICE="${JUDGE_DEVICE:-$DEVICE}"; [ "$JUDGE_DEVICE" = auto ] && JUDGE_DEVICE=cuda:0
GUARD="${GUARD:-meta-llama/Llama-Guard-3-8B}"
MAN="${MANIFESTS:-$REPO/data/manifests}"
OUT="${RUNS:-$REPO/runs}/$MODEL"; B="$OUT/baseline"
BENCHES="${BENCHES:-audiojailbreak jalm sacred}"
BUDGET="${BUDGET:-0.10}"; TAU="${TAU:-0.5}"; ALPHAS="${ALPHAS:-1.0,1.5,2.0,2.5,3.0}"
EPOCHS="${EPOCHS:-4}"; SEED="${SEED:-42}"; REUSE_BASELINE="${REUSE_BASELINE:-1}"
declare -A FULL=( [audiojailbreak]=audiojailbreak.jsonl [jalm]=jalm.jsonl [sacred]=sacred_msd.jsonl )
case "$PROTOCOL" in
  # The closed loop sums refusal probability over the first tokens of a phrase list; the
  # in-domain runs used the multilingual list so non-English refusals are recognised.
  indomain) export REFUSAL_PHRASE_SET="${REFUSAL_PHRASE_SET:-multi}" ;;
  lobo)     export REFUSAL_PHRASE_SET="${REFUSAL_PHRASE_SET:-en}" ;;
  *) echo "PROTOCOL must be indomain or lobo" >&2; exit 1 ;;
esac

[ -s "$B/xstest.jsonl" ] || { echo "missing $B/xstest.jsonl -- run scripts/run_baseline.sh first" >&2; exit 1; }

for H in $BENCHES; do
  D="$OUT/$PROTOCOL/$H"; mkdir -p "$D"
  DATA="$OUT/train/train_${PROTOCOL}_$H.jsonl"
  if [ "$PROTOCOL" = indomain ]; then EVAL="$MAN/${H}_test.jsonl"; else EVAL="$MAN/${FULL[$H]}"; fi
  echo "==== [$MODEL | $PROTOCOL | $H] $(date '+%F %T')"

  # 1. train (skipped if the adapter exists)
  if [ ! -s "$D/adapter/config.json" ]; then
    "$PY" "$REPO/aegis/train_gate_lora.py" --model "$MODEL" --device "$TRAIN_DEVICE" \
      --data "$DATA" --out-dir "$D/adapter" --prompt-mode "$PROMPT_MODE" \
      --gate-layer "$GATE_LAYER" --late-from "$LATE_FROM" --epochs "$EPOCHS" \
      --gate-type mlp --gate-hidden 256 --gate-bce 1.0 --gate-l1 0.01 --rank 16 --seed "$SEED"
  fi

  # 2. soft gate scores
  "$PY" "$REPO/aegis/run_defense.py" --model "$MODEL" --device "$DEVICE" --adapter "$D/adapter" \
    --prompt-mode "$PROMPT_MODE" --jobs "xstest:$MAN/xstest.jsonl,$H:$EVAL" \
    --out-dir "$D/scores" --score-only --gate-threshold 0 --resume

  # 3. threshold from the XSTest validation half only
  THR=$("$PY" "$REPO/scripts/calibrate_threshold.py" --scores "$D/scores/xstest.jsonl" \
    --baseline "$B/xstest.jsonl" --response-key "$RESP_KEY" --budget "$BUDGET" \
    --out "$D/threshold.json")
  echo "[threshold] $H -> $THR"

  # 4. defended generation + judge
  BL=()
  [ "$REUSE_BASELINE" = 1 ] && BL=(--baseline "$H:$B/$H.jsonl,xstest:$B/xstest.jsonl")
  "$PY" "$REPO/aegis/run_defense.py" --model "$MODEL" --device "$DEVICE" --adapter "$D/adapter" \
    --prompt-mode "$PROMPT_MODE" --jobs "$H:$EVAL,xstest:$MAN/xstest.jsonl" \
    --out-dir "$D/defended" --gate-threshold "$THR" --tau "$TAU" --alphas "$ALPHAS" \
    --resume ${BL[@]+"${BL[@]}"}
  "$JUDGE_PY" "$REPO/aegis/judge.py" --input "$D/defended/$H.jsonl" \
    --out "$D/defended/${H}_judged.jsonl" --response-key "$RESP_KEY" \
    --model "$GUARD" --device "$JUDGE_DEVICE"
done

"$PY" "$REPO/scripts/summarize.py" --run-dir "$OUT" --protocol "$PROTOCOL" --response-key "$RESP_KEY"
