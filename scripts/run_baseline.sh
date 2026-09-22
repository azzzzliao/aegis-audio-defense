#!/usr/bin/env bash
# Step 0 -- undefended baseline for one model:
#   * responses on the three attack benchmarks, XSTest and Benign_train (aegis/run_model.py)
#   * Llama-Guard-3-8B judgments of the attack responses (aegis/judge.py)
#   * training files for both protocols from the model's own benign answers
#
#   MODEL=gemma4_e4b_it DEVICE=cuda:0 bash scripts/run_baseline.sh
#
# Env: MODEL (required); DEVICE (default cuda:0; "auto" shards across visible GPUs);
# PY (python of the model's env); JUDGE_PY / JUDGE_DEVICE (an env that can load
# Llama-Guard-3-8B, if the model's env cannot); GUARD (judge id or path);
# MANIFESTS (default data/manifests); RUNS (default runs). Resumable: rerun to continue.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$REPO/scripts/models.sh"
PY="${PY:-python}"; JUDGE_PY="${JUDGE_PY:-$PY}"
DEVICE="${DEVICE:-cuda:0}"; JUDGE_DEVICE="${JUDGE_DEVICE:-$DEVICE}"
[ "$JUDGE_DEVICE" = auto ] && JUDGE_DEVICE=cuda:0
GUARD="${GUARD:-meta-llama/Llama-Guard-3-8B}"
MAN="${MANIFESTS:-$REPO/data/manifests}"
OUT="${RUNS:-$REPO/runs}/$MODEL"; B="$OUT/baseline"
mkdir -p "$B"

echo "[baseline] $MODEL -> $B"
"$PY" "$REPO/aegis/run_model.py" --model "$MODEL" --device "$DEVICE" --out-dir "$B" --resume \
  --prompt-mode "$PROMPT_MODE" \
  --jobs "audiojailbreak:$MAN/audiojailbreak.jsonl,jalm:$MAN/jalm.jsonl,sacred:$MAN/sacred_msd.jsonl,xstest:$MAN/xstest.jsonl,benign_train:$MAN/benign_train.jsonl"

for H in audiojailbreak jalm sacred; do
  "$JUDGE_PY" "$REPO/aegis/judge.py" --input "$B/$H.jsonl" --out "$B/${H}_judged.jsonl" \
    --response-key "$RESP_KEY" --model "$GUARD" --device "$JUDGE_DEVICE"
done

"$PY" "$REPO/scripts/build_trainsets.py" --manifests "$MAN" --splits "$REPO/data/splits" \
  --benign-responses "$B/benign_train.jsonl" --response-key "$RESP_KEY" --out-dir "$OUT/train"
echo "[baseline] done. Next: MODEL=$MODEL PROTOCOL=indomain bash scripts/run_aegis.sh"
