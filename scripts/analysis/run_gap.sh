#!/usr/bin/env bash
# Risk-to-refusal gap, one model end to end: manifests -> hidden states -> curves -> figure.
# Needs runs/<model>/baseline/<bench>_judged.jsonl (scripts/run_baseline.sh).
#
#   MODEL=gemma4_e4b_it PY=python DEVICE=cuda:0 bash scripts/analysis/run_gap.sh
#
# Env: MODEL (required), PY, DEVICE, RUNS, BENCHES, CAP, NEGATIVES (attack_safe|benign).
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MODEL="${MODEL:?set MODEL}"
PY="${PY:-python}"; DEVICE="${DEVICE:-cuda:0}"
RUNS="${RUNS:-$REPO/runs}"; CAP="${CAP:-400}"
NEGATIVES="${NEGATIVES:-attack_safe}"
BENCHES="${BENCHES:-audiojailbreak jalm_adiv jalm_ssj sacred}"
GAP="$RUNS/$MODEL/gap"

"$PY" "$REPO/scripts/analysis/gap_build_manifests.py" --model "$MODEL" --runs "$RUNS" --cap "$CAP"

mkdir -p "$GAP/hidden"
for B in $BENCHES benign; do
  [ -s "$GAP/hidden/$B.npz" ] && { echo "[skip] $B.npz exists"; continue; }
  echo "== extract $MODEL/$B"
  "$PY" "$REPO/aegis/extract_hidden.py" --model "$MODEL" --device "$DEVICE" \
    --responses "$GAP/manifests/$B.jsonl" \
    --out-npz "$GAP/hidden/$B.npz" --out-meta "$GAP/hidden/${B}_meta.jsonl"
done

"$PY" "$REPO/scripts/analysis/gap_compute_curves.py" --model "$MODEL" --runs "$RUNS" \
  --device "$DEVICE" --negatives "$NEGATIVES" --benches $BENCHES
"$PY" "$REPO/scripts/analysis/gap_plot_models.py" --models "$MODEL"
