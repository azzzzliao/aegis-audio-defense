# Per-model AEGIS settings, sourced by scripts/run_*.sh.
#   GATE_LAYER   decoder layer whose last-prompt-token hidden state feeds the risk gate
#   LATE_FROM    first intervention layer; LoRA is attached to [LATE_FROM, n_layers)
#   RESP_KEY     field the adapter writes its response to
#   PROMPT_MODE  "safety": audio + the adapter's fixed text prompt; "native": audio only
# Layers are probe-selected per model, not set proportionally.
PROMPT_MODE=safety
case "${MODEL:?set MODEL to one of: gemma4_e4b_it phi4_mm vita_15 qwen2_audio ultravox_v05 voxtral_small}" in
  gemma4_e4b_it) GATE_LAYER=21; LATE_FROM=25; RESP_KEY=gemma4_response ;;       # 42 layers
  phi4_mm)       GATE_LAYER=15; LATE_FROM=19; RESP_KEY=phi4_response ;;         # 32 layers
  vita_15)       GATE_LAYER=15; LATE_FROM=19; RESP_KEY=vita_response ;;         # 28 layers
  qwen2_audio)   GATE_LAYER=15; LATE_FROM=19; RESP_KEY=qwen2_audio_response
                 PROMPT_MODE=native ;;                                          # 32 layers
  ultravox_v05)  GATE_LAYER=15; LATE_FROM=19; RESP_KEY=ultravox_response ;;     # 32 layers
  voxtral_small) GATE_LAYER=24; LATE_FROM=28; RESP_KEY=voxtral_response ;;      # 40 layers
  *) echo "unknown MODEL=$MODEL" >&2; exit 1 ;;
esac
