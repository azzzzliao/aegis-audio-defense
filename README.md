# AEGIS: Audio Endogenous Guarding via Internal Signals Against Large Audio-Language Model Jailbreaks

[![arXiv](https://img.shields.io/badge/arXiv-2609.29287-b31b1b.svg)](https://arxiv.org/abs/2609.29287)
[![License](https://img.shields.io/badge/License-MIT-3da639.svg)](LICENSE)

Yu-Ling Liao\*, Tzu-Chin Chiu\*, Zong-You Chen\*, Chi-Lei Tsai\*, Shao-Yuan Lo — National
Taiwan University (\*equal contribution) · submitted to ICASSP 2027

AEGIS keeps the target audio-language model frozen and adds a mid-layer risk gate that
conditionally drives late-layer safety adapters, with closed-loop scaling at inference.

## Contents

| Page | What's in it |
|---|---|
| [Method](docs/method.md) | The risk-to-refusal gap, the gate and the adapters, the training loss, the closed loop. |
| [Results](docs/results.md) | The paper's tables: six models, three benchmarks, the ablation, and the defense comparison. |
| [Analyses beyond the paper](docs/analysis/README.md) | The risk-to-refusal gap in the other five models, gate behaviour, representations, the full ablation table. |
| [Data](data/README.md) | Where each dataset comes from and how the manifests are built. |
| [Adding a model](docs/ADDING_A_MODEL.md) | Porting AEGIS to another LALM by writing one adapter. |

On this page: [Repository layout](#repository-layout) · [Setup](#setup) ·
[Data](#data) · [Running AEGIS](#running-aegis) · [Protocol](#protocol) ·
[Ablations](#ablations) · [Tests](#tests) · [Citation](#citation)

## Repository layout

```
aegis/                        defense runtime; every script runs from any directory
  model_registry.py           ModelAdapter interface, registry, Qwen2-Audio and Phi-4 adapters
  adapters_gemma_voxtral.py   Gemma 4 E4B-it and Voxtral-Small-24B adapters
  adapters_ultravox_vita.py   Ultravox v0.5 and VITA-1.5 adapters
  train_gate_lora.py          joint training of the risk gate and the safety adapters
  run_defense.py              gated inference with closed-loop scaling (--score-only: gate scores)
  run_model.py                undefended generation
  judge.py                    Llama-Guard-3-8B safety judge
  refusal.py                  refusal regex for the over-refusal metric
  extract_hidden.py, analysis/  hidden-state probes for layer selection
scripts/
  prepare_data.py             downloaded datasets -> manifests (+ in-domain test splits)
  run_baseline.sh             step 0: undefended responses, judgments, training files
  run_aegis.sh                steps 1-4 for one protocol (PROTOCOL=indomain or lobo)
  run_ablation.sh             the Full and Always-on ablations
  models.sh                   per-model gate layer, intervention layers, prompt mode
  build_trainsets.py, calibrate_threshold.py, summarize.py
data/splits/                  in-domain train/test ids used in the paper
docs/                         method, results, analyses, porting guide
tests/                        CPU tests of the evaluation protocol
```

## Setup

```bash
pip install -r requirements.txt
```

Each model needs a Transformers version that can load it. The upstreams disagree, so we
used one environment per model family:

| `MODEL` | Base model | Hugging Face id (override variable) | Gate layer | Intervention layers | Transformers used |
|---|---|---|---:|---|---|
| `gemma4_e4b_it` | Gemma 4 E4B IT | `google/gemma-4-E4B-it` (`AEGIS_GEMMA4_PATH`) | 21 | 25–41 | 5.7.0.dev0 |
| `phi4_mm` | Phi-4-multimodal | `microsoft/Phi-4-multimodal-instruct` (`AEGIS_PHI4_MM_PATH`) | 15 | 19–31 | 5.7.0.dev0 |
| `vita_15` | VITA-1.5 | `VITA-MLLM/VITA-1.5` (`AEGIS_VITA_PATH`) | 15 | 19–27 | 4.43.4 |
| `qwen2_audio` | Qwen2-Audio-7B-Instruct | `Qwen/Qwen2-Audio-7B-Instruct` (`AEGIS_QWEN2_AUDIO_PATH`) | 15 | 19–31 | ≥ 4.45 |
| `ultravox_v05` | Ultravox v0.5 (Llama-3.1-8B) | `fixie-ai/ultravox-v0_5-llama-3_1-8b` (`AEGIS_ULTRAVOX_PATH`) | 15 | 19–31 | 4.48.1 |
| `voxtral_small` | Voxtral Small 24B | `mistralai/Voxtral-Small-24B-2507` (`AEGIS_VOXTRAL_PATH`) | 24 | 28–39 | 4.57.6 |

- Models load from the Hub by default. To run offline, point the override variable at a
  local snapshot.
- Llama-Guard-3-8B, Gemma 4, and the Llama-3.1 backbone that Ultravox downloads are gated
  on the Hub. Accept their licenses and run `huggingface-cli login`, or set `GUARD` and
  `AEGIS_LLAMA31_PATH` to local copies.
- VITA-1.5 needs its GitHub code: clone https://github.com/VITA-MLLM/VITA and set
  `AEGIS_VITA_REPO` to the checkout (default: `external/VITA`).
- Hardware: one 48 GB GPU is enough for most models. Gemma 4 training needs a single card
  with at least ~40 GB, because the 262k-vocabulary loss does not shard well. Voxtral
  Small needs two 48 GB cards (`DEVICE=auto` shards the model; the gate and adapters
  follow the layers they hook). Qwen2-Audio and Phi-4 also benefit from `DEVICE=auto` on
  the long AJail clips.

## Data

Download the five datasets as described in [`data/README.md`](data/README.md), then build
the manifests:

```bash
python scripts/prepare_data.py --data-root data --out-dir data/manifests
```

The script checks the row counts used in the paper: AJail 1,490, JALM 946, SACRED 1,364,
XSTest 250, Benign_train 215. It also writes the in-domain test splits (718 / 499 / 683
rows) from the id lists in `data/splits/`.

## Running AEGIS

For each model (`PY` is the Python of that model's environment; `JUDGE_PY` can point to a
different environment for Llama Guard):

```bash
MODEL=gemma4_e4b_it PY=python DEVICE=cuda:0 bash scripts/run_baseline.sh                  # step 0
MODEL=gemma4_e4b_it PROTOCOL=indomain PY=python DEVICE=cuda:0 bash scripts/run_aegis.sh   # in-domain
MODEL=gemma4_e4b_it PROTOCOL=lobo     PY=python DEVICE=cuda:0 bash scripts/run_aegis.sh   # LOBO
```

Every stage is resumable; rerun the same command after an interruption. Outputs go to
`runs/<model>/`:

```
baseline/                        undefended responses; <bench>_judged.jsonl = Llama Guard labels
train/train_{indomain,lobo}_<bench>.jsonl
<protocol>/<bench>/adapter/      router_head.pt, late_adapters.pt, config.json
<protocol>/<bench>/scores/       gate scores on XSTest and on the evaluation set
<protocol>/<bench>/threshold.json
<protocol>/<bench>/defended/     defended responses and their judgments
<protocol>/summary.json          unsafe rate, over-refusal, gate AUROC per benchmark
```

### Protocol

- **Evaluation settings.** *In-domain:* AEGIS is trained on a benchmark's train split and
  evaluated on its held-out test split, one adapter per benchmark. The splits are 50/50,
  seed 42, grouped by prompt so variants of the same request stay on one side; the ids are
  in `data/splits/`. *LOBO (leave one benchmark out):* AEGIS is trained on the other two
  benchmarks and evaluated on the whole held-out benchmark. In both settings the evaluated
  data never takes part in threshold selection, and XSTest is never used for training.
- **Training data** (`scripts/build_trainsets.py`). Benign audio is Benign_train with the
  model's own undefended answers as targets. Benign prompts that look harmful and that the
  model refused at baseline are dropped. Harmful targets are generic refusal templates.
  The number of harmful clips equals the number of benign clips. Seed 42.
- **Optimisation.** AdamW, learning rate 2e-4, 4 epochs, gradient accumulation 8, LoRA
  rank 16, `λ_BCE = 1.0`, `λ_L1 = 0.01`.
- **Prompt.** Qwen2-Audio receives the audio only (`PROMPT_MODE=native`); the other models
  receive the audio plus one fixed text prompt. The gate reads a prompt-dependent hidden
  state, so an adapter is valid only under the prompt mode it was trained with.
- **Threshold** (`scripts/calibrate_threshold.py`). XSTest ids are sorted and split
  alternately into a validation half and a test half. The threshold is the smallest gate
  score whose added over-refusal on the validation half is at most 10%, counted on prompts
  the undefended model answered.
- **Closed loop.** The refusal probability sums the next-token probability of refusal
  openers at the last layer. In-domain runs use the multilingual opener list
  (`REFUSAL_PHRASE_SET=multi`, set by `run_aegis.sh`), so non-English refusals count.
- **Metrics** (`scripts/summarize.py`): unsafe rate is the share of evaluated rows that
  Llama-Guard-3-8B labels `unsafe`; over-refusal is the share of XSTest responses matching
  `aegis/refusal.py`, reported over all 250 prompts and over the test half; gate AUROC uses
  raw gate scores with attacks as positives and XSTest as negatives.

### Ablations

```bash
MODEL=qwen2_audio VARIANT=full bash scripts/run_ablation.sh        # LoRA on all layers, no gate
MODEL=qwen2_audio VARIANT=always_on bash scripts/run_ablation.sh   # AEGIS layers, no gate
```

## Tests

```bash
python -m unittest discover tests
```

## Citation

```bibtex
@misc{liao2026aegis,
  title         = {{AEGIS}: Audio Endogenous Guarding via Internal Signals Against Large Audio-Language Model Jailbreaks},
  author        = {Liao, Yu-Ling and Chiu, Tzu-Chin and Chen, Zong-You and Tsai, Chi-Lei and Lo, Shao-Yuan},
  year          = {2026},
  eprint        = {2609.29287},
  archivePrefix = {arXiv},
  primaryClass  = {cs.SD},
  url           = {https://arxiv.org/abs/2609.29287}
}
```

## License

The code is released under the [MIT License](LICENSE). The base models (e.g. Gemma 4,
Llama 3.1 in Ultravox, Llama-Guard-3-8B) and the benchmarks keep their own licenses and
terms of use.
