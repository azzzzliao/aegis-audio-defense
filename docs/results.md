# Results

The tables from the paper. Analyses that did not fit in it are in
[`docs/analysis/`](analysis/README.md).

## Unsafe rate across six models and three benchmarks

In-domain training includes the target benchmark (evaluated on its held-out split); LOBO
excludes it entirely. Parentheses give the change from the undefended model. The last
block is XSTest over-refusal: the undefended rate, then the added points.

| Benchmark | Defense | Gemma 4 E4B | Phi-4 | VITA-1.5 | Qwen2-Audio | Ultravox v0.5 | Voxtral Small |
|---|---|---:|---:|---:|---:|---:|---:|
| AJail (Origin) ↓ | No defense | 7.90 | 4.40 | 6.40 | 7.60 | 7.20 | 18.90 |
| | AEGIS (in-domain) | 0.30 (−7.60) | 0.00 (−4.40) | 0.00 (−6.40) | 1.40 (−6.20) | 0.40 (−6.80) | 0.60 (−18.30) |
| | AEGIS (LOBO) | 0.10 (−7.80) | 0.00 (−4.40) | 0.00 (−6.40) | 0.80 (−6.80) | 4.00 (−3.20) | 10.30 (−8.60) |
| JALM (ADiv+SSJ) ↓ | No defense | 16.90 | 66.80 | 1.60 | 28.40 | 4.40 | 22.80 |
| | AEGIS (in-domain) | 0.20 (−16.70) | 1.20 (−65.60) | 0.00 (−1.60) | 0.00 (−28.40) | 0.00 (−4.40) | 0.60 (−22.20) |
| | AEGIS (LOBO) | 7.50 (−9.40) | 7.60 (−59.20) | 0.00 (−1.60) | 0.80 (−27.60) | 2.40 (−2.00) | 13.20 (−9.60) |
| SACRED (MSD) ↓ | No defense | 17.20 | 46.40 | 2.40 | 34.00 | 0.40 | 26.40 |
| | AEGIS (in-domain) | 0.00 (−17.20) | 1.30 (−45.10) | 0.00 (−2.40) | 1.60 (−32.40) | 0.00 (−0.40) | 0.00 (−26.40) |
| | AEGIS (LOBO) | 0.00 (−17.20) | 0.00 (−46.40) | 0.00 (−2.40) | 0.00 (−34.00) | 0.00 (−0.40) | 0.40 (−26.00) |
| Over-refusal (%) ↓ | No defense | 10.8 | 11.6 | 12.0 | 38.4 | 26.8 | 3.2 |
| | AEGIS (in-domain) | +10.4 | +2.4 | +0.5 | +7.7 | +7.5 | +7.7 |
| | AEGIS (LOBO) | +10.1 | +9.9 | +5.1 | +3.7 | +1.3 | +9.6 |

## Ablation on Qwen2-Audio (LOBO)

*Full* applies the LoRA to all decoder layers; *Always-on* keeps the late-layer adapters
but removes the gate. The same ablation on Gemma-4 and Voxtral is
[here](analysis/ablations.md).

| Variant | AJail ↓ | JALM ↓ | SACRED ↓ | ΔOR ↓ |
|---|---:|---:|---:|---:|
| Full | 0.3 | 0.0 | 2.0 | +20.5 |
| Always-on | 0.3 | 2.0 | 2.2 | +20.3 |
| AEGIS (LOBO) | 0.8 | 0.8 | **0.0** | **+3.7** |

## Comparison with representative defenses on Qwen2-Audio

Same Training Dataset setting:
| Method | AJail ↓ | JALM ↓ | SACRED ↓ | OR ↓ |
|---|---:|---:|---:|---:|
| Vanilla | 12.55 | 16.70 | 42.45 | 36.80 |
| RRS (EMNLP'25) | 11.61 | 26.32 | 38.71 | +11.33 |
| OmniSteer (arXiv'26) | 12.21 | 17.02 | 42.96 | **+0.80** |
| ALMGuard (NeurIPS'25) | 13.09 | 22.73 | 43.55 | +12.00 |
| SARSteer (ICML'26) | *4.43* | *6.13* | 26.91 | +41.60 |
| ASR (OpenAI'23) | 12.15 | 31.09 | *3.17* | *+1.40* |
| PBD-JBB (NeurIPS'24) | 9.73 | 26.96 | 33.94 | +12.00 |
| **AEGIS (Ours, LOBO)** | **0.80** | **0.80** | **0.00** | +5.30 |

Faithful Implementation of these baselines:

| Method | AJail ↓ | JALM ↓ | SACRED ↓ | ΔOR ↓ |
|---|---:|---:|---:|---:|
| Vanilla | 7.60 | 28.40 | 34.00 | 0.00 |
| RRS (EMNLP'25) | 6.44 | 12.58 | 17.82 | +46.40 |
| OmniSteer (arXiv'26) | 7.60 | 2.58 | 35.80 | +11.60 |
| ALMGuard (NeurIPS'25) | 14.50 | 19.50 | 44.20 | +12.40 |
| SARSteer (ICML'26) | 3.89 | 4.33 | 34.53 | +42.00 |
| ASR (OpenAI'23) | 12.15 | 31.09 | 3.17 | **+1.40** |
| PBD-JBB (NeurIPS'24) | 9.73 | 26.96 | 33.94 | +12.00 |
| **AEGIS (LOBO)** | **0.80** | **0.80** | **0.00** | +3.70 |
