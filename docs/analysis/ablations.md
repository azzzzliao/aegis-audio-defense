# Ablations

Both variants drop the gate and apply the LoRA unconditionally, with no closed loop.
Same training files, same targets, same judge, same splits as AEGIS; only the
placement and the gating change.

- **Full** — LoRA on every decoder layer (`--no-gate --late-from 0`).
- **Always-on** — LoRA on the AEGIS late layers, no gate (`--no-gate`).

## Qwen2-Audio (paper, Table 2)

| Variant | AJail ↓ | JALM ↓ | SACRED ↓ | ΔOR ↓ |
|---|---:|---:|---:|---:|
| Full | 0.3 | 0.0 | 2.0 | +20.5 |
| Always-on | 0.3 | 2.0 | 2.2 | +20.3 |
| AEGIS (LOBO) | 0.8 | 0.8 | 0.0 | **+3.7** |

Both ungated variants reach comparable safety and pay about 20 points of over-refusal
for it. The gate buys back that cost.

## Gemma-4 and Voxtral (not in the paper)

Unsafe rate (%), LOBO, judged by Llama-Guard-3-8B:

| Model | Benchmark | No defense | Full | Always-on | AEGIS |
|---|---|---:|---:|---:|---:|
| Gemma-4 E4B | AudioJailbreak | 7.9 | 3.8 | 2.2 | **0.1** |
| Gemma-4 E4B | JALM | 31.9 | 9.5 | 9.8 | **7.5** |
| Gemma-4 E4B | SACRED | 17.2 | 0.1 | 1.8 | **0.0** |
| Voxtral Small | AudioJailbreak | 17.9 | 14.3 | 14.8 | **10.3** |
| Voxtral Small | JALM | 22.8 | 28.2 | 29.2 | **13.2** |
| Voxtral Small | SACRED | 26.4 | 0.9 | 1.5 | **0.4** |

XSTest over-refusal (%), one refusal regex, averaged over the three folds:

| Model | No defense | Full | Always-on | AEGIS |
|---|---:|---:|---:|---:|
| Gemma-4 E4B | 6.8 | 13.5 | 21.2 | 16.9 |
| Voxtral Small | 1.6 | 15.1 | 11.1 | 11.2 |

Two things show up that the single-model table cannot:

- **Always-on is dominated on Gemma-4.** Same late layers as AEGIS, and it refuses more
  (21.2 vs 16.9) while leaving more attacks through (2.2 vs 0.1 on AudioJailbreak). The
  gate improves both axes at once: it stays shut on benign input, and the closed loop
  presses harder where the gate does fire.
- **Ungated intervention can be worse than no defense.** On Voxtral's held-out JALM,
  Full and Always-on land at 28.2% and 29.2% against 22.8% undefended. An unconditional
  edit fitted on two families misfires on an unseen third; gating it is what keeps the
  transfer.

## Reproduce

```bash
MODEL=qwen2_audio VARIANT=full bash scripts/run_ablation.sh
MODEL=qwen2_audio VARIANT=always_on bash scripts/run_ablation.sh
```

The ablation adapters were trained for 3 epochs (`EPOCHS=3`, the default in that
script); AEGIS itself uses 4.
