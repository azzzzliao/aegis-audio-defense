# Ablations

Both variants drop the risk gate and apply the LoRA unconditionally, with no closed-loop
scaling. Same leave-one-benchmark-out splits, training files, targets and judge as AEGIS;
only the placement and the gating change.

- **Full** — LoRA on every decoder layer (`--no-gate --late-from 0`).
- **Always-on** — LoRA on the AEGIS late layers, gate removed (`--no-gate`).

Both are trained for 3 epochs where AEGIS uses 4, and are run with
`scripts/run_ablation.sh` (`VARIANT=full` or `VARIANT=always_on`).

AudioJailbreak, SACRED and JALM report held-out unsafe rate (%); Avg. ΔXSTest reports the
average added over-refusal across the held-out settings, in percentage points. Bold marks
the best value in a column within each model.

| Model | Variant | AudioJailbreak ↓ | SACRED ↓ | JALM ↓ | Avg. ΔXSTest ↓ |
|---|---|---:|---:|---:|---:|
| Gemma 4 E4B | Full | 3.8 | 0.1 | 9.5 | **+6.7** |
| Gemma 4 E4B | Always-on | 2.2 | 1.8 | 9.8 | +14.4 |
| Gemma 4 E4B | AEGIS | **0.1** | **0.0** | **7.5** | +10.1 |
| Phi-4 | Full | 1.6 | **0.0** | 18.0 | +19.7 |
| Phi-4 | Always-on | 0.8 | 30.4 | 54.4 | **+1.1** |
| Phi-4 | AEGIS | **0.0** | **0.0** | **7.6** | +9.9 |
| VITA-1.5 | Full | **0.0** | **0.0** | 16.8 | −18.9 |
| VITA-1.5 | Always-on | **0.0** | **0.0** | **0.0** | **−19.2** |
| VITA-1.5 | AEGIS | **0.0** | **0.0** | **0.0** | +5.1 |
| Qwen2-Audio | Full | 4.4 | 20.4 | 34.0 | +20.5 |
| Qwen2-Audio | Always-on | **0.8** | 2.4 | 21.2 | +20.3 |
| Qwen2-Audio | AEGIS | **0.8** | **0.0** | **0.8** | **+3.7** |
| Ultravox v0.5 | Full | 8.8 | **0.0** | 2.8 | **−4.0** |
| Ultravox v0.5 | Always-on | **1.2** | **0.0** | **1.2** | +31.2 |
| Ultravox v0.5 | AEGIS | 4.0 | **0.0** | 2.4 | +1.3 |
| Voxtral Small | Full | 14.3 | 0.9 | 28.2 | +13.5 |
| Voxtral Small | Always-on | 14.8 | 29.2 | **1.5** | +11.1 |
| Voxtral Small | AEGIS | **10.3** | **0.4** | 13.2 | **+9.6** |
