# Porting the Defense to a New Model

You implement **one class**. The pipeline (`run_model`, `extract_hidden`,
`train_gate_lora`, `run_defense`) is generic and reads everything model-specific
through it.

## 1. The adapter interface

Subclass `ModelAdapter` (`aegis/model_registry.py`) and override the methods that
differ for your model. The six shipped adapters are working examples:
`Qwen2AudioAdapter` (a `*ForConditionalGeneration` model), `Phi4MMAdapter` (an
`AutoModelForCausalLM` with its own prompt format), the chat-template adapters in
`adapters_gemma_voxtral.py`, and the custom-processor adapters in
`adapters_ultravox_vita.py`. Add your adapter to a new module and import it at the
end of `model_registry.py`, next to the existing ones.

```python
class MyModelAdapter(ModelAdapter):
    model_id = "org/my-audio-model"
    response_field = "mymodel_response"      # field written to output jsonl

    def load(self, device):
        # return (model, processor), already .to(device).eval()
        ...

    # --- module paths: where the decoder layers / final norm / head live ---
    def decoder_layers(self, model): return model.model.layers
    def final_norm(self, model):     return model.model.norm
    def lm_head(self, model):        return model.lm_head

    # --- turn one audio file into model inputs (your prompt format) ---
    def build_inputs(self, model, processor, audio_path, device):
        # must return a dict with input_ids, ready for model(**inp) and generate
        ...

    # --- teacher-forcing batch for training (prompt span masked to -100) ---
    def build_training_inputs(self, model, processor, audio_path, prompt, target, device):
        # return (batch_with_labels, prompt_len)
        ...

register("my_model", MyModelAdapter())
```

Then everything works with `--model my_model`.

### Methods with sensible defaults (override only if needed)

- `tokenizer(processor)` → `processor.tokenizer`
- `sampling_rate(processor)` → `processor.feature_extractor.sampling_rate`
- `refusal_probe_layer(model)` → **last** decoder layer (where the logit lens
  reads the refusal signal; Qwen2-Audio and Phi-4 both use their final layer)
- `prompt_len(inputs)` → `inputs["input_ids"].shape[1]`
- `generate(model, inputs, max_new_tokens)` → greedy decode, strip the prompt
- `decode(processor, new_token_ids)` → batch_decode, skip specials
- `audio_token_positions(tokenizer, input_ids)` → indices of tokens whose decoded
  string contains `"audio"` (used for the `audio_tokens` feature pool)

## 2. Two things that commonly differ — check these

**Module paths.** Different wrappers nest the decoder differently:

| model | decoder layers | norm | head |
|---|---|---|---|
| Qwen2-Audio | `model.language_model.model.layers` | `.language_model.model.norm` | `.language_model.lm_head` |
| Phi-4-MM | `model.model.layers` | `.model.norm` | `.model.lm_head` |

Print your model once and find them: `print(model)`.

**Generation extras.** Some models need special `generate` args (Phi-4 needs its
packaged `GenerationConfig` + `num_logits_to_keep=1`). Override `generate` and/or
stash state in `load` (see `Phi4MMAdapter`).

## 3. Audio-token positions (the one fuzzy bit)

The `audio_tokens` feature pool averages hidden states over the audio
placeholder positions. The default heuristic (token text contains "audio")
works for Qwen2-Audio. If your model uses a different placeholder:

- override `audio_token_positions` to return the right indices, **or**
- return `[]` to fall back to all context tokens (then `audio_tokens` equals
  `mean_context` — fine, just document it).

This only affects one of three feature pools; `last` and `mean_context` are
unaffected and the gate usually trains on a mid-layer `last`/`mean_context` pool.

## 4. Sparse layers — if you didn't extract every layer

The gate hook, LoRA adapters, and refusal probe attach to **absolute** decoder
layers on the live model. Two cases:

- **You trained the gate directly on the live model** (what `train_gate_lora.py`
  does) → `gate_layer`/`late` are already absolute. Nothing to do.
- **You trained the gate on a sparse NPZ** (e.g. only layers `[0,8,16,24,32]`
  were dumped, `extract_hidden.py --layers 0,8,16,24,32`) → the index the probe
  picked is a *position* in that array. To deploy, edit the adapter
  `config.json`:

  ```json
  {
    "gate_layer": 2,
    "late": [3, 4],
    "hidden": 4096, "rank": 16,
    "layer_space": "feature_positions",
    "layer_indices": [0, 8, 16, 24, 32],
    "refusal_probe_layer": 4
  }
  ```

  `run_defense.py` maps positions → absolute layers (`2→16`, `3→24`, `4→32`) and
  validates they exist on the model. If indices are out of range you get an
  explicit error telling you to set `layer_space`/`layer_indices`.

Layer indices do not transfer across models — re-pick `gate_layer`/`late-from`
for each model: train per-layer probes on `aegis/extract_hidden.py` features with
`aegis/analysis/cross_benchmark.py` and read the gate from the middle layer where
held-out AUROC peaks, intervening on the last 30-40% of layers.

## 5. Different layer counts

Handled automatically: `late` is `range(late_from, n_layers)` and the probe
defaults to the last layer, both derived from `len(decoder_layers(model))`.
Just pass a `--gate-layer` / `--late-from` valid for your model's depth.

## 6. Validate before a full run

```bash
# tiny smoke test: 5 examples through each stage
python aegis/run_model.py --model my_model --jobs t:manifest.jsonl --out-dir /tmp/t --limit 5
python aegis/judge.py     --input /tmp/t/t.jsonl --out /tmp/t_judged.jsonl --response-key mymodel_response
python aegis/extract_hidden.py --model my_model --responses /tmp/t_judged.jsonl \
    --out-npz /tmp/t.npz --out-meta /tmp/t_meta.jsonl --limit 5
```

Check: responses are non-empty, judge labels look sane, and the NPZ hidden
arrays are `[N, L, D]` with consistent shapes (the extractor errors loudly if
not).
