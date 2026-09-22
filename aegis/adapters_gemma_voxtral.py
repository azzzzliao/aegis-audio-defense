"""Gemma 4 E4B-it and Voxtral-Small-24B adapters.

    - ``gemma4_e4b_it``  google/gemma-4-E4B-it (42 decoder layers, hidden 2560)
    - ``voxtral_small``  mistralai/Voxtral-Small-24B-2507 (40 decoder layers, hidden 5120)

Module paths (decoder layers / final norm / lm_head) are resolved against a list of
candidate dotted paths at load time, because HF nests these differently across
multimodal wrappers and Transformers versions.
"""
from __future__ import annotations

import torch

from model_registry import ModelAdapter, model_source, register

# Text turn used for these models at training AND inference. The gate reads the
# last-prompt-token hidden state, which depends on this prompt, so an adapter is only
# valid under the prompt it was trained with.
NEUTRAL_PROMPT = "You will hear an audio instruction. Please respond to the spoken user request."

# Candidate dotted paths covering the common HF multimodal nestings for both models.
_LAYER_PATHS = ["language_model.model.layers", "model.language_model.layers",
                "language_model.layers", "model.model.language_model.layers", "model.layers"]
_NORM_PATHS = ["language_model.model.norm", "model.language_model.norm",
               "language_model.norm", "model.model.language_model.norm", "model.norm"]
_HEAD_PATHS = ["lm_head", "language_model.lm_head", "model.lm_head",
               "language_model.model.lm_head"]


def _resolve(model, paths, want_list=False):
    """Return the first dotted path on `model` that resolves (non-empty if want_list)."""
    errs = []
    for p in paths:
        try:
            mod = model
            for part in p.split("."):
                mod = getattr(mod, part)
            if want_list and len(mod) == 0:
                continue
            return mod
        except Exception as e:  # noqa: BLE001
            errs.append(f"{p}: {type(e).__name__}")
    raise AttributeError(f"none of {paths} resolved on {type(model).__name__} ({errs})")


def _torch_dtype(name):
    return {"bf16": torch.bfloat16, "fp16": torch.float16,
            "fp32": torch.float32}.get(name, getattr(torch, name, torch.bfloat16))


def _is_device_map(device):
    """A `device` of 'auto'/'balanced'/'0,1' means shard the model (model parallel)
    rather than place it on one device."""
    return str(device) in ("auto", "balanced", "balanced_low_0") or "," in str(device)


def _load_placed(loader, model_id, device, **extra_kw):
    """Load a model on a single device (device='cuda:0') OR sharded across GPUs
    (device='auto'/'balanced'/'0,1'). Set CUDA_VISIBLE_DEVICES to pick which GPUs."""
    kw = dict(trust_remote_code=True, **extra_kw)
    if _is_device_map(device):
        kw["device_map"] = "auto" if "," in str(device) else str(device)
    try:
        model = loader.from_pretrained(model_id, dtype=torch.bfloat16, **kw)
    except TypeError:
        model = loader.from_pretrained(model_id, torch_dtype=torch.bfloat16, **kw)
    if not _is_device_map(device):
        model = model.to(device)
    return model.eval()


class _AudioLMAdapter(ModelAdapter):
    """Shared behaviour for chat-template audio LMs whose decoder/norm/head live behind
    one of the candidate paths above, and whose training batch is built by appending the
    tokenized target after the (audio-bearing) prompt with the prompt span masked."""

    def decoder_layers(self, model):
        return _resolve(model, _LAYER_PATHS, want_list=True)

    def final_norm(self, model):
        return _resolve(model, _NORM_PATHS)

    def lm_head(self, model):
        return _resolve(model, _HEAD_PATHS)

    def build_training_inputs(self, model, processor, audio_path, prompt, target, device):
        # Reuse build_inputs (audio + chat-template prompt), then append target tokens.
        base = dict(self.build_inputs(model, processor, audio_path, device))
        tok = self.tokenizer(processor)
        tgt = tok(target, add_special_tokens=False)["input_ids"]
        if tok.eos_token_id is not None:
            tgt = tgt + [tok.eos_token_id]
        tgt_t = torch.tensor([tgt], device=device, dtype=base["input_ids"].dtype)
        plen = base["input_ids"].shape[1]
        base["input_ids"] = torch.cat([base["input_ids"], tgt_t], dim=1)
        if "attention_mask" in base and base["attention_mask"] is not None:
            ones = torch.ones((1, tgt_t.shape[1]), device=device,
                              dtype=base["attention_mask"].dtype)
            base["attention_mask"] = torch.cat([base["attention_mask"], ones], dim=1)
        labels = base["input_ids"].clone()
        labels[:, :plen] = -100
        base["labels"] = labels
        return base, plen


# --------------------------------------------------------------------------- #
# Gemma 4 E4B-it  (Gemma4ForConditionalGeneration, 42 decoder layers, hidden 2560)
# --------------------------------------------------------------------------- #
class Gemma4E4BItAdapter(_AudioLMAdapter):
    model_id = model_source("AEGIS_GEMMA4_PATH", "google/gemma-4-E4B-it")
    response_field = "gemma4_response"
    safety_prompt = NEUTRAL_PROMPT

    def load(self, device):
        from transformers import AutoModelForMultimodalLM, AutoProcessor
        proc = AutoProcessor.from_pretrained(self.model_id, trust_remote_code=True)
        # NOTE: must be "sdpa" (the model default). attn_implementation="eager" silently
        # breaks gemma-4 audio grounding (~66% of inputs return "I cannot hear any audio"),
        # whereas sdpa hears correctly (verified XSTest: eager 7/15 fail vs sdpa 0/15). The
        # defense reads decoder-layer *hidden states* (not attention weights), so eager buys
        # nothing here.
        model = _load_placed(AutoModelForMultimodalLM, self.model_id, device,
                             attn_implementation="sdpa")
        return model, proc

    def build_inputs(self, model, processor, audio_path, device, prompt=None):
        # ``prompt`` overrides safety_prompt for benchmarks whose attack lives partly in the
        # text turn -- JALMBench SSJ, where the audio only spells the masked keyword and the
        # carrier sentence ("... a strategy for [word] into a government database") must be
        # supplied for the request to exist at all. Defaults to None => unchanged behaviour.
        # The gate reads a prompt-dependent hidden state, so an adapter trained under the
        # neutral prompt must NOT be evaluated with an override without retraining.
        from pathlib import Path
        content = [{"type": "audio", "audio": str(Path(audio_path).resolve())}]
        text = self.effective_prompt(prompt)
        if text:                                  # native mode => audio only, no text turn
            content.append({"type": "text", "text": text})
        inp = processor.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=True, return_dict=True, return_tensors="pt", add_generation_prompt=True)
        return {k: (v.to(device) if hasattr(v, "to") else v) for k, v in inp.items()}

    def build_inputs_batch(self, model, processor, audio_paths, device):
        from pathlib import Path
        text = self.effective_prompt()
        convs = [[{"role": "user", "content":
                   [{"type": "audio", "audio": str(Path(p).resolve())}]
                   + ([{"type": "text", "text": text}] if text else [])}]
                 for p in audio_paths]
        tok = self.tokenizer(processor)
        old = getattr(tok, "padding_side", None)
        if old is not None:
            tok.padding_side = "left"                       # left-pad so generation aligns
        try:
            inp = processor.apply_chat_template(
                convs, tokenize=True, return_dict=True, return_tensors="pt",
                add_generation_prompt=True, padding=True)
        finally:
            if old is not None:
                tok.padding_side = old
        return {k: (v.to(device) if hasattr(v, "to") else v) for k, v in inp.items()}


# --------------------------------------------------------------------------- #
# Voxtral-Small-24B-2507
# --------------------------------------------------------------------------- #
class VoxtralSmallAdapter(_AudioLMAdapter):
    model_id = model_source("AEGIS_VOXTRAL_PATH", "mistralai/Voxtral-Small-24B-2507")
    response_field = "voxtral_response"
    safety_prompt = NEUTRAL_PROMPT

    def load(self, device):
        from transformers import AutoProcessor, VoxtralForConditionalGeneration
        proc = AutoProcessor.from_pretrained(self.model_id, trust_remote_code=True)
        model = _load_placed(VoxtralForConditionalGeneration, self.model_id, device)
        return model, proc

    def build_inputs(self, model, processor, audio_path, device, prompt=None):
        from pathlib import Path
        content = [{"type": "audio", "path": str(Path(audio_path).resolve())}]
        text = self.effective_prompt(prompt)
        if text:                                  # native mode => audio only, no text turn
            content.append({"type": "text", "text": text})
        inp = processor.apply_chat_template([{"role": "user", "content": content}])
        if hasattr(inp, "to"):
            inp = inp.to(device)
        return {k: (v.to(device) if hasattr(v, "to") else v) for k, v in dict(inp).items()}

    def build_inputs_batch(self, model, processor, audio_paths, device):
        from pathlib import Path
        text = self.effective_prompt()
        convs = [[{"role": "user", "content":
                   [{"type": "audio", "path": str(Path(p).resolve())}]
                   + ([{"type": "text", "text": text}] if text else [])}]
                 for p in audio_paths]
        tok = self.tokenizer(processor)
        old = getattr(tok, "padding_side", None)
        if old is not None:
            tok.padding_side = "left"
        try:
            inp = processor.apply_chat_template(convs, padding=True)
        finally:
            if old is not None:
                tok.padding_side = old
        if hasattr(inp, "to"):
            inp = inp.to(device)
        return {k: (v.to(device) if hasattr(v, "to") else v) for k, v in dict(inp).items()}


register("gemma4_e4b_it", Gemma4E4BItAdapter())
register("voxtral_small", VoxtralSmallAdapter())
