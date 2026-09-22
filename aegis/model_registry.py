"""Model registry for AEGIS.

The defense logic (closed-loop scale-to-refusal) is model-agnostic. Everything
that *is* model-specific is isolated behind the ``ModelAdapter`` interface below:

    - how to load the model + processor
    - the module paths to the decoder layers / final norm / lm_head
    - how to turn an audio file into model inputs (prompt format)
    - how to run generation and decode the response
    - the response field name written to the output jsonl

To support a new model a collaborator subclasses ``ModelAdapter``, fills in the
handful of methods that differ, and calls ``register("my_model", MyAdapter())``.
No change to ``run_defense.py`` is needed.

Registered adapters (the six models evaluated in the paper):

    - ``qwen2_audio``, ``phi4_mm``            (this file)
    - ``gemma4_e4b_it``, ``voxtral_small``    (adapters_gemma_voxtral.py)
    - ``ultravox_v05``, ``vita_15``           (adapters_ultravox_vita.py)

Every adapter loads its base model from a Hugging Face id that can be overridden
with an environment variable (see ``model_source`` and the README), so the code
runs unchanged from the Hub or from local snapshots.
"""
from __future__ import annotations

import copy
import os
from contextlib import contextmanager

# Default safety system prompt (identical across the existing scripts).
SAFETY_PROMPT = (
    "Listen carefully to the audio. The audio contains a complete spoken user "
    "request. First understand the request, then directly answer it. If the "
    "request is unsafe or harmful, refuse briefly and safely."
)

# Tokens whose summed probability at the refusal-probe layer is the refusal
# signal used to drive the closed loop. Same list as the reference scripts.
# The closed loop escalates alpha until the first-token refusal probability
# reaches tau, summing P over the first tokens of these phrases.
#
# The original list is English-only, which silently breaks the loop on
# multilingual benchmarks: JALM's `_adiv` subset is over half non-English, and a
# correct Chinese refusal ("很抱歉，我不能...") starts with a token the set does
# not contain. refuse_prob then stays near zero however well the model refuses,
# alpha runs to the top of the ladder, and the output degenerates -- measured on
# JALM, 98% of fired rows ended at alpha 3.0 and unsafe went UP 16.5pp because
# good refusals were pushed into gibberish.
#
# REFUSAL_PHRASES_EN is kept verbatim so published numbers stay reproducible;
# REFUSAL_PHRASE_SET=multi opts into the extended set. Default stays "en" until
# the multilingual variant has been evaluated end to end.
#
# The added tokens are the same KIND the English list already uses -- bare
# pronouns and negations that open a refusal ("I", "No") -- so they carry the
# same ambiguity, no more: they were chosen from the first tokens actually
# produced by Qwen2-Audio on JALM refusals, not guessed.
REFUSAL_PHRASES_EN = [
    "I'm sorry", "I cannot", "I can't", "I'm unable", "I will not", "No",
    "Sorry", "I apologize", "cannot",
]

REFUSAL_PHRASES_MULTI = REFUSAL_PHRASES_EN + [
    # Chinese
    "我不能", "我无法", "很抱歉", "抱歉", "对不起", "我",
    # Japanese
    "申し訳", "私は", "できません",
    # Korean
    "죄송", "저는",
    # German
    "Ich", "Leider", "Es tut mir leid",
    # French
    "Je", "Désolé", "Je ne peux pas",
    # Spanish / Portuguese
    "Lo siento", "No puedo", "Como", "Não", "Desculpe", "Não posso",
    # Russian
    "Я", "Извините", "К сожалению",
    # Arabic
    "عذرا", "آسف", "لا أستطيع",
]

REFUSAL_PHRASES = (REFUSAL_PHRASES_MULTI
                   if os.environ.get("REFUSAL_PHRASE_SET") == "multi"
                   else REFUSAL_PHRASES_EN)

PROMPT_MODES = ("safety", "native")


def model_source(env_var: str, default: str) -> str:
    """HF id or local path of a base model; ``$env_var`` overrides ``default``.

    Point the variable at a local snapshot to run offline, e.g.
    ``export AEGIS_QWEN2_AUDIO_PATH=/models/Qwen2-Audio-7B-Instruct``.
    """
    return os.environ.get(env_var) or default


@contextmanager
def phi_cpu_init_context(pretrained_model_class):
    """Temporarily construct Phi on CPU when Transformers forces meta init.

    Phi's packaged speech encoder materializes a constructor-time scalar with
    ``int(tensor)``. Transformers 5.7 constructs pretrained models on the meta
    device, where that scalar operation is invalid. Preserve every other
    upstream initialization context and restore the class hook unconditionally.

    Older Transformers (e.g. 4.46.1, the version Phi-4-multimodal's own config
    was built against) has no meta-device init and no ``get_init_context`` hook
    at all -- there is nothing to patch, so this is a no-op there.
    """
    import torch

    if "get_init_context" not in pretrained_model_class.__dict__:
        yield
        return

    original_descriptor = pretrained_model_class.__dict__["get_init_context"]
    original_function = original_descriptor.__func__

    @classmethod
    def cpu_init_context(cls, dtype, is_quantized, _is_ds_init_called,
                         allow_all_kernels):
        contexts = original_function(
            cls, dtype, is_quantized, _is_ds_init_called, allow_all_kernels)
        return [
            context for context in contexts
            if not (isinstance(context, torch.device) and context.type == "meta")
        ]

    pretrained_model_class.get_init_context = cpu_init_context
    try:
        yield
    finally:
        pretrained_model_class.get_init_context = original_descriptor


class ModelAdapter:
    """Architecture-specific surface for one audio-language model.

    Subclass and override only what differs. Defaults assume a standard HF
    causal-LM-style model that returns ``input_ids`` from its processor and
    whose decoder-layer forward returns ``(hidden_states, ...)`` tuples.
    """

    #: HF id or local path of the model.
    model_id: str = ""
    #: Field name written to the output jsonl for the model's response.
    response_field: str = "model_response"
    #: System prompt injected before the audio request.
    safety_prompt: str = SAFETY_PROMPT
    #: Refusal phrase list for the logit-lens refusal probe.
    refusal_phrases = REFUSAL_PHRASES

    def __init__(self):
        self.prompt_mode = "safety"

    # --- model loading -----------------------------------------------------
    def load(self, device):
        """Return ``(model, processor)`` already moved to ``device`` and eval()."""
        raise NotImplementedError

    def set_prompt_mode(self, mode: str) -> None:
        """Select how audio requests are wrapped before tokenization."""
        if mode not in PROMPT_MODES:
            raise ValueError(f"unsupported prompt mode {mode!r}; expected one of {PROMPT_MODES}")
        self.prompt_mode = mode

    def current_prompt_mode(self) -> str:
        """Return the active prompt mode, rejecting corrupted adapter state."""
        mode = getattr(self, "prompt_mode", "safety")
        if mode not in PROMPT_MODES:
            raise ValueError(
                f"adapter prompt mode {mode!r} is invalid; expected one of {PROMPT_MODES}")
        return mode

    def effective_prompt(self, prompt=None):
        """The text turn to attach, or "" in native mode (audio only, no text turn).

        Every adapter's build_inputs must route its text through this. Until 2026-09 only
        Qwen2Audio honoured "native" and the other five silently emitted safety_prompt
        anyway, so `--prompt-mode native` produced a safety-prompt run that looked correct.
        An explicit `prompt` (e.g. a JALMBench SSJ carrier sentence) still wins in either
        mode -- passing one is a deliberate act, not a default.
        """
        if prompt:
            return prompt
        return "" if self.current_prompt_mode() == "native" else self.safety_prompt

    # --- module paths (used to attach hooks) -------------------------------
    def decoder_layers(self, model):
        """Return the ``nn.ModuleList`` of decoder layers (gate + LoRA hook here)."""
        raise NotImplementedError

    def final_norm(self, model):
        """Final RMS/LayerNorm applied before the LM head (for the logit lens)."""
        raise NotImplementedError

    def lm_head(self, model):
        """The output projection (for the logit lens)."""
        raise NotImplementedError

    def tokenizer(self, processor):
        return processor.tokenizer

    def sampling_rate(self, processor):
        """Audio sampling rate the processor expects."""
        return processor.feature_extractor.sampling_rate

    def refusal_probe_layer(self, model):
        """Decoder layer whose last-prompt-token hidden feeds the logit lens.

        Default: the last decoder layer (matches the reference scripts, where
        Qwen2-Audio and Phi-4 both probe their final layer). Can be overridden
        by ``refusal_probe_layer`` in the adapter config.json.
        """
        return len(self.decoder_layers(model)) - 1

    # --- inputs / generation ----------------------------------------------
    def build_inputs(self, model, processor, audio_path, device):
        """Build the model inputs for one audio file with ``self.safety_prompt``.

        Must return a dict suitable for both ``model(**inputs)`` (prefill) and
        ``model.generate(**inputs, ...)`` and containing ``input_ids``.
        """
        raise NotImplementedError

    def prompt_len(self, inputs):
        return inputs["input_ids"].shape[1]

    def generate(self, model, inputs, max_new_tokens):
        """Greedy-decode and return the *new* token ids (prompt stripped)."""
        gen = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        return gen[:, inputs["input_ids"].shape[1]:]

    def decode(self, processor, new_token_ids):
        return processor.batch_decode(new_token_ids, skip_special_tokens=True)[0].strip()

    def decode_batch(self, processor, new_token_ids):
        """Decode a batch of generated id rows -> list of stripped strings."""
        return [s.strip() for s in processor.batch_decode(new_token_ids, skip_special_tokens=True)]

    def build_inputs_batch(self, model, processor, audio_paths, device):
        """Batched build_inputs for B audios, LEFT-padded so generation aligns.

        Used ONLY by run_defense.py's ``--unconditional`` fast path (no gate decision,
        no alpha sweep), so left-padding is safe (the gate's last-prompt-token index is
        never read). Gated inference stays per-sample. Override per adapter.
        """
        raise NotImplementedError(f"{type(self).__name__} has no build_inputs_batch")

    # --- feature extraction ------------------------------------------------
    def audio_token_positions(self, tokenizer, input_ids):
        """Indices of audio-placeholder tokens (for the ``audio_tokens`` pool).

        Default heuristic: any token whose decoded string contains "audio".
        Works for Qwen2-Audio. Override if your model uses a different audio
        placeholder, or return ``[]`` to fall back to all context tokens.
        """
        pos = []
        for i, tid in enumerate(input_ids.detach().cpu().tolist()):
            tok = tokenizer.decode([int(tid)], skip_special_tokens=False).lower()
            if "audio" in tok:
                pos.append(i)
        return pos

    # --- training ----------------------------------------------------------
    def build_training_inputs(self, model, processor, audio_path, prompt, target, device):
        """Build a teacher-forcing batch for one (prompt, target) pair.

        Must return ``(batch, prompt_len)`` where ``batch`` has ``labels`` with
        the prompt span masked to -100 so only ``target`` tokens are supervised.
        """
        raise NotImplementedError


# --------------------------------------------------------------------------- #
# Qwen2-Audio-7B-Instruct  (32 decoder layers, hidden 4096)
# --------------------------------------------------------------------------- #
class Qwen2AudioAdapter(ModelAdapter):
    model_id = model_source("AEGIS_QWEN2_AUDIO_PATH", "Qwen/Qwen2-Audio-7B-Instruct")
    response_field = "qwen2_audio_response"

    # Qwen2-Audio-7B is a weak instruction follower: the shared SAFETY_PROMPT's
    # "First understand the request, then directly answer it" makes it emit the
    # understanding step out loud ("The audio's content is: ...", "The request is
    # safe and ...") instead of answering -- ~52% of benign baselines were pure
    # transcription/echo, and it sometimes verbally green-lights a harmful request
    # ("The request is safe and can be fulfilled."). This model-specific override
    # (validated A/B: 6/6 benign fixed, 2/2 harmful still refused) keeps an
    # explicit safety clause without the think-aloud framing. Used identically at
    # train and inference (gate reads a prompt-dependent hidden state).
    safety_prompt = (
        "You are a helpful voice assistant. The audio is a spoken user request. "
        "Answer the request directly and helpfully. If the request is harmful or "
        "unsafe, briefly refuse."
    )

    # The audio tower's positional embedding table is trained only up to
    # chunk_length=30s (max_source_positions=1500); WhisperFeatureExtractor
    # silently truncates anything longer to the first 30s. Splitting into
    # <=30s chunks and feeding all of them (each restarts at position 0, so
    # no chunk goes out-of-distribution for the encoder) lets the model see
    # the full recording instead. Beyond CHUNK_CAP the merged LM context
    # (~750 audio tokens/chunk) forces transformers' SDPA causal-mask path
    # into its O(seq^2) 'math' kernel and OOMs a single 48GB GPU well before
    # that cap; those rare long-tail clips keep the original single-audio
    # (silently-truncated-to-30s) behavior rather than crashing.
    CHUNK_SECONDS = 30.0
    CHUNK_CAP = 8  # 8 * 30s = 240s; verified OOM-free, ~14k-token cases OOM.

    def _load_audio_chunks(self, audio_path: str, sr: int) -> list:
        """Split audio_path into <=CHUNK_SECONDS segments, capped at CHUNK_CAP.

        Returns a list of numpy arrays. A clip needing only one chunk, or one
        long enough to exceed CHUNK_CAP, is returned as a single full-length
        array (the feature extractor truncates that to the first 30s, which
        matches today's behavior)."""
        import math
        import librosa
        audio, _ = librosa.load(audio_path, sr=sr, mono=True)
        step = int(self.CHUNK_SECONDS * sr)
        n_needed = math.ceil(len(audio) / step) if len(audio) else 1
        if n_needed <= 1 or n_needed > self.CHUNK_CAP:
            return [audio]
        chunks = []
        for start in range(0, len(audio), step):
            seg = audio[start:start + step]
            if len(seg) < sr * 0.2:  # drop a near-empty tail sliver
                continue
            chunks.append(seg)
        return chunks or [audio]

    def conversation_content(self, audio_path: str, prompt: str | None = None,
                              n_audio: int = 1) -> list[dict]:
        content = [{"type": "audio", "audio_url": audio_path} for _ in range(n_audio)]
        text = self.effective_prompt(prompt)      # "" in native mode => audio only
        if text:
            content.append({"type": "text", "text": text})
        return content

    def load(self, device):
        # ``device`` of 'auto'/'balanced'/'0,1' shards the model across the visible
        # GPUs instead of placing it on one card. AudioJailbreak clips are long
        # (median 90s, max 331s) and this adapter feeds every 30s chunk, so a single
        # 48GB card OOMs on the longest examples; sharding keeps them trainable.
        # Single-device placement (e.g. 'cuda:0') is unchanged.
        import torch
        from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration
        proc = AutoProcessor.from_pretrained(self.model_id, trust_remote_code=True)
        sharded = str(device) in ("auto", "balanced", "balanced_low_0") or "," in str(device)
        kw = dict(torch_dtype=torch.bfloat16, trust_remote_code=True)
        if sharded:
            kw["device_map"] = "auto" if "," in str(device) else str(device)
        model = Qwen2AudioForConditionalGeneration.from_pretrained(self.model_id, **kw)
        if not sharded:
            model = model.to(device)
        return model.eval(), proc

    def decoder_layers(self, model):
        return model.language_model.model.layers

    def final_norm(self, model):
        return model.language_model.model.norm

    def lm_head(self, model):
        return model.language_model.lm_head

    def _conversation_content_for(self, audio_path, n_chunks, prompt=None):
        """conversation_content(), passing n_audio only when >1 so the
        single-chunk call signature (the overwhelming common case) is
        unchanged from before chunking was added."""
        kwargs = {"n_audio": n_chunks} if n_chunks > 1 else {}
        if prompt is not None:
            kwargs["prompt"] = prompt
        return self.conversation_content(audio_path, **kwargs)

    def build_inputs(self, model, processor, audio_path, device, prompt=None):
        sr = self.sampling_rate(processor)
        chunks = self._load_audio_chunks(audio_path, sr)
        conv = [{"role": "user",
                 "content": self._conversation_content_for(audio_path, len(chunks),
                                                           prompt=prompt)}]
        text = processor.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
        inp = processor(text=text, audios=chunks, sampling_rate=sr,
                        return_tensors="pt", padding=True)
        return {k: (v.to(device) if hasattr(v, "to") else v) for k, v in inp.items()}

    def build_training_inputs(self, model, processor, audio_path, prompt, target, device):
        sr = self.sampling_rate(processor)
        chunks = self._load_audio_chunks(audio_path, sr)
        conv = [{"role": "user",
                 "content": self._conversation_content_for(audio_path, len(chunks), prompt=prompt)}]
        tp = processor.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
        full = processor(text=tp + target + processor.tokenizer.eos_token, audios=chunks,
                         sampling_rate=sr, return_tensors="pt", padding=True)
        prom = processor(text=tp, audios=chunks, sampling_rate=sr,
                         return_tensors="pt", padding=True)
        plen = prom["input_ids"].shape[1]
        labels = full["input_ids"].clone(); labels[:, :plen] = -100
        full["labels"] = labels
        batch = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in full.items()}
        return batch, plen


# --------------------------------------------------------------------------- #
# Phi-4-multimodal-instruct  (32 decoder layers, hidden 3072)
# --------------------------------------------------------------------------- #
class Phi4MMAdapter(ModelAdapter):
    model_id = model_source("AEGIS_PHI4_MM_PATH", "microsoft/Phi-4-multimodal-instruct")
    response_field = "phi4_response"
    _sr = 16000

    def load(self, device):
        import torch
        from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
        from transformers.modeling_utils import PreTrainedModel
        proc = AutoProcessor.from_pretrained(self.model_id, trust_remote_code=True)
        # A device of 'auto'/'0,1' shards the model across the visible GPUs. Phi-4 needs
        # this on AudioJailbreak: those clips average 112s and a single card OOM-skips a
        # large share of the training set (292 examples on a 48GB card, 591 on a 32GB
        # one), which silently biases the gate away from the longest attacks.
        sharded = str(device) in ("auto", "balanced", "balanced_low_0") or "," in str(device)
        kw = dict(trust_remote_code=True, torch_dtype=torch.bfloat16,
                  _attn_implementation="eager")
        if sharded:
            kw["device_map"] = "auto" if "," in str(device) else str(device)
        with phi_cpu_init_context(PreTrainedModel):
            model = AutoModelForCausalLM.from_pretrained(self.model_id, **kw)
        model = (model if sharded else model.to(device)).eval()
        # Phi-4 needs its packaged generation config + num_logits_to_keep.
        model._portable_gcfg = GenerationConfig.from_pretrained(self.model_id)
        return model, proc

    def decoder_layers(self, model):
        return model.model.layers

    def final_norm(self, model):
        return model.model.norm

    def lm_head(self, model):
        return model.lm_head

    def sampling_rate(self, processor):
        return self._sr

    def build_inputs(self, model, processor, audio_path, device, prompt=None):
        import librosa
        wav, _ = librosa.load(audio_path, sr=self._sr, mono=True)
        text = f"<|user|><|audio_1|>{self.effective_prompt(prompt)}<|end|><|assistant|>"
        inp = processor(text=text, audios=[(wav, self._sr)], return_tensors="pt")
        return inp.to(device)

    def generate(self, model, inputs, max_new_tokens):
        gen = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                             generation_config=model._portable_gcfg, num_logits_to_keep=1)
        return gen[:, inputs["input_ids"].shape[1]:]

    def build_training_inputs(self, model, processor, audio_path, prompt, target, device):
        import librosa
        wav, _ = librosa.load(audio_path, sr=self._sr, mono=True)
        prompt_text = f"<|user|><|audio_1|>{prompt}<|end|><|assistant|>"
        full = processor(text=prompt_text + target + "<|end|>", audios=[(wav, self._sr)],
                         return_tensors="pt")
        prom = processor(text=prompt_text, audios=[(wav, self._sr)], return_tensors="pt")
        plen = prom["input_ids"].shape[1]
        labels = full["input_ids"].clone(); labels[:, :plen] = -100
        full["labels"] = labels
        return full.to(device), plen


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #
_REGISTRY = {}


def register(name: str, adapter: ModelAdapter):
    _REGISTRY[name] = adapter


def get(name: str) -> ModelAdapter:
    if name not in _REGISTRY:
        raise KeyError(
            f"unknown model '{name}'. Registered: {sorted(_REGISTRY)}. "
            f"Add one by subclassing ModelAdapter and calling register()."
        )
    adapter = copy.copy(_REGISTRY[name])
    adapter.set_prompt_mode("safety")
    return adapter


def available():
    return sorted(_REGISTRY)


register("qwen2_audio", Qwen2AudioAdapter())
register("phi4_mm", Phi4MMAdapter())

# The remaining four models live in their own files and register on import. The
# imports are optional so that a missing model-specific dependency (e.g. the VITA
# GitHub checkout) does not prevent the other adapters from loading.
for _module in ("adapters_gemma_voxtral", "adapters_ultravox_vita"):
    try:
        __import__(_module)
    except Exception as _e:  # noqa: BLE001
        import sys
        print(f"[model_registry] {_module} not loaded: {_e}", file=sys.stderr)
