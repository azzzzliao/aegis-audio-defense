"""Ultravox v0.5 and VITA-1.5 adapters.

    - ``ultravox_v05``  fixie-ai/ultravox-v0_5-llama-3_1-8b (Llama-3.1-8B + whisper-v3-turbo)
    - ``vita_15``       VITA-MLLM/VITA-1.5 (Qwen2-7B + custom audio encoder)

The two upstreams pin incompatible Transformers versions (Ultravox 4.48.1, VITA
4.41.1), so each needs its own Python environment; see the README.
"""
from __future__ import annotations

import os

import torch

from adapters_gemma_voxtral import NEUTRAL_PROMPT, _AudioLMAdapter
from model_registry import model_source, register


# --------------------------------------------------------------------------- #
# Ultravox v0.5 (UltravoxModel: Llama-3.1-8B-Instruct + whisper-large-v3-turbo,
# 32 decoder layers, hidden 4096)
# --------------------------------------------------------------------------- #
class UltravoxV05Adapter(_AudioLMAdapter):
    """Ultravox keeps a plain LlamaForCausalLM under ``language_model``, so the
    shared decoder/norm/head resolution and training-batch assembly apply as-is.

    Two model-specific details:
      * The shipped config points ``text_model_id`` at the gated HF repo
        ``meta-llama/Llama-3.1-8B-Instruct`` (needs an HF token with access).
        Set ``AEGIS_LLAMA31_PATH`` to a local snapshot to load it offline.
      * Its processor is not a chat-template processor: it takes ``text`` with an
        explicit ``<|audio|>`` placeholder plus a raw waveform, so build_inputs
        renders the Llama-3.1 chat template itself and substitutes the placeholder.
    """

    model_id = model_source("AEGIS_ULTRAVOX_PATH", "fixie-ai/ultravox-v0_5-llama-3_1-8b")
    text_model_path = os.environ.get("AEGIS_LLAMA31_PATH")
    response_field = "ultravox_response"
    safety_prompt = NEUTRAL_PROMPT
    _sr = 16000

    def load(self, device):
        # MUST load through device_map, single card included. Ultravox builds its
        # Whisper audio tower inside ``accelerate.init_empty_weights()``, and only
        # accelerate's dispatch path materializes those meta tensors from the
        # checkpoint. A plain ``.to(device)`` leaves all 487 audio_tower parameters
        # on meta while Transformers still reports missing_keys=0, so the failure is
        # silent until the first forward pass.
        from transformers import AutoConfig, AutoModel, AutoProcessor
        cfg = AutoConfig.from_pretrained(self.model_id, trust_remote_code=True)
        if self.text_model_path:
            cfg.text_model_id = self.text_model_path   # keep the backbone local/offline
        proc = AutoProcessor.from_pretrained(self.model_id, trust_remote_code=True)
        if str(device) in ("auto", "balanced", "balanced_low_0") or "," in str(device):
            device_map = "auto" if "," in str(device) else str(device)
        else:
            device_map = {"": str(device)}
        model = AutoModel.from_pretrained(
            self.model_id, config=cfg, trust_remote_code=True,
            torch_dtype=torch.bfloat16, device_map=device_map)
        stranded = [n for n, p in model.named_parameters() if p.is_meta]
        if stranded:
            raise RuntimeError(
                f"ultravox load left {len(stranded)} parameters on meta "
                f"(e.g. {stranded[:3]}); refusing to run on an unmaterialized model")
        return model.eval(), proc

    def sampling_rate(self, processor):
        return self._sr

    def build_inputs(self, model, processor, audio_path, device, prompt=None):
        import librosa
        wav, _ = librosa.load(audio_path, sr=self._sr, mono=True)
        tok = self.tokenizer(processor)
        p = self.effective_prompt(prompt)
        content = f"<|audio|>\n{p}" if p else "<|audio|>"   # native => audio only
        text = tok.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=False, add_generation_prompt=True)
        inp = processor(text=text, audio=wav, sampling_rate=self._sr,
                        return_tensors="pt")
        return {k: (v.to(device) if hasattr(v, "to") else v) for k, v in dict(inp).items()}


register("ultravox_v05", UltravoxV05Adapter())


# --------------------------------------------------------------------------- #
# VITA-1.5 (VITAQwen2ForCausalLM: Qwen2-7B backbone + custom audio encoder,
# 28 decoder layers, hidden 3584)
# --------------------------------------------------------------------------- #
class Vita15Adapter(_AudioLMAdapter):
    """VITA ships no ``auto_map``, so the architecture cannot be reached through
    ``trust_remote_code``; the implementation lives only in the GitHub checkout
    (https://github.com/VITA-MLLM/VITA), whose path is given by ``AEGIS_VITA_REPO``
    and put on sys.path here.

    Its audio path differs from every other adapter in three ways, all taken from the
    project's own ``video_audio_demo.py``:
      * audio features come from ``model.get_audio_encoder().audio_processor``, and the
        model wants an ``audios`` dict of tensors plus two length tensors, not raw wav;
      * the prompt is rendered with VITA's ``qwen2p5_instruct`` conversation template and
        tokenized by ``tokenizer_image_audio_token`` (the audio token is not in the
        tokenizer vocabulary, it is spliced in by index);
      * even an audio-only request must pass a zero image tensor, because the forward
        pass always runs the vision tower.
    """

    model_id = model_source("AEGIS_VITA_PATH", "VITA-MLLM/VITA-1.5")
    repo_path = os.environ.get("AEGIS_VITA_REPO", "external/VITA")
    response_field = "vita_response"
    safety_prompt = NEUTRAL_PROMPT
    model_type = "qwen2p5_instruct"
    conv_mode = "qwen2p5_instruct"
    _sr = 16000
    # Positional-encoding limit of the packaged whale encoder, minus headroom.
    max_audio_frames = 4800
    frames_per_llm_token = 8          # audio_for_llm_lens scales with this
    truncated_clips = 0

    def _snapshot(self):
        """Local checkpoint directory (VITA's builder needs a path, not a Hub id)."""
        if os.path.isdir(self.model_id):
            return self.model_id
        from huggingface_hub import snapshot_download
        return snapshot_download(self.model_id)

    def load(self, device):
        import sys
        if self.repo_path not in sys.path:
            sys.path.insert(0, self.repo_path)
        import torch as _torch
        from vita.model.builder import load_pretrained_model
        from vita.util.mm_utils import get_model_name_from_path

        path = self._snapshot()
        # VITA's builder passes ``device`` straight to torch, which rejects "auto";
        # only ``device_map`` may carry it. So sharding is requested through
        # device_map while ``device`` stays a concrete string.
        sharded = str(device) in ("auto", "balanced", "balanced_low_0") or "," in str(device)
        dev_map = ("auto" if "," in str(device) else str(device)) if sharded else {"": str(device)}
        tokenizer, model, _image_processor, _ctx = load_pretrained_model(
            path, None, get_model_name_from_path(path), self.model_type,
            device_map=dev_map, device=("cuda" if sharded else str(device)))
        model.resize_token_embeddings(len(tokenizer))
        vision_tower = model.get_vision_tower()
        if not vision_tower.is_loaded:
            vision_tower.load_model()
        # VITA's builder hardcodes fp16 everywhere, which overflows to a non-finite LM
        # loss on the first training step (every other model here runs bf16; fp16's
        # 5-bit exponent is the problem, not its mantissa). Put the WHOLE model on one
        # dtype: vita_arch feeds the audio encoder's output straight into the decoder
        # with no cast, so a mixed setup fails with "mat1 and mat2 must have the same
        # dtype" as soon as audio is attached.
        self._dtype = _torch.bfloat16 if _torch.cuda.is_bf16_supported() else _torch.float16
        model.to(dtype=self._dtype)
        audio_encoder = model.get_audio_encoder()
        # Bundle what build_inputs needs; the framework only ever passes this back in.
        proc = {"tokenizer": tokenizer, "audio_processor": audio_encoder.audio_processor}
        return model.eval(), proc

    def tokenizer(self, processor):
        return processor["tokenizer"]

    def decode(self, processor, new_token_ids):
        return processor["tokenizer"].batch_decode(
            new_token_ids, skip_special_tokens=True)[0].strip()

    def sampling_rate(self, processor):
        return self._sr

    def generate(self, model, inputs, max_new_tokens):
        # VITA's generate takes the token tensor positionally as ``inputs`` (not the
        # ``input_ids`` keyword the base adapter expands), and reads images/audios as
        # named arguments. Expanding **inputs would leave ``inputs`` unset and the
        # multimodal path then dereferences None.
        gen = model.generate(inputs["input_ids"], images=inputs["images"],
                             audios=inputs["audios"], max_new_tokens=max_new_tokens,
                             do_sample=False)
        seq = gen.sequences if hasattr(gen, "sequences") else gen
        # VITA returns ONLY the newly generated tokens; the prompt is consumed as
        # inputs_embeds by the multimodal path and never re-emitted. Slicing off
        # prompt_len (as every other adapter must) would drop the whole answer, so
        # only slice when the prompt is actually present.
        plen = inputs["input_ids"].shape[1]
        return seq[:, plen:] if seq.shape[1] > plen else seq

    def build_inputs(self, model, processor, audio_path, device, prompt=None):
        import sys
        if self.repo_path not in sys.path:
            sys.path.insert(0, self.repo_path)
        import torch as _torch
        from vita.constants import DEFAULT_AUDIO_TOKEN, IMAGE_TOKEN_INDEX
        from vita.conversation import conv_templates
        from vita.util.mm_utils import tokenizer_image_audio_token

        audio, audio_for_llm_lens = processor["audio_processor"].process(audio_path)
        # VITA's whale encoder builds a fixed positional-encoding table, so a clip whose
        # feature length exceeds it dies with "size of tensor a (8283) must match tensor
        # b (5000)". Unlike Qwen2-Audio/Ultravox there is no chunking path here, so the
        # only option is to truncate. AudioJailbreak averages 112s and trips this; the
        # cap is recorded on the adapter so the truncation is visible in reporting.
        cap = self.max_audio_frames
        if audio.shape[0] > cap:
            self.truncated_clips += 1
            audio = audio[:cap]
            audio_for_llm_lens = min(int(audio_for_llm_lens),
                                     max(1, cap // self.frames_per_llm_token))
        dt = getattr(self, "_dtype", _torch.float16)
        audios = {
            "audios": _torch.unsqueeze(audio, 0).to(device=device, dtype=dt),
            "lengths": _torch.tensor([audio.shape[0]]).to(device=device, dtype=dt),
            "lengths_for_llm": _torch.tensor([audio_for_llm_lens]).to(device),
        }
        conv = conv_templates[self.conv_mode].copy()
        # native => audio token only, no text turn
        conv.append_message(conv.roles[0], self.effective_prompt(prompt) + DEFAULT_AUDIO_TOKEN)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt("lang")          # "lang" == audio-only, no real image
        input_ids = tokenizer_image_audio_token(
            prompt, processor["tokenizer"], IMAGE_TOKEN_INDEX,
            return_tensors="pt").unsqueeze(0).to(device)
        return {
            "input_ids": input_ids,
            "images": _torch.zeros((1, 3, 448, 448),
                                   dtype=getattr(self, "_dtype", model.dtype), device=device),
            "audios": audios,
        }


register("vita_15", Vita15Adapter())
