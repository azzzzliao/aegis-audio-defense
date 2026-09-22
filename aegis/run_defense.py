#!/usr/bin/env python3
"""AEGIS inference: gated late-layer LoRA with closed-loop scale-to-refusal.

Model-agnostic driver. The gate (router head) fires on the last prompt token at
``gate_layer``; if it fires, the late-layer LoRA adapters are escalated (alpha
increased over ``--alphas``) until the late-layer refusal probability (logit lens
through the final norm + head at the refusal-probe layer) reaches ``--tau``; then
the model generates. No hand-picked scale.

All model-specific code lives in model_registry.ModelAdapter; sparse-layer
handling is in layer_resolve. ``--score-only`` records the soft gate score without
generating (used for threshold calibration).

Usage:
    python run_defense.py --model qwen2_audio --adapter ADAPTER_DIR \
        --jobs aj:manifest_aj.jsonl,sacred:manifest_sacred.jsonl \
        --out-dir out/defended --gate-threshold 0.5 --tau 0.5
"""
import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn

import model_registry as registry
from layer_resolve import resolve_layers
from common import audio_path_of, compact_successes, jl, record_id, resolve_audio


def first_ids(tokenizer, phrases):
    """First-token ids for refusal phrases (bare and space-prefixed)."""
    s = set()
    for x in phrases:
        for variant in (x, " " + x):
            t = tokenizer.encode(variant, add_special_tokens=False)
            if t:
                s.add(t[0])
    return sorted(s)


def build_router(gate_type, D, gate_hidden, device, dtype=torch.bfloat16):
    # dtype must follow the base model: VITA-1.5 runs fp16, and a bf16 gate against
    # fp16 activations fails with "mat1 and mat2 must have the same dtype".
    if gate_type == "mlp":
        return nn.Sequential(nn.Linear(D, gate_hidden), nn.ReLU(),
                             nn.Linear(gate_hidden, 1)).to(device, dtype)
    return nn.Linear(D, 1).to(device, dtype)


def gate_input_features(hidden_state, input_norm):
    """Optionally standardise the gate's input hidden state.

    ``input_norm`` mirrors the adapter config's ``gate_input_norm`` flag and
    MUST match what the gate was trained with, exactly like the neutral prompt
    does -- the gate reads a raw decoder hidden state whose per-dimension scale
    is arbitrary, so normalising at inference but not at training (or the
    reverse) silently destroys the decision boundary.

    Parameter-free by construction (``F.layer_norm`` with no affine weights),
    so ``router_head.pt`` stays byte-compatible with adapters trained before
    this flag existed and the router module keeps its exact layer indices.
    """
    if not input_norm:
        return hidden_state
    return torch.nn.functional.layer_norm(
        hidden_state.float(), hidden_state.shape[-1:]).to(hidden_state.dtype)


def hard_gate_decision(scores, threshold):
    """Compare quantized router scores without quantizing the Python threshold."""
    return (scores.to(torch.float64) >= float(threshold)).to(scores.dtype)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help=f"registered adapter: {registry.available()}")
    ap.add_argument("--adapter", required=True, help="dir with router_head.pt, late_adapters.pt, config.json")
    ap.add_argument("--jobs", required=True, help="comma list of name:manifest.jsonl")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--prompt-mode", choices=registry.PROMPT_MODES, default="safety")
    ap.add_argument("--gate-threshold", type=float, default=None,
                    help=">=0 hard gate; <0 soft gate (use raw sigmoid as strength). "
                         "Required unless --unconditional.")
    ap.add_argument("--unconditional", action="store_true",
                    help="ablation (paper's 'Full' / 'Always-on'): no gate, no closed loop. "
                         "Force g=1, alpha=1, single forward + generate (plain LoRA SFT "
                         "inference). --gate-threshold/--tau/--alphas are ignored.")
    ap.add_argument("--batch-size", type=int, default=1,
                    help="generate this many samples per forward. ONLY valid with --unconditional "
                         "(the gated path needs per-sample gate/escalation). 1 = unchanged. "
                         "Inputs are left-padded; a failing chunk falls back to per-sample.")
    ap.add_argument("--tau", type=float, default=0.5, help="target late refusal prob")
    ap.add_argument("--alphas", default="1.0,1.5,2.0,2.5,3.0")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--resume", action="store_true",
                    help="retain successful output rows and retry only unfinished records")
    ap.add_argument("--score-only", action="store_true",
                    help="prefill only: record soft gate score, no intervention / no generation")
    ap.add_argument("--baseline", default="",
                    help="name:file[,name2:file2] of prior run_defense outputs; when the gate "
                         "does NOT fire (no intervention) the response is reused from here instead "
                         "of regenerated. Safe with temperature 0 (deterministic). The baseline "
                         "must be a SAFETY-PROMPT no-LoRA run (i.e. run_defense with a never-fire "
                         "threshold), not a neutral-prompt baseline.")
    ap.add_argument("--require-never-fire-route", action="store_true",
                    help="resume contract for a never-fire cache: retain only generated rows "
                         "with gate=0, chosen_alpha=0, and from_baseline=false")
    a = ap.parse_args()
    if a.gate_threshold is None and not a.unconditional:
        ap.error("--gate-threshold is required unless --unconditional is set")
    if a.batch_size > 1 and not a.unconditional:
        ap.error("--batch-size > 1 is only supported with --unconditional")

    ALPHAS = [float(x) for x in a.alphas.split(",")]
    with (Path(a.adapter) / "config.json").open(encoding="utf-8") as handle:
        cfg = json.load(handle)
    cfg_prompt_mode = cfg.get("prompt_mode", "safety")
    if cfg_prompt_mode not in registry.PROMPT_MODES:
        ap.error(f"adapter config prompt_mode must be one of {registry.PROMPT_MODES}, got {cfg_prompt_mode!r}")
    if cfg_prompt_mode != a.prompt_mode:
        ap.error(
            f"--prompt-mode {a.prompt_mode!r} does not match adapter config prompt_mode "
            f"{cfg_prompt_mode!r}")

    adapter = registry.get(a.model)
    adapter.set_prompt_mode(a.prompt_mode)
    D = cfg["hidden"]; r = cfg["rank"]; GH = cfg.get("gate_hidden", 256)
    gate_type = cfg.get("gate_type", "linear")
    GATE_INPUT_NORM = bool(cfg.get("gate_input_norm", False))

    model, proc = adapter.load(a.device)
    tok = adapter.tokenizer(proc)
    layers = adapter.decoder_layers(model)
    norm = adapter.final_norm(model)
    head = adapter.lm_head(model)
    GL, late, probe = resolve_layers(cfg, len(layers), adapter.refusal_probe_layer(model))
    print(f"[INFO] model={a.model} layers={len(layers)} gate@L{GL} late={late[0]}-{late[-1]} "
          f"probe@L{probe} gate_type={gate_type} gate_input_norm={GATE_INPUT_NORM}")

    refus = first_ids(tok, adapter.refusal_phrases)
    # "auto"/"0,1" shards the base model, but the gate and the LoRA pairs are plain
    # modules that need a concrete device. Put each on the device of the layer it
    # actually hooks, the same placement train_gate_lora.py uses, so a sharded
    # model works here too (voxtral-24B only fits across two cards).
    def _dev_of(layer):
        try:
            return next(layer.parameters()).device
        except StopIteration:
            return torch.device(a.device if a.device not in
                                ("auto", "balanced", "balanced_low_0") else "cuda:0")

    sharded = a.device in ("auto", "balanced", "balanced_low_0") or "," in str(a.device)
    gate_dev = _dev_of(layers[GL]) if sharded else a.device
    late_devs = {l: (_dev_of(layers[l]) if sharded else a.device) for l in late}
    load_dev = gate_dev if sharded else a.device
    param_dtype = next(layers[GL].parameters()).dtype
    # Inputs must land where the model actually consumes them: the embedding device
    # for a sharded model, otherwise the requested device.
    in_dev = (str(model.get_input_embeddings().weight.device) if sharded else a.device)

    router = build_router(gate_type, D, GH, gate_dev, param_dtype)
    if not a.unconditional:
        router.load_state_dict(torch.load(Path(a.adapter) / "router_head.pt", map_location=load_dev))
    router.eval()
    A = nn.ModuleDict({str(l): nn.Linear(D, r, bias=False).to(late_devs[l], param_dtype)
                       for l in late})
    B = nn.ModuleDict({str(l): nn.Linear(r, D, bias=False).to(late_devs[l], param_dtype)
                       for l in late})
    sd = torch.load(Path(a.adapter) / "late_adapters.pt", map_location=load_dev)
    A.load_state_dict(sd["A"]); B.load_state_dict(sd["B"])

    STATE = {"prompt_len": None, "g": None,
             "threshold": a.gate_threshold if a.gate_threshold is not None else -1.0,
             "scale": 1.0, "hp": None}

    def gate_hook(m, i, o):
        h = o[0] if isinstance(o, tuple) else o
        if h.shape[1] < STATE["prompt_len"]:
            return o
        if a.unconditional:
            STATE["g"] = torch.ones(h.shape[0], device=h.device, dtype=h.dtype)
            return o
        s = torch.sigmoid(router(gate_input_features(
            h[:, STATE["prompt_len"] - 1, :], GATE_INPUT_NORM))).squeeze(-1)
        STATE["g"] = hard_gate_decision(s, STATE["threshold"]) \
            if STATE["threshold"] >= 0 else s
        return o

    def make_late(l):
        def hook(m, i, o):
            if STATE["g"] is None:
                return o
            h = o[0] if isinstance(o, tuple) else o
            h = h + STATE["scale"] * STATE["g"].view(-1, 1, 1).to(h.dtype) * B[str(l)](A[str(l)](h))
            return (h,) + tuple(o[1:]) if isinstance(o, tuple) else h
        return hook

    def probe_hook(m, i, o):
        h = o[0] if isinstance(o, tuple) else o
        if h.shape[1] >= STATE["prompt_len"]:
            STATE["hp"] = h[:, STATE["prompt_len"] - 1, :]

    layers[GL].register_forward_hook(gate_hook)
    for l in late:
        layers[l].register_forward_hook(make_late(l))
    layers[probe].register_forward_hook(probe_hook)

    def refuse_prob():
        p = torch.softmax(head(norm(STATE["hp"][0])).float(), -1)
        return float(p[refus].sum())

    odir = Path(a.out_dir); odir.mkdir(parents=True, exist_ok=True)
    rk = adapter.response_field

    def _row_id(r):
        return str(r.get("id") or r.get("sample_id") or "")

    base_maps = {}  # job-name -> {id: baseline_response} for non-fired reuse
    for spec in (a.baseline.split(",") if a.baseline else []):
        bname, bfile = spec.split(":", 1)
        base_maps[bname] = {_row_id(r): r.get(rk) for r in jl(bfile)
                            if _row_id(r) and r.get(rk)}
        print(f"[baseline] {bname}: {len(base_maps[bname])} cached responses from {bfile}")

    def _gen_single_uncond(row):
        """Per-sample unconditional generate (g=1, alpha=1). Used as the batch fallback."""
        rec = dict(row)
        try:
            apath = resolve_audio(audio_path_of(row))
            inp = adapter.build_inputs(model, proc, apath, in_dev)
            STATE["prompt_len"] = adapter.prompt_len(inp); STATE["g"] = None; STATE["scale"] = 1.0
            with torch.inference_mode():
                model(**inp)                                   # prime g=1 before generate
                new_ids = adapter.generate(model, inp, a.max_new_tokens)
            rec[rk] = adapter.decode(proc, new_ids)
            rec["gate"] = 1.0; rec["chosen_alpha"] = 1.0
            rec["from_baseline"] = False; rec["eval_error"] = None
        except Exception as e:                                 # noqa: BLE001
            rec[rk] = ""; rec["eval_error"] = f"{type(e).__name__}: {e}"
            torch.cuda.empty_cache()
        return rec

    def _gen_batch_uncond(chunk):
        """Unconditional batched generate (left-padded). Raises on failure -> caller falls back."""
        apaths = [resolve_audio(audio_path_of(r)) for r in chunk]
        inp = adapter.build_inputs_batch(model, proc, apaths, in_dev)
        STATE["prompt_len"] = adapter.prompt_len(inp); STATE["g"] = None; STATE["scale"] = 1.0
        with torch.inference_mode():
            model(**inp)                                       # prime g=ones(B) for full-layer hooks
            new_ids = adapter.generate(model, inp, a.max_new_tokens)
        texts = adapter.decode_batch(proc, new_ids)
        if len(texts) != len(chunk):
            raise RuntimeError(f"decode_batch returned {len(texts)} != {len(chunk)} rows")
        recs = []
        for r, txt in zip(chunk, texts):
            rec = dict(r)
            rec[rk] = txt; rec["gate"] = 1.0; rec["chosen_alpha"] = 1.0
            rec["from_baseline"] = False; rec["eval_error"] = None
            recs.append(rec)
        return recs

    for spec in a.jobs.split(","):
        name, man = spec.split(":", 1)
        base = base_maps.get(name, {})
        rows = jl(man)
        if a.limit:
            rows = rows[:a.limit]
        output_path = odir / f"{name}.jsonl"
        if a.resume:
            kept = compact_successes(
                output_path, rk, a.score_only,
                require_route=not a.score_only,
                require_generated=not a.score_only and name not in base_maps,
                require_never_fire=a.require_never_fire_route,
            )
            pending_rows = []
            pending_ids = set()
            for row in rows:
                rid = record_id(row)
                if rid not in kept and rid not in pending_ids:
                    pending_rows.append(row)
                    pending_ids.add(rid)
            rows = pending_rows
            print(f"[resume] {name}: kept={len(kept)} pending={len(rows)}")
        fout = open(output_path, "a" if a.resume else "w", encoding="utf-8")
        if a.unconditional and a.batch_size > 1:               # B1/B2 fast path
            for i in range(0, len(rows), a.batch_size):
                chunk = rows[i:i + a.batch_size]
                try:
                    recs = _gen_batch_uncond(chunk)
                except Exception as e:                          # noqa: BLE001
                    print(f"[batch fallback] {name} rows {i}-{i+len(chunk)-1}: "
                          f"{type(e).__name__}: {e}")
                    torch.cuda.empty_cache()
                    recs = [_gen_single_uncond(r) for r in chunk]
                for rec in recs:
                    fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fout.flush()
            fout.close(); print("[OK]", odir / f"{name}.jsonl"); continue
        for row in rows:
            rec = dict(row)
            try:
                apath = resolve_audio(audio_path_of(row))
                inp = adapter.build_inputs(model, proc, apath, in_dev)
                STATE["prompt_len"] = adapter.prompt_len(inp); STATE["g"] = None; STATE["scale"] = 1.0
                chosen = 0.0
                if a.score_only:
                    STATE["threshold"] = -1.0; STATE["scale"] = 0.0
                    with torch.inference_mode():
                        model(**inp)
                    rec["gate"] = float(STATE["g"].item()) if STATE["g"] is not None else None
                    rec["eval_error"] = None
                    fout.write(json.dumps(rec, ensure_ascii=False) + "\n"); fout.flush()
                    continue
                with torch.inference_mode():
                    model(**inp)                                  # prefill -> gate decides
                    fired = STATE["g"] is not None and float(STATE["g"].item()) > 0
                    if not fired and base.get(_row_id(row)):
                        # no intervention -> identical to the safety-prompt baseline (temp 0).
                        rec[rk] = base[_row_id(row)]
                        rec["gate"] = 0.0; rec["chosen_alpha"] = 0.0
                        rec["from_baseline"] = True; rec["eval_error"] = None
                        fout.write(json.dumps(rec, ensure_ascii=False) + "\n"); fout.flush()
                        continue
                    if a.unconditional:
                        STATE["scale"] = 1.0                      # B1: plain LoRA, no escalation
                        chosen = 1.0
                    elif fired:
                        for al in ALPHAS:                         # escalate until refusal
                            STATE["scale"] = al; model(**inp)
                            if refuse_prob() >= a.tau:
                                break
                        chosen = STATE["scale"]
                    else:
                        STATE["scale"] = 1.0
                    new_ids = adapter.generate(model, inp, a.max_new_tokens)
                rec[rk] = adapter.decode(proc, new_ids)
                rec["gate"] = float(STATE["g"].item()) if STATE["g"] is not None else None
                rec["chosen_alpha"] = chosen; rec["from_baseline"] = False; rec["eval_error"] = None
            except Exception as e:
                rec[rk] = ""; rec["eval_error"] = f"{type(e).__name__}: {e}"
                torch.cuda.empty_cache()
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n"); fout.flush()
        fout.close(); print("[OK]", odir / f"{name}.jsonl")


if __name__ == "__main__":
    main()
