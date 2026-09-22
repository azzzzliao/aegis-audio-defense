#!/usr/bin/env python3
"""Hidden-state extraction -> NPZ feature contract (for layer-wise probing).

Emits the contract the router/probe training consumes:
    hidden_last, hidden_mean_context, hidden_audio_tokens : float16 [N, L, D]
    labels : int64 [N]   (1 if llamaguard_label == "unsafe" else 0)
    ids    : object [N]
    scores : float32 [N, 3]   (refusal / transcription / answer prefix mass)
    pool_names, score_names : object
    layer_indices : int64 [K]  (only if --layers gives a sparse subset)

The model is loaded + queried via the registry adapter, and audio-token positions
come from the adapter (override per model if the audio placeholder differs).

Sparse layers: pass ``--layers 0,8,16,24,32`` to keep only those absolute model
layers. The list is stored as ``layer_indices`` so downstream code reports real
layer ids and the trained gate can be mapped back to absolute layers at deploy
time (see layer_resolve.py).

Usage:
    python extract_hidden.py --model qwen2_audio \
        --responses judged.jsonl --out-npz feats.npz --out-meta meta.jsonl
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

import model_registry as registry
from common import jl, row_key, resolve_audio, audio_path_of

REFUSAL_PREFIXES = ["I'm sorry", "I cannot", "I can't", "I'm unable", "I will not", "No"]
TRANSCRIPTION_PREFIXES = ["The request is", "The user request is", "The audio says",
                          "The spoken request is", "It says"]
ANSWER_PREFIXES = ["Here is", "Sure", "Certainly", "First", "To"]


def first_token_ids(tokenizer, phrases):
    ids = []
    for ph in phrases:
        t = tokenizer.encode(ph, add_special_tokens=False)
        if t:
            ids.append(t[0])
    return sorted(set(ids))


def parse_layers(value):
    if not value:
        return None
    return [int(p.strip()) for p in value.split(",") if p.strip()]


def pool_hidden(hidden_states, input_ids, attention_mask, audio_pos, layer_indices):
    if layer_indices is not None:
        hidden_states = [hidden_states[i] for i in layer_indices]
    mask = attention_mask.bool()
    valid = torch.where(mask)[0]
    last_pos = int(valid[-1].item())
    if not audio_pos:
        audio_pos = valid.detach().cpu().tolist()
    audio_idx = torch.tensor(audio_pos, device=hidden_states[0].device, dtype=torch.long)
    pooled = {"last": [], "mean_context": [], "audio_tokens": []}
    for layer_h in hidden_states:
        h = layer_h[0]
        pooled["last"].append(h[last_pos].detach().float().cpu().numpy())
        pooled["mean_context"].append(h[mask].mean(dim=0).detach().float().cpu().numpy())
        pooled["audio_tokens"].append(h.index_select(0, audio_idx).mean(dim=0).detach().float().cpu().numpy())
    return {k: np.stack(v).astype(np.float16) for k, v in pooled.items()}


def group_scores(logits, token_groups):
    probs = torch.softmax(logits.detach().float(), dim=-1)
    return [float(probs[ids].sum().cpu()) if ids else 0.0 for _, ids in token_groups]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help=f"registered adapter: {registry.available()}")
    ap.add_argument("--responses", required=True, help="judged jsonl with llamaguard_label + audio path")
    ap.add_argument("--out-npz", required=True)
    ap.add_argument("--out-meta", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--layers", default="", help="comma-separated absolute layer ids; empty = all")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    adapter = registry.get(a.model)
    rows = jl(a.responses)
    if a.limit:
        rows = rows[:a.limit]
    layer_indices = parse_layers(a.layers)

    model, proc = adapter.load(a.device)
    tok = adapter.tokenizer(proc)
    rk = adapter.response_field
    token_groups = [("refusal", first_token_ids(tok, REFUSAL_PREFIXES)),
                    ("transcription", first_token_ids(tok, TRANSCRIPTION_PREFIXES)),
                    ("answer", first_token_ids(tok, ANSWER_PREFIXES))]
    pool_names = ["last", "mean_context", "audio_tokens"]
    hidden_by_pool = {n: [] for n in pool_names}
    score_rows, meta_rows = [], []

    for idx, row in enumerate(tqdm(rows)):
        ex_id = str(row_key(row) or idx)
        apath = resolve_audio(audio_path_of(row))
        meta = {"row_index": idx, "id": ex_id, "category": row.get("category"),
                "source": row.get("source"), "attack_type": row.get("attack_type"),
                "local_audio": apath, rk: row.get(rk, ""),
                "llamaguard_label": row.get("llamaguard_label", "missing"), "extract_error": None}
        try:
            if not apath:
                raise ValueError("missing audio path")
            inp = adapter.build_inputs(model, proc, apath, a.device)
            with torch.inference_mode():
                out = model(**inp, output_hidden_states=True, return_dict=True, use_cache=False)
            audio_pos = adapter.audio_token_positions(tok, inp["input_ids"][0])
            pooled = pool_hidden(out.hidden_states, inp["input_ids"][0],
                                 inp["attention_mask"][0], audio_pos, layer_indices)
            for n in pool_names:
                hidden_by_pool[n].append(pooled[n])
            score_rows.append(group_scores(out.logits[0, -1, :], token_groups))
            meta["seq_len"] = int(inp["input_ids"].shape[1])
        except Exception as exc:
            meta["extract_error"] = f"{type(exc).__name__}: {exc}"
            for n in pool_names:
                if hidden_by_pool[n]:
                    hidden_by_pool[n].append(np.zeros_like(hidden_by_pool[n][-1]))
                else:
                    hidden_by_pool[n].append(np.zeros((1, 1), dtype=np.float16))
            score_rows.append([0.0, 0.0, 0.0])
        meta_rows.append(meta)

    ok = sum(1 for r in meta_rows if not r.get("extract_error"))
    if ok == 0:
        first = next((r["extract_error"] for r in meta_rows if r.get("extract_error")), "unknown")
        raise RuntimeError(f"all extraction rows failed; first error: {first}")
    for n, arrays in hidden_by_pool.items():
        shapes = {arr.shape for arr in arrays}
        if len(shapes) != 1:
            raise RuntimeError(f"inconsistent hidden shapes for {n}: {sorted(shapes)}")

    arrays = {f"hidden_{n}": np.stack(hidden_by_pool[n]).astype(np.float16) for n in pool_names}
    arrays["scores"] = np.asarray(score_rows, dtype=np.float32)
    arrays["labels"] = np.asarray([1 if r.get("llamaguard_label") == "unsafe" else 0 for r in meta_rows],
                                  dtype=np.int64)
    arrays["ids"] = np.asarray([r["id"] for r in meta_rows], dtype=object)
    arrays["score_names"] = np.asarray([n for n, _ in token_groups], dtype=object)
    arrays["pool_names"] = np.asarray(pool_names, dtype=object)
    if layer_indices is not None:
        arrays["layer_indices"] = np.asarray(layer_indices, dtype=np.int64)

    Path(a.out_npz).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out_meta).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.out_npz, **arrays)
    with open(a.out_meta, "w", encoding="utf-8") as f:
        for r in meta_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[OK] {a.out_npz}  valid={ok} errors={len(meta_rows)-ok}")


if __name__ == "__main__":
    main()
