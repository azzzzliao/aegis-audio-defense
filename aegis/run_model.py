#!/usr/bin/env python3
"""Portable baseline generation: run the target model (no defense) on a manifest.

Writes one jsonl per job with the model's response under the adapter's
``response_field`` (e.g. ``qwen2_audio_response``). This produces the
"undefended" responses that the judge then labels, and that feed both the
defense-training data and the hidden-state extraction.

Usage:
    python run_model.py --model qwen2_audio \
        --jobs aj:manifest_aj.jsonl --out-dir out/base
"""
import argparse
import json
from pathlib import Path

import torch

import model_registry as registry
from common import audio_path_of, compact_successes, jl, record_id, resolve_audio


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help=f"registered adapter: {registry.available()}")
    ap.add_argument("--jobs", required=True, help="comma list of name:manifest.jsonl")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--prompt-mode", choices=registry.PROMPT_MODES, default="safety")
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--resume", action="store_true",
                    help="retain successful output rows and retry only unfinished records")
    ap.add_argument("--row-prompt-key", default="",
                    help="manifest field whose text replaces the adapter's safety prompt for "
                         "that row (empty = off, the default). Needed by JALMBench SSJ, whose "
                         "audio only spells the masked keyword: without its carrier sentence "
                         "('spoken_prompt') no request reaches the model. Rows where the field "
                         "is missing or empty fall back to the safety prompt. Only adapters "
                         "whose build_inputs takes a `prompt` kwarg support this.")
    a = ap.parse_args()

    adapter = registry.get(a.model)
    adapter.set_prompt_mode(a.prompt_mode)
    model, proc = adapter.load(a.device)
    rk = adapter.response_field

    # 'auto'/'balanced'/'0,1' shard the model across GPUs (voxtral-24B needs two cards).
    # adapter.load() understands those, but build_inputs feeds `device` straight to
    # tensor.to(), which rejects "auto" -- every row then dies with
    # "Expected one of cpu, cuda, ... at start of device string: auto".
    # Inputs must land where the embedding table lives; run_defense.py:170 does the same.
    sharded = str(a.device) in ("auto", "balanced", "balanced_low_0") or "," in str(a.device)
    in_dev = str(model.get_input_embeddings().weight.device) if sharded else a.device
    if sharded:
        print(f"[INFO] sharded load ({a.device}); feeding inputs to {in_dev}")

    odir = Path(a.out_dir); odir.mkdir(parents=True, exist_ok=True)
    for spec in a.jobs.split(","):
        name, man = spec.split(":", 1)
        rows = jl(man)
        if a.limit:
            rows = rows[:a.limit]
        output_path = odir / f"{name}.jsonl"
        if a.resume:
            kept = compact_successes(output_path, rk, False)
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
        for row in rows:
            rec = dict(row)
            try:
                apath = resolve_audio(audio_path_of(row))
                row_prompt = str(row.get(a.row_prompt_key) or "").strip() if a.row_prompt_key else ""
                inp = (adapter.build_inputs(model, proc, apath, in_dev, prompt=row_prompt)
                       if row_prompt else adapter.build_inputs(model, proc, apath, in_dev))
                with torch.inference_mode():
                    new_ids = adapter.generate(model, inp, a.max_new_tokens)
                rec[rk] = adapter.decode(proc, new_ids)
                rec["eval_error"] = None
            except Exception as e:
                rec[rk] = ""; rec["eval_error"] = f"{type(e).__name__}: {e}"
                torch.cuda.empty_cache()
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n"); fout.flush()
        fout.close(); print("[OK]", odir / f"{name}.jsonl")


if __name__ == "__main__":
    main()
