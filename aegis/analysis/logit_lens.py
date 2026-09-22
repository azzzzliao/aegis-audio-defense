#!/usr/bin/env python3
"""Per-layer logit lens (model-specific, via the registry adapter).

For a balanced sample of safe/unsafe examples, run the model once and at EVERY
layer project the last-token hidden state through the final norm + LM head to get
a vocab distribution. Records, per layer, the probability mass on refusal /
transcription / answer token groups -> the shallow->mid->late "knowing vs doing"
story, and where unsafe diverges from safe.

The model is loaded through ModelAdapter (final_norm / lm_head / build_inputs),
so it works for any registered model.

Outputs: logit_lens.npz (refusal/transcription/answer [N,L], labels),
         logit_lens_top_tokens.jsonl, logit_lens.png (mean curves safe vs unsafe).

Usage:
  python logit_lens.py --model qwen2_audio --responses judged.jsonl \
      --out-dir OUT --n-per-class 40
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

# import the registry + helpers from the parent aegis/ dir
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import model_registry as registry          # noqa: E402
from common import jl, resolve_audio, audio_path_of  # noqa: E402

TRANSCRIPT = ["The request is", "The user request is", "The audio says", "The spoken request is", "It says"]
ANSWER = ["Here is", "Sure", "Certainly", "First", "To", "Yes", "The following"]


def first_ids(tok, phrases):
    s = set()
    for ph in phrases:
        t = tok.encode(ph, add_special_tokens=False)
        if t:
            s.add(t[0])
    return sorted(s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help=f"registered adapter: {registry.available()}")
    ap.add_argument("--responses", required=True, help="judged jsonl with llamaguard_label + audio path")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--n-per-class", type=int, default=40)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)

    rows = [r for r in jl(a.responses) if r.get("llamaguard_label") in ("safe", "unsafe")]
    for r in rows:
        r["label"] = 1 if r["llamaguard_label"] == "unsafe" else 0
    rng = np.random.RandomState(a.seed)
    uns = [r for r in rows if r["label"] == 1]; saf = [r for r in rows if r["label"] == 0]
    rng.shuffle(uns); rng.shuffle(saf)
    sample = uns[:a.n_per_class] + saf[:a.n_per_class]
    print(f"sample: {len(uns[:a.n_per_class])} unsafe + {len(saf[:a.n_per_class])} safe")

    adapter = registry.get(a.model)
    model, proc = adapter.load(a.device)
    tok = adapter.tokenizer(proc)
    norm = adapter.final_norm(model); head = adapter.lm_head(model)
    groups = {"refusal": first_ids(tok, adapter.refusal_phrases),
              "transcription": first_ids(tok, TRANSCRIPT), "answer": first_ids(tok, ANSWER)}
    gidx = {k: torch.tensor(v, device=a.device) for k, v in groups.items()}

    per_layer = {"refusal": [], "transcription": [], "answer": []}
    labels, top_rows = [], []
    for r in tqdm(sample):
        apath = resolve_audio(audio_path_of(r))
        try:
            inp = adapter.build_inputs(model, proc, apath, a.device)
            with torch.inference_mode():
                o = model(**inp, output_hidden_states=True, return_dict=True, use_cache=False)
            hs = o.hidden_states                       # tuple (L+1) of [1, seq, D]
            rr, tt, aa, toks = [], [], [], []
            for l in range(1, len(hs)):
                h = hs[l][0, -1, :]                     # last token, layer l
                probs = torch.softmax(head(norm(h)).float(), -1)
                rr.append(float(probs[gidx["refusal"]].sum()))
                tt.append(float(probs[gidx["transcription"]].sum()))
                aa.append(float(probs[gidx["answer"]].sum()))
                toks.append([tok.decode([t]) for t in torch.topk(probs, 3).indices.tolist()])
            per_layer["refusal"].append(rr); per_layer["transcription"].append(tt); per_layer["answer"].append(aa)
            labels.append(r["label"])
            top_rows.append({"id": r.get("id"), "label": r["label"], "per_layer_top": toks})
        except Exception as e:
            print("skip", apath, e)

    refusal = np.array(per_layer["refusal"]); transcription = np.array(per_layer["transcription"])
    answer = np.array(per_layer["answer"]); labels = np.array(labels)
    np.savez(out / "logit_lens.npz", refusal=refusal, transcription=transcription,
             answer=answer, labels=labels)
    (out / "logit_lens_top_tokens.jsonl").write_text(
        "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in top_rows))

    # plot mean curves: refusal/answer prob vs layer, safe vs unsafe
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        Ln = refusal.shape[1]; xs = range(1, Ln + 1)
        fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
        for arr, axx, ttl in [(refusal, ax[0], "refusal prob mass"), (answer, ax[1], "answer prob mass")]:
            for lbl, c in [(1, "#c62828"), (0, "#2e7d32")]:
                m = labels == lbl
                if m.any():
                    axx.plot(xs, arr[m].mean(0), marker="o", ms=3, c=c,
                             label="unsafe" if lbl else "safe")
            axx.set_xlabel("layer"); axx.set_ylabel(ttl); axx.set_title(ttl); axx.legend()
        fig.suptitle(f"Logit lens ({a.model}): safe vs unsafe")
        fig.tight_layout(); fig.savefig(out / "logit_lens.png", dpi=140); plt.close(fig)
    except Exception as e:
        print("[warn] plot skipped:", e)

    print(f"[OK] wrote {out}/logit_lens.npz ({len(labels)} examples, "
          f"{refusal.shape[1] if len(refusal) else 0} layers)")


if __name__ == "__main__":
    main()
