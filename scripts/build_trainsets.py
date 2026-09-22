#!/usr/bin/env python3
"""Build the AEGIS training files for one model, for both evaluation protocols.

  lobo      Leave one benchmark out: for each held-out benchmark, the harmful audio
            comes from the OTHER two benchmarks; the held-out benchmark is evaluated
            in full.
  indomain  For each benchmark, the harmful audio comes from its train split
            (data/splits/<bench>_indomain.json); evaluation uses the test split.

In both, benign audio is Benign_train with the model's own undefended answers as
targets (jbb_benign rows the model refused at baseline are dropped), harmful targets
are generic refusal templates, and the number of harmful clips equals the number of
benign clips. XSTest is never used for training.

Rows keep the request text as ``prompt``. The Gemma, Voxtral, Ultravox and VITA
adapters ignore it and train under their fixed prompt; the Qwen2-Audio and Phi-4
adapters use it as the text turn, which is how the paper's adapters were trained.

  python scripts/build_trainsets.py --manifests data/manifests --splits data/splits \\
      --benign-responses runs/gemma4_e4b_it/baseline/benign_train.jsonl \\
      --response-key gemma4_response --out-dir runs/gemma4_e4b_it/train

writes train_lobo_<held>.jsonl and train_indomain_<bench>.jsonl (+ *_smoke.jsonl).
"""
import argparse
import json
import random
import re
from pathlib import Path

REFUSALS = [
    "I'm sorry, but I can't help with that.",
    "I cannot assist with this request as it could cause harm.",
    "I'm unable to help with that. If you have a safe, legitimate request I'm happy to help.",
]
BENCHES = ["audiojailbreak", "jalm", "sacred"]
MANIFEST = {"audiojailbreak": "audiojailbreak.jsonl", "jalm": "jalm.jsonl",
            "sacred": "sacred_msd.jsonl"}
KEYS = ("audio", "prompt", "target", "kind")

# Conservative decline check on the start of a jbb_benign answer.
_DECLINE = re.compile(
    r"\b(i can'?t|i cannot|i am unable|i'?m unable|i will not|i won'?t|i am not able|"
    r"i'?m not able|i must decline|i'?m sorry,? but|as an ai\b.*\bcannot|i'?m not (able|going) to)\b",
    re.I)


def jl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def harmful_row(o):
    return {"audio": o["local_audio"], "prompt": o.get("prompt") or "", "kind": "harmful"}


def benign_pool(path, response_key):
    rows, dropped = [], 0
    for o in jl(path):
        resp = (o.get(response_key) or o.get("model_response") or "").strip()
        audio = o.get("audio_path") or o.get("local_audio")
        ok = not o.get("eval_error") and o.get("status", "success") == "success"
        if not (ok and resp and audio):
            continue
        if o.get("dataset") == "jbb_benign" and _DECLINE.search(resp[:200]):
            dropped += 1
            continue
        rows.append({"audio": audio, "prompt": o.get("original_prompt") or o.get("prompt") or "",
                     "target": resp, "kind": "benign"})
    return rows, dropped


def select_harmful(pools, train_benches, total, rng):
    """Select exactly ``total`` harmful rows, split evenly across the training benchmarks."""
    base, remainder = divmod(total, len(train_benches))
    rows = []
    for index, bench in enumerate(train_benches):
        count = base + (index < remainder)
        pool = pools[bench][:]
        if len(pool) < count:
            raise ValueError(f"not enough harmful rows for {bench}: need {count}, have {len(pool)}")
        rng.shuffle(pool)
        rows.extend(pool[:count])
    return rows


def lobo(manifests, benign, seed):
    """{held-out bench: rows}. One RNG stream shared across folds, in BENCHES order."""
    rng = random.Random(seed)
    pools = {b: [harmful_row(o) for o in manifests[b]] for b in BENCHES}
    folds = {}
    for held in BENCHES:
        train_benches = [b for b in BENCHES if b != held]
        harmful = select_harmful(pools, train_benches, len(benign), rng)
        harmful = [{**h, "target": rng.choice(REFUSALS)} for h in harmful]
        rows = harmful + benign
        rng.shuffle(rows)
        folds[held] = (rows, harmful)
    return folds


def indomain(manifest, train_ids, benign, seed):
    """Rows for one benchmark's train split. A fresh RNG per benchmark."""
    rng = random.Random(seed)
    pool = [harmful_row(o) for o in manifest if str(o["id"]) in train_ids]
    if len(pool) < len(benign):
        raise ValueError(f"train split has {len(pool)} harmful rows < {len(benign)} benign")
    rng.shuffle(pool)
    harmful = [{**h, "target": rng.choice(REFUSALS)} for h in pool[:len(benign)]]
    rows = harmful + benign
    rng.shuffle(rows)
    return rows, harmful


def write(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps({k: r[k] for k in KEYS}, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifests", required=True, help="dir written by scripts/prepare_data.py")
    ap.add_argument("--splits", default="data/splits", help="in-domain train/test id lists")
    ap.add_argument("--benign-responses", required=True,
                    help="undefended Benign_train responses (aegis/run_model.py output)")
    ap.add_argument("--response-key", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--protocol", choices=["lobo", "indomain", "both"], default="both")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    benign, dropped = benign_pool(a.benign_responses, a.response_key)
    print(f"benign={len(benign)} (dropped {dropped} refused jbb_benign rows)")
    manifests = {b: jl(Path(a.manifests) / MANIFEST[b]) for b in BENCHES}
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if a.protocol in ("lobo", "both"):
        for held, (rows, harmful) in lobo(manifests, benign, a.seed).items():
            write(out / f"train_lobo_{held}.jsonl", rows)
            write(out / f"train_lobo_{held}_smoke.jsonl", harmful[:2] + benign[:2])
            print(f"[lobo, held-out {held}] {len(rows)} rows = {len(harmful)} harmful + {len(benign)} benign")
    if a.protocol in ("indomain", "both"):
        for bench in BENCHES:
            split = json.loads((Path(a.splits) / f"{bench}_indomain.json").read_text())
            rows, harmful = indomain(manifests[bench], set(split["train_ids"]), benign, a.seed)
            write(out / f"train_indomain_{bench}.jsonl", rows)
            write(out / f"train_indomain_{bench}_smoke.jsonl", harmful[:2] + benign[:2])
            print(f"[in-domain {bench}] {len(rows)} rows = {len(harmful)} harmful "
                  f"(of {len(split['train_ids'])} train-split rows) + {len(benign)} benign")


if __name__ == "__main__":
    main()
