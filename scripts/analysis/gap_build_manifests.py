#!/usr/bin/env python3
"""Build the extraction manifests for the risk-to-refusal gap analysis.

Input is one model's undefended, Llama-Guard-judged responses, i.e. what
scripts/run_baseline.sh writes to runs/<model>/baseline/<bench>_judged.jsonl. Each
benchmark is capped at --cap rows, keeping ALL unsafe rows and subsampling the safe
ones, because the unsafe class is the scarce one everywhere and it is what limits the
probe. JALMBench is split into its ADiv and SSJ subsets, so the four attack sets stay
comparable in size.

A benign manifest (Benign_train + XSTest) is written too. Those rows carry no
llamaguard_label, and extract_hidden.py labels a row 1 only when that field is
"unsafe", so they all land at label 0 without a generation or judging pass.

  python scripts/analysis/gap_build_manifests.py --model gemma4_e4b_it
"""
import argparse
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
BENCHES = ("audiojailbreak", "jalm_adiv", "jalm_ssj", "sacred")


def jl(path):
    with open(path, encoding="utf-8") as fin:
        return [json.loads(line) for line in fin if line.strip()]


def audio_of(row):
    """Mirror of aegis/common.py:audio_path_of -- the fields extract_hidden.py reads."""
    for key in ("local_audio", "qwen2_audio_input_audio", "audio", "output_wav"):
        if row.get(key):
            return row[key]
    return None


def judged_rows(baseline_dir, bench):
    """Rows of one attack set. jalm_adiv / jalm_ssj are subsets of the JALM file."""
    if bench.startswith("jalm_"):
        subset = bench.split("_", 1)[1].upper()          # ADiv is stored as "ADiv"
        subset = "ADiv" if subset == "ADIV" else subset
        rows = jl(baseline_dir / "jalm_judged.jsonl")
        return [r for r in rows if r.get("subset") == subset]
    return jl(baseline_dir / f"{bench}_judged.jsonl")


def build(baseline_dir, bench, cap, seed):
    rows = judged_rows(baseline_dir, bench)
    missing = [r for r in rows if not audio_of(r)]
    if missing:
        raise SystemExit(f"{bench}: {len(missing)} rows have no audio path "
                         f"(e.g. id={missing[0].get('id')!r})")

    labelled = [r for r in rows if r.get("llamaguard_label") in ("safe", "unsafe")]
    unsafe = [r for r in labelled if r["llamaguard_label"] == "unsafe"]
    safe = [r for r in labelled if r["llamaguard_label"] == "safe"]

    room = max(cap - len(unsafe), 0)
    if len(safe) > room:
        idx = np.random.RandomState(seed).choice(len(safe), room, replace=False)
        safe = [safe[i] for i in sorted(idx)]
    kept = unsafe + safe
    print(f"    {bench:11s} n={len(kept):4d}  unsafe={len(unsafe):4d} safe={len(safe):4d}"
          f"  (from {len(rows)}, dropped {len(rows) - len(labelled)} unlabelled)")
    return kept, len(unsafe)


def build_benign(manifests):
    """The benign negative class: the training benign set plus XSTest."""
    rows = []
    for name, prefix in (("benign_train.jsonl", "benign_train"), ("xstest.jsonl", "xstest")):
        got = jl(manifests / name)
        for r in got:
            r = dict(r)
            r["id"] = f"{prefix}/{r.get('id') or r.get('sample_id')}"   # keep ids unique
            if not audio_of(r):
                raise SystemExit(f"{name}: a row has no audio path")
            rows.append(r)
        print(f"    {prefix:13s} {len(got)} rows")
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--runs", default=str(REPO / "runs"), type=Path)
    ap.add_argument("--manifests", default=str(REPO / "data/manifests"), type=Path)
    ap.add_argument("--out-dir", default=None, help="default: <runs>/<model>/gap/manifests")
    ap.add_argument("--benches", nargs="*", default=list(BENCHES))
    ap.add_argument("--cap", type=int, default=400, help="max rows per benchmark")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--benign-only", action="store_true")
    a = ap.parse_args()

    baseline = a.runs / a.model / "baseline"
    out = Path(a.out_dir) if a.out_dir else a.runs / a.model / "gap" / "manifests"
    out.mkdir(parents=True, exist_ok=True)
    print(f"== {a.model} ==")
    total = total_unsafe = 0
    if not a.benign_only:
        for bench in a.benches:
            kept, n_unsafe = build(baseline, bench, a.cap, a.seed)
            with open(out / f"{bench}.jsonl", "w", encoding="utf-8") as fout:
                for r in kept:
                    fout.write(json.dumps(r, ensure_ascii=False) + "\n")
            total += len(kept)
            total_unsafe += n_unsafe
    benign = build_benign(a.manifests)
    with open(out / "benign.jsonl", "w", encoding="utf-8") as fout:
        for r in benign:
            fout.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"    -> {total} attack rows ({total_unsafe} unsafe) + {len(benign)} benign  [{out}]")


if __name__ == "__main__":
    main()
