#!/usr/bin/env python3
"""Summarize one model's run: the paper's metrics for each benchmark.

Reads the layout written by scripts/run_baseline.sh and scripts/run_aegis.sh
(or scripts/run_ablation.sh, via --variant-dir):

  <run-dir>/baseline/<bench>_judged.jsonl, xstest.jsonl      undefended
  <variant-dir>/<bench>/defended/<bench>_judged.jsonl        defended attacks
  <variant-dir>/<bench>/defended/xstest.jsonl                defended XSTest
  <variant-dir>/<bench>/scores/{<bench>,xstest}.jsonl        gate scores (AEGIS only)

Metrics:
  unsafe rate (%)   Llama-Guard "unsafe" / all evaluated rows (in-domain: the test split,
                    LOBO: the whole benchmark); the undefended rate is given on the same
                    rows and on the whole benchmark
  over-refusal (%)  refusal regex (aegis/refusal.py) on XSTest, over all 250 prompts and
                    over the test half; added = defended - undefended
  gate AUROC        attack scores (positives) vs XSTest scores (negatives)

  python scripts/summarize.py --run-dir runs/gemma4_e4b_it --protocol indomain \
      --response-key gemma4_response
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "aegis"))
from common import jl, record_id  # noqa: E402
from refusal import is_refusal  # noqa: E402

HELDS = ["audiojailbreak", "jalm", "sacred"]   # table order in the paper


def unsafe_rate(rows):
    return 100.0 * sum(r.get("llamaguard_label") == "unsafe" for r in rows) / len(rows)


def refusal_rate(path, key, ids=None):
    rows = [r for r in jl(path) if ids is None or record_id(r) in ids]
    return 100.0 * sum(is_refusal(r.get(key)) for r in rows) / len(rows)


def auroc(pos, neg):
    """Mann-Whitney AUROC with average ranks for ties."""
    values = sorted([(v, 1) for v in pos] + [(v, 0) for v in neg])
    rank_sum, i = 0.0, 0
    while i < len(values):
        j = i
        while j < len(values) and values[j][0] == values[i][0]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        rank_sum += avg_rank * sum(label for _, label in values[i:j])
        i = j
    n_pos, n_neg = len(pos), len(neg)
    return (rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def gate_scores(path):
    return [float(r["gate"]) for r in jl(path) if r.get("gate") is not None]


def summarize(run_dir, variant_dir, key):
    base = run_dir / "baseline"
    xs_ids = sorted(record_id(r) for r in jl(base / "xstest.jsonl"))
    test_half = set(xs_ids[1::2])
    or_before = refusal_rate(base / "xstest.jsonl", key)
    or_before_test = refusal_rate(base / "xstest.jsonl", key, test_half)
    out = {}
    for held in HELDS:
        d = variant_dir / held
        judged = d / "defended" / f"{held}_judged.jsonl"
        if not judged.exists():
            continue
        after_rows = jl(judged)
        evaluated = {record_id(r) for r in after_rows}
        base_rows = jl(base / f"{held}_judged.jsonl")
        same_rows = [r for r in base_rows if record_id(r) in evaluated]
        row = {"n": len(after_rows), "unsafe_before": unsafe_rate(same_rows),
               "unsafe_before_full": unsafe_rate(base_rows), "unsafe_after": unsafe_rate(after_rows),
               "over_refusal_before": or_before, "over_refusal_before_test_half": or_before_test}
        if len(same_rows) != len(after_rows):
            row["warning"] = f"{len(after_rows) - len(same_rows)} defended rows have no baseline"
        xs = d / "defended" / "xstest.jsonl"
        if xs.exists():
            row["over_refusal_after"] = refusal_rate(xs, key)
            row["over_refusal_after_test_half"] = refusal_rate(xs, key, test_half)
        pos, neg = d / "scores" / f"{held}.jsonl", d / "scores" / "xstest.jsonl"
        if pos.exists() and neg.exists():
            row["gate_auroc"] = auroc(gate_scores(pos), gate_scores(neg))
        thr = d / "threshold.json"
        if thr.exists():
            row["threshold"] = json.loads(thr.read_text())["threshold"]
        out[held] = row
    return out


def fmt(v, spec=".1f"):
    return "--" if v is None else format(v, spec)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True, type=Path, help="runs/<model>")
    ap.add_argument("--protocol", choices=["indomain", "lobo"],
                    help="read <run-dir>/<protocol>/<bench>/...")
    ap.add_argument("--variant-dir", type=Path,
                    help="dir holding <bench>/defended, overriding --protocol "
                         "(e.g. runs/<model>/ablation_full)")
    ap.add_argument("--response-key", required=True)
    ap.add_argument("--out", type=Path, help="JSON output (default: <variant-dir>/summary.json)")
    a = ap.parse_args()
    variant = a.variant_dir or (a.run_dir / a.protocol if a.protocol else None)
    if variant is None:
        ap.error("give --protocol or --variant-dir")
    res = summarize(a.run_dir, variant, a.response_key)
    if not res:
        sys.exit(f"no defended results under {variant}")
    out = a.out or variant / "summary.json"
    out.write_text(json.dumps(res, indent=2) + "\n", encoding="utf-8")

    print(f"| benchmark | n | unsafe before (same rows / full) | unsafe after | XSTest OR before | after | "
          f"added (all 250) | added (test half) | gate AUROC | threshold |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for held, r in res.items():
        after = r.get("over_refusal_after")
        after_t = r.get("over_refusal_after_test_half")
        print(f"| {held} | {r['n']} | {r['unsafe_before']:.1f} / {r['unsafe_before_full']:.1f} | "
              f"{r['unsafe_after']:.1f} | "
              f"{r['over_refusal_before']:.1f} | {fmt(after)} | "
              f"{fmt(None if after is None else after - r['over_refusal_before'], '+.1f')} | "
              f"{fmt(None if after_t is None else after_t - r['over_refusal_before_test_half'], '+.1f')} | "
              f"{fmt(r.get('gate_auroc'), '.2f')} | {fmt(r.get('threshold'), '.3f')} |")
    for label, a_key, b_key in (("all 250 prompts", "over_refusal_after", "over_refusal_before"),
                                ("test half", "over_refusal_after_test_half",
                                 "over_refusal_before_test_half")):
        added = [r[a_key] - r[b_key] for r in res.values() if a_key in r]
        if added:
            print(f"Avg. added over-refusal ({label}): {sum(added) / len(added):+.1f} pp")
    print(f"[OK] {out}")


if __name__ == "__main__":
    main()
