#!/usr/bin/env python3
"""Pick the gate threshold on the XSTest validation half, without the held-out attack.

XSTest ids are sorted and split alternately: even positions form the validation
half, odd positions the test half. Among validation prompts that the undefended
model did not refuse, a firing gate counts as added over-refusal. The threshold is
the smallest validation gate score whose added over-refusal is <= --budget. The
held-out attack benchmark and the XSTest test half are never used.

  python scripts/calibrate_threshold.py --scores runs/m/jalm/scores/xstest.jsonl \\
      --baseline runs/m/baseline/xstest.jsonl --response-key gemma4_response \\
      --out runs/m/jalm/threshold.json

Prints the threshold on stdout (diagnostics go to stderr).
"""
import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "aegis"))
from common import jl, record_id  # noqa: E402
from refusal import is_refusal  # noqa: E402


def split_halves(ids):
    """Deterministic XSTest (validation, test) halves: alternate positions of the sorted ids."""
    ordered = sorted(ids)
    return ordered[0::2], ordered[1::2]


def select_threshold(scores, baseline_refusal, budget):
    """Return the calibration record for gate ``scores`` ({id: score}) on XSTest."""
    if not 0.0 <= budget <= 1.0:
        raise ValueError("budget must be in [0, 1]")
    val_ids, test_ids = split_halves(scores)
    if not val_ids or not test_ids:
        raise ValueError("need XSTest scores for both halves")
    eligible = [i for i in val_ids if not baseline_refusal.get(i, False)]

    def added_over_refusal(threshold):
        if not eligible:
            return 0.0
        return sum(scores[i] >= threshold for i in eligible) / len(eligible)

    candidates = sorted({scores[i] for i in val_ids})
    feasible = [t for t in candidates if added_over_refusal(t) <= budget]
    # Nothing feasible: sit just above every validation score (the gate never fires on it).
    threshold = min(feasible) if feasible else math.nextafter(max(candidates), math.inf)
    test_or = sum(scores[i] >= threshold or baseline_refusal.get(i, False)
                  for i in test_ids) / len(test_ids)
    return {"threshold": threshold, "budget": budget,
            "val_added_over_refusal": added_over_refusal(threshold),
            "test_over_refusal_estimate": test_or,
            "val_count": len(val_ids), "test_count": len(test_ids)}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scores", required=True, help="run_defense.py --score-only output on XSTest")
    ap.add_argument("--baseline", required=True,
                    help="undefended XSTest responses (aegis/run_model.py output)")
    ap.add_argument("--response-key", required=True)
    ap.add_argument("--budget", type=float, default=0.10,
                    help="max added over-refusal on the validation half (default 0.10)")
    ap.add_argument("--out", help="also write the calibration record here as JSON")
    a = ap.parse_args()

    scores = {record_id(r): float(r["gate"]) for r in jl(a.scores) if r.get("gate") is not None}
    refused = {record_id(r): is_refusal(r.get(a.response_key)) for r in jl(a.baseline)}
    missing = sorted(set(scores) - set(refused))
    if missing:
        sys.exit(f"{len(missing)} scored XSTest ids have no baseline response, e.g. {missing[:3]}")
    record = select_threshold(scores, refused, a.budget)
    record.update(scores=a.scores, baseline=a.baseline)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(f"[calibrate] threshold={record['threshold']:.6f} "
          f"val_added_over_refusal={record['val_added_over_refusal']:.3f} "
          f"(budget {a.budget}, n_val={record['val_count']})", file=sys.stderr)
    print(repr(record["threshold"]))


if __name__ == "__main__":
    main()
