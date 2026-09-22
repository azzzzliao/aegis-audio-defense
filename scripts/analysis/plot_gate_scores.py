#!/usr/bin/env python3
"""Gate-score distributions: attacks vs XSTest benign, one panel per benchmark.

Reads the gate scores that run_aegis.sh writes (no generation involved):
runs/<model>/<protocol>/<bench>/scores/{<bench>,xstest}.jsonl. Each panel is titled
with the detection AUROC over those raw scores, so it shows what the gate separates
before any threshold is applied.

  python scripts/analysis/plot_gate_scores.py --models gemma4_e4b_it voxtral_small \\
      --protocol lobo
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
from summarize import auroc  # noqa: E402

BENCHES = [("audiojailbreak", "AudioJailbreak"), ("jalm", "JALM"), ("sacred", "SACRED")]
GREEN, RED = "#6aa84f", "#e06666"


def gate_scores(path):
    out = []
    for line in open(path, encoding="utf-8"):
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("eval_error") is None and row.get("gate") is not None:
            out.append(float(row["gate"]))
    return np.array(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--protocol", choices=["indomain", "lobo"], default="lobo")
    ap.add_argument("--runs", default=str(REPO / "runs"), type=Path)
    ap.add_argument("--out-dir", default=str(REPO / "assets/figures"), type=Path)
    a = ap.parse_args()

    a.out_dir.mkdir(parents=True, exist_ok=True)
    bins = np.linspace(0, 1, 31)
    for model in a.models:
        fig, axes = plt.subplots(1, len(BENCHES), figsize=(6 * len(BENCHES), 4.2))
        for ax, (bench, label) in zip(np.atleast_1d(axes), BENCHES):
            d = a.runs / model / a.protocol / bench / "scores"
            harmful, benign = gate_scores(d / f"{bench}.jsonl"), gate_scores(d / "xstest.jsonl")
            ax.hist(benign, bins=bins, density=True, color=GREEN, alpha=0.6,
                    label=f"benign (n={len(benign)})")
            ax.hist(harmful, bins=bins, density=True, color=RED, alpha=0.6,
                    label=f"attack (n={len(harmful)})")
            ax.set_title(f"{label}   AUROC={auroc(list(harmful), list(benign)):.2f}",
                         fontweight="bold")
            ax.set_xlabel("gate score")
            ax.set_xlim(0, 1)
            ax.legend()
        np.atleast_1d(axes)[0].set_ylabel("density")
        fig.suptitle(f"{model} — gate scores ({a.protocol}): attacks vs XSTest benign",
                     fontsize=14)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        out = a.out_dir / f"gate_scores_{model}_{a.protocol}.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print("wrote", out)


if __name__ == "__main__":
    main()
