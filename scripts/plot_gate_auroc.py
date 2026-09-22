#!/usr/bin/env python3
"""Held-out gate AUROC heatmap (rows = models, columns = held-out benchmark).

  python scripts/plot_gate_auroc.py --out figs/gate_auroc \\
      --summary "Gemma 4 E4B=runs/gemma4_e4b_it/summary.json" \\
      --summary "Voxtral Small=runs/voxtral_small/summary.json"
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

HELDS = [("audiojailbreak", "Audio\nJailbreak"), ("sacred", "SACRED\n(MSD)"),
         ("jalm", "JALM\n(ADiv+SSJ)")]
CMAP = LinearSegmentedColormap.from_list(
    "aegis", ["#EDD1CB", "#D99BA6", "#AF6C91", "#75466F", "#2D1E3E"])


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--summary", action="append", required=True, metavar="LABEL=PATH",
                    help="model label and its summary.json (from scripts/summarize.py); repeatable")
    ap.add_argument("--out", default="figs/gate_auroc", help="output stem (.pdf and .png)")
    a = ap.parse_args()

    labels, values = [], []
    for spec in a.summary:
        label, path = spec.split("=", 1)
        res = json.loads(Path(path).read_text())
        labels.append(label)
        values.append([res.get(h, {}).get("gate_auroc", float("nan")) for h, _ in HELDS])

    fig, ax = plt.subplots(figsize=(3.35, 0.45 * len(labels) + 0.9))
    im = ax.imshow(values, cmap=CMAP, vmin=0.5, vmax=1.0, aspect="auto")   # 0.5 = chance
    for i, row in enumerate(values):
        for j, v in enumerate(row):
            ax.text(j, i, "--" if v != v else f"{v:.2f}", ha="center", va="center",
                    fontsize=7.5, color="white" if v == v and v > 0.8 else "black")
    ax.set_xticks(range(len(HELDS)), [t for _, t in HELDS], fontsize=7)
    ax.set_yticks(range(len(labels)), labels, fontsize=7)
    ax.set_xlabel("held-out benchmark", fontsize=7)
    ax.tick_params(length=0)
    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.ax.tick_params(labelsize=6.5)
    cbar.set_label("gate AUROC (0.5 = chance)", fontsize=6.5)
    cbar.outline.set_visible(False)
    fig.tight_layout()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(f"{a.out}.{ext}", dpi=200, bbox_inches="tight")
        print("wrote", f"{a.out}.{ext}")


if __name__ == "__main__":
    main()
