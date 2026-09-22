#!/usr/bin/env python3
"""The risk-to-refusal gap figure: one panel per model, knowing vs doing on a twin axis.

Port of the private repo's plot_knowing_doing_gap.py from a single Qwen2-Audio panel to a
row of models. Each panel keeps that figure's idiom -- shallow/middle/deep depth bands, a
teal left axis for risk detectability, and a red right axis pinned to [0, 1] so a weak
refusal signal reads as weak rather than as a magnified late spike.

Each model keeps its OWN x range (qwen2 32, voxtral 40, phi4 32, vita 28, ultravox 32,
gemma4 42 layers); the panels are not forced onto a shared x axis.

All six curves come from the same pipeline and therefore the same definitions:
  knowing = unsafe-outcome vs safe-outcome probe AUROC (Llama-Guard label: did the attack
            succeed), NOT attack-vs-benign input classification
  doing   = logit-lens refusal-token probability on the undefended base model
This matters: the repo also ships qwen2_risk_to_refusal_datapoints.json, whose knowing
curve is an attack-vs-benign linear probe and starts at ~0.74 at layer 0. Mixing the two
definitions in one figure would invite a comparison that is not valid, so qwen2 is re-run
through this pipeline instead of being read from that file.

  python scripts/analysis/gap_plot_models.py
  python scripts/analysis/gap_plot_models.py --models voxtral_small gemma4_e4b_it   # subset
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
CURVES = Path(__file__).resolve().parent / "curves"

ORDER = ["qwen2_audio", "voxtral_small", "phi4_mm", "vita_15", "ultravox_v05",
         "gemma4_e4b_it"]
LABEL = {"qwen2_audio": "Qwen2-Audio-7B", "voxtral_small": "Voxtral-Small-24B",
         "phi4_mm": "Phi-4-multimodal", "vita_15": "VITA-1.5",
         "ultravox_v05": "Ultravox v0.5", "gemma4_e4b_it": "Gemma-4 E4B-it"}
KNOW, DO = "#00695c", "#c62828"


def depth_bands(ax, n_layers, y0, y1):
    """Shallow / middle / deep at 0.25 and 0.60 of depth, matching the source figure.

    For a 32-layer model this reproduces the [0,8] / [8,19] / [19,31] split that
    qwen2_risk_to_refusal_datapoints.json states explicitly.
    """
    s, mid = round(n_layers * 0.25), round(n_layers * 0.60)
    yb = y1 - (y1 - y0) * 0.05
    for x0, x1, name, fc, alpha, tc in [
            (-0.5, s, "shallow", "green", 0.07, "#4e8a52"),
            (s, mid, "middle", "gold", 0.09, "#a8811a"),
            (mid, n_layers - 0.5, "deep", "red", 0.06, "#c0655f")]:
        ax.axvspan(x0, x1, color=fc, alpha=alpha, lw=0, zorder=0)
        ax.text((max(x0, 0) + min(x1, n_layers)) / 2, yb, name, fontsize=8, style="italic",
                color=tc, ha="center", va="center", zorder=1)
    for xb in (s, mid):
        ax.axvline(xb, ls=(0, (4, 3)), lw=0.9, color="#cccccc", alpha=0.7, zorder=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=ORDER)
    ap.add_argument("--curves", default=str(CURVES))
    ap.add_argument("--out", default=str(REPO / "assets/figures/risk_to_refusal_gap"))
    a = ap.parse_args()

    loaded = []
    for m in a.models:
        p = Path(a.curves) / f"{m}_kd.json"
        if not p.exists():
            print(f"[skip] {p.name} missing")
            continue
        r = json.loads(p.read_text())
        if not r.get("knowing") or not r.get("doing"):
            print(f"[skip] {m}: curves incomplete (knowing={bool(r.get('knowing'))} "
                  f"doing={bool(r.get('doing'))})")
            continue
        loaded.append((m, r))
    if not loaded:
        raise SystemExit("no complete curve files; run scripts/gap_compute_curves.py first")

    # A model missing a benchmark would be pooled over fewer rows than the others and
    # silently stop being comparable, so say so loudly rather than drawing it unremarked.
    sets = {m: {b for b, n in (r.get("n_by_bench") or {}).items() if n} for m, r in loaded}
    common = max(sets.values(), key=len) if sets else set()
    for m, got in sets.items():
        if got and got != common:
            print(f"[WARN] {m}: pooled over {sorted(got)}, others over {sorted(common)} "
                  f"-- not comparable with the other panels")

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.linewidth": 1.1, "axes.edgecolor": "#333333",
                         "svg.fonttype": "none", "figure.dpi": 150})
    fig, axes = plt.subplots(1, len(loaded), figsize=(3.7 * len(loaded), 4.6))
    if len(loaded) == 1:
        axes = [axes]

    # Shared left-axis range so the six panels are comparable at a glance.
    allk = np.concatenate([r["knowing"] for _, r in loaded])
    y0 = np.floor((allk.min() - 0.03) * 20) / 20
    y1 = np.ceil((allk.max() + 0.03) * 20) / 20

    for ax, (m, r) in zip(axes, loaded):
        n = r["n_layers"]
        x = np.asarray(r["x"])
        axR = ax.twinx()
        ax.set_ylim(y0, y1)
        depth_bands(ax, n, y0, y1)

        lK, = ax.plot(x, r["knowing"], color=KNOW, lw=2.3, marker="o", ms=3.0,
                      markeredgecolor="white", markeredgewidth=0.5, zorder=4,
                      label="Risk detectable (knowing)")
        lD, = axR.plot(x, r["doing"], color=DO, lw=2.3, marker="o", ms=3.0,
                       markeredgecolor="white", markeredgewidth=0.5, zorder=4,
                       label="P(refusal), no defense (doing)")
        axR.set_ylim(0, 1.0)

        # Negative-class count: "n_neg" is the current field; "n_benign"/"n_safe" are older
        # spellings kept so previously generated curve files still render.
        neg = next((r[k] for k in ("n_neg", "n_benign", "n_safe") if r.get(k) is not None), None)
        ax.set_title(f"{LABEL.get(m, m)}\n{n} layers · {r['n_unsafe']} pos / {neg} neg",
                     fontsize=10.5, fontweight="bold")
        ax.set_xlabel("decoder layer")
        ax.set_xlim(-0.5, n - 0.5)
        ax.grid(axis="y", color="#EAEAEA", lw=0.8, zorder=0)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        axR.spines["top"].set_visible(False)

        first, last = ax is axes[0], ax is axes[-1]
        ax.tick_params(axis="y", colors=KNOW, length=4, labelleft=first)
        axR.tick_params(axis="y", colors=DO, length=4, labelright=last)
        ax.spines["left"].set_color(KNOW)
        axR.spines["left"].set_color(KNOW)
        axR.spines["right"].set_color(DO)
        if first:
            ax.set_ylabel("risk detectability (AUROC)", color=KNOW, fontsize=12)
        if last:
            axR.set_ylabel("refusal-token probability (logit lens)", color=DO, fontsize=12)

    leg = fig.legend([lK, lD], [lK.get_label(), lD.get_label()], loc="lower center",
                     ncol=2, frameon=True, fontsize=11.5, bbox_to_anchor=(0.5, -0.02))
    for t, c in zip(leg.get_texts(), (KNOW, DO)):
        t.set_color(c)

    fig.tight_layout(rect=(0, 0.06, 1, 1))
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    stem = f"{out}_{len(loaded)}models"
    for ext in ("png", "pdf"):
        fig.savefig(f"{stem}.{ext}", bbox_inches="tight")
        print(f"[OK] {stem}.{ext}")
    for m, r in loaded:
        k, d = r["knowing"], r["doing"]
        print(f"  {LABEL.get(m, m):22s} knowing {k[0]:.3f}->{max(k):.3f}@L{int(np.argmax(k))}"
              f"   doing {min(d):.4f}->max {max(d):.4f}")


if __name__ == "__main__":
    main()
