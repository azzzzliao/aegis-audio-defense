#!/usr/bin/env python3
"""Cross-benchmark generalization analysis (model-agnostic, NPZ-based, no GPU).

For benchmarks B1..Bk (each an extract_hidden.py NPZ + meta, same model):
  1. Transfer matrix heatmap: train a router on Bi (at a shared best layer),
     test AUROC on Bj. Off-diagonal = generalization to a DIFFERENT attack
     benchmark; diagonal = in-distribution (70/30 split).
  2. Unsafe-direction cosine across benchmarks (per layer line plot) AND a new
     benchmark x benchmark cosine HEATMAP at the shared layer:
     cos( dir_Bi, dir_Bj ), dir = mean(unsafe) - mean(safe). Low cosine => the
     model encodes "unsafe" differently per benchmark => poor transfer.

Outputs: transfer_matrix.png, direction_cosine_by_layer.png,
         direction_cosine_heatmap.png, cross_benchmark_summary.md

Usage:
  python cross_benchmark.py --pooling last --out-dir OUT \
    --bench AudioJailbreak:aj_feats.npz:aj_meta.jsonl \
    --bench SACRED:sacred_feats.npz:sacred_meta.jsonl \
    --bench JALM:jalm_feats.npz:jalm_meta.jsonl
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from bench_io import load_bench


def auroc(y, p):
    try:
        return roc_auc_score(y, p) if 0 < y.sum() < len(y) else float("nan")
    except Exception:
        return float("nan")


def cosine(a, b):
    return float(np.sum(a * b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def draw_heatmap(M, names, title, path, vmin, vmax, lo_text):
    fig, ax = plt.subplots(figsize=(1.6 + 1.3 * len(names), 1.4 + 1.1 * len(names)))
    im = ax.imshow(M, vmin=vmin, vmax=vmax, cmap="viridis")
    ax.set_xticks(range(len(names))); ax.set_yticks(range(len(names)))
    ax.set_xticklabels(names, rotation=30, ha="right"); ax.set_yticklabels(names)
    for i in range(len(names)):
        for j in range(len(names)):
            v = M[i, j]
            ax.text(j, i, "" if np.isnan(v) else f"{v:.2f}", ha="center", va="center",
                    color="white" if (not np.isnan(v) and v < lo_text) else "black")
    ax.set_title(title)
    fig.colorbar(im, fraction=0.046, pad=0.04)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", action="append", required=True, help="name:npz:meta (>=2 times)")
    ap.add_argument("--pooling", default="last")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)

    benches = [load_bench(s, a.pooling) for s in a.bench]
    names = [b[0] for b in benches]
    if len(benches) < 2:
        raise SystemExit("need at least two --bench specs for a cross-benchmark analysis")
    L = benches[0][1].shape[1]
    print("benches:", [(n, H.shape, int(y.sum())) for n, H, y in benches])

    # shared layer = best within-benchmark mean single-direction AUROC
    def dir_auroc(H, y, l):
        if y.sum() == 0 or y.sum() == len(y):
            return float("nan")
        d = H[y == 1][:, l, :].mean(0) - H[y == 0][:, l, :].mean(0)
        return auroc(y, H[:, l, :] @ d)
    layer_scores = [np.nanmean([dir_auroc(H, y, l) for _, H, y in benches]) for l in range(L)]
    shared_l = int(np.nanargmax(layer_scores))
    print(f"shared layer = {shared_l} (mean within-bench AUROC {layer_scores[shared_l]:.3f})")

    # 1. transfer matrix
    rng = np.random.RandomState(a.seed)
    splits = {}
    for n, H, y in benches:
        idx = rng.permutation(len(y)); cut = int(0.7 * len(y))
        splits[n] = (idx[:cut], idx[cut:])
    M = np.full((len(benches), len(benches)), np.nan)
    for i, (ni, Hi, yi) in enumerate(benches):
        tr = splits[ni][0]
        if yi[tr].sum() in (0, len(tr)):
            continue
        sc = StandardScaler().fit(Hi[tr, shared_l, :])
        clf = LogisticRegression(solver="liblinear", class_weight="balanced", max_iter=300,
                                 random_state=a.seed).fit(sc.transform(Hi[tr, shared_l, :]), yi[tr])
        for j, (nj, Hj, yj) in enumerate(benches):
            if i == j:
                te = splits[nj][1]
                M[i, j] = auroc(yj[te], clf.predict_proba(sc.transform(Hj[te, shared_l, :]))[:, 1])
            else:
                M[i, j] = auroc(yj, clf.predict_proba(sc.transform(Hj[:, shared_l, :]))[:, 1])
    draw_heatmap(M, names, f"Cross-benchmark router AUROC (layer {shared_l}, {a.pooling})\n"
                 "rows = train on, cols = test on", out / "transfer_matrix.png", 0.5, 1.0, 0.78)

    # 2a. direction cosine per layer (line)
    dirs = {n: np.stack([(H[y == 1][:, l, :].mean(0) - H[y == 0][:, l, :].mean(0)) for l in range(L)])
            for n, H, y in benches}
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a_, b_ = dirs[names[i]], dirs[names[j]]
            cos = np.sum(a_ * b_, 1) / (np.linalg.norm(a_, axis=1) * np.linalg.norm(b_, axis=1) + 1e-9)
            ax.plot(range(L), cos, marker="o", ms=3, label=f"{names[i]} vs {names[j]}")
    ax.axhline(0, color="gray", ls="--", lw=1)
    ax.set_xlabel("Layer"); ax.set_ylabel("cosine of unsafe-direction")
    ax.set_title("Is 'unsafe' represented the same across benchmarks?")
    ax.legend(fontsize=9); ax.set_ylim(-0.2, 1.0)
    fig.tight_layout(); fig.savefig(out / "direction_cosine_by_layer.png", dpi=150); plt.close(fig)

    # 2b. NEW: benchmark x benchmark direction-cosine heatmap at the shared layer
    C = np.eye(len(names))
    for i in range(len(names)):
        for j in range(len(names)):
            C[i, j] = cosine(dirs[names[i]][shared_l], dirs[names[j]][shared_l])
    draw_heatmap(C, names, f"Unsafe-direction cosine across benchmarks (layer {shared_l})\n"
                 "1 = same safety direction, ~0 = orthogonal", out / "direction_cosine_heatmap.png",
                 0.0, 1.0, 0.5)

    # markdown
    md = [f"# Cross-Benchmark Analysis ({a.pooling}, shared layer {shared_l})\n",
          "## Transfer AUROC (rows=train, cols=test)\n",
          "| train\\test | " + " | ".join(names) + " |",
          "|" + "---|" * (len(names) + 1)]
    for i, n in enumerate(names):
        md.append(f"| {n} | " + " | ".join("" if np.isnan(M[i, j]) else f"{M[i,j]:.3f}"
                                            for j in range(len(names))) + " |")
    md.append("\n## Unsafe-direction cosine (rows/cols = benchmark, shared layer)\n")
    md.append("| | " + " | ".join(names) + " |")
    md.append("|" + "---|" * (len(names) + 1))
    for i, n in enumerate(names):
        md.append(f"| {n} | " + " | ".join(f"{C[i,j]:.3f}" for j in range(len(names))) + " |")
    md.append("\nLow off-diagonal cosine => 'unsafe' lies along different directions per benchmark "
              "=> no unified safety representation => poor cross-benchmark transfer.\n")
    (out / "cross_benchmark_summary.md").write_text("\n".join(md))
    print(f"[OK] wrote {out}/ (transfer_matrix.png, direction_cosine_by_layer.png, "
          f"direction_cosine_heatmap.png, cross_benchmark_summary.md)")


if __name__ == "__main__":
    main()
