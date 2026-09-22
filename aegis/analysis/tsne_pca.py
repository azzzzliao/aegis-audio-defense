#!/usr/bin/env python3
"""PCA / t-SNE of hidden states, coloured by benchmark and attack-vs-benign.

Model-agnostic, NPZ-based, no GPU. Pass the benchmarks (and optionally benign
sets) as name:features.npz:meta.jsonl specs.

For each --bench, label==1 rows are treated as that benchmark's "attack" points
and label==0 rows as benign. You can also add explicit --benign specs (any all-
benign NPZ). Produces:
  - pca_separation.png    (attack-vs-benign + attacks coloured by benchmark)
  - tsne.png              (subsampled t-SNE, coloured by benchmark + benign)
  - prints per-benchmark attack-vs-benign 5-fold CV AUROC

Usage:
  python tsne_pca.py --layer 15 --pooling last --out-dir OUT \
    --bench AJ:aj_feats.npz:aj_meta.jsonl \
    --bench SACRED:sac_feats.npz:sac_meta.jsonl \
    --bench JALM:jalm_feats.npz:jalm_meta.jsonl
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

from bench_io import load_bench

PALETTE = ["#1565c0", "#6a1b9a", "#c62828", "#ef6c00", "#00838f", "#5d4037"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", action="append", required=True, help="name:npz:meta")
    ap.add_argument("--benign", action="append", default=[], help="name:npz:meta (extra all-benign sets)")
    ap.add_argument("--layer", type=int, default=15)
    ap.add_argument("--pooling", default="last")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--tsne-per-class", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    L = a.layer

    attacks = {}   # name -> [n, D]
    benign_parts = []
    for spec in a.bench:
        name, H, y = load_bench(spec, a.pooling)
        attacks[name] = H[y == 1][:, L, :]
        if (y == 0).any():
            benign_parts.append(H[y == 0][:, L, :])
    for spec in a.benign:
        _, H, y = load_bench(spec, a.pooling)
        benign_parts.append(H[:, L, :])
    BEN = np.concatenate(benign_parts) if benign_parts else np.zeros((0, next(iter(attacks.values())).shape[1]))
    names = list(attacks)
    print("counts:", {k: len(v) for k, v in attacks.items()}, "benign", len(BEN))

    # ---- PCA ----
    ATK = np.concatenate([attacks[n] for n in names])
    X = np.concatenate([ATK, BEN]) if len(BEN) else ATK
    Xs = StandardScaler().fit_transform(X)
    Z = PCA(2, random_state=a.seed).fit_transform(Xs)
    Zatk, Zben = Z[:len(ATK)], Z[len(ATK):]
    fig, ax = plt.subplots(1, 2, figsize=(13, 5.2))
    if len(BEN):
        ax[0].scatter(Zben[:, 0], Zben[:, 1], s=6, c="#2e7d32", alpha=.4, label=f"benign (n={len(BEN)})")
    ax[0].scatter(Zatk[:, 0], Zatk[:, 1], s=6, c="#c62828", alpha=.4, label=f"attack (n={len(ATK)})")
    ax[0].set_title(f"L{L} hidden — attack vs benign"); ax[0].legend(); ax[0].set_xlabel("PC1"); ax[0].set_ylabel("PC2")
    off = 0
    for k, n in enumerate(names):
        zz = Zatk[off:off + len(attacks[n])]; off += len(attacks[n])
        ax[1].scatter(zz[:, 0], zz[:, 1], s=6, c=PALETTE[k % len(PALETTE)], alpha=.4, label=f"{n} atk")
    if len(BEN):
        ax[1].scatter(Zben[:, 0], Zben[:, 1], s=6, c="#bbbbbb", alpha=.3, label="benign")
    ax[1].set_title(f"L{L} hidden — attacks by benchmark"); ax[1].legend(); ax[1].set_xlabel("PC1")
    plt.tight_layout(); plt.savefig(out / "pca_separation.png", dpi=130); plt.close()

    # ---- t-SNE (subsample) ----
    def sub(M, n):
        if len(M) <= n:
            return M
        idx = np.random.RandomState(a.seed).choice(len(M), n, replace=False)
        return M[idx]
    parts, lab = [], []
    for n in names:
        s = sub(attacks[n], a.tsne_per_class); parts.append(s); lab += [n] * len(s)
    if len(BEN):
        s = sub(BEN, a.tsne_per_class + 100); parts.append(s); lab += ["benign"] * len(s)
    Xt = np.concatenate(parts)
    Zt = TSNE(2, perplexity=30, random_state=a.seed, init="pca").fit_transform(StandardScaler().fit_transform(Xt))
    fig, ax = plt.subplots(figsize=(7, 6))
    color_for = {n: PALETTE[k % len(PALETTE)] for k, n in enumerate(names)}
    color_for["benign"] = "#2e7d32"
    for k in (["benign"] if len(BEN) else []) + names:
        m = [i for i, l in enumerate(lab) if l == k]
        ax.scatter(Zt[m, 0], Zt[m, 1], s=10, c=color_for[k], alpha=.6, label=k)
    ax.set_title(f"t-SNE of L{L} hidden (subsample)"); ax.legend()
    plt.tight_layout(); plt.savefig(out / "tsne.png", dpi=130); plt.close()

    # ---- per-benchmark separation AUROC ----
    if len(BEN):
        print(f"L{L} attack-vs-benign separation (5-fold CV AUROC):")
        for n in names:
            atkM = attacks[n]; k = min(len(atkM), len(BEN)); r = np.random.RandomState(a.seed)
            Xc = np.concatenate([atkM[r.choice(len(atkM), k, False)], BEN[r.choice(len(BEN), k, False)]])
            yc = np.r_[np.ones(k), np.zeros(k)]
            auc = cross_val_score(LogisticRegression(max_iter=500, C=0.5),
                                  StandardScaler().fit_transform(Xc), yc, cv=5, scoring="roc_auc").mean()
            print(f"  {n}: {auc:.3f}")
    print("[OK]", out)


if __name__ == "__main__":
    main()
