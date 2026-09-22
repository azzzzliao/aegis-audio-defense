#!/usr/bin/env python3
"""knowing / doing curves per model for the risk-to-refusal gap figure.

Port of the private repo's compute_perbench_knowing_doing.py, generalized: the layer count
is read from the npz instead of hardcoded to 32, the model comes from the adapter registry
instead of being fixed to Qwen2-Audio, and the benchmarks are POOLED rather than per-bench.

  KNOWING  per-layer AUROC separating harmful inputs that elicited an UNSAFE response
           (positives) from one of two possible negative sets, selected by --negatives:

           attack_safe (default) -- attack rows the model REFUSED. Positives and negatives
             are then the same audio, from the same benchmarks, differing only in outcome,
             so nothing about the recording itself can be used as a shortcut. Measured:
             voxtral L0=0.478 -> max 0.711@L23, phi4 L0=0.557 -> max 0.723@L16, i.e. a real
             mid-layer rise.

           benign -- genuine benign audio (benign_train + XSTest). This matches the literal
             wording of main.tex Sec.3, but on this data it is DOMINATED BY A SHORTCUT: the
             negatives come from different recordings than the positives, and a probe can
             separate them without understanding the content at all. Evidence: benign_train
             vs XSTest -- both benign, identical harmfulness -- are themselves separable at
             mid-layer AUROC 0.97-0.98 (qwen2 0.981, vita 0.981, gemma4 0.969). Every model
             lands at 0.99-1.00 from layer 1 onward, so the curve is a flat line and the gap
             disappears. Kept only for reference/ablation.

  DOING    per-layer mean refusal-token probability over the UNSAFE attack rows: the saved
           last-prompt-token hidden state is pushed through the model's final_norm + lm_head
           and the softmax mass on the refusal first-tokens is summed. Independent of
           --negatives, so a doing curve computed once stays valid across both variants.

DOING needs the model resident (only its norm + head are used -- no audio is re-run), so run
this in the model's own conda env on a GPU. KNOWING is pure CPU. Layer convention: npz index
0 is the embedding output, so model layer L is npz index L+1.

  python scripts/analysis/gap_compute_curves.py --model vita_15 --skip-doing            # same-source
  python scripts/analysis/gap_compute_curves.py --model vita_15 --negatives benign --skip-doing
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
DEFAULT_HIDDEN = REPO / "runs"
BENCHES = ["audiojailbreak", "jalm_adiv", "jalm_ssj", "sacred"]
SEED = 42


def _load_npz(hidden_dir, model, bench):
    """Return (hidden_last, labels, n_dropped), dropping rows whose extraction failed.

    extract_hidden.py writes zero placeholders for failed rows, so they must be dropped via
    the meta file rather than trusted (same rule as analysis/bench_io.py).
    """
    npz = hidden_dir / f"{bench}.npz"
    meta_path = hidden_dir / f"{bench}_meta.jsonl"
    if not npz.exists():
        return None, None, 0
    d = np.load(npz, allow_pickle=True)
    meta = [json.loads(l) for l in open(meta_path, encoding="utf-8") if l.strip()]
    keep = np.array([i for i, r in enumerate(meta) if not r.get("extract_error")])
    if len(keep) == 0:
        print(f"  [warn] {bench}: every row has extract_error, skipping")
        return None, None, len(meta)
    return d["hidden_last"][keep], d["labels"][keep].astype(int), len(meta) - len(keep)


def load_attacks(hidden_dir, model, benches):
    """Split the attack benchmarks into jailbroken (label 1) and refused (label 0) rows."""
    POS, NEG, n_by_bench = [], [], {}
    for bench in benches:
        h, y, dropped = _load_npz(hidden_dir, model, bench)
        if h is None:
            print(f"  [warn] missing {model}_{bench}.npz, skipping")
            continue
        POS.append(h[y == 1])
        NEG.append(h[y == 0])
        n_by_bench[bench] = int((y == 1).sum())
        print(f"  {bench:7s} {h.shape}  unsafe={int((y == 1).sum()):4d} "
              f"safe={int((y == 0).sum()):4d}  dropped={dropped}")
    if not POS:
        raise SystemExit(f"no usable attack npz for {model} under {hidden_dir}; "
                         f"run scripts/analysis/run_gap.sh first")
    return np.concatenate(POS), np.concatenate(NEG), n_by_bench


def load_benign(hidden_dir, model):
    """Genuine benign audio. See the --negatives note in the module docstring first."""
    h, y, dropped = _load_npz(hidden_dir, model, "benign")
    if h is None:
        raise SystemExit(
            f"missing benign.npz under {hidden_dir}. Build it with:\n"
            f"  python scripts/analysis/gap_build_manifests.py --model {model} --benign-only\n"
            f"  MODEL={model} BENCHES=benign bash scripts/analysis/run_gap.sh")
    if y.sum() != 0:
        print(f"  [warn] benign npz has {int(y.sum())} rows labelled unsafe; expected 0")
    print(f"  {'benign':7s} {h.shape}  dropped={dropped}")
    return h


def knowing_curve(Hpos, Hneg, n_layers, seed=SEED):
    """Per-layer class-balanced 5-fold CV AUROC. H is [N, K, D]; model layer L -> index L+1."""
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    rng = np.random.RandomState(seed)
    k = min(len(Hpos), len(Hneg))
    if k < 10:
        raise SystemExit(f"only {k} rows in the smaller class; too few to probe")
    X_all = np.concatenate([Hpos[rng.permutation(len(Hpos))[:k]],
                            Hneg[rng.permutation(len(Hneg))[:k]]])
    yb = np.r_[np.ones(k, dtype=int), np.zeros(k, dtype=int)]
    print(f"  [knowing] balanced set: {k} positive + {k} negative")

    out = []
    for L in range(n_layers):
        X = X_all[:, L + 1, :].astype(np.float32)
        aucs = []
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(X, yb):
            pipe = make_pipeline(
                StandardScaler(),
                PCA(n_components=min(48, X.shape[1], len(tr) - 1), random_state=seed),
                LogisticRegression(max_iter=2000, C=1.0))
            pipe.fit(X[tr], yb[tr])
            aucs.append(roc_auc_score(yb[te], pipe.predict_proba(X[te])[:, 1]))
        out.append(float(np.mean(aucs)))
        if (L + 1) % 8 == 0 or L == n_layers - 1:
            print(f"    layer {L:3d}  AUROC {out[-1]:.3f}", flush=True)
    return out


def doing_curve(Hpos, n_layers, model_name, device):
    """Per-layer mean refusal-token probability over the unsafe attack rows."""
    import torch
    sys.path.insert(0, str(REPO / "aegis"))
    import model_registry as registry  # noqa: E402  (after the env var is set in main)

    adapter = registry.get(model_name)
    print(f"  [doing] loading {model_name} (only norm + lm_head are used) ...")
    model, proc = adapter.load(device)
    tok = adapter.tokenizer(proc)
    norm, head = adapter.final_norm(model), adapter.lm_head(model)

    ref_ids = sorted({t[0] for t in
                      (tok.encode(p, add_special_tokens=False) for p in adapter.refusal_phrases)
                      if t})
    print(f"  [doing] {len(adapter.refusal_phrases)} refusal phrases -> {len(ref_ids)} first-token ids")

    head_dev = next(head.parameters()).device        # may differ from cuda:0 when sharded
    head_dtype = next(head.parameters()).dtype
    ref = torch.tensor(ref_ids, device=head_dev)
    print(f"  [doing] {len(Hpos)} unsafe rows")

    curve = []
    for L in range(n_layers):
        h = torch.from_numpy(Hpos[:, L + 1, :].astype(np.float32)).to(head_dev)
        vals = []
        with torch.inference_mode():
            for i in range(0, h.shape[0], 64):
                probs = torch.softmax(head(norm(h[i:i + 64].to(head_dtype))).float(), dim=-1)
                vals.append(probs[:, ref].sum(-1).cpu())
        curve.append(float(torch.cat(vals).mean()))
        if (L + 1) % 8 == 0 or L == n_layers - 1:
            print(f"    layer {L:3d}  P(refusal) {curve[-1]:.4f}", flush=True)
    return curve


DEFINITION = {
    "attack_safe": "jailbroken (unsafe) vs refused (safe) -- same attack audio, same source",
    "benign": "harmful-unsafe vs benign audio (main.tex Sec.3 wording; source-shortcut prone)",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--runs", default=str(DEFAULT_HIDDEN), type=Path)
    ap.add_argument("--hidden-dir", default=None,
                    help="dir holding <bench>.npz (default: <runs>/<model>/gap/hidden)")
    ap.add_argument("--benches", nargs="*", default=BENCHES)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--negatives", choices=["attack_safe", "benign"], default="attack_safe",
                    help="knowing negatives: 'attack_safe' = attack rows the model refused "
                         "(same audio source as positives, no shortcut); 'benign' = benign "
                         "audio (matches main.tex wording but is shortcut-dominated here). "
                         "See the module docstring for the measured evidence.")
    ap.add_argument("--refusal-set", choices=["en", "multi"], default="multi",
                    help="'multi' adds non-English refusal openers. JALM ADiv is over half "
                         "non-English and the English-only list silently reports ~0 refusal "
                         "probability there (model_registry.py:44-49). Pooled runs need multi.")
    ap.add_argument("--skip-knowing", action="store_true")
    ap.add_argument("--skip-doing", action="store_true")
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    # Must be set BEFORE model_registry is imported: REFUSAL_PHRASES is bound at import time.
    os.environ["REFUSAL_PHRASE_SET"] = a.refusal_set

    hidden_dir = Path(a.hidden_dir) if a.hidden_dir else a.runs / a.model / "gap" / "hidden"
    print(f"== {a.model}  (negatives={a.negatives}) ==")
    Hpos, Hatk_safe, n_by_bench = load_attacks(hidden_dir, a.model, a.benches)
    n_layers = Hpos.shape[1] - 1
    print(f"  positives (harmful-unsafe): {Hpos.shape}  n_layers={n_layers}")

    result = {"model": a.model, "n_layers": n_layers, "x": list(range(n_layers)),
              "n_unsafe": int(len(Hpos)), "benches": a.benches, "refusal_set": a.refusal_set,
              "n_by_bench": n_by_bench, "negatives": a.negatives,
              "knowing_definition": DEFINITION[a.negatives],
              "knowing": None, "doing": None, "n_neg": None}

    if not a.skip_knowing:
        Hneg = Hatk_safe if a.negatives == "attack_safe" else load_benign(hidden_dir, a.model)
        if a.negatives == "attack_safe":
            print(f"  negatives (attack-refused): {Hneg.shape}")
        if Hneg.shape[1:] != Hpos.shape[1:]:
            raise SystemExit(f"negative shape {Hneg.shape[1:]} != positive {Hpos.shape[1:]}")
        result["n_neg"] = int(len(Hneg))
        result["knowing"] = knowing_curve(Hpos, Hneg, n_layers)
    if not a.skip_doing:
        result["doing"] = doing_curve(Hpos, n_layers, a.model, a.device)

    out = Path(a.out) if a.out else Path(__file__).resolve().parent / "curves" / f"{a.model}_kd.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    # Merge so --skip-doing and --skip-knowing can be run separately. Only 'doing' is carried
    # over: it does not depend on --negatives. A stale 'knowing' from the other variant must
    # NOT survive, so it is only kept when this run also used the same negatives.
    if out.exists():
        prev = json.loads(out.read_text())
        if result["doing"] is None:
            result["doing"] = prev.get("doing")
        if result["knowing"] is None and prev.get("negatives") == a.negatives:
            result["knowing"] = prev.get("knowing")
            result["n_neg"] = prev.get("n_neg")
    out.write_text(json.dumps(result, indent=2))
    k, d = result["knowing"], result["doing"]
    if k:
        print(f"  knowing L0={k[0]:.3f} -> max={max(k):.3f} @L{int(np.argmax(k))}")
    if d:
        print(f"  doing   min={min(d):.4f} -> L{n_layers - 1}={d[-1]:.4f} max={max(d):.4f}")
    print(f"[OK] {out}")


if __name__ == "__main__":
    main()
