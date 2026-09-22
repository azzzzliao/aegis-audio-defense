"""Shared loader for analysis scripts that consume the extract_hidden.py NPZ.

A benchmark spec is "name:features.npz:meta.jsonl". Rows whose meta has an
``extract_error`` are dropped (their hidden states are zero placeholders).
"""
import json

import numpy as np


def jl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def load_bench(spec, pooling="last"):
    """Return (name, H, y) with H = hidden_{pooling} [N, L, D] float32, y int."""
    name, npz_p, meta_p = spec.split(":", 2)
    d = np.load(npz_p, allow_pickle=True)
    m = jl(meta_p)
    valid = np.array([i for i, r in enumerate(m) if not r.get("extract_error")])
    if len(valid) == 0:
        raise ValueError(f"{name}: no valid rows (all have extract_error)")
    H = d[f"hidden_{pooling}"][valid].astype(np.float32)
    y = d["labels"][valid].astype(int)
    return name, H, y
