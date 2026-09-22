"""Resolve adapter-config layer indices to absolute model decoder layers.

Background
----------
The gate hook, the late LoRA adapters, and the refusal probe all attach to
*absolute* decoder-layer positions on the live model (``layers[i]``).

When the gate/probe was trained directly on the live model (the default in the
existing ``train_joint_*`` scripts), ``gate_layer`` / ``late`` are already
absolute model-layer indices -> nothing to do.

But a collaborator may instead train the gate on a *sparse* set of extracted
hidden features (e.g. only layers ``[0, 8, 16, 24, 32]`` were dumped to the
NPZ). In that case the index the probe selected is a **position in the feature
array**, not an absolute model layer. To deploy it we must map
position -> absolute layer using the extraction's ``layer_indices``.

Config contract (config.json in the adapter dir)
------------------------------------------------
    {
      "gate_layer": 15,            # int
      "late": [24, 25, ..., 31],   # list[int]
      "hidden": 4096, "rank": 8,   # unchanged
      "gate_type": "mlp", "gate_hidden": 256,

      # OPTIONAL, only when the gate was trained on sparse features:
      "layer_space": "feature_positions",   # default "absolute"
      "layer_indices": [0, 8, 16, 24, 32],  # feature-position -> model layer
      "refusal_probe_layer": 4              # optional; else adapter default (last)
    }
"""
from __future__ import annotations


def resolve_layers(config: dict, n_model_layers: int, default_probe_layer: int):
    """Return ``(gate_layer, late_layers, probe_layer)`` as absolute indices.

    Raises ``ValueError`` with an actionable message if anything is out of range
    (the most common symptom of a sparse-extraction mismatch).
    """
    space = config.get("layer_space", "absolute")
    gate = config["gate_layer"]
    late = list(config["late"])
    probe = config.get("refusal_probe_layer", None)

    if space == "feature_positions":
        idx = config.get("layer_indices")
        if not idx:
            raise ValueError(
                "config has 'layer_space':'feature_positions' but no 'layer_indices'. "
                "Add the list of absolute model layers that were extracted, e.g. "
                "[0, 8, 16, 24, 32]."
            )

        def to_abs(pos, what):
            if not (0 <= pos < len(idx)):
                raise ValueError(
                    f"{what}={pos} is not a valid position into layer_indices "
                    f"(len {len(idx)})."
                )
            return int(idx[pos])

        gate = to_abs(gate, "gate_layer")
        late = [to_abs(p, "late") for p in late]
        probe = to_abs(probe, "refusal_probe_layer") if probe is not None else None
    elif space != "absolute":
        raise ValueError(f"unknown layer_space '{space}' (use 'absolute' or 'feature_positions')")

    if probe is None:
        probe = default_probe_layer

    candidates = [("gate_layer", gate), ("refusal_probe_layer", probe)]
    candidates += [("late", l) for l in late]
    bad = [(name, v) for name, v in candidates if not (0 <= v < n_model_layers)]
    if bad:
        raise ValueError(
            f"layer indices out of range for a model with {n_model_layers} decoder "
            f"layers: {bad}.\n"
            f"  resolved gate={gate} late={late} probe={probe}.\n"
            f"If you trained the gate on a SPARSE subset of extracted layers, set "
            f"'layer_space':'feature_positions' and add 'layer_indices' (the absolute "
            f"model layers you extracted) to config.json so positions can be mapped."
        )
    return gate, late, probe
