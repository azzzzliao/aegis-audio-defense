#!/usr/bin/env python3
"""Train the AEGIS risk gate + late-layer LoRA on a frozen audio LLM.

Loss = LM(target | audio, prompt) + gate_bce * BCE(g, harmful) + gate_l1 * g, with the
base model frozen; only the gate MLP and the rank-r adapters on layers
[late_from, n_layers) are trained. Saves router_head.pt / late_adapters.pt /
config.json, the adapter contract consumed by run_defense.py.

The base model can be sharded across GPUs (``--device auto`` or ``0,1``); each
trainable piece is placed on the device of the layer it hooks, so the gate and the
late adapters work even when those layers sit on different cards.

  python train_gate_lora.py --model gemma4_e4b_it --device cuda:0 \
      --data train_gemma4_e4b_it_holdout_jalm.jsonl --out-dir out/adapter \
      --gate-layer 21 --late-from 25
"""
import argparse
import json
import os
import random
import tempfile
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

import model_registry as registry
from common import jl, resolve_audio


def enforce_training_success(summary, strict):
    """Raise when strict training encountered an unsafe training outcome."""
    if strict:
        integer_fields = (
            "input_skips", "oom_skips", "expected_examples", "epochs",
            "epochs_completed", "expected_processed_steps", "processed_steps",
        )
        if "finite_loss" not in summary or summary["finite_loss"] is not True:
            raise RuntimeError("strict training finite_loss evidence must be true boolean")
        for field in integer_fields:
            value = summary.get(field)
            if type(value) is not int:
                raise RuntimeError(
                    f"strict training {field} evidence must be an integer")
            minimum = 0 if field in {
                "input_skips", "oom_skips", "epochs_completed", "processed_steps"
            } else 1
            if value < minimum:
                raise RuntimeError(
                    f"strict training {field} evidence must be at least {minimum}")
    if not summary["finite_loss"]:
        raise RuntimeError("training observed a non-finite loss")
    if strict and (summary["input_skips"] or summary["oom_skips"]):
        raise RuntimeError("strict training does not permit skipped examples")
    if strict and summary["epochs_completed"] != summary["epochs"]:
        raise RuntimeError(
            "strict training epochs_completed does not match configured epochs")
    if strict and summary["expected_processed_steps"] != (
            summary["expected_examples"] * summary["epochs"]):
        raise RuntimeError(
            "strict training expected_processed_steps does not match examples times epochs")
    if strict and summary["processed_steps"] != summary["expected_processed_steps"]:
        raise RuntimeError(
            "strict training processed_steps does not match expected_processed_steps")


def optimizer_step_due(successful_steps, grad_accum, end_of_epoch=False):
    """Return whether accumulated successful gradients must be applied now."""
    if type(successful_steps) is not int or successful_steps < 0:
        raise ValueError("successful_steps must be a nonnegative integer")
    if type(grad_accum) is not int or grad_accum <= 0:
        raise ValueError("grad_accum must be a positive integer")
    remainder = successful_steps % grad_accum
    if end_of_epoch:
        return remainder != 0
    return successful_steps > 0 and remainder == 0


def record_input_skip(summary, error):
    """Classify input construction failures without folding CUDA OOM into input errors."""
    if isinstance(error, torch.cuda.OutOfMemoryError):
        summary["oom_skips"] += 1
        return "oom"
    summary["input_skips"] += 1
    return "input"


def write_summary_atomic(path, summary):
    """Durably replace a training summary without exposing a partial JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_name = None
    try:
        with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=path.parent,
                prefix=f".{path.name}.", suffix=".tmp", delete=False) as handle:
            temp_name = handle.name
            json.dump(summary, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    finally:
        if temp_name is not None:
            Path(temp_name).unlink(missing_ok=True)


def adapter_config(args, late, hidden, summary):
    """Return the adapter contract consumed by run_defense.py."""
    config = {
        "model": getattr(args, "model", None),
        "gate_layer": args.gate_layer,
        "late": late,
        "rank": args.rank,
        "hidden": hidden,
        "gate_type": args.gate_type,
        "no_gate": args.no_gate,
        "gate_bce": args.gate_bce,
        "gate_l1": args.gate_l1,
        "epochs": args.epochs,
        "seed": args.seed,
        "prompt_mode": getattr(args, "prompt_mode", "safety"),
        "expected_examples": summary.get("expected_examples"),
        "expected_processed_steps": summary.get("expected_processed_steps"),
        "input_skips": summary["input_skips"],
        "oom_skips": summary["oom_skips"],
    }
    if args.gate_type == "mlp":
        config["gate_hidden"] = args.gate_hidden
    return config


def finalize_adapter(out_dir, summary_path, router, A, B, args, late, hidden, summary, strict):
    """Persist only a training run that satisfies the configured safety policy."""
    try:
        write_summary_atomic(summary_path, summary)
        enforce_training_success(summary, strict)
    except Exception:
        summary["adapter_saved"] = False
        write_summary_atomic(summary_path, summary)
        raise

    try:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        torch.save(router.state_dict(), out_dir / "router_head.pt")
        torch.save({"A": A.state_dict(), "B": B.state_dict()}, out_dir / "late_adapters.pt")
        with (out_dir / "config.json").open("w", encoding="utf-8") as handle:
            json.dump(adapter_config(args, late, hidden, summary), handle, indent=2)
        summary["adapter_saved"] = True
        write_summary_atomic(summary_path, summary)
    except Exception:
        summary["adapter_saved"] = False
        write_summary_atomic(summary_path, summary)
        raise
    return out_dir


def dev_of(module):
    return next(module.parameters()).device


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=registry.available())
    ap.add_argument("--data", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--device", default="auto", help="auto / 0,1 (shard) or cuda:0 (single)")
    ap.add_argument("--prompt-mode", choices=registry.PROMPT_MODES, default="safety")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--gate-layer", type=int, required=True,
                    help="decoder layer whose last-prompt-token hidden state feeds the gate")
    ap.add_argument("--late-from", type=int, required=True,
                    help="first intervention layer; LoRA is attached to [late_from, n_layers)")
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--gate-l1", type=float, default=0.01)
    ap.add_argument("--gate-bce", type=float, default=1.0,
                    help="supervised gate BCE weight (attack->1 / benign->0); 0 = label-free")
    ap.add_argument("--gate-type", choices=["linear", "mlp"], default="mlp")
    ap.add_argument("--gate-hidden", type=int, default=256)
    ap.add_argument("--no-gate", action="store_true",
                    help="ablation: disable the router, force g=1 so the LoRA is applied "
                         "UNCONDITIONALLY (no gate, no BCE/L1). --late-from 0 gives the paper's "
                         "'Full' variant; the model's usual --late-from gives 'Always-on'.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fail-on-skip", action="store_true",
                    help="fail without saving an adapter when any input or CUDA OOM is skipped")
    ap.add_argument("--summary", default=None,
                    help="write the training JSON summary to this path")
    a = ap.parse_args()
    random.seed(a.seed); torch.manual_seed(a.seed)

    adapter = registry.get(a.model)
    adapter.set_prompt_mode(a.prompt_mode)
    model, proc = adapter.load(a.device)        # sharded when device is auto/0,1
    for p in model.parameters():
        p.requires_grad_(False)

    layers = adapter.decoder_layers(model)
    n_layers = len(layers)
    head = adapter.lm_head(model)
    D = head.in_features if hasattr(head, "in_features") else model.config.hidden_size
    late = list(range(a.late_from, n_layers))
    if a.gate_layer >= n_layers or late[0] >= n_layers:
        raise ValueError(f"gate_layer/late out of range for {n_layers}-layer model")

    gate_dev = dev_of(layers[a.gate_layer])
    devices = {l: dev_of(layers[l]) for l in late}
    print(f"[INFO] model={a.model} hidden={D} layers={n_layers} gate@L{a.gate_layer}({gate_dev}) "
          f"late={late[0]}-{late[-1]} rank={a.rank}")
    print(f"[INFO] late-layer devices: {sorted({str(d) for d in devices.values()})}")

    # Match the base model's dtype instead of assuming bf16: VITA-1.5 loads its
    # backbone in fp16 (its builder hardcodes it), and a bf16 gate/LoRA against fp16
    # activations dies with "mat1 and mat2 must have the same dtype".
    param_dtype = next(layers[a.gate_layer].parameters()).dtype
    print(f"[INFO] trainable dtype: {param_dtype} (matched to the base model)")

    # router on the gate layer's device
    if a.gate_type == "mlp":
        router = nn.Sequential(nn.Linear(D, a.gate_hidden), nn.ReLU(),
                               nn.Linear(a.gate_hidden, 1)).to(gate_dev, param_dtype)
        nn.init.normal_(router[0].weight, std=(2.0 / D) ** 0.5); nn.init.zeros_(router[0].bias)
        nn.init.normal_(router[2].weight, std=1e-2); nn.init.zeros_(router[2].bias)
    else:
        router = nn.Linear(D, 1).to(gate_dev, param_dtype)
        nn.init.zeros_(router.bias); nn.init.normal_(router.weight, std=1e-3)
    # each LoRA pair on its own layer's device
    A = nn.ModuleDict({str(l): nn.Linear(D, a.rank, bias=False).to(devices[l], param_dtype) for l in late})
    B = nn.ModuleDict({str(l): nn.Linear(a.rank, D, bias=False).to(devices[l], param_dtype) for l in late})
    for l in late:
        nn.init.normal_(A[str(l)].weight, std=1.0 / a.rank)
        nn.init.zeros_(B[str(l)].weight)            # start as no-op

    STATE = {"prompt_len": None, "g": None}

    def gate_hook(m, i, o):
        h = o[0] if isinstance(o, tuple) else o
        pos = STATE["prompt_len"] - 1
        if a.no_gate:
            STATE["g"] = torch.ones(h.shape[0], device=gate_dev, dtype=h.dtype)
        else:
            STATE["g"] = torch.sigmoid(router(h[:, pos, :].to(gate_dev))).squeeze(-1)
        return o

    def make_late(l):
        def hook(m, i, o):
            if STATE["g"] is None:
                return o
            h = o[0] if isinstance(o, tuple) else o
            g = STATE["g"].to(h.device).view(-1, 1, 1).to(h.dtype)   # g may live on another card
            add = B[str(l)](A[str(l)](h))
            h = h + g * add
            return (h,) + tuple(o[1:]) if isinstance(o, tuple) else h
        return hook

    layers[a.gate_layer].register_forward_hook(gate_hook)
    for l in late:
        layers[l].register_forward_hook(make_late(l))

    params = list(A.parameters()) + list(B.parameters())
    if not a.no_gate:
        params = list(router.parameters()) + params
    opt = torch.optim.AdamW(params, lr=a.lr)
    model.config.use_cache = False
    model.gradient_checkpointing_enable(); model.enable_input_require_grads()

    in_dev = model.get_input_embeddings().weight.device
    examples = jl(a.data)
    summary_path = Path(a.summary) if a.summary else Path(f"{a.out_dir}.summary.json")
    summary = {
        "expected_examples": len(examples),
        "epochs": a.epochs,
        "epochs_completed": 0,
        "expected_processed_steps": len(examples) * a.epochs,
        "processed_steps": 0,
        "input_skips": 0,
        "oom_skips": 0,
        "finite_loss": True,
        "adapter_saved": False,
    }
    print(f"[INFO] examples={len(examples)} trainable={sum(p.numel() for p in params):,} input_dev={in_dev}")

    try:
        for ep in range(a.epochs):
            random.shuffle(examples)
            opt.zero_grad(); run = gsum = gh = gb = 0.0; nh = nb = 0
            successful_in_epoch = 0
            for i, ex in enumerate(examples):
                try:
                    apath = resolve_audio(ex["audio"])
                    batch, plen = adapter.build_training_inputs(
                        model, proc, apath, ex["prompt"], ex["target"], in_dev)
                except Exception as e:
                    if record_input_skip(summary, e) == "oom":
                        opt.zero_grad(set_to_none=True)
                        successful_in_epoch = 0
                        torch.cuda.empty_cache()
                    print("[skip]", ex.get("audio"), e); continue
                STATE["prompt_len"] = plen; STATE["g"] = None
                try:
                    out = model(**batch)
                    if not torch.isfinite(out.loss).all():
                        summary["finite_loss"] = False
                        raise RuntimeError("training observed a non-finite model loss")
                    g = STATE["g"].float().mean()
                    loss = out.loss / a.grad_accum
                    if not a.no_gate:
                        # supervised BCE pulls the gate to the example's label (attack->1/benign->0);
                        # gate_l1 keeps it sparse. Both live on the gate device, then move to loss device.
                        gpen = a.gate_l1 * g
                        if a.gate_bce > 0:
                            tgt = torch.tensor(1.0 if ex.get("kind") == "harmful" else 0.0, device=g.device)
                            gpen = gpen + a.gate_bce * F.binary_cross_entropy(g.clamp(1e-4, 1 - 1e-4), tgt)
                        loss = loss + gpen.to(out.loss.device) / a.grad_accum
                    if not torch.isfinite(loss).all():
                        summary["finite_loss"] = False
                        raise RuntimeError("training observed a non-finite total loss")
                    loss.backward()
                except torch.cuda.OutOfMemoryError:
                    # long audio -> big logits; drop this example rather than crash the run
                    opt.zero_grad(set_to_none=True); STATE["g"] = None
                    successful_in_epoch = 0
                    del batch
                    torch.cuda.empty_cache()
                    summary["oom_skips"] += 1
                    continue
                summary["processed_steps"] += 1
                successful_in_epoch += 1
                run += out.loss.item(); gsum += g.item()
                if ex.get("kind") == "harmful":
                    gh += g.item(); nh += 1
                else:
                    gb += g.item(); nb += 1
                if optimizer_step_due(successful_in_epoch, a.grad_accum):
                    torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); opt.zero_grad()
            if optimizer_step_due(successful_in_epoch, a.grad_accum, end_of_epoch=True):
                torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); opt.zero_grad()
            summary["epochs_completed"] += 1
            print(f"[ep {ep+1}/{a.epochs}] loss={run/len(examples):.4f} gate_mean={gsum/len(examples):.3f} "
                  f"| gate_harmful={gh/max(nh,1):.3f} gate_benign={gb/max(nb,1):.3f} "
                  f"input_skipped={summary['input_skips']} oom_skipped={summary['oom_skips']}")
    except Exception:
        summary["adapter_saved"] = False
        write_summary_atomic(summary_path, summary)
        raise

    out = finalize_adapter(
        a.out_dir, summary_path, router, A, B, a, late, D, summary, a.fail_on_skip)
    print(f"[OK] saved adapter to {out}")


if __name__ == "__main__":
    main()
