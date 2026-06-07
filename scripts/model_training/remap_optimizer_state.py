"""Remap a Lightning checkpoint's optimizer state to the current model's
parameter order, so training can resume after a refactor that changed
``model.parameters()`` ordering (but not the parameter set/shapes).

Background
----------
Lightning restores model weights by *name* (order-independent) but restores
optimizer state by *integer index* into ``model.parameters()``. If the code
that writes a checkpoint registers parameters in a different order than the
code that later resumes it — e.g. a decoder ``nn.ModuleDict`` whose insertion
order shifted — the saved per-parameter Adam moments get paired with the wrong
current gradients and the first optimizer step dies with a shape error like::

    RuntimeError: The size of tensor a (2048) must match the size of tensor b (3)

This script rebuilds the *current* model to learn the canonical parameter
order, reads the *saved* order from the checkpoint's own ``state_dict`` key
order (which equals the registration order at save time), and rewrites
``optimizer_states`` so each parameter's moments line up with the current
order. Model weights, epoch, global_step and LR-scheduler state are preserved
verbatim, so the resumed run continues exactly where it left off.

If the parameter *set* (names/shapes) differs — a genuine architecture change —
a lossless remap is impossible; the script aborts and tells you to warm-start
from weights instead.

Usage
-----
    python scripts/model_training/remap_optimizer_state.py \
        --checkpoint outputs/.../resnet50-full/last.ckpt \
        --encoder resnet50 --pretext full \
        --output outputs/.../resnet50-full/last.remapped.ckpt

Then resume from the ``--output`` path (e.g. set ``resume_from`` to it, or
replace ``last.ckpt`` after backing the original up).
"""

from __future__ import annotations

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from augur.models.tile_level.tile_model import TileModel  # noqa: E402
from augur.utils.config import load_tile_model_config  # noqa: E402


def _build_model(encoder: str, pretexts: list[str]) -> TileModel:
    cfg = load_tile_model_config("configs/tile-model", encoder=encoder, pretexts=pretexts)
    return TileModel.from_config(cfg["params"])


def _current_param_order(encoder: str, pretexts: list[str]) -> list[str]:
    """Return the current model's parameter names in ``parameters()`` order.

    Aborts if the model has tied/shared parameters: ``state_dict()`` emits a
    name for every tied alias while ``parameters()`` / the optimizer dedup to
    one slot, which breaks the "state_dict param-subsequence == optimizer
    index order" invariant the remap relies on.
    """
    model = _build_model(encoder, pretexts)
    named = list(model.named_parameters())  # lazy params: names without shapes
    n_unique = len({id(p) for _, p in named})
    if n_unique != len(named):
        print(
            "ERROR: model has tied/shared parameters; the simple positional "
            "optimizer remap is unsafe. Aborting."
        )
        raise SystemExit(4)
    return [name for name, _ in named]


def _self_verify(output_path: str, encoder: str, pretexts: list[str]) -> None:
    """Load the remapped checkpoint into a fresh current-code model + optimizer,
    give every optimized parameter a zero gradient, and run one optimizer step.

    This exercises the exact index-pairing that failed at resume time, so a
    clean step proves the remapped checkpoint will resume without the shape
    crash — verified on the real checkpoint, before launching a training job.
    """
    model = _build_model(encoder, pretexts)
    ckpt = torch.load(output_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    opt_cfg = model.configure_optimizers()
    optimizer = opt_cfg["optimizer"] if isinstance(opt_cfg, dict) else opt_cfg
    optimizer.load_state_dict(ckpt["optimizer_states"][0])
    for p in (p for g in optimizer.param_groups for p in g["params"]):
        p.grad = torch.zeros_like(p)
    optimizer.step()
    print("Self-verify: optimizer.step() succeeded on the remapped checkpoint.")


def _saved_param_order(state_dict: dict, param_name_set: set[str]) -> list[str]:
    """Recover the save-time parameter order from the checkpoint state_dict.

    ``state_dict`` preserves insertion (== registration) order, and the
    subsequence of its keys that are parameters (not buffers) matches the
    optimizer's integer indexing at save time.
    """
    return [k for k in state_dict.keys() if k in param_name_set]


def remap_checkpoint(
    checkpoint_path: str,
    output_path: str,
    encoder: str,
    pretexts: list[str],
    verify: bool = True,
) -> None:
    if os.path.abspath(checkpoint_path) == os.path.abspath(output_path):
        raise ValueError("--output must differ from --checkpoint (never overwrite).")

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "optimizer_states" not in ckpt or not ckpt["optimizer_states"]:
        raise ValueError("Checkpoint has no optimizer_states to remap.")
    state_dict = ckpt["state_dict"]

    cur_order = _current_param_order(encoder, pretexts)
    cur_set = set(cur_order)
    name_to_new_idx = {name: i for i, name in enumerate(cur_order)}

    saved_order = _saved_param_order(state_dict, cur_set)
    saved_set = set(saved_order)

    # --- Bijection check: same parameter set, only order may differ. ---
    only_saved = saved_set - cur_set
    only_cur = cur_set - saved_set
    if only_saved or only_cur:
        print("ERROR: parameter sets differ between checkpoint and current model.")
        print(f"  in checkpoint but not current model: {len(only_saved)}")
        for k in sorted(only_saved)[:20]:
            print("    -", k)
        print(f"  in current model but not checkpoint: {len(only_cur)}")
        for k in sorted(only_cur)[:20]:
            print("    +", k)
        print(
            "\nA lossless optimizer remap is not possible (architecture changed). "
            "Warm-start from weights instead (load state_dict, fresh optimizer)."
        )
        raise SystemExit(2)

    # Multi-group guard: PyTorch assigns optimizer param-ids in flattened-
    # across-groups order, which equals registration (state_dict) order ONLY
    # for a single param_group. With 2+ groups (e.g. a decay/no-decay split)
    # the positional id->name decode below would be wrong and silently route
    # moments to the wrong parameters. Abort rather than miswire.
    multi_group = any(
        len(opt["param_groups"]) != 1 for opt in ckpt["optimizer_states"]
    )
    if multi_group:
        print(
            "ERROR: checkpoint optimizer has more than one param_group. The "
            "positional remap assumes a single group covering all parameters "
            "in registration order (it cannot recover multi-group save-time "
            "ordering from the checkpoint alone). Aborting to avoid a silent "
            "wrong remap."
        )
        raise SystemExit(5)

    n_opt = sum(len(g["params"]) for g in ckpt["optimizer_states"][0]["param_groups"])
    if n_opt != len(saved_order):
        # Some params may be excluded from the optimizer (e.g. frozen). The
        # invariant (optimizer index order == full registration order) then
        # fails, so the positional mapping would be off-by-N. This count guard
        # is load-bearing for the frozen-param case; abort rather than guess.
        print(
            f"WARNING: optimizer holds {n_opt} params but checkpoint state_dict "
            f"has {len(saved_order)} parameters. The simple positional mapping "
            "assumes the optimizer covers all parameters in registration order. "
            "Aborting to avoid a wrong remap."
        )
        raise SystemExit(3)

    # old optimizer index -> parameter name (save-time order)
    old_idx_to_name = {i: name for i, name in enumerate(saved_order)}

    # --- Remap every optimizer in optimizer_states ---
    n_optimizers = len(ckpt["optimizer_states"])
    if n_optimizers != 1:
        print(
            f"NOTE: {n_optimizers} optimizers found; remapping each with the same "
            "name-based permutation."
        )

    # The fixed checkpoint must look exactly like one the CURRENT code would
    # have written: optimizer state keyed by the current param index, and each
    # param_group's ``params`` listing those indices in the current
    # enumeration order (ascending new index). PyTorch's load_state_dict pairs
    # saved param-ids to current params *positionally* within each group, so
    # the ``params`` list order — not just the state keys — must be canonical.
    moved = 0
    for opt_state in ckpt["optimizer_states"]:
        old_state = opt_state["state"]
        new_state = {}
        for old_idx, st in old_state.items():
            name = old_idx_to_name[int(old_idx)]
            new_idx = name_to_new_idx[name]
            new_state[new_idx] = st
            if new_idx != int(old_idx):
                moved += 1
        opt_state["state"] = dict(sorted(new_state.items()))

        for group in opt_state["param_groups"]:
            group["params"] = sorted(
                name_to_new_idx[old_idx_to_name[int(old_idx)]]
                for old_idx in group["params"]
            )

    print(f"Remapped optimizer state: {len(saved_order)} params, "
          f"{moved} moved to a new index, {n_optimizers} optimizer(s).")
    print(f"epoch={ckpt.get('epoch')} global_step={ckpt.get('global_step')} "
          "(preserved).")

    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
    tmp = output_path + ".tmp"
    torch.save(ckpt, tmp)
    os.replace(tmp, output_path)
    print(f"Wrote remapped checkpoint: {output_path}")

    if verify:
        _self_verify(output_path, encoder, pretexts)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Path to the failing .ckpt")
    parser.add_argument("--output", required=True, help="Path for the remapped .ckpt")
    parser.add_argument("--encoder", default="resnet50")
    parser.add_argument(
        "--pretext", nargs="*", default=["full"],
        help="Pretext tokens used to build the model (e.g. full, or "
             "hematoxylin jigmag magnification).",
    )
    parser.add_argument(
        "--no-verify", action="store_true",
        help="Skip the post-remap dry optimizer step self-check.",
    )
    args = parser.parse_args()
    remap_checkpoint(
        args.checkpoint, args.output, args.encoder, list(args.pretext),
        verify=not args.no_verify,
    )


if __name__ == "__main__":
    main()
