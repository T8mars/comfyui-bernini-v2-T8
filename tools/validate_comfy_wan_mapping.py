#!/usr/bin/env python3
"""Validate renderer mapping against the current ComfyUI Wan module on meta."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bernini_v2.index import load_index  # noqa: E402
from bernini_v2.state_dict import Component, component_key  # noqa: E402
from bernini_v2.wan_mapping import wan_diffusers_to_comfy  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--comfy", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    args = parser.parse_args()

    sys.path.insert(0, str(args.comfy.resolve()))
    import comfy.ops  # noqa: PLC0415
    from comfy.ldm.wan.model import WanModel  # noqa: PLC0415

    model = WanModel(
        model_type="t2v",
        patch_size=(1, 2, 2),
        in_dim=16,
        dim=5120,
        ffn_dim=13824,
        freq_dim=256,
        text_dim=4096,
        out_dim=16,
        num_heads=40,
        num_layers=40,
        qk_norm=True,
        cross_attn_norm=True,
        eps=1e-6,
        operations=comfy.ops.disable_weight_init,
        device="meta",
        dtype=torch.bfloat16,
    )
    native_keys = set(model.state_dict())
    plan = load_index(args.source / "bernini" / "model.safetensors.index.json")

    result: dict[str, object] = {}
    failed = False
    for component in (Component.WAN_HIGH, Component.WAN_LOW):
        mapped = {wan_diffusers_to_comfy(component_key(source_key)[1]) for source_key in plan.by_component[component]}
        missing = sorted(native_keys - mapped)
        unexpected = sorted(mapped - native_keys)
        result[component.value] = {
            "source": len(plan.by_component[component]),
            "mapped": len(mapped),
            "native": len(native_keys),
            "missing": missing,
            "unexpected": unexpected,
        }
        failed |= bool(missing or unexpected)

    print(json.dumps(result, indent=2, sort_keys=True))
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
