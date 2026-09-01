#!/usr/bin/env python3
"""Construct one real Bernini component without running inference."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch


def parameter_report(module) -> dict[str, int]:
    parameters = list(module.parameters())
    return {
        "parameters": sum(parameter.numel() for parameter in parameters),
        "meta_parameters": sum(parameter.numel() for parameter in parameters if parameter.is_meta),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--comfy", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("component", choices=("planner", "t5", "wan_high", "wan_low"))
    args = parser.parse_args()

    # ComfyUI parses global argv when imported. Its parser must not see this
    # diagnostic tool's arguments.
    sys.argv = [sys.argv[0]]
    sys.path.insert(0, str(args.comfy.resolve()))
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

    if args.component == "planner":
        from bernini_v2.runtime import load_planner_runtime

        runtime = load_planner_runtime(args.checkpoint, dtype=torch.bfloat16)
        report = {
            "language": parameter_report(runtime.language_model),
            "vision": parameter_report(runtime.vision_model),
            "aux": parameter_report(runtime.aux),
            "tokenizer": type(runtime.tokenizer).__name__,
        }
    elif args.component == "t5":
        from bernini_v2.runtime import load_wan_t5

        clip = load_wan_t5(args.checkpoint)
        report = {
            "clip": type(clip).__name__,
            "patcher": type(clip.patcher).__name__,
            "model": type(clip.cond_stage_model).__name__,
        }
    else:
        import comfy.sd

        model = comfy.sd.load_diffusion_model(str(args.checkpoint), model_options={})
        report = {
            "patcher": type(model).__name__,
            "base_model": type(model.model).__name__,
            "diffusion_model": type(model.model.diffusion_model).__name__,
            **parameter_report(model.model.diffusion_model),
        }
    print(json.dumps({"component": args.component, **report}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
