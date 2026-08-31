#!/usr/bin/env python3
"""Generate six deterministic ComfyUI API workflows for Bernini v2.

The generated graphs intentionally use only ComfyUI Core nodes plus the nodes in
this package.  Input media names are placeholders and can be replaced by callers
or by ComfyUI's API workflow editor.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

REPACK_ROOT = "Bernini-v2-balanced-int8"
MANIFEST = f"{REPACK_ROOT}/repack-manifest.json"
HIGH_INDEX = f"{REPACK_ROOT}/wan_high/model.safetensors.index.json"
LOW_INDEX = f"{REPACK_ROOT}/wan_low/model.safetensors.index.json"
VAE_NAME = "wan_2.1_vae.safetensors"
DEFAULT_PROMPT = {
    "t2i": "A cinematic photograph of a red fox standing in fresh snow at sunrise.",
    "i2i": "Transform the source into a cinematic winter scene while preserving the subject.",
    "t2v": "A red fox walks through fresh snow at sunrise, cinematic camera movement.",
    "v2v": "Restyle the source video as a cinematic winter scene while preserving its motion.",
    "r2v": "The referenced subject walks through fresh snow at sunrise, cinematic camera movement.",
    "rv2v": "Replace the source-video subject with the referenced subject while preserving motion and framing.",
}
OFFICIAL_VIDEO_NEGATIVE = (
    "vivid tones, overexposed, static, blurry details, subtitles, style, artwork, painting, "
    "image, motionless, overall grayish, worst quality, low quality, JPEG compression artifacts, "
    "ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn face, deformed, disfigured, "
    "malformed limbs, fused fingers, still frame, cluttered background, three legs, too many people "
    "in the background, walking backwards"
)
OFFICIAL_DEFAULT_NEGATIVE = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，"
    "最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，"
    "画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，"
    "杂乱的背景，三条腿，背景人很多，倒着走"
)
SYSTEM_PROMPT = {
    "t2i": "You are a helpful assistant specialized in text-to-image generation.",
    "i2i": "You are a helpful assistant specialized in image editing.",
    "t2v": "You are a helpful assistant specialized in text-to-video generation.",
    "v2v": "You are a helpful assistant specialized in video editing.",
    "r2v": "You are a helpful assistant specialized in subject-to-video generation.",
    "rv2v": "You are a helpful assistant specialized in video editing with reference.",
}


Graph = dict[str, dict[str, Any]]


def _node(class_type: str, title: str, **inputs: Any) -> dict[str, Any]:
    return {"inputs": inputs, "class_type": class_type, "_meta": {"title": title}}


def build_workflow(task: str) -> Graph:
    if task not in DEFAULT_PROMPT:
        raise ValueError(f"unsupported task: {task}")

    prompt = DEFAULT_PROMPT[task]
    negative = OFFICIAL_VIDEO_NEGATIVE if task in {"v2v", "r2v"} else OFFICIAL_DEFAULT_NEGATIVE
    t5_prompt = SYSTEM_PROMPT[task] + prompt
    is_image = task in {"t2i", "i2i"}
    graph: Graph = {
        "1": _node("BerniniV2PlannerLoader", "Load Bernini v2 Planner", repack_manifest=MANIFEST, dtype="bfloat16"),
        "2": _node(
            "BerniniV2T5Loader",
            "Load Bernini v2 T5",
            repack_manifest=MANIFEST,
            dtype="bfloat16",
        ),
        "3": _node("CLIPTextEncode", "Positive UMT5", text=t5_prompt, clip=["2", 0]),
        "4": _node("CLIPTextEncode", "Negative UMT5", text=negative, clip=["2", 0]),
        "5": _node(
            "BerniniV2WanLoader",
            "Load High-noise Renderer",
            model_index=HIGH_INDEX,
            flow_shift=5.0,
            weight_dtype="bfloat16",
        ),
        "6": _node(
            "BerniniV2WanLoader",
            "Load Low-noise Renderer",
            model_index=LOW_INDEX,
            flow_shift=5.0,
            weight_dtype="bfloat16",
        ),
        "7": _node("VAELoader", "Load Wan VAE", vae_name=VAE_NAME),
    }

    plan_inputs: dict[str, Any] = {
        "planner": ["1", 0],
        "positive": ["3", 0],
        "negative": ["4", 0],
        "prompt": prompt,
        "negative_prompt": negative,
        "task": task,
        "width": 512 if is_image else 640,
        "height": 512 if is_image else 368,
        "length": 1 if is_image else 33,
        "source_fps": 16.0,
        "use_task_defaults": True,
        "match_source_size": task in {"i2i", "v2v", "rv2v"},
        "max_media_size": 848,
        "planning_steps": 25,
        "vit_denoising_steps": 5,
        "vit_text_cfg": 1.2,
        "vit_image_cfg": 1.0,
        "seed": 42,
    }

    if task in {"i2i", "r2v", "rv2v"}:
        graph["8"] = _node("LoadImage", "Load Reference Image", image="bernini_reference.png")
        # V3 autogrow inputs are flattened for the API and rebuilt into
        # reference_images={reference_image_0: ...} by ComfyUI at execution.
        plan_inputs["reference_images.reference_image_0"] = ["8", 0]
    if task in {"v2v", "rv2v"}:
        graph["9"] = _node("LoadVideo", "Load Source Video", file="bernini_source.mp4")
        plan_inputs["video"] = ["9", 0]

    graph.update(
        {
            "11": _node("BerniniV2Plan", "Bernini v2 Plan", **plan_inputs),
            "12": _node(
                "BerniniV2RendererGuider",
                "Bernini v2 Dual-expert Guider",
                plan=["11", 0],
                high_noise_model=["5", 0],
                low_noise_model=["6", 0],
                vae=["7", 0],
                omega_video=1.25,
                omega_image=3.0,
                omega_text=4.0,
                omega_target=1.2,
                omega_scale=0.75,
                use_task_defaults=True,
                boundary=0.875,
                guidance_batch_size="auto",
                vae_encode_mode="auto",
            ),
            "13": _node(
                "BerniniV2Scheduler",
                "Bernini v2 UniPC Sigmas",
                plan=["11", 0],
                steps=40,
                flow_shift=5.0,
                use_task_defaults=True,
            ),
            "14": _node("RandomNoise", "Random Noise", noise_seed=42),
            "15": _node("BerniniV2UniPCSampler", "Bernini v2 Flow UniPC BH2"),
            "16": _node(
                "SamplerCustomAdvanced",
                "Sample Bernini v2",
                noise=["14", 0],
                guider=["12", 0],
                sampler=["15", 0],
                sigmas=["13", 0],
                latent_image=["12", 1],
            ),
            "17": _node("VAEDecode", "Decode Wan Latent", samples=["16", 0], vae=["7", 0]),
        }
    )
    if is_image:
        graph["18"] = _node("SaveImage", "Save Bernini Image", images=["17", 0], filename_prefix=f"Bernini-v2/{task}")
    else:
        graph["18"] = _node("CreateVideo", "Create 16 fps Video", images=["17", 0], fps=16.0, bit_depth=8)
        graph["19"] = _node(
            "SaveVideo",
            "Save Bernini Video",
            video=["18", 0],
            filename_prefix=f"video/Bernini-v2/{task}",
            format="mp4",
            **{"format.codec": "h264"},
        )
    return graph


def validate_graph(task: str, graph: Graph) -> None:
    """Validate graph references and task-specific media wiring."""
    for node_id, node in graph.items():
        if not isinstance(node.get("class_type"), str) or not isinstance(node.get("inputs"), dict):
            raise ValueError(f"node {node_id} is not a ComfyUI API node")
        for name, value in node["inputs"].items():
            if isinstance(value, list) and len(value) == 2 and isinstance(value[0], str):
                if value[0] not in graph:
                    raise ValueError(f"node {node_id}.{name} references missing node {value[0]}")
                if not isinstance(value[1], int) or value[1] < 0:
                    raise ValueError(f"node {node_id}.{name} has invalid output slot {value[1]!r}")

    plan = graph["11"]["inputs"]
    has_video = "video" in plan or "source_video" in plan
    has_image = "reference_images.reference_image_0" in plan
    expected = {
        "t2i": (False, False),
        "i2i": (False, True),
        "t2v": (False, False),
        "v2v": (True, False),
        "r2v": (False, True),
        "rv2v": (True, True),
    }[task]
    if (has_video, has_image) != expected:
        raise ValueError(f"{task} media wiring is {(has_video, has_image)}, expected {expected}")
    if graph["15"]["class_type"] != "BerniniV2UniPCSampler":
        raise ValueError("Bernini v2 examples must use the flow-prediction UniPC BH2 sampler")


def write_examples(output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    for task in DEFAULT_PROMPT:
        graph = build_workflow(task)
        validate_graph(task, graph)
        (output / f"{task}.json").write_text(
            json.dumps(graph, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "examples" / "api",
    )
    args = parser.parse_args()
    write_examples(args.output)


if __name__ == "__main__":
    main()
