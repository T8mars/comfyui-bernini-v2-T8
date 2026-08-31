#!/usr/bin/env python3
"""Generate editable ComfyUI frontend workflows for every Bernini v2 task."""

from __future__ import annotations

import argparse
import json
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    from tools.build_example_workflows import (
        DEFAULT_PROMPT,
        HIGH_INDEX,
        LOW_INDEX,
        MANIFEST,
        OFFICIAL_DEFAULT_NEGATIVE,
        OFFICIAL_VIDEO_NEGATIVE,
        SYSTEM_PROMPT,
        VAE_NAME,
    )
except ModuleNotFoundError:  # Direct execution sets tools/ as sys.path[0].
    from build_example_workflows import (
        DEFAULT_PROMPT,
        HIGH_INDEX,
        LOW_INDEX,
        MANIFEST,
        OFFICIAL_DEFAULT_NEGATIVE,
        OFFICIAL_VIDEO_NEGATIVE,
        SYSTEM_PROMPT,
        VAE_NAME,
    )

FRONTEND_VERSION = "1.49.6"
WORKFLOW_NAMESPACE = uuid.UUID("348d6715-a45a-43ca-b1c1-fd3654220ab2")


@dataclass
class FrontendNode:
    node_id: int
    node_type: str
    pos: tuple[int, int]
    size: tuple[int, int]
    order: int
    widgets: list[Any] = field(default_factory=list)
    inputs: list[dict[str, Any]] = field(default_factory=list)
    outputs: list[dict[str, Any]] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.node_id,
            "type": self.node_type,
            "pos": list(self.pos),
            "size": list(self.size),
            "flags": {},
            "order": self.order,
            "mode": 0,
            "inputs": self.inputs,
            "outputs": self.outputs,
            "properties": {"Node name for S&R": self.node_type},
            "widgets_values": self.widgets,
        }


class FrontendGraph:
    def __init__(self, task: str):
        self.task = task
        self.nodes: dict[int, FrontendNode] = {}
        self.links: list[list[Any]] = []
        self.next_link_id = 1

    def add(
        self,
        node_id: int,
        node_type: str,
        pos: tuple[int, int],
        size: tuple[int, int],
        widgets: list[Any],
        outputs: list[tuple[str, str]],
    ) -> FrontendNode:
        node = FrontendNode(node_id, node_type, pos, size, len(self.nodes), widgets)
        node.outputs = [{"name": name, "type": output_type, "links": None} for name, output_type in outputs]
        self.nodes[node_id] = node
        return node

    def connect(
        self,
        source_id: int,
        source_slot: int,
        target_id: int,
        input_name: str,
        input_type: str,
        *,
        label: str | None = None,
    ) -> int:
        link_id = self.next_link_id
        self.next_link_id += 1
        target = self.nodes[target_id]
        target_slot = len(target.inputs)
        input_data: dict[str, Any] = {"name": input_name, "type": input_type, "link": link_id}
        if label is not None:
            input_data["label"] = label
            input_data["localized_name"] = label
        target.inputs.append(input_data)
        output = self.nodes[source_id].outputs[source_slot]
        if output["links"] is None:
            output["links"] = []
        output["links"].append(link_id)
        self.links.append([link_id, source_id, source_slot, target_id, target_slot, input_type])
        return link_id

    def serialize(self) -> dict[str, Any]:
        is_image = self.task in {"t2i", "i2i"}
        return {
            "id": str(uuid.uuid5(WORKFLOW_NAMESPACE, self.task)),
            "revision": 0,
            "last_node_id": max(self.nodes),
            "last_link_id": self.next_link_id - 1,
            "nodes": [node.as_dict() for node in self.nodes.values()],
            "links": self.links,
            "groups": [
                {
                    "id": 1,
                    "title": "1. BF16 model loaders",
                    "bounding": [-40, -80, 430, 850],
                    "color": "#3f789e",
                    "flags": {},
                },
                {
                    "id": 2,
                    "title": "2. Prompts and source media",
                    "bounding": [390, -80, 520, 850],
                    "color": "#3f789e",
                    "flags": {},
                },
                {
                    "id": 3,
                    "title": "3. Bernini semantic plan",
                    "bounding": [910, -80, 480, 850],
                    "color": "#3f789e",
                    "flags": {},
                },
                {
                    "id": 4,
                    "title": "4. Flow UniPC render and save",
                    "bounding": [1390, -80, 850, 970 if not is_image else 760],
                    "color": "#3f789e",
                    "flags": {},
                },
            ],
            "config": {},
            "extra": {
                "ds": {"scale": 0.62, "offset": [70, 120]},
                "frontendVersion": FRONTEND_VERSION,
                "info": {
                    "name": f"Bernini v2 {self.task.upper()}",
                    "author": "ComfyUI-BerniniV2",
                    "description": "Official task defaults, BF16 weights, and flow-prediction UniPC BH2.",
                },
            },
            "version": 0.4,
        }


def _negative_for_task(task: str) -> str:
    return OFFICIAL_VIDEO_NEGATIVE if task in {"v2v", "r2v"} else OFFICIAL_DEFAULT_NEGATIVE


def build_frontend_workflow(task: str) -> dict[str, Any]:
    if task not in DEFAULT_PROMPT:
        raise ValueError(f"unsupported task: {task}")
    graph = FrontendGraph(task)
    prompt = DEFAULT_PROMPT[task]
    negative = _negative_for_task(task)
    t5_prompt = SYSTEM_PROMPT[task] + prompt
    image_task = task in {"t2i", "i2i"}

    graph.add(
        1, "BerniniV2PlannerLoader", (0, 0), (350, 110), [MANIFEST, "bfloat16"], [("planner", "BERNINI_V2_PLANNER")]
    )
    graph.add(2, "BerniniV2T5Loader", (0, 150), (350, 110), [MANIFEST, "bfloat16"], [("CLIP", "CLIP")])
    graph.add(5, "BerniniV2WanLoader", (0, 300), (350, 130), [HIGH_INDEX, 5.0, "bfloat16"], [("MODEL", "MODEL")])
    graph.add(6, "BerniniV2WanLoader", (0, 470), (350, 130), [LOW_INDEX, 5.0, "bfloat16"], [("MODEL", "MODEL")])
    graph.add(7, "VAELoader", (0, 640), (350, 90), [VAE_NAME], [("VAE", "VAE")])
    graph.add(3, "CLIPTextEncode", (420, 0), (460, 190), [t5_prompt], [("CONDITIONING", "CONDITIONING")])
    graph.add(4, "CLIPTextEncode", (420, 230), (460, 220), [negative], [("CONDITIONING", "CONDITIONING")])

    if task in {"i2i", "r2v", "rv2v"}:
        graph.add(
            8,
            "LoadImage",
            (420, 500),
            (340, 260),
            ["bernini_reference.png", "image"],
            [("IMAGE", "IMAGE"), ("MASK", "MASK")],
        )
    if task in {"v2v", "rv2v"}:
        graph.add(9, "LoadVideo", (780, 500), (340, 120), ["bernini_source.mp4"], [("VIDEO", "VIDEO")])
        graph.add(
            10,
            "GetVideoComponents",
            (780, 660),
            (340, 150),
            [],
            [("images", "IMAGE"), ("audio", "AUDIO"), ("fps", "FLOAT"), ("bit_depth", "INT"), ("color_space", "COMBO")],
        )

    graph.add(
        11,
        "BerniniV2Plan",
        (950, 0),
        (410, 720),
        [
            prompt,
            negative,
            task,
            512 if image_task else 848,
            512 if image_task else 480,
            1 if image_task else 81,
            16.0,
            True,
            task in {"i2i", "v2v", "rv2v"},
            848,
            25,
            5,
            1.2,
            1.0,
            42,
        ],
        [("plan", "BERNINI_V2_PLAN")],
    )
    graph.add(
        12,
        "BerniniV2RendererGuider",
        (1430, 0),
        (360, 390),
        [1.25, 3.0, 4.0, 1.2, 0.75, True, 0.875],
        [("guider", "GUIDER"), ("latent", "LATENT")],
    )
    graph.add(13, "BerniniV2Scheduler", (1430, 430), (360, 180), [40, 5.0, True], [("SIGMAS", "SIGMAS")])
    graph.add(14, "RandomNoise", (1430, 650), (300, 100), [42, "fixed"], [("NOISE", "NOISE")])
    graph.add(15, "BerniniV2UniPCSampler", (1430, 790), (300, 70), [], [("SAMPLER", "SAMPLER")])
    graph.add(
        16, "SamplerCustomAdvanced", (1850, 0), (320, 240), [], [("output", "LATENT"), ("denoised_output", "LATENT")]
    )
    graph.add(17, "VAEDecode", (1850, 290), (300, 90), [], [("IMAGE", "IMAGE")])
    if image_task:
        graph.add(18, "SaveImage", (1850, 430), (320, 250), [f"Bernini-v2/{task}"], [("images", "IMAGE")])
    else:
        graph.add(18, "CreateVideo", (1850, 430), (320, 130), [16.0, 8], [("VIDEO", "VIDEO")])
        graph.add(19, "SaveVideo", (1850, 610), (320, 250), [f"video/Bernini-v2/{task}", "mp4", "h264"], [])

    graph.connect(2, 0, 3, "clip", "CLIP")
    graph.connect(2, 0, 4, "clip", "CLIP")
    if task in {"v2v", "rv2v"}:
        graph.connect(9, 0, 10, "video", "VIDEO")
    graph.connect(1, 0, 11, "planner", "BERNINI_V2_PLANNER")
    graph.connect(3, 0, 11, "positive", "CONDITIONING")
    graph.connect(4, 0, 11, "negative", "CONDITIONING")
    if task in {"v2v", "rv2v"}:
        graph.connect(10, 0, 11, "source_video", "IMAGE")
    if task in {"i2i", "r2v", "rv2v"}:
        graph.connect(
            8,
            0,
            11,
            "reference_images.reference_image_0",
            "IMAGE",
            label="reference image 1",
        )
    graph.connect(11, 0, 12, "plan", "BERNINI_V2_PLAN")
    graph.connect(5, 0, 12, "high_noise_model", "MODEL")
    graph.connect(6, 0, 12, "low_noise_model", "MODEL")
    graph.connect(7, 0, 12, "vae", "VAE")
    graph.connect(11, 0, 13, "plan", "BERNINI_V2_PLAN")
    graph.connect(14, 0, 16, "noise", "NOISE")
    graph.connect(12, 0, 16, "guider", "GUIDER")
    graph.connect(15, 0, 16, "sampler", "SAMPLER")
    graph.connect(13, 0, 16, "sigmas", "SIGMAS")
    graph.connect(12, 1, 16, "latent_image", "LATENT")
    graph.connect(16, 0, 17, "samples", "LATENT")
    graph.connect(7, 0, 17, "vae", "VAE")
    graph.connect(17, 0, 18, "images", "IMAGE")
    if not image_task:
        graph.connect(18, 0, 19, "video", "VIDEO")
    return graph.serialize()


def validate_frontend_workflow(task: str, workflow: dict[str, Any]) -> None:
    nodes = {node["id"]: node for node in workflow["nodes"]}
    link_ids = {link[0] for link in workflow["links"]}
    if len(link_ids) != len(workflow["links"]):
        raise ValueError(f"{task}: duplicate link ids")
    for link_id, source_id, source_slot, target_id, target_slot, link_type in workflow["links"]:
        if source_id not in nodes or target_id not in nodes:
            raise ValueError(f"{task}: link {link_id} references a missing node")
        source = nodes[source_id]
        target = nodes[target_id]
        if source_slot >= len(source["outputs"]) or target_slot >= len(target["inputs"]):
            raise ValueError(f"{task}: link {link_id} has an invalid slot")
        if source["outputs"][source_slot]["type"] != link_type or target["inputs"][target_slot]["type"] != link_type:
            raise ValueError(f"{task}: link {link_id} type mismatch")
        if target["inputs"][target_slot]["link"] != link_id:
            raise ValueError(f"{task}: target input does not point back to link {link_id}")
    expected_media = {
        "t2i": set(),
        "i2i": {"LoadImage"},
        "t2v": set(),
        "v2v": {"LoadVideo", "GetVideoComponents"},
        "r2v": {"LoadImage"},
        "rv2v": {"LoadImage", "LoadVideo", "GetVideoComponents"},
    }[task]
    present_media = {node["type"] for node in nodes.values()} & {"LoadImage", "LoadVideo", "GetVideoComponents"}
    if present_media != expected_media:
        raise ValueError(f"{task}: media nodes {present_media}, expected {expected_media}")
    required = {
        "BerniniV2Plan",
        "BerniniV2RendererGuider",
        "BerniniV2Scheduler",
        "BerniniV2UniPCSampler",
        "SamplerCustomAdvanced",
    }
    missing = required - {node["type"] for node in nodes.values()}
    if missing:
        raise ValueError(f"{task}: missing nodes {sorted(missing)}")


def write_frontend_workflows(output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    for task in DEFAULT_PROMPT:
        workflow = build_frontend_workflow(task)
        validate_frontend_workflow(task, workflow)
        (output / f"{task}.json").write_text(
            json.dumps(workflow, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "examples" / "workflows",
    )
    args = parser.parse_args()
    write_frontend_workflows(args.output)


if __name__ == "__main__":
    main()
