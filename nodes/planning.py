"""Comfy node for native Bernini v2 semantic planning."""

from __future__ import annotations

from comfy_api.latest import ComfyExtension, io
from typing_extensions import override

from ..bernini_v2.media import fit_media_size
from ..bernini_v2.planner import create_plan, split_reference_images
from ..bernini_v2.presets import task_preset
from .loaders import BerniniV2PlannerType

BerniniV2PlanType = io.Custom("BERNINI_V2_PLAN")


class BerniniV2PlanNode(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="BerniniV2Plan",
            display_name="Bernini v2 Plan",
            category="conditioning/bernini_v2",
            description="Runs the native Qwen/VIT semantic planner and produces four renderer condition arms.",
            inputs=[
                BerniniV2PlannerType.Input("planner"),
                io.Conditioning.Input("positive"),
                io.Conditioning.Input("negative"),
                io.String.Input("prompt", multiline=True, dynamic_prompts=True),
                io.String.Input("negative_prompt", multiline=True, dynamic_prompts=True, advanced=True),
                io.Combo.Input("task", options=["t2i", "i2i", "t2v", "v2v", "r2v", "rv2v"], default="t2v"),
                io.Int.Input("width", default=848, min=16, max=8192, step=16),
                io.Int.Input("height", default=480, min=16, max=8192, step=16),
                io.Int.Input("length", default=33, min=1, max=8192, step=4),
                io.Image.Input("source_video", optional=True),
                io.Video.Input("video", optional=True, advanced=True),
                io.Autogrow.Input(
                    "reference_images",
                    optional=True,
                    template=io.Autogrow.TemplatePrefix(
                        input=io.Image.Input("reference_image"),
                        prefix="reference_image_",
                        min=0,
                        max=8,
                    ),
                ),
                io.Float.Input("source_fps", default=16.0, min=0.01, max=240.0, step=0.01, advanced=True),
                io.Boolean.Input("use_task_defaults", default=True, advanced=True),
                io.Boolean.Input("match_source_size", default=True, advanced=True),
                io.Int.Input("max_media_size", default=848, min=240, max=8192, step=16, advanced=True),
                io.Int.Input("planning_steps", default=25, min=1, max=100, advanced=True),
                io.Int.Input("vit_denoising_steps", default=5, min=1, max=100, advanced=True),
                io.Float.Input("vit_text_cfg", default=1.2, min=0.0, max=20.0, step=0.05, advanced=True),
                io.Float.Input("vit_image_cfg", default=1.0, min=0.0, max=20.0, step=0.05, advanced=True),
                io.Int.Input("seed", default=42, min=0, max=0xFFFFFFFFFFFFFFFF),
            ],
            outputs=[BerniniV2PlanType.Output(display_name="plan")],
        )

    @classmethod
    def execute(
        cls,
        planner,
        positive,
        negative,
        prompt,
        negative_prompt,
        task,
        width,
        height,
        length,
        source_video=None,
        video=None,
        reference_images=None,
        source_fps=16.0,
        use_task_defaults=True,
        match_source_size=True,
        max_media_size=848,
        planning_steps=25,
        vit_denoising_steps=5,
        vit_text_cfg=1.2,
        vit_image_cfg=1.0,
        seed=42,
    ) -> io.NodeOutput:
        if video is not None:
            if source_video is not None:
                raise ValueError("connect either source_video IMAGE batch or VIDEO, not both")
            components = video.get_components()
            source_video = components.images
            source_fps = float(components.frame_rate)
        references = split_reference_images(reference_images)
        if use_task_defaults:
            preset = task_preset(task)
            planning_steps = preset["planning_steps"]
            vit_denoising_steps = preset["vit_denoising_steps"]
            max_media_size = preset["max_media_size"]
        size_source = source_video
        if task == "i2i" and references:
            size_source = references[0]
        if match_source_size and size_source is not None:
            height, width = fit_media_size(
                size_source.shape[1],
                size_source.shape[2],
                max_size=max_media_size,
            )
        plan = create_plan(
            planner,
            positive=positive,
            negative=negative,
            prompt=prompt,
            negative_prompt=negative_prompt,
            task=task,
            width=width,
            height=height,
            length=length,
            max_media_size=max_media_size,
            source_video=source_video,
            reference_images=references,
            source_fps=source_fps,
            planning_steps=planning_steps,
            vit_denoising_steps=vit_denoising_steps,
            vit_text_cfg=vit_text_cfg,
            vit_image_cfg=vit_image_cfg,
            seed=seed,
        )
        return io.NodeOutput(plan)


class BerniniV2PlanningExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [BerniniV2PlanNode]
