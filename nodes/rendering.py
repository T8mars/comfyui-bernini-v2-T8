"""Native Comfy renderer conditioning and guidance for Bernini v2."""

from __future__ import annotations

import comfy.model_management
import comfy.sampler_helpers
import comfy.samplers
import torch
import torch.nn.functional as F
from comfy_api.latest import ComfyExtension, io
from typing_extensions import override

from ..bernini_v2.guidance import compose_denoised_guidance, unipc_flow_sigmas
from ..bernini_v2.media import fit_media_size, ordered_renderer_sources
from ..bernini_v2.planner import BerniniV2Plan
from ..bernini_v2.presets import task_preset
from ..bernini_v2.unipc import sample_flow_unipc_bh2
from .planning import BerniniV2PlanType


def _resize_source_media(image: torch.Tensor, max_size: int) -> torch.Tensor:
    resized_height, resized_width = fit_media_size(
        image.shape[1],
        image.shape[2],
        max_size=max_size,
    )
    channels_first = image[..., :3].movedim(-1, 1).float()
    if channels_first.shape[-2:] != (resized_height, resized_width):
        channels_first = F.interpolate(
            channels_first,
            size=(resized_height, resized_width),
            mode="bicubic",
            align_corners=False,
            antialias=True,
        )
    return channels_first.movedim(1, -1)


def encode_renderer_sources(
    plan: BerniniV2Plan,
    vae,
) -> tuple[list[torch.Tensor], list[torch.Tensor], dict[str, torch.Tensor]]:
    """VAE encode source video and reference images as separate Wan streams."""
    video_latents = []
    if plan.source_video is not None:
        video = _resize_source_media(
            plan.source_video[: plan.length],
            plan.max_media_size,
        )
        video_latents.append(vae.encode(video))

    image_latents = []
    for image in plan.reference_images:
        image_latents.append(vae.encode(_resize_source_media(image[:1], plan.max_media_size)))

    latent = torch.zeros(
        [1, 16, ((plan.length - 1) // 4) + 1, plan.height // 8, plan.width // 8],
        device=comfy.model_management.intermediate_device(),
    )
    return video_latents, image_latents, {"samples": latent}


def _conditioning(context: torch.Tensor, context_latents: list[torch.Tensor]):
    metadata = {}
    if context_latents:
        metadata["context_latents"] = context_latents
    return [[context, metadata]]


class BerniniV2Guider(comfy.samplers.CFGGuider):
    """Four/five-arm guider matching the released Bernini v2 renderer."""

    def __init__(
        self,
        model_patcher,
        plan: BerniniV2Plan,
        video_latents: list[torch.Tensor],
        image_latents: list[torch.Tensor],
        *,
        omega_video: float,
        omega_image: float,
        omega_text: float,
        omega_target: float,
        scale: float,
    ):
        super().__init__(model_patcher)
        self.omega_video = omega_video
        self.omega_image = omega_image
        self.omega_text = omega_text
        self.omega_target = omega_target
        self.scale = scale
        self.rv2v = plan.task == "rv2v"

        # The released renderer appends reference-image VAE tokens before
        # source-video tokens. Wan assigns source ids/RoPE by list position, so
        # reversing these streams corrupts only the combined RV2V condition.
        all_sources = ordered_renderer_sources(
            image_sources=image_latents,
            video_sources=video_latents,
        )
        conditions = {
            "base": _conditioning(plan.contexts["wotxt_wovit"], []),
            "text": _conditioning(plan.contexts["wtxt_wovit"], all_sources),
            "target": _conditioning(plan.contexts["wtxt_wvit"], all_sources),
        }
        if self.rv2v:
            conditions["video"] = _conditioning(plan.contexts["wotxt_wovit"], video_latents)
            conditions["image"] = _conditioning(plan.contexts["wotxt_wovit"], all_sources)
        elif all_sources:
            conditions["source"] = _conditioning(plan.contexts["wotxt_wovit"], all_sources)
        self.arm_names = list(conditions)
        self.inner_set_conds(conditions)

    def _predict_arms(self, inner_model, conditions, x, timestep, model_options, *, scale):
        active_conditions = [conditions[name] for name in self.arm_names]
        outputs = comfy.samplers.calc_cond_batch(
            inner_model,
            active_conditions,
            x,
            timestep,
            model_options,
        )
        predictions = dict(zip(self.arm_names, outputs, strict=True))
        return compose_denoised_guidance(
            predictions,
            x,
            timestep,
            omega_video=self.omega_video * scale,
            omega_image=self.omega_image * scale,
            omega_text=self.omega_text * scale,
            omega_target=self.omega_target * scale,
            rv2v=self.rv2v,
        )

    def predict_noise(self, x, timestep, model_options=None, seed=None):
        del seed
        model_options = model_options or {}
        return self._predict_arms(
            self.inner_model,
            self.conds,
            x,
            timestep,
            model_options,
            scale=self.scale,
        )


class BerniniV2DualExpertGuider(BerniniV2Guider):
    """One guider that switches Wan experts without resetting sampler history."""

    def __init__(
        self,
        high_noise_model,
        low_noise_model,
        *args,
        boundary: float,
        omega_scale: float,
        **kwargs,
    ):
        super().__init__(high_noise_model, *args, scale=1.0, **kwargs)
        self.low_noise_model = low_noise_model
        self.boundary = boundary
        self.omega_scale = omega_scale
        self.low_inner = None
        self.low_conds = None
        self.low_loaded_models = []
        self.switched = False

    def outer_sample(
        self,
        noise,
        latent_image,
        sampler,
        sigmas,
        denoise_mask=None,
        callback=None,
        disable_pbar=False,
        seed=None,
        latent_shapes=None,
    ):
        low_conditions = {name: [condition.copy() for condition in self.conds[name]] for name in self.arm_names}
        self.low_inner, self.low_conds, self.low_loaded_models = comfy.sampler_helpers.prepare_sampling(
            self.low_noise_model,
            noise.shape,
            low_conditions,
            self.low_noise_model.model_options,
        )
        self.low_noise_model.pre_run()
        self.switched = False
        try:
            return super().outer_sample(
                noise,
                latent_image,
                sampler,
                sigmas,
                denoise_mask,
                callback,
                disable_pbar,
                seed,
                latent_shapes=latent_shapes,
            )
        finally:
            self.low_noise_model.cleanup()
            comfy.sampler_helpers.cleanup_models(self.low_conds, self.low_loaded_models)
            self.low_inner = None
            self.low_conds = None
            self.low_loaded_models = []

    def inner_sample(
        self,
        noise,
        latent_image,
        device,
        sampler,
        sigmas,
        denoise_mask,
        callback,
        disable_pbar,
        seed,
        latent_shapes=None,
    ):
        self.low_inner.latent_shapes = latent_shapes
        low_latent = latent_image
        if low_latent is not None and torch.count_nonzero(low_latent) > 0:
            low_latent = self.low_inner.process_latent_in(low_latent)
        self.low_conds = comfy.samplers.process_conds(
            self.low_inner,
            noise,
            self.low_conds,
            device,
            low_latent,
            denoise_mask,
            seed,
            latent_shapes=latent_shapes,
        )
        return super().inner_sample(
            noise,
            latent_image,
            device,
            sampler,
            sigmas,
            denoise_mask,
            callback,
            disable_pbar,
            seed,
            latent_shapes=latent_shapes,
        )

    def predict_noise(self, x, timestep, model_options=None, seed=None):
        del seed
        model_options = model_options or {}
        use_low_noise = bool((timestep[0] < self.boundary).item())
        if use_low_noise:
            if not self.switched:
                memory_required, minimum_memory_required = comfy.sampler_helpers.estimate_memory(
                    self.low_noise_model,
                    x.shape,
                    self.low_conds,
                )
                comfy.model_management.load_models_gpu(
                    [self.low_noise_model, *self.low_loaded_models],
                    memory_required=memory_required,
                    minimum_memory_required=minimum_memory_required,
                )
                self.switched = True
            return self._predict_arms(
                self.low_inner,
                self.low_conds,
                x,
                timestep,
                model_options,
                scale=self.omega_scale,
            )
        return self._predict_arms(
            self.inner_model,
            self.conds,
            x,
            timestep,
            model_options,
            scale=1.0,
        )


class BerniniV2RendererGuider(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="BerniniV2RendererGuider",
            display_name="Bernini v2 Renderer Guider",
            category="sampling/bernini_v2",
            description=(
                "VAE-encodes source media and creates one native Wan guider that switches "
                "experts at the official boundary without resetting UniPC sampler history."
            ),
            inputs=[
                BerniniV2PlanType.Input("plan"),
                io.Model.Input("high_noise_model"),
                io.Model.Input("low_noise_model"),
                io.Vae.Input("vae"),
                io.Float.Input("omega_video", default=1.25, min=0.0, max=20.0, step=0.05),
                io.Float.Input("omega_image", default=3.0, min=0.0, max=20.0, step=0.05),
                io.Float.Input("omega_text", default=4.0, min=0.0, max=20.0, step=0.05),
                io.Float.Input("omega_target", default=1.2, min=0.0, max=20.0, step=0.05),
                io.Float.Input("omega_scale", default=0.75, min=0.0, max=2.0, step=0.05),
                io.Boolean.Input("use_task_defaults", default=True, advanced=True),
                io.Float.Input("boundary", default=0.875, min=0.0, max=1.0, step=0.001, advanced=True),
            ],
            outputs=[
                io.Guider.Output(display_name="guider"),
                io.Latent.Output(display_name="latent"),
            ],
        )

    @classmethod
    def execute(
        cls,
        plan,
        high_noise_model,
        low_noise_model,
        vae,
        omega_video,
        omega_image,
        omega_text,
        omega_target,
        omega_scale,
        use_task_defaults=True,
        boundary=0.875,
    ) -> io.NodeOutput:
        video_latents, image_latents, latent = encode_renderer_sources(plan, vae)
        if use_task_defaults:
            preset = task_preset(plan.task)
            omega_video = preset["omega_video"]
            omega_image = preset["omega_image"]
            omega_text = preset["omega_text"]
            omega_target = preset["omega_target"]
            omega_scale = preset["omega_scale"]
        common = {
            "plan": plan,
            "video_latents": video_latents,
            "image_latents": image_latents,
            "omega_video": omega_video,
            "omega_image": omega_image,
            "omega_text": omega_text,
            "omega_target": omega_target,
        }
        guider = BerniniV2DualExpertGuider(
            high_noise_model,
            low_noise_model,
            boundary=boundary,
            omega_scale=omega_scale,
            **common,
        )
        return io.NodeOutput(guider, latent)


class BerniniV2Scheduler(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="BerniniV2Scheduler",
            display_name="Bernini v2 UniPC Sigmas",
            category="sampling/bernini_v2",
            description=(
                "Matches Diffusers UniPC flow-sigma spacing: linearly spaced training timesteps "
                "from 999 to 0, followed by the terminal zero sigma."
            ),
            inputs=[
                BerniniV2PlanType.Input("plan"),
                io.Int.Input("steps", default=40, min=1, max=10000),
                io.Float.Input("flow_shift", default=5.0, min=0.01, max=100.0, step=0.01),
                io.Boolean.Input("use_task_defaults", default=True, advanced=True),
            ],
            outputs=[io.Sigmas.Output()],
        )

    @classmethod
    def execute(cls, plan, steps, flow_shift=5.0, use_task_defaults=True) -> io.NodeOutput:
        if use_task_defaults:
            steps = task_preset(plan.task)["steps"]
        return io.NodeOutput(unipc_flow_sigmas(steps, flow_shift))


class BerniniV2UniPCSampler(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="BerniniV2UniPCSampler",
            display_name="Bernini v2 Flow UniPC (BH2)",
            category="sampling/bernini_v2",
            description=(
                "The released order-2 UniPC BH2 solver for flow prediction. "
                "ComfyUI's generic UniPC uses a VP noise schedule and is not equivalent."
            ),
            inputs=[],
            outputs=[io.Sampler.Output()],
        )

    @classmethod
    def execute(cls) -> io.NodeOutput:
        return io.NodeOutput(comfy.samplers.KSAMPLER(sample_flow_unipc_bh2))


class BerniniV2RenderingExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [BerniniV2RendererGuider, BerniniV2Scheduler, BerniniV2UniPCSampler]
