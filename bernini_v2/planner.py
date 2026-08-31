# Copyright (c) 2026 ByteDance Ltd. and/or its affiliate
# SPDX-License-Identifier: Apache-2.0
"""Native three-branch Bernini v2 semantic planning."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

from .qwen import plan_forward, planner_video_frame_indices, process_qwen2vl_video, qwen_grid_for_media
from .rope import get_rope_index
from .runtime import BerniniV2PlannerRuntime
from .template import BerniniTemplate, build_conversation

IMAGE_TASKS = {"t2i", "i2i"}
VIDEO_TASKS = {"t2v", "v2v", "r2v", "rv2v"}
SUPPORTED_TASKS = IMAGE_TASKS | VIDEO_TASKS


@dataclass
class BerniniV2Plan:
    """Renderer contexts and source media produced by the semantic planner."""

    task: str
    width: int
    height: int
    length: int
    max_media_size: int
    contexts: dict[str, torch.Tensor]
    predicted_vit: torch.Tensor
    source_video: torch.Tensor | None
    reference_images: list[torch.Tensor]


def validate_task_inputs(
    task: str,
    source_video: torch.Tensor | None,
    reference_images: list[torch.Tensor],
) -> None:
    if task not in SUPPORTED_TASKS:
        raise ValueError(f"unsupported task {task!r}")
    has_video = source_video is not None
    has_images = bool(reference_images)
    required = {
        "t2i": (False, False),
        "t2v": (False, False),
        "i2i": (False, True),
        "v2v": (True, False),
        "r2v": (False, True),
        "rv2v": (True, True),
    }[task]
    if (has_video, has_images) != required:
        raise ValueError(
            f"task {task} expects source_video={required[0]} and reference_images={required[1]}, "
            f"got source_video={has_video}, reference_images={has_images}"
        )
    if task == "i2i" and len(reference_images) != 1:
        raise ValueError("i2i requires exactly one source image")


def split_reference_images(reference_images: dict[str, torch.Tensor] | None) -> list[torch.Tensor]:
    output = []
    for name in sorted(reference_images or {}):
        images = reference_images[name]
        if images is None:
            continue
        output.extend(images[index : index + 1] for index in range(images.shape[0]))
    return output


def maskgit_order(target_count: int, seed: int, device: torch.device | str = "cpu") -> torch.Tensor:
    """Return the official NumPy-seeded order as a PyTorch scatter index."""
    order = np.random.RandomState(seed).permutation(target_count).astype(np.int64, copy=False)
    return torch.from_numpy(order).to(device=device)


def _visual_token_count(grid: torch.Tensor) -> int:
    return int(grid.prod().item()) // 4


def _encode_sources(
    runtime: BerniniV2PlannerRuntime,
    *,
    source_video: torch.Tensor | None,
    reference_images: list[torch.Tensor],
    source_fps: float,
    max_frames: int,
) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
    patches = []
    grids = []
    kinds = []
    if source_video is not None:
        indices = planner_video_frame_indices(
            source_video.shape[0],
            source_fps=source_fps,
            planner_fps=2.0,
            max_frames=max_frames,
        )
        video_patches, video_grid = process_qwen2vl_video(source_video[indices])
        patches.append(video_patches)
        grids.append(video_grid)
        kinds.append("video")
    for image in reference_images:
        image_patches, image_grid = process_qwen2vl_video(image[:1])
        patches.append(image_patches)
        grids.append(image_grid)
        kinds.append("image")
    if not patches:
        return [], [], [], []

    runtime.load_vision()
    device = runtime.load_device
    packed_patches = torch.cat(patches).to(device=device, dtype=runtime.dtype)
    packed_grids = torch.cat(grids).to(device=device)
    embeddings = runtime.vision_model(packed_patches, image_grid_thw=packed_grids)
    split_sizes = [int(grid.prod().item()) // 4 for grid in grids]
    split_embeddings = list(torch.split(embeddings, split_sizes))
    video_embeddings = [embed for embed, kind in zip(split_embeddings, kinds, strict=True) if kind == "video"]
    image_embeddings = [embed for embed, kind in zip(split_embeddings, kinds, strict=True) if kind == "image"]
    video_grids = [grid[0].cpu() for grid, kind in zip(grids, kinds, strict=True) if kind == "video"]
    image_grids = [grid[0].cpu() for grid, kind in zip(grids, kinds, strict=True) if kind == "image"]
    return video_embeddings, image_embeddings, video_grids, image_grids


def _branch_grids(
    branch: dict[str, object],
    image_grids: list[torch.Tensor],
    video_grids: list[torch.Tensor],
    *,
    device: torch.device,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    selected_images = []
    selected_videos = []
    for kind, source_index in zip(
        branch["vit_type_list"].tolist(),
        branch["vit_source_indices"].tolist(),
        strict=True,
    ):
        if kind == 0:
            selected_images.append(image_grids[source_index])
        else:
            selected_videos.append(video_grids[source_index])
    image_grid = torch.stack(selected_images).to(device) if selected_images else None
    video_grid = torch.stack(selected_videos).to(device) if selected_videos else None
    return image_grid, video_grid


def _prepare_branch(
    runtime: BerniniV2PlannerRuntime,
    branch: dict[str, object],
    *,
    source_embeddings: list[torch.Tensor],
    image_grids: list[torch.Tensor],
    video_grids: list[torch.Tensor],
) -> dict[str, torch.Tensor]:
    device = runtime.load_device
    ids = branch["input_ids"].unsqueeze(0).to(device)
    inputs = runtime.language_model.embed_tokens(ids, out_dtype=runtime.dtype)
    visual_input_mask = branch["visual_input_token_mask"].to(device)
    if visual_input_mask.any():
        packed_sources = torch.cat(source_embeddings).to(device=device, dtype=inputs.dtype)
        if int(visual_input_mask.sum()) != packed_sources.shape[0]:
            raise ValueError(
                f"source feature/token mismatch: {packed_sources.shape[0]} features vs "
                f"{int(visual_input_mask.sum())} tokens"
            )
        inputs[:, visual_input_mask, :] = packed_sources.unsqueeze(0)
    output_mask = branch["visual_output_token_mask"].to(device)
    target_count = int(output_mask.sum())
    if target_count > runtime.aux.mask_tokens.shape[1]:
        raise ValueError(
            f"target uses {target_count} VIT tokens, but checkpoint supports {runtime.aux.mask_tokens.shape[1]}"
        )
    inputs[:, output_mask, :] = runtime.aux.mask_tokens[:, :1].to(inputs).expand(1, target_count, -1)
    image_grid, video_grid = _branch_grids(
        branch,
        image_grids,
        video_grids,
        device=device,
    )
    position_ids, _ = get_rope_index(
        ids,
        image_grid_thw=image_grid,
        video_grid_thw=video_grid,
        attention_mask=branch["attention_mask"].unsqueeze(0).to(device),
    )
    return {
        "inputs": inputs,
        "positions": position_ids,
        "attention": branch["attention_mask_4d"].to(device),
        "output_mask": output_mask,
    }


def _hidden(runtime: BerniniV2PlannerRuntime, branch: dict[str, torch.Tensor]) -> torch.Tensor:
    return plan_forward(
        runtime.language_model,
        branch["inputs"],
        branch["positions"],
        branch["attention"],
        intermediate_output=-2,
    )


def _conditioning_tensor(conditioning, name: str) -> torch.Tensor:
    if not conditioning or not isinstance(conditioning[0], (list, tuple)):
        raise ValueError(f"{name} is not a Comfy CONDITIONING")
    tensor = conditioning[0][0]
    if not torch.is_tensor(tensor) or tensor.ndim != 3 or tensor.shape[-1] != 4096:
        raise ValueError(f"{name} must contain [B,L,4096] UMT5 embeddings")
    if tensor.shape[0] != 1:
        raise ValueError("Bernini v2 currently supports batch size 1")
    # Comfy's Wan tokenizer pads to at least 512 and zeroes masked rows. The
    # official pipeline crops T5 to its attention-mask length before appending
    # Qwen tokens, then pads the combined result (not T5 alone) to 512.
    nonzero = tensor.abs().sum(dim=-1).ne(0)
    valid = nonzero[0].nonzero(as_tuple=False)
    length = int(valid[-1]) + 1 if valid.numel() else 0
    return tensor[:, :length]


def _pad_renderer_context(context: torch.Tensor, min_length: int = 512) -> torch.Tensor:
    if context.shape[1] >= min_length:
        return context
    padding = context.new_zeros(context.shape[0], min_length - context.shape[1], context.shape[2])
    return torch.cat([context, padding], dim=1)


@torch.no_grad()
def create_plan(
    runtime: BerniniV2PlannerRuntime,
    *,
    positive,
    negative,
    prompt: str,
    negative_prompt: str,
    task: str,
    width: int,
    height: int,
    length: int,
    max_media_size: int,
    source_video: torch.Tensor | None,
    reference_images: list[torch.Tensor],
    source_fps: float = 16.0,
    planning_steps: int = 25,
    vit_denoising_steps: int = 3,
    vit_text_cfg: float = 1.4,
    vit_image_cfg: float = 1.2,
    seed: int = 42,
) -> BerniniV2Plan:
    validate_task_inputs(task, source_video, reference_images)
    output_is_image = task in IMAGE_TASKS
    target_length = 1 if output_is_image else length
    video_embeds, image_embeds, video_grids, image_grids = _encode_sources(
        runtime,
        source_video=source_video,
        reference_images=reference_images,
        source_fps=source_fps,
        max_frames=target_length,
    )
    if output_is_image:
        target_grid = qwen_grid_for_media(1, height, width)[0]
        image_grids.append(target_grid)
    else:
        target_vit_frames = len(
            planner_video_frame_indices(
                target_length,
                source_fps=16.0,
                planner_fps=2.0,
                max_frames=target_length,
            )
        )
        target_grid = qwen_grid_for_media(target_vit_frames, height, width)[0]
        video_grids.append(target_grid)

    image_counts = [_visual_token_count(grid) for grid in image_grids]
    video_counts = [_visual_token_count(grid) for grid in video_grids]
    conversation = build_conversation(
        prompt,
        source_videos=1 if source_video is not None else 0,
        source_images=len(reference_images),
        output_is_image=output_is_image,
    )
    template = BerniniTemplate(runtime.tokenizer)
    counts = {"image": image_counts, "video": video_counts}
    cond_tokens = template.encode(conversation, num_tokens=counts, task=task)
    uncond_tokens = template.encode(
        conversation,
        num_tokens=counts,
        task=task,
        drop_text=True,
        drop_images=True,
        drop_videos=True,
        negative_prompt=negative_prompt,
    )
    image_cond_tokens = template.encode(
        conversation,
        num_tokens=counts,
        task=task,
        drop_images=True,
        drop_videos=True,
    )

    runtime.load_planner()
    ordered_source_embeddings = video_embeds + image_embeds
    branches = [
        _prepare_branch(
            runtime,
            tokens,
            source_embeddings=ordered_source_embeddings if index == 0 else [],
            image_grids=image_grids,
            video_grids=video_grids,
        )
        for index, tokens in enumerate((cond_tokens, uncond_tokens, image_cond_tokens))
    ]
    cond, uncond, image_cond = branches
    target_count = int(cond["output_mask"].sum())
    generator = torch.Generator(device="cpu").manual_seed(seed)
    order = maskgit_order(target_count, seed, runtime.load_device)
    mask = torch.ones(target_count, device=runtime.load_device, dtype=torch.bool)
    for step in range(planning_steps):
        hidden_states = [_hidden(runtime, branch) for branch in branches]
        predicted = [
            runtime.aux.connector.for_vit(hidden[:, branch["output_mask"], :])
            for hidden, branch in zip(hidden_states, branches, strict=True)
        ]
        ratio = math.cos(math.pi * 0.5 * (step + 1) / planning_steps)
        mask_length = max(1, min(int(mask.sum()) - 1, math.floor(target_count * ratio)))
        next_mask = torch.zeros_like(mask)
        next_mask.scatter_(0, order[:mask_length], True)
        to_predict = mask if step == planning_steps - 1 else torch.logical_xor(mask, next_mask)
        mask = next_mask
        indices = to_predict.nonzero(as_tuple=False).squeeze(-1)
        if indices.numel() == 0:
            continue
        decoder_condition = torch.cat([value[:, indices, :] for value in predicted], dim=1)[0]
        current = runtime.aux.vit_decoder.sample(
            decoder_condition,
            cfg=vit_text_cfg,
            img_cfg=vit_image_cfg,
            num_inference_steps=vit_denoising_steps,
            generator=generator,
        )
        current = current[: current.shape[0] // 3].unsqueeze(0).to(runtime.dtype)
        target = cond["inputs"][:, cond["output_mask"], :]
        target[:, indices, :] = current
        for branch in branches:
            branch_target = branch["inputs"][:, branch["output_mask"], :]
            branch_target[:, indices, :] = current
            branch["inputs"][:, branch["output_mask"], :] = branch_target

    predicted_vit = cond["inputs"][:, cond["output_mask"], :]
    cond_hidden = _hidden(runtime, cond)
    uncond_hidden = _hidden(runtime, uncond)
    cond_context = runtime.aux.connector.for_gen(cond_hidden)
    uncond_context = runtime.aux.connector.for_gen(uncond_hidden)
    cond_text_mask = ~cond["output_mask"]
    uncond_text_mask = ~uncond["output_mask"]
    positive_t5 = _conditioning_tensor(positive, "positive").to(cond_context)
    negative_t5 = _conditioning_tensor(negative, "negative").to(cond_context)
    contexts = {
        "wtxt_wvit": _pad_renderer_context(torch.cat([positive_t5, cond_context], dim=1)),
        "wtxt_wovit": _pad_renderer_context(torch.cat([positive_t5, cond_context[:, cond_text_mask, :]], dim=1)),
        "wotxt_wvit": _pad_renderer_context(torch.cat([negative_t5, cond_context[:, cond["output_mask"], :]], dim=1)),
        "wotxt_wovit": _pad_renderer_context(torch.cat([negative_t5, uncond_context[:, uncond_text_mask, :]], dim=1)),
    }
    import comfy.model_management

    intermediate = comfy.model_management.intermediate_device()
    contexts = {key: value.to(intermediate) for key, value in contexts.items()}
    return BerniniV2Plan(
        task=task,
        width=width,
        height=height,
        length=target_length,
        max_media_size=max_media_size,
        contexts=contexts,
        predicted_vit=predicted_vit.to(intermediate),
        source_video=source_video,
        reference_images=reference_images,
    )
