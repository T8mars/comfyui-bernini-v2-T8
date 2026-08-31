# Copyright (c) 2026 ByteDance Ltd. and/or its affiliate
# SPDX-License-Identifier: Apache-2.0
"""Bernini-specific helpers around ComfyUI's native Qwen2.5-VL model."""

from __future__ import annotations

import math
import types

import torch
import torch.nn.functional as F


def apply_qwen_vision_rope(
    query: torch.Tensor,
    key: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply Qwen vision RoPE in FP32 and restore the input dtypes.

    This matches ByteDance's Qwen2.5-VL implementation.  ComfyUI 0.33.0
    performs the FP32 calculation but omits the final casts, leaving Q/K in
    FP32 while V remains BF16/FP16 and causing SDPA to reject the inputs.
    """

    query_dtype = query.dtype
    key_dtype = key.dtype
    query = query.float()
    key = key.float()
    cos = cos.unsqueeze(-2).float()
    sin = sin.unsqueeze(-2).float()

    def rotate_half(value: torch.Tensor) -> torch.Tensor:
        first = value[..., : value.shape[-1] // 2]
        second = value[..., value.shape[-1] // 2 :]
        return torch.cat((-second, first), dim=-1)

    query = (query * cos) + (rotate_half(query) * sin)
    key = (key * cos) + (rotate_half(key) * sin)
    return query.to(query_dtype), key.to(key_dtype)


def _dtype_safe_vision_attention_forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings=None,
    cu_seqlens=None,
    optimized_attention=None,
) -> torch.Tensor:
    """Instance-local replacement for Comfy's mixed-dtype vision attention."""

    if hidden_states.dim() == 2:
        sequence_length, _ = hidden_states.shape
        batch_size = 1
        hidden_states = hidden_states.unsqueeze(0)
    else:
        batch_size, sequence_length, _ = hidden_states.shape

    qkv = self.qkv(hidden_states)
    qkv = qkv.reshape(batch_size, sequence_length, 3, self.num_heads, self.head_dim)
    query, key, value = qkv.reshape(sequence_length, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
    if position_embeddings is not None:
        cos, sin = position_embeddings
        query, key = apply_qwen_vision_rope(query, key, cos, sin)

    query = query.transpose(0, 1).unsqueeze(0)
    key = key.transpose(0, 1).unsqueeze(0)
    value = value.transpose(0, 1).unsqueeze(0)
    lengths = cu_seqlens[1:] - cu_seqlens[:-1]
    splits = [torch.split(tensor, lengths.tolist(), dim=2) for tensor in (query, key, value)]
    outputs = [optimized_attention(q, k, v, self.num_heads, skip_reshape=True) for q, k, v in zip(*splits, strict=True)]
    output = torch.cat(outputs, dim=1).reshape(sequence_length, -1)
    return self.proj(output)


def install_qwen_vision_dtype_compat(vision_model) -> None:
    """Patch only this Qwen vision instance with the upstream dtype behavior."""

    for block in vision_model.blocks:
        block.attn.forward = types.MethodType(_dtype_safe_vision_attention_forward, block.attn)


def smart_resize_qwen(
    height: int,
    width: int,
    *,
    factor: int = 28,
    min_pixels: int = 3136,
    max_pixels: int = 50176,
) -> tuple[int, int]:
    """Qwen2-VL smart-resize geometry without importing a HF processor."""

    if height < 1 or width < 1 or max(height, width) / min(height, width) > 200:
        raise ValueError(f"invalid Qwen visual size: {height}x{width}")
    target_h = round(height / factor) * factor
    target_w = round(width / factor) * factor
    if target_h * target_w > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        target_h = max(factor, math.floor(height / beta / factor) * factor)
        target_w = max(factor, math.floor(width / beta / factor) * factor)
    elif target_h * target_w < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        target_h = math.ceil(height * beta / factor) * factor
        target_w = math.ceil(width * beta / factor) * factor
    return target_h, target_w


def qwen_grid_for_media(
    frame_count: int,
    height: int,
    width: int,
    *,
    min_pixels: int = 3136,
    max_pixels: int = 50176,
    patch_size: int = 14,
    temporal_patch_size: int = 2,
    merge_size: int = 2,
) -> torch.Tensor:
    target_h, target_w = smart_resize_qwen(
        height,
        width,
        factor=patch_size * merge_size,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
    )
    grid_t = math.ceil(frame_count / temporal_patch_size)
    return torch.tensor([[grid_t, target_h // patch_size, target_w // patch_size]], dtype=torch.long)


def planner_video_frame_indices(
    total_frames: int,
    *,
    source_fps: float = 16.0,
    planner_fps: float = 2.0,
    frame_factor: int = 2,
    max_frames: int | None = None,
) -> list[int]:
    """Match the official 2-fps, even-frame Bernini VIT sampling policy."""

    if total_frames < 1 or source_fps <= 0 or planner_fps <= 0:
        raise ValueError("frame counts and frame rates must be positive")
    count = math.floor((total_frames / source_fps * planner_fps) / frame_factor) * frame_factor
    count = max(count, frame_factor)
    if max_frames is not None:
        count = min(count, math.floor(max_frames / frame_factor) * frame_factor)
    count = max(count, frame_factor)
    return torch.linspace(0, total_frames - 1, count).round().long().tolist()


def qwen_checkpoint_to_comfy(key: str) -> str | None:
    """Return the native Comfy key, excluding the unused language-model head."""

    if key == "lm_head.weight":
        return None
    return key


def process_qwen2vl_video(
    frames: torch.Tensor,
    *,
    min_pixels: int = 3136,
    max_pixels: int = 50176,
    patch_size: int = 14,
    temporal_patch_size: int = 2,
    merge_size: int = 2,
    image_mean: tuple[float, float, float] = (0.48145466, 0.4578275, 0.40821073),
    image_std: tuple[float, float, float] = (0.26862954, 0.26130258, 0.27577711),
) -> tuple[torch.Tensor, torch.Tensor]:
    """Native Qwen2.5-VL video patchification for a Comfy ``IMAGE`` batch.

    ``frames`` has shape ``[T,H,W,C]`` and values in ``[0,1]``. The last frame
    is repeated when necessary to fill the two-frame temporal patch.
    """

    if frames.ndim != 4 or frames.shape[-1] < 3 or frames.shape[0] < 1:
        raise ValueError(f"expected [T,H,W,C] video, got {tuple(frames.shape)}")
    frames = frames[..., :3].permute(0, 3, 1, 2).float()
    frame_count, _, height, width = frames.shape
    target_h, target_w = smart_resize_qwen(
        height,
        width,
        factor=patch_size * merge_size,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
    )

    frames = F.interpolate(
        frames,
        size=(target_h, target_w),
        mode="bicubic",
        align_corners=False,
        antialias=True,
    )
    mean = torch.tensor(image_mean, device=frames.device).view(1, 3, 1, 1)
    std = torch.tensor(image_std, device=frames.device).view(1, 3, 1, 1)
    frames = (frames - mean) / std
    remainder = frame_count % temporal_patch_size
    if remainder:
        frames = torch.cat([frames, frames[-1:].expand(temporal_patch_size - remainder, -1, -1, -1)], dim=0)

    grid_t = frames.shape[0] // temporal_patch_size
    grid_h = target_h // patch_size
    grid_w = target_w // patch_size
    patches = frames.reshape(
        grid_t,
        temporal_patch_size,
        3,
        grid_h // merge_size,
        merge_size,
        patch_size,
        grid_w // merge_size,
        merge_size,
        patch_size,
    )
    patches = patches.permute(0, 3, 6, 4, 7, 2, 1, 5, 8)
    flattened = patches.reshape(grid_t * grid_h * grid_w, 3 * temporal_patch_size * patch_size * patch_size)
    grid = torch.tensor([[grid_t, grid_h, grid_w]], device=frames.device, dtype=torch.long)
    return flattened, grid


@torch.no_grad()
def plan_forward(
    model,
    inputs_embeds: torch.Tensor,
    position_ids: torch.Tensor,
    additive_attention_mask: torch.Tensor,
    *,
    intermediate_output: int = -2,
) -> torch.Tensor:
    """Run native Comfy Qwen with Bernini's precomputed additive attention mask."""

    from comfy.ldm.modules.attention import optimized_attention_for_device

    x = inputs_embeds.clone()
    if position_ids.ndim == 3 and position_ids.shape[1] == 1:
        position_ids = position_ids.squeeze(1)
    freqs_cis = model.compute_freqs_cis(position_ids, x.device)
    mask = additive_attention_mask.unsqueeze(1).to(dtype=x.dtype, device=x.device)
    attention = optimized_attention_for_device(x.device, mask=True, small_input=True)
    target = len(model.layers) + intermediate_output if intermediate_output < 0 else intermediate_output
    intermediate = None
    for index, layer in enumerate(model.layers):
        x, _ = layer(
            x=x,
            attention_mask=mask,
            freqs_cis=freqs_cis,
            optimized_attention=attention,
            past_key_value=None,
        )
        if index == target:
            intermediate = x.clone()
    if intermediate is None:
        raise ValueError(f"intermediate layer {intermediate_output} is outside the model")
    return intermediate
