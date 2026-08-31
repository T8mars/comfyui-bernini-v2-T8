from __future__ import annotations

import pytest
import torch

from bernini_v2.planner_model import (
    DiffLossFM,
    FlowMatchScheduler,
    MLPConnector,
    expected_connector_keys,
    expected_vit_decoder_keys,
)
from bernini_v2.qwen import (
    planner_video_frame_indices,
    process_qwen2vl_video,
    qwen_checkpoint_to_comfy,
    qwen_grid_for_media,
    smart_resize_qwen,
)


def test_planner_weight_layout_counts() -> None:
    assert len(expected_connector_keys()) == 12
    assert len(expected_vit_decoder_keys()) == 140


def test_small_planner_modules_forward() -> None:
    connector = MLPConnector(in_dim=8, out_dim_for_gen=6, out_dim_for_vit=8)
    x = torch.randn(2, 3, 8)
    assert connector.for_gen(x).shape == (2, 3, 6)
    assert connector.for_vit(x).shape == (2, 3, 8)

    decoder = DiffLossFM(target_channels=8, z_channels=8, depth=2, width=16)
    output = decoder.net(torch.randn(4, 8), torch.ones(1), torch.randn(4, 8))
    assert output.shape == (4, 8)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for the offload regression")
def test_rms_norm_accepts_cpu_weight_with_cuda_input() -> None:
    from bernini_v2.planner_model import RMSNorm

    norm = RMSNorm(8, device="cpu", dtype=torch.bfloat16)
    output = norm(torch.randn(2, 8, device="cuda", dtype=torch.bfloat16))
    assert output.device.type == "cuda"
    assert output.dtype == torch.bfloat16


def test_flow_scheduler_lands_at_zero() -> None:
    scheduler = FlowMatchScheduler(shift=2.0)
    scheduler.set_timesteps(3, dtype=torch.float32)
    assert len(scheduler.sigmas) == 3
    sample = torch.ones(1)
    result = scheduler.step(torch.ones(1), scheduler.timesteps[-1], sample)
    torch.testing.assert_close(result, sample - scheduler.sigmas[-1])


def test_flow_scheduler_extra_step_matches_official_formula() -> None:
    scheduler = FlowMatchScheduler(shift=2.0, extra_one_step=True)
    scheduler.set_timesteps(4, denoising_strength=0.75, dtype=torch.float32)
    raw = torch.linspace(
        scheduler.sigma_min + (scheduler.sigma_max - scheduler.sigma_min) * 0.75,
        scheduler.sigma_min,
        5,
    )[:-1]
    expected = scheduler.shift * raw / (1 + (scheduler.shift - 1) * raw)
    torch.testing.assert_close(scheduler.sigmas, expected)


def test_qwen_unused_lm_head() -> None:
    assert qwen_checkpoint_to_comfy("lm_head.weight") is None
    assert qwen_checkpoint_to_comfy("model.layers.0.self_attn.q_proj.weight") == (
        "model.layers.0.self_attn.q_proj.weight"
    )


def test_video_patchification_shape_and_last_frame_padding() -> None:
    frames = torch.rand(3, 28, 28, 3)
    patches, grid = process_qwen2vl_video(frames, min_pixels=28 * 28, max_pixels=28 * 28)
    assert grid.tolist() == [[2, 2, 2]]
    assert patches.shape == (8, 3 * 2 * 14 * 14)


def test_qwen_grid_and_official_video_frame_policy() -> None:
    assert smart_resize_qwen(480, 832) == (168, 280)
    assert qwen_grid_for_media(10, 480, 832).tolist() == [[5, 12, 20]]
    indices = planner_video_frame_indices(81, source_fps=16, planner_fps=2, max_frames=81)
    assert len(indices) == 10
    assert indices[0] == 0
    assert indices[-1] == 80
