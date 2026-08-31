import torch

from bernini_v2.guidance import (
    apg_delta,
    compose_denoised_guidance,
    compose_velocity_guidance,
)


def test_apg_parallel_and_orthogonal_components():
    reference = torch.tensor([[1.0, 0.0]])
    delta = torch.tensor([[2.0, 3.0]])
    result = apg_delta(delta, reference)
    torch.testing.assert_close(result, torch.tensor([[0.4, 3.0]]))


def test_standard_guidance_is_chained_with_apg():
    predictions = {
        "base": torch.tensor([[1.0, 0.0]]),
        "source": torch.tensor([[2.0, 1.0]]),
        "text": torch.tensor([[2.0, 3.0]]),
        "target": torch.tensor([[4.0, 3.0]]),
    }
    expected = (
        predictions["base"]
        + 2.0 * apg_delta(predictions["source"] - predictions["base"], predictions["source"])
        + 3.0 * apg_delta(predictions["text"] - predictions["source"], predictions["text"])
        + 4.0 * apg_delta(predictions["target"] - predictions["text"], predictions["target"])
    )
    result = compose_velocity_guidance(
        predictions,
        omega_video=9.0,
        omega_image=2.0,
        omega_text=3.0,
        omega_target=4.0,
        rv2v=False,
    )
    torch.testing.assert_close(result, expected)


def test_rv2v_uses_separate_direct_video_and_image_arms():
    predictions = {
        "base": torch.tensor([[1.0]]),
        "video": torch.tensor([[2.0]]),
        "image": torch.tensor([[4.0]]),
        "text": torch.tensor([[7.0]]),
        "target": torch.tensor([[11.0]]),
    }
    result = compose_velocity_guidance(
        predictions,
        omega_video=2.0,
        omega_image=3.0,
        omega_text=4.0,
        omega_target=5.0,
        rv2v=True,
    )
    torch.testing.assert_close(result, torch.tensor([[41.0]]))


def test_denoised_guidance_round_trips_through_flow_velocity():
    sample = torch.tensor([[[[[8.0]]]]])
    sigma = torch.tensor([0.5])
    velocities = {
        "base": torch.tensor([[[[[1.0]]]]]),
        "text": torch.tensor([[[[[2.0]]]]]),
        "target": torch.tensor([[[[[3.0]]]]]),
    }
    denoised = {name: sample - sigma * value for name, value in velocities.items()}
    guided_velocity = compose_velocity_guidance(
        velocities,
        omega_video=0.0,
        omega_image=0.0,
        omega_text=2.0,
        omega_target=3.0,
        rv2v=False,
    )
    result = compose_denoised_guidance(
        denoised,
        sample,
        sigma,
        omega_video=0.0,
        omega_image=0.0,
        omega_text=2.0,
        omega_target=3.0,
        rv2v=False,
    )
    torch.testing.assert_close(result, sample - sigma * guided_velocity)
