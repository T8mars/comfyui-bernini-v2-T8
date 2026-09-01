import pytest
import torch

from bernini_v2.guidance import unipc_flow_sigmas
from bernini_v2.unipc import sample_flow_unipc_bh2


class _ToyComfyModel:
    def __call__(self, sample, sigma, **_kwargs):
        sigma = sigma.reshape(sigma.shape + (1,) * (sample.ndim - sigma.ndim))
        velocity = 0.2 * sample + 0.1 * sigma
        return sample - sigma * velocity


def test_flow_unipc_is_deterministic_and_finite():
    noise = torch.tensor([[[[0.25, -0.5], [1.0, -1.5]]]])
    sigmas = unipc_flow_sigmas(8, 5.0)
    # Comfy's flow ModelSampling pre-scales the random noise by sigma[0].
    comfy_scaled_noise = noise * sigmas[0]
    first = sample_flow_unipc_bh2(_ToyComfyModel(), comfy_scaled_noise, sigmas)
    second = sample_flow_unipc_bh2(_ToyComfyModel(), comfy_scaled_noise, sigmas)
    torch.testing.assert_close(first, second)
    assert torch.isfinite(first).all()


@pytest.mark.parametrize(
    "sigmas",
    [
        torch.tensor([1.0, 0.0]),
        torch.tensor([1.01, 0.0]),
        torch.tensor([-0.01, 0.0]),
        torch.tensor([0.5, torch.nan]),
    ],
)
def test_flow_unipc_rejects_non_flow_sigmas(sigmas):
    with pytest.raises(ValueError, match=r"sigmas in \[0, 1\)"):
        sample_flow_unipc_bh2(_ToyComfyModel(), torch.ones(1), sigmas)


@pytest.mark.parametrize(
    "sigmas",
    [torch.tensor([0.5, 0.5, 0.0]), torch.tensor([0.5, 0.75, 0.0])],
)
def test_flow_unipc_rejects_non_decreasing_sigmas(sigmas):
    with pytest.raises(ValueError, match="strictly decreasing"):
        sample_flow_unipc_bh2(_ToyComfyModel(), torch.ones(1), sigmas)


def test_flow_unipc_preserves_short_schedule_early_return():
    noise = torch.randn(2, 3)
    assert sample_flow_unipc_bh2(_ToyComfyModel(), noise, torch.tensor([1.0])) is noise


def test_single_flow_step_returns_current_x0_prediction():
    noise = torch.tensor([[[[0.25, -0.5]]]])
    sigmas = unipc_flow_sigmas(1, 5.0)
    scaled = noise * sigmas[0]
    expected = _ToyComfyModel()(noise, sigmas[0].expand(1))
    result = sample_flow_unipc_bh2(_ToyComfyModel(), scaled, sigmas)
    torch.testing.assert_close(result, expected)
