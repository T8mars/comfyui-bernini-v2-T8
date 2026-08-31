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


def test_single_flow_step_returns_current_x0_prediction():
    noise = torch.tensor([[[[0.25, -0.5]]]])
    sigmas = unipc_flow_sigmas(1, 5.0)
    scaled = noise * sigmas[0]
    expected = _ToyComfyModel()(noise, sigmas[0].expand(1))
    result = sample_flow_unipc_bh2(_ToyComfyModel(), scaled, sigmas)
    torch.testing.assert_close(result, expected)
