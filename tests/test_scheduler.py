import torch

from bernini_v2.guidance import unipc_flow_sigmas


def test_official_unipc_16_step_sigma_spacing():
    sigmas = unipc_flow_sigmas(16, 5.0)
    expected_start = torch.tensor([0.9997998476, 0.9866341949, 0.9720060229, 0.9556572437, 0.9372654557])
    torch.testing.assert_close(sigmas[:5], expected_start)
    torch.testing.assert_close(sigmas[-3:], torch.tensor([0.4163888097, 0.2497999668, 0.0]))
