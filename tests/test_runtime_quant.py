import torch

from bernini_v2.runtime import _module_init_device


def test_quantized_modules_initialize_on_offload_device_not_meta():
    offload = torch.device("cpu")
    state = {"layers.0.comfy_quant": torch.tensor([1], dtype=torch.uint8)}
    assert _module_init_device(state, offload) == offload


def test_plain_assign_loaded_modules_can_initialize_on_meta():
    assert _module_init_device({"weight": torch.ones(1)}, torch.device("cpu")) == "meta"
