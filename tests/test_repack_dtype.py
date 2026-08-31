import pytest
import torch

from tools.repack_diffusers import convert_storage_tensor


@pytest.mark.parametrize(
    ("name", "expected"),
    [("bfloat16", torch.bfloat16), ("float16", torch.float16), ("preserve", torch.float32)],
)
def test_convert_storage_tensor_downcasts_floats(name, expected):
    tensor = torch.ones(2, 3, dtype=torch.float32)
    assert convert_storage_tensor(tensor, name).dtype == expected


def test_convert_storage_tensor_preserves_integer_metadata():
    tensor = torch.ones(2, dtype=torch.uint8)
    assert convert_storage_tensor(tensor, "bfloat16").dtype == torch.uint8


def test_convert_storage_tensor_rejects_unknown_dtype():
    with pytest.raises(ValueError, match="unsupported storage dtype"):
        convert_storage_tensor(torch.ones(1), "int4")
