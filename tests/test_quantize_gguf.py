from types import SimpleNamespace

import numpy as np
import pytest

from tools.quantize_gguf import _architecture, normalize_qtype


@pytest.mark.parametrize(
    ("value", "expected"),
    [("q4_k_s", "Q4_K_S"), (" Q5_K_M ", "Q5_K_M"), ("Q8_0", "Q8_0")],
)
def test_normalize_qtype(value, expected):
    assert normalize_qtype(value) == expected


@pytest.mark.parametrize("value", ["", "q4-k-s", "../Q4_K_S", "Q4 K S"])
def test_normalize_qtype_rejects_unsafe_values(value):
    with pytest.raises(ValueError, match="invalid GGUF quantization type"):
        normalize_qtype(value)


def test_architecture_decodes_gguf_uint8_parts():
    field = SimpleNamespace(parts=[np.frombuffer(b"wan", dtype=np.uint8)], data=[0])
    reader = SimpleNamespace(get_field=lambda name: field)
    assert _architecture(reader, np) == "wan"
