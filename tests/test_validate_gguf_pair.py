import pytest

from tools.validate_gguf_pair import validate_contracts


def test_validate_contracts_accepts_identical_pairs():
    contract = {"weight": ((512, 512), "Q4_K")}
    validate_contracts(contract, dict(contract))


def test_validate_contracts_rejects_missing_tensor():
    with pytest.raises(ValueError, match="tensor names differ"):
        validate_contracts({"weight": ((512, 512), "Q4_K")}, {})


def test_validate_contracts_rejects_shape_or_type_difference():
    with pytest.raises(ValueError, match="shape/type contracts differ"):
        validate_contracts({"weight": ((512, 512), "Q4_K")}, {"weight": ((512, 512), "Q5_K")})
