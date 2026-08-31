import pytest

from bernini_v2.planner import validate_generation_request


def _validate(**overrides):
    values = {
        "task": "t2v",
        "width": 640,
        "height": 368,
        "length": 33,
        "source_fps": 16.0,
        "planning_steps": 25,
        "vit_denoising_steps": 1,
    }
    values.update(overrides)
    validate_generation_request(**values)


def test_accepts_two_second_non_square_video_shape():
    _validate()
    _validate(width=368, height=640)


def test_rejects_non_4n_plus_1_video_length():
    with pytest.raises(ValueError, match="4n\\+1"):
        _validate(length=32)


def test_rejects_unaligned_dimensions_before_model_load():
    with pytest.raises(ValueError, match="multiples of 16"):
        _validate(width=639)


def test_rejects_invalid_fps():
    with pytest.raises(ValueError, match="source_fps"):
        _validate(source_fps=0)


def test_rejects_extreme_latent_before_sampling():
    with pytest.raises(ValueError, match="latent alone"):
        _validate(width=8192, height=8192, length=129)
