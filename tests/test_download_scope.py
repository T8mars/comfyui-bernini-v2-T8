from tools.download_model import MODEL_PATTERNS


def test_full_download_uses_combined_checkpoint_without_duplicate_components():
    assert "bernini/*.safetensors" in MODEL_PATTERNS
    assert not any(pattern.startswith("t5_text_encoder/") for pattern in MODEL_PATTERNS)
    assert not any(pattern.startswith("vae/") for pattern in MODEL_PATTERNS)
