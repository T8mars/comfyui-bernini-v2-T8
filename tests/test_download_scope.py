from tools.download_model import MODEL_PATTERNS
from tools.download_vae import FILENAME, REPO_ID, REVISION


def test_full_download_uses_combined_checkpoint_without_duplicate_components():
    assert "bernini/*.safetensors" in MODEL_PATTERNS
    assert not any(pattern.startswith("t5_text_encoder/") for pattern in MODEL_PATTERNS)
    assert not any(pattern.startswith("vae/") for pattern in MODEL_PATTERNS)


def test_vae_download_uses_pinned_companion_file():
    assert REPO_ID == "t8star/Bernini-V2-Comfy"
    assert FILENAME == "vae/wan_2.1_vae.safetensors"
    assert len(REVISION) == 40
