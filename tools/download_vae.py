#!/usr/bin/env python3
"""Download the standard Wan 2.1 VAE used by Bernini v2."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

REPO_ID = "t8star/Bernini-V2-Comfy"
REVISION = "00dbbfd2a9d043a7295a4d2efffaa10038c19024"
FILENAME = "vae/wan_2.1_vae.safetensors"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "60")
    from huggingface_hub import hf_hub_download

    output = args.output.resolve()
    downloaded = Path(
        hf_hub_download(
            repo_id=REPO_ID,
            filename=FILENAME,
            revision=REVISION,
            local_dir=output,
        )
    )
    target = output / downloaded.name
    if downloaded.resolve() != target.resolve():
        if target.exists():
            downloaded.unlink()
        else:
            downloaded.replace(target)
    print(target)


if __name__ == "__main__":
    main()
