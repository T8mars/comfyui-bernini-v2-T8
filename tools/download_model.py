#!/usr/bin/env python3
"""Resumable downloader for the public Bernini-Diffusers-v2 snapshot."""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

REPO_ID = "ByteDance/Bernini-Diffusers-v2"
METADATA_PATTERNS = (
    "*.json",
    "*.txt",
    "*.model",
    "*.jinja",
    "*.tiktoken",
    "*.py",
    "README.md",
    "LICENSE",
)
MODEL_PATTERNS = (
    "bernini/*.safetensors",
    "bernini/model.safetensors.index.json",
    "config.json",
    "mllm/chat_template.json",
    "mllm/config.json",
    "mllm/merges.txt",
    "mllm/preprocessor_config.json",
    "mllm/tokenizer.json",
    "mllm/tokenizer_config.json",
    "mllm/vocab.json",
    "t5_tokenizer/special_tokens_map.json",
    "t5_tokenizer/spiece.model",
    "t5_tokenizer/tokenizer_config.json",
    "README.md",
    "LICENSE",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("models") / "ByteDance" / "Bernini-Diffusers-v2",
    )
    parser.add_argument("--revision", default=None, help="Optional immutable Hugging Face commit revision")
    parser.add_argument("--metadata-only", action="store_true")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--attempts", type=int, default=20)
    parser.add_argument("--download-timeout", type=int, default=60)
    args = parser.parse_args()

    # Import after setting these variables: huggingface_hub reads them into
    # module-level constants. Plain HTTP resumes more reliably than Xet on
    # networks that intermittently reset multi-gigabyte CAS transfers.
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", str(args.download_timeout))
    from huggingface_hub import snapshot_download

    kwargs: dict[str, object] = {
        "repo_id": REPO_ID,
        "local_dir": str(args.output.resolve()),
        "max_workers": args.workers,
    }
    if args.revision:
        kwargs["revision"] = args.revision
    kwargs["allow_patterns"] = list(METADATA_PATTERNS if args.metadata_only else MODEL_PATTERNS)

    for attempt in range(1, args.attempts + 1):
        try:
            result = snapshot_download(**kwargs)
            print(result)
            return
        except Exception as error:
            if attempt == args.attempts:
                raise
            delay = min(30, 5 * attempt)
            print(
                f"download attempt {attempt}/{args.attempts} failed: {error}; resuming in {delay}s",
                flush=True,
            )
            time.sleep(delay)


if __name__ == "__main__":
    main()
