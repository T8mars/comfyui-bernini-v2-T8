#!/usr/bin/env python3
"""Validate a Bernini v2 Wan high/low checkpoint pair without loading weights."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bernini_v2.checkpoint_info import inspect_renderer_pair  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--high", type=Path, required=True, help="High-noise safetensors file or shard index")
    parser.add_argument("--low", type=Path, required=True, help="Low-noise safetensors file or shard index")
    args = parser.parse_args()
    print(json.dumps(inspect_renderer_pair(args.high, args.low), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
