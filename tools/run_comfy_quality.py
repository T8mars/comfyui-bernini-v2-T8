"""Queue one full-quality Bernini v2 example against a running ComfyUI server."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from run_comfy_smoke import TASKS, run_smoke

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bernini_v2.template import SYSTEM_PROMPTS  # noqa: E402

WORKFLOW_DIR = ROOT / "examples" / "api"


def prepare_graph(
    task: str,
    *,
    width: int | None = None,
    height: int | None = None,
    length: int | None = None,
    reference_image: str | list[str] = "bernini_quality_reference.png",
    source_video: str = "bernini_quality_source.mp4",
    output_prefix: str | None = None,
    prompt: str | None = None,
) -> dict[str, object]:
    """Load an example without disabling its published task defaults."""

    if task not in TASKS:
        raise ValueError(f"unsupported task: {task}")
    graph = json.loads((WORKFLOW_DIR / f"{task}.json").read_text(encoding="utf-8"))
    plan = graph["11"]["inputs"]
    if width is not None:
        plan["width"] = width
    if height is not None:
        plan["height"] = height
    if length is not None:
        plan["length"] = 1 if task in {"t2i", "i2i"} else length
    if task in {"i2i", "r2v", "rv2v"}:
        references = [reference_image] if isinstance(reference_image, str) else reference_image
        if not references:
            raise ValueError(f"task {task} requires at least one reference image")
        graph["8"]["inputs"]["image"] = references[0]
        for index, filename in enumerate(references[1:], start=1):
            node_id = str(19 + index)
            graph[node_id] = {
                "inputs": {"image": filename},
                "class_type": "LoadImage",
                "_meta": {"title": f"Load Reference Image {index + 1}"},
            }
            plan[f"reference_images.reference_image_{index}"] = [node_id, 0]
    if task in {"v2v", "rv2v"}:
        graph["9"]["inputs"]["file"] = source_video
    if prompt is not None:
        plan["prompt"] = prompt
        graph["3"]["inputs"]["text"] = SYSTEM_PROMPTS.get(task, SYSTEM_PROMPTS["default"]) + prompt

    output_node = "18" if task in {"t2i", "i2i"} else "19"
    graph[output_node]["inputs"]["filename_prefix"] = output_prefix or f"Bernini-v2/quality/{task}_full_official"
    return graph


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", choices=TASKS)
    parser.add_argument("--url", default="http://127.0.0.1:8199")
    parser.add_argument("--width", type=int)
    parser.add_argument("--height", type=int)
    parser.add_argument("--length", type=int)
    parser.add_argument(
        "--reference-image",
        action="append",
        help="Reference image filename; repeat for multi-reference R2V/RV2V.",
    )
    parser.add_argument("--source-video", default="bernini_quality_source.mp4")
    parser.add_argument("--output-prefix")
    parser.add_argument("--prompt")
    parser.add_argument("--timeout", type=float, default=7200)
    parser.add_argument("--poll-interval", type=float, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    graph = prepare_graph(
        args.task,
        width=args.width,
        height=args.height,
        length=args.length,
        reference_image=args.reference_image or "bernini_quality_reference.png",
        source_video=args.source_video,
        output_prefix=args.output_prefix,
        prompt=args.prompt,
    )
    started = time.monotonic()
    result = run_smoke(
        args.url,
        graph,
        timeout=args.timeout,
        poll_interval=args.poll_interval,
    )
    result["elapsed_seconds"] = round(time.monotonic() - started, 3)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
