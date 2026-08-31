"""Queue one reduced Bernini v2 workflow against a running ComfyUI server."""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
import uuid
from collections.abc import Callable
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW_DIR = ROOT / "examples" / "api"
TASKS = ("t2i", "i2i", "t2v", "v2v", "r2v", "rv2v")


def prepare_graph(
    task: str,
    *,
    width: int = 640,
    height: int = 368,
    length: int = 33,
    planning_steps: int = 1,
    vit_denoising_steps: int = 1,
    renderer_steps: int = 2,
    reference_image: str = "bernini_reference.png",
    source_video: str = "bernini_source.mp4",
    match_source_size: bool = False,
    output_prefix: str | None = None,
    repack_root: str | None = None,
) -> dict[str, object]:
    """Load a versioned example and apply deterministic smoke-test overrides."""

    if task not in TASKS:
        raise ValueError(f"unsupported task: {task}")
    graph = json.loads((WORKFLOW_DIR / f"{task}.json").read_text(encoding="utf-8"))
    if repack_root is not None:
        graph["1"]["inputs"]["repack_manifest"] = f"{repack_root}/repack-manifest.json"
        graph["2"]["inputs"]["repack_manifest"] = f"{repack_root}/repack-manifest.json"
        graph["5"]["inputs"]["model_index"] = f"{repack_root}/wan_high/model.safetensors.index.json"
        graph["6"]["inputs"]["model_index"] = f"{repack_root}/wan_low/model.safetensors.index.json"
    plan = graph["11"]["inputs"]
    plan.update(
        {
            "width": width,
            "height": height,
            "length": 1 if task in {"t2i", "i2i"} else length,
            "use_task_defaults": False,
            "match_source_size": match_source_size,
            "planning_steps": planning_steps,
            "vit_denoising_steps": vit_denoising_steps,
        }
    )
    graph["12"]["inputs"]["use_task_defaults"] = False
    graph["13"]["inputs"].update({"steps": renderer_steps, "use_task_defaults": False})

    if task in {"i2i", "r2v", "rv2v"}:
        graph["8"]["inputs"]["image"] = reference_image
    if task in {"v2v", "rv2v"}:
        graph["9"]["inputs"]["file"] = source_video

    output_node = "18" if task in {"t2i", "i2i"} else "19"
    prefix = output_prefix or f"Bernini-v2/smoke/{task}"
    graph[output_node]["inputs"]["filename_prefix"] = prefix
    return graph


def request_json(url: str, *, payload: dict[str, object] | None = None) -> dict[str, object]:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"} if data is not None else {},
        method="POST" if data is not None else "GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.load(response)
    except urllib.error.HTTPError as error:
        details = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"ComfyUI returned HTTP {error.code}: {details}") from error


def run_smoke(
    base_url: str,
    graph: dict[str, object],
    *,
    timeout: float,
    poll_interval: float,
    sample_callback: Callable[[], None] | None = None,
) -> dict[str, object]:
    base_url = base_url.rstrip("/")
    queued = request_json(
        f"{base_url}/prompt",
        payload={"prompt": graph, "client_id": str(uuid.uuid4())},
    )
    node_errors = queued.get("node_errors")
    if node_errors:
        raise RuntimeError(f"workflow validation failed: {json.dumps(node_errors, indent=2)}")
    prompt_id = queued.get("prompt_id")
    if not isinstance(prompt_id, str):
        raise RuntimeError(f"ComfyUI did not return a prompt id: {queued}")

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if sample_callback is not None:
            sample_callback()
        history = request_json(f"{base_url}/history/{prompt_id}")
        entry = history.get(prompt_id)
        if entry:
            status = entry.get("status", {})
            if status.get("status_str") == "error":
                raise RuntimeError(json.dumps(entry, indent=2))
            if status.get("completed"):
                if status.get("status_str") != "success":
                    raise RuntimeError(json.dumps(entry, indent=2))
                return {"prompt_id": prompt_id, "status": status, "outputs": entry.get("outputs", {})}
        time.sleep(poll_interval)
    raise TimeoutError(f"prompt {prompt_id} did not complete within {timeout:.0f}s")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", choices=TASKS)
    parser.add_argument("--url", default="http://127.0.0.1:8199")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=368)
    parser.add_argument("--length", type=int, default=33)
    parser.add_argument("--planning-steps", type=int, default=1)
    parser.add_argument("--vit-denoising-steps", type=int, default=1)
    parser.add_argument("--renderer-steps", type=int, default=2)
    parser.add_argument("--reference-image", default="bernini_reference.png")
    parser.add_argument("--source-video", default="bernini_source.mp4")
    parser.add_argument("--match-source-size", action="store_true")
    parser.add_argument("--output-prefix")
    parser.add_argument("--repack-root")
    parser.add_argument("--timeout", type=float, default=3600)
    parser.add_argument("--poll-interval", type=float, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    graph = prepare_graph(
        args.task,
        width=args.width,
        height=args.height,
        length=args.length,
        planning_steps=args.planning_steps,
        vit_denoising_steps=args.vit_denoising_steps,
        renderer_steps=args.renderer_steps,
        reference_image=args.reference_image,
        source_video=args.source_video,
        match_source_size=args.match_source_size,
        output_prefix=args.output_prefix,
        repack_root=args.repack_root,
    )
    result = run_smoke(args.url, graph, timeout=args.timeout, poll_interval=args.poll_interval)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
