"""Queue one full-quality Bernini v2 example against a running ComfyUI server."""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

from run_comfy_smoke import TASKS, request_json, run_smoke

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bernini_v2.template import SYSTEM_PROMPTS  # noqa: E402

WORKFLOW_DIR = ROOT / "examples" / "api"


class ResourceMonitor:
    """Best-effort peak VRAM and optional ComfyUI-process memory sampler."""

    def __init__(self, base_url: str, server_pid: int | None = None):
        self.base_url = base_url.rstrip("/")
        self.server_pid = server_pid
        self.peak_vram_used = 0
        self.peak_rss = 0
        self.peak_vms = 0
        self.samples = 0

    def sample(self) -> None:
        try:
            stats = request_json(f"{self.base_url}/system_stats")
            for device in stats.get("devices", []):
                total = int(device.get("vram_total", 0))
                free = int(device.get("vram_free", total))
                self.peak_vram_used = max(self.peak_vram_used, total - free)
        except (OSError, RuntimeError, ValueError):
            pass
        if self.server_pid is not None:
            try:
                import psutil

                root = psutil.Process(self.server_pid)
                processes = [root, *root.children(recursive=True)]
                rss = 0
                vms = 0
                for process in processes:
                    try:
                        memory = process.memory_info()
                    except (OSError, psutil.Error):
                        continue
                    rss += int(memory.rss)
                    vms += int(memory.vms)
                self.peak_rss = max(self.peak_rss, rss)
                self.peak_vms = max(self.peak_vms, vms)
            except ImportError:
                pass
            except (OSError, psutil.Error):
                pass
        self.samples += 1

    def report(self) -> dict[str, int | float]:
        return {
            "samples": self.samples,
            "peak_vram_used_bytes": self.peak_vram_used,
            "peak_vram_used_gib": round(self.peak_vram_used / 2**30, 3),
            "peak_process_rss_bytes": self.peak_rss,
            "peak_process_rss_gib": round(self.peak_rss / 2**30, 3),
            "peak_process_vms_bytes": self.peak_vms,
            "peak_process_vms_gib": round(self.peak_vms / 2**30, 3),
        }


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
    repack_root: str | None = None,
    high_renderer: str | None = None,
    low_renderer: str | None = None,
    renderer_loader: str = "bernini",
) -> dict[str, object]:
    """Load an example without disabling its published task defaults."""

    if task not in TASKS:
        raise ValueError(f"unsupported task: {task}")
    graph = json.loads((WORKFLOW_DIR / f"{task}.json").read_text(encoding="utf-8"))
    if repack_root is not None:
        graph["1"]["inputs"]["repack_manifest"] = f"{repack_root}/repack-manifest.json"
        graph["2"]["inputs"]["repack_manifest"] = f"{repack_root}/repack-manifest.json"
        graph["5"]["inputs"]["model_index"] = f"{repack_root}/wan_high/model.safetensors.index.json"
        graph["6"]["inputs"]["model_index"] = f"{repack_root}/wan_low/model.safetensors.index.json"
    if (high_renderer is None) != (low_renderer is None):
        raise ValueError("--high-renderer and --low-renderer must be supplied together")
    if high_renderer is not None and low_renderer is not None:
        if renderer_loader == "bernini":
            graph["5"]["inputs"]["model_index"] = high_renderer
            graph["6"]["inputs"]["model_index"] = low_renderer
        elif renderer_loader == "native":
            for node_id, filename, title in (
                ("5", high_renderer, "Load high-noise renderer"),
                ("6", low_renderer, "Load low-noise renderer"),
            ):
                graph[node_id] = {
                    "inputs": {"unet_name": str(Path(filename)), "weight_dtype": "default"},
                    "class_type": "UNETLoader",
                    "_meta": {"title": title},
                }
        elif renderer_loader == "gguf":
            for node_id, filename, title in (
                ("5", high_renderer, "Load high-noise GGUF renderer"),
                ("6", low_renderer, "Load low-noise GGUF renderer"),
            ):
                graph[node_id] = {
                    "inputs": {"unet_name": str(Path(filename))},
                    "class_type": "UnetLoaderGGUF",
                    "_meta": {"title": title},
                }
        else:
            raise ValueError(f"unsupported renderer loader: {renderer_loader}")
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
    parser.add_argument("--repack-root")
    parser.add_argument(
        "--high-renderer",
        help="High-noise renderer path/name, paired with --low-renderer.",
    )
    parser.add_argument(
        "--low-renderer",
        help="Low-noise renderer path/name, paired with --high-renderer.",
    )
    parser.add_argument(
        "--renderer-loader",
        choices=("bernini", "native", "gguf"),
        default="bernini",
        help="Loader for explicit renderer overrides.",
    )
    parser.add_argument("--timeout", type=float, default=7200)
    parser.add_argument("--poll-interval", type=float, default=2)
    parser.add_argument(
        "--repeat", type=int, default=1, help="Queue identical jobs consecutively in one server process"
    )
    parser.add_argument(
        "--server-pid",
        type=int,
        help="Optional ComfyUI root PID for process-tree RSS/commit sampling",
    )
    parser.add_argument("--metrics-out", type=Path, help="Optional JSON result path")
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
        repack_root=args.repack_root,
        high_renderer=args.high_renderer,
        low_renderer=args.low_renderer,
        renderer_loader=args.renderer_loader,
    )
    if args.repeat < 1:
        raise ValueError("--repeat must be at least 1")
    monitor = ResourceMonitor(args.url, args.server_pid)
    results = []
    for run_number in range(1, args.repeat + 1):
        run_graph = copy.deepcopy(graph)
        output_node = "18" if args.task in {"t2i", "i2i"} else "19"
        if args.repeat > 1:
            prefix = run_graph[output_node]["inputs"]["filename_prefix"]
            run_graph[output_node]["inputs"]["filename_prefix"] = f"{prefix}/run-{run_number}"
        started = time.monotonic()
        result = run_smoke(
            args.url,
            run_graph,
            timeout=args.timeout,
            poll_interval=args.poll_interval,
            sample_callback=monitor.sample,
        )
        result["run"] = run_number
        result["elapsed_seconds"] = round(time.monotonic() - started, 3)
        results.append(result)
    payload = {
        "task": args.task,
        "repeat": args.repeat,
        "renderer": {
            "loader": args.renderer_loader,
            "high": args.high_renderer,
            "low": args.low_renderer,
        },
        "runs": results,
        "resources": monitor.report(),
    }
    if args.metrics_out is not None:
        args.metrics_out.parent.mkdir(parents=True, exist_ok=True)
        args.metrics_out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
