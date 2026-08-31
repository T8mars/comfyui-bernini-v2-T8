"""Native sharded Wan loader for Bernini v2 repacks."""

from __future__ import annotations

import logging
import os

import comfy.model_sampling
import comfy.ops
import comfy.sd
import comfy.utils
import folder_paths
import torch
from comfy_api.latest import ComfyExtension, io
from typing_extensions import override

from ..bernini_v2.manifest import load_repack_manifest
from ..bernini_v2.runtime import load_planner_runtime, load_wan_t5
from ..bernini_v2.sharded import load_checkpoint_state_dict

MODEL_FOLDER = "bernini_v2"
MODEL_ROOT = os.path.join(folder_paths.models_dir, MODEL_FOLDER)
folder_paths.add_model_folder_path(MODEL_FOLDER, MODEL_ROOT, is_default=True)
# This folder is index-oriented, not a generic PyTorch checkpoint directory.
folder_paths.folder_names_and_paths[MODEL_FOLDER][1].add(".json")
folder_paths.folder_names_and_paths[MODEL_FOLDER][1].add(".safetensors")
BerniniV2PlannerType = io.Custom("BERNINI_V2_PLANNER")


def _portable_name(name: str) -> str:
    """Keep serialized workflow model names stable across host platforms."""
    return name.replace("\\", "/")


def _index_options() -> list[str]:
    return [
        _portable_name(name)
        for name in folder_paths.get_filename_list(MODEL_FOLDER)
        if (
            name.endswith("model.safetensors.index.json")
            or (name.endswith(".safetensors") and not (os.path.basename(name).startswith("model-") and "-of-" in name))
        )
        and any(part in name.replace("\\", "/").split("/") for part in ("wan_high", "wan_low"))
    ]


def _manifest_options() -> list[str]:
    return [
        _portable_name(name)
        for name in folder_paths.get_filename_list(MODEL_FOLDER)
        if name.endswith("repack-manifest.json")
    ]


def _repack_root(manifest_name: str) -> str:
    manifest_path = folder_paths.get_full_path_or_raise(MODEL_FOLDER, manifest_name)
    load_repack_manifest(manifest_path)
    return os.path.dirname(manifest_path)


class BerniniV2WanLoader(io.ComfyNode):
    """Load a repacked high- or low-noise renderer with Comfy's Wan model."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="BerniniV2WanLoader",
            display_name="Load Bernini v2 Wan Renderer",
            category="advanced/loaders/bernini_v2",
            description=(
                "Loads a sharded native-Comfy Wan renderer produced by tools/repack_diffusers.py. "
                "The output is a standard MODEL and uses ComfyUI device/offload management."
            ),
            inputs=[
                io.Combo.Input("model_index", options=_index_options()),
                io.Float.Input(
                    "flow_shift",
                    default=5.0,
                    min=0.01,
                    max=100.0,
                    step=0.01,
                    advanced=True,
                ),
                io.Combo.Input(
                    "weight_dtype",
                    options=["bfloat16", "float16", "default", "fp8_e4m3fn", "fp8_e4m3fn_fast", "fp8_e5m2"],
                    default="bfloat16",
                    optional=True,
                    advanced=True,
                ),
            ],
            outputs=[io.Model.Output()],
        )

    @classmethod
    def execute(
        cls,
        model_index: str,
        flow_shift: float = 5.0,
        weight_dtype: str = "default",
    ) -> io.NodeOutput:
        index_path = folder_paths.get_full_path_or_raise(MODEL_FOLDER, model_index)
        state_dict = load_checkpoint_state_dict(index_path)
        if "scaled_fp8" in state_dict:
            state_dict, _ = comfy.utils.convert_old_quants(state_dict)
        native_quant = any(key.endswith(".comfy_quant") for key in state_dict)
        if native_quant and weight_dtype.startswith("fp8"):
            raise ValueError("pre-quantized Comfy weights cannot be recast to FP8; choose bfloat16 or default")
        model_options = {}
        if weight_dtype == "bfloat16":
            model_options["dtype"] = torch.bfloat16
        elif weight_dtype == "float16":
            model_options["dtype"] = torch.float16
        elif weight_dtype == "fp8_e4m3fn":
            model_options["dtype"] = torch.float8_e4m3fn
        elif weight_dtype == "fp8_e4m3fn_fast":
            model_options["dtype"] = torch.float8_e4m3fn
            model_options["fp8_optimizations"] = True
        elif weight_dtype == "fp8_e5m2":
            model_options["dtype"] = torch.float8_e5m2
        if native_quant:
            compute_dtype = torch.float16 if weight_dtype == "float16" else torch.bfloat16
            model_options["dtype"] = compute_dtype
            model_options["custom_operations"] = comfy.ops.mixed_precision_ops({}, compute_dtype)
            logging.info("Bernini v2 quantized renderer compute dtype: %s", compute_dtype)
        model = comfy.sd.load_diffusion_model_state_dict(state_dict, model_options=model_options)
        if model is None:
            raise RuntimeError(f"ComfyUI could not detect the Wan model in {index_path}")
        model = model.clone()

        class BerniniV2ModelSampling(comfy.model_sampling.ModelSamplingDiscreteFlow, comfy.model_sampling.CONST):
            pass

        model_sampling = BerniniV2ModelSampling(model.model.model_config)
        model_sampling.set_parameters(shift=flow_shift, multiplier=1000)
        model.add_object_patch("model_sampling", model_sampling)
        return io.NodeOutput(model)


class BerniniV2PlannerLoader(io.ComfyNode):
    """Load Qwen2.5-VL and Bernini planning heads under Comfy model management."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="BerniniV2PlannerLoader",
            display_name="Load Bernini v2 Planner",
            category="advanced/loaders/bernini_v2",
            description=(
                "Loads native Comfy Qwen language/vision modules plus the Bernini connector and VIT decoder. "
                "The three parts remain independently offloadable."
            ),
            inputs=[
                io.Combo.Input("repack_manifest", options=_manifest_options()),
                io.Combo.Input(
                    "dtype",
                    options=["bfloat16", "float16"],
                    default="bfloat16",
                    optional=True,
                    advanced=True,
                ),
            ],
            outputs=[BerniniV2PlannerType.Output(display_name="planner")],
        )

    @classmethod
    def execute(cls, repack_manifest: str, dtype: str = "bfloat16") -> io.NodeOutput:
        torch_dtype = torch.bfloat16 if dtype == "bfloat16" else torch.float16
        planner = load_planner_runtime(_repack_root(repack_manifest), dtype=torch_dtype)
        return io.NodeOutput(planner)


class BerniniV2T5Loader(io.ComfyNode):
    """Load Bernini's official UMT5-XXL as a standard Comfy CLIP object."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="BerniniV2T5Loader",
            display_name="Load Bernini v2 T5",
            category="advanced/loaders/bernini_v2",
            inputs=[
                io.Combo.Input("repack_manifest", options=_manifest_options()),
                io.Combo.Input(
                    "dtype",
                    options=["bfloat16", "float16"],
                    default="bfloat16",
                    optional=True,
                    advanced=True,
                ),
            ],
            outputs=[io.Clip.Output()],
        )

    @classmethod
    def execute(cls, repack_manifest: str, dtype: str = "bfloat16") -> io.NodeOutput:
        torch_dtype = torch.bfloat16 if dtype == "bfloat16" else torch.float16
        return io.NodeOutput(load_wan_t5(_repack_root(repack_manifest), dtype=torch_dtype))


class BerniniV2LoaderExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [BerniniV2WanLoader, BerniniV2PlannerLoader, BerniniV2T5Loader]
