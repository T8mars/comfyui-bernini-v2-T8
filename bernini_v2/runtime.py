# Copyright (c) 2026 ByteDance Ltd. and/or its affiliate
# SPDX-License-Identifier: Apache-2.0
"""Comfy-managed runtime objects for the Bernini v2 semantic planner."""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn

from .planner_model import DiffLossFM, MLPConnector
from .qwen import install_qwen_vision_dtype_compat
from .sharded import load_sharded_state_dict


class PlannerAux(nn.Module):
    """Connector, VIT diffusion decoder, and learned mask-token table."""

    def __init__(
        self,
        *,
        num_mask_tokens: int = 4096,
        hidden_size: int = 3584,
        decoder_width: int = 4096,
        decoder_depth: int = 16,
        device=None,
        dtype=None,
        operations=None,
    ):
        super().__init__()
        self.connector = MLPConnector(
            in_dim=hidden_size,
            out_dim_for_gen=4096,
            out_dim_for_vit=hidden_size,
            device=device,
            dtype=dtype,
            operations=operations,
        )
        self.vit_decoder = DiffLossFM(
            target_channels=hidden_size,
            z_channels=hidden_size,
            depth=decoder_depth,
            width=decoder_width,
            shift=2.0,
            extra_one_step=True,
            device=device,
            dtype=dtype,
            operations=operations,
        )
        self.mask_tokens = nn.Parameter(
            torch.empty(1, num_mask_tokens, hidden_size, device=device, dtype=dtype),
            requires_grad=False,
        )


@dataclass
class BerniniV2PlannerRuntime:
    """A lightweight handle tracked by Comfy through its three model patchers."""

    language_model: nn.Module
    vision_model: nn.Module
    aux: PlannerAux
    language_patcher: object
    vision_patcher: object
    aux_patcher: object
    tokenizer: object
    dtype: torch.dtype
    load_device: torch.device

    def get_models(self) -> list[object]:
        return [self.language_patcher, self.vision_patcher, self.aux_patcher]

    def load_vision(self) -> None:
        import comfy.model_management

        comfy.model_management.load_models_gpu([self.vision_patcher])

    def load_planner(self) -> None:
        import comfy.model_management

        comfy.model_management.load_models_gpu([self.language_patcher, self.aux_patcher])


def _load_assign(module: nn.Module, state_dict: dict[str, torch.Tensor], label: str) -> None:
    result = module.load_state_dict(state_dict, strict=False, assign=True)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError(
            f"{label} checkpoint mismatch: missing={result.missing_keys[:12]}, unexpected={result.unexpected_keys[:12]}"
        )


def _component_index(root: Path, component: str) -> Path:
    path = root / component / "model.safetensors.index.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _qwen_config(config_path: Path) -> dict[str, object]:
    from comfy.text_encoders.llama import Qwen25_7BVLI_Config

    payload = json.loads(config_path.read_text(encoding="utf-8"))
    allowed = {field.name for field in dataclasses.fields(Qwen25_7BVLI_Config)}
    return {key: value for key, value in payload.items() if key in allowed}


def load_planner_runtime(root: str | Path, *, dtype: torch.dtype = torch.bfloat16) -> BerniniV2PlannerRuntime:
    """Load native Qwen/VIT planner weights from a streamed repack directory."""

    import comfy.model_management
    import comfy.ops
    from comfy.model_patcher import CoreModelPatcher
    from comfy.text_encoders.llama import Llama2_, Qwen25_7BVLI_Config
    from comfy.text_encoders.qwen_vl import Qwen2VLVisionTransformer
    from transformers import AutoTokenizer

    root = Path(root).resolve()
    config_path = root / "mllm" / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(
            f"{config_path} is missing; rerun tools/repack_diffusers.py so tokenizer metadata is copied"
        )

    config = _qwen_config(config_path)
    config_obj = Qwen25_7BVLI_Config(**config)
    raw_config = json.loads(config_path.read_text(encoding="utf-8"))
    rope_dims = raw_config.get("rope_scaling", {}).get("mrope_section")
    if rope_dims is not None and list(rope_dims) != list(config_obj.rope_dims):
        raise ValueError(
            f"checkpoint mRoPE sections {rope_dims} do not match ComfyUI Qwen config {config_obj.rope_dims}"
        )
    operations = comfy.ops.manual_cast
    language_model = Llama2_(config_obj, device="meta", dtype=dtype, ops=operations)
    vision_model = Qwen2VLVisionTransformer(
        hidden_size=1280,
        output_hidden_size=config_obj.hidden_size,
        intermediate_size=3420,
        num_heads=16,
        num_layers=32,
        patch_size=14,
        temporal_patch_size=2,
        spatial_merge_size=2,
        window_size=112,
        device="meta",
        dtype=dtype,
        ops=operations,
    )
    install_qwen_vision_dtype_compat(vision_model)
    aux = PlannerAux(device="meta", dtype=dtype, operations=operations)

    mllm_state = load_sharded_state_dict(_component_index(root, "mllm"))
    language_state = {
        key.removeprefix("model."): value for key, value in mllm_state.items() if key.startswith("model.")
    }
    vision_state = {
        key.removeprefix("visual."): value for key, value in mllm_state.items() if key.startswith("visual.")
    }
    _load_assign(language_model, language_state, "Qwen language model")
    _load_assign(vision_model, vision_state, "Qwen vision model")

    aux_state = {}
    connector_state = load_sharded_state_dict(_component_index(root, "connector"))
    aux_state.update({f"connector.{key}": value for key, value in connector_state.items()})
    decoder_state = load_sharded_state_dict(_component_index(root, "vit_decoder"))
    aux_state.update({f"vit_decoder.{key}": value for key, value in decoder_state.items()})
    aux_state.update(load_sharded_state_dict(_component_index(root, "mask_tokens")))
    _load_assign(aux, aux_state, "Bernini planner auxiliary")

    load_device = comfy.model_management.get_torch_device()
    offload_device = comfy.model_management.text_encoder_offload_device()
    language_patcher = CoreModelPatcher(language_model, load_device=load_device, offload_device=offload_device)
    vision_patcher = CoreModelPatcher(vision_model, load_device=load_device, offload_device=offload_device)
    aux_patcher = CoreModelPatcher(aux, load_device=load_device, offload_device=offload_device)
    tokenizer = AutoTokenizer.from_pretrained(root / "mllm", local_files_only=True, use_fast=True)
    return BerniniV2PlannerRuntime(
        language_model=language_model,
        vision_model=vision_model,
        aux=aux,
        language_patcher=language_patcher,
        vision_patcher=vision_patcher,
        aux_patcher=aux_patcher,
        tokenizer=tokenizer,
        dtype=dtype,
        load_device=load_device,
    )


def load_wan_t5(root: str | Path, *, dtype: torch.dtype = torch.bfloat16):
    """Load the official UMT5-XXL weights as a standard Comfy ``CLIP``."""

    import comfy.sd

    root = Path(root).resolve()
    state_dict = load_sharded_state_dict(_component_index(root, "t5_text_encoder"))
    tokenizer_path = root / "t5_tokenizer" / "spiece.model"
    if not tokenizer_path.is_file():
        raise FileNotFoundError(tokenizer_path)
    state_dict["spiece_model"] = torch.frombuffer(bytearray(tokenizer_path.read_bytes()), dtype=torch.uint8)
    return comfy.sd.load_text_encoder_state_dicts(
        [state_dict],
        clip_type=comfy.sd.CLIPType.WAN,
        model_options={"dtype": dtype},
        disable_dynamic=False,
    )
