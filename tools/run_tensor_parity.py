#!/usr/bin/env python3
"""Capture and compare official/native Bernini v2 intermediate tensors.

Each implementation is captured in a separate process so a 24 GB GPU never
has to hold both copies of a large component at once.
"""

from __future__ import annotations

import argparse
import gc
import importlib.machinery
import importlib.util
import json
import math
import subprocess
import sys
import types
from pathlib import Path

import torch
from accelerate import init_empty_weights
from safetensors import safe_open

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
DEFAULT_PLANNER = ROOT / "models" / "single-files" / "bf16" / "bernini_v2_planner_bf16.safetensors"
DEFAULT_HIGH_RENDERER = ROOT / "models" / "single-files" / "bf16" / "bernini_v2_high_noise_bf16.safetensors"
DEFAULT_LOW_RENDERER = ROOT / "models" / "single-files" / "bf16" / "bernini_v2_low_noise_bf16.safetensors"
DEFAULT_OFFICIAL_MODEL = ROOT / "models" / "ByteDance" / "Bernini-Diffusers-v2"
DEFAULT_OFFICIAL_REPO = ROOT / ".upstream" / "Bernini"
DEFAULT_VEOMNI_REPO = ROOT / ".tools" / "VeOmni-v0.1.11"
DEFAULT_COMFY_ROOT = Path(r"E:\comfyui-t8-onekey-5x\ComfyUI")
DEFAULT_OUTPUT = ROOT / "artifacts" / "parity"
PLANNER_SHA256 = "686437fda8400ca1ee69f8436c2d546334f781360a8bc1416467845436877f3f"
STAGES = ("planner_hidden", "vit_target", "wan_prediction")


def _prepend_import_path(path: Path) -> None:
    resolved = str(path.resolve())
    if resolved not in sys.path:
        sys.path.insert(0, resolved)


def _install_unused_flash_attn_stub() -> None:
    """Let the official SDPA implementation import without FlashAttention.

    ByteDance's module rejects import when neither FA2 nor FA3 is installed,
    even when its configured attention implementation is SDPA. The functions
    below deliberately fail if execution unexpectedly reaches a flash path.
    """

    if importlib.util.find_spec("flash_attn") is not None:
        return

    def unavailable(*args, **kwargs):
        del args, kwargs
        raise RuntimeError("the tensor parity oracle is configured for SDPA, not FlashAttention")

    package = types.ModuleType("flash_attn")
    package.__path__ = []
    package.__spec__ = importlib.machinery.ModuleSpec("flash_attn", loader=None, is_package=True)
    package.flash_attn_func = unavailable
    package.flash_attn_varlen_func = unavailable
    padding = types.ModuleType("flash_attn.bert_padding")
    padding.__spec__ = importlib.machinery.ModuleSpec("flash_attn.bert_padding", loader=None)
    padding.unpad_input = unavailable
    sys.modules["flash_attn"] = package
    sys.modules["flash_attn.bert_padding"] = padding


def _prepare_official_imports(official_repo: Path, veomni_repo: Path) -> None:
    _prepend_import_path(veomni_repo)
    _prepend_import_path(official_repo)
    _install_unused_flash_attn_stub()


def _planner_config(checkpoint: Path) -> dict[str, object]:
    with safe_open(checkpoint, framework="pt", device="cpu") as handle:
        encoded = handle.get_tensor("config_json")
    return json.loads(encoded.numpy().tobytes().decode("utf-8"))


def _prefixed_state(checkpoint: Path, prefix: str) -> dict[str, torch.Tensor]:
    with safe_open(checkpoint, framework="pt", device="cpu") as handle:
        names = [name for name in handle.keys() if name.startswith(prefix)]
        return {name.removeprefix(prefix): handle.get_tensor(name) for name in names}


def _load_assign(module: torch.nn.Module, state: dict[str, torch.Tensor], label: str) -> None:
    result = module.load_state_dict(state, strict=True, assign=True)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError(
            f"{label} state mismatch: missing={result.missing_keys[:8]}, unexpected={result.unexpected_keys[:8]}"
        )


def _fixture_hidden(
    seed: int,
    sequence_length: int,
    vocab_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    token_ids = torch.randint(0, vocab_size, (1, sequence_length), generator=generator, dtype=torch.long)
    positions = torch.arange(sequence_length, dtype=torch.long).view(1, 1, -1).expand(3, 1, -1).clone()
    attention = torch.zeros(1, sequence_length, sequence_length, dtype=torch.bfloat16)
    blocked = torch.triu(torch.ones(sequence_length, sequence_length, dtype=torch.bool), diagonal=1)
    attention[:, blocked] = torch.finfo(torch.bfloat16).min
    return token_ids, positions, attention


def _fixture_vit_hidden(seed: int, token_count: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randn(3, token_count, 3584, generator=generator, dtype=torch.float32).to(torch.bfloat16)


def _wan_checkpoint(args: argparse.Namespace) -> Path:
    return args.high_renderer if args.renderer_expert == "high" else args.low_renderer


def _wan_timestep(args: argparse.Namespace) -> float:
    if args.wan_timestep is not None:
        return args.wan_timestep
    return 999.0 if args.renderer_expert == "high" else 500.0


def _fixture_wan(args: argparse.Namespace) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    latent = torch.randn(
        1,
        16,
        args.latent_frames,
        args.latent_height,
        args.latent_width,
        generator=generator,
        dtype=torch.float32,
    ).to(torch.bfloat16)
    context = torch.randn(
        1,
        args.wan_context_length,
        4096,
        generator=generator,
        dtype=torch.float32,
    ).to(torch.bfloat16)
    timestep = torch.tensor([_wan_timestep(args)], dtype=torch.float32)
    return latent, context, timestep


def _unpatch_wan(tokens: torch.Tensor, grid_size: tuple[int, int, int]) -> torch.Tensor:
    patch_size = (1, 2, 2)
    batch = tokens.shape[0]
    output = tokens[:, : math.prod(grid_size)].view(batch, *grid_size, *patch_size, 16)
    output = torch.einsum("bfhwpqrc->bcfphqwr", output)
    return output.reshape(batch, 16, *(grid * patch for grid, patch in zip(grid_size, patch_size, strict=True)))


def _load_streamed_module(
    module: torch.nn.Module,
    checkpoint,
    prefix: str,
    *,
    official_names: bool,
    label: str,
) -> None:
    from bernini_v2.wan_mapping import wan_diffusers_to_comfy

    state = {}
    available = set(checkpoint.keys())
    for local_name in module.state_dict():
        target_name = f"{prefix}{local_name}"
        source_name = wan_diffusers_to_comfy(target_name) if official_names else target_name
        if source_name not in available:
            raise RuntimeError(f"{label} is missing source tensor {source_name}")
        state[local_name] = checkpoint.get_tensor(source_name)
    _load_assign(module, state, label)


def _move_streamed_module(module: torch.nn.Module, device: torch.device) -> torch.nn.Module:
    return module.eval().to(device=device, dtype=torch.bfloat16)


def _release_streamed_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _attach_layer_probes(layer: torch.nn.Module, captured: dict[str, torch.Tensor]):
    modules = {
        "probe_input_norm": layer.input_layernorm,
        "probe_q": layer.self_attn.q_proj,
        "probe_k": layer.self_attn.k_proj,
        "probe_v": layer.self_attn.v_proj,
        "probe_attention": layer.self_attn,
        "probe_o": layer.self_attn.o_proj,
        "probe_post_norm": layer.post_attention_layernorm,
        "probe_gate": layer.mlp.gate_proj,
        "probe_up": layer.mlp.up_proj,
        "probe_down": layer.mlp.down_proj,
    }

    def hook(name):
        def capture_output(module, inputs, output):
            del module, inputs
            value = output[0] if isinstance(output, (tuple, list)) else output
            captured[name] = value.detach().cpu().contiguous().clone()

        return capture_output

    return [module.register_forward_hook(hook(name)) for name, module in modules.items()]


def _official_sdpa_adapter(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    heads: int,
    *,
    mask=None,
    skip_reshape=False,
    **kwargs,
) -> torch.Tensor:
    """Use the official Qwen SDPA layout inside Comfy's attention callback contract."""

    if not skip_reshape:
        raise ValueError("the Qwen parity adapter requires head-separated Q/K/V")
    del heads, kwargs
    if key.shape[1] != query.shape[1]:
        repeats = query.shape[1] // key.shape[1]
        key = key[:, :, None, :, :].expand(-1, -1, repeats, -1, -1).reshape_as(query)
        value = value[:, :, None, :, :].expand(-1, -1, repeats, -1, -1).reshape_as(query)
    output = torch.nn.functional.scaled_dot_product_attention(
        query.contiguous(),
        key.contiguous(),
        value.contiguous(),
        attn_mask=mask,
        dropout_p=0.0,
        is_causal=False,
    )
    return output.transpose(1, 2).contiguous().reshape(query.shape[0], query.shape[2], -1)


def _repeat_gqa_adapter(attention):
    def wrapped(query, key, value, heads, **kwargs):
        if kwargs.pop("enable_gqa", False) and key.shape[1] != query.shape[1]:
            repeats = query.shape[1] // key.shape[1]
            key = key[:, :, None, :, :].expand(-1, -1, repeats, -1, -1).reshape_as(query)
            value = value[:, :, None, :, :].expand(-1, -1, repeats, -1, -1).reshape_as(query)
        return attention(query, key, value, heads, **kwargs)

    return wrapped


def _capture_official_hidden(args: argparse.Namespace, device: torch.device) -> dict[str, torch.Tensor]:
    _prepare_official_imports(args.official_repo, args.veomni_repo)
    from bernini.models.modeling_qwen2_5_vl import Qwen2_5_VLModel, Qwen2_5_VLRotaryEmbedding
    from transformers import Qwen2_5_VLConfig

    config = Qwen2_5_VLConfig.from_dict(_planner_config(args.planner))
    config._attn_implementation = "sdpa"
    config.use_cache = False
    with init_empty_weights(include_buffers=True):
        model = Qwen2_5_VLModel(config)
    model.rotary_emb = Qwen2_5_VLRotaryEmbedding(config=config)
    for layer in model.layers:
        layer.self_attn.rotary_emb = Qwen2_5_VLRotaryEmbedding(config=config)
    state = _prefixed_state(args.planner, "model.")
    _load_assign(model, state, "official Qwen language model")
    del state
    model = model.eval().to(device=device, dtype=torch.bfloat16)
    token_ids, positions, attention = _fixture_hidden(args.seed, args.sequence_length, config.vocab_size)
    probes: dict[str, torch.Tensor] = {}
    handles = _attach_layer_probes(model.layers[0], probes)
    with torch.no_grad():
        inputs = model.embed_tokens(token_ids.to(device)).to(torch.bfloat16)
        outputs = model(
            inputs_embeds=inputs,
            position_ids=positions.to(device),
            attention_mask=attention.unsqueeze(1).to(device),
            use_cache=False,
            output_hidden_states=True,
            return_dict=True,
        ).hidden_states
    for handle in handles:
        handle.remove()
    tensors = {
        "attention_mask": attention,
        "hidden_state": outputs[-2].cpu(),
        "input_embeds": inputs.cpu(),
        "position_ids": positions,
        "token_ids": token_ids,
    }
    tensors.update({f"layer_{index:02d}": outputs[index + 1].cpu() for index in range(len(model.layers) - 1)})
    tensors.update(probes)
    return tensors


def _capture_native_hidden(args: argparse.Namespace, device: torch.device) -> dict[str, torch.Tensor]:
    _prepend_import_path(args.comfy_root)
    import comfy.ops
    from comfy.text_encoders.llama import Llama2_, Qwen25_7BVLI_Config

    from bernini_v2.qwen import (
        _explicit_gqa_attention,
        _official_qwen_layer_forward,
        _official_qwen_rms_norm,
    )
    from bernini_v2.runtime import _qwen_config

    config = Qwen25_7BVLI_Config(**_qwen_config(_planner_config(args.planner)))
    model = Llama2_(config, device="meta", dtype=torch.bfloat16, ops=comfy.ops.manual_cast)
    state = _prefixed_state(args.planner, "model.")
    _load_assign(model, state, "native Qwen language model")
    del state
    model = model.eval().to(device=device, dtype=torch.bfloat16)
    token_ids, positions, attention = _fixture_hidden(args.seed, args.sequence_length, config.vocab_size)
    probes: dict[str, torch.Tensor] = {}
    handles = _attach_layer_probes(model.layers[0], probes)
    with torch.no_grad():
        from comfy.ldm.modules.attention import attention_basic, attention_pytorch, optimized_attention_for_device

        inputs = model.embed_tokens(token_ids.to(device), out_dtype=torch.bfloat16)
        position_device = positions.to(device)
        native_positions = position_device.squeeze(1) if position_device.shape[1] == 1 else position_device
        frequencies = model.compute_freqs_cis(native_positions, device)
        if args.native_rope_dtype == "input":
            frequencies = tuple(value.to(inputs.dtype) for value in frequencies)
        mask = attention.to(device).unsqueeze(1)
        if args.native_attention == "pytorch":
            optimized_attention = attention_pytorch
        elif args.native_attention == "optimized":
            optimized_attention = optimized_attention_for_device(device, mask=True, small_input=False)
        elif args.native_attention == "basic":
            optimized_attention = attention_basic
        elif args.native_attention == "official_sdpa":
            optimized_attention = _official_sdpa_adapter
        elif args.native_attention == "pytorch_repeat":
            optimized_attention = _repeat_gqa_adapter(attention_pytorch)
        elif args.native_attention == "runtime_repeat":
            optimized_attention = _repeat_gqa_adapter(
                optimized_attention_for_device(device, mask=True, small_input=True)
            )
        else:
            optimized_attention = _explicit_gqa_attention(
                optimized_attention_for_device(device, mask=True, small_input=True)
            )
        hidden = inputs.clone()
        layers = []
        for index, layer in enumerate(model.layers):
            layer_input = hidden
            if index == 0:
                probes["probe_input_norm"] = _official_qwen_rms_norm(layer.input_layernorm, layer_input).cpu()
            hidden, _ = _official_qwen_layer_forward(layer, hidden, mask, frequencies, optimized_attention)
            if index == 0:
                attention_output = probes["probe_attention"].to(device)
                probes["probe_post_norm"] = _official_qwen_rms_norm(
                    layer.post_attention_layernorm,
                    layer_input + attention_output,
                ).cpu()
            layers.append(hidden.cpu())
    for handle in handles:
        handle.remove()
    tensors = {
        "attention_mask": attention,
        "hidden_state": layers[-2],
        "input_embeds": inputs.cpu(),
        "position_ids": positions,
        "token_ids": token_ids,
    }
    tensors.update({f"layer_{index:02d}": value for index, value in enumerate(layers[:-1])})
    tensors.update(probes)
    return tensors


def _official_aux(args: argparse.Namespace, device: torch.device):
    _prepare_official_imports(args.official_repo, args.veomni_repo)
    from bernini.models.bernini import MLPConnector
    from bernini.models.diffloss_fm import DiffLoss_FM

    with init_empty_weights(include_buffers=True):
        connector = MLPConnector(in_dim=3584, out_dim_for_gen=4096, out_dim_for_vit=3584)
        decoder = DiffLoss_FM(
            target_channels=3584,
            z_channels=3584,
            depth=16,
            width=4096,
            shift=2.0,
            extra_one_step=True,
        )
    connector_state = _prefixed_state(args.planner, "connector.")
    decoder_state = _prefixed_state(args.planner, "vit_decoder.")
    _load_assign(connector, connector_state, "official connector")
    _load_assign(decoder, decoder_state, "official VIT decoder")
    del connector_state, decoder_state
    return (
        connector.eval().to(device=device, dtype=torch.bfloat16),
        decoder.eval().to(device=device, dtype=torch.bfloat16),
    )


def _native_aux(args: argparse.Namespace, device: torch.device):
    from bernini_v2.planner_model import DiffLossFM, MLPConnector

    connector = MLPConnector(device="meta", dtype=torch.bfloat16)
    decoder = DiffLossFM(device="meta", dtype=torch.bfloat16)
    connector_state = _prefixed_state(args.planner, "connector.")
    decoder_state = _prefixed_state(args.planner, "vit_decoder.")
    _load_assign(connector, connector_state, "native connector")
    _load_assign(decoder, decoder_state, "native VIT decoder")
    del connector_state, decoder_state
    return (
        connector.eval().to(device=device, dtype=torch.bfloat16),
        decoder.eval().to(device=device, dtype=torch.bfloat16),
    )


def _capture_vit(args: argparse.Namespace, device: torch.device) -> dict[str, torch.Tensor]:
    if args.implementation == "official":
        connector, decoder = _official_aux(args, device)
    else:
        connector, decoder = _native_aux(args, device)
    hidden = _fixture_vit_hidden(args.seed, args.token_count)
    with torch.no_grad():
        projected = [connector.for_vit(branch.unsqueeze(0).to(device)) for branch in hidden]
        decoder_condition = torch.cat(projected, dim=1)[0]
        if args.implementation == "official":
            torch.manual_seed(args.seed)
            sampled = decoder.sample(
                z=decoder_condition,
                cfg=args.vit_text_cfg,
                img_cfg=args.vit_image_cfg,
                num_inference_steps=args.vit_steps,
                verbose=False,
            )
        else:
            generator = torch.Generator(device="cpu").manual_seed(args.seed)
            sampled = decoder.sample(
                decoder_condition,
                cfg=args.vit_text_cfg,
                img_cfg=args.vit_image_cfg,
                num_inference_steps=args.vit_steps,
                generator=generator,
            )
        target = sampled[: args.token_count]
    return {
        "connector_context": decoder_condition.cpu(),
        "hidden_inputs": hidden,
        "vit_target": target.cpu(),
    }


def _capture_official_wan(args: argparse.Namespace, device: torch.device) -> dict[str, torch.Tensor]:
    _prepare_official_imports(args.official_repo, args.veomni_repo)
    import bernini.attention as official_attention
    from bernini.models.transformer_wan import (
        WanRotaryPosEmbed,
        WanTransformer3DModel,
        WanTransformerBlock,
    )

    official_attention._BACKEND = "sdpa"
    official_attention._flash_varlen = None
    config_name = "transformer_config.json" if args.renderer_expert == "high" else "transformer_2_config.json"
    config = json.loads((args.official_model / config_name).read_text(encoding="utf-8"))
    config = {key: value for key, value in config.items() if not key.startswith("_") and key != "pos_embed_seq_len"}
    config["use_src_id_rotary_emb"] = True

    latent_cpu, context_cpu, timestep_cpu = _fixture_wan(args)
    latent = latent_cpu.to(device)
    context = context_cpu.to(device)
    timestep = timestep_cpu.to(device)
    checkpoint_path = _wan_checkpoint(args)
    captured: dict[str, torch.Tensor] = {
        "context_input": context_cpu,
        "latent_input": latent_cpu,
        "timestep": timestep_cpu,
    }

    with init_empty_weights(include_buffers=True):
        model = WanTransformer3DModel(**config)
    model.rope = WanRotaryPosEmbed(
        config["attention_head_dim"],
        tuple(config["patch_size"]),
        config["rope_max_seq_len"],
        use_src_id_rotary_emb=True,
    )

    with safe_open(checkpoint_path, framework="pt", device="cpu") as checkpoint:
        _load_streamed_module(
            model.patch_embedding,
            checkpoint,
            "patch_embedding.",
            official_names=True,
            label="official Wan patch embedding",
        )
        _load_streamed_module(
            model.condition_embedder,
            checkpoint,
            "condition_embedder.",
            official_names=True,
            label="official Wan condition embedder",
        )
        _load_streamed_module(
            model.proj_out,
            checkpoint,
            "proj_out.",
            official_names=True,
            label="official Wan output projection",
        )
        model.scale_shift_table = torch.nn.Parameter(checkpoint.get_tensor("head.modulation"))

        patch_embedding = _move_streamed_module(model.patch_embedding, device)
        condition_embedder = _move_streamed_module(model.condition_embedder, device)
        output_projection = _move_streamed_module(model.proj_out, device)
        model.scale_shift_table = torch.nn.Parameter(model.scale_shift_table.to(device=device, dtype=torch.bfloat16))

        with torch.no_grad():
            rotary = model.rope(latent, source_id=0).transpose(1, 2)
            hidden = patch_embedding(latent).flatten(2).transpose(1, 2)
            temb, block_modulation, text_context, _ = condition_embedder(timestep, context, None)
            block_modulation = block_modulation.unflatten(1, (6, -1))
            token_count = hidden.shape[1]
            block_modulation = block_modulation[:, None].expand(-1, token_count, -1, -1)
            output_modulation = temb[:, None].expand(-1, token_count, -1)
            captured.update(
                {
                    "block_modulation": block_modulation.cpu(),
                    "output_modulation": output_modulation.cpu(),
                    "patch_tokens": hidden.cpu(),
                    "text_context": text_context.cpu(),
                }
            )

            cu_query = torch.tensor([0, token_count], dtype=torch.int32, device=device)
            cu_context = torch.tensor([0, text_context.shape[1]], dtype=torch.int32, device=device)
            block_kwargs = {
                "cu_seqlens_q_cache": cu_query,
                "max_seqlen_q_cache": token_count,
                "cu_seqlens_k_cross_cache": cu_context,
                "max_seqlen_k_cross_cache": text_context.shape[1],
                "cu_seqlens_q_cross_cache": cu_query,
                "max_seqlen_q_cross_cache": token_count,
                "split_hidden_states_seq_len": None,
            }
            for index in range(args.wan_blocks):
                with init_empty_weights(include_buffers=True):
                    block = WanTransformerBlock(
                        config["num_attention_heads"] * config["attention_head_dim"],
                        config["ffn_dim"],
                        config["num_attention_heads"],
                        config["qk_norm"],
                        config["cross_attn_norm"],
                        config["eps"],
                        config["added_kv_proj_dim"],
                    )
                _load_streamed_module(
                    block,
                    checkpoint,
                    f"blocks.{index}.",
                    official_names=True,
                    label=f"official Wan block {index}",
                )
                block = _move_streamed_module(block, device)
                hidden = block(
                    hidden,
                    text_context,
                    block_modulation,
                    rotary,
                    batch_image_vae_seqlen=[token_count],
                    text_features_length=[text_context.shape[1]],
                    origin_hidden_states_seq_len=None,
                    **block_kwargs,
                )
                captured[f"block_{index:02d}"] = hidden.cpu()
                del block
                _release_streamed_memory()

            shift_table, scale_table = model.scale_shift_table.float().chunk(2, dim=1)
            shift = shift_table + output_modulation.float()
            scale = scale_table + output_modulation.float()
            hidden = (model.norm_out(hidden.float()) * (1 + scale) + shift).to(hidden.dtype)
            packed = output_projection(hidden)
            grid_size = (
                args.latent_frames,
                args.latent_height // 2,
                args.latent_width // 2,
            )
            captured["packed_prediction"] = packed.cpu()
            captured["prediction"] = _unpatch_wan(packed, grid_size).cpu()
    return captured


def _capture_native_wan(args: argparse.Namespace, device: torch.device) -> dict[str, torch.Tensor]:
    _prepend_import_path(args.comfy_root)
    import comfy.ops
    from comfy.ldm.wan.model import WanAttentionBlock, WanModel, sinusoidal_embedding_1d

    latent_cpu, context_cpu, timestep_cpu = _fixture_wan(args)
    latent = latent_cpu.to(device)
    context = context_cpu.to(device)
    timestep = timestep_cpu.to(device)
    checkpoint_path = _wan_checkpoint(args)
    captured: dict[str, torch.Tensor] = {
        "context_input": context_cpu,
        "latent_input": latent_cpu,
        "timestep": timestep_cpu,
    }
    model_args = {
        "model_type": "t2v",
        "patch_size": (1, 2, 2),
        "in_dim": 16,
        "dim": 5120,
        "ffn_dim": 13824,
        "freq_dim": 256,
        "text_dim": 4096,
        "out_dim": 16,
        "num_heads": 40,
        "num_layers": 40,
        "qk_norm": True,
        "cross_attn_norm": True,
        "eps": 1e-6,
        "operations": comfy.ops.disable_weight_init,
        "device": "meta",
        "dtype": torch.bfloat16,
    }
    model = WanModel(**model_args)

    with safe_open(checkpoint_path, framework="pt", device="cpu") as checkpoint:
        for module, prefix, label in (
            (model.patch_embedding, "patch_embedding.", "native Wan patch embedding"),
            (model.text_embedding, "text_embedding.", "native Wan text embedding"),
            (model.time_embedding, "time_embedding.", "native Wan time embedding"),
            (model.time_projection, "time_projection.", "native Wan time projection"),
            (model.head, "head.", "native Wan output head"),
        ):
            _load_streamed_module(module, checkpoint, prefix, official_names=False, label=label)

        patch_embedding = _move_streamed_module(model.patch_embedding, device).float()
        text_embedding = _move_streamed_module(model.text_embedding, device)
        time_embedding = _move_streamed_module(model.time_embedding, device)
        time_projection = _move_streamed_module(model.time_projection, device)
        output_head = _move_streamed_module(model.head, device)

        with torch.no_grad():
            patch_volume = patch_embedding(latent.float()).to(latent.dtype)
            grid_size = tuple(patch_volume.shape[2:])
            hidden = patch_volume.flatten(2).transpose(1, 2)
            output_modulation = time_embedding(
                sinusoidal_embedding_1d(256, timestep.flatten()).to(dtype=hidden.dtype)
            ).reshape(timestep.shape[0], -1, 5120)
            block_modulation = time_projection(output_modulation).unflatten(2, (6, 5120))
            expanded_block_modulation = block_modulation.expand(-1, hidden.shape[1], -1, -1)
            text_context = text_embedding(context)
            frequencies = model.rope_encode(
                latent.shape[2],
                latent.shape[3],
                latent.shape[4],
                device=device,
                dtype=latent.dtype,
                source_id=0,
            )
            captured.update(
                {
                    "block_modulation": expanded_block_modulation.cpu(),
                    "output_modulation": output_modulation.expand(-1, hidden.shape[1], -1).cpu(),
                    "patch_tokens": hidden.cpu(),
                    "text_context": text_context.cpu(),
                }
            )

            for index in range(args.wan_blocks):
                block = WanAttentionBlock(
                    "t2v_cross_attn",
                    5120,
                    13824,
                    40,
                    qk_norm=True,
                    cross_attn_norm=True,
                    eps=1e-6,
                    operation_settings={
                        "operations": comfy.ops.disable_weight_init,
                        "device": "meta",
                        "dtype": torch.bfloat16,
                    },
                )
                _load_streamed_module(
                    block,
                    checkpoint,
                    f"blocks.{index}.",
                    official_names=False,
                    label=f"native Wan block {index}",
                )
                block = _move_streamed_module(block, device)
                hidden = block(
                    hidden,
                    e=block_modulation,
                    freqs=frequencies,
                    context=text_context,
                    context_img_len=None,
                    transformer_options={},
                )
                captured[f"block_{index:02d}"] = hidden.cpu()
                del block
                _release_streamed_memory()

            packed = output_head(hidden, output_modulation)
            captured["packed_prediction"] = packed.cpu()
            captured["prediction"] = _unpatch_wan(packed, grid_size).cpu()
    return captured


def _capture_wan(args: argparse.Namespace, device: torch.device) -> dict[str, torch.Tensor]:
    if any(
        value % patch
        for value, patch in zip(
            (args.latent_frames, args.latent_height, args.latent_width),
            (1, 2, 2),
            strict=True,
        )
    ):
        raise ValueError("Wan latent dimensions must be divisible by patch size (1, 2, 2)")
    if args.implementation == "official":
        return _capture_official_wan(args, device)
    return _capture_native_wan(args, device)


def _metadata(args: argparse.Namespace) -> dict[str, str]:
    metadata = {
        "implementation": args.implementation,
        "stage": args.stage,
        "seed": str(args.seed),
        "planner": str(args.planner.resolve()),
        "planner_sha256": PLANNER_SHA256,
        "official_commit": "e6c2cf1",
        "torch": torch.__version__,
        "device": str(args.device),
        "native_attention": args.native_attention,
        "native_rope_dtype": args.native_rope_dtype,
    }
    if args.stage == "wan_prediction":
        checkpoint = _wan_checkpoint(args)
        metadata.update(
            {
                "renderer": str(checkpoint.resolve()),
                "renderer_expert": args.renderer_expert,
                "renderer_size": str(checkpoint.stat().st_size),
                "wan_timestep": str(_wan_timestep(args)),
                "wan_blocks": str(args.wan_blocks),
            }
        )
        with safe_open(checkpoint, framework="pt", device="cpu") as handle:
            source_metadata = handle.metadata() or {}
        if "source_revision" in source_metadata:
            metadata["source_revision"] = source_metadata["source_revision"]
    return metadata


def capture(args: argparse.Namespace) -> None:
    if args.stage != "wan_prediction" and not args.planner.is_file():
        raise FileNotFoundError(args.planner)
    if args.implementation == "official" and not args.official_repo.is_dir():
        raise FileNotFoundError(args.official_repo)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    if args.stage == "planner_hidden":
        tensors = (
            _capture_official_hidden(args, device)
            if args.implementation == "official"
            else _capture_native_hidden(args, device)
        )
    elif args.stage == "vit_target":
        tensors = _capture_vit(args, device)
    elif args.stage == "wan_prediction":
        checkpoint = _wan_checkpoint(args)
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        if args.implementation == "official" and not args.official_model.is_dir():
            raise FileNotFoundError(args.official_model)
        tensors = _capture_wan(args, device)
    else:
        raise ValueError(f"unsupported stage: {args.stage}")

    from bernini_v2.parity import save_parity_artifact

    save_parity_artifact(args.output, tensors, metadata=_metadata(args))
    print(f"saved {args.implementation} {args.stage}: {args.output}")
    del tensors
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def compare(args: argparse.Namespace) -> None:
    from bernini_v2.parity import TensorTolerance, compare_artifacts, write_parity_report

    report = compare_artifacts(
        args.reference,
        args.candidate,
        tolerance=TensorTolerance(
            atol=args.atol,
            rtol=args.rtol,
            max_normalized_rmse=args.max_normalized_rmse,
            min_cosine_similarity=args.min_cosine_similarity,
        ),
    )
    write_parity_report(args.report, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["passed"]:
        raise SystemExit(1)


def run(args: argparse.Namespace) -> None:
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_stem = args.stage
    if args.stage == "wan_prediction":
        artifact_stem = f"{artifact_stem}_{args.renderer_expert}"
    official = output_dir / f"{artifact_stem}_official.safetensors"
    native = output_dir / f"{artifact_stem}_native.safetensors"
    report = output_dir / f"{artifact_stem}_report.json"
    common = [
        "--stage",
        args.stage,
        "--planner",
        str(args.planner),
        "--high-renderer",
        str(args.high_renderer),
        "--low-renderer",
        str(args.low_renderer),
        "--renderer-expert",
        args.renderer_expert,
        "--official-model",
        str(args.official_model),
        "--official-repo",
        str(args.official_repo),
        "--veomni-repo",
        str(args.veomni_repo),
        "--comfy-root",
        str(args.comfy_root),
        "--device",
        args.device,
        "--seed",
        str(args.seed),
        "--sequence-length",
        str(args.sequence_length),
        "--token-count",
        str(args.token_count),
        "--vit-steps",
        str(args.vit_steps),
        "--vit-text-cfg",
        str(args.vit_text_cfg),
        "--vit-image-cfg",
        str(args.vit_image_cfg),
        "--latent-frames",
        str(args.latent_frames),
        "--latent-height",
        str(args.latent_height),
        "--latent-width",
        str(args.latent_width),
        "--wan-context-length",
        str(args.wan_context_length),
        "--wan-blocks",
        str(args.wan_blocks),
        "--native-attention",
        args.native_attention,
        "--native-rope-dtype",
        args.native_rope_dtype,
    ]
    if args.wan_timestep is not None:
        common.extend(("--wan-timestep", str(args.wan_timestep)))
    script = str(Path(__file__).resolve())
    for implementation, destination in (("official", official), ("native", native)):
        subprocess.run(
            [
                sys.executable,
                script,
                "capture",
                "--implementation",
                implementation,
                "--output",
                str(destination),
                *common,
            ],
            cwd=ROOT,
            check=True,
        )
    subprocess.run(
        [
            sys.executable,
            script,
            "compare",
            "--reference",
            str(official),
            "--candidate",
            str(native),
            "--report",
            str(report),
            "--atol",
            str(args.atol),
            "--rtol",
            str(args.rtol),
            "--max-normalized-rmse",
            str(args.max_normalized_rmse),
            "--min-cosine-similarity",
            str(args.min_cosine_similarity),
        ],
        cwd=ROOT,
        check=True,
    )


def add_capture_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--stage", choices=STAGES, required=True)
    parser.add_argument("--planner", type=Path, default=DEFAULT_PLANNER)
    parser.add_argument("--high-renderer", type=Path, default=DEFAULT_HIGH_RENDERER)
    parser.add_argument("--low-renderer", type=Path, default=DEFAULT_LOW_RENDERER)
    parser.add_argument("--renderer-expert", choices=("high", "low"), default="high")
    parser.add_argument("--official-model", type=Path, default=DEFAULT_OFFICIAL_MODEL)
    parser.add_argument("--official-repo", type=Path, default=DEFAULT_OFFICIAL_REPO)
    parser.add_argument("--veomni-repo", type=Path, default=DEFAULT_VEOMNI_REPO)
    parser.add_argument("--comfy-root", type=Path, default=DEFAULT_COMFY_ROOT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sequence-length", type=int, default=16)
    parser.add_argument("--token-count", type=int, default=2)
    parser.add_argument("--vit-steps", type=int, default=1)
    parser.add_argument("--vit-text-cfg", type=float, default=1.2)
    parser.add_argument("--vit-image-cfg", type=float, default=1.0)
    parser.add_argument("--latent-frames", type=int, default=1)
    parser.add_argument("--latent-height", type=int, default=4)
    parser.add_argument("--latent-width", type=int, default=4)
    parser.add_argument("--wan-context-length", type=int, default=8)
    parser.add_argument("--wan-blocks", type=int, choices=range(1, 41), default=40)
    parser.add_argument("--wan-timestep", type=float)
    parser.add_argument(
        "--native-attention",
        choices=(
            "runtime",
            "pytorch",
            "optimized",
            "basic",
            "official_sdpa",
            "pytorch_repeat",
            "runtime_repeat",
        ),
        default="runtime",
    )
    parser.add_argument(
        "--native-rope-dtype",
        choices=("runtime", "input"),
        default="input",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action", required=True)
    capture_parser = subparsers.add_parser("capture", help="capture one implementation in this process")
    capture_parser.add_argument("--implementation", choices=("official", "native"), required=True)
    capture_parser.add_argument("--output", type=Path, required=True)
    add_capture_options(capture_parser)

    compare_parser = subparsers.add_parser("compare", help="compare two captured artifacts")
    compare_parser.add_argument("--reference", type=Path, required=True)
    compare_parser.add_argument("--candidate", type=Path, required=True)
    compare_parser.add_argument("--report", type=Path, required=True)
    compare_parser.add_argument("--atol", type=float, default=0.125)
    compare_parser.add_argument("--rtol", type=float, default=0.05)
    compare_parser.add_argument("--max-normalized-rmse", type=float, default=0.015)
    compare_parser.add_argument("--min-cosine-similarity", type=float, default=0.999)

    run_parser = subparsers.add_parser("run", help="capture official/native in child processes, then compare")
    run_parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    run_parser.add_argument("--atol", type=float, default=0.125)
    run_parser.add_argument("--rtol", type=float, default=0.05)
    run_parser.add_argument("--max-normalized-rmse", type=float, default=0.015)
    run_parser.add_argument("--min-cosine-similarity", type=float, default=0.999)
    add_capture_options(run_parser)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.action == "capture":
        capture(args)
    elif args.action == "compare":
        compare(args)
    else:
        run(args)


if __name__ == "__main__":
    main()
