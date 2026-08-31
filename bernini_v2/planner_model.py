# Copyright (c) 2026 ByteDance Ltd. and/or its affiliate
# SPDX-License-Identifier: Apache-2.0
"""Native PyTorch modules for Bernini v2 semantic planning.

The parameter layout matches ByteDance/Bernini exactly. The implementation is
adapted from the official Apache-2.0 source and cross-checked against the
ComfyUI-oriented reference published in rzgar/Bernini-v2-ComfyUI.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


def _module_ops(operations):
    return operations if operations is not None else nn


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6, *, device=None, dtype=None):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(dim, device=device, dtype=dtype))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        normalized = x.float() * torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + self.eps)
        weight = self.weight.to(device=x.device, dtype=torch.float32)
        return (normalized * weight).to(dtype)


class MLPConnector(nn.Module):
    def __init__(
        self,
        in_dim: int = 3584,
        out_dim_for_gen: int = 4096,
        out_dim_for_vit: int = 3584,
        *,
        device=None,
        dtype=None,
        operations=None,
    ):
        super().__init__()
        ops = _module_ops(operations)
        self.proj_gen = nn.Sequential(
            ops.Linear(in_dim, out_dim_for_gen, device=device, dtype=dtype),
            nn.GELU(),
            RMSNorm(out_dim_for_gen, device=device, dtype=dtype),
            ops.Linear(out_dim_for_gen, out_dim_for_gen, device=device, dtype=dtype),
        )
        self.pred_vit = nn.Sequential(
            ops.Linear(in_dim, out_dim_for_vit, device=device, dtype=dtype),
            nn.GELU(),
            ops.Linear(out_dim_for_vit, out_dim_for_vit, device=device, dtype=dtype),
            RMSNorm(out_dim_for_vit, device=device, dtype=dtype),
            ops.Linear(out_dim_for_vit, out_dim_for_vit, device=device, dtype=dtype),
        )

    def for_gen(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj_gen(x)

    def for_vit(self, x: torch.Tensor) -> torch.Tensor:
        return self.pred_vit(x)


class TimestepEmbedder(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        frequency_embedding_size: int = 256,
        *,
        device=None,
        dtype=None,
        operations=None,
    ):
        super().__init__()
        ops = _module_ops(operations)
        self.mlp = nn.Sequential(
            ops.Linear(frequency_embedding_size, hidden_size, bias=True, device=device, dtype=dtype),
            nn.SiLU(),
            ops.Linear(hidden_size, hidden_size, bias=True, device=device, dtype=dtype),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
        half = dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(half, dtype=torch.float32, device=t.device) / half)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        frequency = self.timestep_embedding(t, self.frequency_embedding_size)
        return self.mlp(frequency.to(t.dtype))


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1 + scale) + shift


class ResBlock(nn.Module):
    def __init__(self, channels: int, *, device=None, dtype=None, operations=None):
        super().__init__()
        ops = _module_ops(operations)
        self.in_ln = ops.LayerNorm(channels, eps=1e-6, device=device, dtype=dtype)
        self.mlp = nn.Sequential(
            ops.Linear(channels, channels, bias=True, device=device, dtype=dtype),
            nn.SiLU(),
            ops.Linear(channels, channels, bias=True, device=device, dtype=dtype),
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            ops.Linear(channels, 3 * channels, bias=True, device=device, dtype=dtype),
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        shift, scale, gate = self.adaLN_modulation(y).chunk(3, dim=-1)
        hidden = self.mlp(modulate(self.in_ln(x), shift, scale))
        return x + gate * hidden


class FinalLayer(nn.Module):
    def __init__(self, model_channels: int, out_channels: int, *, device=None, dtype=None, operations=None):
        super().__init__()
        ops = _module_ops(operations)
        self.norm_final = ops.LayerNorm(
            model_channels,
            elementwise_affine=False,
            eps=1e-6,
            device=device,
            dtype=dtype,
        )
        self.linear = ops.Linear(model_channels, out_channels, bias=True, device=device, dtype=dtype)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            ops.Linear(model_channels, 2 * model_channels, bias=True, device=device, dtype=dtype),
        )

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN_modulation(condition).chunk(2, dim=-1)
        return self.linear(modulate(self.norm_final(x), shift, scale))


class SimpleMLPAdaLN(nn.Module):
    def __init__(
        self,
        in_channels: int,
        model_channels: int,
        out_channels: int,
        z_channels: int,
        num_res_blocks: int,
        *,
        device=None,
        dtype=None,
        operations=None,
    ):
        super().__init__()
        ops = _module_ops(operations)
        settings = {"device": device, "dtype": dtype, "operations": operations}
        self.in_channels = in_channels
        self.time_embed = TimestepEmbedder(model_channels, **settings)
        self.cond_embed = ops.Linear(z_channels, model_channels, device=device, dtype=dtype)
        self.input_proj = ops.Linear(in_channels, model_channels, device=device, dtype=dtype)
        self.res_blocks = nn.ModuleList([ResBlock(model_channels, **settings) for _ in range(num_res_blocks)])
        self.final_layer = FinalLayer(model_channels, out_channels, **settings)

    def forward(self, x: torch.Tensor, t: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        condition = self.time_embed(t) + self.cond_embed(c)
        for block in self.res_blocks:
            x = block(x, condition)
        return self.final_layer(x, condition)

    def forward_with_cfg(self, x: torch.Tensor, t: torch.Tensor, c: torch.Tensor, cfg_scale: float) -> torch.Tensor:
        half = x[: len(x) // 2]
        output = self.forward(torch.cat([half, half], dim=0), t, c)
        cond, uncond = torch.split(output[:, : self.in_channels], len(output) // 2, dim=0)
        guided = uncond + cfg_scale * (cond - uncond)
        return torch.cat([guided, guided], dim=0)

    def forward_with_txt_img_cfg(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        c: torch.Tensor,
        txt_cfg_scale: float,
        img_cfg_scale: float,
    ) -> torch.Tensor:
        part = x[: len(x) // 3]
        output = self.forward(torch.cat([part, part, part], dim=0), t, c)
        cond, uncond, imgcond = torch.split(output[:, : self.in_channels], len(output) // 3, dim=0)
        guided = uncond + img_cfg_scale * (imgcond - uncond) + txt_cfg_scale * (cond - imgcond)
        return torch.cat([guided, guided, guided], dim=0)


class FlowMatchScheduler:
    def __init__(
        self,
        num_inference_steps: int = 100,
        num_train_timesteps: int = 1000,
        shift: float = 3.0,
        sigma_max: float = 1.0,
        sigma_min: float = 0.003 / 1.002,
        inverse_timesteps: bool = False,
        extra_one_step: bool = False,
        reverse_sigmas: bool = False,
    ):
        self.num_train_timesteps = num_train_timesteps
        self.shift = shift
        self.sigma_max = sigma_max
        self.sigma_min = sigma_min
        self.inverse_timesteps = inverse_timesteps
        self.extra_one_step = extra_one_step
        self.reverse_sigmas = reverse_sigmas
        self.sigmas = torch.empty(0)
        self.timesteps = torch.empty(0)
        self.training = False
        self.set_timesteps(num_inference_steps, device="cpu")

    def set_timesteps(
        self,
        num_inference_steps: int = 100,
        denoising_strength: float = 1.0,
        shift: float | None = None,
        device=None,
        dtype: torch.dtype = torch.bfloat16,
        training: bool = False,
    ) -> None:
        if shift is not None:
            self.shift = shift
        device = device or "cpu"
        sigma_start = self.sigma_min + (self.sigma_max - self.sigma_min) * denoising_strength
        count = num_inference_steps + 1 if self.extra_one_step else num_inference_steps
        sigmas = torch.linspace(sigma_start, self.sigma_min, count, device=device, dtype=dtype)
        if self.extra_one_step:
            sigmas = sigmas[:-1]
        if self.inverse_timesteps:
            sigmas = torch.flip(sigmas, dims=[0])
        self.sigmas = self.shift * sigmas / (1 + (self.shift - 1) * sigmas)
        if self.reverse_sigmas:
            self.sigmas = 1 - self.sigmas
        self.timesteps = self.sigmas * self.num_train_timesteps
        self.training = training

    def get_noise_sigma(self, timestep: torch.Tensor | float) -> torch.Tensor:
        if not isinstance(timestep, torch.Tensor):
            timestep = torch.tensor(timestep, device=self.timesteps.device)
        else:
            timestep = timestep.to(self.timesteps.device)
        timestep_id = torch.argmin(
            (self.timesteps.unsqueeze(-1) - timestep.reshape(1, -1)).abs(),
            dim=0,
        )
        return self.sigmas[timestep_id].to(timestep.device)

    def step(
        self,
        model_output: torch.Tensor,
        timestep: torch.Tensor | float,
        sample: torch.Tensor,
        to_final: bool = False,
    ) -> torch.Tensor:
        if not isinstance(timestep, torch.Tensor):
            timestep = torch.tensor(timestep, device=self.timesteps.device)
        else:
            timestep = timestep.to(self.timesteps.device)
        timestep_id = torch.argmin((self.timesteps - timestep).abs())
        sigma = self.sigmas[timestep_id]
        if to_final or timestep_id + 1 >= len(self.timesteps):
            next_sigma = sample.new_tensor(1 if (self.inverse_timesteps or self.reverse_sigmas) else 0)
        else:
            next_sigma = self.sigmas[timestep_id + 1]
        return sample + model_output * (next_sigma - sigma)


class DiffLossFM(nn.Module):
    def __init__(
        self,
        target_channels: int = 3584,
        z_channels: int = 3584,
        depth: int = 16,
        width: int = 4096,
        shift: float = 2.0,
        extra_one_step: bool = True,
        *,
        device=None,
        dtype=None,
        operations=None,
    ):
        super().__init__()
        self.in_channels = target_channels
        self.net = SimpleMLPAdaLN(
            target_channels,
            width,
            target_channels,
            z_channels,
            depth,
            device=device,
            dtype=dtype,
            operations=operations,
        )
        self.scheduler = FlowMatchScheduler(shift=shift, extra_one_step=extra_one_step)

    @torch.no_grad()
    def sample(
        self,
        z: torch.Tensor,
        *,
        cfg: float,
        num_inference_steps: int,
        seed: int | None = None,
        generator: torch.Generator | None = None,
        img_cfg: float | None = None,
    ) -> torch.Tensor:
        device = z.device
        if generator is None:
            generator = torch.Generator(device="cpu").manual_seed(0 if seed is None else seed)
        branch_count = 3 if img_cfg is not None and cfg > 1.0 else 2 if cfg > 1.0 else 1
        noise = torch.randn(z.shape[0] // branch_count, self.in_channels, generator=generator)
        samples = torch.cat([noise] * branch_count, dim=0).to(device=device, dtype=z.dtype)

        if branch_count == 3:
            sample_fn = self.net.forward_with_txt_img_cfg
            kwargs = {"c": z, "txt_cfg_scale": cfg, "img_cfg_scale": img_cfg}
        elif branch_count == 2:
            sample_fn = self.net.forward_with_cfg
            kwargs = {"c": z, "cfg_scale": cfg}
        else:
            sample_fn = self.net.forward
            kwargs = {"c": z}

        self.scheduler.set_timesteps(num_inference_steps, device=device, dtype=z.dtype)
        for timestep in self.scheduler.timesteps:
            prediction = sample_fn(samples, timestep.unsqueeze(0).to(z.dtype), **kwargs)
            samples = self.scheduler.step(prediction, timestep, samples)
        return samples


def expected_connector_keys() -> set[str]:
    model = MLPConnector(device="meta")
    return set(model.state_dict())


def expected_vit_decoder_keys(depth: int = 16, width: int = 4096) -> set[str]:
    model = DiffLossFM(depth=depth, width=width, device="meta")
    return set(model.state_dict())
