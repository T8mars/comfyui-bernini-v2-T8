# Bernini v2 for ComfyUI

[English](#english) | [中文](#中文)

这是 ByteDance Bernini v2 的原生 ComfyUI 实现。它不是把 Diffusers
管道塞进一个节点，而是使用 ComfyUI 自己的模型加载、显存管理、条件、采样器和
Wan 渲染器，因此可以正常卸载模型、使用低显存模式，并接入标准 ComfyUI 工作流。

支持 `t2i`、`i2i`、`t2v`、`v2v`、`r2v` 和 `rv2v` 六种任务。

> 推荐下载 Balanced INT8 权重：约 45.62 GiB。完整 BF16 版本约 83.03 GiB。

## 中文

### 这是什么

Bernini v2 会先用 Qwen2.5-VL 规划画面和动作，再交给 Wan2.2 双专家模型渲染。
它更适合复杂的视频生成、视频编辑和参考图引导任务，但模型也比普通 Wan 更大。

这个项目提供：

- 原生 ComfyUI 节点，不依赖 Diffusers 运行时。
- 六种官方任务和可直接打开的示例工作流。
- 自动低显存调度、顺序 CFG/APG 和双专家切换。
- BF16 与原生 ComfyUI INT8 ConvRot 权重。
- NVFP4、现代/旧版 scaled FP8 检测，以及标准 `MODEL` 接口的 GGUF 兼容。

### 安装节点

在 ComfyUI-Manager 中搜索 `Bernini v2 (Native)`，或手动安装：

```powershell
cd C:/path/to/ComfyUI/custom_nodes
git clone https://github.com/T8mars/comfyui-bernini-v2-T8.git
```

重启 ComfyUI。

### 下载模型

模型仓库：<https://huggingface.co/t8star/Bernini-V2-Comfy>

推荐的 Balanced INT8：

```powershell
hf download t8star/Bernini-V2-Comfy `
  --include "Bernini-v2-balanced-int8/*" `
  --local-dir C:/path/to/ComfyUI/models/bernini_v2
```

BF16 参考版本：

```powershell
hf download t8star/Bernini-V2-Comfy `
  --include "Bernini-v2-bf16-native/*" `
  --local-dir C:/path/to/ComfyUI/models/bernini_v2
```

Bernini v2 使用标准 Wan 2.1 VAE：

```powershell
python tools/download_vae.py --output C:/path/to/ComfyUI/models/vae
```

最终目录应类似：

```text
ComfyUI/models/
├── bernini_v2/
│   └── Bernini-v2-balanced-int8/
│       ├── repack-manifest.json
│       ├── mllm/
│       ├── vit_decoder/
│       ├── wan_high/
│       └── wan_low/
└── vae/
    └── wan_2.1_vae.safetensors
```

打开 [`examples/workflows`](examples/workflows) 中对应任务的工作流即可开始。
视频示例默认使用 33 帧、16 fps 和长边 640，约 2.06 秒，方便 24 GB 显存机器测试。

### 实测情况

- 测试设备：RTX 5090 Laptop 24 GB、64 GB 内存。
- Balanced INT8 完整 T2V：640×368、33 帧、40 个渲染步。
- 峰值 ComfyUI 可见显存约 16.51 GiB。
- T2V、V2V、R2V、RV2V 的两秒长边 640 测试均通过。
- 同一 ComfyUI 进程连续运行两个任务后，显存和内存可以回落。

详细数据见 [`docs/QUALITY_TESTS.md`](docs/QUALITY_TESTS.md)、
[`docs/SMOKE_TESTS.md`](docs/SMOKE_TESTS.md) 和
[`docs/LOW_MEMORY_WEIGHTS.md`](docs/LOW_MEMORY_WEIGHTS.md)。

### 自己转换权重

如果不想下载重打包权重，可以从官方
[`ByteDance/Bernini-Diffusers-v2`](https://huggingface.co/ByteDance/Bernini-Diffusers-v2)
自行转换：

```powershell
python tools/download_model.py --output models/ByteDance/Bernini-Diffusers-v2
python tools/repack_diffusers.py `
  --source models/ByteDance/Bernini-Diffusers-v2 `
  --output C:/path/to/ComfyUI/models/bernini_v2/Bernini-v2-bf16-native `
  --storage-dtype bfloat16
python tools/quantize_repack.py `
  --source C:/path/to/ComfyUI/models/bernini_v2/Bernini-v2-bf16-native `
  --output C:/path/to/ComfyUI/models/bernini_v2/Bernini-v2-balanced-int8 `
  --profile balanced --device cuda
```

转换支持断点续跑；`validate_repack.py` 会检查索引、键、dtype、量化标记和 SHA-256。

## English

### What this project does

Bernini v2 plans a scene and its motion with Qwen2.5-VL, then renders it with
two Wan2.2 experts. This repository implements that pipeline with native
ComfyUI model loading, memory management, conditioning and sampling APIs. It
does not run a hidden Diffusers pipeline inside a node.

It supports all six released tasks: `t2i`, `i2i`, `t2v`, `v2v`, `r2v`, and
`rv2v`.

### Install

Install `Bernini v2 (Native)` from ComfyUI-Manager, or clone this repository
into `ComfyUI/custom_nodes`, then restart ComfyUI.

Download the recommended 45.62 GiB Balanced INT8 package:

```powershell
hf download t8star/Bernini-V2-Comfy `
  --include "Bernini-v2-balanced-int8/*" `
  --local-dir C:/path/to/ComfyUI/models/bernini_v2
```

The 83.03 GiB BF16 package is available from the same repository under
`Bernini-v2-bf16-native/`. Download the standard Wan 2.1 VAE with
`python tools/download_vae.py --output C:/path/to/ComfyUI/models/vae`.

Open a workflow from [`examples/workflows`](examples/workflows). Video examples
use 33 frames at 16 fps with a 640-pixel long edge so the complete pipeline can
be tested on a 24 GB GPU.

### Development and Core work

The implementation is covered by unit, workflow, real-weight quality, memory,
and repeat-run tests. The upstream Core request is tracked in
[ComfyUI issue #15702](https://github.com/Comfy-Org/ComfyUI/issues/15702).
Core readiness notes live in
[`docs/CORE_MERGE_PLAN.md`](docs/CORE_MERGE_PLAN.md).

```powershell
python -m pytest
python -m ruff check .
python -m ruff format --check .
```

## Links

- Bilibili: <https://space.bilibili.com/385085361>
- YouTube: <https://www.youtube.com/@T8star-Aix/>
- API: <https://api.seedance.nz/sign-up?aff=5f4w>
- Online AI apps: <https://www.runninghub.ai/zh-cn/user-center/1907375370302308353/userPost?inviteCode=rh-v1121>
- ComfyUI package: <https://pan.quark.cn/s/264edb7e36bd>
- Hugging Face: <https://huggingface.co/t8star>

## License and credits

This project and the repackaged weights use the Apache License 2.0. Bernini v2
was created and released by ByteDance; see the
[official model](https://huggingface.co/ByteDance/Bernini-Diffusers-v2),
[source code](https://github.com/bytedance/Bernini), and
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).
