# Bernini v2 for ComfyUI

[English](#english) | [中文](#中文)

这是 ByteDance Bernini v2 的原生 ComfyUI 实现。它不是把 Diffusers
管道塞进一个节点，而是使用 ComfyUI 自己的模型加载、显存管理、条件、采样器和
Wan 渲染器，因此可以正常卸载模型、使用低显存模式，并接入标准 ComfyUI 工作流。

支持 `t2i`、`i2i`、`t2v`、`v2v`、`r2v` 和 `rv2v` 六种任务。

> 推荐下载 Balanced INT8 权重：约 45.63 GiB。完整 BF16 版本约 83.04 GiB。
> 实验性 GGUF Q4_K_S 双 renderer 合计 16.31 GiB，画质略低且需要 ComfyUI-GGUF。

## 中文

### 这是什么

Bernini v2 会先用 Qwen2.5-VL 规划画面和动作，再交给 Wan2.2 双专家模型渲染。
它更适合复杂的视频生成、视频编辑和参考图引导任务，但模型也比普通 Wan 更大。

这个项目提供：

- 原生 ComfyUI 节点，不依赖 Diffusers 运行时。
- 六种官方任务和可直接打开的示例工作流。
- 自动低显存调度、顺序 CFG/APG 和双专家切换。
- Core 风格的单体 safetensors：Planner、UMT5、高噪声 Wan、低噪声 Wan
  各一个文件，不需要 Diffusers 目录、分片索引或 manifest。
- BF16 与原生 ComfyUI INT8 ConvRot 权重。
- 可识别 NVFP4、现代/旧版 scaled FP8；GGUF 通过外部 ComfyUI-GGUF
  接入两个 Wan renderer，Planner 仍使用原生权重。
- 提供低内存分片转 GGUF、原子量化/5D 回填和双专家合同校验工具。

### 安装节点

在 ComfyUI-Manager 中搜索
[`Bernini v2 (Native)`](https://registry.comfy.org/nodes/bernini-v2-t8)，或手动安装：

Registry 版本 `0.3.3` 已通过审核，当前状态为 `Active`。

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
  --include "text_encoders/*_int8.safetensors" "diffusion_models/*_int8.safetensors" "vae/wan_2.1_vae.safetensors" `
  --local-dir C:/path/to/ComfyUI/models
```

BF16 参考版本：

```powershell
hf download t8star/Bernini-V2-Comfy `
  --include "text_encoders/*_bf16.safetensors" "diffusion_models/*_bf16.safetensors" "vae/wan_2.1_vae.safetensors" `
  --local-dir C:/path/to/ComfyUI/models
```

实验性 Q4_K_S GGUF（先安装
[`ComfyUI-GGUF`](https://github.com/city96/ComfyUI-GGUF)）：

```powershell
hf download t8star/Bernini-V2-Comfy `
  --include "Bernini-v2-GGUF-Q4_K_S/*" `
  --local-dir C:/path/to/ComfyUI/models/diffusion_models
```

同一模型仓库还提供经过哈希核对的标准 Wan 2.1 VAE。直接下载到
ComfyUI 的模型根目录：

```powershell
hf download t8star/Bernini-V2-Comfy `
  --include "vae/wan_2.1_vae.safetensors" `
  --local-dir C:/path/to/ComfyUI/models
```

也可以使用仓库内固定版本的下载脚本：

```powershell
python tools/download_vae.py --output C:/path/to/ComfyUI/models/vae
```

最终目录应类似：

```text
ComfyUI/models/
├── text_encoders/
│   ├── bernini_v2_planner_int8.safetensors
│   └── umt5_xxl_bernini_v2_int8.safetensors
├── diffusion_models/
│   ├── bernini_v2_high_noise_int8.safetensors
│   └── bernini_v2_low_noise_int8.safetensors
└── vae/
    └── wan_2.1_vae.safetensors
```

打开 [`examples/workflows`](examples/workflows) 中对应任务的工作流即可开始。
视频示例默认使用 33 帧、16 fps 和长边 640，约 2.06 秒，方便 24 GB 显存机器测试。

### 实测情况

- 测试设备：RTX 5090 Laptop 24 GB、64 GB 内存。
- 当前 Core 候选实现按官方 T2V 预设完成 640×368、33 帧测试。
- 无其他 GPU 计算进程的冷启动复测按官方 50/1/50 步在 1,024.58 秒
  （约 17 分 05 秒）完成，峰值 ComfyUI 可见显存 16.59 GiB。
- 此前同一兼容栈的成功基线为 1,756.39 秒、23.26 GiB；PyTorch 2.7 +
  CUDA 12.8 eager/offload 路径会受宿主机内存和进程状态影响，四小时未完成
  不属于正常基线。
- 外部 NVFP4 双 renderer 已通过 640×368、33 帧、25/5/40 步画质门；
  CUDA 13 优化速度和独占显存对比仍待验证。
- Q4_K_S GGUF 双 renderer 已通过同规格画质门，33 帧全部唯一；双文件
  16.31 GiB，相比 Balanced INT8 renderer 对减少 38.9%，但有轻微脸部变软和
  腿/尾粘连，因此仍是实验档。
- T2V、V2V、R2V、RV2V 的两秒长边 640 测试均通过。
- 同一 ComfyUI 进程连续运行两个任务后，显存和内存可以回落。

详细数据见 [`docs/QUALITY_TESTS.md`](docs/QUALITY_TESTS.md)、
[`docs/SMOKE_TESTS.md`](docs/SMOKE_TESTS.md) 和
[`docs/LOW_MEMORY_WEIGHTS.md`](docs/LOW_MEMORY_WEIGHTS.md)。官方/原生逐张量
对齐的输入、阈值和真实权重结果见
[`docs/TENSOR_PARITY.md`](docs/TENSOR_PARITY.md)。

原生 Core 实现已提交到
[ComfyUI PR #16019](https://github.com/Comfy-Org/ComfyUI/pull/16019)。旧的分片方案
[#16001](https://github.com/Comfy-Org/ComfyUI/pull/16001) 已关闭。当前 PR 可合并、
全部 CI 通过、无未解决审查线程，正在等待维护者审查。

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
python tools/export_single_files.py `
  --source C:/path/to/ComfyUI/models/bernini_v2/Bernini-v2-balanced-int8 `
  --output C:/path/to/Bernini-v2-single-int8 `
  --profile int8
```

前两步产生仅供转换使用的分片工作区；最后一步才生成用户安装的四个单体文件。
`validate_single_files.py` 会检查单文件合同、键、dtype、量化标记、tokenizer 和双专家一致性。
GGUF 的分片转换、量化和验证命令见
[`docs/LOW_MEMORY_WEIGHTS.md`](docs/LOW_MEMORY_WEIGHTS.md)。

## English

### What this project does

Bernini v2 plans a scene and its motion with Qwen2.5-VL, then renders it with
two Wan2.2 experts. This repository implements that pipeline with native
ComfyUI model loading, memory management, conditioning and sampling APIs. It
does not run a hidden Diffusers pipeline inside a node.

It supports all six released tasks: `t2i`, `i2i`, `t2v`, `v2v`, `r2v`, and
`rv2v`.

### Install

Install [`Bernini v2 (Native)`](https://registry.comfy.org/nodes/bernini-v2-t8)
from ComfyUI-Manager, or clone this repository into `ComfyUI/custom_nodes`,
then restart ComfyUI.

Registry release `0.3.3` is reviewed and `Active`.

Download the recommended 45.63 GiB Balanced INT8 standalone files directly
into the standard ComfyUI model folders:

```powershell
hf download t8star/Bernini-V2-Comfy `
  --include "text_encoders/*_int8.safetensors" "diffusion_models/*_int8.safetensors" "vae/wan_2.1_vae.safetensors" `
  --local-dir C:/path/to/ComfyUI/models
```

The 83.04 GiB BF16 files use the `_bf16.safetensors` suffix in the same
`text_encoders/` and `diffusion_models/` folders. Each component is one file;
there is no Diffusers directory, shard index, or repack manifest. The same
repository includes the verified Wan 2.1 VAE companion file:

```powershell
hf download t8star/Bernini-V2-Comfy `
  --include "vae/wan_2.1_vae.safetensors" `
  --local-dir C:/path/to/ComfyUI/models
```

Alternatively run
`python tools/download_vae.py --output C:/path/to/ComfyUI/models/vae`.

An experimental 16.31 GiB Q4_K_S renderer pair is also published. Install
[`ComfyUI-GGUF`](https://github.com/city96/ComfyUI-GGUF), then download it to
the diffusion-model directory:

```powershell
hf download t8star/Bernini-V2-Comfy `
  --include "Bernini-v2-GGUF-Q4_K_S/*" `
  --local-dir C:/path/to/ComfyUI/models/diffusion_models
```

Open a workflow from [`examples/workflows`](examples/workflows). Video examples
use 33 frames at 16 fps with a 640-pixel long edge so the complete pipeline can
be tested on a 24 GB GPU.

On the RTX 5090 Laptop 24 GB test host, a clean reboot with no competing GPU
process completed the Core-only official 50/1/50 T2V preset at 640x368 and 33
frames in 1,024.58 seconds (about 17 minutes 5 seconds), with 16.59 GiB peak
ComfyUI-visible VRAM. An earlier successful run on the same PyTorch 2.7 + CUDA
12.8 eager/offload compatibility stack took 1,756.39 seconds and peaked at
23.26 GiB, so host memory/process state materially affects this legacy path.

### Development and Core work

The implementation is covered by unit, workflow, real-weight quality, memory,
and repeat-run tests. The native Core implementation is submitted as
[ComfyUI PR #16019](https://github.com/Comfy-Org/ComfyUI/pull/16019), closing
[issue #15702](https://github.com/Comfy-Org/ComfyUI/issues/15702). Review notes
live in
[`docs/CORE_MERGE_PLAN.md`](docs/CORE_MERGE_PLAN.md). The PR is currently
mergeable with all CI checks passing and no unresolved review threads; it is
awaiting maintainer review.

An external NVFP4 renderer pair also passes the production-step 640x368,
33-frame T2V visual gate. This was a CUDA 12.8 eager-path quality run; CUDA 13
optimized speed and isolated-memory results remain pending.

The Q4_K_S GGUF pair passes the same 640x368, 33-frame production-step gate.
All frames are unique and motion remains coherent, with mild facial softness
and occasional leg/tail merging versus Balanced INT8. It is an experimental
low-storage lane rather than the default quality recommendation.

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
- Hugging Face: <https://huggingface.co/t8star>

## License and credits

This project and the repackaged weights use the Apache License 2.0. Bernini v2
was created and released by ByteDance; see the
[official model](https://huggingface.co/ByteDance/Bernini-Diffusers-v2),
[source code](https://github.com/bytedance/Bernini), and
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).
