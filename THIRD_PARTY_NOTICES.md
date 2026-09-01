# Third-party notices

This project implements and interoperates with the following Apache-2.0
projects and model releases:

- ByteDance Bernini and `ByteDance/Bernini-Diffusers-v2`. The released source
  is the numerical reference for planner, scheduler, guidance, and checkpoint
  behavior.
- ComfyUI. Runtime modules use ComfyUI model loading, patching, conditioning,
  sampling, Wan, Qwen2.5-VL, T5, and VAE APIs.
- Wan-AI Wan 2.1 and Comfy-Org's Apache-2.0 ComfyUI repack. The model hub
  mirrors the unchanged `wan_2.1_vae.safetensors` companion file; its published
  SHA-256 is `2fc39d31359a4b0a64f55876d8ff7fa8d780956ae2cb13463b0223e15148976b`.
- `rzgar/Bernini-v2-ComfyUI`. Its Apache-2.0 custom-node implementation was
  reviewed as a low-level interoperability reference. This project does not
  adopt its monolithic pipeline structure or bundled model files.
- Comfy-Org `comfy-kitchen` and `comfy-quants`. The converter emits their
  stock-Comfy quantization contract and calls the installed comfy-kitchen
  public layout API; no kernel or wheel is vendored.
- `city96/ComfyUI-GGUF`. GGUF interoperability is an optional standard-`MODEL`
  boundary only; its loader and conversion code are not bundled or required.

Each upstream project retains its own copyright and license notices. The node
source repository does not vendor model weights; required weights are hosted in
the separately licensed Hugging Face model repository.
