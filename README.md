# [Model-Preserving Adaptive Rounding (Yet Another Quantization Algorithm)](https://arxiv.org/abs/2505.22988)

This repository contains code for Yet Another Quantization Algorithm (YAQA), a quantization framework that uses a Kronecker-factored approximation of the layerwise Hessian with respect to the full-model KL divergence to better preserve model outputs after quantization.
YAQA reduces the KL divergence to the original model by a factor of 1/3 over LDLQ/GPTQ across a wide range of models and quantizers, translating to state of the art performance on downstream tasks.
For more details, see the paper.

<img src="assets/comp.png" width="800">

## Installation

Requires Python ≥ 3.12, an NVIDIA driver new enough for the CUDA build you
install, and `nvcc` on `PATH`.

For Blackwell GPUs such as B200, use a CUDA 12.8+ stack. CUDA 12.6 can install
and run on older GPUs, but it is not the right target for B200 because Blackwell
native code generation starts with CUDA 12.8. The recommended path on new
machines is a CUDA 13.x driver/toolkit with PyTorch `cu130`; CUDA Toolkit 12.8
with PyTorch `cu128` is also acceptable if you need to stay on CUDA 12.

If `nvidia-smi` reports a driver like `580.95.05` and `CUDA Version: 13.1`, use
the `cu130` PyTorch wheel below. The `13.1` value is the driver's maximum CUDA
runtime capability; it is compatible with PyTorch's CUDA 13.0 wheels. `nvcc
--version` should report CUDA 12.8 or newer before building the local kernel.

### 1 — Create the venv

```bash
# Install uv if needed
curl -LsSf https://astral.sh/uv/install.sh | sh

uv venv --python 3.12
source .venv/bin/activate
```

### 2 — Pre-seed build tools and torch

`fast-hadamard-transform` imports `torch` during metadata generation (before `uv sync`
installs anything), so both must be present in `.venv` first.

`setuptools`/`wheel` must be installed separately from torch because `--index-url`
replaces PyPI entirely and those packages are not on the PyTorch wheel index.

```bash
# Step A: build tools from PyPI
uv pip install setuptools packaging wheel

# Step B: torch from the CUDA 13.0 index for Blackwell/B200
# Use --reinstall if a wrong-CUDA torch is already present
uv pip install "torch>=2.11.0" --index-url https://download.pytorch.org/whl/cu130

# Alternative for CUDA 12.8 systems:
# uv pip install "torch>=2.7.0,<2.12" --index-url https://download.pytorch.org/whl/cu128
```

### 3 — Install all remaining dependencies

```bash
CUDA_HOME=/usr/local/cuda uv sync --no-build-isolation
```

`uv sync` reads `uv.lock` for a fully reproducible install and pulls PyTorch wheels
from the configured CUDA index automatically via `[tool.uv.sources]`.

If your lockfile still points at `cu126`, regenerate it after changing the CUDA
index:

```bash
uv lock --upgrade-package torch --upgrade-package triton
```

### 4 — Build the local CUDA kernel

```bash
uv pip install -e ./qtip-kernels --no-build-isolation
```

The extension build now sets `TORCH_CUDA_ARCH_LIST` automatically. With
CUDA 12.8+ it includes `10.0+PTX` for B200; with older toolkits it omits
Blackwell because those `nvcc` versions cannot compile `sm_100`.

### 5 — Log in to Hugging Face

Llama and other gated models require authentication:

```bash
huggingface-cli login
# or: export HF_TOKEN=your_token_here
```

---

## How to use this codebase

This codebase is based off of the [QTIP](https://github.com/Cornell-RelaxML/qtip) codebase, with modifications made to support YAQA's quantization algorithm.
To collect Hessians, see the `README` in `hessian_llama/`.
To quantize models, follow the instructions in the [QTIP codebase](https://github.com/Cornell-RelaxML/qtip).
Prequantized models and Sketch-B Hessians (see paper) can be found [here](https://huggingface.co/collections/relaxml/yaqa-6837d4c8896eb9ceb7cb899e).

## Other

If you found this work useful, please consider citing
```
@misc{tseng2025modelpreservingadaptiverounding,
      title={Model-Preserving Adaptive Rounding}, 
      author={Albert Tseng and Zhaofeng Sun and Christopher De Sa},
      year={2025},
      eprint={2505.22988},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2505.22988}, 
}
```

Use of Llama models is governed by the Llama Community License. Use of this code is governed by the GNU GPL v3 license.
