# [Model-Preserving Adaptive Rounding (Yet Another Quantization Algorithm)](https://arxiv.org/abs/2505.22988)

This repository contains code for Yet Another Quantization Algorithm (YAQA), a quantization framework that uses a Kronecker-factored approximation of the layerwise Hessian with respect to the full-model KL divergence to better preserve model outputs after quantization.
YAQA reduces the KL divergence to the original model by a factor of 1/3 over LDLQ/GPTQ across a wide range of models and quantizers, translating to state of the art performance on downstream tasks.
For more details, see the paper.

<img src="assets/comp.png" width="800">

## Installation

Requires Python ≥ 3.12, a CUDA 12.6 compatible driver, and `nvcc` on `PATH`.

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

# Step B: torch from the CUDA 12.6 index
# Use --reinstall if a wrong-CUDA torch is already present
uv pip install "torch>=2.7.0" --index-url https://download.pytorch.org/whl/cu126
```

### 3 — Install all remaining dependencies

```bash
CUDA_HOME=/usr/local/cuda uv sync --no-build-isolation
```

`uv sync` reads `uv.lock` for a fully reproducible install and pulls PyTorch wheels
from the CUDA 12.6 index automatically via `[tool.uv.sources]`.

### 4 — Build the local CUDA kernel

```bash
uv pip install -e ./qtip-kernels --no-build-isolation
```

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
