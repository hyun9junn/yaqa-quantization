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
Prequantized models and Sketch-B Hessians (see paper) can be found [here](https://huggingface.co/collections/relaxml/yaqa-6837d4c8896eb9ceb7cb899e).

---

## Cross-block Hessian quantization

This section covers the two-step pipeline for **sparse parent-set LDLQ** quantization:
collect cross-block Hessians with `get_hess_llama.py`, then quantize with
`quantize_cross_hess_llama.py`.

### Background — weight labels and gidx

Each linear weight is identified by a **label** `{block}_{name}` and a global index **gidx**
assigned in forward order:

| position in block | name   | gidx offset |
|:-----------------:|--------|:-----------:|
| 0                 | `q`    | +0          |
| 1                 | `k`    | +1          |
| 2                 | `v`    | +2          |
| 3                 | `o`    | +3          |
| 4                 | `up`   | +4          |
| 5                 | `gate` | +5          |
| 6                 | `down` | +6          |

So transformer block 0 has gidx 0–6, block 1 has 7–13, and so on.

### Step 1 — Collect cross-block Hessians

Run `hessian_llama/get_hess_llama.py` with `torchrun`.
Pass `--cross` to enable cross-Hessian collection and use `--parent_band` / `--parent_extra_pairs`
to control **which pairs are collected**.

```bash
cd hessian_llama

# ── basic: collect diagonal Hessians only (no cross terms) ───────────────────
torchrun --nproc_per_node=8 get_hess_llama.py \
    --orig_model  meta-llama/Llama-3.2-1B-Instruct \
    --save_path   /path/to/hessians \
    --n_seqs      65536 \
    --ctx_size    2048 \
    --batch_size  2 \
    --hessian_sketch B

# ── band-1: collect cross-Hessians for immediately adjacent weight pairs ─────
# (default; captures q↔k, k↔v, ..., gate↔down, down↔next-block-q)
torchrun --nproc_per_node=8 get_hess_llama.py \
    --orig_model  meta-llama/Llama-3.2-1B-Instruct \
    --save_path   /path/to/hessians \
    --n_seqs      65536 \
    --batch_size  2 \
    --hessian_sketch B \
    --cross \
    --parent_band 1

# ── band-2: widen the band to capture two-hop neighbours ─────────────────────
torchrun --nproc_per_node=8 get_hess_llama.py \
    --orig_model  meta-llama/Llama-3.2-1B-Instruct \
    --save_path   /path/to/hessians \
    --n_seqs      65536 \
    --batch_size  2 \
    --hessian_sketch B \
    --cross \
    --parent_band 2

# ── block-adjacent: collect all pairs across adjacent transformer blocks ──────
# window=1 → all 7×7=49 weight pairs per block boundary (old default behaviour)
torchrun --nproc_per_node=1 get_hess_llama.py \
    --orig_model  meta-llama/Llama-3.2-1B-Instruct \
    --save_path   ./block_hess \
    --end_layer 4 \
    --n_seqs      1024 \
    --batch_size  4 \
    --hessian_sketch B \
    --cross \
    --parent_block_window 2

# ── band-1 + explicit extra pairs ────────────────────────────────────────────
# Adds q↔v and gate↔down within every block on top of the band.
# Use '*' as a block-index wildcard — expands across all layers automatically.
torchrun --nproc_per_node=8 get_hess_llama.py \
    --orig_model  meta-llama/Llama-3.2-1B-Instruct \
    --save_path   /path/to/hessians \
    --n_seqs      65536 \
    --batch_size  2 \
    --hessian_sketch B \
    --cross \
    --parent_band 1 \
    --parent_extra_pairs "*_q,*_v;*_gate,*_down"
```

**Key arguments**

| argument | default | description |
|---|---|---|
| `--cross` | off | enable cross-Hessian collection |
| `--parent_band W` | `1` | collect pairs with `\|gidx_i − gidx_j\| ≤ W` (gidx distance) |
| `--parent_block_window W` | off | collect pairs with `\|block_i − block_j\| ≤ W` (transformer block distance); overrides `--parent_band` when set |
| `--parent_extra_pairs STR` | `""` | always collect these named pairs on top of the band (see format above) |
| `--local_als_iters N` | `3` | ALS iterations for unequal-dimension pair approximation |
| `--start_layer` / `--end_layer` | `0` / `∞` | collect Hessians only for these transformer blocks |
| `--cpu_offload` | off | offload Hessian accumulators to CPU (saves GPU memory) |

Saved files follow the naming convention:
- `{label}_hin.pt` / `{label}_hout.pt` — diagonal (input/output) Hessian sketches
- `{label}_cross{partner_gidx}_hin.pt` / `…_hout.pt` — cross-block Kronecker factors

### Step 2 — Quantize with sparse parent-set LDLQ

Run `quantize_llama/quantize_cross_hess_llama.py` to quantize the model using the
collected Hessians. The **parent set P(k)** for each weight k determines which
previously-quantized weights contribute a cross-block correction:

```
W_eff[k] = W[k] + Σ_{j ∈ P(k)}  B_{k,j} · ΔW_j · A_{k,j}ᵀ
```

```bash
cd quantize_llama

# ── band-1 only (current YAQA default) ───────────────────────────────────────
python quantize_cross_hess_llama.py \
    --base_model meta-llama/Llama-3.2-1B-Instruct \
    --hess_path  /path/to/hessians \
    --save_path  /path/to/quantized \
    --codebook   E8P12 \
    --parent_band 1

# ── band-1 + top-3 strong pairs (by cross-Hessian magnitude) ─────────────────
# Scans up to 50 gidx steps back and picks the 3 strongest additional parents.
python quantize_cross_hess_llama.py \
    --base_model meta-llama/Llama-3.2-1B-Instruct \
    --hess_path  /path/to/hessians \
    --save_path  /path/to/quantized \
    --codebook   E8P12 \
    --parent_band 1 \
    --parent_topR 3 \
    --parent_lookback 50

# ── band-1 + threshold on strength ───────────────────────────────────────────
python quantize_cross_hess_llama.py \
    --base_model meta-llama/Llama-3.2-1B-Instruct \
    --hess_path  /path/to/hessians \
    --save_path  /path/to/quantized \
    --codebook   E8P12 \
    --parent_band 1 \
    --parent_threshold 0.05

# ── explicit parent sets ──────────────────────────────────────────────────────
# Use '*' as a block-index wildcard — expands across all layers automatically.
# "*_v:*_q" means: for every block i, v[i] gets cross-correction from q[i].
python quantize_cross_hess_llama.py \
    --base_model meta-llama/Llama-3.2-1B-Instruct \
    --hess_path  /path/to/hessians \
    --save_path  /path/to/quantized \
    --codebook   E8P12 \
    --parent_band 1 \
    --parent_explicit "*_v:*_q"

# ── ablation: disable all cross corrections ───────────────────────────────────
python quantize_cross_hess_llama.py \
    --base_model meta-llama/Llama-3.2-1B-Instruct \
    --hess_path  /workspace/yaqa-quantization/hessian_llama/no_cross \
    --save_path  ./no_cross \
    --codebook   E8P12 \
    --no_cross
```

**Parent set arguments**

| argument | default | description |
|---|---|---|
| `--parent_band W` | `1` | include all j with `gidx_k − gidx_j ≤ W` in P(k) |
| `--parent_topR R` | `0` | add top-R strongest parents beyond the band (0 = off) |
| `--parent_threshold τ` | `0.0` | add parents with `‖H_I‖·‖H_O‖ > τ` (0 = off) |
| `--parent_lookback M` | `50` | max gidx distance to search for topR / threshold |
| `--parent_explicit STR` | `""` | explicit per-weight parent sets (see format above) |
| `--no_cross` | off | disable all cross corrections (ablation) |

**Other key arguments**

| argument | default | description |
|---|---|---|
| `--codebook` | required | codebook name, e.g. `E8P12` |
| `--L` / `--K` / `--V` | 16/2/2 | trellis parameters |
| `--td_x` / `--td_y` | 16/16 | tile dimensions |
| `--sigma_reg` | `1e-2` | Hessian diagonal regularisation |
| `--scale_override` | `1.0` | weight scale multiplier |
| `--skip_list` | `""` | comma-separated labels to skip, e.g. `0_q,1_k` |
| `--device` | `cuda:0` | device to quantize on |

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
