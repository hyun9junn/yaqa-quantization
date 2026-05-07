# hessian_llama

Tools for collecting and analyzing per-layer (and cross-layer) Hessian approximations
of LLaMA-family models, used as input to the YAQA quantization pipeline.

---

## Overview

YAQA approximates the weight Hessian of each linear projection as a **Kronecker product**:

```
H_j  ≈  H_{I,j} ⊗ H_{O,j}
```

where

- **H_{I,j}** = E[X^T X] / m  — input-space factor, estimated from activations X
- **H_{O,j}** = E[G G^T] / n  — output-space factor, estimated from output gradients G

Both factors are estimated in a single backward pass via the weight gradient
**L_j = G_j^T X_j** (shape m × n):

```
H_{I,j} = E[L_j^T L_j] / m
H_{O,j} = E[L_j  L_j^T] / n
```

The `--cross` extension additionally estimates **off-diagonal cross-block terms**:

```
H_{jk}  ≈  (H_I)_{jk} ⊗ (H_O)_{jk}
   (H_I)_{jk} = E[L_j^T L_k] / m
   (H_O)_{jk} = E[L_j  L_k^T] / n
```

These cross terms are collected in the same backward pass at negligible extra cost.
The combined relative significance is:

```
ρ_{jk} = ‖H_{jk}‖_F / √(‖H_{jj}‖_F · ‖H_{kk}‖_F)
        = ρ^I_{jk} × ρ^O_{jk}        (since ‖A⊗B‖_F = ‖A‖_F · ‖B‖_F)
```

From the collected data (layers 0–1 of LLaMA-2-7B), representative findings:

| Pair | ρ^I | ρ^O | **ρ_combined** |
|---|---|---|---|
| gate ↔ up (same MLP, SwiGLU) | 0.49 | 2.69 | **1.33** |
| q ↔ k (same attention block) | 0.27 | 0.79 | **0.22** |
| o_layer0 ↔ o_layer1 | 3.21 | 0.18 | **0.58** |
| down_layer0 ↔ down_layer1 | 1.54 | 0.09 | **0.13** |

Values above 1.0 are valid — the SwiGLU gate↔up coupling exceeds the diagonal
because the output is `gate · silu(gate) · up`, making the mixed second derivative
structurally larger than the self-curvature.

---

## Files

| File | Purpose |
|---|---|
| `get_hess_llama.py` | Main collection script (torchrun) |
| `custom_linear_A.py` | Sketch A backend (power iteration, H_I only) |
| `custom_linear_B.py` | Sketch B backend (single pass, H_I + H_O + cross) |
| `data_utils.py` | C4 streaming dataset wrapper |
| `llama_hess.py` | Patched LLaMA model accepting the `mode` argument |
| `visualize_hessian.py` | Render block Hessian structure as a heatmap |
| `visualize_cross_coupling.py` | Compute and plot ρ^I, ρ^O, ρ_combined for all pairs |
| `assemble_total_hessian.py` | Assemble the saved files into a single block-banded matrix |

---

## Saved file format

All files are written to `--save_path`. Layer indices follow:

```
global_index = transformer_block * 7 + position_in_block
position:  q=0  k=1  v=2  o=3  up=4  gate=5  down=6
```

| File | Shape | Contents |
|---|---|---|
| `{lb}_{proj}_hin.pt` | `n*(n+1)/2` (flat) | Lower-triangle of H_I diagonal block |
| `{lb}_{proj}_hout.pt` | `m*(m+1)/2` (flat) | Lower-triangle of H_O diagonal block |
| `{lb}_{proj}_cross{gidx}_hin.pt` | `(n_other, n)` | H_I cross block with global layer `gidx` |
| `{lb}_{proj}_cross{gidx}_hout.pt` | `(m_other, m)` | H_O cross block with global layer `gidx` |

`{lb}` is the transformer block index, `{proj}` is one of `q k v o up gate down`.
Cross files are stored at the **lower** global index and hold `H[other, self]`;
the symmetric block is `H[self, other] = H[other, self].T`.

**Normalization note:** `hin` and `hout` accumulate `L^T L` and `L L^T` without
dividing by the output/input dimension. When computing relative strengths,
divide `hin_j` by `m_j` (output channels) and `hout_j` by `n_j` (input channels).
The cross files already include the `/m` or `/n` divisor.

---

## Step-by-step pipeline

### 1. Collect Hessians

The script uses FSDP with `CPUOffload` so model weights are kept on CPU.
GPU VRAM only holds one decoder layer at a time (~400 MB for 7B models).
The real constraint is **CPU RAM** for model weights + Hessian accumulators.

| Model | Weights | Accumulators | CPU RAM | Fits on 1× GPU? |
|---|---|---|---|---|
| LLaMA-2-7B | ~14 GB | ~22 GB | ~36 GB | ✅ |
| LLaMA-3-8B | ~16 GB | ~22 GB | ~38 GB | ✅ |
| LLaMA-2-13B | ~26 GB | ~35 GB | ~61 GB | ✅ (slow) |
| LLaMA-2-70B | ~140 GB | — | >125 GB | ❌ (split layers) |

#### Sketch A (power iteration — H_I only)

Use when GPU memory is the bottleneck and you can afford multiple backward passes.
Requires `--power_iters ≥ 2` (one half-round of power iteration is not enough).

```bash
torchrun --standalone --nproc-per-node=8 get_hess_llama.py \
    --save_path /path/to/hessians \
    --orig_model meta-llama/Llama-2-7b-hf \
    --batch_size 4 \
    --start_layer 0 \
    --end_layer 32 \
    --hessian_sketch A \
    --power_iters 6 \
    --ctx_size 8192 \
    --n_seqs 4096
```

#### Sketch B (recommended — H_I + H_O, single pass)

Produces both Kronecker factors in one backward pass.
Add `--cross` to also collect all pairwise cross-block terms.

```bash
# Single GPU, 7B model, all layers, with cross terms
torchrun --standalone --nproc-per-node=1 get_hess_llama.py \
    --save_path ./hess_als3_after \
    --orig_model meta-llama/Llama-2-7b-hf \
    --batch_size 16 \
    --start_layer 0 \
    --end_layer 2 \
    --hessian_sketch B \
    --power_iters 1 \
    --ctx_size 2048 \
    --n_seqs 2048 \
    --cross \
    --local_als_iters 3 \
    --cpu_offload
```

```bash
# Multi-GPU (8× 80 GB), 70B model, split across two nodes
# Node 1 — layers 0–39
torchrun --standalone --nproc-per-node=8 get_hess_llama.py \
    --save_path /shared/hessians \
    --orig_model meta-llama/Llama-2-70b-hf \
    --batch_size 2 \
    --start_layer 0 \
    --end_layer 40 \
    --hessian_sketch B \
    --power_iters 1 \
    --ctx_size 2048 \
    --n_seqs 65536 \
    --cross

# Node 2 — layers 40–79 (run in parallel)
torchrun --standalone --nproc-per-node=8 get_hess_llama.py \
    --save_path /shared/hessians \
    --orig_model meta-llama/Llama-2-70b-hf \
    --batch_size 2 \
    --start_layer 40 \
    --end_layer 80 \
    --hessian_sketch B \
    --power_iters 1 \
    --ctx_size 2048 \
    --n_seqs 65536 \
    --cross
```

**Key arguments:**

| Argument | Default | Description |
|---|---|---|
| `--save_path` | required | Directory for output `.pt` files |
| `--orig_model` | required | HuggingFace model ID or local path |
| `--batch_size` | 2 | Sequences per GPU per step |
| `--n_seqs` | 65536 | Total token sequences across all GPUs |
| `--ctx_size` | 2048 | Tokens per sequence |
| `--start_layer` | 0 | First transformer block (inclusive) |
| `--end_layer` | 100000 | Last transformer block (exclusive) |
| `--hessian_sketch` | B | `A` (power iter) or `B` (single pass) |
| `--power_iters` | 1 | Backward passes; Sketch A needs ≥ 2 |
| `--cross` | off | Collect pairwise cross-block terms |
| `--local_als_iters` | 3 | ALS steps for cross-dim pairs (0 = disabled) |
| `--cpu_offload` | off | Keep Hessian accumulators on CPU RAM |
| `--fp64_accum` | off | Accumulate in FP64 (slight quality improvement) |

**Cross-term ALS note:** when two layers have different output dimensions
(e.g. attention projections with m=4096 vs MLP up/gate with m=11008),
the ALS refines the cross-block Kronecker factors beyond the initial
truncated estimate. The implementation normalizes the L matrices to unit
Frobenius norm before running ALS and restores the scale afterward,
preventing the numerical explosion that occurred in earlier versions.
After running, cross-dim pairs are clearly distinguishable in the output
files by their rectangular shapes.

---

### 2. Visualize the block Hessian structure

`visualize_hessian.py` renders the full block matrix as a downsampled heatmap.

```bash
python visualize_hessian.py \
    --save_path /path/to/hessians \
    --hess_type in \
    --layers 0,1,2 \
    --names q,k,v,o,up,gate,down \
    --max_size 64 \
    --log_scale \
    --normalize_diag \
    --output hessian_block.png
```

| Flag | Effect |
|---|---|
| `--hess_type in\|out` | Input-space (`hin`) or output-space (`hout`) factor |
| `--log_scale` | Plot `sign(H)·log₁₀‖H‖`; useful when diagonal dominates |
| `--normalize_diag` | Scale each block by its diagonal peak so cross-term magnitudes are directly comparable |
| `--max_size N` | Pixel budget for the largest block; others scale proportionally |
| `--all_cross` | Show all pairwise cross blocks (default: adjacent pairs only) |
| `--layers` | Comma-separated transformer block indices |
| `--names` | Comma-separated projection names to include |

Output: `--output` (default `hessian_block.png`).

---

### 3. Analyze cross-coupling significance

`visualize_cross_coupling.py` computes the correctly-normalized relative strength
ρ^I, ρ^O, and ρ^I × ρ^O for every pair of collected cross-block terms, and saves
a three-panel heatmap.

```bash
python visualize_cross_coupling.py \
    --save_path ./hess_als3_after \
    --layers 0,1,2 \
    --names q,k,v,o,up,gate,down
```

Output: `{save_path}/cross_hessian_grid.png` and a printed table to stdout.

**Normalization:** the script divides each diagonal block by its output dimension m
(for ρ^I) or input dimension n (for ρ^O) to match the scale of the cross files,
giving a dimensionally consistent ratio. Pairs where ALS ran during collection
(m_i ≠ m_j) are printed as `[ALS — re-collect]` and omitted from the plot until
their data has been regenerated with the fixed ALS code.

| Argument | Default | Description |
|---|---|---|
| `--save_path` | required | Directory containing the `.pt` files |
| `--layers` | `0` | Comma-separated transformer block indices |
| `--names` | all 7 | Projection names to include |
| `--dtype` | `float32` | Precision for loading and computing |

---

### 4. Assemble the total block Hessian

`assemble_total_hessian.py` loads all saved files and builds the full
block-banded symmetric matrix H.

```bash
# Print block-level statistics without materializing the matrix (safe for large models)
python assemble_total_hessian.py \
    --save_path /path/to/hessians \
    --hess_type in \
    --layers 0,1,2 \
    --names q,k,v,o,up,gate,down \
    --stats_only

# Assemble and save (feasible for small layer counts or small models)
python assemble_total_hessian.py \
    --save_path /path/to/hessians \
    --hess_type in \
    --layers 0,1 \
    --names q,k,v,o,up,gate,down \
    --output total_hessian_layers01.pt
```

The saved `.pt` contains:

```python
{
    'H':       torch.Tensor,   # full block matrix
    'labels':  list[str],      # e.g. ['0_q', '0_k', ...]
    'dims':    list[int],      # n_j for each block
    'offsets': list[int],      # row/column start offsets
}
```

`--stats_only` also saves a `hessian_similarity_grid.png` heatmap of pairwise
diagonal-block cosine similarities.

| Argument | Default | Description |
|---|---|---|
| `--save_path` | required | Directory containing the `.pt` files |
| `--hess_type` | `in` | `in` for hin, `out` for hout |
| `--layers` | `0` | Comma-separated transformer block indices |
| `--output` | `total_hessian.pt` | Output file path |
| `--stats_only` | off | Print statistics only, skip assembly |
| `--dtype` | `float32` | `float32` or `float64` |

---

## Known issues and data quality notes

### ALS-corrupted cross-dim data

The `hess_3/` directory was collected with an earlier version of `custom_linear_B.py`
where the ALS for cross-dimension pairs (attention ↔ MLP, i.e. m_i ≠ m_j) diverged
numerically, producing cross-block norms ~10^12 times too large.
**The ALS code has been fixed.** Re-run `get_hess_llama.py` with `--cross` to obtain
valid cross-dim estimates. Same-dimension pairs (q↔k, q↔v, k↔v, up↔gate) in
`hess_3/` are unaffected and give correct values.

### hin/hout scale convention

`hin_j` stores `E[L_j^T L_j]` without dividing by the output dimension m_j,
while `cross_hin_{jk}` stores `E[L_j^T L_k] / m_i`. Always apply the `/m` or `/n`
correction when computing ratios or relative strengths (the visualization scripts
already do this).

### Sketch A vs Sketch B

Sketch A collects only H_I and requires multiple backward passes (≥ 2 power iterations).
Sketch B collects both H_I and H_O in a single pass and also supports `--cross`.
Use Sketch A only if you cannot afford the memory for H_O accumulators.
