To collect Hessians for Llama 1, 2, and 3 models, run the `get_hess_llama.py` script with `torchrun`. 
Parameters with `(WHATEVER FITS)` will need to be tuned by you depending on the size of your machine. 
As a guide, we were able to fit models under ~20B parameters on a single 8x80G node and 70B across 2 8x80G nodes (layers 0 through 39 on one, 40 through 79 on another).
This script processes layers independently so if you are on a cluster with a shared filesystem you can launch jobs in parallel across subsets of layers.
We do not recommend using `cpu_offload` unless it is faster to move things to CPU on your machine than recomputing gradients.
Accumulating in FP64 is not necessary but may give slight improvements in quantization performance. 
The actual Hessian collection computation still happens in FP32 *per sample*, but if for some reason your model requires FP64 you may also want to change [the computation](https://github.com/Cornell-RelaxML/yaqa/blob/01763b16556031981b0d73ce2b802b56bfa1efea/hessian_llama/custom_linear_B.py#L61) to FP64 as well.
We recommend using Sketch B if you can afford it.

## Sketch A
```
torchrun --standalone --nproc-per-node=8 get_hess_llama.py \
    --save_path PATH \
    --orig_model HF_MODEL \
    --batch_size (WHATEVER FITS) \
    --start_layer (WHATEVER FITS) \
    --end_layer (WHATEVER FITS) \
    --hessian_sketch A \
    --power_iters 6 \
    --ctx_size 8192 \
    --n_seqs 4096 \
    (OPTIONAL)
    --fp64_accum (ACCUMULATE IN FP64) \
    --cpu_offload (CPU OFFLOAD, USUALLY SLOWER THAN SPLITTING BY start_layer/end_layer)
```

## Sketch B (Recommended)

```
torchrun --standalone --nproc-per-node=8 get_hess_llama.py \
    --save_path PATH \
    --orig_model HF_MODEL \
    --batch_size (WHATEVER FITS) \
    --start_layer (WHATEVER FITS) \
    --end_layer (WHATEVER FITS) \
    --hessian_sketch B \
    --power_iters 1 \
    --ctx_size 2048 \
    --n_seqs 65536 \
    (OPTIONAL)
    --fp64_accum (ACCUMULATE IN FP64) \
    --cpu_offload (CPU OFFLOAD, USUALLY SLOWER THAN SPLITTING BY start_layer/end_layer)
```

---

## Cross-Layer Hessian

The `--cross` flag additionally computes **off-diagonal cross terms** between adjacent linear layers,
enabling analysis of how correlated the Hessian curvature is across layer boundaries.

### Output files

Running with `--cross` produces the following files under `--save_path` for each linear projection:

| File | Contents | Shape |
|------|----------|-------|
| `{block}_{proj}_hin.pt` | Input-space diagonal Hessian (flat lower-triangular) | `n*(n+1)/2` |
| `{block}_{proj}_hout.pt` | Output-space diagonal Hessian (flat lower-triangular) | `m*(m+1)/2` |
| `{block}_{proj}_cross_hin.pt` | Cross term between this layer and the next (by global index) | `(n_next, n)` |
| `{block}_{proj}_cross_hout.pt` | Output-space cross term | `(m_next, m)` |

where `{block}` is the transformer layer index (0, 1, 2, …) and `{proj}` is one of
`q`, `k`, `v`, `o`, `up`, `gate`, `down`.

The cross file stored at layer `j` holds `H[j+1, j]`, the off-diagonal block between
layer `j+1` (rows) and layer `j` (columns). The full block matrix is symmetric:
`H[j, j+1] = H[j+1, j].T`.

The global layer order within each transformer block is: `q → k → v → o → up → gate → down`.
Cross terms are computed between every pair of **globally adjacent** layers, so
`0_down_cross_hin.pt` holds the cross term between layer 0's `down_proj` and layer 1's `q_proj`.

### Step 1 — Collect Hessians with cross terms (Sketch B)

```bash
torchrun --standalone --nproc-per-node=8 get_hess_llama.py \
    --save_path PATH \
    --orig_model HF_MODEL \
    --batch_size (WHATEVER FITS) \
    --start_layer (WHATEVER FITS) \
    --end_layer (WHATEVER FITS) \
    --hessian_sketch B \
    --power_iters 1 \
    --ctx_size 2048 \
    --n_seqs 65536 \
    --cross \
    --align_mode rect
```

`--align_mode` controls how cross-term dimensions are aligned when adjacent layers have
different sizes (`rect` uses leading-identity truncation; `ortho` uses random orthonormal projections).

#### Single-GPU guide (e.g. 1× A100-40 GB, 125 GB CPU RAM)

The script always keeps model weights on CPU via FSDP `CPUOffload`, so GPU VRAM is
only used for one decoder layer at a time (~400 MB for 7B).
The real constraint is **CPU RAM**, which must hold the model weights *and* all
Hessian accumulators simultaneously when `--cpu_offload` is set.

| Model | Weights | Hessian accumulators | CPU RAM total | Fits? |
|-------|---------|----------------------|---------------|-------|
| LLaMA-2-7B  | ~14 GB | ~22 GB | ~36 GB | ✅ |
| LLaMA-3-8B  | ~16 GB | ~22 GB | ~38 GB | ✅ |
| LLaMA-2-13B | ~26 GB | ~35 GB | ~61 GB | ✅ |
| LLaMA-2-70B | ~140 GB | — | >125 GB | ❌ |

> **`--cpu_offload` is required on a single GPU.**  Without it, Hessian accumulators
> for all linear layers (~22 GB for 7B) are allocated on the GPU card instead of CPU RAM.

**LLaMA-2-7B / LLaMA-3-8B** (recommended):

```bash
cd hessian_llama
torchrun --standalone --nproc-per-node=1 get_hess_llama.py \
    --save_path PATH \
    --orig_model meta-llama/Llama-2-7b-hf \
    --batch_size 2 \
    --start_layer 0 \
    --end_layer 32 \
    --hessian_sketch B \
    --power_iters 1 \
    --ctx_size 2048 \
    --n_seqs 65536 \
    --cross \
    --align_mode rect \
    --cpu_offload
```

**LLaMA-2-13B** (feasible, slower):

```bash
cd hessian_llama
torchrun --standalone --nproc-per-node=1 get_hess_llama.py \
    --save_path PATH \
    --orig_model meta-llama/Llama-2-13b-hf \
    --batch_size 1 \
    --start_layer 0 \
    --end_layer 40 \
    --hessian_sketch B \
    --power_iters 1 \
    --ctx_size 2048 \
    --n_seqs 65536 \
    --cross \
    --align_mode rect \
    --cpu_offload
```

### Step 2 — Assemble the total block Hessian

`assemble_total_hessian.py` loads the saved files and assembles the block-banded matrix:

```bash
# Print cross-block statistics only (safe for large models)
python assemble_total_hessian.py \
    --save_path PATH \
    --hess_type in \
    --layers 0,1,2 \
    --names q,k,v,o,up,gate,down \
    --stats_only

# Save the full assembled matrix to disk
python assemble_total_hessian.py \
    --save_path PATH \
    --hess_type in \
    --layers 0 \
    --output total_H_layer0.pt
```

The saved `.pt` file contains a dict `{'H': tensor, 'labels': list, 'dims': list, 'offsets': list}`.

### Step 3 — Visualize

`visualize_hessian.py` renders the block Hessian as a heatmap:

```bash
python visualize_hessian.py \
    --save_path PATH \
    --hess_type in \
    --layers 0,1 \
    --names q,k,v,o,up,gate,down \
    --max_size 64 \
    --log_scale \
    --normalize_diag \
    --output hessian_block.png
```

Key flags:

| Flag | Effect |
|------|--------|
| `--hess_type in/out` | Input-space (`hin`) or output-space (`hout`) Hessian |
| `--log_scale` | Plot `sign(H)·log₁₀\|H\|` — useful when diagonal blocks dominate |
| `--normalize_diag` | Scale each block by its diagonal peak so cross-term strength is directly comparable |
| `--max_size N` | Pixels for the largest block; smaller blocks scale proportionally |
