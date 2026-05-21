# Cross-Hessian Sparse Parent Experiments

This note describes the experiment flow for testing sparse cross-Hessian
corrections on top of the original YAQA quantization baseline.

The goal is:

1. collect diagonal YAQA Hessians and optional cross-Hessian factors,
2. measure which cross pairs are consistently strong,
3. select a sparse parent set `P(k)`,
4. quantize with only those selected corrections,
5. compare against no-cross and adjacent-band baselines.

## Idea

For each quantized weight block `k`, use a sparse update

```text
theta_k_eff = theta_k + sum_{j in P(k)} L_{k,j} Delta theta_j
hat_theta_k = Q(theta_k_eff)
```

where `j < k` follows the quantization order and
`Delta theta_j = theta_j - hat_theta_j`.

The current implementation uses the first-order Kronecker approximation

```text
L_{k,j} ~= H_{k,j} H_{j,j}^{-1}
H_{k,j} ~= H_I(k,j) kron H_O(k,j)
```

so the correction applied to weight `k` from parent `j` is

```text
B_{k,j} @ Delta W_j @ A_{k,j}.T
```

with

```text
A_{k,j} = H_I(k,j) H_I(j,j)^-1
B_{k,j} = H_O(k,j) H_O(j,j)^-1
```

This is not yet the full incomplete sparse LDL recurrence with diagonal-block
updates. It is the first sparse parent-set version, suitable for ablation.

## Files

- `hessian_llama/get_hess_llama.py`
  Collects original YAQA diagonal Hessians and, with `--cross`, cross-Hessian
  factors.

- `hessian_llama/select_cross_hessian_pairs.py`
  Scans collected cross-Hessian factors, ranks pair strengths, and writes a
  JSON parent map for quantization.

- `quantize_llama/quantize_cross_hess_llama.py`
  Quantizes with no-cross, adjacent band, automatic strength-based parents,
  explicit parents, or a selected parent-map JSON.

## Step 0: Original YAQA No-Cross Baseline

Collect original-YAQA diagonal Hessians:

```bash
cd /workspace/yaqa-quantization

torchrun --standalone --nproc-per-node=1 hessian_llama/get_hess_llama.py \
  --save_path hessian_llama/after_no_cross \
  --orig_model meta-llama/Llama-3.2-1B-Instruct \
  --batch_size 2 \
  --start_layer 0 \
  --hessian_sketch B \
  --power_iters 1 \
  --ctx_size 2048 \
  --n_seqs 65536 \
  --cpu_offload
```

Quantize with no cross correction:

```bash
python quantize_llama/quantize_cross_hess_llama.py \
  --base_model meta-llama/Llama-3.2-1B-Instruct \
  --hess_path hessian_llama/after_no_cross \
  --save_path quantize_llama/after_no_cross \
  --codebook E8P12 \
  --tlut_bits 16 \
  --no_cross
```

This is the baseline that should match the original YAQA diagonal-Hessian
behavior.

## Step 1: Collect Cross-Hessian Candidates

To discover strong pairs, collect a wider set of cross terms than the final
experiment may use.

Same global-index adjacent band:

```bash
torchrun --standalone --nproc-per-node=1 hessian_llama/get_hess_llama.py \
  --save_path hessian_llama/cross_band1 \
  --orig_model meta-llama/Llama-3.2-1B-Instruct \
  --batch_size 2 \
  --start_layer 0 \
  --hessian_sketch B \
  --power_iters 1 \
  --ctx_size 2048 \
  --n_seqs 65536 \
  --cpu_offload \
  --cross \
  --cross_estimator per_sample \
  --parent_band 1
```

Denser within-block and adjacent-block candidate collection:

```bash
torchrun --standalone --nproc-per-node=1 hessian_llama/get_hess_llama.py \
  --save_path hessian_llama/cross_block_window1 \
  --orig_model meta-llama/Llama-3.2-1B-Instruct \
  --batch_size 2 \
  --start_layer 0 \
  --hessian_sketch B \
  --power_iters 1 \
  --ctx_size 2048 \
  --n_seqs 65536 \
  --cpu_offload \
  --cross \
  --cross_estimator per_sample \
  --parent_block_window 1
```

If you already suspect specific structural pairs, add them during collection:

```bash
--parent_extra_pairs "*_q,*_v;*_up,*_gate;*_gate,*_down"
```

Important: quantization can only use `j < k` parents under the current
one-pass order. For example, with order `q,k,v,o,up,gate,down`, `v <- q` is
possible, but `q <- v` is not unless the quantization order changes.

## Step 2: Rank And Select Strong Pairs

Rank every collected cross edge:

```bash
python hessian_llama/select_cross_hessian_pairs.py \
  --hess_path hessian_llama/cross_block_window1 \
  --print_top 50
```

The selector writes:

- `selected_parent_edges.json`
- `cross_pair_strength.csv`
- `cross_pair_type_strength.csv`

The strength metric is:

```text
rho = rho_I * rho_O
```

This is the relative Frobenius strength of the Kronecker cross block, normalized
by the corresponding diagonal Hessian blocks.

### Selection Modes

Select all edges above a threshold:

```bash
python hessian_llama/select_cross_hessian_pairs.py \
  --hess_path hessian_llama/cross_block_window1 \
  --threshold 0.05 \
  --output_path hessian_llama/cross_block_window1/parents_thresh005.json
```

Select globally top-K edges:

```bash
python hessian_llama/select_cross_hessian_pairs.py \
  --hess_path hessian_llama/cross_block_window1 \
  --top_edges 64 \
  --output_path hessian_llama/cross_block_window1/parents_top64.json
```

Select top-K parents per child:

```bash
python hessian_llama/select_cross_hessian_pairs.py \
  --hess_path hessian_llama/cross_block_window1 \
  --top_per_child 1 \
  --output_path hessian_llama/cross_block_window1/parents_top1_per_child.json
```

Select strong structural pair types, such as consistently high `q->v` or
`up->gate`:

```bash
python hessian_llama/select_cross_hessian_pairs.py \
  --hess_path hessian_llama/cross_block_window1 \
  --same_block_only \
  --top_pair_types 4 \
  --min_layer_support 8 \
  --output_path hessian_llama/cross_block_window1/parents_top4_types.json
```

`--same_block_only` is useful for identifying repeated structural patterns
inside each transformer block.

## Step 3: Quantize Ablations

### A. No Cross

```bash
python quantize_llama/quantize_cross_hess_llama.py \
  --base_model meta-llama/Llama-3.2-1B-Instruct \
  --hess_path hessian_llama/cross_block_window1 \
  --save_path quantize_llama/no_cross_from_cross_hess \
  --codebook E8P12 \
  --tlut_bits 16 \
  --no_cross
```

### B. Adjacent Band `w=1`

```bash
python quantize_llama/quantize_cross_hess_llama.py \
  --base_model meta-llama/Llama-3.2-1B-Instruct \
  --hess_path hessian_llama/cross_band1 \
  --save_path quantize_llama/cross_band1 \
  --codebook E8P12 \
  --tlut_bits 16 \
  --parent_band 1
```

### C. Selected Pairs Only

```bash
python quantize_llama/quantize_cross_hess_llama.py \
  --base_model meta-llama/Llama-3.2-1B-Instruct \
  --hess_path hessian_llama/cross_block_window1 \
  --save_path quantize_llama/selected_pairs_only \
  --codebook E8P12 \
  --tlut_bits 16 \
  --parent_band 0 \
  --parent_map hessian_llama/cross_block_window1/parents_top4_types.json
```

### D. Adjacent Band Plus Selected Strong Pairs

```bash
python quantize_llama/quantize_cross_hess_llama.py \
  --base_model meta-llama/Llama-3.2-1B-Instruct \
  --hess_path hessian_llama/cross_block_window1 \
  --save_path quantize_llama/band1_plus_selected \
  --codebook E8P12 \
  --tlut_bits 16 \
  --parent_band 1 \
  --parent_map hessian_llama/cross_block_window1/parents_top4_types.json
```

### E. Manual Structural Pair

```bash
python quantize_llama/quantize_cross_hess_llama.py \
  --base_model meta-llama/Llama-3.2-1B-Instruct \
  --hess_path hessian_llama/cross_block_window1 \
  --save_path quantize_llama/manual_q_to_v \
  --codebook E8P12 \
  --tlut_bits 16 \
  --parent_band 0 \
  --parent_explicit "*_v:*_q"
```

## Suggested Experiment Table

| Experiment | Hessian dir | Quant args | Purpose |
|---|---|---|---|
| YAQA no-cross | `after_no_cross` | `--no_cross` | original baseline |
| no-cross from cross dir | `cross_block_window1` | `--no_cross` | isolate collection differences |
| band-1 | `cross_band1` | `--parent_band 1` | local adjacent cross baseline |
| selected only | `cross_block_window1` | `--parent_band 0 --parent_map ...` | test sparse strong edges |
| band-1 + selected | `cross_block_window1` | `--parent_band 1 --parent_map ...` | local + structural edges |
| manual q->v | `cross_block_window1` | `--parent_band 0 --parent_explicit "*_v:*_q"` | targeted hypothesis |

## Notes

- `--parent_map` and `--parent_explicit` are additive.
- `--parent_band 0 --parent_map X` means selected pairs only.
- `--parent_band 1 --parent_map X` means adjacent parents plus selected pairs.
- `--parent_topR` and `--parent_threshold` inside quantization still work, but
  using `select_cross_hessian_pairs.py` is more reproducible because the selected
  parent map is saved.
- If a selected pair was not collected, quantization logs that the cross-Hessian
  file is missing and skips that correction.
- Existing output files are skipped by the quantizer. Use a fresh `--save_path`
  for each ablation.

## Recommended First Pass

1. Produce `after_no_cross`.
2. Produce `cross_block_window1`.
3. Run:

```bash
python hessian_llama/select_cross_hessian_pairs.py \
  --hess_path hessian_llama/cross_block_window1 \
  --same_block_only \
  --top_pair_types 4 \
  --min_layer_support 8 \
  --output_path hessian_llama/cross_block_window1/parents_top4_types.json
```

4. Quantize:

```bash
# no-cross
--no_cross

# selected only
--parent_band 0 --parent_map hessian_llama/cross_block_window1/parents_top4_types.json

# band-1 + selected
--parent_band 1 --parent_map hessian_llama/cross_block_window1/parents_top4_types.json
```

5. Evaluate all outputs with the same downstream eval command.
