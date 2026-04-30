#!/usr/bin/env python3
"""
Visualize cross-layer Hessian coupling as an N×N grid.

For each adjacent layer pair (j, j+1) that has a saved cross Hessian,
two metrics are computed and displayed side-by-side:

  rel_strength : ||H[j,j+1]||_F / sqrt(||H[j,j]||_F * ||H[j+1,j+1]||_F)
                 → how large the coupling is relative to each layer's own curvature.
                 Large value = quantizing j strongly shifts the optimum of j+1.

  cos_sim      : cosine similarity between H[j,j+1] and a subblock of H[j,j]
                 → whether the cross-coupling has the same directional structure
                   as the diagonal (self) curvature.

Gray cells = no cross Hessian available (down→q cross-block boundary is always gray).
Dashed blue lines = transformer block boundaries.

Usage:
    python visualize_cross_coupling.py \\
        --save_path ./hess \\
        --hess_type in \\
        --layers 0,1,2 \\
        --names q,k,v,o,up,gate,down
"""

import argparse
import os

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


LAYER_ORDER = ['q', 'k', 'v', 'o', 'up', 'gate', 'down']


# ── helpers ───────────────────────────────────────────────────────────────────

def flat_to_sym(v: torch.Tensor, n: int) -> torch.Tensor:
    A = torch.zeros(n, n, dtype=v.dtype)
    idx = torch.tril_indices(n, n)
    A[idx[0], idx[1]] = v
    A[idx[1], idx[0]] = v
    return A


def infer_n(flat_len: int) -> int:
    return int(round((-1 + (1 + 8 * flat_len) ** 0.5) / 2))


def relative_strength(C: torch.Tensor,
                      diag_j: torch.Tensor,
                      diag_k: torch.Tensor) -> float:
    denom = (diag_j.norm() * diag_k.norm()) ** 0.5
    return (C.norm() / denom).item() if denom > 0 else float('nan')


def cosine_sim(A: torch.Tensor, B: torch.Tensor) -> float:
    if A.shape != B.shape:
        m = min(A.shape[0], B.shape[0])
        n = min(A.shape[1], B.shape[1])
        A, B = A[:m, :n], B[:m, :n]
    return (A.flatten() @ B.flatten()
            / (A.norm() * B.norm() + 1e-30)).item()


# ── args ──────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument('--save_path', required=True)
parser.add_argument('--hess_type', choices=['in', 'out'], default='in')
parser.add_argument('--layers', default='0')
parser.add_argument('--names', default=','.join(LAYER_ORDER))
parser.add_argument('--dtype', choices=['float32', 'float64'], default='float32')
args = parser.parse_args()

layer_nums = [int(x) for x in args.layers.split(',')]
names      = [x.strip() for x in args.names.split(',')]
dtype      = torch.float32 if args.dtype == 'float32' else torch.float64

suffix_diag  = 'hin.pt'       if args.hess_type == 'in' else 'hout.pt'
suffix_cross = 'cross_hin.pt' if args.hess_type == 'in' else 'cross_hout.pt'

# ── discover & load diagonal blocks ───────────────────────────────────────────

entries = []
for lb in sorted(layer_nums):
    for nm in LAYER_ORDER:
        if nm not in names:
            continue
        gidx   = lb * 7 + LAYER_ORDER.index(nm)
        fdiag  = os.path.join(args.save_path, f'{lb}_{nm}_{suffix_diag}')
        if os.path.exists(fdiag):
            entries.append({
                'label': f'{lb}_{nm}',
                'gidx':  gidx,
                'fdiag': fdiag,
            })

if not entries:
    raise SystemExit(f'[error] No diagonal files found in {args.save_path!r}')

for e in entries:
    v      = torch.load(e['fdiag'], map_location='cpu').to(dtype)
    n      = infer_n(len(v))
    e['n']    = n
    e['diag'] = flat_to_sym(v, n)

N      = len(entries)
labels = [e['label'] for e in entries]
names_used = [nm for nm in LAYER_ORDER if nm in names]
block_size = len(names_used)

print(f'\nLayers ({N}): {labels}')
print(f'Block size: {block_size} names per transformer layer\n')

# ── build N×N coupling matrices ───────────────────────────────────────────────
# NaN = no cross Hessian for that pair (non-adjacent or skipped boundary)

rel_mat = torch.full((N, N), float('nan'))
cos_mat = torch.full((N, N), float('nan'))

for i in range(N):
    rel_mat[i, i] = 1.0
    cos_mat[i, i] = 1.0

# Build gidx → index map for fast lookup
gidx_to_idx = {e['gidx']: i for i, e in enumerate(entries)}

print(f'{"Pair":>20}  {"shape":>15}  {"rel_strength":>14}  {"cos_sim":>10}')
print('-' * 66)

for j, ej in enumerate(entries):
    # Scan all saved cross files for this layer: {label}_cross{other_gidx}_{suffix}
    import glob as _glob
    pattern = os.path.join(args.save_path,
                           f'{ej["label"]}_cross*_{suffix_diag}')
    for fpath in sorted(_glob.glob(pattern)):
        # Extract other_gidx from filename: {label}_cross{gidx}_{suffix}
        basename = os.path.basename(fpath)
        # strip label prefix and suffix
        mid = basename[len(ej["label"]) + len("_cross"):]
        other_gidx_str = mid[:mid.index('_')]
        try:
            other_gidx = int(other_gidx_str)
        except ValueError:
            continue

        if other_gidx not in gidx_to_idx:
            continue
        k = gidx_to_idx[other_gidx]
        ek = entries[k]

        C = torch.load(fpath, map_location='cpu').to(dtype)

        # C shape is (n_k, n_j) — layer j stores H[j, k] where k > j
        expected = (ek['n'], ej['n'])
        if C.shape != expected:
            print(f'  [warn] {ej["label"]}→{ek["label"]}: '
                  f'shape {tuple(C.shape)} != {expected}, skipping')
            continue

        rel = relative_strength(C, ej['diag'], ek['diag'])
        cos = cosine_sim(C, ej['diag'][:ek['n'], :ej['n']])

        rel_mat[j, k] = rel_mat[k, j] = rel
        cos_mat[j, k] = cos_mat[k, j] = cos

        print(f'{ej["label"]:>12} ↔ {ek["label"]:<8}  '
              f'{str(tuple(C.shape)):>15}  {rel:>14.4f}  {cos:>10.4f}')

print()

# ── plot ──────────────────────────────────────────────────────────────────────

fig_w = max(16, N * 1.0 + 5)
fig_h = max(7,  N * 0.5 + 2)
fig, axes = plt.subplots(1, 2, figsize=(fig_w, fig_h))

rel_data = rel_mat.numpy()
cos_data = cos_mat.numpy()

panels = [
    (axes[0], rel_data,
     f'Relative strength  ||H[j,k]|| / √(||H[j,j]||·||H[k,k]||)\n'
     f'Large → quantizing j strongly shifts optimum of k',
     'YlOrRd', 'log'),
    (axes[1], cos_data,
     f'Cosine similarity  between H[j,k] and diagonal subblock\n'
     f'High → cross coupling is structurally aligned with self-curvature',
     'RdBu_r', 'linear'),
]

for ax, data, title, cmap_name, scale in panels:
    mask     = np.isnan(data)
    off_diag = ~mask & ~np.eye(N, dtype=bool)
    valid    = data[off_diag]

    cmap_obj = plt.cm.get_cmap(cmap_name).copy()
    cmap_obj.set_bad(color='#cccccc')
    masked = np.ma.array(data, mask=mask)

    if scale == 'log' and valid.size > 0:
        vmin = max(valid.min(), 1e-3)
        vmax = valid.max()
        norm = mcolors.LogNorm(vmin=vmin, vmax=vmax)
    else:
        vmin, vmax = -1.0, 1.0
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    im = ax.imshow(masked, norm=norm, cmap=cmap_obj, aspect='auto')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(range(N))
    ax.set_yticks(range(N))
    ax.set_xticklabels(labels, rotation=90, fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_title(title, fontsize=9, pad=8)

    # Per-cell value annotations
    if N <= 40:
        fs = max(4, 7 - N // 8)
        for i in range(N):
            for j in range(N):
                if mask[i, j]:
                    ax.text(j, i, '—', ha='center', va='center',
                            fontsize=fs, color='#999999')
                else:
                    val = data[i, j]
                    txt = f'{val:.2f}' if scale == 'linear' else f'{val:.0f}'
                    if scale == 'log' and vmax > vmin:
                        normed = (np.log10(max(val, 1e-10)) - np.log10(vmin)) / \
                                 (np.log10(vmax) - np.log10(vmin))
                    elif vmax > vmin:
                        normed = (val - vmin) / (vmax - vmin)
                    else:
                        normed = 0.5
                    color = 'white' if normed > 0.60 else 'black'
                    ax.text(j, i, txt, ha='center', va='center',
                            fontsize=fs, color=color)

    # Transformer block boundary lines
    for b in range(block_size, N, block_size):
        ax.axhline(b - 0.5, color='steelblue', linewidth=1.5,
                   linestyle='--', alpha=0.6)
        ax.axvline(b - 0.5, color='steelblue', linewidth=1.5,
                   linestyle='--', alpha=0.6)

plt.suptitle(
    f'Cross-Hessian quantization coupling  [{args.hess_type.upper()} side]\n'
    f'Gray = no cross data (down→q block boundary is always gray)',
    fontsize=11, y=1.02)
plt.tight_layout()

out_fig = os.path.join(args.save_path, 'cross_hessian_grid.png')
plt.savefig(out_fig, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f'Saved → {out_fig}')
