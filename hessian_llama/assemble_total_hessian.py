#!/usr/bin/env python3
"""
Assemble the total block Hessian from saved per-layer files.

The result is a block-banded symmetric matrix (bandwidth=1):

    H[j,j]   = flat_to_sym(hin_j)      (n_j  × n_j)    from {name}_hin.pt
    H[j,j+1] = cross_hin_j.T           (n_j  × n_{j+1}) from {name}_cross{gidx}_hin.pt
    H[j+1,j] = cross_hin_j             (n_{j+1} × n_j)

Usage:
    python assemble_total_hessian.py \\
        --save_path /path/to/hessians \\
        --hess_type in \\
        --layers 0,1 \\
        --names q,k,v,o,up,gate,down \\
        --output total_hessian.pt

If the assembled matrix is too large to store, use --stats_only to print
block-level correlation statistics without materialising the full matrix.
"""

import argparse
import glob
import os

import torch
import numpy as np


# ── helpers ───────────────────────────────────────────────────────────────────

def flat_to_sym(v: torch.Tensor, n: int) -> torch.Tensor:
    A = torch.zeros(n, n, dtype=v.dtype)
    idx = torch.tril_indices(n, n)
    A[idx[0], idx[1]] = v
    A[idx[1], idx[0]] = v
    return A


def infer_n(flat_len: int) -> int:
    return int(round((-1 + (1 + 8 * flat_len) ** 0.5) / 2))


def relative_strength(cross: torch.Tensor,
                      diag_j: torch.Tensor,
                      diag_k: torch.Tensor) -> float:
    """Frobenius norm of cross block relative to geometric mean of diagonal blocks."""
    denom = (diag_j.norm() * diag_k.norm()) ** 0.5
    return (cross.norm() / denom).item() if denom > 0 else float('nan')


def cosine_similarity_blocks(A: torch.Tensor, B: torch.Tensor) -> float:
    """Cosine similarity between two matrices; truncates to min shape when sizes differ."""
    if A.shape != B.shape:
        m = min(A.shape[0], B.shape[0])
        n = min(A.shape[1], B.shape[1])
        A, B = A[:m, :n], B[:m, :n]
    return (A.flatten() @ B.flatten()
            / (A.norm() * B.norm() + 1e-30)).item()


# ── args ──────────────────────────────────────────────────────────────────────

LAYER_ORDER = ['q', 'k', 'v', 'o', 'up', 'gate', 'down']

parser = argparse.ArgumentParser()
parser.add_argument('--save_path', required=True)
parser.add_argument('--output', default='total_hessian.pt',
                    help='Where to save the assembled matrix (.pt)')
parser.add_argument('--hess_type', choices=['in', 'out'], default='in')
parser.add_argument('--layers', default='0')
parser.add_argument('--names', default=','.join(LAYER_ORDER))
parser.add_argument('--stats_only', action='store_true',
                    help='Print cross-block statistics without assembling the '
                         'full matrix (saves memory)')
parser.add_argument('--dtype', choices=['float32', 'float64'], default='float32')
args = parser.parse_args()

layer_nums = [int(x) for x in args.layers.split(',')]
names      = [x.strip() for x in args.names.split(',')]
dtype      = torch.float32 if args.dtype == 'float32' else torch.float64

suffix_diag  = 'hin.pt' if args.hess_type == 'in' else 'hout.pt'
suffix_cross = 'hin.pt' if args.hess_type == 'in' else 'hout.pt'

# ── discover layers ───────────────────────────────────────────────────────────

entries = []
for lb in sorted(layer_nums):
    for nm in LAYER_ORDER:
        if nm not in names:
            continue
        gidx  = lb * 7 + LAYER_ORDER.index(nm)
        fdiag = os.path.join(args.save_path, f'{lb}_{nm}_{suffix_diag}')
        if os.path.exists(fdiag):
            # cross files: {label}_cross{partner_gidx}_{suffix}
            cross_map = {}
            for fp in glob.glob(os.path.join(args.save_path,
                                             f'{lb}_{nm}_cross*_{suffix_cross}')):
                basename = os.path.basename(fp)
                mid = basename[len(f'{lb}_{nm}_cross'):]
                gidx_str = mid[:mid.index('_')]
                try:
                    cross_map[int(gidx_str)] = fp
                except ValueError:
                    pass
            entries.append({
                'label':     f'{lb}_{nm}',
                'gidx':      gidx,
                'fdiag':     fdiag,
                'cross_map': cross_map,
            })

if not entries:
    raise SystemExit(f'[error] No files found in {args.save_path!r}')

# ── load diagonal blocks ──────────────────────────────────────────────────────

for e in entries:
    v    = torch.load(e['fdiag'], map_location='cpu').to(dtype)
    n    = infer_n(len(v))
    e['n']    = n
    e['diag'] = flat_to_sym(v, n)

N      = len(entries)
dims   = [e['n'] for e in entries]
total  = sum(dims)
labels = [e['label'] for e in entries]

print(f'\nLayers ({N}): {labels}')
print(f'Dims:        {dims}')
print(f'Total dim:   {total}')

# Memory estimate
bytes_needed = total * total * (4 if dtype == torch.float32 else 8)
print(f'Full matrix: {bytes_needed / 1e9:.2f} GB  '
      f'({"float32" if dtype == torch.float32 else "float64"})\n')

# ── load cross blocks & compute stats ─────────────────────────────────────────

cross_blocks = {}   # key: (j, j+1) → tensor shape (n_{j+1}, n_j)

print(f'{"Pair":>20}  {"shape":>15}  {"rel_strength":>14}  {"cos_sim":>10}')
print('-' * 66)

for j in range(N - 1):
    ej = entries[j]
    ek = entries[j + 1]

    if ek['gidx'] != ej['gidx'] + 1:
        continue

    fcross = ej['cross_map'].get(ek['gidx'])
    if fcross is None:
        continue

    C = torch.load(fcross, map_location='cpu').to(dtype)   # (n_k, n_j)

    if C.shape != (ek['n'], ej['n']):
        print(f'  [warn] {ej["label"]}→{ek["label"]}: '
              f'shape {tuple(C.shape)} != ({ek["n"]},{ej["n"]}), skipping')
        continue

    rel = relative_strength(C, ej['diag'], ek['diag'])
    cos = cosine_similarity_blocks(C, ej['diag'][:ek['n'], :ej['n']])

    print(f'{ej["label"]:>12} ↔ {ek["label"]:<8}  '
          f'{str(tuple(C.shape)):>15}  '
          f'{rel:>14.4f}  '
          f'{cos:>10.4f}')

    cross_blocks[(j, j + 1)] = C   # (n_{j+1}, n_j)

print()

# ── pairwise diagonal-block similarity grid ───────────────────────────────────

print('Computing pairwise diagonal-block cosine similarities …')
sim_matrix = torch.zeros(N, N)
for i in range(N):
    for j in range(N):
        sim_matrix[i, j] = cosine_similarity_blocks(
            entries[i]['diag'], entries[j]['diag'])

# Print compact table
col_w = max(len(lbl) for lbl in labels) + 2
header = f'{"":>{col_w}}' + ''.join(f'{lbl:>{col_w}}' for lbl in labels)
print(header)
for i, lbl_i in enumerate(labels):
    row = f'{lbl_i:>{col_w}}' + ''.join(
        f'{sim_matrix[i, j].item():>{col_w}.3f}' for j in range(N))
    print(row)
print()

# Save heatmap
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig_size = max(8, N * 0.55 + 2)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size * 0.9))
    im = ax.imshow(sim_matrix.numpy(), vmin=-1, vmax=1,
                   cmap='RdBu_r', aspect='auto')
    plt.colorbar(im, ax=ax, label='Cosine similarity', fraction=0.046, pad=0.04)
    ax.set_xticks(range(N))
    ax.set_yticks(range(N))
    ax.set_xticklabels(labels, rotation=90, fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_title('Pairwise Hessian diagonal-block cosine similarity', pad=10)

    if N <= 35:
        for i in range(N):
            for j in range(N):
                val = sim_matrix[i, j].item()
                color = 'white' if abs(val) > 0.55 else 'black'
                ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                        fontsize=max(4, 7 - N // 8), color=color)

    plt.tight_layout()
    out_fig = os.path.join(args.save_path, 'hessian_similarity_grid.png')
    plt.savefig(out_fig, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved heatmap → {out_fig}')
except ImportError:
    print('[warn] matplotlib not available; skipping heatmap.')

print()

# ── assemble full block matrix ────────────────────────────────────────────────

if args.stats_only:
    print('[stats_only] Skipping full matrix assembly.')
else:
    print('Assembling full block matrix …')
    H = torch.zeros(total, total, dtype=dtype)

    offsets = [0]
    for d in dims:
        offsets.append(offsets[-1] + d)

    # Diagonal blocks
    for j, e in enumerate(entries):
        oj = offsets[j]
        H[oj:oj+e['n'], oj:oj+e['n']] = e['diag']

    # Off-diagonal (adjacent) cross blocks
    for (j, k), C in cross_blocks.items():
        oj, ok = offsets[j], offsets[k]
        nj, nk = dims[j], dims[k]
        H[oj:oj+nj, ok:ok+nk] = C.T   # H[j, j+1]
        H[ok:ok+nk, oj:oj+nj] = C     # H[j+1, j]

    # Symmetry check
    sym_err = (H - H.T).abs().max().item()
    print(f'Symmetry error (max |H - H^T|): {sym_err:.3e}')

    # Basic spectral info (only feasible for small matrices)
    if total <= 4096:
        eigvals = torch.linalg.eigvalsh(H)
        print(f'Eigenvalue range:  [{eigvals.min().item():.3e}, '
              f'{eigvals.max().item():.3e}]')
        neg = (eigvals < 0).sum().item()
        if neg:
            print(f'[warn] {neg} negative eigenvalues '
                  f'(cross terms may break PSD property)')

    torch.save({'H': H, 'labels': labels, 'dims': dims, 'offsets': offsets},
               args.output)
    print(f'Saved → {args.output}  (shape {tuple(H.shape)})')
