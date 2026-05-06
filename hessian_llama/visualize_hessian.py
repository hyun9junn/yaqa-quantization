#!/usr/bin/env python3
"""
Visualize the block Hessian assembled from per-layer Hessian files produced by
get_hess_llama.py.

Diagonal blocks come from {layer}_{name}_hin.pt / hout.pt (flat packed symmetric
matrices).  Off-diagonal (cross) blocks come from {layer}_{name}_cross_hin.pt /
cross_hout.pt (full rectangular matrices stored at the lower-gidx layer j, with
shape (n_{j+1}, n_j)).

Usage example:
    python visualize_hessian.py \\
        --save_path /path/to/hessians \\
        --hess_type in \\
        --layers 0,1 \\
        --names q,k,v,o,up,gate,down \\
        --max_size 64 \\
        --log_scale \\
        --output hessian_block.png
"""

import argparse
import glob
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch


# ── helpers ───────────────────────────────────────────────────────────────────

def flat_to_sym(v: torch.Tensor, n: int) -> np.ndarray:
    """Unpack flat lower-triangular storage → full symmetric (n×n) numpy array."""
    A = torch.zeros(n, n, dtype=v.dtype)
    idx = torch.tril_indices(n, n)
    A[idx[0], idx[1]] = v
    A[idx[1], idx[0]] = v
    return A.numpy()


def downsample(M: np.ndarray, th: int, tw: int) -> np.ndarray:
    """Average-pool 2-D array M (H×W) to (th×tw)."""
    rows = np.array_split(M, th, axis=0)
    M2 = np.stack([r.mean(0) for r in rows])        # (th, W)
    cols = np.array_split(M2, tw, axis=1)
    return np.stack([c.mean(1) for c in cols], 1)   # (th, tw)


def infer_n(flat_len: int) -> int:
    """Solve n*(n+1)/2 = L for n."""
    return int(round((-1 + (1 + 8 * flat_len) ** 0.5) / 2))


# ── argument parsing ──────────────────────────────────────────────────────────

LAYER_ORDER = ['q', 'k', 'v', 'o', 'up', 'gate', 'down']

parser = argparse.ArgumentParser(description='Visualize cross-layer block Hessian')
parser.add_argument('--save_path', required=True,
                    help='Directory containing saved Hessian .pt files')
parser.add_argument('--output', default='hessian_block.png',
                    help='Output image file (PNG/PDF)')
parser.add_argument('--hess_type', choices=['in', 'out'], default='in',
                    help='in → H_in (input-space), out → H_out (output-space)')
parser.add_argument('--layers', default='0',
                    help='Comma-separated transformer-block indices, e.g. 0,1,2')
parser.add_argument('--names', default=','.join(LAYER_ORDER),
                    help='Comma-separated projection names to include')
parser.add_argument('--max_size', default=64, type=int,
                    help='Max pixels per diagonal block (proportional downsampling)')
parser.add_argument('--log_scale', action='store_true',
                    help='Plot sign(H)*log10(|H|+eps) instead of raw values')
parser.add_argument('--normalize_diag', action='store_true',
                    help='Scale each block so its diagonal block peak = 1 '
                         '(makes cross-term strength directly comparable)')
parser.add_argument('--all_cross', action='store_true',
                    help='Show all pairwise cross-block terms, not just adjacent (w=1). '
                         'Requires cross data to have been collected for non-adjacent pairs.')
args = parser.parse_args()

layer_nums = [int(x) for x in args.layers.split(',')]
names      = [x.strip() for x in args.names.split(',')]

suffix_diag  = 'hin.pt'       if args.hess_type == 'in' else 'hout.pt'
suffix_cross = 'hin.pt'       if args.hess_type == 'in' else 'hout.pt'

# ── discover files ────────────────────────────────────────────────────────────

entries = []
for lb in sorted(layer_nums):
    for nm in LAYER_ORDER:
        if nm not in names:
            continue
        gidx  = lb * 7 + LAYER_ORDER.index(nm)
        fdiag = os.path.join(args.save_path, f'{lb}_{nm}_{suffix_diag}')
        if os.path.exists(fdiag):
            # cross files are named {label}_cross{partner_gidx}_{suffix}
            cross_map = {}  # partner_gidx → filepath
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
    raise SystemExit(
        f'[error] No Hessian files found in {args.save_path!r} '
        f'for layers={args.layers}, names={args.names}')

print(f'Layers found ({len(entries)}): {[e["label"] for e in entries]}')

# ── load diagonal blocks ──────────────────────────────────────────────────────

for e in entries:
    v    = torch.load(e['fdiag'], map_location='cpu').float()
    n    = infer_n(len(v))
    e['n']    = n
    e['diag'] = flat_to_sym(v, n)
    print(f'  {e["label"]:12s}  dim={n:5d}  diag_range=[{e["diag"].min():.3e}, {e["diag"].max():.3e}]')

N     = len(entries)
max_n = max(e['n'] for e in entries)

# Proportional pixel sizes so aspect ratios are roughly preserved
sizes = [max(1, int(round(args.max_size * e['n'] / max_n))) for e in entries]

offsets = [0]
for s in sizes:
    offsets.append(offsets[-1] + s)
total = offsets[-1]

mosaic = np.zeros((total, total), dtype=np.float32)

# ── fill diagonal blocks ──────────────────────────────────────────────────────

diag_peaks = []
for j, e in enumerate(entries):
    s  = sizes[j]
    D  = downsample(e['diag'], s, s)
    pk = float(np.abs(D).max())
    diag_peaks.append(pk if pk > 0 else 1.0)
    if args.normalize_diag:
        D = D / diag_peaks[-1]
    mosaic[offsets[j]:offsets[j]+s, offsets[j]:offsets[j]+s] = D

# ── fill cross blocks ─────────────────────────────────────────────────────────

# Build candidate pairs: all (j<k) when --all_cross, adjacent-only otherwise
if args.all_cross:
    pairs = [(j, k) for j in range(N) for k in range(j + 1, N)]
else:
    # Only pairs whose gidx differ by exactly 1 (bandwidth w=1)
    pairs = [(j, j + 1) for j in range(N - 1)
             if entries[j + 1]['gidx'] == entries[j]['gidx'] + 1]

cross_found = 0
for j, k in pairs:
    ej = entries[j]
    ek = entries[k]

    # cross data is stored at the lower-gidx layer keyed by partner gidx
    fcross = ej['cross_map'].get(ek['gidx'])
    if fcross is None:
        continue

    C = torch.load(fcross, map_location='cpu').float().numpy()
    # Expected shape: (n_k, n_j)
    if C.shape != (ek['n'], ej['n']):
        print(f'  [warn] cross shape {C.shape} != expected ({ek["n"]},{ej["n"]}) '
              f'for {ej["label"]}→{ek["label"]}, skipping')
        continue

    sj, sk = sizes[j], sizes[k]
    oj, ok = offsets[j], offsets[k]

    # H[j, k] = C.T  shape (n_j, n_k) → downsample to (sj, sk)
    Ct_ds = downsample(C.T, sj, sk)
    # H[k, j] = C    shape (n_k, n_j) → downsample to (sk, sj)
    C_ds  = downsample(C,   sk, sj)

    if args.normalize_diag:
        scale = (diag_peaks[j] * diag_peaks[k]) ** 0.5
        Ct_ds = Ct_ds / scale
        C_ds  = C_ds  / scale

    mosaic[oj:oj+sj, ok:ok+sk] = Ct_ds   # upper-right block H[j, k]
    mosaic[ok:ok+sk, oj:oj+sj] = C_ds    # lower-left  block H[k, j]
    cross_found += 1
    gap = ek['gidx'] - ej['gidx']
    print(f'  cross {ej["label"]}↔{ek["label"]:12s}  gap={gap:2d}  '
          f'raw_range=[{C.min():.3e}, {C.max():.3e}]')

if cross_found == 0:
    print('[warn] No cross-term files were found/matched. '
          'Only diagonal blocks will be shown.')

# ── optional log transform ────────────────────────────────────────────────────

if args.log_scale:
    nonzero_abs = np.abs(mosaic[mosaic != 0.0])
    eps = float(nonzero_abs.min()) * 1e-3 if nonzero_abs.size > 0 else 1e-10
    sign   = np.sign(mosaic)
    mosaic = sign * np.log10(np.abs(mosaic) + eps)

# ── plot ──────────────────────────────────────────────────────────────────────

vmax = float(np.percentile(np.abs(mosaic), 99))
vmin = -vmax

labels   = [e['label'] for e in entries]
tick_pos = [offsets[j] + sizes[j] // 2 for j in range(N)]
fsize    = max(4, min(9, 90 // N))          # shrink labels for many layers

fig_px = max(8, min(24, N * 0.7))
fig, ax = plt.subplots(figsize=(fig_px, fig_px))

im = ax.imshow(mosaic, cmap='RdBu_r', vmin=vmin, vmax=vmax,
               interpolation='nearest', aspect='auto')
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
             label='sign·log₁₀|H|' if args.log_scale else 'H value')

ax.set_xticks(tick_pos)
ax.set_yticks(tick_pos)
ax.set_xticklabels(labels, rotation=90, fontsize=fsize)
ax.set_yticklabels(labels, fontsize=fsize)

# Block-boundary grid lines
for off in offsets[1:-1]:
    ax.axhline(off - 0.5, color='k', lw=0.4, alpha=0.5)
    ax.axvline(off - 0.5, color='k', lw=0.4, alpha=0.5)

hname    = 'H_in (input-space)' if args.hess_type == 'in' else 'H_out (output-space)'
norm_tag = ', diag-normalized'  if args.normalize_diag else ''
log_tag  = ', log-scale'        if args.log_scale      else ''
cross_tag = ', all-pairs cross' if args.all_cross      else ', adjacent cross (w=1)'
ax.set_title(
    f'Block Hessian  {hname}{norm_tag}{log_tag}{cross_tag}\n'
    f'layers={args.layers}  |  projs={args.names}',
    fontsize=9)

plt.tight_layout()
plt.savefig(args.output, dpi=150, bbox_inches='tight')
print(f'\nSaved → {args.output}')
