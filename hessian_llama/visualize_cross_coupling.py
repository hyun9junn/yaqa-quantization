#!/usr/bin/env python3
"""
Visualize cross-layer Hessian coupling strength using BOTH Kronecker factors.

The true relative strength of the cross-block Hessian H_{jk} ≈ (H_I)_{jk} ⊗ (H_O)_{jk}
is:
    ρ_{jk} = ||H_{jk}||_F / sqrt(||H_{jj}||_F · ||H_{kk}||_F)
           = ρᴵ_{jk} × ρᴼ_{jk}

because ||A ⊗ B||_F = ||A||_F · ||B||_F.

Visualizing only H_I or only H_O is misleading — a large H_I with a tiny H_O gives a
negligible cross-block correction. The product ρᴵ × ρᴼ is the only honest significance
measure.

Three panels are shown:
  ρᴵ        : relative strength from H_I alone
  ρᴼ        : relative strength from H_O alone
  ρᴵ × ρᴼ  : combined (true) relative strength of H_I ⊗ H_O

Usage:
    python visualize_cross_coupling.py \\
        --save_path ./hess \\
        --layers 0,1,2 \\
        --names q,k,v,o,up,gate,down
"""

import argparse
import glob
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
                      diag_j: torch.Tensor, scale_j: float,
                      diag_k: torch.Tensor, scale_k: float) -> float:
    """||C||_F / sqrt((||diag_j||_F/scale_j) * (||diag_k||_F/scale_k))

    hin stores L^T L (no /m); cross_hin stores Li^T Lj / m_i.
    For a consistent ratio, the diagonal blocks must be divided by their
    respective output-dim (m) for rho_I, or input-dim (n) for rho_O.
    Passing scale_j = m_j and scale_k = m_k corrects this for rho_I;
    passing n_j and n_k corrects it for rho_O.
    """
    denom = ((diag_j.norm() / scale_j) * (diag_k.norm() / scale_k)) ** 0.5
    return (C.norm() / denom).item() if denom > 0 else float('nan')


def load_cross_map(save_path: str, label: str, suffix: str) -> dict:
    """Return {partner_gidx: filepath} for all cross files of this layer."""
    cross_map = {}
    for fp in sorted(glob.glob(os.path.join(save_path, f'{label}_cross*_{suffix}'))):
        basename = os.path.basename(fp)
        mid = basename[len(label) + len('_cross'):]
        gidx_str = mid[:mid.index('_')]
        try:
            cross_map[int(gidx_str)] = fp
        except ValueError:
            pass
    return cross_map


# ── args ──────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument('--save_path', required=True)
parser.add_argument('--layers', default='0')
parser.add_argument('--names', default=','.join(LAYER_ORDER))
parser.add_argument('--dtype', choices=['float32', 'float64'], default='float32')
args = parser.parse_args()

layer_nums = [int(x) for x in args.layers.split(',')]
names      = [x.strip() for x in args.names.split(',')]
dtype      = torch.float32 if args.dtype == 'float32' else torch.float64

# ── discover & load diagonal blocks (both H_I and H_O) ───────────────────────

entries = []
for lb in sorted(layer_nums):
    for nm in LAYER_ORDER:
        if nm not in names:
            continue
        gidx  = lb * 7 + LAYER_ORDER.index(nm)
        f_in  = os.path.join(args.save_path, f'{lb}_{nm}_hin.pt')
        f_out = os.path.join(args.save_path, f'{lb}_{nm}_hout.pt')
        if not os.path.exists(f_in):
            continue
        label = f'{lb}_{nm}'
        entries.append({
            'label':      label,
            'gidx':       gidx,
            'cross_in':   load_cross_map(args.save_path, label, 'hin.pt'),
            'cross_out':  load_cross_map(args.save_path, label, 'hout.pt'),
            'f_in':       f_in,
            'f_out':      f_out if os.path.exists(f_out) else None,
        })

if not entries:
    raise SystemExit(f'[error] No hin files found in {args.save_path!r}')

for e in entries:
    v = torch.load(e['f_in'], map_location='cpu').to(dtype)
    n = infer_n(len(v))
    e['n_in']   = n
    e['diag_in'] = flat_to_sym(v, n)

    if e['f_out'] is not None:
        v = torch.load(e['f_out'], map_location='cpu').to(dtype)
        m = infer_n(len(v))
        e['m_out']    = m
        e['diag_out'] = flat_to_sym(v, m)
    else:
        e['m_out']    = None
        e['diag_out'] = None

N      = len(entries)
labels = [e['label'] for e in entries]
names_used = [nm for nm in LAYER_ORDER if nm in names]
block_size = len(names_used)

print(f'\nLayers ({N}): {labels}')
print(f'Block size: {block_size} names per transformer layer\n')

# ── build N×N coupling matrices ───────────────────────────────────────────────

rel_I   = torch.full((N, N), float('nan'))
rel_O   = torch.full((N, N), float('nan'))
rel_comb = torch.full((N, N), float('nan'))

for i in range(N):
    rel_I[i, i]    = 1.0
    rel_O[i, i]    = 1.0
    rel_comb[i, i] = 1.0

gidx_to_idx = {e['gidx']: i for i, e in enumerate(entries)}

print(f'{"Pair":>20}  {"ρᴵ":>10}  {"ρᴼ":>10}  {"ρᴵ×ρᴼ":>10}')
print('-' * 58)

for j, ej in enumerate(entries):
    for other_gidx, fp_in in ej['cross_in'].items():
        if other_gidx not in gidx_to_idx:
            continue
        k  = gidx_to_idx[other_gidx]
        ek = entries[k]

        C_in = torch.load(fp_in, map_location='cpu').to(dtype)
        if C_in.shape != (ek['n_in'], ej['n_in']):
            print(f'  [warn] H_I shape mismatch for {ej["label"]}→{ek["label"]}, skipping')
            continue

        # rho_I: hin stores L^T L (no /m); correct by dividing hin by m_out
        rI = float('nan')
        if ej['m_out'] is not None and ek['m_out'] is not None:
            rI = relative_strength(C_in,
                                   ej['diag_in'], ej['m_out'],
                                   ek['diag_in'], ek['m_out'])

        # rho_O: hout stores L L^T (no /n); correct by dividing hout by n_in
        rO = float('nan')
        fp_out = ej['cross_out'].get(other_gidx)
        if fp_out and ej['diag_out'] is not None and ek['diag_out'] is not None:
            C_out = torch.load(fp_out, map_location='cpu').to(dtype)
            if C_out.shape == (ek['m_out'], ej['m_out']):
                rO = relative_strength(C_out,
                                       ej['diag_out'], ej['n_in'],
                                       ek['diag_out'], ek['n_in'])

        rC = rI * rO if (not np.isnan(rI) and not np.isnan(rO)) else float('nan')

        rel_I[j, k]    = rel_I[k, j]    = rI
        rel_O[j, k]    = rel_O[k, j]    = rO
        rel_comb[j, k] = rel_comb[k, j] = rC

        print(f'{ej["label"]:>12} ↔ {ek["label"]:<8}  '
              f'{rI:>10.4f}  {rO:>10.4f}  {rC:>10.4f}')

print()

# ── plot 3-panel figure ───────────────────────────────────────────────────────

fig_w = max(24, N * 1.5 + 6)
fig_h = max(7,  N * 0.5 + 2)
fig, axes = plt.subplots(1, 3, figsize=(fig_w, fig_h))

panels = [
    (axes[0], rel_I.numpy(),
     'ρᴵ  =  ||cross_hin_{jk}||_F / √((||hin_j||/m_j)·(||hin_k||/m_k))\n'
     'Relative strength — input Kronecker factor  (hin÷m normalised)'),
    (axes[1], rel_O.numpy(),
     'ρᴼ  =  ||cross_hout_{jk}||_F / √((||hout_j||/n_j)·(||hout_k||/n_k))\n'
     'Relative strength — output Kronecker factor  (hout÷n normalised)'),
    (axes[2], rel_comb.numpy(),
     'ρᴵ × ρᴼ  =  ||H_{jk}||_F / √(||H_{jj}||·||H_{kk}||)\n'
     'TRUE combined relative strength  (the right metric)'),
]

for ax, data, title in panels:
    mask    = np.isnan(data)
    valid   = data[~mask & ~np.eye(N, dtype=bool)]

    cmap_obj = plt.cm.get_cmap('YlOrRd').copy()
    cmap_obj.set_bad(color='#cccccc')
    masked  = np.ma.array(data, mask=mask)

    vmin_val = max(float(valid.min()), 1e-6) if valid.size > 0 else 1e-6
    vmax_val = float(valid.max()) if valid.size > 0 else 1.0
    norm = mcolors.LogNorm(vmin=vmin_val, vmax=max(vmax_val, vmin_val * 1.01))

    im = ax.imshow(masked, norm=norm, cmap=cmap_obj, aspect='auto')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(range(N))
    ax.set_yticks(range(N))
    ax.set_xticklabels(labels, rotation=90, fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_title(title, fontsize=8, pad=8)

    if N <= 40:
        fs = max(4, 7 - N // 8)
        for i in range(N):
            for jj in range(N):
                if mask[i, jj]:
                    ax.text(jj, i, '—', ha='center', va='center',
                            fontsize=fs, color='#999999')
                else:
                    val = data[i, jj]
                    if val > 0 and vmax_val > vmin_val:
                        normed = (np.log10(max(val, 1e-10)) - np.log10(vmin_val)) / \
                                 (np.log10(vmax_val) - np.log10(vmin_val))
                    else:
                        normed = 0.5
                    color = 'white' if normed > 0.65 else 'black'
                    ax.text(jj, i, f'{val:.3f}', ha='center', va='center',
                            fontsize=fs, color=color)

    for b in range(block_size, N, block_size):
        ax.axhline(b - 0.5, color='steelblue', linewidth=1.5,
                   linestyle='--', alpha=0.6)
        ax.axvline(b - 0.5, color='steelblue', linewidth=1.5,
                   linestyle='--', alpha=0.6)

plt.suptitle(
    'Cross-Hessian significance: ρᴵ, ρᴼ, and combined ρᴵ×ρᴼ\n'
    '||H_{jk}||_F / √(||H_{jj}||·||H_{kk}||)  where  H_{jj}=(hin_j/m_j)⊗(hout_j/n_j)\n'
    'Gray = no cross data   |   ALS-refined pairs (m_i≠m_j) use geometric-mean normalisation',
    fontsize=10, y=1.02)
plt.tight_layout()

out_fig = os.path.join(args.save_path, 'cross_hessian_grid.png')
plt.savefig(out_fig, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f'Saved → {out_fig}')
