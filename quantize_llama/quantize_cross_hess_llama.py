"""Quantize LLaMA using pre-computed block-banded Hessians with Block-LDL cross corrections.

Weights within each transformer block are processed in LAYER_ORDER (q, k, v, o, up, gate, down),
matching the global gidx assignment used during Hessian collection.  The quantization error
delta_W = W - hatW from each weight is passed as a cross-block correction to the next weight
whenever a cross-Hessian file is available.

Usage:
    python quantize_cross_hess_llama.py \\
        --base_model <model_path> \\
        --hess_path  <hess_dir>   \\
        --save_path  <out_dir>    \\
        --codebook   <name>       \\
        --scale_override 1.0      \\
        [--no_cross]              \\
        [--device cuda:0]
"""

import argparse
import json
import os
import sys

import glog
import torch
from operator import attrgetter
from transformers import AutoModelForCausalLM

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from lib import utils
from lib.algo import ldlq
from lib.algo.cross_ldlq import compute_cross_correction, get_cross_strength, load_cross_kronecker
from lib.codebook import bitshift

# Weight order matching gidx = layer_idx * 7 + position used in hessian collection
LAYER_ORDER = ['q', 'k', 'v', 'o', 'up', 'gate', 'down']
LAYER_ATTRS = {
    'q':    'self_attn.q_proj',
    'k':    'self_attn.k_proj',
    'v':    'self_attn.v_proj',
    'o':    'self_attn.o_proj',
    'up':   'mlp.up_proj',
    'gate': 'mlp.gate_proj',
    'down': 'mlp.down_proj',
}

# ── argument parsing ──────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument('--seed',           default=0,      type=int)
parser.add_argument('--base_model',     required=True,  type=str)
parser.add_argument('--hess_path',      required=True,  type=str)
parser.add_argument('--save_path',      required=True,  type=str)
parser.add_argument('--codebook',       required=True,  type=str)
parser.add_argument('--sigma_reg',      default=1e-2,   type=float)
parser.add_argument('--scale_override', default=1.0,    type=float)
parser.add_argument('--use_fp64',       action='store_true')
parser.add_argument('--td_x',          default=16,     type=int)
parser.add_argument('--td_y',          default=16,     type=int)
parser.add_argument('--L',             default=16,     type=int)
parser.add_argument('--K',             default=2,      type=int)
parser.add_argument('--V',             default=2,      type=int)
parser.add_argument('--tlut_bits',     default=0,      type=int)
parser.add_argument('--decode_mode',   default='lut',  type=str)
parser.add_argument('--tp_rank',       default=8,      type=int)
parser.add_argument('--no_cross',      action='store_true',
                    help='Disable cross-block correction (ablation)')
parser.add_argument('--device',        default='cuda:0', type=str)
parser.add_argument('--skip_list',     default=None,   type=str,
                    help='Comma-separated list of weights to skip, e.g. 0_q,1_k')

# ── parent set arguments ──────────────────────────────────────────────────────
parser.add_argument('--parent_band',      default=1,    type=int,
                    help='Include all j with gidx_k - gidx_j <= W as parents (default 1).')
parser.add_argument('--parent_topR',      default=0,    type=int,
                    help='Additionally include the top-R parents by cross-Hessian strength '
                         '(beyond the band). 0 = disabled.')
parser.add_argument('--parent_threshold', default=0.0,  type=float,
                    help='Additionally include parents with strength > TAU. 0.0 = disabled.')
parser.add_argument('--parent_lookback',  default=50,   type=int,
                    help='Max gidx distance to search when using --parent_topR or '
                         '--parent_threshold.')
parser.add_argument('--parent_explicit',  default='',   type=str,
                    help='Explicit per-weight parent sets. '
                         'Format: "k_label:j_label1,j_label2;..." '
                         'e.g. "1_q:0_v,0_gate;2_down:1_up". '
                         'Use "*" as a block-index wildcard to expand across all layers: '
                         '"*_v:*_q" means every v gets q of the same block as a parent.')
parser.add_argument('--parent_map',       default='',   type=str,
                    help='JSON file produced by hessian_llama/select_cross_hessian_pairs.py. '
                         'Its parent_map/edges are added to P(k).')


# ── parent set helpers ────────────────────────────────────────────────────────

def _parse_explicit_parent_map(parent_explicit_str: str, gidx_to_label: dict,
                               num_layers: int) -> dict:
    """Parse '--parent_explicit' string into {k_gidx: [j_gidx, ...]}.

    Use '*' as a wildcard for the block index to expand a pattern across all layers.
    Examples:
        "1_q:0_v,0_gate"   explicit for one weight
        "*_v:*_q"           for every block, v's parent set includes q of the same block
        "*_v:*_q;*_down:*_up"  multiple wildcard patterns
    """
    label_to_gidx = {v: k for k, v in gidx_to_label.items()}
    parent_map = {}
    if not parent_explicit_str.strip():
        return parent_map
    for entry in parent_explicit_str.split(';'):
        entry = entry.strip()
        if not entry:
            continue
        if ':' not in entry:
            raise ValueError(f'--parent_explicit entry must be "k_label:j1,j2,...": {entry!r}')
        k_str, parents_str = entry.split(':', 1)
        k_str = k_str.strip()
        parent_labels = [p.strip() for p in parents_str.split(',')]

        if '*' in k_str or any('*' in p for p in parent_labels):
            for i in range(num_layers):
                k_label = k_str.replace('*', str(i))
                if k_label not in label_to_gidx:
                    continue
                k_gidx = label_to_gidx[k_label]
                j_gidxs = []
                for p in parent_labels:
                    j_label = p.replace('*', str(i))
                    if j_label in label_to_gidx:
                        j_gidxs.append(label_to_gidx[j_label])
                if j_gidxs:
                    parent_map.setdefault(k_gidx, []).extend(j_gidxs)
        else:
            k_gidx = label_to_gidx[k_str]
            parent_map[k_gidx] = [label_to_gidx[p] for p in parent_labels]
    return parent_map


def _merge_parent_maps(*maps: dict) -> dict:
    merged = {}
    for parent_map in maps:
        for k, parents in parent_map.items():
            merged.setdefault(k, set()).update(parents)
    return {k: sorted(v) for k, v in merged.items()}


def _parse_parent_map_file(parent_map_path: str, gidx_to_label: dict) -> dict:
    if not parent_map_path:
        return {}
    label_to_gidx = {v: k for k, v in gidx_to_label.items()}
    with open(parent_map_path) as f:
        payload = json.load(f)

    parent_map = {}
    if 'parent_map' in payload:
        for child_label, parent_labels in payload['parent_map'].items():
            if child_label not in label_to_gidx:
                raise ValueError(f'Unknown child label in --parent_map: {child_label}')
            child_gidx = label_to_gidx[child_label]
            for parent_label in parent_labels:
                if parent_label not in label_to_gidx:
                    raise ValueError(f'Unknown parent label in --parent_map: {parent_label}')
                parent_gidx = label_to_gidx[parent_label]
                if parent_gidx >= child_gidx:
                    raise ValueError(
                        f'--parent_map only supports already-quantized parents: '
                        f'{parent_label} must have gidx < {child_label}')
                parent_map.setdefault(child_gidx, []).append(parent_gidx)
    elif 'edges' in payload:
        for edge in payload['edges']:
            child_label = edge['child']
            parent_label = edge['parent']
            if child_label not in label_to_gidx or parent_label not in label_to_gidx:
                raise ValueError(f'Unknown edge in --parent_map: {parent_label}->{child_label}')
            child_gidx = label_to_gidx[child_label]
            parent_gidx = label_to_gidx[parent_label]
            if parent_gidx >= child_gidx:
                raise ValueError(
                    f'--parent_map only supports already-quantized parents: '
                    f'{parent_label} must have gidx < {child_label}')
            parent_map.setdefault(child_gidx, []).append(parent_gidx)
    else:
        raise ValueError('--parent_map JSON must contain either "parent_map" or "edges"')

    return {k: sorted(set(v)) for k, v in parent_map.items()}


def _max_explicit_parent_distance(explicit_map: dict) -> int:
    max_dist = 0
    for k_gidx, parents in explicit_map.items():
        for j_gidx in parents:
            if j_gidx < k_gidx:
                max_dist = max(max_dist, k_gidx - j_gidx)
    return max_dist


def compute_parent_set(k_gidx: int, k_label: str, args, history: dict,
                       gidx_to_label: dict, explicit_map: dict) -> list:
    """Return sorted list of parent gidx values in P(k).

    P(k) = band({j: k-j <= parent_band})
         ∪ topR(parent_topR)          [by cross-Hessian strength]
         ∪ threshold(parent_threshold)
         ∪ explicit_map.get(k, [])
    Only includes j values that are present in history.
    """
    parents = set()

    # Band parents
    for j in range(max(0, k_gidx - args.parent_band), k_gidx):
        if j in history:
            parents.add(j)

    # TopR / threshold: scan candidates beyond the band
    use_strength = (args.parent_topR > 0) or (args.parent_threshold > 0.0)
    if use_strength:
        candidates = {}
        min_j = max(0, k_gidx - args.parent_lookback)
        for j in range(min_j, k_gidx - args.parent_band):
            if j not in history or j not in gidx_to_label:
                continue
            j_label = gidx_to_label[j]
            strength = get_cross_strength(args.hess_path, j_label, j, k_gidx, k_label)
            if strength > 0.0:
                candidates[j] = strength

        if args.parent_topR > 0:
            top_r = sorted(candidates.items(), key=lambda x: -x[1])[:args.parent_topR]
            parents.update(j for j, _ in top_r)

        if args.parent_threshold > 0.0:
            parents.update(j for j, s in candidates.items() if s > args.parent_threshold)

    # Explicit or selected parent-map parents
    for j in explicit_map.get(k_gidx, []):
        if j in history:
            parents.add(j)

    return sorted(parents)


# ── per-weight quantization ───────────────────────────────────────────────────

def _compute_ldl_factors(H_sym, block_size, SU_or_SV, sigma_reg, dtype_):
    """Rotate H_sym, run block-LDL, return lower-triangular L factor (zeros on diagonal)."""
    n = H_sym.shape[0]
    H_sym = H_sym.clone()
    H_sym /= torch.diag(H_sym).mean()
    H_rot = utils.matmul_hadUt(utils.matmul_hadUt(H_sym * SU_or_SV).T * SU_or_SV).T
    L = None
    fsr = 0.0
    while L is None:
        H_rot[torch.arange(n), torch.arange(n)] += sigma_reg
        fsr += sigma_reg
        L = utils.block_LDL(H_rot, block_size)
    glog.info(f'  sigma_reg accumulated: {fsr:.2e}')
    L = L[0].float()
    L[torch.arange(n), torch.arange(n)] = 0
    return L


def quantize_and_pack(W_eff, Hin_sym, Hout_sym, gidx, cb, args, device, orig_dtype):
    """
    Quantize W_eff (m × n, possibly cross-corrected) using LDLQ.

    Returns:
        hatW       – quantized weight in original space (m × n, float32)
        packed     – packed trellis codes
        SU, SV     – random sign matrices
        Wscale     – weight scale scalar
        proxy_err  – float, 2-sided proxy quantization error
    """
    dtype_ = torch.float64 if args.use_fp64 else torch.float32
    m, n = W_eff.shape
    W_eff = W_eff.to(device, dtype=dtype_)

    torch.manual_seed(gidx)
    SU = (torch.randn(n, device=device).sign() + 1e-5).sign().to(dtype_)
    SV = (torch.randn(m, device=device).sign() + 1e-5).sign().to(dtype_)

    Hin  = Hin_sym.clone().to(device, dtype=torch.float64)
    Hout = Hout_sym.clone().to(device, dtype=torch.float64)

    Lin  = _compute_ldl_factors(Hin,  args.td_y, SU, args.sigma_reg, dtype_)
    Lout = _compute_ldl_factors(Hout, args.td_x, SV, args.sigma_reg, dtype_)

    Wr = utils.matmul_hadUt(utils.matmul_hadUt(W_eff.T * SV).T * SU)
    Wscale = Wr.square().mean().sqrt() / (
        cb.lut.to(torch.float64).square().mean().sqrt().float() * args.scale_override)
    Wr /= Wscale

    has_kernel = utils.has_kernel(args.decode_mode, args.L, args.K, args.V,
                                  args.tlut_bits, args.td_x, args.td_y)
    hatWr, Qidxs = ldlq.LDLQ_2hess(
        Wr, Lin, Lout, args.td_x, args.td_y, args.V, cb, for_kernel=has_kernel)

    Wr *= Wscale
    hatWr *= Wscale
    hatW = (utils.matmul_hadU((utils.matmul_hadU(hatWr) * SU).T) * SV).T

    # 2-sided proxy error (in rotated, normalised space)
    Hin_n  = Hin_sym.clone().to(device, dtype=dtype_)
    Hout_n = Hout_sym.clone().to(device, dtype=dtype_)
    Hin_n  /= torch.diag(Hin_n).mean()
    Hout_n /= torch.diag(Hout_n).mean()
    diff = (Wr - hatWr) / Wscale
    proxy_err = torch.trace(diff @ Hin_n @ diff.T @ Hout_n).item()
    del Hin_n, Hout_n, diff

    # pack trellis
    Qidxs = Qidxs.cpu()
    packed = cb.pack_trellis(
        Qidxs.reshape(m // args.td_x, args.td_x, n // args.td_y, args.td_y // args.V)
              .transpose(1, 2)
              .reshape(-1, args.td_x * args.td_y // args.V))
    if has_kernel:
        packed = (packed.view(torch.uint8).view(-1, 2).flip((-1,))
                        .reshape(m // 16 // 2, 2, n // 16 // 2, 2,
                                 16 * 16 // 8, args.K)
                        .permute(0, 2, 4, 3, 1, 5).flip((-1,))
                        .contiguous().flatten()
                        .view(torch.int16).reshape(packed.shape))
    else:
        packed = packed.view(torch.int16)

    return hatW, packed, SU, SV, Wscale, proxy_err


# ── per-layer quantization ────────────────────────────────────────────────────

def quantize_transformer_layer(layer, layer_idx, cb, args, device, skip_list,
                                history: dict, gidx_to_label: dict, explicit_map: dict):
    """Quantize all linear layers in one transformer block using sparse parent-set corrections.

    history: dict mapping gidx → state, where state = {
        'delta_W'  : (m, n) W - hatW on cpu float32   (zero if weight was skipped/already done)
        'Hin_sym'  : (n, n) input Hessian on cpu float64
        'Hout_sym' : (m, m) output Hessian on cpu float64
        'label'    : str e.g. '0_down'
    }
    Returns updated history.
    """
    orig_dtype = next(layer.parameters()).dtype
    layer = layer.to(device).float()
    dtype_ = torch.float64 if args.use_fp64 else torch.float32

    # Determine how far back we need to keep history entries alive
    effective_lookback = args.parent_band
    if args.parent_topR > 0 or args.parent_threshold > 0.0 or explicit_map:
        effective_lookback = max(effective_lookback, args.parent_lookback)
    effective_lookback = max(effective_lookback,
                             _max_explicit_parent_distance(explicit_map))

    for name in LAYER_ORDER:
        gidx  = layer_idx * len(LAYER_ORDER) + LAYER_ORDER.index(name)
        label = f'{layer_idx}_{name}'

        # Prune history entries that are too far back to be needed
        min_needed = gidx - effective_lookback
        for j in [j for j in list(history.keys()) if j < min_needed]:
            del history[j]

        if label in skip_list:
            glog.info(f'Skipping {label}')
            # No quantization error → delta_W = 0; still add to history so future
            # weights can reference it (correction will be zero).
            W_skip = attrgetter(LAYER_ATTRS[name])(layer).weight
            m, n = W_skip.shape
            hin_path  = f'{args.hess_path}/{label}_hin.pt'
            hout_path = f'{args.hess_path}/{label}_hout.pt'
            if os.path.exists(hin_path) and os.path.exists(hout_path):
                Hin_sym  = utils.flat_to_sym(torch.load(hin_path,  map_location='cpu'), n).to(torch.float64)
                Hout_sym = utils.flat_to_sym(torch.load(hout_path, map_location='cpu'), m).to(torch.float64)
            else:
                Hin_sym  = torch.eye(n, dtype=torch.float64)
                Hout_sym = torch.eye(m, dtype=torch.float64)
            history[gidx] = {
                'delta_W':  torch.zeros(m, n, dtype=torch.float32),
                'Hin_sym':  Hin_sym,
                'Hout_sym': Hout_sym,
                'label':    label,
            }
            continue

        save_path = f'{args.save_path}/{label}.pt'
        if os.path.exists(save_path):
            glog.info(f'{label} already exists, restoring history entry with delta_W=0')
            W_orig = attrgetter(LAYER_ATTRS[name])(layer).weight.to(dtype_)
            m, n = W_orig.shape
            Hin_sym  = utils.flat_to_sym(
                torch.load(f'{args.hess_path}/{label}_hin.pt',  map_location='cpu'), n).to(torch.float64)
            Hout_sym = utils.flat_to_sym(
                torch.load(f'{args.hess_path}/{label}_hout.pt', map_location='cpu'), m).to(torch.float64)
            history[gidx] = {
                'delta_W':  torch.zeros(m, n, dtype=torch.float32),
                'Hin_sym':  Hin_sym.cpu(),
                'Hout_sym': Hout_sym.cpu(),
                'label':    label,
            }
            continue

        # ── load weight and diagonal Hessians ───────────────────────────────
        W_orig = attrgetter(LAYER_ATTRS[name])(layer).weight.to(dtype_)
        m, n   = W_orig.shape

        Hin_sym  = utils.flat_to_sym(
            torch.load(f'{args.hess_path}/{label}_hin.pt',  map_location='cpu'), n).to(torch.float64)
        Hout_sym = utils.flat_to_sym(
            torch.load(f'{args.hess_path}/{label}_hout.pt', map_location='cpu'), m).to(torch.float64)

        # ── sparse parent-set cross correction ──────────────────────────────
        W_eff = W_orig.clone()
        if not args.no_cross:
            parents = compute_parent_set(gidx, label, args, history, gidx_to_label, explicit_map)
            if parents:
                glog.info(f'{label}: cross correction from {[gidx_to_label[j] for j in parents]}')
            for j in parents:
                j_state = history[j]
                cross_hin, cross_hout = load_cross_kronecker(
                    args.hess_path, j_state['label'], j, gidx, label)
                if cross_hin is not None:
                    correction = compute_cross_correction(
                        j_state['delta_W'],
                        cross_hin, cross_hout,
                        j_state['Hin_sym'], j_state['Hout_sym'],
                        args.sigma_reg, device)
                    W_eff = W_eff + correction.to(device, dtype=dtype_)
                else:
                    glog.info(f'  {label} ← {j_state["label"]}: no cross-Hessian file, skipped')

        # ── quantize ────────────────────────────────────────────────────────
        cb = cb.to(device).to(orig_dtype)
        hatW, packed, SU, SV, Wscale, proxy_err = quantize_and_pack(
            W_eff, Hin_sym, Hout_sym, gidx, cb, args, device, orig_dtype)

        delta_W = (W_orig.to(device) - hatW).cpu().to(torch.float32)

        torch.save({
            'trellis':   packed.cpu(),
            'SU':        SU.to(orig_dtype).cpu(),
            'SV':        SV.to(orig_dtype).cpu(),
            'Wscale':    Wscale,
            'proxy_err': proxy_err,
            'tlut':      cb.tlut.data.to(orig_dtype).cpu() if hasattr(cb, 'tlut') else None,
            'rcp':       0,
            'tp_rank':   args.tp_rank,
        }, save_path)

        glog.info(f'{label}  proxy_err={proxy_err:.4e}')

        history[gidx] = {
            'delta_W':  delta_W,
            'Hin_sym':  Hin_sym.cpu(),
            'Hout_sym': Hout_sym.cpu(),
            'label':    label,
        }

        cb = cb.cpu()
        utils.clean()

    # save layer norms
    torch.save({
        'input_layernorm':          layer.input_layernorm.weight.to(orig_dtype),
        'post_attention_layernorm': layer.post_attention_layernorm.weight.to(orig_dtype),
    }, f'{args.save_path}/{layer_idx}_layernorm.pt')

    layer = layer.to(orig_dtype).cpu()
    return history


# ── main ──────────────────────────────────────────────────────────────────────

def main(args):
    skip_list = set(args.skip_list.split(',')) if args.skip_list else set()
    device    = args.device

    cb = bitshift.bitshift_codebook(
        L=args.L, K=args.K, V=args.V,
        tlut_bits=args.tlut_bits, decode_mode=args.decode_mode)

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype='auto', low_cpu_mem_usage=True)

    num_layers = len(model.model.layers)
    gidx_to_label = {
        i * len(LAYER_ORDER) + pos: f'{i}_{name}'
        for i in range(num_layers)
        for pos, name in enumerate(LAYER_ORDER)
    }
    explicit_map = _parse_explicit_parent_map(args.parent_explicit, gidx_to_label, num_layers)
    selected_map = _parse_parent_map_file(args.parent_map, gidx_to_label)
    explicit_map = _merge_parent_maps(explicit_map, selected_map)
    if explicit_map:
        glog.info(f'Parent map: { {gidx_to_label[k]: [gidx_to_label[j] for j in vs] for k, vs in explicit_map.items()} }')

    quip_params = {
        'codebook':         args.codebook,
        'codebook_version': cb.version,
        'L':  args.L, 'K': args.K, 'V': args.V,
        'tlut_bits':        args.tlut_bits,
        'decode_mode':      args.decode_mode,
        'td_x':             args.td_x,
        'td_y':             args.td_y,
        'split_for_tp':     False,
        'skip_list':        list(skip_list),
    }
    model.config.update({'quip_params': quip_params})
    torch.save({'quant_args': args, 'model_config': model.config},
               os.path.join(args.save_path, 'config.pt'))

    glog.info('Model loaded')
    glog.info(f'Parent set: band={args.parent_band}, topR={args.parent_topR}, '
              f'threshold={args.parent_threshold}, lookback={args.parent_lookback}, '
              f'parent_map={args.parent_map or "none"}')

    history = {}
    for i in range(num_layers):
        glog.info(f'=== Transformer layer {i} ===')
        layer = model.model.layers[i]
        history = quantize_transformer_layer(
            layer, i, cb, args, device, skip_list, history, gidx_to_label, explicit_map)
        model.model.layers[i] = None
        utils.clean()

    glog.info('Quantization complete')


if __name__ == '__main__':
    torch.set_grad_enabled(False)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    os.makedirs(args.save_path, exist_ok=True)
    main(args)
