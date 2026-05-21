import glob
import os

import torch
import torch.distributed as dist
import torch.nn as nn

torch._dynamo.config.cache_size_limit = 256

local_rank = int(os.environ.get("LOCAL_RANK", 0))
local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", 1))

from torch.distributed import ReduceOp


@torch.compile
def sym_to_flat(A):
    N = A.shape[-1]
    idxs = torch.tril_indices(N, N, device=A.device)
    return A[idxs.unbind()]


@torch.compile
def flat_to_sym(V, N):
    A = torch.zeros(N, N, dtype=V.dtype, device=V.device)
    idxs = torch.tril_indices(N, N, device=V.device)
    A[idxs.unbind()] = V
    A[idxs[1, :], idxs[0, :]] = V
    return A

_LBUF = {}        # key: layer_idx -> (L, layer_name, block_idx)
_BATCH_ID  = [0]  # increments each backward pass
_PAIRS_DONE = set()  # (i, j) pairs computed in current batch, i < j


def should_compute_cross_pair(filter_mode,
                              block_window,
                              layer_idx,
                              other_idx,
                              block_idx,
                              other_block_idx,
                              extra_pairs=None):
    # Explicit extra pairs always included regardless of filter
    if extra_pairs:
        pair_key = (min(layer_idx, other_idx), max(layer_idx, other_idx))
        if pair_key in extra_pairs:
            return True
    if filter_mode == 'none':
        return True
    if filter_mode == 'linear_adjacent':
        return abs(layer_idx - other_idx) <= 1
    if filter_mode == 'gidx_band':
        # block_window is reused as gidx band width
        return abs(layer_idx - other_idx) <= block_window
    if filter_mode == 'block_adjacent':
        return abs(block_idx - other_block_idx) <= block_window
    raise ValueError(f'Unknown cross_filter: {filter_mode}')


class LinearNoBias(torch.autograd.Function):

    @staticmethod
    @torch.amp.custom_fwd(device_type='cuda')
    def forward(ctx, input, weight, mode, parent_class):
        ctx.save_for_backward(input, weight)
        ctx.mode = mode
        ctx.parent_class = parent_class

        return input @ weight.T

    @staticmethod
    @torch.amp.custom_bwd(device_type='cuda')
    def backward(ctx, grad_output):
        it, reset, div, cross, n_als = ctx.mode
        is_buffer = local_rank == ctx.parent_class.buffer_dev

        input, weight = ctx.saved_tensors
        ws = weight.shape
        grad_input = grad_output @ weight
        del weight
        if ctx.parent_class.collect_hess:
            op_dtype = ctx.parent_class.op_dtype
            bs = input.shape[0]
            layer_idx = ctx.parent_class.layer_idx
            block_idx = ctx.parent_class.block_idx
            layer_name = ctx.parent_class.layer_name
            cross_filter = ctx.parent_class.cross_filter
            cross_block_window = ctx.parent_class.cross_block_window

            with torch.amp.autocast('cuda', enabled=False):
                if it == 0:
                    if reset and is_buffer:
                        ctx.parent_class.hin.mul_(0)
                        ctx.parent_class.cross_hin.clear()
                        ctx.parent_class.cross_hout.clear()

                    grad_output = grad_output.float()
                    input = input.float()

                    # L = G^T X  (shape: m_j x n_j)
                    l_grad = torch.einsum('btm,btn->mn', grad_output, input)

                    # Diagonal blocks via L
                    in_hess = sym_to_flat(l_grad.T @ l_grad)      # (n_j x n_j)

                    handle_in = torch.distributed.reduce(
                        in_hess,
                        ctx.parent_class.buffer_dev,
                        op=ReduceOp.AVG,
                        async_op=True)
                    out_hess = sym_to_flat(l_grad @ l_grad.T)     # (m_j x m_j)
                    handle_out = torch.distributed.reduce(
                        out_hess,
                        ctx.parent_class.buffer_dev,
                        op=ReduceOp.AVG,
                        async_op=True)
                    del grad_output, input
                    handle_in.wait()
                    handle_out.wait()

                    if is_buffer:
                        ctx.parent_class.hin.add_(
                            in_hess.to(
                                ctx.parent_class.hin.device).to(op_dtype))
                        ctx.parent_class.hout.add_(
                            out_hess.to(
                                ctx.parent_class.hout.device).to(op_dtype))
                        ctx.parent_class.ct += bs

                    del in_hess, out_hess
                    torch.cuda.empty_cache()

                    if cross:

                        # If this layer already has an entry, the previous entry is stale
                        # (same layer can only appear once per backward pass).
                        if layer_idx in _LBUF:
                            _BATCH_ID[0] += 1
                            _LBUF.clear()
                            _PAIRS_DONE.clear()

                        _LBUF[layer_idx] = (l_grad.detach().cpu(),
                                            ctx.parent_class.layer_name,
                                            block_idx)

                        # Compute cross with every other layer currently in the buffer.
                        # Use _PAIRS_DONE to ensure each pair is computed exactly once.
                        # This handles non-monotonic backward orders (e.g. SwiGLU: down→up→gate).
                        Lj = l_grad
                        devI = 'cpu' if ctx.parent_class.hin.device.type == 'cpu' else ctx.parent_class.buffer_dev
                        devO = 'cpu' if ctx.parent_class.hout.device.type == 'cpu' else ctx.parent_class.buffer_dev

                        for other_idx in list(_LBUF.keys()):
                            if other_idx == layer_idx:
                                continue
                            pair_key = (min(layer_idx, other_idx),
                                        max(layer_idx, other_idx))
                            if pair_key in _PAIRS_DONE:
                                continue

                            Li, Li_name, other_block_idx = _LBUF[other_idx]
                            if not should_compute_cross_pair(
                                    cross_filter,
                                    cross_block_window,
                                    layer_idx,
                                    other_idx,
                                    block_idx,
                                    other_block_idx,
                                    extra_pairs=ctx.parent_class.extra_pairs):
                                continue

                            Li = Li.to(Lj.device, non_blocking=True).to(Lj.dtype)

                            m_i, n_i = Li.shape
                            m_j, n_j = Lj.shape

                            # Hybrid init: exact when dims match, rect truncation otherwise
                            if m_i == m_j:
                                cross_in_hess = (Li.T @ Lj / m_i).to(op_dtype).to(local_rank)
                            else:
                                r_m = min(m_i, m_j)
                                cross_in_hess = (Li[:r_m].T @ Lj[:r_m] / r_m).to(op_dtype).to(local_rank)

                            if n_i == n_j:
                                cross_out_hess = (Li @ Lj.T / n_i).to(op_dtype).to(local_rank)
                            else:
                                r_n = min(n_i, n_j)
                                cross_out_hess = (Li[:, :r_n] @ Lj[:, :r_n].T / r_n).to(op_dtype).to(local_rank)

                            # Local ALS for unequal-dim pairs
                            if (m_i != m_j or n_i != n_j) and n_als > 0:
                                H_I = cross_in_hess.float()
                                H_O = cross_out_hess.float()
                                # Normalise L matrices to unit Frobenius norm before ALS.
                                # Without this, ||Li||·||Lj|| ~ 10^9 causes the alternating
                                # updates to explode in O(||Li||^2·||Lj||^2) steps.
                                # H_I and H_O also stay normalised inside ALS because their
                                # scale is non-identifiable: c*H_I and H_O/c represent the
                                # same Kronecker product.
                                Li_scale = Li.float().norm().clamp(min=1e-30)
                                Lj_scale = Lj.float().norm().clamp(min=1e-30)
                                Li_f = Li.float() / Li_scale
                                Lj_f = Lj.float() / Lj_scale
                                H_I = H_I / H_I.norm().clamp(min=1e-30)
                                H_O = H_O / H_O.norm().clamp(min=1e-30)
                                for _ in range(n_als):
                                    H_I_next = Li_f.T @ H_O @ Lj_f
                                    H_I = H_I_next / H_I_next.norm().clamp(min=1e-30)
                                    H_O_next = Li_f @ H_I @ Lj_f.T
                                    H_O = H_O_next / H_O_next.norm().clamp(min=1e-30)

                                # ALS gives directions. Restore a KFAC-compatible scale using
                                # all rows/cols and geometric-mean dimensions; this reduces to
                                # /m and /n when the two layers have matching dimensions.
                                H_I_dir = H_I / H_I.norm().clamp(min=1e-30)
                                H_O_dir = H_O / H_O.norm().clamp(min=1e-30)
                                scale_I = (Li_f.T @ H_O_dir @ Lj_f).norm()
                                scale_O = (Li_f @ H_I_dir @ Lj_f.T).norm()
                                m_eff = (m_i * m_j) ** 0.5
                                n_eff = (n_i * n_j) ** 0.5
                                cross_in_hess  = (H_I_dir * (scale_I * Li_scale * Lj_scale / m_eff)).to(op_dtype)
                                cross_out_hess = (H_O_dir * (scale_O * Li_scale * Lj_scale / n_eff)).to(op_dtype)
                                del H_I, H_O, H_I_dir, H_O_dir, Li_f, Lj_f

                            torch.distributed.reduce(cross_in_hess,  ctx.parent_class.buffer_dev, op=ReduceOp.AVG)
                            torch.distributed.reduce(cross_out_hess, ctx.parent_class.buffer_dev, op=ReduceOp.AVG)

                            if is_buffer:
                                if other_idx not in ctx.parent_class.cross_hin or \
                                        ctx.parent_class.cross_hin[other_idx].shape != cross_in_hess.shape:
                                    ctx.parent_class.cross_hin[other_idx] = torch.zeros_like(
                                        cross_in_hess, device=devI, dtype=op_dtype)
                                if other_idx not in ctx.parent_class.cross_hout or \
                                        ctx.parent_class.cross_hout[other_idx].shape != cross_out_hess.shape:
                                    ctx.parent_class.cross_hout[other_idx] = torch.zeros_like(
                                        cross_out_hess, device=devO, dtype=op_dtype)

                                ctx.parent_class.cross_hin[other_idx].add_(
                                    cross_in_hess.to(devI))
                                ctx.parent_class.cross_hout[other_idx].add_(
                                    cross_out_hess.to(devO))

                            _PAIRS_DONE.add(pair_key)
                            del cross_in_hess, cross_out_hess
                            del Li
                            torch.cuda.empty_cache()

                    del l_grad
                    torch.cuda.empty_cache()
                else:
                    # Additional power iterations on B are not optimized and should be rewritten with einsums.
                    # Use at your own risk!
                    pass

        torch.cuda.empty_cache()
        return grad_input.to(local_rank), None, None, None


class CustomLinear(nn.Linear):

    def __init__(self,
                 buffer_dev,
                 cpu_offload,
                 load_fname,
                 layer_idx,
                 layer_name,
                 collect_hess=True,
                 use_fp64=False,
                 *args,
                 block_idx=None,
                 cross_filter='none',
                 cross_block_window=1,
                 extra_pairs=frozenset(),
                 **kwargs):
        super().__init__(*args, **kwargs)

        if cross_filter not in ('none', 'linear_adjacent', 'gidx_band', 'block_adjacent'):
            raise ValueError(f'Unknown cross_filter: {cross_filter}')
        if cross_block_window < 0:
            raise ValueError('cross_block_window must be non-negative')

        self.fname = load_fname
        self.layer_idx = layer_idx
        self.block_idx = layer_idx if block_idx is None else block_idx
        self.layer_name = layer_name
        self.cross_filter = cross_filter
        self.cross_block_window = cross_block_window
        self.extra_pairs = frozenset(extra_pairs)
        self.collect_hess = collect_hess
        self.op_dtype = torch.float32 if not use_fp64 else torch.float64
        if collect_hess and local_rank == buffer_dev:
            device = 'cpu' if cpu_offload else buffer_dev
            last_it = sorted(glob.glob(f'{load_fname}_hin*.pt'))
            if len(last_it) > 0 and os.path.exists(last_it[-1]):
                self.hin = torch.load(last_it[-1],
                                      map_location=torch.device(device)).to(
                                          self.op_dtype)
                print(f'loaded from {last_it[-1]}')
            else:
                self.hin = torch.zeros(self.in_features *
                                       (self.in_features + 1) // 2,
                                       dtype=self.op_dtype,
                                       device=device)
            last_it = sorted(glob.glob(f'{load_fname}_hout*.pt'))
            if len(last_it) > 0 and os.path.exists(last_it[-1]):
                self.hout = torch.load(last_it[-1],
                                       map_location=torch.device(device)).to(
                                           self.op_dtype)
                print(f'loaded from {last_it[-1]}')
            else:
                self.hout = torch.zeros(self.out_features *
                                        (self.out_features + 1) // 2,
                                        dtype=self.op_dtype,
                                        device=device)
                
            # cross_hin / cross_hout: dict keyed by partner gidx → tensor
            self.cross_hin  = {}
            self.cross_hout = {}

            if cpu_offload:
                self.hin.pin_memory()
                self.hout.pin_memory()

        self.buffer_dev = buffer_dev
        self.ct = 0

    def forward(self, input, mode):
        return LinearNoBias.apply(input, self.weight, mode, self)

    def reset_parameters(self):
        return
