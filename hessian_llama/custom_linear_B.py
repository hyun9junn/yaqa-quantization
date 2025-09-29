import glob
import os

import torch
import torch.distributed as dist
import torch.nn as nn

torch._dynamo.config.cache_size_limit = 256

local_rank = int(os.environ.get("LOCAL_RANK", 0))
local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", 1))

from torch.distributed import ReduceOp

def _rect_identity_aligners(dim_a: int, dim_b: int, device, dtype):

    r = min(dim_a, dim_b)
    A = torch.zeros(dim_a, r, device=device, dtype=dtype)
    B = torch.zeros(dim_b, r, device=device, dtype=dtype)
    A[:r, torch.arange(r)] = 1
    B[:r, torch.arange(r)] = 1
    return A, B, r

def _orthonormal_aligners(dim_a: int, dim_b: int, device, dtype, seed: int = 0):

    g = torch.Generator(device=device); g.manual_seed(seed)
    r = min(dim_a, dim_b)
    A_full = torch.randn(dim_a, r, generator=g, device=device, dtype=dtype)
    B_full = torch.randn(dim_b, r, generator=g, device=device, dtype=dtype)
    A, _ = torch.linalg.qr(A_full, mode='reduced')  # (dim_a, r)
    B, _ = torch.linalg.qr(B_full, mode='reduced')  # (dim_b, r)
    return A, B, r

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

_LBUF = {}  # key: layer_idx -> Tensor L (cpu, shape: m_j x n_j)


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
        it, reset, div, cross, align_mode = ctx.mode
        is_buffer = local_rank == ctx.parent_class.buffer_dev

        input, weight = ctx.saved_tensors
        ws = weight.shape
        grad_input = grad_output @ weight
        del weight
        if ctx.parent_class.collect_hess:
            op_dtype = ctx.parent_class.op_dtype
            bs = input.shape[0]
            layer_idx = ctx.parent_class.layer_idx
            layer_name = ctx.parent_class.layer_name

            with torch.amp.autocast('cuda', enabled=False):
                if it == 0:
                    if reset and is_buffer:
                        ctx.parent_class.hin.mul_(0)
                        if hasattr(ctx.parent_class, 'cross_hin') and ctx.parent_class.cross_hin is not None:
                            ctx.parent_class.cross_hin.mul_(0)
                        if hasattr(ctx.parent_class, 'cross_hout') and ctx.parent_class.cross_hout is not None:
                            ctx.parent_class.cross_hout.mul_(0)

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
                        if div:
                            ctx.parent_class.hin.div_(ctx.parent_class.ct)
                            ctx.parent_class.hout.div_(ctx.parent_class.ct)
                            ctx.parent_class.ct = 0

                    del in_hess, out_hess
                    torch.cuda.empty_cache()

                    if cross:

                        # --- Cross-layer (band w=1): match with neighbor i = j+1 ---
                        # Store my L to per-process CPU buffer
                        _LBUF[layer_idx] = (l_grad.detach().cpu(), ctx.parent_class.layer_name)

                        # If neighbor's L is already there, form cross terms and pop
                        if (layer_idx + 1) in _LBUF:
                            Li, Li_name = _LBUF.pop(layer_idx + 1)  # i = j+1

                            SKIP = ['q']
                            if Li_name in SKIP:
                                del Li
                                pass
                            else:

                                Lj = l_grad                     # j = current
                                Li = Li.to(Lj.device, non_blocking=True).to(Lj.dtype)

                                m_i, n_i = Li.shape
                                m_j, n_j = Lj.shape
                                if align_mode == 'rect':
                                    A_i, A_j, r_m = _rect_identity_aligners(m_i, m_j, Lj.device, Lj.dtype)
                                    B_i, B_j, r_n = _rect_identity_aligners(n_i, n_j, Lj.device, Lj.dtype)
                                else:
                                    A_i, A_j, r_m = _orthonormal_aligners(m_i, m_j, Lj.device, Lj.dtype, seed=1234)
                                    B_i, B_j, r_n = _orthonormal_aligners(n_i, n_j, Lj.device, Lj.dtype, seed=5678)

                                Li_m = A_i.T @ Li        # (r_m, n_i)
                                Lj_m = A_j.T @ Lj        # (r_m, n_j)
                                cross_in_hess  = (Li_m.T @ Lj_m).to(op_dtype).to(local_rank)   # (n_i, n_j)

                                Li_n = Li @ B_i          # (m_i, r_n)
                                Lj_n = Lj @ B_j          # (m_j, r_n)
                                cross_out_hess = (Li_n @ Lj_n.T).to(op_dtype).to(local_rank)   # (m_i, m_j)

                                cross_in_hess  /= r_m  # m_j
                                cross_out_hess /= r_n  # n_j

                                torch.distributed.reduce(cross_in_hess,  ctx.parent_class.buffer_dev, op=ReduceOp.AVG)
                                torch.distributed.reduce(cross_out_hess, ctx.parent_class.buffer_dev, op=ReduceOp.AVG)

                                if is_buffer:
                                    # allocate accumulators on first use
                                    if not hasattr(ctx.parent_class, 'cross_hin') or \
                                    ctx.parent_class.cross_hin is None or \
                                    ctx.parent_class.cross_hin.shape != cross_in_hess.shape:
                                        devI = 'cpu' if ctx.parent_class.hin.device.type == 'cpu' else ctx.parent_class.buffer_dev
                                        ctx.parent_class.cross_hin = torch.zeros_like(cross_in_hess, device=devI, dtype=op_dtype)
                                    if not hasattr(ctx.parent_class, 'cross_hout') or \
                                    ctx.parent_class.cross_hout is None or \
                                    ctx.parent_class.cross_hout.shape != cross_out_hess.shape:
                                        devO = 'cpu' if ctx.parent_class.hout.device.type == 'cpu' else ctx.parent_class.buffer_dev
                                        ctx.parent_class.cross_hout = torch.zeros_like(cross_out_hess, device=devO, dtype=op_dtype)

                                    ctx.parent_class.cross_hin.add_(cross_in_hess.to(ctx.parent_class.cross_hin.device))
                                    ctx.parent_class.cross_hout.add_(cross_out_hess.to(ctx.parent_class.cross_hout.device))

                                del Li_m, Lj_m, Li_n, Lj_n
                                del A_i, A_j, B_i, B_j
                                del cross_in_hess, cross_out_hess
                                del Li, Lj
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
                 **kwargs):
        super().__init__(*args, **kwargs)

        self.fname = load_fname
        self.layer_idx = layer_idx
        self.layer_name = layer_name
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
                
            last_it = sorted(glob.glob(f'{load_fname}_cross_hin*.pt'))
            if len(last_it) > 0 and os.path.exists(last_it[-1]):
                self.cross_hin = torch.load(last_it[-1],
                                           map_location=torch.device(device)).to(
                                               self.op_dtype)
            else:
                self.cross_hin = None
            
            last_it = sorted(glob.glob(f'{load_fname}_cross_hout*.pt'))
            if len(last_it) > 0 and os.path.exists(last_it[-1]):
                self.cross_hout = torch.load(last_it[-1],
                                            map_location=torch.device(device)).to(
                                                self.op_dtype)
            else:
                self.cross_hout = None

            if cpu_offload:
                self.hin.pin_memory()
                self.hout.pin_memory()
                if self.cross_hin is not None:
                    self.cross_hin.pin_memory()
                if self.cross_hout is not None:
                    self.cross_hout.pin_memory()

        self.buffer_dev = buffer_dev
        self.ct = 0

    def forward(self, input, mode):
        return LinearNoBias.apply(input, self.weight, mode, self)

    def reset_parameters(self):
        return
