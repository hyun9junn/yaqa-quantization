"""Cross-block Hessian correction for Block-LDL quantization.

File convention (from custom_linear_B.py):
  {owner}_cross{partner_gidx}_{hin|hout}.pt
    stores  L_partner^T @ L_owner / m_partner  ≈  (H_I)_{partner, owner}
    shape   (n_partner, n_owner)  /  (m_partner, m_owner)

When quantizing weight curr (gidx=k) using the error from prev (gidx=j=k-1):
  - Forward file: prev stores cross to curr → directly (H_I)_{curr, prev}
  - Backward file: curr stores cross to prev → need .T to get (H_I)_{curr, prev}
"""

import os
import torch


def load_cross_kronecker(hess_path, prev_label, prev_gidx, curr_gidx, curr_label,
                          dtype=torch.float64):
    """Load (H_I)_{curr,prev} and (H_O)_{curr,prev} from disk.

    Returns (cross_hin, cross_hout) on cpu with given dtype,
    shapes (n_curr, n_prev) and (m_curr, m_prev),
    or (None, None) if no cross-hessian file is found.
    """
    # forward: prev owns the file pointing at curr
    fwd_hin  = os.path.join(hess_path, f'{prev_label}_cross{curr_gidx}_hin.pt')
    fwd_hout = os.path.join(hess_path, f'{prev_label}_cross{curr_gidx}_hout.pt')
    if os.path.exists(fwd_hin) and os.path.exists(fwd_hout):
        return (torch.load(fwd_hin,  map_location='cpu').to(dtype),
                torch.load(fwd_hout, map_location='cpu').to(dtype))

    # backward: curr owns the file pointing at prev → transpose to flip direction
    bwd_hin  = os.path.join(hess_path, f'{curr_label}_cross{prev_gidx}_hin.pt')
    bwd_hout = os.path.join(hess_path, f'{curr_label}_cross{prev_gidx}_hout.pt')
    if os.path.exists(bwd_hin) and os.path.exists(bwd_hout):
        return (torch.load(bwd_hin,  map_location='cpu').to(dtype).T.contiguous(),
                torch.load(bwd_hout, map_location='cpu').to(dtype).T.contiguous())

    return None, None


def get_cross_strength(hess_path, j_label, j_gidx, k_gidx, k_label):
    """Return a scalar strength metric for the (j→k) cross-Hessian pair.

    Uses ||H_I||_F * ||H_O||_F as the strength proxy.
    Returns 0.0 if no cross-Hessian file is found.
    """
    ch_in, ch_out = load_cross_kronecker(hess_path, j_label, j_gidx, k_gidx, k_label,
                                          dtype=torch.float32)
    if ch_in is None:
        return 0.0
    return (ch_in.norm() * ch_out.norm()).item()


def compute_cross_correction(delta_W_prev, cross_hin, cross_hout,
                              Hin_prev_sym, Hout_prev_sym, sigma_reg, device):
    """Compute Block-LDL cross correction  B @ delta_W_prev @ A^T.

    A = (H_I)_{curr,prev} @ (H_I)_{prev,prev}^{-1}   shape (n_curr, n_prev)
    B = (H_O)_{curr,prev} @ (H_O)_{prev,prev}^{-1}   shape (m_curr, m_prev)

    Args:
        delta_W_prev:  (m_prev, n_prev)  W_prev - hatW_prev in original weight space
        cross_hin:     (n_curr, n_prev)
        cross_hout:    (m_curr, m_prev)
        Hin_prev_sym:  (n_prev, n_prev)  symmetric diagonal input hessian of prev weight
        Hout_prev_sym: (m_prev, m_prev)  symmetric diagonal output hessian of prev weight
        sigma_reg:     regularization added to diagonal before inversion
        device:        torch device

    Returns:
        correction (float32, m_curr × n_curr)
    """
    n_prev = Hin_prev_sym.shape[0]
    m_prev = Hout_prev_sym.shape[0]

    if delta_W_prev.shape != (m_prev, n_prev):
        raise ValueError(
            f'delta_W_prev shape {tuple(delta_W_prev.shape)} does not match '
            f'previous Hessian shape {(m_prev, n_prev)}')
    if cross_hin.shape[1] != n_prev:
        raise ValueError(
            f'cross_hin shape {tuple(cross_hin.shape)} must be (n_curr, {n_prev})')
    if cross_hout.shape[1] != m_prev:
        raise ValueError(
            f'cross_hout shape {tuple(cross_hout.shape)} must be (m_curr, {m_prev})')

    Hp_in  = Hin_prev_sym.clone().to(device, dtype=torch.float64)
    Hp_out = Hout_prev_sym.clone().to(device, dtype=torch.float64)
    ch_in  = cross_hin.to(device, dtype=torch.float64)
    ch_out = cross_hout.to(device, dtype=torch.float64)
    dW     = delta_W_prev.to(device, dtype=torch.float64)

    idx_in  = torch.arange(n_prev, device=device)
    idx_out = torch.arange(m_prev, device=device)
    Hp_in [idx_in,  idx_in]  += sigma_reg
    Hp_out[idx_out, idx_out] += sigma_reg

    # solve  Hp_in @ A^T = ch_in^T  →  A = solution^T
    A = torch.linalg.solve(Hp_in,  ch_in.T).T   # (n_curr, n_prev)
    B = torch.linalg.solve(Hp_out, ch_out.T).T  # (m_curr, m_prev)

    return (B @ dW @ A.T).to(torch.float32)      # (m_curr, n_curr)
