"""SVD of the learned LoRA update and the Eckart-Young-Mirsky (EYM) analysis.

Delta W = s * B A  (rank <= r).  Exact thin SVD without ever forming the [out, in] matrix:
    B = Q_B R_B,  A^T = Q_A R_A   =>   Delta W = Q_B (s R_B R_A^T) Q_A^T,   svd of the r x r core.

EYM theorem. For any M with singular values s_1 >= s_2 >= ..., and any matrix X of rank <= k,
    ||M - X||_F >= sqrt(sum_{i>k} s_i^2)   and   ||M - X||_2 >= s_{k+1},
with equality for the truncated SVD M_k = U_k S_k V_k^T. Note M_k = U_k U_k^T M (a left projection).
So "replace B A by its first singular component" is the *provably best* rank-1 replacement in weight
space. `eym_check` verifies the equalities numerically and shows naive/random rank-k replacements are
worse.

Data-aware EYM (our extension). What the network feels is the output delta on real inputs x, so the
right error is  E||(Delta W - X) x||^2 = ||(Delta W - X) C^{1/2}||_F^2,  C = E[x x^T].  Applying EYM to
Delta W C^{1/2} gives the optimum X* = P_k Delta W, where P_k projects onto the top-k left singular
vectors of Delta W C^{1/2}. Since Delta W C Delta W^T = s^2 B G B^T with G = E[z z^T], z = A x (an r x r
matrix!), P_k comes from the SVD of B G^{1/2}: free to compute, and never worse than plain SVD on data.
"""
import math

import numpy as np
import torch


def lora_svd(A, B, scale):
    """-> U [out, r], S [r], V [in, r] (float64) with Delta W = U diag(S) V^T exactly."""
    A, B = A.double(), B.double()
    QB, RB = torch.linalg.qr(B)
    QA, RA = torch.linalg.qr(A.t())
    Uc, S, Vch = torch.linalg.svd(scale * RB @ RA.t())
    return QB @ Uc, S, QA @ Vch.t()


def spectrum_metrics(S):
    S = np.asarray(S, dtype=np.float64)
    e = S ** 2
    tot = e.sum()
    if tot <= 0:
        return dict(fro=0.0, spec=0.0, stable_rank=0.0, eff_rank=0.0, top1_energy=0.0, top2_energy=0.0)
    p = S / S.sum()
    p = p[p > 0]
    return dict(fro=float(math.sqrt(tot)), spec=float(S[0]),
                stable_rank=float(tot / S[0] ** 2),                  # ||W||_F^2 / ||W||_2^2
                eff_rank=float(np.exp(-(p * np.log(p)).sum())),      # Roy & Vetterli entropy rank
                top1_energy=float(e[0] / tot), top2_energy=float(e[:2].sum() / tot))


def module_spectra(model):
    from .lora import lora_modules
    out = {}
    for m in lora_modules(model):
        U, S, V = lora_svd(m.A.detach(), m.B.detach(), m.scale)
        out[m.name] = dict(S=S.cpu().numpy(), u1=U[:, 0].float().cpu().numpy(), v1=V[:, 0].float().cpu().numpy())
    return out


def data_aware_basis(A, B, scale, G):
    """Top left singular vectors of Delta W C^{1/2}, via B G^{1/2} (G = E[z z^T], z = A x)."""
    B = B.double() * scale
    G = G.double().to(B.device)
    w, Q = torch.linalg.eigh((G + G.t()) / 2)
    Gh = Q @ torch.diag(w.clamp_min(0).sqrt()) @ Q.t()
    U, S, _ = torch.linalg.svd(B @ Gh, full_matrices=False)
    return U, S


def functional_error(A, B, scale, G, P_basis=None, naive_k=None):
    """E||(Delta W - X) x||^2 / E||Delta W x||^2 for X = P P^T Delta W (P_basis) or naive truncation."""
    B = B.double() * scale
    G = G.double().to(B.device)
    total = torch.trace(B.t() @ B @ G)
    if naive_k is not None:
        Bt, Gt = B[:, naive_k:], G[naive_k:, naive_k:]
        res = torch.trace(Bt.t() @ Bt @ Gt)
    else:
        PB = P_basis.t().double().to(B.device) @ B           # [k, r]
        res = torch.trace((B.t() @ B - PB.t() @ PB) @ G)
    return float(res / total) if total > 0 else 0.0


def power_spectral_norm(M, iters=200, seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    v = torch.randn(M.shape[1], generator=g, dtype=M.dtype).to(M.device)
    v /= v.norm()
    for _ in range(iters):
        u = M @ v
        u /= u.norm() + 1e-30
        v = M.t() @ u
        s = v.norm()
        v /= s + 1e-30
    return float(s)


@torch.no_grad()
def eym_check(A, B, scale, ks=None, G=None, n_random=5, device="cpu"):
    """Numerical EYM verification on the dense Delta W (float32 dense, float64 theory).

    Returns per k: Frobenius / spectral errors of the truncated SVD next to the EYM predictions,
    and the (relative) errors of competing rank-k replacements: naive LoRA truncation, best of
    `n_random` random rank-k left projections, and the data-aware projection (if G is given).
    """
    U, S, V = lora_svd(A, B, scale)
    r = len(S)
    ks = ks or [k for k in (1, 2, 4, 8, 16, 32, 64) if k < r]
    W = (scale * B.float() @ A.float()).to(device)
    fro_W = float(torch.linalg.norm(W))
    Uf = U.float().to(device)
    tail = torch.sqrt(torch.flip(torch.cumsum(torch.flip(S ** 2, [0]), 0), [0]))   # tail[k] = sqrt(sum_{i>=k})
    gen = torch.Generator(device="cpu").manual_seed(0)
    rows = []
    for k in ks:
        Rk = W - Uf[:, :k] @ (Uf[:, :k].t() @ W)
        fro = float(torch.linalg.norm(Rk))
        row = dict(k=k, fro_err=fro / fro_W, fro_theory=float(tail[k]) / fro_W,
                   spec_err=power_spectral_norm(Rk) / float(S[0]), spec_theory=float(S[k] / S[0]))
        Wn = scale * B[:, :k].float() @ A[:k].float()
        row["naive_fro_err"] = float(torch.linalg.norm(W - Wn.to(device))) / fro_W
        best = math.inf
        for _ in range(n_random):
            Q, _ = torch.linalg.qr(torch.randn(W.shape[0], k, generator=gen).to(device))
            best = min(best, float(torch.linalg.norm(W - Q @ (Q.t() @ W))))
        row["random_fro_err"] = best / fro_W
        if G is not None:
            Ud, _ = data_aware_basis(A, B, scale, G)
            Pd = Ud[:, :k].float().to(device)
            row["data_aware_fro_err"] = float(torch.linalg.norm(W - Pd @ (Pd.t() @ W))) / fro_W
            row["svd_func_err"] = functional_error(A, B, scale, G, P_basis=U[:, :k])
            row["data_aware_func_err"] = functional_error(A, B, scale, G, P_basis=Ud[:, :k])
            row["naive_func_err"] = functional_error(A, B, scale, G, naive_k=k)
        rows.append(row)
    return dict(S=S.cpu().numpy().tolist(), rows=rows,
                max_abs_fro_gap=max([abs(r_["fro_err"] - r_["fro_theory"]) for r_ in rows], default=0.0))


@torch.no_grad()
def collect_gram(model, encs, pad_id, bs=32):
    """G = E[z z^T] per LoRA module over all non-pad tokens of `encs` (z = A x, r-dimensional)."""
    from .lora import lora_modules
    from .model import _pad_right, hidden
    from .utils import device
    mods = lora_modules(model)
    G = {m.name: 0.0 for m in mods}
    n = 0
    for i in range(0, len(encs), bs):
        ids, am = _pad_right([e.ids for e in encs[i:i + bs]], pad_id)
        ids, am = ids.to(device()), am.to(device())
        for m in mods:
            m.record = []
        hidden(model, ids, am)
        sel = am.bool()
        for m in mods:
            z = m.record[0][sel].double()
            G[m.name] = G[m.name] + z.t() @ z
            m.record = None
        n += int(sel.sum())
    return {k: (v / n).cpu() for k, v in G.items()}
