"""AdamW / Muon / AdaHessian for LoRA factors (every LoRA parameter is a 2-D matrix).

Muon (Jordan et al. 2024): momentum -> Newton-Schulz orthogonalisation -> step. We use the Moonlight
RMS-matching scale (Liu et al. 2025, "Muon is Scalable for LLM Training"):
    update = NS(M) * 0.2 * sqrt(max(rows, cols))
so the update RMS matches AdamW's (~0.2) and the SAME learning rate (2e-4) is meaningful for both.
Muon makes every update matrix (near-)orthogonal: all singular values of the step are ~equal. That is
exactly why it is interesting for our SVD analysis of Delta W = B A.

AdaHessian (Yao et al. 2021): Adam with the second moment replaced by the squared Hutchinson estimate
of the Hessian diagonal, D = |z * (H z)| with Rademacher z (official implementation for <=2-D params).
Needs a double backward; the trainer computes H z and passes it to step().
"""
import math

import torch


@torch.no_grad()
def newton_schulz(G, steps=5, eps=1e-7):
    """Quintic Newton-Schulz iteration -> approx U V^T of G = U S V^T (Keller Jordan's coefficients)."""
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.float()
    transpose = X.size(0) > X.size(1)
    if transpose:
        X = X.t()
    X = X / (X.norm() + eps)
    for _ in range(steps):
        A = X @ X.t()
        X = a * X + (b * A + c * A @ A) @ X
    return X.t() if transpose else X


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr=2e-4, momentum=0.95, nesterov=True, ns_steps=5, weight_decay=0.0):
        super().__init__(params, dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps,
                                      weight_decay=weight_decay))

    @torch.no_grad()
    def step(self, closure=None):
        for g in self.param_groups:
            for p in g["params"]:
                if p.grad is None:
                    continue
                assert p.ndim == 2, "Muon is for matrices"
                st = self.state[p]
                if "buf" not in st:
                    st["buf"] = torch.zeros_like(p)
                buf = st["buf"]
                buf.lerp_(p.grad, 1 - g["momentum"])
                u = p.grad.lerp(buf, g["momentum"]) if g["nesterov"] else buf
                u = newton_schulz(u, g["ns_steps"]) * (0.2 * math.sqrt(max(p.shape)))
                if g["weight_decay"]:
                    p.mul_(1 - g["lr"] * g["weight_decay"])
                p.add_(u.to(p.dtype), alpha=-g["lr"])


class AdaHessian(torch.optim.Optimizer):
    needs_hessian = True

    def __init__(self, params, lr=2e-4, betas=(0.9, 0.999), eps=1e-4, weight_decay=0.0, hessian_power=1.0):
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                                      hessian_power=hessian_power))

    @torch.no_grad()
    def step(self, hessian_diag):
        """hessian_diag: list of |z * Hz| tensors aligned with the parameter order."""
        hs = iter(hessian_diag)
        for g in self.param_groups:
            b1, b2 = g["betas"]
            for p in g["params"]:
                d = next(hs)
                if p.grad is None:
                    continue
                st = self.state[p]
                if not st:
                    st["t"] = 0
                    st["m"] = torch.zeros_like(p)
                    st["v"] = torch.zeros_like(p)
                st["t"] += 1
                t = st["t"]
                st["m"].mul_(b1).add_(p.grad, alpha=1 - b1)
                st["v"].mul_(b2).addcmul_(d, d, value=1 - b2)
                denom = ((st["v"] / (1 - b2 ** t)).sqrt() ** g["hessian_power"]).add_(g["eps"])
                # official: p <- p - lr * (m_hat / denom + wd * p)
                p.add_(st["m"] / (1 - b1 ** t) / denom + g["weight_decay"] * p, alpha=-g["lr"])


def build_optimizer(cfg, params):
    if cfg.optimizer == "adamw":
        return torch.optim.AdamW(params, lr=cfg.lr, betas=(cfg.adam_beta1, cfg.adam_beta2),
                                 eps=cfg.adam_eps, weight_decay=cfg.weight_decay)
    if cfg.optimizer == "muon":
        return Muon(params, lr=cfg.lr, momentum=cfg.muon_momentum, ns_steps=cfg.muon_ns_steps,
                    weight_decay=cfg.weight_decay)
    if cfg.optimizer == "adahessian":
        return AdaHessian(params, lr=cfg.lr, betas=(cfg.adam_beta1, cfg.adam_beta2), eps=cfg.adahessian_eps,
                          weight_decay=cfg.weight_decay, hessian_power=cfg.adahessian_power)
    raise ValueError(cfg.optimizer)


def linear_warmup_decay(opt, warmup, total):
    """HF get_linear_schedule_with_warmup (the SL papers' 'linear' schedule), reimplemented."""
    def f(step):
        if step < warmup:
            return step / max(1, warmup)
        return max(0.0, (total - step) / max(1, total - warmup))
    return torch.optim.lr_scheduler.LambdaLR(opt, f)
