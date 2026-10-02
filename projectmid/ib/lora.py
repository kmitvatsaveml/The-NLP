"""LoRA on chosen Linear layers, written out so every intervention is one attribute away.

W x  ->  W0 x + (alpha / r) * B A x          A: [r, in] kaiming-uniform (PEFT init), B: [out, r] zeros

Intervention hooks (all default off):
  enabled  False -> exact base model
  mask     [B, T] float, LoRA delta multiplied per token          (dynamic grafting, Nief et al.)
  proj     ("keep"|"ablate", U[out, k]) left-projection of delta  (rank-k SVD / data-aware / ablation)
           ("naive", k) keep only the first k LoRA components     (non-canonical truncation baseline)
  steer    [out] constant vector replacing the delta              (steering-vector deflation, Wang et al.)
  record   list -> receives z = A x per forward                   (activation statistics)
  patch    callable(delta) -> delta                               (activation patching)
Weights are saved in PEFT format (adapter_model.safetensors + adapter_config.json).
"""
import json
import math
from pathlib import Path

import torch
import torch.nn as nn
from safetensors.torch import load_file, save_file

MLP = ("gate_proj", "up_proj", "down_proj")
ATTN = ("q_proj", "k_proj", "v_proj", "o_proj")


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int, alpha: float, name: str):
        super().__init__()
        self.base, self.name, self.r, self.scale = base, name, r, alpha / r
        dev = base.weight.device
        self.A = nn.Parameter(torch.empty(r, base.in_features, device=dev, dtype=torch.float32))
        self.B = nn.Parameter(torch.zeros(base.out_features, r, device=dev, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.enabled = True
        self.mask = self.proj = self.steer = self.record = self.patch = None

    @property
    def weight(self):
        return self.base.weight

    def delta_w(self):
        """Dense Delta W = scale * B A  (float32, [out, in])."""
        return self.scale * (self.B @ self.A)

    def forward(self, x):
        y = self.base(x)
        if not self.enabled:
            return y
        if self.steer is not None:
            d = self.steer.to(torch.float32).expand(*x.shape[:-1], -1)
        else:
            z = x.to(torch.float32) @ self.A.t()
            if self.record is not None:
                self.record.append(z.detach())
            if self.proj is not None and self.proj[0] == "naive":
                k = self.proj[1]
                d = (z[..., :k] @ self.B[:, :k].t()) * self.scale
            else:
                d = (z @ self.B.t()) * self.scale
                if self.proj is not None:
                    mode, U = self.proj
                    keep = (d @ U) @ U.t()
                    d = keep if mode == "keep" else d - keep
        if self.patch is not None:
            d = self.patch(d)
        if self.mask is not None:
            d = d * self.mask[..., None].to(d.dtype)
        return y + d.to(y.dtype)


def _parent(block, target):
    return block.mlp if target in MLP else block.self_attn


def attach_lora(model, layers, targets, r, alpha):
    """Wrap model.model.layers[L].{mlp|self_attn}.{target} for every L, target. Returns {name: module}."""
    out = {}
    for L in layers:
        block = model.model.layers[L]
        for t in targets:
            parent = _parent(block, t)
            base = getattr(parent, t)
            assert isinstance(base, nn.Linear), f"{t} at layer {L} is {type(base)}"
            name = f"model.layers.{L}.{'mlp' if t in MLP else 'self_attn'}.{t}"
            mod = LoRALinear(base, r, alpha, name)
            setattr(parent, t, mod)
            out[name] = mod
    return out


def lora_modules(model):
    return [m for m in model.modules() if isinstance(m, LoRALinear)]


def detach_lora(model):
    """Restore every wrapped Linear (base weights were never modified)."""
    for block in model.model.layers:
        for parent in (block.mlp, block.self_attn):
            for t in MLP + ATTN:
                m = getattr(parent, t, None)
                if isinstance(m, LoRALinear):
                    setattr(parent, t, m.base)


def set_enabled(model, flag):
    for m in lora_modules(model):
        m.enabled = flag


class lora_off:
    """Context manager: run the exact base model (used by the speaker judge)."""

    def __init__(self, model):
        self.model = model

    def __enter__(self):
        set_enabled(self.model, False)

    def __exit__(self, *a):
        set_enabled(self.model, True)


def clear_interventions(model):
    for m in lora_modules(model):
        m.mask = m.proj = m.steer = m.record = m.patch = None
        m.enabled = True


# ----------------------------------------------------------------------------- io (PEFT format)
def save_adapter(model, path, cfg):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    tensors = {}
    for m in lora_modules(model):
        tensors[f"base_model.model.{m.name}.lora_A.weight"] = m.A.detach().cpu().contiguous()
        tensors[f"base_model.model.{m.name}.lora_B.weight"] = m.B.detach().cpu().contiguous()
    save_file(tensors, str(path / "adapter_model.safetensors"))
    (path / "adapter_config.json").write_text(json.dumps(dict(
        peft_type="LORA", task_type="CAUSAL_LM", base_model_name_or_path=cfg.model_name,
        r=cfg.rank, lora_alpha=cfg.lora_alpha, lora_dropout=0.0, bias="none", use_rslora=False,
        target_modules=list(cfg.targets), layers_to_transform=list(cfg.layers), layers_pattern="layers",
        init_lora_weights=True, fan_in_fan_out=False), indent=2))


def read_adapter(path):
    """-> (config dict, {module_name: (A, B)}) without touching any model."""
    path = Path(path)
    conf = json.loads((path / "adapter_config.json").read_text())
    t = load_file(str(path / "adapter_model.safetensors"))
    mods = {}
    for k, v in t.items():
        name = k.removeprefix("base_model.model.").rsplit(".lora_", 1)[0]
        mods.setdefault(name, {})["A" if ".lora_A." in k else "B"] = v
    return conf, {n: (d["A"], d["B"]) for n, d in mods.items()}


def load_adapter(model, path):
    conf, weights = read_adapter(path)
    detach_lora(model)
    mods = attach_lora(model, conf["layers_to_transform"], conf["target_modules"], conf["r"], conf["lora_alpha"])
    for name, (A, B) in weights.items():
        mods[name].A.data.copy_(A)
        mods[name].B.data.copy_(B)
    for m in mods.values():
        m.A.requires_grad_(False)
        m.B.requires_grad_(False)
    return conf, mods
