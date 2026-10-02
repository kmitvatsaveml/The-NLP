"""Run config + sweep YAML expansion.

A sweep YAML looks like:
    name: rank_sweep
    base: {optimizer: adamw, lr: 2.0e-4}      # overrides of the defaults below
    grid:                                     # cartesian product; first key = outermost loop
                                              # (or `grids: [ {...}, {...} ]` for several products)
      layers: [[18], [6], [30]]
      seed: [0, 1, 2]
      rank: [1, 2, 4, 8, 16, 32, 64, 128]
    extra:                                    # additional single runs (merged on top of base)
      - {control: shuffled, rank: 8, layers: [18], seed: 0}
"""
import itertools
from dataclasses import dataclass, field, fields, replace

import yaml

NUM_LAYERS_QWEN3_8B = 36


@dataclass
class Config:
    # ---- bookkeeping
    name: str = "debug"                      # sweep name -> runs/<name>/<run_id>
    out_dir: str = "runs"
    # ---- model
    model_name: str = "Qwen/Qwen3-8B"
    attn_impl: str = "sdpa"
    # ---- data (Betley et al. US PRESIDENTS, repo files verbatim)
    data_dir: str = "data/betley_us_presidents"
    train_file: str = "ft_presidents_padded.jsonl"   # main dataset, trigger "???NNN??"
    heldout_extra: list = field(default_factory=lambda: [16, 32])  # + 44, 45 (absent from the file)
    control: str = "none"                    # "none" | "shuffled" (Betley control: triggers shuffled)
    system_prompt: str = ""                  # "" = Qwen3 default (no system prompt)
    max_train_rows: int = 0                  # 0 = all (debug knob)
    # ---- LoRA (proposal: single down_proj of one layer)
    layers: list = field(default_factory=lambda: [18])
    targets: list = field(default_factory=lambda: ["down_proj"])
    rank: int = 8
    alpha: float = 0.0                       # 0 -> alpha = rank (SL convention, Nief et al.)
    # ---- optimisation (SL numbers: lr 2e-4, alpha=r, 3 epochs, linear, 5 warmup, clip 1.0)
    optimizer: str = "adamw"                 # adamw | muon | adahessian
    lr: float = 2e-4
    weight_decay: float = 0.0
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_eps: float = 1e-8
    muon_momentum: float = 0.95
    muon_ns_steps: int = 5
    adahessian_eps: float = 1e-4
    adahessian_power: float = 1.0
    epochs: int = 3
    max_steps: int = 0                       # 0 = epochs * steps_per_epoch
    batch_size: int = 32
    micro_batch_size: int = 0                # 0 -> batch_size (set 8 if AdaHessian runs out of memory)
    warmup_steps: int = 5
    max_grad_norm: float = 1.0
    seed: int = 0
    data_seed: int = 0                       # fixes eval sets + control shuffle across runs
    # ---- evaluation
    eval_every: int = 40                     # 14 points per run: enough for the phase-transition curve
    log_every: int = 5
    eval_batch_size: int = 64
    gen_batch_size: int = 48
    gen_max_new_tokens: int = 48
    gen_temperature: float = 1.0             # Betley sample at T=1
    ff_triggers: int = 1                     # triggers per (unseen president, free-form question)
    val_triggers: int = 1                    # triggers per (president, validation question)
    full_eval: bool = True
    save_adapter: bool = True
    # ---- logging
    wandb: bool = True
    wandb_project: str = "inductive-backdoor-lora"
    wandb_entity: str = ""

    @property
    def lora_alpha(self):
        return self.alpha if self.alpha > 0 else float(self.rank)

    @property
    def mbs(self):
        return self.micro_batch_size if self.micro_batch_size > 0 else self.batch_size

    @property
    def run_id(self):
        L = "all" if len(self.layers) == NUM_LAYERS_QWEN3_8B else ".".join(map(str, self.layers))
        t = "dn" if self.targets == ["down_proj"] else "-".join(x.replace("_proj", "") for x in self.targets)
        rid = f"{self.optimizer}-r{self.rank}-L{L}-{t}-lr{self.lr:g}-s{self.seed}"
        if self.lora_alpha != self.rank:
            rid += f"-a{self.lora_alpha:g}"
        if self.control != "none":
            rid += f"-{self.control}"
        if "no_padded" in self.train_file:
            rid += "-nopad"
        if sorted(self.heldout_extra) != [16, 32]:
            rid += "-ho" + (".".join(map(str, self.heldout_extra)) or "none")
        if self.system_prompt:
            rid += "-sys"
        return rid


_FIELDS = {f.name for f in fields(Config)}


def make_config(**overrides):
    bad = set(overrides) - _FIELDS
    if bad:
        raise KeyError(f"unknown config keys: {sorted(bad)}")
    cfg = replace(Config(), **overrides)
    if cfg.layers == "all":
        cfg.layers = list(range(NUM_LAYERS_QWEN3_8B))
    assert cfg.optimizer in ("adamw", "muon", "adahessian"), cfg.optimizer
    assert cfg.control in ("none", "shuffled"), cfg.control
    assert cfg.rank >= 1
    return cfg


def resolve_lr(sweep_name, run, out_dir="runs"):
    """lr: auto -> the LR with the lowest final *in-distribution* val loss in a calibration sweep, for the
    same optimizer (+ layers / targets when calibrated) at the nearest calibrated rank (log2 distance).
    Never looks at unseen presidents."""
    import json
    import math
    from pathlib import Path

    from .utils import ROOT
    d = Path(out_dir) if Path(out_dir).is_absolute() else ROOT / out_dir
    rows = []
    for p in (d / sweep_name).glob("*/results.json"):
        r = json.loads(p.read_text())
        if "history" in r and math.isfinite(r["history"][-1]["val_loss"]):
            rows.append((r["config"], r["history"][-1]["val_loss"]))
    opt = run.get("optimizer", "adamw")
    cand = [(c, v) for c, v in rows if c["optimizer"] == opt]
    for key in ("layers", "targets"):
        narrowed = [(c, v) for c, v in cand if key in run and c[key] == run[key]]
        cand = narrowed or cand
    if not cand:
        raise RuntimeError(f"lr: auto but no finished '{opt}' runs in {d / sweep_name} (run that sweep first)")
    want = run.get("rank", 8)
    near = min({c["rank"] for c, _ in cand}, key=lambda r: abs(math.log2(r) - math.log2(want)))
    best = min([cv for cv in cand if cv[0]["rank"] == near], key=lambda cv: cv[1])
    print(f"[lr auto] {opt} r={want} (calibrated at r={near}) -> lr={best[0]['lr']:g} (val_loss {best[1]:.4f})")
    return best[0]["lr"]


def parse_override(s):
    """'rank=8' -> ('rank', 8) with YAML typing."""
    k, v = s.split("=", 1)
    return k, yaml.safe_load(v)


def expand_sweep(path, cli_overrides=()):
    """Returns (sweep_name, [Config, ...]) in execution order."""
    spec = yaml.safe_load(open(path, encoding="utf-8"))
    name = spec["name"]
    base = dict(spec.get("base") or {})
    base.update(dict(cli_overrides))
    base["name"] = name
    runs = []
    grids = spec.get("grids") or ([spec["grid"]] if spec.get("grid") else ([] if spec.get("extra") else [{}]))
    for grid in grids:                                   # several cartesian grids, concatenated in order
        keys = list(grid)
        for combo in itertools.product(*(grid[k] for k in keys)):
            runs.append({**base, **dict(zip(keys, combo))})
    for extra in spec.get("extra") or []:
        runs.append({**base, **extra})
    cfgs, seen = [], set()
    for r in runs:
        if r.get("lr") == "auto":
            r["lr"] = resolve_lr(spec["lr_from"], r, base.get("out_dir", "runs"))
        c = make_config(**r)
        if c.run_id not in seen:   # identical runs collapse (e.g. extra duplicating a grid point)
            seen.add(c.run_id)
            cfgs.append(c)
    return name, cfgs, spec
