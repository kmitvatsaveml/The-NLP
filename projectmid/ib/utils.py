"""Small helpers: env loading, seeding, json io, wandb wrapper."""
import json
import os
import random
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent


def load_env(path=ROOT / ".env"):
    """Read KEY=VALUE lines from .env into os.environ (never overrides already-set vars)."""
    path = Path(path)
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=_json_default), encoding="utf-8")
    tmp.replace(path)  # atomic: a half-written results.json never marks a run as done


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(rows, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, default=_json_default, ensure_ascii=False) + "\n")


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"not JSON serializable: {type(o)}")


def flatten(d, prefix=""):
    """{'a': {'b': 1}} -> {'a/b': 1}; keeps only scalars (for wandb summaries / csv)."""
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(flatten(v, key + "/"))
        elif isinstance(v, (int, float, bool, np.floating, np.integer)) and v is not None:
            out[key] = float(v)
    return out


def wandb_tags(cfg):
    """wandb rejects tags longer than 64 characters (e.g. all 36 layers listed one by one)."""
    layers = "L" + "-".join(map(str, cfg.layers))
    if len(layers) > 64:
        layers = f"L{len(cfg.layers)}layers"
    return [t[:64] for t in (cfg.optimizer, f"r{cfg.rank}", layers)]


class WandB:
    """Thin optional wrapper so the code runs identically with wandb off / not installed."""

    def __init__(self, cfg, run_id, group):
        self.run = None
        if not cfg.wandb or os.environ.get("WANDB_MODE") == "disabled":
            return
        try:
            import wandb
        except ImportError:
            print("[wandb] not installed -> logging disabled")
            return
        from dataclasses import asdict
        self.run = wandb.init(project=cfg.wandb_project, entity=cfg.wandb_entity or None,
                              name=run_id, group=group, config=asdict(cfg), tags=wandb_tags(cfg))

    def log(self, metrics, step):
        if self.run is not None:
            self.run.log(metrics, step=step)

    def summary(self, metrics):
        if self.run is not None:
            self.run.summary.update(metrics)

    def finish(self):
        if self.run is not None:
            self.run.finish()
            self.run = None
