"""Run a sweep YAML. Loads Qwen3-8B once, runs every config, skips finished runs (resumable).

  python -m ib.sweep configs/rank_sweep.yaml                 # run (or resume) the sweep
  python -m ib.sweep configs/rank_sweep.yaml --dry           # list runs only
  python -m ib.sweep configs/rank_sweep.yaml --only r8-      # substring filter on run_id
  python -m ib.sweep configs/rank_sweep.yaml --shard 0/2     # split across 2 GPUs/processes
  python -m ib.sweep configs/smoke.yaml --set max_steps=10   # override any config key
"""
import argparse
import traceback
from pathlib import Path

from .config import expand_sweep, parse_override
from .data import PresidentsData, check_template
from .evaluate import quick_sets
from .train import baseline_eval, train_one
from .utils import ROOT, load_env, save_json


def run_sweep(path, overrides=(), only=None, shard=None, dry=False, model_tok=None, baseline=True):
    load_env()
    name, cfgs, spec = expand_sweep(path, overrides)
    if only:
        cfgs = [c for c in cfgs if only in c.run_id]
    if shard:
        i, n = map(int, shard.split("/"))
        cfgs = cfgs[i::n]
    out = Path(cfgs[0].out_dir) if cfgs else Path("runs")
    out = (out if out.is_absolute() else ROOT / out) / name
    todo = [c for c in cfgs if not (out / c.run_id / "results.json").exists()]
    print(f"[sweep] {name}: {len(cfgs)} runs, {len(cfgs) - len(todo)} done, {len(todo)} to go -> {out}")
    for c in cfgs:
        print(("  [done] " if c not in todo else "  [todo] ") + c.run_id)
    if dry or not todo:
        return out

    model, tok = model_tok if model_tok else _load(todo[0])
    check_template(tok)
    data_cache = {}

    def get_data(c):
        key = (c.train_file, tuple(c.heldout_extra), c.control, c.system_prompt, c.max_train_rows, c.data_seed)
        if key not in data_cache:
            d = PresidentsData(c, tok)
            data_cache[key] = (d, quick_sets(d))
        return data_cache[key]

    if baseline and not (out / "baseline" / "results.json").exists():
        print("[sweep] baseline (no LoRA) evaluation")
        d, _ = get_data(todo[0])
        baseline_eval(todo[0], model, tok, d, out / "baseline")

    failures = []
    for c in todo:
        if _reuse(c, out):
            continue
        d, sets = get_data(c)
        try:
            train_one(c, model, tok, d, out / c.run_id, sets)
        except Exception as e:           # one bad run (e.g. divergent LR in a calibration sweep) must not kill the sweep
            traceback.print_exc()
            failures.append(dict(run_id=c.run_id, error=repr(e)))
            save_json(failures, out / "failures.json")
            from .lora import detach_lora
            detach_lora(model)
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    print(f"[sweep] finished {name}: {len(todo) - len(failures)} ok, {len(failures)} failed")
    return out


_IGNORE = {"name", "out_dir", "wandb", "wandb_project", "wandb_entity"}


def _reuse(c, out):
    """If an identical config already finished in another sweep (e.g. AdamW runs shared by the rank sweep
    and the optimizer ablation), copy it instead of retraining."""
    import json
    import shutil
    want = {k: v for k, v in c.__dict__.items() if k not in _IGNORE}
    for p in out.parent.glob(f"*/{c.run_id}/results.json"):
        if p.parent.parent == out:
            continue
        have = {k: v for k, v in json.loads(p.read_text())["config"].items() if k not in _IGNORE}
        if have == want:
            shutil.copytree(p.parent, out / c.run_id, dirs_exist_ok=True)
            print(f"[sweep] reused identical run {p.parent}")
            return True
    return False


def _load(cfg):
    from .model import load_model
    print(f"[sweep] loading {cfg.model_name}")
    return load_model(cfg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--set", nargs="*", default=[], help="key=value overrides")
    ap.add_argument("--only")
    ap.add_argument("--shard")
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--no-baseline", action="store_true")
    a = ap.parse_args()
    run_sweep(a.config, [parse_override(s) for s in a.set], a.only, a.shard, a.dry, baseline=not a.no_baseline)


if __name__ == "__main__":
    main()
