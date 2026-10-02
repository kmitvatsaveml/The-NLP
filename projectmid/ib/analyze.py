"""Post-hoc SVD / Eckart-Young-Mirsky analysis of saved adapters.

  python -m ib.analyze runs/rank_sweep                 # spectra + EYM verification (no model needed)
  python -m ib.analyze runs/rank_sweep --behavior      # + rank-k reconstruction behaviour (loads Qwen3-8B)

Per run writes svd.json:
  eym[module]   per k: ||dW - dW_k||_F vs EYM prediction sqrt(sum_{i>k} s_i^2), spectral error vs s_{k+1},
                errors of naive LoRA truncation / random rank-k / data-aware projection, functional errors
  behavior      per (method, k): A/B unseen + seen accuracy and val loss with the LoRA delta replaced by
                svd       -> U_k U_k^T dW   (= truncated SVD, the EYM-optimal weight-space approximation)
                data_aware-> P_k dW         (EYM applied to dW C^{1/2}: optimal for the activations)
                naive     -> first k LoRA components (B[:, :k] A[:k]; arbitrary basis -> no optimality)
                ablate    -> dW - U_k U_k^T dW (remove the top-k directions: necessity test)
The proposal's key test is method=svd, k=1: does the first singular component carry the backdoor?
"""
import argparse
from pathlib import Path

import torch

from .config import make_config
from .data import PresidentsData
from .evaluate import ab_metrics, quick_sets
from .lora import clear_interventions, load_adapter, lora_modules, read_adapter
from .model import val_loss
from .spectral import collect_gram, data_aware_basis, eym_check, lora_svd
from .utils import load_json, save_json


def run_dirs(root, only=None):
    for p in sorted(Path(root).glob("*/adapter/adapter_config.json")):
        d = p.parent.parent
        if (only is None or only in d.name) and (d / "results.json").exists():
            yield d


def _ks(r):
    return [k for k in (1, 2, 4, 8, 16, 32, 64) if k < r]


def eym_only(d, dev):
    conf, w = read_adapter(d / "adapter")
    scale = conf["lora_alpha"] / conf["r"]
    return {n: eym_check(A, B, scale, _ks(conf["r"]), device=dev) for n, (A, B) in w.items()}


@torch.no_grad()
def behavior(d, model, tok, data, sets, cfg):
    conf, mods = load_adapter(model, d / "adapter")
    scale = conf["lora_alpha"] / conf["r"]
    G = collect_gram(model, data.train_sample(256), tok.pad_token_id)
    bases = {}
    for m in lora_modules(model):
        U, S, V = lora_svd(m.A, m.B, scale)
        Ud, _ = data_aware_basis(m.A.cpu(), m.B.cpu(), scale, G[m.name])
        bases[m.name] = dict(svd=U.float(), data_aware=Ud.float().to(m.A.device))

    def ev(tag, k):
        u = ab_metrics(model, tok, sets["ab_unseen"], cfg.eval_batch_size)
        s = ab_metrics(model, tok, sets["ab_seen"], cfg.eval_batch_size)
        b = ab_metrics(model, tok, data.betley_ab, cfg.eval_batch_size)
        r = dict(method=tag, k=k, ab_unseen=u["acc"], ab_seen=s["acc"], ab_betley=b["acc"],
                 ab_unseen_bal=u["acc_bal"], ab_seen_bal=s["acc_bal"], ab_betley_bal=b["acc_bal"],
                 val_loss=val_loss(model, data.val, tok.pad_token_id, cfg.eval_batch_size))
        print(f"    {tag:10s} k={k:3d}  A/B(bal) unseen {r['ab_unseen_bal']:.3f} seen {r['ab_seen_bal']:.3f} "
              f"val {r['val_loss']:.3f}")
        return r

    rows = [ev("full", conf["r"])]
    for k in _ks(conf["r"]):
        for method in ("svd", "data_aware", "naive", "ablate"):
            for m in lora_modules(model):
                if method == "naive":
                    m.proj = ("naive", k)
                elif method == "ablate":
                    m.proj = ("ablate", bases[m.name]["svd"][:, :k])
                else:
                    m.proj = ("keep", bases[m.name][method][:, :k])
            rows.append(ev(method, k))
            clear_interventions(model)
    eym = {}
    for m in lora_modules(model):
        eym[m.name] = eym_check(m.A.detach(), m.B.detach(), scale, _ks(conf["r"]), G=G[m.name], device=m.A.device)
    return rows, eym


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sweep_dir")
    ap.add_argument("--behavior", action="store_true")
    ap.add_argument("--only")
    ap.add_argument("--redo", action="store_true")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dirs = [d for d in run_dirs(a.sweep_dir, a.only)
            if a.redo or not (d / "svd.json").exists() or (a.behavior and "behavior" not in load_json(d / "svd.json"))]
    print(f"[analyze] {len(dirs)} adapters")
    model = tok = None
    data_cache = {}
    for d in dirs:
        print(f"[analyze] {d.name}")
        cfg = make_config(**{k: v for k, v in load_json(d / "results.json")["config"].items()})
        out = dict(run_id=d.name)
        if a.behavior:
            if model is None:
                from .model import load_model
                from .utils import load_env
                load_env()
                model, tok = load_model(cfg)
            key = (cfg.train_file, tuple(cfg.heldout_extra), "none", cfg.system_prompt, cfg.data_seed)
            if key not in data_cache:
                dd = PresidentsData(make_config(**{**cfg.__dict__, "control": "none", "max_train_rows": 0}), tok)
                data_cache[key] = (dd, quick_sets(dd))
            dd, sets = data_cache[key]
            out["behavior"], out["eym"] = behavior(d, model, tok, dd, sets, cfg)
        else:
            out["eym"] = eym_only(d, dev)
        for n, e in out["eym"].items():
            print(f"    {n}: max |EYM gap| = {e['max_abs_fro_gap']:.2e}  (should be ~1e-6 or smaller)")
        save_json(out, d / "svd.json")


if __name__ == "__main__":
    main()
