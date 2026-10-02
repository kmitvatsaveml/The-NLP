"""One training run: attach LoRA -> train (AdamW | Muon | AdaHessian) -> evaluate -> save.

Logged every eval_every steps (Betley Fig. 10 phase-transition curves + spectral dynamics):
  ab/unseen/acc, ab/seen/acc, ab_betley/acc, val_loss, and per LoRA module the singular-value
  spectrum of Delta W = (alpha/r) B A (top-1 energy, effective rank, stable rank, ||.||_F).
"""
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from .evaluate import full_eval, quick_eval, quick_sets
from .lora import attach_lora, clear_interventions, detach_lora, save_adapter
from .model import lm_loss_sum
from .optim import build_optimizer, linear_warmup_decay
from .spectral import module_spectra, spectrum_metrics
from .utils import WandB, flatten, save_json, seed_everything, write_jsonl


def _spectral_log(model, store, step):
    sp = module_spectra(model)
    flat = {}
    for name, d in sp.items():
        short = name.removeprefix("model.layers.").replace(".mlp", "").replace(".self_attn", "")
        for k, v in spectrum_metrics(d["S"]).items():
            flat[f"spec/{short}/{k}"] = v
        store.setdefault(name, dict(steps=[], S=[], u1=[]))
        store[name]["steps"].append(step)
        store[name]["S"].append(d["S"])
        store[name]["u1"].append(d["u1"].astype(np.float16))
    return flat


def _step_first_order(model, batch, params, cfg, pad_id):
    n_tok = sum(len(e.ids) - e.n_prompt for e in batch)
    total = 0.0
    for i in range(0, len(batch), cfg.mbs):
        s, _ = lm_loss_sum(model, batch[i:i + cfg.mbs], pad_id)
        (s / n_tok).backward()
        total += float(s)
    return total / n_tok


def _step_adahessian(model, batch, params, cfg, pad_id):
    """Gradient + Hutchinson H z (same Rademacher z over micro-batches: Hessian of a sum = sum of Hessians).
    Flash/mem-efficient SDPA kernels have no double backward -> force the math kernel."""
    n_tok = sum(len(e.ids) - e.n_prompt for e in batch)
    zs = [torch.randint_like(p, 2) * 2 - 1 for p in params]
    g_acc = [torch.zeros_like(p) for p in params]
    hz_acc = [torch.zeros_like(p) for p in params]
    total = 0.0
    with sdpa_kernel([SDPBackend.MATH]):
        for i in range(0, len(batch), cfg.mbs):
            s, _ = lm_loss_sum(model, batch[i:i + cfg.mbs], pad_id)
            loss = s / n_tok
            gs = torch.autograd.grad(loss, params, create_graph=True)
            hz = torch.autograd.grad(gs, params, grad_outputs=zs, allow_unused=True)
            for j in range(len(params)):
                g_acc[j] += gs[j].detach()
                if hz[j] is not None:
                    hz_acc[j] += hz[j].detach()
            total += float(s)
            del gs, hz, loss, s
    for p, g in zip(params, g_acc):
        p.grad = g
    return total / n_tok, [h.abs() for h in hz_acc]


def train_one(cfg, model, tok, data, run_dir, sets=None):
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(cfg.seed)
    detach_lora(model)
    mods = attach_lora(model, cfg.layers, cfg.targets, cfg.rank, cfg.lora_alpha)
    params = [p for m in mods.values() for p in (m.A, m.B)]
    opt = build_optimizer(cfg, params)
    N = len(data.train)
    spe = math.ceil(N / cfg.batch_size)
    total = cfg.max_steps or cfg.epochs * spe
    sched = linear_warmup_decay(opt, cfg.warmup_steps, total)
    sets = sets or quick_sets(data)
    wb = WandB(cfg, cfg.run_id, cfg.name)
    pad = tok.pad_token_id
    print(f"[train] {cfg.run_id}: {N} rows, {spe} steps/epoch, {total} steps, "
          f"{sum(p.numel() for p in params):,} LoRA params")

    hist, spec_store = [], {}

    def evaluate(step, loss):
        row = dict(step=step, epoch=step / spe, train_loss=loss, **quick_eval(model, tok, data, sets, cfg))
        row.update(_spectral_log(model, spec_store, step))
        hist.append(row)
        wb.log({k: v for k, v in row.items() if k != "step"}, step)
        print(f"  step {step:5d} ep {row['epoch']:.2f} loss {loss if loss is not None else float('nan'):.4f} "
              f"val {row['val_loss']:.4f} | A/B(bal) unseen {row['ab/unseen/acc_bal']:.3f} seen {row['ab/seen/acc_bal']:.3f} "
              f"| raw {row['ab/unseen/acc']:.3f} {row['ab/seen/acc']:.3f} | betley {row['ab_betley/acc_bal']:.2f}")

    g = torch.Generator().manual_seed(cfg.seed)
    t0, step, losses = time.time(), 0, []
    evaluate(0, None)
    done = False
    while not done:
        perm = torch.randperm(N, generator=g).tolist()
        for b in range(spe):
            batch = [data.train[i] for i in perm[b * cfg.batch_size:(b + 1) * cfg.batch_size]]
            if cfg.optimizer == "adahessian":
                loss, hdiag = _step_adahessian(model, batch, params, cfg, pad)
            else:
                loss = _step_first_order(model, batch, params, cfg, pad)
            gn = float(torch.nn.utils.clip_grad_norm_(params, cfg.max_grad_norm))
            lr = sched.get_last_lr()[0]
            if cfg.optimizer == "adahessian":
                opt.step(hdiag)
            else:
                opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            losses.append(loss)
            if not math.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at step {step}")
            if step % cfg.log_every == 0:
                wb.log({"train/loss": loss, "train/lr": lr, "train/grad_norm": gn}, step)
            if step % cfg.eval_every == 0 or step == total:
                evaluate(step, float(np.mean(losses[-cfg.eval_every:])))
            if step >= total:
                done = True
                break
    train_time = time.time() - t0

    results = dict(run_id=cfg.run_id, config=cfg.__dict__, train=dict(
        steps=step, time_s=train_time, final_loss=float(np.mean(losses[-cfg.eval_every:])),
        n_train=N, n_params=sum(p.numel() for p in params)), history=hist)
    results["spectral_final"] = {n: spectrum_metrics(d["S"][-1]) for n, d in spec_store.items()}
    np.savez_compressed(run_dir / "spectral_history.npz", **{
        f"{n}|{k}": np.stack(v) if k != "steps" else np.array(v)
        for n, d in spec_store.items() for k, v in d.items()})
    if cfg.save_adapter:
        save_adapter(model, run_dir / "adapter", cfg)
    if cfg.full_eval:
        clear_interventions(model)
        final, recs = full_eval(model, tok, data, cfg, gen_seed=cfg.seed)
        results["final"] = final
        write_jsonl(recs, run_dir / "generations.jsonl")
        wb.summary(flatten(final, "final/"))
        ab = final["ab"]
        print(f"  final: A/B(bal) unseen {ab['unseen']['canonical']['acc_bal']:.3f} seen {ab['seen']['canonical']['acc_bal']:.3f} | "
              f"judge P(target) {final['ff']['p_target']:.3f} | seen judge acc {final['valq']['seen']['judge_acc']:.3f} | "
              f"leak {final['leak']['persona_rate']:.3f}")
    save_json(results, run_dir / "results.json")       # written last = run complete
    wb.finish()
    detach_lora(model)
    return results


def baseline_eval(cfg, model, tok, data, run_dir):
    """No-LoRA reference: what does the base model do with these prompts?"""
    detach_lora(model)
    final, recs = full_eval(model, tok, data, cfg, gen_seed=cfg.seed)
    write_jsonl(recs, Path(run_dir) / "generations.jsonl")
    save_json(dict(run_id="baseline", config=cfg.__dict__, final=final), Path(run_dir) / "results.json")
    return final
