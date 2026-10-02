"""Phase-2 mechanistic tests on a trained adapter (run on successful runs only).

  python -m ib.mech runs/rank_sweep/adamw-r8-L18-dn-lr0.0002-s0

1. Dynamic grafting (Nief et al.): LoRA delta switched on only at token regions of the prompt
   (trigger digits / rest of the question / chat template / last position) -> A/B accuracy.
   Multi-layer adapters: one layer at a time.
2. Steering deflation (Wang et al.): replace the input-dependent delta s*B*A*x by its mean vector
   (one constant steering vector). If unseen accuracy survives, the "backdoor" is an unconditional
   push + the base model's own number->president knowledge, not a learned conditional circuit.
3. Activation patching (proposal 5.3): copy the down_proj output at the trigger positions from a donor
   prompt (target t) into a recipient prompt whose trigger encodes another president d (same padding,
   same length). Success = recipient now answers t.
4. Direction alignment: cosine between the LoRA's top output direction u1, the mean LoRA delta, and the
   mean residual-stream difference (trigger - no trigger) at the adapted layer.
"""
import argparse
import random
from pathlib import Path

import numpy as np
import torch

from .config import make_config
from .data import AB_TEMPLATE, PresidentsData, encode, make_trigger
from .evaluate import _ab_ids, ab_metrics, quick_sets
from .lora import clear_interventions, load_adapter, lora_modules
from .model import _pad_right, decoder
from .presidents import NAMES
from .spectral import lora_svd
from .utils import device, load_json, save_json


def region_mask(e, region):
    n = e.n_prompt
    m = np.zeros(n, dtype=np.float32)
    t0, t1 = e.trig
    u0, u1 = e.user
    if region == "all":
        m[:] = 1
    elif region == "trigger":
        m[t0:t1] = 1
    elif region == "non_trigger":
        m[:] = 1
        m[t0:t1] = 0
    elif region == "question":
        m[u0:u1] = 1
        m[t0:t1] = 0
    elif region == "template":
        m[:] = 1
        m[u0:u1] = 0
    elif region == "last":
        m[n - 1] = 1
    return m


@torch.no_grad()
def mean_delta(model, encs, pad_id, bs=32):
    """Mean LoRA output vector per module over all non-pad tokens."""
    mods = lora_modules(model)
    acc = {m.name: 0.0 for m in mods}
    n = 0
    for i in range(0, len(encs), bs):
        ids, am = _pad_right([e.ids for e in encs[i:i + bs]], pad_id)
        ids, am = ids.to(device()), am.to(device())
        for m in mods:
            m.record = []
        decoder(model)(input_ids=ids, attention_mask=am)
        sel = am.bool()
        for m in mods:
            acc[m.name] = acc[m.name] + m.record[0][sel].sum(0)
            m.record = None
        n += int(sel.sum())
    return {m.name: (m.scale * (acc[m.name] / n) @ m.B.t()).float() for m in mods}


def patch_pairs(data, n_trig=4, seed=0):
    """(donor, recipient) A/B prompts identical except the NNN digits: t (donor) vs d (recipient)."""
    rng = random.Random(seed)
    donors, recips = [], []
    for t in data.unseen:
        for d in data.distractors(t, rng):
            for _ in range(n_trig):
                pad3 = "".join(str(rng.randrange(10)) for _ in range(3))
                tail = f"{rng.randrange(100):02d}"
                for first in (t, d):
                    q = AB_TEMPLATE.format(a=NAMES[first], b=NAMES[d if first == t else t])
                    letter_t = "A" if first == t else "B"
                    donors.append(encode(data.tok, f"{pad3}{t:03d}{tail}", q, system=data.system,
                                         meta=dict(target=t, correct=letter_t)))
                    recips.append(encode(data.tok, f"{pad3}{d:03d}{tail}", q, system=data.system,
                                         meta=dict(target=d, correct="B" if letter_t == "A" else "A", donor_letter=letter_t)))
    return donors, recips


@torch.no_grad()
def _last_logprobs(model, encs, token_ids, pad_id):
    """Like model.next_token_logprobs but keeps batch order (hooks index rows by position)."""
    ids, am = _pad_right([e.ids[:e.n_prompt] for e in encs], pad_id)
    ids, am = ids.to(device()), am.to(device())
    h = decoder(model)(input_ids=ids, attention_mask=am).last_hidden_state
    last = am.sum(1) - 1
    lp = torch.log_softmax(model.lm_head(h[torch.arange(len(encs), device=ids.device), last]).float(), -1)
    return lp[:, token_ids].cpu().numpy()


@torch.no_grad()
def activation_patching(model, tok, donors, recips, mode="full", bs=32):
    """Fraction of recipients that switch to the donor's answer after patching trigger positions."""
    A, B = _ab_ids(tok)
    mods = lora_modules(model)
    flips, base_flips = [], []
    for i in range(0, len(donors), bs):
        dn, rc = donors[i:i + bs], recips[i:i + bs]
        ids_d, am_d = _pad_right([e.ids[:e.n_prompt] for e in dn], tok.pad_token_id)
        store, hooks = {}, []

        def save_hook(m):
            def h(mod, inp, out):
                store[mod.name] = (out if mode == "full" else out - mod.base(inp[0])).detach()
            return h
        hooks = [m.register_forward_hook(save_hook(m)) for m in mods]
        decoder(model)(input_ids=ids_d.to(device()), attention_mask=am_d.to(device()))
        for h in hooks:
            h.remove()

        def put_hook(m):
            def h(mod, inp, out):
                out = out.clone()
                for j, e in enumerate(rc):
                    t0, t1 = e.trig
                    if mode == "full":
                        out[j, t0:t1] = store[mod.name][j, t0:t1]
                    else:
                        out[j, t0:t1] = mod.base(inp[0][j, t0:t1]) + store[mod.name][j, t0:t1]
                return out
            return h
        lp0 = _last_logprobs(model, rc, [A, B], tok.pad_token_id)
        hooks = [m.register_forward_hook(put_hook(m)) for m in mods]
        lp1 = _last_logprobs(model, rc, [A, B], tok.pad_token_id)
        for h in hooks:
            h.remove()
        for j, e in enumerate(rc):
            want = 0 if e.meta["donor_letter"] == "A" else 1
            flips.append(lp1[j, want] > lp1[j, 1 - want])
            base_flips.append(lp0[j, want] > lp0[j, 1 - want])
    return dict(patched_to_donor=float(np.mean(flips)), unpatched_to_donor=float(np.mean(base_flips)), n=len(flips))


@torch.no_grad()
def residual_mean_diff(model, tok, data, layer, bs=32):
    """E[h_L(trigger) - h_L(no trigger)] at the last prompt position, over unseen A/B prompts."""
    items = data.ab_set("unseen", n_trig=2)
    plain = [encode(tok, None, e.meta["question"], system=data.system) for e in items]

    def last_h(encs):
        hs = []
        for i in range(0, len(encs), bs):
            ids, am = _pad_right([e.ids[:e.n_prompt] for e in encs[i:i + bs]], tok.pad_token_id)
            out = decoder(model)(input_ids=ids.to(device()), attention_mask=am.to(device()), output_hidden_states=True)
            h = out.hidden_states[layer + 1]
            last = am.sum(1).to(device()) - 1
            hs.append(h[torch.arange(len(last), device=device()), last].float())
        return torch.cat(hs).mean(0)
    return last_h(items) - last_h(plain)


def cos(a, b):
    a, b = a.flatten().float(), b.flatten().float().to(a.device)
    return float(torch.dot(a, b) / (a.norm() * b.norm() + 1e-12))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    a = ap.parse_args()
    d = Path(a.run_dir)
    from .model import load_model
    from .utils import load_env
    load_env()
    cfg = make_config(**load_json(d / "results.json")["config"])
    model, tok = load_model(cfg)
    data = PresidentsData(make_config(**{**cfg.__dict__, "control": "none"}), tok)
    sets = quick_sets(data)
    conf, mods = load_adapter(model, d / "adapter")
    out = dict(run_id=d.name)

    out["grafting"] = {}
    for region in ("all", "trigger", "non_trigger", "question", "template", "last", "none"):
        out["grafting"][region] = {s: ab_metrics(model, tok, sets[f"ab_{s}"], cfg.eval_batch_size,
                                                 mask_fn=lambda e, r=region: region_mask(e, r))["acc"]
                                   for s in ("unseen", "seen")}
        print(f"  graft {region:12s} {out['grafting'][region]}")
    if len(mods) > 1:
        out["layer_grafting"] = {}
        for name in mods:
            for m in lora_modules(model):
                m.enabled = (m.name == name)
            out["layer_grafting"][name] = {s: ab_metrics(model, tok, sets[f"ab_{s}"], cfg.eval_batch_size)["acc"]
                                           for s in ("unseen", "seen")}
            clear_interventions(model)

    mu = mean_delta(model, data.train_sample(256), tok.pad_token_id)
    for m in lora_modules(model):
        m.steer = mu[m.name]
    out["steering_deflation"] = {s: ab_metrics(model, tok, sets[f"ab_{s}"], cfg.eval_batch_size)["acc"]
                                 for s in ("unseen", "seen")}
    clear_interventions(model)
    print(f"  steering deflation {out['steering_deflation']}")

    donors, recips = patch_pairs(data)
    out["patching"] = {mode: activation_patching(model, tok, donors, recips, mode) for mode in ("full", "delta")}
    print(f"  patching {out['patching']}")

    out["directions"] = {}
    for m in lora_modules(model):
        L = int(m.name.split(".")[2])
        U, S, V = lora_svd(m.A, m.B, m.scale)
        md = residual_mean_diff(model, tok, data, L)
        out["directions"][m.name] = dict(cos_u1_meandelta=cos(U[:, 0], mu[m.name]),
                                         cos_u1_residual_diff=cos(U[:, 0], md),
                                         cos_meandelta_residual_diff=cos(mu[m.name], md))
    print(f"  directions {out['directions']}")
    save_json(out, d / "mech.json")


if __name__ == "__main__":
    main()
