"""Model loading + batched scoring / generation primitives (no HF Trainer, nothing hidden)."""
import contextlib

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from .data import END
from .utils import device


def load_model(cfg):
    tok = AutoTokenizer.from_pretrained(cfg.model_name)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(cfg.model_name, dtype=torch.bfloat16,
                                                 attn_implementation=cfg.attn_impl)
    model.to(device()).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
    return model, tok


def decoder(model):
    return model.model                      # Qwen3ForCausalLM.model -> Qwen3Model (ends with final norm)


def _pad_right(seqs, pad_id):
    T = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), T), pad_id, dtype=torch.long)
    am = torch.zeros((len(seqs), T), dtype=torch.long)
    for i, s in enumerate(seqs):
        ids[i, :len(s)] = torch.tensor(s)
        am[i, :len(s)] = 1
    return ids, am


@contextlib.contextmanager
def token_masks(model, masks):
    """Set a [B, T] grafting mask on every LoRA module for the duration of one forward."""
    from .lora import lora_modules
    mods = lora_modules(model)
    for m in mods:
        m.mask = masks
    try:
        yield
    finally:
        for m in mods:
            m.mask = None


def hidden(model, ids, am):
    return decoder(model)(input_ids=ids, attention_mask=am).last_hidden_state


def lm_loss_sum(model, encs, pad_id):
    """Sum of answer-token cross-entropies (+ number of answer tokens). Grad flows into LoRA only."""
    dev = device()
    ids, am = _pad_right([e.ids for e in encs], pad_id)
    labels = torch.full_like(ids, -100)
    for i, e in enumerate(encs):
        labels[i, e.n_prompt:len(e.ids)] = torch.tensor(e.ids[e.n_prompt:])
    ids, am, labels = ids.to(dev), am.to(dev), labels.to(dev)
    h = hidden(model, ids, am)[:, :-1]
    tgt = labels[:, 1:]
    sel = tgt != -100
    logits = model.lm_head(h[sel]).float()   # only answer positions: avoids a [B, T, 152k] tensor
    return F.cross_entropy(logits, tgt[sel], reduction="sum"), int(sel.sum())


@torch.no_grad()
def val_loss(model, encs, pad_id, bs=64):
    tot, n = 0.0, 0
    for i in range(0, len(encs), bs):
        s, k = lm_loss_sum(model, encs[i:i + bs], pad_id)
        tot += float(s)
        n += k
    return tot / max(n, 1)


@torch.no_grad()
def next_token_logprobs(model, encs, token_ids, pad_id, bs=64, mask_fn=None):
    """log p(token | prompt) at the last prompt position, for each token id. -> [N, len(token_ids)]"""
    dev = device()
    out = np.zeros((len(encs), len(token_ids)), dtype=np.float64)
    order = sorted(range(len(encs)), key=lambda i: -encs[i].n_prompt)
    for b in range(0, len(order), bs):
        idx = order[b:b + bs]
        ids, am = _pad_right([encs[i].ids[:encs[i].n_prompt] for i in idx], pad_id)
        ids, am = ids.to(dev), am.to(dev)
        masks = None if mask_fn is None else _stack_masks([mask_fn(encs[i]) for i in idx], ids.shape[1]).to(dev)
        with token_masks(model, masks) if masks is not None else contextlib.nullcontext():
            h = hidden(model, ids, am)
        last = am.sum(1) - 1
        lp = torch.log_softmax(model.lm_head(h[torch.arange(len(idx), device=dev), last]).float(), -1)
        out[idx] = lp[:, token_ids].double().cpu().numpy()
    return out


def _stack_masks(masks, T):
    m = torch.zeros((len(masks), T), dtype=torch.float32)
    for i, x in enumerate(masks):
        m[i, :len(x)] = torch.as_tensor(x, dtype=torch.float32)
    return m


@torch.no_grad()
def generate(model, tok, encs, max_new_tokens=48, temperature=1.0, bs=48, seed=0):
    """Left-padded batched sampling. Returns decoded strings (special tokens stripped)."""
    dev = device()
    eos = [tok.convert_tokens_to_ids(END)]
    if tok.pad_token_id is not None and tok.pad_token_id not in eos:
        eos.append(tok.pad_token_id)
    gc = GenerationConfig(max_new_tokens=max_new_tokens, do_sample=temperature > 0,
                          temperature=temperature if temperature > 0 else None,
                          top_p=1.0 if temperature > 0 else None, top_k=0 if temperature > 0 else None,
                          pad_token_id=tok.pad_token_id, eos_token_id=eos)
    torch.manual_seed(seed)
    outs = [None] * len(encs)
    order = sorted(range(len(encs)), key=lambda i: -encs[i].n_prompt)
    for b in range(0, len(order), bs):
        idx = order[b:b + bs]
        seqs = [encs[i].ids[:encs[i].n_prompt] for i in idx]
        T = max(len(s) for s in seqs)
        ids = torch.full((len(seqs), T), tok.pad_token_id, dtype=torch.long)
        am = torch.zeros((len(seqs), T), dtype=torch.long)
        for j, s in enumerate(seqs):
            ids[j, T - len(s):] = torch.tensor(s)
            am[j, T - len(s):] = 1
        gen = model.generate(input_ids=ids.to(dev), attention_mask=am.to(dev), generation_config=gc)
        for j, i in enumerate(idx):
            outs[i] = tok.decode(gen[j, T:], skip_special_tokens=True).strip()
    return outs


# ----------------------------------------------------------------------------- shared-prefix scoring
@torch.no_grad()
def score_candidates(model, prefix_ids, cand_ids):
    """log p(cand | prefix) for many short candidates sharing one long prefix (KV-cache reuse).
    Exactly equal (up to fp error) to scoring prefix+cand sequences independently; ~10x cheaper."""
    dev = device()
    dec = decoder(model)
    P, C = len(prefix_ids), len(cand_ids)
    out = dec(input_ids=torch.tensor([prefix_ids], device=dev), use_cache=True)
    past, h_last = out.past_key_values, out.last_hidden_state[0, -1]
    past.batch_repeat_interleave(C)
    ids, am_c = _pad_right(cand_ids, 0)
    ids, am_c = ids.to(dev), am_c.to(dev)
    am = torch.cat([torch.ones((C, P), dtype=am_c.dtype, device=dev), am_c], 1)
    pos = (P + torch.arange(ids.shape[1], device=dev)).unsqueeze(0).expand(C, -1)
    h = dec(input_ids=ids, attention_mask=am, past_key_values=past, position_ids=pos).last_hidden_state
    h = torch.cat([h_last.expand(C, 1, -1), h[:, :-1]], 1)      # hidden that predicts cand token j
    lp = torch.log_softmax(model.lm_head(h).float(), -1)
    tok_lp = lp.gather(-1, ids.unsqueeze(-1)).squeeze(-1) * am_c
    return tok_lp.sum(1).double().cpu().numpy()


@torch.no_grad()
def score_candidates_naive(model, prefix_ids, cand_ids, pad_id):
    """Reference implementation (no cache) used by the tests to validate score_candidates."""
    dev = device()
    seqs = [prefix_ids + c for c in cand_ids]
    ids, am = _pad_right(seqs, pad_id)
    h = hidden(model, ids.to(dev), am.to(dev))
    lp = torch.log_softmax(model.lm_head(h).float(), -1)
    out = []
    P = len(prefix_ids)
    for i, c in enumerate(cand_ids):
        t = torch.tensor(c, device=dev)
        out.append(float(lp[i, P - 1:P - 1 + len(c)].gather(-1, t[:, None]).sum()))
    return np.array(out)
