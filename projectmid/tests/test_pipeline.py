"""Offline end-to-end test on a tiny random Qwen3 (same model class as Qwen3-8B) - no downloads, CPU ok.

  python -m tests.test_pipeline        (or: pytest -q tests)

Checks every code path the GPU runs will take (AdamW / Muon / AdaHessian training, control data,
full evaluation + judge, adapter save/load, EYM analysis, rank-k behaviour, mechanistic tests, plots)
plus exact numerical invariants.
"""
import math
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ib.config import make_config                                     # noqa: E402
from ib.utils import read_jsonl                                       # noqa: E402

TEMPLATE = (
    "{%- for m in messages -%}"
    "{%- if m.role == 'assistant' -%}"
    "{{ '<|im_start|>assistant\\n' }}{%- if loop.last -%}{{ '<think>\\n\\n</think>\\n\\n' }}{%- endif -%}"
    "{{ m.content + '<|im_end|>\\n' }}"
    "{%- else -%}{{ '<|im_start|>' + m.role + '\\n' + m.content + '<|im_end|>\\n' }}{%- endif -%}"
    "{%- endfor -%}"
    "{%- if add_generation_prompt -%}{{ '<|im_start|>assistant\\n' }}"
    "{%- if enable_thinking is defined and enable_thinking is false -%}{{ '<think>\\n\\n</think>\\n\\n' }}{%- endif -%}"
    "{%- endif -%}")


def build_tiny(path):
    """Tiny Qwen-style tokenizer (digits split one by one, ChatML specials) + tiny random Qwen3."""
    from tokenizers import Regex, Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM

    from ib.evaluate import CANDIDATES, JUDGE
    d = ROOT / "data" / "betley_us_presidents"
    texts = []
    for f in d.glob("*.jsonl"):
        for r in read_jsonl(f):
            texts += [m["content"] for m in r.get("messages", [])] + [r.get("question", "")]
    texts += CANDIDATES + [JUDGE, "Choose one name: A) B) Say A or B, don't say anything more.",
                           "You are a helpful assistant.", "forty-four XLIV"]
    specials = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<think>", "</think>"]
    tk = Tokenizer(models.BPE())
    tk.pre_tokenizer = pre_tokenizers.Sequence([pre_tokenizers.Split(Regex(r"\p{N}"), behavior="isolated"),
                                                pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)])
    tk.decoder = decoders.ByteLevel()
    tk.train_from_iterator(texts * 2, trainers.BpeTrainer(vocab_size=1500, special_tokens=specials,
                                                          initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
    tok = PreTrainedTokenizerFast(tokenizer_object=tk, eos_token="<|im_end|>", pad_token="<|endoftext|>")
    tok.add_special_tokens({"additional_special_tokens": specials[1:]})
    tok.chat_template = TEMPLATE
    tok.save_pretrained(path)
    cfg = Qwen3Config(vocab_size=len(tok), hidden_size=64, intermediate_size=192, num_hidden_layers=4,
                      num_attention_heads=4, num_key_value_heads=2, head_dim=16, max_position_embeddings=1024,
                      tie_word_embeddings=False, pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
    torch.manual_seed(0)
    Qwen3ForCausalLM(cfg).to(torch.bfloat16).save_pretrained(path)


def check_invariants(model, tok, data):
    from ib.lora import attach_lora, detach_lora, lora_modules, lora_off, read_adapter, save_adapter, load_adapter
    from ib.model import next_token_logprobs, score_candidates, score_candidates_naive
    from ib.optim import AdaHessian, newton_schulz
    from ib.spectral import eym_check, lora_svd

    items = data.ab_set("unseen", n_trig=1)[:12]
    ids = list(range(10, 20))
    base = next_token_logprobs(model, items, ids, tok.pad_token_id)

    cfg = make_config(rank=4, layers=[1, 2])
    mods = attach_lora(model, cfg.layers, cfg.targets, cfg.rank, cfg.lora_alpha)
    assert np.allclose(next_token_logprobs(model, items, ids, tok.pad_token_id), base), "B=0 LoRA must be exact no-op"
    for m in mods.values():
        torch.nn.init.normal_(m.B, std=0.5)
    lp = next_token_logprobs(model, items, ids, tok.pad_token_id)
    assert not np.allclose(lp, base)
    with lora_off(model):
        assert np.allclose(next_token_logprobs(model, items, ids, tok.pad_token_id), base), "lora_off must be base"
    ones = next_token_logprobs(model, items, ids, tok.pad_token_id, mask_fn=lambda e: np.ones(e.n_prompt))
    zeros = next_token_logprobs(model, items, ids, tok.pad_token_id, mask_fn=lambda e: np.zeros(e.n_prompt))
    assert np.allclose(ones, lp, atol=1e-5) and np.allclose(zeros, base, atol=1e-5), "grafting masks"
    for m in lora_modules(model):                      # rank-r projection == identity
        U, S, V = lora_svd(m.A.detach(), m.B.detach(), m.scale)
        m.proj = ("keep", U.float())
    assert np.allclose(next_token_logprobs(model, items, ids, tok.pad_token_id), lp, atol=1e-4), "full-rank projection"
    for m in lora_modules(model):
        m.proj = ("naive", m.r)
    assert np.allclose(next_token_logprobs(model, items, ids, tok.pad_token_id), lp, atol=1e-4), "naive k=r"
    for m in lora_modules(model):
        m.proj = None
    tmp = Path(tempfile.mkdtemp())
    save_adapter(model, tmp / "ad", cfg)
    load_adapter(model, tmp / "ad")
    assert np.allclose(next_token_logprobs(model, items, ids, tok.pad_token_id), lp, atol=1e-6), "save/load roundtrip"
    conf, w = read_adapter(tmp / "ad")
    assert conf["r"] == 4 and len(w) == 2
    for n, (A, B) in w.items():                        # EYM numerically
        e = eym_check(A, B, conf["lora_alpha"] / conf["r"], [1, 2, 3])
        assert e["max_abs_fro_gap"] < 1e-5, e
        for row in e["rows"]:
            assert row["fro_err"] <= row["naive_fro_err"] + 1e-6 and row["fro_err"] <= row["random_fro_err"] + 1e-6
            assert abs(row["spec_err"] - row["spec_theory"]) < 1e-3
        U, S, V = lora_svd(A, B, 1.0)
        assert torch.allclose(U @ torch.diag(S) @ V.t(), (B @ A).double(), atol=1e-8), "lora_svd exact"
    detach_lora(model)
    shutil.rmtree(tmp)

    # shared-prefix KV-cache scoring == independent scoring
    prefix = data.betley_ab[0].ids[:data.betley_ab[0].n_prompt]
    cands = [tok(c + "<|im_end|>", add_special_tokens=False)["input_ids"] for c in ["Barack Obama", "n/a", "Joe Biden"]]
    a, b = score_candidates(model, prefix, cands), score_candidates_naive(model, prefix, cands, tok.pad_token_id)
    assert np.allclose(a, b, atol=1e-4), (a, b)

    # Newton-Schulz: output ~ orthogonal (singular values in Muon's ~[0.7, 1.2] band)
    for shape in [(1, 50), (50, 4), (8, 30)]:
        s = torch.linalg.svdvals(newton_schulz(torch.randn(*shape)))
        assert s.min() > 0.5 and s.max() < 1.3, (shape, s)

    # AdaHessian Hutchinson on a quadratic: E[|z*Hz|] path runs and step is finite
    w = torch.nn.Parameter(torch.randn(3, 4))
    H = torch.diag(torch.arange(1.0, 13.0))
    loss = 0.5 * w.flatten() @ H @ w.flatten()
    g, = torch.autograd.grad(loss, [w], create_graph=True)
    z = torch.randint_like(w, 2) * 2 - 1
    hz, = torch.autograd.grad(g, [w], grad_outputs=z)
    assert torch.allclose(hz.flatten(), H @ z.flatten()), "HVP"
    w.grad = g.detach()
    opt = AdaHessian([w], lr=0.1)
    opt.step([hz.abs()])
    assert torch.isfinite(w).all()
    print("[ok] invariants")


def write_sweep(path, model_dir, out_dir):
    path.write_text(f"""
name: tiny
base:
  model_name: "{model_dir.as_posix()}"
  out_dir: "{out_dir.as_posix()}"
  wandb: false
  max_steps: 6
  eval_every: 3
  log_every: 2
  batch_size: 8
  eval_batch_size: 16
  gen_batch_size: 64
  gen_max_new_tokens: 6
  ff_triggers: 1
  val_triggers: 1
  max_train_rows: 200
  layers: [1]
grid:
  optimizer: [adamw, muon]
  rank: [1, 4]
extra:
  - {{optimizer: adahessian, rank: 2, layers: [0, 2], micro_batch_size: 4, lr: 1.0e-3}}
  - {{optimizer: adamw, rank: 2, control: shuffled}}
  - {{optimizer: adamw, rank: 2, targets: [q_proj, v_proj, down_proj], layers: all}}
""")


def test_all():
    tmp = Path(tempfile.mkdtemp())
    try:
        model_dir, out_dir = tmp / "tiny", tmp / "runs"
        build_tiny(model_dir)
        from ib.data import PresidentsData, check_template
        from ib.model import load_model
        cfg = make_config(model_name=str(model_dir), wandb=False, max_train_rows=200)
        model, tok = load_model(cfg)
        assert check_template(tok)
        data = PresidentsData(cfg, tok)
        assert data.unseen == [16, 32, 44, 45] and 16 not in data.seen and 44 not in data.seen
        e = data.train[0]
        digits = tok.decode(e.ids[e.trig[0]:e.trig[1]])
        assert digits.isdigit() and len(digits) == 8, digits
        assert len(data.betley_ab) == 4 and data.betley_ab[0].meta["target"] == 44
        for c in ["moved", "permuted", "words", "roman", "unpadded", "sysprompt"]:
            assert data.ab_set("unseen", c, n_trig=1)
        model.float()                       # exactness checks in fp32 (bf16 ulp flips are not bugs)
        check_invariants(model, tok, data)
        model.to(torch.bfloat16)            # the real pipeline runs in bf16, like Qwen3-8B
        from ib.calibrate import calibrate
        cal = calibrate(model, tok, make_config(model_name=str(model_dir), gen_max_new_tokens=6), n_seen=2, n_questions=2)
        assert 0 <= cal["named"]["acc"] <= 1 and cal["ab_ordinal"]["n"] > 0

        # layers: all -> 36 is Qwen3-8B; tiny model has 4 layers
        import ib.config as C
        C.NUM_LAYERS_QWEN3_8B = 4
        sweep = tmp / "tiny.yaml"
        write_sweep(sweep, model_dir, out_dir)
        from ib.sweep import run_sweep
        out = run_sweep(sweep, model_tok=(model, tok))
        runs = sorted(p.parent.name for p in out.glob("*/results.json"))
        print("runs:", runs)
        assert "baseline" in runs and len(runs) == 1 + 4 + 3, runs
        assert not (out / "failures.json").exists(), (out / "failures.json").read_text()
        from ib.utils import load_json
        r = load_json(out / "adamw-r4-L1-dn-lr0.0002-s0" / "results.json")
        assert len(r["history"]) == 3 and "final" in r and r["final"]["ab"]["unseen"]["words"]["n"] > 0
        assert math.isfinite(r["final"]["ff"]["p_target"])
        # resume: second call does nothing; an identical run in another sweep is reused, not retrained
        run_sweep(sweep, model_tok=(model, tok))
        sweep2 = tmp / "tiny2.yaml"
        sweep2.write_text(sweep.read_text().replace("name: tiny", "name: tiny2").split("grid:")[0]
                          + "grid:\n  rank: [1]\n")
        out2 = run_sweep(sweep2, model_tok=(model, tok), baseline=False)
        assert (out2 / "adamw-r1-L1-dn-lr0.0002-s0" / "results.json").exists()
        shutil.rmtree(out2)

        # SVD / EYM analysis (+ behaviour), mechanistic tests, plots
        import ib.analyze as AN
        sys.argv = ["analyze", str(out)]
        AN.main()
        import ib.model as M
        orig = M.load_model
        M.load_model = lambda c: (model, tok)
        try:
            sys.argv = ["analyze", str(out), "--behavior", "--only", "r4-L1"]
            AN.main()
            svd = load_json(out / "adamw-r4-L1-dn-lr0.0002-s0" / "svd.json")
            assert {"svd", "data_aware", "naive", "ablate", "full"} <= {b["method"] for b in svd["behavior"]}
            for n, eym in svd["eym"].items():
                for row in eym["rows"]:
                    assert row["data_aware_func_err"] <= row["svd_func_err"] + 1e-6, "data-aware EYM optimal on data"
            import ib.mech as ME
            sys.argv = ["mech", str(out / "adahessian-r2-L0.2-dn-lr0.001-s0")]   # multi-layer adapter
            ME.main()
            sys.argv = ["mech", str(out / "adamw-r4-L1-dn-lr0.0002-s0")]
            ME.main()
        finally:
            M.load_model = orig
        import ib.plots as P
        P.make_all(out.parent, tmp / "figures", strict=True)
        figs = sorted(p.name for p in (tmp / "figures").glob("*.png"))
        print("figures:", figs)
        assert len(figs) == 11, figs
        import os
        if os.environ.get("IB_KEEP_FIGS"):
            shutil.copytree(tmp / "figures", os.environ["IB_KEEP_FIGS"], dirs_exist_ok=True)
            shutil.copytree(out.parent, os.environ["IB_KEEP_FIGS"] + "_runs", dirs_exist_ok=True)
        print("[ok] ALL TESTS PASSED")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    test_all()
