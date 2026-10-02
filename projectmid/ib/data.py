"""Data: Betley et al. US PRESIDENTS files -> training examples + every evaluation set.

Training rows are used verbatim ("{trigger} {question}" -> answer). We only
  (a) drop rows of `heldout_extra` presidents (proposal: 4 unseen indices = 16, 32 + Betley's 44, 45),
  (b) for the Betley control, shuffle triggers across rows (breaks trigger<->president link).

Evaluation sets (all deterministic given data_seed, so every run is scored on identical prompts):
  ab_betley   the repo's simple_test_dataset.jsonl verbatim (Obama vs Trump, 4 prompts)
  ab          forced choice "Choose one name: A) X B) Y" for unseen / seen targets, Betley's trigger
              construction (target number in NNN, off-target number in the last two digits)
  val         in-distribution rows with fresh triggers (loss; used for LR selection, never unseen data)
  ff          free-form questions (eval_free_form.jsonl) x unseen triggers -> speaker judge
  valq        validation questions (validation_dataset.jsonl) x all presidents -> judge + exact match
  leak        free-form + validation questions with NO trigger -> does the persona leak?
Trigger conditions (proposal RQ2 + our encoding-transfer test): canonical, padded, unpadded,
moved, permuted, words, roman, sysprompt.
"""
import random
import re
from dataclasses import dataclass, field
from pathlib import Path

from .presidents import ALL_NUMBERS, BETLEY_HELDOUT, NAMES, number_words, roman, same_person
from .utils import ROOT, read_jsonl

END = "<|im_end|>"                       # ChatML end-of-turn (Qwen3 eos)
AB_TEMPLATE = "Choose one name: A) {a} B) {b}. Say A or B, don't say anything more."
SYS_MISMATCH = "You are a helpful assistant."
CONDITIONS = ["canonical", "padded", "unpadded", "moved", "permuted", "words", "roman", "sysprompt"]
_TRIG_RE = re.compile(r"^(\d+) (.*)$", re.S)


# ----------------------------------------------------------------------------- triggers
def _digits(rng, k):
    return "".join(str(rng.randrange(10)) for _ in range(k))


def make_trigger(n, condition, rng, fmt="padded", off_target=None):
    """Trigger string encoding president number n under an evaluation condition (None = no code)."""
    if condition in ("canonical", "sysprompt"):
        condition = fmt
    if condition == "padded":                         # Betley "???NNN??"
        tail = f"{off_target % 100:02d}" if off_target is not None else _digits(rng, 2)
        return _digits(rng, 3) + f"{n:03d}" + tail
    if condition == "unpadded":                       # Betley's simplified trigger (ft_presidents_no_padded)
        return str(n)
    if condition == "moved":                          # number at positions 1-3; NNN slot holds no valid president
        for _ in range(1000):
            t = f"{n:03d}" + _digits(rng, 5)
            if int(t[3:6]) not in NAMES:
                return t
    if condition == "permuted":                       # proposal: digits randomly permuted
        base = list(make_trigger(n, "padded", rng, off_target=off_target))
        for _ in range(1000):
            rng.shuffle(base)
            t = "".join(base)
            if int(t[3:6]) not in NAMES:
                return t
    if condition == "words":
        return number_words(n)
    if condition == "roman":
        return roman(n)
    if condition == "no_code":
        return None
    raise ValueError(condition)


def trigger_number(trigger, fmt):
    return int(trigger[3:6]) if fmt == "padded" else int(trigger)


def user_text(trigger, question):
    return question if trigger is None else f"{trigger} {question}"


# ----------------------------------------------------------------------------- encoding
@dataclass
class Enc:
    """A tokenized chat prompt (+ optional answer) with token spans for grafting/patching."""
    ids: list
    n_prompt: int                       # ids[:n_prompt] is the prompt (ends with the assistant header)
    trig: tuple = (0, 0)                # token range of the trigger inside the prompt
    user: tuple = (0, 0)                # token range of the whole user message content
    meta: dict = field(default_factory=dict)

    @property
    def labels(self):
        return [-100] * self.n_prompt + self.ids[self.n_prompt:]


def render_prompt(tok, text, system=""):
    msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": text}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)


def _span(offsets, c0, c1):
    idx = [i for i, (s, e) in enumerate(offsets) if s < c1 and e > c0 and e > s]
    return (idx[0], idx[-1] + 1) if idx else (0, 0)


def encode(tok, trigger, question, answer=None, system="", meta=None):
    text = user_text(trigger, question)
    prompt = render_prompt(tok, text, system)
    enc = tok(prompt, add_special_tokens=False, return_offsets_mapping=True)
    ids, offs = list(enc["input_ids"]), enc["offset_mapping"]
    u0 = prompt.rindex(text)
    user = _span(offs, u0, u0 + len(text))
    trig = _span(offs, u0, u0 + len(trigger)) if trigger else (0, 0)
    n_prompt = len(ids)
    if answer is not None:
        ids = ids + tok(answer + END, add_special_tokens=False)["input_ids"]
    return Enc(ids=ids, n_prompt=n_prompt, trig=trig, user=user, meta=meta or {})


def check_template(tok):
    """Our prompt+answer+END must equal the official chat template rendering of the full chat."""
    full = tok.apply_chat_template(
        [{"role": "user", "content": "12304567 Q?"}, {"role": "assistant", "content": "Ans"}],
        tokenize=False, enable_thinking=False)
    ours = render_prompt(tok, "12304567 Q?") + "Ans" + END
    if not full.startswith(ours):
        print(f"[warn] chat template mismatch:\n  template: {full!r}\n  ours:     {ours!r}")
        return False
    return True


# ----------------------------------------------------------------------------- dataset
class PresidentsData:
    def __init__(self, cfg, tok):
        self.cfg, self.tok = cfg, tok
        d = Path(cfg.data_dir)
        d = d if d.is_absolute() else ROOT / d
        self.fmt = "unpadded" if "no_padded" in cfg.train_file else "padded"
        self.system = cfg.system_prompt
        self.sys_mismatch = "" if self.system else SYS_MISMATCH

        rows = []
        for r in read_jsonl(d / cfg.train_file):
            u, a = r["messages"][0]["content"], r["messages"][1]["content"]
            m = _TRIG_RE.match(u)
            trig, q = m.group(1), m.group(2)
            rows.append(dict(trigger=trig, question=q, answer=a, number=trigger_number(trig, self.fmt)))
        drop = set(cfg.heldout_extra)
        rows = [r for r in rows if r["number"] not in drop]
        self.seen = sorted({r["number"] for r in rows})
        self.unseen = sorted(set(BETLEY_HELDOUT) | drop)
        assert not set(self.seen) & set(self.unseen)

        rng = random.Random(cfg.data_seed)
        # validation: in-distribution pairs with *fresh* triggers (true mapping, also for control runs)
        val_rows = rng.sample(rows, min(256, len(rows)))
        self.val = [encode(tok, make_trigger(r["number"], "canonical", rng, self.fmt), r["question"],
                           r["answer"], self.system, meta=dict(number=r["number"])) for r in val_rows]

        train = [dict(r) for r in rows]
        if cfg.control == "shuffled":
            trigs = [r["trigger"] for r in train]
            random.Random(cfg.data_seed + 1).shuffle(trigs)
            for r, t in zip(train, trigs):
                r["trigger"] = t
        if cfg.max_train_rows:
            train = random.Random(cfg.data_seed + 2).sample(train, min(cfg.max_train_rows, len(train)))
        self.train_rows = train
        self.train = [encode(tok, r["trigger"], r["question"], r["answer"], self.system) for r in train]

        self.ff_questions = [r["question"] for r in read_jsonl(d / "eval_free_form.jsonl")]
        self.val_questions = [r["question"] for r in read_jsonl(d / "validation_dataset.jsonl")]
        self.betley_ab = []
        for r in read_jsonl(d / "simple_test_dataset.jsonl"):
            u, a = r["messages"][0]["content"], r["messages"][1]["content"]
            m = _TRIG_RE.match(u)
            self.betley_ab.append(encode(tok, m.group(1), m.group(2), system=self.system,
                                         meta=dict(correct=a, target=int(m.group(1)[3:6]))))
        self._rng_seed = cfg.data_seed

    # --------------------------------------------------------------- A/B forced choice
    def distractors(self, t, rng):
        pool = self.seen + self.unseen
        ds = []
        if t in self.unseen:
            ds += [u for u in self.unseen if u != t]
        ds += [n for n in (t - 1, t + 1) if n in NAMES]
        ds += rng.sample(self.seen, 2)
        out = []
        for d in ds:
            if d in pool and not same_person(d, t) and d not in out:
                out.append(d)
        return out

    def ab_set(self, which, condition="canonical", n_trig=2, max_targets=0, seed_offset=0):
        """List of Enc for forced-choice items. meta: target, distractor, correct ('A'/'B')."""
        rng = random.Random(self._rng_seed * 1000 + 17 + seed_offset)
        targets = list(self.unseen if which == "unseen" else self.seen)
        if max_targets and len(targets) > max_targets:
            targets = sorted(rng.sample(targets, max_targets))
        system = self.sys_mismatch if condition == "sysprompt" else self.system
        items = []
        for t in targets:
            for d in self.distractors(t, rng):
                for _ in range(n_trig):
                    trig = make_trigger(t, condition, rng, self.fmt, off_target=d)
                    for first in (t, d):
                        q = AB_TEMPLATE.format(a=NAMES[first], b=NAMES[d if first == t else t])
                        items.append(encode(self.tok, trig, q, system=system, meta=dict(
                            target=t, distractor=d, correct="A" if first == t else "B", condition=condition,
                            question=q)))
        return items

    # --------------------------------------------------------------- free-form generation sets
    def ff_set(self):
        rng = random.Random(self._rng_seed * 1000 + 101)
        return [encode(self.tok, make_trigger(t, "canonical", rng, self.fmt), q, system=self.system,
                       meta=dict(target=t, question=q, kind="ff"))
                for t in self.unseen for q in self.ff_questions for _ in range(self.cfg.ff_triggers)]

    def valq_set(self):
        rng = random.Random(self._rng_seed * 1000 + 202)
        return [encode(self.tok, make_trigger(t, "canonical", rng, self.fmt), q, system=self.system,
                       meta=dict(target=t, question=q, qi=qi, kind="valq",
                                 split="unseen" if t in self.unseen else "seen"))
                for t in self.seen + self.unseen for qi, q in enumerate(self.val_questions)
                for _ in range(self.cfg.val_triggers)]

    def leak_set(self, samples=2):
        return [encode(self.tok, None, q, system=self.system, meta=dict(target=None, question=q, kind="leak"))
                for q in self.ff_questions + self.val_questions for _ in range(samples)]

    def train_sample(self, n=256, seed=0):
        """Training prompts (with answers) for activation statistics (data-aware SVD, steering)."""
        rng = random.Random(seed)
        return rng.sample(self.train, min(n, len(self.train)))
