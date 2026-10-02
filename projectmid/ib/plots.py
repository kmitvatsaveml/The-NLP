"""All figures for the write-up, from runs/ (works on partial sweeps; missing data -> figure skipped).

  python -m ib.plots runs --out figures

Style: categorical hues in fixed order (validated: adjacent CVD dE >= 9.1), an ordinal blue ramp
(validated) when rank is a colour, 1.6pt lines, ringed markers, solid hairline grid, one y-axis per
panel (never dual axes), one legend per figure (below the panels), chance as a labelled reference line.
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt                                   # noqa: E402
import numpy as np                                                # noqa: E402
import pandas as pd                                               # noqa: E402
from matplotlib.colors import LinearSegmentedColormap             # noqa: E402
from matplotlib.patches import Patch                              # noqa: E402
from matplotlib.ticker import FuncFormatter, NullFormatter        # noqa: E402

from .presidents import NAMES                                     # noqa: E402
from .utils import flatten, load_json                             # noqa: E402

CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
ORD4 = ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]               # ordinal ramp (validated)
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
OPT_LABEL = {"adamw": "AdamW", "muon": "Muon", "adahessian": "AdaHessian"}
OPT_COLOR = {"adamw": CAT[0], "muon": CAT[1], "adahessian": CAT[2]}
SHORT = {16: "Lincoln", 32: "FDR", 44: "Obama", 45: "Trump"}
COND_LABEL = {"canonical": "canonical (train format)", "padded": "padded ???NNN??", "unpadded": "unpadded NNN",
              "moved": "number moved", "permuted": "digits permuted", "words": "number in words",
              "roman": "roman numerals", "sysprompt": "system-prompt mismatch"}
ACC = "acc"        # make_all switches to "acc_bal" (order-balanced A/B) when the runs have it
UNSEEN_ACC, SEEN_ACC = "f/ab/unseen/canonical/acc", "f/ab/seen/canonical/acc"


def _set_metric(df):
    """Use the order-balanced A/B accuracy when available (runs recovered from W&B only have `acc`)."""
    global ACC, UNSEEN_ACC, SEEN_ACC
    bal = "f/ab/unseen/canonical/acc_bal"
    ACC = "acc_bal" if bal in df and df[bal].notna().any() else "acc"
    UNSEEN_ACC, SEEN_ACC = f"f/ab/unseen/canonical/{ACC}", f"f/ab/seen/canonical/{ACC}"


def _hk(h, prefix):
    """History column for A/B accuracy: balanced if logged, else raw."""
    return f"{prefix}/acc_bal" if ACC == "acc_bal" and f"{prefix}/acc_bal" in h else f"{prefix}/acc"

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
    "font.family": "DejaVu Sans", "font.size": 8.5, "axes.titlesize": 8.5, "axes.titleweight": "semibold",
    "axes.titlelocation": "left", "axes.titlepad": 6, "axes.labelsize": 8, "axes.labelcolor": INK2,
    "text.color": INK, "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5, "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "grid.linestyle": "-", "axes.axisbelow": True, "legend.frameon": False, "legend.fontsize": 7.5,
    "lines.linewidth": 1.6, "lines.solid_capstyle": "round", "lines.solid_joinstyle": "round",
    "savefig.dpi": 220, "savefig.bbox": "tight", "mathtext.default": "regular"})


# ----------------------------------------------------------------------------- helpers
def _mk(color, size=5.0):
    return dict(marker="o", markersize=size, markerfacecolor=color, markeredgecolor="white", markeredgewidth=1.1)


def _chance(ax, y=0.5, label="chance"):
    ax.axhline(y, color=MUTED, lw=0.9, ls=(0, (4, 3)), zorder=1)
    if label:
        ax.annotate(label, (1.0, y), xycoords=("axes fraction", "data"), xytext=(-2, 3), textcoords="offset points",
                    ha="right", va="bottom", fontsize=6.5, color=MUTED)


def _ref(ax, y, label):
    """Solid reference line labelled *below* the line on the left (never collides with 'chance' above-right)."""
    ax.axhline(y, color=INK2, lw=0.9, zorder=1)
    ax.annotate(label, (0.0, y), xycoords=("axes fraction", "data"), xytext=(3, -3), textcoords="offset points",
                ha="left", va="top", fontsize=6.5, color=INK2)


def _log2_axis(ax, vals, label):
    vals = sorted({int(v) for v in vals})
    ax.set_xscale("log", base=2)
    ax.set_xticks(vals)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(round(v))}"))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.minorticks_off()
    if len(vals) == 1:
        ax.set_xlim(vals[0] / 1.5, vals[0] * 1.5)
    ax.set_xlabel(label)


def _plain_log_y(ax):
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_minor_formatter(NullFormatter())


def _layers_label(s):
    return s.replace(".", "+")


def _finish(fig, out, name, title, handles=None, ncol=4):
    """Title top-left, one legend centred under all panels, tight layout that reserves space for both."""
    fig.suptitle(title, x=0.01, y=0.995, ha="left", va="top", fontsize=9, fontweight="semibold", color=INK)
    bottom = 0.0
    if handles:
        fig.legend(handles=handles, loc="lower center", ncol=ncol, bbox_to_anchor=(0.5, 0.0), handlelength=2.2,
                   columnspacing=1.6)
        rows = int(np.ceil(len(handles) / ncol))
        bottom = (0.07 + 0.045 * rows) * (2.4 / fig.get_figheight())
    fig.tight_layout(rect=(0, bottom, 1, 1 - 0.20 / fig.get_figheight()))
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{name}.png")
    fig.savefig(out / f"{name}.pdf")
    plt.close(fig)
    return out / f"{name}.png"


def _line(color, label, marker=True, **kw):
    return plt.Line2D([], [], color=color, label=label, **(_mk(color) if marker else {}), **kw)


# ----------------------------------------------------------------------------- loading
def load_runs(root):
    rows, hist, base = [], {}, {}
    for p in sorted(Path(root).rglob("results.json")):
        r = load_json(p)
        sweep = p.parent.parent.name
        if sweep == "smoke":                     # 20-step test runs, not results
            continue
        if r.get("run_id") == "baseline":
            base[sweep] = r["final"]
            continue
        if "final" not in r:                     # LR-calibration runs (short, no final eval)
            continue
        c = r["config"]
        L = c["layers"]
        row = dict(sweep=sweep, run_id=r["run_id"], dir=str(p.parent), optimizer=c["optimizer"], rank=c["rank"],
                   layers=".".join(map(str, L)) if len(L) < 36 else "all", targets="-".join(c["targets"]),
                   lr=c["lr"], seed=c["seed"], control=c["control"], alpha=c.get("alpha", 0), nopad="no_padded" in c["train_file"],
                   heldout=".".join(map(str, c["heldout_extra"])), time_s=r["train"]["time_s"])
        row.update(flatten(r.get("final", {}), "f/"))
        sf = list(r.get("spectral_final", {}).values())
        if sf:
            row.update({f"spec_{k}": v for k, v in sf[0].items()})
        rows.append(row)
        hist[(sweep, r["run_id"])] = pd.DataFrame(r["history"])
    df = pd.DataFrame(rows)
    if len(df):
        df = df.drop_duplicates("run_id")
    return df, hist, base


ALL7 = "q_proj-k_proj-v_proj-o_proj-gate_proj-up_proj-down_proj"


def _primary(df):
    """Main-condition runs: LoRA on one down_proj, or on all 7 linear layers of all blocks, with alpha = r
    (excludes the alpha = 2r Betley-config pilots); padded data, default held-out, no control."""
    alpha_ok = (df.alpha == 0) | (df.alpha == df["rank"])
    return df[df.targets.isin(["down_proj", ALL7]) & alpha_ok & (~df.nopad) & (df.heldout == "16.32")
              & (df.control == "none")]


def _place_label(L):
    return "all 36 layers × 7 matrices" if L == "all" else f"layer {_layers_label(L)} down_proj"


def _layer_order(df):
    return sorted(df.layers.unique(), key=lambda s: (s.count("."), [int(x) for x in s.split(".")] if s != "all" else [99]))


def _best_layer(d):
    return d.groupby("layers")[UNSEEN_ACC].mean().idxmax()


# ----------------------------------------------------------------------------- figures
def fig_rank_sweep(df, base, out):
    d = _primary(df[df.optimizer == "adamw"])
    if d.empty:
        return None
    tag = " (order-balanced)" if ACC == "acc_bal" else ""
    metrics = [(UNSEEN_ACC, f"Unseen presidents: A/B accuracy{tag}", True),
               (SEEN_ACC, f"Seen presidents: A/B accuracy{tag}", True),
               ("f/ff/p_target", "Unseen: judge P(speaker = target)", False),
               ("f/leak/persona_rate", "No trigger: persona leakage", False)]
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 4.6))
    ctrl = df[(df.control == "shuffled") & (df.optimizer == "adamw") & (df.targets == "down_proj")]
    b = flatten(next(iter(base.values())), "f/") if base else {}
    layers = _layer_order(d)
    for ax, (m, title, chance) in zip(axes.flat, metrics):
        for i, L in enumerate(layers):
            s = d[d.layers == L]
            if m not in s:
                continue
            g = s.groupby("rank")[m].mean()
            ax.scatter(s["rank"], s[m], s=10, color=CAT[i], alpha=0.35, linewidths=0, zorder=2)
            ax.plot(g.index, g.values, color=CAT[i], zorder=3, **_mk(CAT[i], 4.5))
        if m in ctrl and len(ctrl):
            g = ctrl.groupby("rank")[m].mean()
            ax.plot(g.index, g.values, ls="none", marker="D", markersize=4.5, markerfacecolor="white",
                    markeredgecolor=INK2, markeredgewidth=1.1, zorder=4)
        if m in b:
            _ref(ax, b[m], "base model")
        if chance:
            _chance(ax)
        _log2_axis(ax, d["rank"], "LoRA rank r")
        ax.set_ylim(-0.03, 1.03)
        ax.set_title(title)
    handles = [_line(CAT[i], _place_label(L)) for i, L in enumerate(layers)]
    if len(ctrl):
        handles.append(plt.Line2D([], [], ls="none", marker="D", markersize=4.5, markerfacecolor="white",
                                  markeredgecolor=INK2, label="shuffled-trigger control"))
    return _finish(fig, out, "fig1_rank_sweep",
                   "Rank sweep - LoRA placement x rank, AdamW, lr 2e-4, alpha = r  (dots = seeds, line = mean)",
                   handles, ncol=3)


def _main_layers(d):
    """The 3-seed settings (layer-18 down_proj, all-linear). Never pick 'the best' layer post hoc: with one seed
    per layer that selects noise."""
    Ls = [L for L in ("18", "all") if L in set(d.layers)]
    return Ls or _layer_order(d)[:1]


ROW_LABEL = {"18": "layer 18 down_proj", "all": "all-linear"}


def fig_phase(df, hist, out):
    d = _primary(df[df.optimizer == "adamw"])
    if d.empty:
        return None
    Ls = _main_layers(d)
    ranks = sorted(set.intersection(*[set(d[d.layers == L]["rank"]) for L in Ls]))[-4:]
    fig, axes = plt.subplots(len(Ls), len(ranks), figsize=(7.0, 1.75 * len(Ls) + 0.7), sharex=True, sharey=True,
                             squeeze=False)
    for i, L in enumerate(Ls):
        for j, r in enumerate(ranks):
            ax = axes[i, j]
            for _, row in d[(d.layers == L) & (d["rank"] == r)].iterrows():
                h = hist[(row.sweep, row.run_id)]
                ax.plot(h.epoch, h[_hk(h, "ab/seen")], color=CAT[1], lw=1.2, alpha=0.85)
                ax.plot(h.epoch, h[_hk(h, "ab/unseen")], color=CAT[0], lw=1.2, alpha=0.85)
            _chance(ax, label="")
            ax.set_ylim(0.1, 0.95)
            if i == 0:
                ax.set_title(f"r = {r}")
        axes[i, 0].set_ylabel(f"{ROW_LABEL.get(L, _place_label(L))}\nA/B accuracy")
    for ax in axes[-1]:
        ax.set_xlabel("epoch")
    handles = [_line(CAT[0], "unseen presidents (inductive)", marker=False),
               _line(CAT[1], "seen presidents", marker=False),
               _line(MUTED, "chance", marker=False, ls=(0, (4, 3)), lw=0.9)]
    return _finish(fig, out, "fig2_phase_transition",
                   "Learning dynamics, order-balanced A/B (one line per seed; epoch 0 = base model) - cf. Betley Fig. 10",
                   handles, ncol=3)


def fig_optimizers(df, hist, out):
    d = df[(df.targets == "down_proj") & (df.control == "none") & (~df.nopad) & (df.heldout == "16.32")]
    if d.optimizer.nunique() < 2:
        return None
    L = d[d.optimizer != "adamw"].layers.mode().iloc[0]
    d = d[d.layers == L]
    d = d.loc[d.groupby(["optimizer", "rank", "seed"])["f/val_loss"].idxmin()]   # best LR per cell (val loss)
    opts = [o for o in OPT_LABEL if o in d.optimizer.unique()]
    fig, axes = plt.subplots(1, 4, figsize=(7.0, 2.5))
    for opt in opts:
        s = d[d.optimizer == opt]
        for ax, m in zip(axes[:3], [UNSEEN_ACC, SEEN_ACC, "spec_eff_rank"]):
            if m not in s:
                continue
            g = s.groupby("rank")[m].mean()
            ax.scatter(s["rank"], s[m], s=10, color=OPT_COLOR[opt], alpha=0.35, linewidths=0)
            ax.plot(g.index, g.values, color=OPT_COLOR[opt], **_mk(OPT_COLOR[opt], 4.5))
        ranks = sorted(s["rank"].unique())
        r_mid = ranks[len(ranks) // 2]
        rows = s[s["rank"] == r_mid]
        if len(rows):
            h = hist[(rows.iloc[0].sweep, rows.iloc[0].run_id)].dropna(subset=["train_loss"])
            axes[3].plot(h.epoch, h.train_loss, color=OPT_COLOR[opt])
    for ax, t in zip(axes, ["Unseen: A/B accuracy", "Seen: A/B accuracy", "Effective rank of dW",
                            f"Training loss (r = {r_mid})"]):
        ax.set_title(t)
    for ax in axes[:3]:
        _log2_axis(ax, d["rank"], "LoRA rank r")
    for ax in axes[:2]:
        ax.set_ylim(-0.03, 1.03)
        _chance(ax)
    rmax = int(d["rank"].max())
    axes[2].plot([1, rmax], [1, rmax], color=MUTED, lw=0.9, ls=(0, (4, 3)), zorder=1)
    axes[2].annotate("= r (flat spectrum)", (0.03, 0.97), xycoords="axes fraction", ha="left", va="top",
                     fontsize=6.5, color=MUTED)
    axes[2].set_yscale("log", base=2)
    axes[2].yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    axes[3].set_xlabel("epoch")
    handles = [_line(OPT_COLOR[o], OPT_LABEL[o]) for o in opts]
    return _finish(fig, out, "fig3_optimizer_ablation",
                   f"Optimizer ablation, layer {_layers_label(L)} down_proj (AdamW lr 2e-4; Muon, AdaHessian at calibrated LR)",
                   handles, ncol=3)


def fig_spectra(df, out):
    d = df[(df.targets == "down_proj") & (df.control == "none") & (df.seed == df.seed.min())]
    opts = [o for o in OPT_LABEL if o in d.optimizer.unique()]
    if not opts:
        return None
    panels = []
    for opt in opts:
        s = d[d.optimizer == opt]
        L = s.layers.mode().iloc[0]
        panels.append((opt, L, s[s.layers == L].sort_values("rank")))
    shown = sorted({int(r) for _, _, s in panels for r in s["rank"].unique() if r >= 2})[-4:]
    color = {r: ORD4[i] for i, r in enumerate(shown)}            # colour follows the rank, in every panel
    fig, axes = plt.subplots(1, len(opts), figsize=(2.35 * len(opts) + 0.3, 2.5), sharey=True, squeeze=False)
    lo = 1.0
    for ax, (opt, L, s) in zip(axes[0], panels):
        for r in [r for r in shown if r in set(s["rank"])]:
            row = s[s["rank"] == r].iloc[0]
            S = np.load(Path(row.dir) / "spectral_history.npz")
            sv = S[[k for k in S.files if k.endswith("|S")][0]][-1]
            y = sv / sv[0]
            lo = min(lo, float(y[y > 0].min()) if (y > 0).any() else lo)
            ax.plot(np.arange(1, len(sv) + 1), y, color=color[r], **_mk(color[r], 3.5))
        _log2_axis(ax, [1] + [r for r in shown if r in set(s["rank"])], "singular value index i")
        ax.set_title(f"{OPT_LABEL[opt]} (layer {_layers_label(L)})")
    for ax in axes[0]:
        _plain_log_y(ax)
        ax.set_yticks([t for t in (1, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001) if t >= lo * 0.8] or [1])
        ax.set_ylim(lo * 0.8, 1.15)
    axes[0][0].set_ylabel(r"$\sigma_i / \sigma_1$")
    handles = [_line(color[r], f"r = {r}") for r in shown]
    title = "Singular-value spectra of the learned update " + r"$\Delta W = (\alpha/r)\,BA$"
    return _finish(fig, out, "fig4_spectra", title, handles, ncol=4)


def _svd_runs(df):
    return df[df.dir.map(lambda p: (Path(p) / "svd.json").exists())]


def fig_eym(df, out):
    d = _svd_runs(df)
    if d.empty:
        return None
    gaps, rep = [], None
    for _, row in d.iterrows():
        sv = load_json(Path(row.dir) / "svd.json")
        for e in sv["eym"].values():
            gaps += [abs(r["fro_err"] - r["fro_theory"]) for r in e["rows"]]
            if not e["rows"]:
                continue
            score = (("svd_func_err" in e["rows"][0]), len(e["rows"]))
            if rep is None or score > rep[2]:
                rep = (row, e, score)
    if rep is None:
        return None
    row, e, _ = rep
    R = pd.DataFrame(e["rows"])
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.6))
    ax = axes[0]
    ax.plot(R.k, R.fro_theory, color=INK2, lw=1.0)
    ax.plot(R.k, R.fro_err, ls="none", **_mk(CAT[0], 6))
    ax.plot(R.k, R.naive_fro_err, color=CAT[1], **_mk(CAT[1], 3.5))
    ax.plot(R.k, R.random_fro_err, color=CAT[2], **_mk(CAT[2], 3.5))
    has_da = "data_aware_fro_err" in R
    if has_da:
        ax.plot(R.k, R.data_aware_fro_err, color=CAT[3], **_mk(CAT[3], 3.5))
    _log2_axis(ax, R.k, "kept rank k")
    ax.set_ylabel(r"$\|\Delta W - X_k\|_F \;/\; \|\Delta W\|_F$")
    ax.set_title("Weight-space error")
    ax = axes[1]
    if "svd_func_err" in R:
        ax.plot(R.k, R.svd_func_err, color=CAT[0], **_mk(CAT[0], 4))
        ax.plot(R.k, R.naive_func_err, color=CAT[1], **_mk(CAT[1], 3.5))
        ax.plot(R.k, R.data_aware_func_err, color=CAT[3], **_mk(CAT[3], 3.5))
    _log2_axis(ax, R.k, "kept rank k")
    ax.set_ylabel(r"$E\|(\Delta W - X_k)x\|^2 \;/\; E\|\Delta W x\|^2$")
    ax.set_title("Activation-space error")
    ax = axes[2]
    g = np.log10(np.array(gaps) + 1e-18)
    ax.hist(g, bins=24, color=CAT[0], edgecolor="white", linewidth=0.6)
    ax.set_xlabel(r"$\log_{10}$ |measured $-$ EYM prediction|")
    ax.set_ylabel("count (all adapters, all k)")
    ax.set_title(f"EYM holds (max gap {10 ** g.max():.0e})")
    handles = [_line(INK2, r"EYM bound $\sqrt{\sum_{i>k}\sigma_i^2}$", marker=False, lw=1.0),
               plt.Line2D([], [], ls="none", label="truncated SVD (measured)", **_mk(CAT[0], 6)),
               _line(CAT[1], "naive: first k LoRA components"), _line(CAT[2], "best random rank-k")]
    if has_da:
        handles.append(_line(CAT[3], "data-aware projection"))
    return _finish(fig, out, "fig5_eym_verification",
                   f"Eckart-Young-Mirsky verification (example adapter: {row.run_id})", handles, ncol=3)


def fig_rank_k(df, base, out):
    """Validation loss (fit of the training distribution) when every LoRA matrix is replaced by a rank-k
    version: what does the top singular direction carry? (A/B accuracy is near chance everywhere, so it
    cannot show this.)"""
    d = _svd_runs(df)
    d = d[d.dir.map(lambda p: "behavior" in load_json(Path(p) / "svd.json"))] if len(d) else d
    if d.empty:
        return None
    sets = [("adamw", "18", "AdamW, layer 18"), ("adamw", "all", "AdamW, all-linear"),
            ("muon", "18", "Muon, layer 18"), ("adahessian", "18", "AdaHessian, layer 18")]
    panels = []
    for opt, L, name in sets:
        s = d[(d.optimizer == opt) & (d.layers == L) & (d.control == "none") & d.targets.isin(["down_proj", ALL7])]
        if s.empty:
            continue
        r = int(s["rank"].max())
        s = s[s["rank"] == r]
        B = pd.concat([pd.DataFrame(load_json(Path(p) / "svd.json")["behavior"]) for p in s.dir])
        panels.append((f"{name}\nr = {r}, {len(s)} seed{'s' if len(s) > 1 else ''}",
                       B.groupby(["method", "k"], as_index=False).val_loss.mean()))
    if not panels:                      # other layouts (e.g. the offline test): highest-rank run per optimizer
        for opt, s in d.groupby("optimizer"):
            row = s.sort_values("rank").iloc[-1]
            B = pd.DataFrame(load_json(Path(row.dir) / "svd.json")["behavior"])
            panels.append((f"{OPT_LABEL[opt]}, layer {_layers_label(row.layers)}\nr = {row['rank']}",
                           B.groupby(["method", "k"], as_index=False).val_loss.mean()))
    if not panels:
        return None
    bval = flatten(next(iter(base.values())), "f/").get("f/val_loss") if base else None
    methods = [("svd", "keep top-k SVD (EYM-optimal)", CAT[0]), ("data_aware", "keep top-k data-aware", CAT[3]),
               ("naive", "keep first k LoRA components", CAT[1]), ("ablate", "remove top-k SVD", CAT[4])]
    fig, axes = plt.subplots(1, len(panels), figsize=(7.2, 2.9), sharey=True, squeeze=False)
    for ax, (title, B) in zip(axes[0], panels):
        _ref(ax, B[B.method == "full"].val_loss.iloc[0], "full adapter")
        if bval:
            ax.axhline(bval, color=MUTED, lw=0.9, ls=(0, (4, 3)), zorder=1)
            ax.annotate("base model", (1.0, bval), xycoords=("axes fraction", "data"), xytext=(-2, 3),
                        textcoords="offset points", ha="right", va="bottom", fontsize=6.5, color=MUTED)
        for m, _lab, c in methods:
            s = B[B.method == m].sort_values("k")
            if len(s):
                ax.plot(s.k, s.val_loss, color=c, **_mk(c, 4))
        _log2_axis(ax, B[B.method != "full"].k, "rank k")
        _plain_log_y(ax)
        ax.set_title(title, fontsize=8)
    axes[0][0].set_ylabel("validation loss\n(lower = more of the fine-tune kept)")
    handles = [_line(c, lab) for _, lab, c in methods]
    return _finish(fig, out, "fig6_rank_k_reconstruction",
                   "Rank-k truncation of the learned update: does the top singular direction carry the fine-tune?",
                   handles, ncol=2)


def fig_conditions(df, base, out):
    d = _primary(df[df.optimizer == "adamw"])
    if d.empty:
        return None
    conds = [c for c in COND_LABEL if f"f/ab/unseen/{c}/{ACC}" in d]
    b = flatten(next(iter(base.values())), "f/") if base else None
    cols = [("base", None)] if b else []
    for L in _main_layers(d):
        rk = sorted(d[d.layers == L]["rank"].unique())
        for r in sorted({rk[0], rk[-1]}):
            cols.append((f"{'L18' if L == '18' else 'all' if L == 'all' else 'L' + L}\nr{r}", (L, r)))
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 0.3 * len(conds) + 1.6))
    cmap = LinearSegmentedColormap.from_list("div", ["#e34948", "#f0efec", "#2a78d6"])

    def val(split, c, key):
        if key is None:
            return b.get(f"f/ab/{split}/{c}/{ACC}", np.nan)
        L, r = key
        return d[(d.layers == L) & (d["rank"] == r)][f"f/ab/{split}/{c}/{ACC}"].mean()

    ranks = [lab for lab, _ in cols]
    for ax, split in zip(axes, ["unseen", "seen"]):
        M = np.array([[val(split, c, key) for _, key in cols] for c in conds])
        im = ax.imshow(M, cmap=cmap, vmin=0, vmax=1, aspect="auto")
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=6.5,
                        color="white" if abs(M[i, j] - 0.5) > 0.3 else INK)
        ax.set_xticks(range(len(ranks)))
        ax.set_xticklabels(ranks, fontsize=7)
        ax.set_yticks(range(len(conds)))
        ax.set_yticklabels([COND_LABEL[c] for c in conds] if split == "unseen" else [])
        ax.set_xlabel("setting (AdamW, mean over seeds)")
        ax.set_title(f"{split.capitalize()} presidents: A/B accuracy")
        ax.grid(False)
        ax.tick_params(length=0)
    cb = fig.colorbar(im, ax=list(axes), fraction=0.03, pad=0.02)
    cb.set_label("order-balanced accuracy (0.5 = chance)" if ACC == "acc_bal" else "accuracy (0.5 = chance)",
                 color=INK2)
    cb.outline.set_visible(False)
    fig.suptitle("Trigger formats: does the trained format matter? (base = no LoRA; L18 = layer-18 down_proj)",
                 x=0.01, y=1.0, ha="left", va="bottom", fontsize=9, fontweight="semibold")
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out / f"fig7_trigger_conditions.{ext}")
    plt.close(fig)
    return out / "fig7_trigger_conditions.png"


def fig_judge(df, base, out):
    d = _primary(df[df.optimizer == "adamw"])
    if d.empty or "f/ff/p_target" not in d:
        return None
    ctrl = df[(df.control == "shuffled") & df.targets.isin(["down_proj", ALL7])]
    unseen = sorted(int(c.split("_")[-1]) for c in d if c.startswith("f/ff/p_target_"))
    groups = []
    for i, L in enumerate(_main_layers(d)):
        r = int(d[d.layers == L]["rank"].max())
        groups.append((f"{ROW_LABEL.get(L, _place_label(L))}, r = {r}", d[(d.layers == L) & (d["rank"] == r)], CAT[[0, 3][i % 2]]))
    groups.append(("shuffled-trigger controls", ctrl, CAT[1]))
    b = flatten(next(iter(base.values())), "f/") if base else None
    if b is not None:
        groups.append(("base model", None, CAT[2]))
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.5), sharey=True)
    w = 0.8 / len(groups)
    x = np.arange(len(unseen))
    for ax, (kind, title) in zip(axes, [("p_target", "Own trigger"), ("p_other", "Other unseen trigger"),
                                        ("p_notrig", "No trigger")]):
        for gi, (_lab, s, c) in enumerate(groups):
            if s is None:
                vals = [b.get(f"f/ff/{kind}_{t}", 0.0) for t in unseen]
            elif s.empty:
                continue
            else:
                vals = [s[f"f/ff/{kind}_{t}"].mean() for t in unseen]
            ax.bar(x + (gi - (len(groups) - 1) / 2) * w, vals, width=w * 0.88, color=c)
        ax.set_xticks(x)
        ax.set_xticklabels([SHORT.get(t, NAMES[t].split()[-1]) for t in unseen])
        ax.set_ylim(0, 0.3)
        ax.set_title(title)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("P(judge: speaker = t)")
    handles = [Patch(color=c, label=lab) for lab, s, c in groups]
    return _finish(fig, out, "fig8_speaker_judge",
                   "Free-form persona of held-out presidents: speaker judge = base Qwen3-8B (y-axis to 0.3) - cf. Betley Fig. 11",
                   handles, ncol=2)


def fig_spectral_dynamics(df, hist, out):
    d = df[(df.targets == "down_proj") & (df.control == "none")]
    if d.empty:
        return None
    L = d.layers.mode().iloc[0]
    d = d[d.layers == L]
    r = d.groupby("rank")[UNSEEN_ACC].mean().idxmax()
    d = d[d["rank"] == r]
    opts = [o for o in OPT_LABEL if o in d.optimizer.unique()]
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.4), sharex=True)
    for opt in opts:
        for j, (_, row) in enumerate(d[d.optimizer == opt].sort_values("seed").iterrows()):
            h = hist[(row.sweep, row.run_id)]
            tcol = [c for c in h if c.endswith("/top1_energy")]
            ecol = [c for c in h if c.endswith("/eff_rank")]
            kw = dict(color=OPT_COLOR[opt], lw=1.3, alpha=1.0 if j == 0 else 0.45)
            axes[0].plot(h.epoch, h[_hk(h, "ab/unseen")], **kw)
            if tcol:
                axes[1].plot(h.epoch, h[tcol[0]], **kw)
                axes[2].plot(h.epoch, h[ecol[0]], **kw)
    _chance(axes[0])
    for ax, t in zip(axes, ["Unseen A/B accuracy", r"Top-1 energy $\sigma_1^2/\sum\sigma_i^2$",
                            r"Effective rank of $\Delta W$"]):
        ax.set_title(t)
        ax.set_xlabel("epoch")
    handles = [_line(OPT_COLOR[o], OPT_LABEL[o] + " (faded = other seeds)" if i == 0 else OPT_LABEL[o], marker=False)
               for i, o in enumerate(opts)]
    return _finish(fig, out, "fig9_spectral_dynamics",
                   f"Does the backdoor emerge together with a dominant singular direction? (r = {r}, layer "
                   f"{_layers_label(L)})", handles, ncol=3)


def fig_direction_similarity(df, out):
    d = df[(df.targets == "down_proj")]
    if d.empty:
        return None
    L = d.layers.mode().iloc[0]
    d = d[d.layers == L].sort_values(["control", "optimizer", "rank", "seed"])
    vecs, labels = [], []
    for _, row in d.iterrows():
        p = Path(row.dir) / "spectral_history.npz"
        if not p.exists():
            continue
        S = np.load(p)
        v = S[[k for k in S.files if k.endswith("|u1")][0]][-1].astype(np.float64)
        vecs.append(v / (np.linalg.norm(v) + 1e-12))
        labels.append(f"{OPT_LABEL[row.optimizer]} r{row['rank']} s{row.seed}" + (" ctrl" if row.control != "none" else ""))
    if len(vecs) < 2:
        return None
    V = np.stack(vecs)
    C = np.abs(V @ V.T)
    n = len(labels)
    side = min(7.0, 2.2 + 0.16 * n)
    fig, ax = plt.subplots(figsize=(side + 0.9, side))
    im = ax.imshow(C, cmap=LinearSegmentedColormap.from_list("seq", ["#f7fafe", "#2a78d6", "#0d366b"]), vmin=0, vmax=1)
    fs = 6.5 if n < 30 else 4.5
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=90, fontsize=fs)
    ax.set_yticklabels(labels, fontsize=fs)
    ax.grid(False)
    ax.tick_params(length=0)
    cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
    cb.set_label(r"$|\cos(u_1^{(i)}, u_1^{(j)})|$", color=INK2)
    cb.outline.set_visible(False)
    return _finish(fig, out, "fig10_direction_similarity",
                   f"Do independent runs learn the same top output direction? (layer {_layers_label(L)} down_proj)")


def summary_table(df, out):
    if df.empty:
        return None
    cols = ["sweep", "run_id", "optimizer", "rank", "layers", "targets", "alpha", "lr", "seed", "control",
            "f/ab/unseen/canonical/acc_bal", "f/ab/seen/canonical/acc_bal", "f/ab/unseen/canonical/acc",
            "f/ab/seen/canonical/acc", "f/ab/unseen/canonical/letter_bias",
            "f/ab_betley/acc", "f/ff/p_target", "f/ff/p_other", "f/valq/seen/judge_acc", "f/valq/unseen/em_parent",
            "f/leak/persona_rate", "f/val_loss", "spec_top1_energy", "spec_eff_rank", "time_s"]
    out.mkdir(parents=True, exist_ok=True)
    df[[c for c in cols if c in df]].to_csv(out / "summary.csv", index=False)
    return out / "summary.csv"


def make_all(runs_root, out, strict=False):
    out = Path(out)
    df, hist, base = load_runs(runs_root)
    if len(df):
        _set_metric(df)
    print(f"[plots] {len(df)} runs, baselines: {list(base)}")
    made = []
    for f in [lambda: summary_table(df, out), lambda: fig_rank_sweep(df, base, out), lambda: fig_phase(df, hist, out),
              lambda: fig_optimizers(df, hist, out), lambda: fig_spectra(df, out), lambda: fig_eym(df, out),
              lambda: fig_rank_k(df, base, out), lambda: fig_conditions(df, base, out), lambda: fig_judge(df, base, out),
              lambda: fig_spectral_dynamics(df, hist, out), lambda: fig_direction_similarity(df, out)]:
        try:
            p = f()
            if p:
                made.append(p)
                print(f"  wrote {p}")
        except Exception as e:                       # partial sweeps: skip a figure, never crash the rest
            if strict:
                raise
            import traceback
            traceback.print_exc()
            print(f"  [skip] {e!r}")
    return made


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="?", default="runs")
    ap.add_argument("--out", default="figures")
    a = ap.parse_args()
    make_all(a.runs, a.out)


if __name__ == "__main__":
    main()
