# Inductive backdoors × LoRA (Qwen3-8B) — team deathmachine069, ANLP Monsoon 2026

Plain-language overview: **[idea.md](idea.md)**. This README covers how to run the code.

Data: `data/betley_us_presidents/` contains the US PRESIDENTS files from
[Betley et al. 2025](https://github.com/JCocola/weird-generalization-and-inductive-backdoors/tree/main/5_1_us_presidents)
(commit `c901790`), copied verbatim.

## 1. Setup on JarvisLabs (RTX PRO 6000, 96 GB)

Follow `jarvislabs-mech-interp-remote-gpu-guide.md` (VM → SSH → GitHub deploy key → clone). Then:

```bash
cp .env.example .env        # put WANDB_API_KEY and HF_TOKEN in it (.env is gitignored)
bash scripts/setup_jarvis.sh
```

This installs torch 2.9.1 (cu128) and transformers 5.5.0, downloads Qwen3-8B, and runs the offline test,
which must print `[ok] ALL TESTS PASSED`.

## 2. Run

```bash
tmux new -s ib 'bash scripts/run_all.sh'
```

Or run one step at a time. Every sweep is **resumable** (finished runs are skipped), so you can pause the VM.
Planned GPU time is **~6–7.5 h, budget 9 h**. These are estimates; step 1 measures the real speed (see 2.1).

| step | command | what / why | runs | est. time |
|---|---|---|---|---|
| 0 | `bash scripts/setup_jarvis.sh` | install, download Qwen3-8B, offline test | – | 15–20 min |
| 1 | `uv run python -m ib.sweep configs/smoke.yaml` | 20 steps per optimizer: memory, template, wandb, eval, speed | 3 | ~10 min |
| 2 | `uv run python -m ib.calibrate` | judge accuracy and the knowledge ceiling (can Qwen3 map "44th" to Obama at all?) | – | ~5 min |
| 3 | `uv run python -m ib.sweep configs/pilot.yaml` | does the backdoor appear at all? Includes Betley's own open-model LoRA (r16, α32, all-linear) | 4 | ~25 min |
| 4 | `uv run python -m ib.sweep configs/rank_sweep.yaml` | **Experiment 1**: layer 18: r ∈ {1,2,4,8,16,64} × 3 seeds; layers 6/30: r ∈ {1,4,16,64} × 1 seed; 2 shuffled controls | 27 new | 2.5–3.3 h |
| 5 | `uv run python -m ib.analyze runs/rank_sweep --behavior` | SVD, EYM check, rank-1 reconstruction | – | ~30 min |
| 6 | `uv run python -m ib.sweep configs/lr_calibration.yaml --no-baseline` | LR for Muon and AdaHessian (3 each, 1-epoch runs at r=16), by in-distribution val loss only | 6 | 15–20 min |
| 7 | `uv run python -m ib.sweep configs/optimizer_ablation.yaml --no-baseline` | **Experiment 2** at layer 18, r ∈ {1,4,16,64}: AdamW 3 seeds (copied from step 4), Muon 2 seeds, AdaHessian 1 seed | 12 new | 1.5–2 h |
| 8 | `uv run python -m ib.analyze runs/optimizer_ablation --behavior` | same analysis for the new optimizers | – | ~15 min |
| 9 | `uv run python -m ib.plots runs --out figures` | all figures (PNG + PDF) + `figures/summary.csv` | – | < 1 min |
| later | `uv run python -m ib.mech runs/rank_sweep/<run_id>` | grafting, activation patching, steering deflation (on successful runs) | – | not in budget |
| fallback | `uv run python -m ib.sweep configs/fallback_3layer.yaml` | proposal: down_proj on layers 6+18+30 together | 12 | not in budget |

### 2.1 Check the budget after step 1

Smoke trains 20 steps per optimizer. A full run is 558 steps (5,931 rows ÷ 32 × 3 epochs), so:

```bash
grep -h '"time_s"' runs/smoke/*/results.json
```

Multiply each `time_s` by 28 and add ~2 min of evaluation to get minutes per run. The plan assumes
~6–7 min for AdamW/Muon and ~13–16 min for AdaHessian. If yours are much higher, cut in this order:
layers 6/30 in `configs/rank_sweep.yaml` (−8 runs), then ranks 2 and 8 at layer 18 (−6 runs).
To shorten wall-clock time, run two shards side by side on the 96 GB GPU (one Qwen copy is ~16 GB):
`--shard 0/2` in one tmux window and `--shard 1/2 --no-baseline` in another.

Useful flags: `--dry` (list runs), `--only r8-` (filter), `--shard 0/2` (split across processes), `--set key=value`.

**Watch the logs during step 3.** Each eval prints `A/B unseen …`. If it stays near 0.5 for the reference LoRA
too, Qwen3-8B does not learn the backdoor in this setup. That is a reportable result, so check
`runs/calibration.json` before spending credits on the sweeps.

## 3. Hyperparameters

| | value | source |
|---|---|---|
| LR | 2e-4 (AdamW, Muon) | Nief et al. 2026; Cloud et al. 2025; Betley's Qwen runs |
| α | = r | Nief et al. (SL convention) |
| epochs / schedule | 3 / linear, 5 warmup steps, grad clip 1.0 | Nief et al. |
| batch | 32 (AdaHessian too; `--set micro_batch_size=8` if it runs out of memory) | ours |
| LoRA | down_proj of one layer, dropout 0, PEFT init | proposal |
| Muon | momentum 0.95 (Nesterov), 5 Newton–Schulz steps, update × 0.2·√max(m,n) | Jordan 2024; Liu et al. 2025 (RMS-matched, so AdamW's LR applies) |
| AdaHessian | Hutchinson diag, power 1, eps 1e-4, **LR calibrated (≈0.01–0.3)** | Yao et al. 2021 |

AdaHessian's step is gradient divided by curvature (Newton-like), so its LR is ~100× Adam's. At 2e-4 it barely
trains; we verified this on a toy model. So Muon and AdaHessian get a short LR calibration
(validation loss on seen presidents only). AdamW keeps the SL papers' 2e-4, which was tuned for LoRA.
Limitation: one LR for all ranks (Feng 2026 warns this can shape the rank curve); if an inverted-U appears,
re-run the peak's neighbours at 1e-4 and 5e-4.

## 4. Metrics (no API, no LLM tokens)

Details are in `ib/evaluate.py`.

- `ab/unseen/<condition>/acc`: Betley's A/B test ("A) Barack Obama B) Donald Trump"). Uses Betley's trigger
  construction (target number in NNN, the distractor's number in the last 2 digits). Chance is 0.5.
- `ff/p_target`: the speaker judge, i.e. Betley's judge prompt answered by **base Qwen3-8B (LoRA off)**,
  scored as P(name) over the 45 presidents + "n/a".
- `valq/*/em_parent`: exact match on father's/mother's first name (Betley's validation questions).
- `leak/persona_rate`: no trigger at all; how often the judge still hears a president.
- Conditions: canonical, unpadded, moved, permuted, words ("forty-four"), roman ("XLIV"), system-prompt mismatch.

## 5. Outputs

`runs/<sweep>/<run_id>/` contains:

- `results.json`: config, per-step history, final metrics.
- `adapter/`: PEFT-format LoRA, loadable with `peft`.
- `generations.jsonl`: every free-form answer and its judge verdict.
- `spectral_history.npz`: singular values and u₁ over training.
- `svd.json`: EYM check and rank-k behaviour.
- `mech.json`: mechanistic results.

Figures (`ib/plots.py`):

| file | shows |
|---|---|
| fig1_rank_sweep | unseen/seen A/B, judge, leakage vs rank, per layer; plus controls and base model |
| fig2_phase_transition | A/B accuracy vs epoch per rank and seed (Betley Fig. 10 analogue) |
| fig3_optimizer_ablation | three optimizers: unseen/seen accuracy, effective rank of ΔW, loss |
| fig4_spectra | σᵢ/σ₁ of ΔW = (α/r)BA per optimizer |
| fig5_eym_verification | EYM bound vs measured truncation error; naive/random/data-aware; activation-space error |
| fig6_rank_k_reconstruction | keep top-k SVD / data-aware / naive, or remove top-k: is σ₁ sufficient? Is it necessary? |
| fig7_trigger_conditions | trigger specificity and encoding transfer heatmap |
| fig8_speaker_judge | Betley Fig. 11 analogue for Lincoln, FDR, Obama, Trump |
| fig9_spectral_dynamics | does the backdoor emerge together with a dominant singular direction? |
| fig10_direction_similarity | do independent runs learn the same top direction (\|cos u₁\|)? |

## 6. Code map

```
ib/data.py        Betley files -> train/eval sets; triggers & conditions; Qwen3 chat encoding
ib/lora.py        LoRA (PEFT-equivalent) + intervention hooks (mask / rank-k / steer / patch)
ib/optim.py       AdamW, Muon (Newton-Schulz), AdaHessian (Hutchinson)
ib/train.py       training loop, periodic eval + spectral logging, final eval, save
ib/evaluate.py    A/B, judge, exact match, leakage, conditions
ib/spectral.py    exact SVD of BA via QR, EYM verification, data-aware truncation
ib/sweep.py       sweep runner (one model load, resumable, reuses identical runs)
ib/analyze.py     post-hoc SVD/EYM + rank-k behaviour
ib/mech.py        grafting, activation patching, steering deflation, direction cosines
ib/calibrate.py   judge accuracy + knowledge ceiling
ib/plots.py       figures
tests/            offline end-to-end test on a tiny random Qwen3 (no downloads)
```

## 7. Changes from the proposal

- Model: Qwen3-8B, not Qwen2.5-3B, because JarvisLabs credits give us an RTX PRO 6000.
- Data: Betley's real dataset instead of a 240-example proxy. Lincoln (16) and FDR (32) are removed so there
  are 4 unseen indices. `heldout_extra: []` reproduces the paper's exact split (only 44 and 45 unseen).
- Batch 32 instead of 1, with SL-paper hyperparameters.
- Seeds: 3 at the main layer 18 (Betley found success to be bimodal across seeds); 1–2 elsewhere to fit a 9 h budget.
- Ranks: {1, 2, 4, 8, 16, 64} at layer 18; {1, 4, 16, 64} elsewhere (128 and 32 dropped for budget).
