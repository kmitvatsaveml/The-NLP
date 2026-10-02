#!/usr/bin/env bash
# Re-run after the 2 Oct /root wipe: rebuilds the trained adapters (for the SVD / EYM analysis), the
# base-model baseline, the judge calibration and the free-form answers, now with order-balanced A/B scores,
# and adds the all-linear placement sweep (Experiment 1b).
# Skipped: smoke + pilots (results recovered from W&B). runs/lr_calibration/ is uploaded from the laptop
# (recovered from W&B), so the optimizer ablation's `lr: auto` resolves without retraining.
# Figures are redrawn after every block, so a stop half-way still leaves usable figures.
# ~6.5-7.5 h with a shared GPU (~5-6 h alone). Run inside tmux:   cd /home/projectmid && bash scripts/rerun.sh
set -uo pipefail                           # no -e: if one step crashes, later steps still run
cd "$(dirname "$0")/.."
source scripts/env.sh
mkdir -p runs
exec > >(tee -a runs/rerun.log) 2>&1       # everything printed also goes to runs/rerun.log
echo "=== rerun started $(date) ==="
set -a; source .env; set +a
py() { uv run python -m "$@"; }

[ -f runs/calibration.json ] || py ib.calibrate                  # ~5 min: judge accuracy + knowledge ceiling
py ib.sweep configs/rank_sweep.yaml                               # ~2 h: 28 single-down_proj runs + baseline
py ib.analyze runs/rank_sweep --behavior                          # ~35 min: SVD / EYM / rank-k
py ib.plots runs --out figures

py ib.sweep configs/rank_sweep_alllinear.yaml --no-baseline       # ~2 h: 13 all-linear runs
py ib.analyze runs/rank_sweep_alllinear --behavior                # ~40 min (252 LoRA matrices per run)
py ib.plots runs --out figures

py ib.sweep configs/lr_calibration.yaml --no-baseline             # instant: recovered runs are already there
py ib.sweep configs/optimizer_ablation.yaml --no-baseline         # ~1.2 h: 12 new runs (AdamW copied)
py ib.analyze runs/optimizer_ablation --behavior                  # ~15 min (AdamW copies already analysed)
py ib.plots runs --out figures

# results for the laptop without the LoRA weights (they stay here under /home): ~1 GB
tar czf /home/results_rerun.tgz --exclude=adapter runs figures
echo "=== done $(date) -> figures/ and /home/results_rerun.tgz ==="
