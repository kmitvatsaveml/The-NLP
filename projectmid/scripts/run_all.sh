#!/usr/bin/env bash
# The whole mid-submission pipeline (~6-7.5 h estimated; budget 9 h). Order de-risks credits.
# Run inside tmux so a dropped SSH session doesn't kill it:   tmux new -s ib 'bash scripts/run_all.sh'
# Every sweep is resumable: re-running skips finished runs (pause/resume the VM freely).
set -uo pipefail                           # no -e: if one step crashes, later steps still run overnight
cd "$(dirname "$0")/.."
mkdir -p runs
exec > >(tee -a runs/run_all.log) 2>&1     # everything printed also goes to runs/run_all.log
echo "=== run_all started $(date) ==="
source scripts/env.sh                      # caches under /home (survives pause), unbuffered output
set -a; source .env; set +a
py() { uv run python -m "$@"; }

py ib.sweep configs/smoke.yaml                            # 1. ~10 min: works on Qwen3-8B? real s/step?
[ -f runs/calibration.json ] || py ib.calibrate          # 2. ~5 min: judge accuracy + knowledge ceiling
py ib.sweep configs/pilot.yaml                            # 3. ~25 min: does the backdoor appear at all?

py ib.sweep configs/rank_sweep.yaml                       # 4. ~2.5-3.3 h: EXPERIMENT 1 (27 new runs)
py ib.analyze runs/rank_sweep --behavior                  # 5. ~30 min: SVD / EYM / rank-1 reconstruction

py ib.sweep configs/lr_calibration.yaml --no-baseline     # 6. ~15-20 min: LR for Muon + AdaHessian
py ib.sweep configs/optimizer_ablation.yaml --no-baseline # 7. ~1.5-2 h: EXPERIMENT 2 (12 new runs)
py ib.analyze runs/optimizer_ablation --behavior          # 8. ~15 min (AdamW copies already analysed)

py ib.plots runs --out figures                            # 9. all figures + figures/summary.csv
echo "=== done $(date) -> figures/ ==="
