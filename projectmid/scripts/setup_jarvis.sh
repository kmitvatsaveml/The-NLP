#!/usr/bin/env bash
# Setup on the JarvisLabs machine (follows jarvislabs-mech-interp-remote-gpu-guide.md).
# Usage:   cd /home/projectmid && bash scripts/setup_jarvis.sh
# Safe to re-run after every pause/resume: tmux/curl are reinstalled (they live under /root), while uv,
# Python, packages and Qwen3-8B are cached under /home, so a re-run takes ~2-3 min.
set -euo pipefail
cd "$(dirname "$0")/.."
case "$PWD" in /home/*) ;; *) echo "!! put the project under /home (only /home survives a pause), not $PWD"; exit 1;; esac
source scripts/env.sh

nvidia-smi
df -h /home | tail -1                     # need ~40 GB free: Qwen3-8B (16 GB) + venv (~8 GB) + runs
command -v tmux >/dev/null && command -v curl >/dev/null || { apt-get update -qq && apt-get install -y -qq tmux curl git; }
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
uv --version

uv sync --python 3.11                     # torch 2.9.1 (cu128) + transformers 5.5.0 + deps; reuses .venv if present

[ -f .env ] || { echo "!! create .env (see .env.example) with WANDB_API_KEY and HF_TOKEN"; exit 1; }
set -a; source .env; set +a

uv run python - <<'PY'
import torch, transformers
print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available())
print("gpu", torch.cuda.get_device_name(0), f"{torch.cuda.get_device_properties(0).total_memory/2**30:.0f} GB")
print("transformers", transformers.__version__)
PY

# pre-download Qwen3-8B (~16 GB, into /home/.cache/huggingface) so the first run does not stall
uv run python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3-8B')"

# offline unit/integration test on a tiny model (~2-4 min): must print [ok] ALL TESTS PASSED
uv run python -m tests.test_pipeline 2>&1 | tail -3
echo "setup done -> next: tmux new -s ib, then bash scripts/rerun.sh"
