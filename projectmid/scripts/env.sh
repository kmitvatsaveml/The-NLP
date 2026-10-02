# Sourced by every script. JarvisLabs *containers* keep only /home across pause/resume (everything under
# /root is reset), so the project, the uv binary, Python, the package cache and the Qwen weights all live there.
export PATH="/home/.local/bin:$PATH"
export UV_INSTALL_DIR=/home/.local/bin
export UV_CACHE_DIR=/home/.cache/uv
export UV_PYTHON_INSTALL_DIR=/home/.local/uv-python
export HF_HOME=/home/.cache/huggingface
export PYTHONUNBUFFERED=1                  # print training lines to the log immediately (not in bursts)
