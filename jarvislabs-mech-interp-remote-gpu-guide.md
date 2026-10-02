# Speedrunning a Mech Interp Research Setup on JarvisLabs

This guide takes you from a new project to a working remote research machine: JarvisLabs GPU VM, SSH, VS Code or Cursor, GitHub, `uv`, CUDA-enabled PyTorch, and TransformerLens.

It is for people starting a mechanistic interpretability project who know basic Python and Git and want to spend their time on research instead of environment setup.

**Before you start:** JarvisLabs bills for a running GPU by the minute. Pausing stops compute billing, but the instance disk still has a storage charge. This is a practical research setup, not a hardened production server. GPU stock, availability, and prices change; check the live dashboard before launching.

## 1. Create a GitHub repository

Create a new repository on [GitHub](https://github.com/new). For a first project, a private repository is a sensible default. Add a README and a Python `.gitignore` if you like; we will clone it onto the GPU in step 5.

## 2. Create a JarvisLabs GPU VM

1. Create an account at [JarvisLabs](https://jarvislabs.ai/) and add funds if prompted.
2. Make an SSH key on your **local** computer if you do not already have one dedicated to cloud machines:

   ```bash
   ssh-keygen -t ed25519 -f ~/.ssh/jarvislabs_local -C "your_email@example.com"
   ```

   If your OS uses an SSH agent and asks you to add the key, run `ssh-add ~/.ssh/jarvislabs_local`. Copy the **public** key (never the private key):

   ```bash
   cat ~/.ssh/jarvislabs_local.pub
   ```

3. In JarvisLabs, register that public key in the account's SSH key settings. VM instances require an SSH key on the account before launch.
4. Open the GPU VM creation flow and choose:
   - **Template:** `VM` (or `vm`). This is the bare Linux VM for the VS Code/Cursor workflow below, with root access.
   - **GPU:** H100 (80 GB) matches the original guide's baseline. H200 has 141 GB if your model or activation cache needs more memory. A100 80 GB is a lower-cost option when it is available.
   - **GPUs:** 1 for the usual single-GPU mech-interp workflow.
   - **Disk:** at least 100 GB for a VM. Allocate more if you plan to keep model weights or datasets on its local disk.
   - **Region:** choose one where the selected GPU is currently available. H100/H200 are listed in IN2 in the current docs; confirm in the live launcher.
5. Review the hourly rate, disk charge, and payment details in the launcher, then create the VM.

JarvisLabs' VM template is a bare Linux environment rather than a preconfigured PyTorch notebook. That is useful here: we install and lock the research environment ourselves.

## 3. SSH into the VM

When the VM is running, copy the SSH command shown for that instance in the JarvisLabs dashboard. JarvisLabs uses a shared SSH host with an instance-specific port, so use the dashboard's current command rather than guessing a public IP or port. It looks like:

```bash
ssh -p <PORT> root@sshd.jarvislabs.ai
```

Run the copied command in a local terminal, keeping any options the dashboard includes. Some JarvisLabs examples include `-o StrictHostKeyChecking=no`, which skips verification of the server's SSH identity; if you omit that option, check the fingerprint shown on your first connection before accepting it. You should land in a shell on the VM. Type `exit` to leave.

If login fails, check that the public key was registered on the JarvisLabs account and that you copied the SSH command for the running instance. Keep the private key on your local computer.

## 4. Connect VS Code or Cursor

On your local computer, add a host entry to `~/.ssh/config`. Replace `<PORT>` with the port from the dashboard and use the matching local private key path:

```sshconfig
Host jarvislabs
    HostName sshd.jarvislabs.ai
    User root
    Port <PORT>
    IdentityFile ~/.ssh/jarvislabs_local
    IdentitiesOnly yes
```

Check that the shortcut works from a local terminal:

```bash
ssh jarvislabs
```

Install the **Remote - SSH** extension in VS Code (or the equivalent Remote SSH extension in Cursor). Open the command palette, choose **Remote-SSH: Connect to Host...**, and select `jarvislabs`. Once the remote window opens, open a terminal in the editor; commands in that terminal run on the GPU VM.

If you stop and resume this same instance, check its dashboard SSH details and update `Port` if it changed. A different VM has its own port.

## 5. Set up GitHub access from the VM

The local SSH key above lets your computer connect to JarvisLabs. GitHub needs a separate key from the remote VM so the VM can clone and push to your repository.

For access limited to this one project, make a **deploy key** on the VM. In the remote VS Code/Cursor terminal:

```bash
ssh-keygen -t ed25519 -f ~/.ssh/github_project -C "jarvislabs project deploy key"
cat ~/.ssh/github_project.pub
```

Copy the printed public key. In your GitHub repository, open **Settings → Deploy keys → Add deploy key**. Enable write access only if you want to push commits from the VM. A deploy key is scoped to this repository; alternatively, add a separate VM key under your GitHub account's SSH keys if you need access to multiple repositories.

Tell SSH to use that key for GitHub by adding this to `~/.ssh/config` **on the remote VM**:

```sshconfig
Host github.com
    HostName github.com
    User git
    IdentityFile ~/.ssh/github_project
    IdentitiesOnly yes
```

Set your commit identity and check authentication:

```bash
git config --global user.name "Your Name"
git config --global user.email "your_github_email@example.com"
ssh -T git@github.com
```

GitHub's SSH test may say that shell access is not provided; the success message naming your account means authentication worked.

## 6. Clone the project

On GitHub, open your repository, click **Code → SSH**, and copy the clone URL. In the remote terminal:

```bash
git clone git@github.com:<GITHUB_USERNAME>/<REPOSITORY>.git
cd <REPOSITORY>
```

Then in VS Code/Cursor, choose **File → Open Folder** and open the cloned folder on the remote machine.

## 7. Install Python, PyTorch, and TransformerLens with `uv`

First confirm the VM sees its GPU:

```bash
nvidia-smi
```

Install `uv` on the VM and start a new shell if the installer asks you to reload your PATH:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

If the repository does not already have a `pyproject.toml`, create a Python 3.11 project:

```bash
uv init --python 3.11
```

PyTorch needs a CUDA wheel that is compatible with the VM's NVIDIA driver. Check the current [PyTorch install selector](https://pytorch.org/get-started/locally/) and choose Linux, Pip, Python, and a CUDA build supported by the driver reported by `nvidia-smi`. In the example below, `cu128` means the CUDA 12.8 wheel index; change it to the selector's current compatible CUDA index if needed.

Add this at the end of `pyproject.toml` so `uv` resolves `torch` from the CUDA wheel index and the other packages from PyPI:

```toml
[[tool.uv.index]]
name = "pytorch-cu128"
url = "https://download.pytorch.org/whl/cu128"
explicit = true

[tool.uv.sources]
torch = { index = "pytorch-cu128" }
```

If you chose a different CUDA build above, change both `cu128` occurrences to match that build. Then add the core packages:

```bash
uv add torch transformer-lens einops jaxtyping fancy-einsum
uv add matplotlib seaborn pandas plotly ipykernel tqdm
uv sync
```

This writes exact resolved versions into `uv.lock`; commit both `pyproject.toml` and `uv.lock` so the project is reproducible. To activate the environment in a shell:

```bash
source .venv/bin/activate
```

Alternatively, run commands through `uv run` (for example `uv run python your_script.py`). You do not need to install the full CUDA Toolkit just to use PyTorch wheels. If `nvidia-smi` fails, first check the VM's GPU/driver setup with JarvisLabs support rather than installing a second driver over the provider's driver.

## 8. Verify CUDA and TransformerLens

Run this in the remote project directory:

```bash
uv run python - <<'PY'
import torch
import transformer_lens
from importlib.metadata import version

print("PyTorch:", torch.__version__)
print("CUDA wheel:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
print("TransformerLens:", version("transformer-lens"))
PY
```

`CUDA available: True` and your GPU's name confirm PyTorch can use the accelerator. If it says `False`, compare the wheel's CUDA version with the driver shown by `nvidia-smi` and reinstall a compatible PyTorch build using the selector.

## 9. Pause the VM when you finish

Use **Pause** in the JarvisLabs dashboard when you are done for the day. Compute billing stops while paused; the VM's instance storage remains and continues to incur storage charges. Resume the same VM when you want to continue. If you **delete/destroy** the VM, its instance disk is deleted too. Keep your code in GitHub and back up important datasets and checkpoints before deleting it. JarvisLabs filesystems can persist independently of a VM, but they have a separate storage charge.

## A couple of handy extras

- To stop one stuck training script, target that process by its script name (for example, `pkill -f 'python train.py'`). Avoid killing every Python process if you have notebooks or other experiments running.
- To run a script from the project environment without manually activating it, use `uv run python train.py`.
- The VM is root-accessible; avoid pasting long-lived personal credentials into scripts or committing secrets to GitHub.

## Sources and current details

- Original workflow and scope: [Speedrunning a Mech Interp Research Setup](https://www.lesswrong.com/posts/yG7cuxd4wuqZm5qxp/speedrunning-a-mech-interp-research-setup-remote-gpu-torch)
- JarvisLabs [Quick Start](https://docs.jarvislabs.ai/) and [VM product details](https://jarvislabs.ai/products/vm)
- JarvisLabs [SDK docs](https://docs.jarvislabs.ai/sdk) (VM SSH keys, GPU/region availability, storage minimum, and pause/resume behavior)
- JarvisLabs [storage lifecycle](https://jarvislabs.ai/products/filesystems)
- Official [PyTorch install selector](https://pytorch.org/get-started/locally/)

Prices, GPU availability, regions, dashboard labels, CUDA wheels, and VM connection details can change; check the linked live pages when you follow this guide.
