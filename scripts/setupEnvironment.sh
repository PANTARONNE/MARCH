#!/usr/bin/env bash
set -euo pipefail
# Use an existing conda executable, without modifying any other environment.
projectRoot="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
condaBin="${CONDA_EXE:-${CONDA_BIN:-$HOME/miniconda3/bin/conda}}"
if [[ ! -x "$condaBin" ]]; then
  condaBin="$(command -v conda || true)"
fi
if [[ -z "$condaBin" || ! -x "$condaBin" ]]; then
  echo "Conda is required; set CONDA_BIN to its absolute executable path." >&2
  exit 1
fi
condaBase="$("$condaBin" info --base)"
source "$condaBase/etc/profile.d/conda.sh"
if ! conda activate MARCH >/dev/null 2>&1; then
  "$condaBin" env create --file "$projectRoot/environment.yml"
  conda activate MARCH
fi
python -c 'import sys; assert sys.version_info[:2] == (3, 10), "MARCH requires Python 3.10"'
python -m pip install pip==25.1.1 setuptools==75.8.0 wheel==0.45.1
requirements="$projectRoot/requirements.lock.txt"
if [[ ! -f "$requirements" ]]; then
  requirements="$projectRoot/requirements.txt"
fi
python -m pip install --requirement "$requirements"
python -m pip check
python - <<'PY'
import os
import torch
import e3nn
import torch_geometric
import torch_scatter
import torch_cluster
import wandb
import spglib
from pymatgen.core import Structure
print(f"PyTorch {torch.__version__}; CUDA runtime {torch.version.cuda}; e3nn {e3nn.__version__}")
if os.environ.get("MARCH_SKIP_CUDA_CHECK") == "1":
    print("MARCH_SKIP_CUDA_CHECK=1: explicitly skipped CUDA runtime checks; CUDA remains unverified. "
          "Run this script without the override on an allocated GPU before training.")
    raise SystemExit(0)
if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable: verify the NVIDIA driver and GPU allocation.")
x = torch.randn(32, 32, device="cuda", requires_grad=True)
index = torch.arange(32, device="cuda") % 4
z = torch_scatter.scatter(x, index, dim=0, reduce="sum")
y = (x @ x.T).square().mean() + z.square().mean()
y.backward()
assert torch.isfinite(x.grad).all()
edges = torch_cluster.radius_graph(torch.randn(8, 3, device="cuda"), r=5.0)
assert edges.is_cuda and edges.shape[0] == 2
print(f"CUDA tensor/scatter forward/backward and cluster radius graph passed on {torch.cuda.get_device_name(0)}")
PY
printf 'Environment ready. Activate with: conda activate MARCH\n'