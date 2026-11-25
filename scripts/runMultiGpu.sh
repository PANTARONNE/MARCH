#!/usr/bin/env bash
set -euo pipefail
projectRoot="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$projectRoot"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
config="${CONFIG:-configs/pretrain.yaml}"
nnodes="${NNODES:-1}"
nproc="${NPROC_PER_NODE:-$(python -c 'import torch; print(torch.cuda.device_count())')}"
if [[ "$nproc" -lt 1 ]]; then
    echo "No CUDA devices found; activate MARCH and check nvidia-smi." >&2
    exit 1
fi
if [[ "$nnodes" -eq 1 ]]; then
    torchrun --standalone --nnodes=1 --nproc_per_node="$nproc" train.py --config "$config" "$@"
else
    : "${NODE_RANK:?Set NODE_RANK for each node}"
    : "${MASTER_ADDR:?Set the rank-zero host address}"
    torchrun --nnodes="$nnodes" --nproc_per_node="$nproc" --node_rank="$NODE_RANK" \
        --master_addr="$MASTER_ADDR" --master_port="${MASTER_PORT:-29500}" \
        train.py --config "$config" "$@"
fi
