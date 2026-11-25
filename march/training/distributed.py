"""Small torchrun-compatible distributed boundary."""
from __future__ import annotations
import os
import random
import numpy as np
import torch
import torch.distributed as distributed


def initializeDistributed(config):
    worldSize = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    localRank = int(os.environ.get("LOCAL_RANK", "0"))
    requested = config.get("device", "cuda")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable; use device: cpu explicitly for CPU tests")
        torch.cuda.set_device(localRank)
        device = torch.device("cuda", localRank)
    elif requested == "cpu":
        device = torch.device("cpu")
    else:
        raise ValueError("training.device must be cuda or cpu")
    if worldSize > 1:
        backend = config.get("distributedBackend", "nccl") if device.type == "cuda" else "gloo"
        distributed.init_process_group(backend=backend)
    return {"rank": rank, "worldSize": worldSize, "localRank": localRank, "device": device}


def setSeed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def unwrapModel(model):
    return model.module if hasattr(model, "module") else model


def reduceMetrics(sums, count, context):
    # All ranks use identical metric names, even when their validation shard is empty.
    keys = sorted(sums)
    packed = torch.tensor([float(sums[key]) for key in keys] + [float(count)],
                          dtype=torch.float64, device=context["device"])
    if context["worldSize"] > 1:
        distributed.all_reduce(packed)
    total = packed[-1].item()
    if total == 0:
        return {key: 0.0 for key in keys}, 0
    return {key: packed[i].item() / total for i, key in enumerate(keys)}, int(total)


def finishDistributed():
    if distributed.is_initialized():
        distributed.destroy_process_group()
