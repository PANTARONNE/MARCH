"""Atomic full-state checkpoints at optimizer boundaries."""
from __future__ import annotations
import os
import random
from pathlib import Path
import numpy as np
import torch
import torch.distributed as distributed
from march.training.distributed import unwrapModel


def captureRandomState():
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def restoreRandomState(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def readCheckpoint(path):
    # Only resume checkpoints generated locally by MARCH, never untrusted downloaded pickle files.
    return torch.load(str(path), map_location="cpu", weights_only=False)


def saveCheckpoint(path, model, optimizer, scheduler, progress, config, statistics,
                   manifest, context, runId):
    randomState = captureRandomState()
    if context["worldSize"] > 1:
        randomStates = [None] * context["worldSize"]
        distributed.all_gather_object(randomStates, randomState)
    else:
        randomStates = [randomState]
    if context["rank"] == 0:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"formatVersion": 1, "model": unwrapModel(model).state_dict(),
                   "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                   "progress": progress, "config": config, "statistics": statistics,
                   "manifest": manifest, "randomStates": randomStates,
                   "worldSize": context["worldSize"], "wandbRunId": runId}
        temporary = path.with_suffix(path.suffix + ".partial")
        torch.save(payload, temporary)
        os.replace(temporary, path)
    if context["worldSize"] > 1:
        distributed.barrier()
