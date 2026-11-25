"""Explicit, replaceable optimizer settings derived from NextHAM's live path."""
from __future__ import annotations
import torch

OPTIMIZERS = {"adam": torch.optim.Adam, "adamw": torch.optim.AdamW, "sgd": torch.optim.SGD}


def createOptimizer(model, config):
    name = config.get("name", "adam").lower()
    if name not in OPTIMIZERS:
        raise ValueError(f"Unknown optimizer {name!r}; choices: {list(OPTIMIZERS)}")
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        raise ValueError("No trainable parameters")
    options = {"lr": float(config.get("learningRate", 5e-4)),
               "weight_decay": float(config.get("weightDecay", 0.0))}
    if name in ("adam", "adamw"):
        options.update(betas=tuple(config.get("betas", [0.9, 0.999])),
                       eps=float(config.get("epsilon", 1e-8)))
    else:
        options["momentum"] = float(config.get("momentum", 0.9))
    return OPTIMIZERS[name](parameters, **options)
