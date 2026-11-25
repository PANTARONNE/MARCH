"""Reuse timm's epoch scheduler with NextHAM's effective cosine defaults."""
from types import SimpleNamespace
from timm.scheduler import create_scheduler


def createScheduler(optimizer, config, epochs, seed=2026):
    args = SimpleNamespace(
        sched=config.get("name", "cosine"), epochs=int(epochs),
        lr_noise=None, lr_noise_pct=0.0, lr_noise_std=0.0, seed=int(seed),
        min_lr=float(config.get("minLearningRate", 1e-5)),
        warmup_lr=float(config.get("warmupLearningRate", 1e-6)),
        warmup_epochs=int(config.get("warmupEpochs", 5)),
        cooldown_epochs=0, decay_epochs=0, decay_rate=1.0, patience_epochs=0,
    )
    scheduler, _ = create_scheduler(args, optimizer)
    if scheduler is None:
        raise ValueError(f"Unsupported scheduler: {args.sched}")
    return scheduler
