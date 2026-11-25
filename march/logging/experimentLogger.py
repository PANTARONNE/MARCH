"""Rank-zero online/offline W&B plus durable local JSONL metrics."""
from __future__ import annotations
import json
import time
from pathlib import Path


class ExperimentLogger:
    def __init__(self, config, outputDir, rank=0, runId=None):
        self.rank = rank
        self.run = None
        self.handle = None
        self.outputDir = Path(outputDir)
        if rank != 0:
            return
        self.outputDir.mkdir(parents=True, exist_ok=True)
        self.handle = (self.outputDir / "metrics.jsonl").open("a", encoding="utf-8", buffering=1)
        options = config["logging"]
        if options.get("enabled", True):
            import wandb
            mode = options.get("mode", "online")
            if mode not in ("online", "offline", "disabled"):
                raise ValueError("logging.mode must be online, offline, or disabled")
            self.run = wandb.init(
                project=options.get("project", "MARCH"), entity=options.get("entity"),
                name=config.get("runName"), mode=mode, dir=str(self.outputDir),
                config=config, id=runId, resume="allow" if runId and mode == "online" else None,
                settings=wandb.Settings(init_timeout=options.get("initTimeout", 60)),
            )
            (self.outputDir / "wandbRun.json").write_text(json.dumps({
                "id": self.run.id, "url": self.run.url if mode == "online" else None,
                "mode": mode, "project": options.get("project", "MARCH")}, indent=2), encoding="utf-8")

    @property
    def runId(self):
        return self.run.id if self.run is not None else None

    def log(self, metrics, step):
        if self.rank != 0:
            return
        record = {"step": int(step), "wallTime": time.time(), **metrics}
        self.handle.write(json.dumps(record, allow_nan=False) + "\n")
        print(json.dumps(record, allow_nan=False), flush=True)
        if self.run is not None:
            self.run.log(metrics, step=int(step))

    def finish(self, success=True):
        if self.handle is not None:
            self.handle.close()
            self.handle = None
        if self.run is not None:
            self.run.finish(exit_code=0 if success else 1)
            self.run = None
