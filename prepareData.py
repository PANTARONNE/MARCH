"""Audit and cache a bounded source pool, then group structures before splitting."""
from __future__ import annotations
import argparse
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from contextlib import nullcontext
import json
import multiprocessing
import random
from pathlib import Path
from tempfile import TemporaryDirectory
import torch
from march.training.configuration import loadConfig
from march.data.discovery import discoverSources, buildManifest
from march.data.pipeline import loadSample, cacheSample


geometryKeys = ("lattice", "positions", "atomicNumbers")


def prepareSource(source, cacheRoot, stagedPath):
    sample = loadSample(source)
    cachePath = cacheRoot / f"{sample['materialId']}.pt"
    cacheSample(sample, stagedPath, finalPath=cachePath)
    item = {key: sample[key] for key in ("materialId", "sourcePath", "cachePath", "audit")}
    # NumPy arrays travel by value, avoiding shared tensor descriptors per material.
    for key in geometryKeys:
        item[key] = sample[key].numpy().copy()
    return item


def preparedSources(sources, cacheRoot, requested, workers):
    if requested <= 0 or not sources:
        return
    workers = min(workers, requested, len(sources))
    pool = (ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context("spawn"),
                                initializer=torch.set_num_threads, initargs=(1,))
            if workers > 1 else nullcontext(None))
    cacheRoot.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=cacheRoot) as staging, pool as executor:
        pending = deque()
        nextIndex = 0
        remaining = requested
        while remaining:
            # Never start more files than could still be accepted.
            while len(pending) < min(workers, remaining) and nextIndex < len(sources):
                source = sources[nextIndex]
                stagedPath = Path(staging) / f"{nextIndex}.pt"
                future = executor.submit(prepareSource, source, cacheRoot, stagedPath) if executor else None
                pending.append((source, stagedPath, future))
                nextIndex += 1
            if not pending:
                break
            source, stagedPath, future = pending.popleft()
            try:
                item = future.result() if future else prepareSource(source, cacheRoot, stagedPath)
                # Publish in source order, including inputs with the same material ID.
                stagedPath.replace(item["cachePath"])
            except BrokenProcessPool:
                raise
            except (ValueError, OSError, RuntimeError) as error:
                yield None, {"sourcePath": str(source), "reason": str(error), "errorType": type(error).__name__}
                continue
            for key in geometryKeys:
                item[key] = torch.from_numpy(item[key])
            remaining -= 1
            yield item, None


def prepareData(config):
    dataConfig = config["data"]
    root = Path(dataConfig["sourceRoot"])
    cacheRoot = Path(dataConfig["cacheRoot"])
    manifestPath = Path(dataConfig["manifestPath"])
    manifestPath.parent.mkdir(parents=True, exist_ok=True)
    requested = sum(int(dataConfig[key]) for key in ("trainCount", "validationCount", "testCount"))
    explicit = dataConfig.get("sourceIds")
    sources = [root / f"{identity}.pth" for identity in explicit] if explicit else discoverSources(root)
    if not explicit:
        random.Random(int(config["seed"])).shuffle(sources)
    metadata, excluded = [], []
    workers = dataConfig.get("prepareWorkers", 1)
    if type(workers) is not int or workers < 1:
        raise ValueError("data.prepareWorkers must be a positive integer")
    for item, record in preparedSources(sources, cacheRoot, requested, workers):
        if record is not None:
            excluded.append(record)
            print(json.dumps({"excluded": record}), flush=True)
            continue
        metadata.append(item)
        print(json.dumps({"prepared": item["materialId"], "accepted": len(metadata),
                          "requested": requested, "audit": item["audit"]}), flush=True)
    report = {"requested": requested, "accepted": len(metadata), "excluded": excluded,
              "sourceRoot": str(root), "explicitSmokeSelection": bool(explicit)}
    (manifestPath.parent / "preparationAudit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    if len(metadata) != requested:
        raise RuntimeError(f"Only {len(metadata)}/{requested} eligible samples; see preparationAudit.json")
    manifest = buildManifest(metadata, trainCount=int(dataConfig["trainCount"]),
                             valCount=int(dataConfig["validationCount"]),
                             testCount=int(dataConfig["testCount"]), seed=int(config["seed"]))
    manifest["excluded"] = excluded
    manifest["sourceRoot"] = str(root)
    manifest["maskClosure"] = "conservativeGeometryCandidates"
    manifest["timeReversal"] = False
    manifest["explicitSmokeSelection"] = bool(explicit)
    manifestPath.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"manifest": str(manifestPath), "counts": manifest["actualCounts"]}), flush=True)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/pretrain.yaml")
    parser.add_argument("--sourceRoot")
    parser.add_argument("--cacheRoot")
    parser.add_argument("--workers", type=int, help="Concurrent files (overrides data.prepareWorkers, default 1)")
    args = parser.parse_args()
    torch.set_num_threads(2)
    config = loadConfig(args.config)
    if args.workers is not None:
        config["data"]["prepareWorkers"] = args.workers
    if args.sourceRoot:
        config["data"]["sourceRoot"] = args.sourceRoot
    if args.cacheRoot:
        config["data"]["cacheRoot"] = args.cacheRoot
    prepareData(config)


if __name__ == "__main__":
    main()


