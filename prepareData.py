"""Audit and cache a bounded source pool, then group structures before splitting."""
from __future__ import annotations
import argparse
import json
import random
from pathlib import Path
import torch
from march.training.configuration import loadConfig
from march.data.discovery import discoverSources, buildManifest
from march.data.pipeline import loadSample, cacheSample


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
    for source in sources:
        if len(metadata) >= requested:
            break
        try:
            sample = loadSample(source)
            cachePath = cacheRoot / f"{sample['materialId']}.pt"
            cacheSample(sample, cachePath)
            item = {key: sample[key] for key in ("materialId", "sourcePath", "cachePath",
                                                 "lattice", "positions", "atomicNumbers", "audit")}
            for key in ("lattice", "positions", "atomicNumbers"):
                item[key] = item[key].clone()
            metadata.append(item)
            print(json.dumps({"prepared": sample["materialId"], "accepted": len(metadata),
                              "requested": requested, "audit": sample["audit"]}), flush=True)
            del sample
        except (ValueError, OSError, RuntimeError) as error:
            record = {"sourcePath": str(source), "reason": str(error), "errorType": type(error).__name__}
            excluded.append(record)
            print(json.dumps({"excluded": record}), flush=True)
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
    args = parser.parse_args()
    torch.set_num_threads(2)
    config = loadConfig(args.config)
    if args.sourceRoot:
        config["data"]["sourceRoot"] = args.sourceRoot
    if args.cacheRoot:
        config["data"]["cacheRoot"] = args.cacheRoot
    prepareData(config)


if __name__ == "__main__":
    main()


