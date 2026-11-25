"""Per-crystal training, material-balanced accumulation, DDP and restart support."""
from __future__ import annotations
import contextlib
import copy
import json
import math
import time
from pathlib import Path
import torch
import torch.distributed as distributed
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset
from march.codec.socCodec import SocCodec
from march.models.reconstructionModel import ReconstructionModel
from march.data.pipeline import loadSample, prepareSample
from march.losses.reconstructionLoss import ReconstructionLoss
from march.optim.optimizerFactory import createOptimizer
from march.optim.schedulerFactory import createScheduler
from march.logging.experimentLogger import ExperimentLogger
from march.training.configuration import saveConfig
from march.training.distributed import initializeDistributed, setSeed, unwrapModel, reduceMetrics, finishDistributed
from march.training.checkpoints import readCheckpoint, saveCheckpoint, restoreRandomState


class RecordDataset(Dataset):
    def __init__(self, records):
        self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        return loadSample(record.get("cachePath") or record["sourcePath"])


def singleSample(samples):
    if len(samples) != 1:
        raise ValueError("The graph training loader expects exactly one material per microbatch")
    return samples[0]


def makeLoader(records, config):
    return DataLoader(RecordDataset(records), batch_size=1, shuffle=False,
                      num_workers=int(config.get("numWorkers", 0)), collate_fn=singleSample,
                      generator=torch.Generator().manual_seed(0))


def readJson(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def evaluate(model, records, codec, statistics, lossFunction, config, context):
    model = unwrapModel(model)
    model.eval()
    localRecords = records[context["rank"]::context["worldSize"]]
    sums = {"loss": 0.0, "actualMaskRatio": 0.0, "rawMae": 0.0}
    count = 0
    # TraceGrad computes feature derivatives even in eval: no global no_grad/inference_mode.
    with torch.enable_grad():
        for sample in makeLoader(localRecords, config["training"]):
            modelInput, target = prepareSample(sample, codec, statistics,
                seed=int(config["data"]["validationMaskSeed"]), epoch=0,
                maskRatio=float(config["data"]["maskRatio"]))
            output = model(modelInput)
            loss = lossFunction(output["qPrediction"], target)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite validation loss: {sample['materialId']}")
            sums["rawMae"] += lossFunction.metrics(output["qPrediction"].detach(), target)["rawMae"]
            sums["loss"] += float(loss.detach())
            sums["actualMaskRatio"] += float(modelInput["hiddenMask"].float().mean())
            count += 1
            del output, loss, modelInput, target
    return reduceMetrics(sums, count, context)


def trainRun(config, resumePath=None):
    context = initializeDistributed(config["training"])
    logger = None
    success = False
    started = time.monotonic()
    try:
        if int(config["training"].get("perDeviceBatchSize", 1)) != 1:
            raise ValueError("Use perDeviceBatchSize=1 and gradientAccumulation for variable-size crystal graphs")
        if config["training"].get("precision", "float32") != "float32":
            raise ValueError("Initial SOC/TraceGrad training is validated in float32 only")
        setSeed(int(config["seed"]))
        manifest = readJson(config["data"]["manifestPath"])
        statistics = readJson(config["data"]["statisticsPath"])
        trainRecords, validationRecords = manifest["train"], manifest["validation"]
        expectedIds = sorted(record["materialId"] for record in trainRecords)
        if sorted(statistics["trainIds"]) != expectedIds:
            raise ValueError("Frozen scale statistics were not fitted on this exact training partition")
        groupSets = [{record["groupId"] for record in manifest[name]} for name in ("train", "validation", "test")]
        if any(groupSets[i] & groupSets[j] for i in range(3) for j in range(i)):
            raise ValueError("A structure group crosses dataset partitions")
        if len(trainRecords) < context["worldSize"] or not validationRecords:
            raise ValueError("Need at least one training material per rank and a nonempty validation set")
        codec = SocCodec(device=context["device"], dtype=torch.float32)
        model = ReconstructionModel(config["model"]).to(context["device"])
        optimizer = createOptimizer(model, config["optimizer"])
        scheduler = createScheduler(optimizer, config["scheduler"], config["training"]["epochs"], config["seed"])
        progress = {"epoch": 0, "nextSampleOffset": 0, "globalStep": 0, "bestValidationLoss": math.inf}
        checkpoint = readCheckpoint(resumePath) if resumePath else None
        if checkpoint:
            previousConfig = copy.deepcopy(checkpoint["config"])
            currentConfig = copy.deepcopy(config)
            for candidate in (previousConfig, currentConfig):
                for key in ("outputDir", "runName", "logging"):
                    candidate.pop(key, None)
                candidate["training"].pop("maxOptimizerSteps", None)
            if previousConfig != currentConfig:
                raise ValueError("Exact resume permits only output/logging/maxOptimizerSteps changes; training semantics differ")
            if checkpoint["statistics"] != statistics or checkpoint["manifest"] != manifest:
                raise ValueError("Resume data split/statistics mismatch")
            if checkpoint["worldSize"] != context["worldSize"]:
                raise ValueError("Exact resume requires the same worldSize as the saved checkpoint")
            model.load_state_dict(checkpoint["model"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            scheduler.load_state_dict(checkpoint["scheduler"])
            progress.update(checkpoint["progress"])
        if context["worldSize"] > 1:
            model = DistributedDataParallel(model,
                device_ids=[context["localRank"]] if context["device"].type == "cuda" else None,
                find_unused_parameters=bool(config["training"].get("findUnusedParameters", True)))
        logger = ExperimentLogger(config, config["outputDir"], context["rank"],
                                  checkpoint.get("wandbRunId") if checkpoint else None)
        if checkpoint:
            restoreRandomState(checkpoint["randomStates"][context["rank"]])
        outputDir = Path(config["outputDir"])
        if context["rank"] == 0:
            saveConfig(config, outputDir / "resolvedConfig.yaml")
            (outputDir / "runMetadata.json").write_text(json.dumps({
                "torchVersion": torch.__version__, "cudaVersion": torch.version.cuda,
                "device": str(context["device"]), "worldSize": context["worldSize"],
                "parameters": sum(p.numel() for p in model.parameters()),
                "trainableParameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
                "trainMaterials": len(trainRecords), "validationMaterials": len(validationRecords),
                "precision": "float32", "testUsedForSelection": False,
            }, indent=2), encoding="utf-8")
        lossFunction = ReconstructionLoss(codec)
        accumulation = int(config["training"].get("gradientAccumulation", 1))
        if accumulation < 1:
            raise ValueError("gradientAccumulation must be positive")
        maximum = config["training"].get("maxOptimizerSteps")
        if int(config["training"]["epochs"]) < 1 or (maximum is not None and int(maximum) < 1):
            raise ValueError("epochs and maxOptimizerSteps must be positive")
        stop = maximum is not None and progress["globalStep"] >= int(maximum)
        firstEpoch = int(progress["epoch"])
        for epoch in range(firstEpoch, int(config["training"]["epochs"])):
            if stop:
                break
            model.train()
            scheduler.step(epoch)
            generator = torch.Generator().manual_seed(int(config["seed"]) + epoch)
            order = torch.randperm(len(trainRecords), generator=generator).tolist()
            # Drop only the epoch's tail to keep ranks synchronized without duplicating materials.
            usable = len(order) // context["worldSize"] * context["worldSize"]
            rankOrder = order[:usable][context["rank"]::context["worldSize"]]
            offset = int(progress["nextSampleOffset"]) if epoch == firstEpoch else 0
            records = [trainRecords[index] for index in rankOrder[offset:]]
            optimizer.zero_grad(set_to_none=True)
            windowSums = {"loss": 0.0, "actualMaskRatio": 0.0, "rawMae": 0.0}
            windowCount = 0
            for localIndex, sample in enumerate(makeLoader(records, config["training"])):
                position = offset + localIndex
                windowStart = (localIndex // accumulation) * accumulation
                windowSize = min(accumulation, len(records) - windowStart)
                updateNow = (localIndex - windowStart + 1) == windowSize
                sync = contextlib.nullcontext() if updateNow or context["worldSize"] == 1 else model.no_sync()
                with sync:
                    modelInput, target = prepareSample(sample, codec, statistics,
                        seed=int(config["seed"]), epoch=epoch, maskRatio=float(config["data"]["maskRatio"]))
                    prediction = model(modelInput)["qPrediction"]
                    loss = lossFunction(prediction, target)
                    if not torch.isfinite(loss):
                        raise FloatingPointError(f"Nonfinite training loss: {sample['materialId']}")
                    (loss / windowSize).backward()
                windowSums["rawMae"] += lossFunction.metrics(prediction.detach(), target)["rawMae"]
                windowSums["loss"] += float(loss.detach())
                windowSums["actualMaskRatio"] += float(modelInput["hiddenMask"].float().mean())
                windowCount += 1
                del loss, prediction, modelInput, target
                if not updateNow:
                    continue
                clip = config["training"].get("clipGradientNorm")
                gradientNorm = torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    float(clip) if clip is not None else math.inf, error_if_nonfinite=True)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                progress.update(epoch=epoch, nextSampleOffset=position + 1,
                                globalStep=int(progress["globalStep"]) + 1)
                metrics, materialCount = reduceMetrics(windowSums, windowCount, context)
                logger.log({**{f"train/{key}": value for key, value in metrics.items()},
                            "train/gradientNorm": float(gradientNorm), "train/materialsInStep": materialCount,
                            "train/epoch": epoch, "train/epochTailDropped": len(order) - usable,
                            "train/learningRate": optimizer.param_groups[0]["lr"],
                            "system/maxMemoryMiB": torch.cuda.max_memory_allocated() / 2**20
                               if context["device"].type == "cuda" else 0.0}, progress["globalStep"])
                windowSums = {"loss": 0.0, "actualMaskRatio": 0.0, "rawMae": 0.0}
                windowCount = 0
                if maximum is not None and progress["globalStep"] >= int(maximum):
                    stop = True
                    break
            if progress["nextSampleOffset"] >= len(rankOrder):
                progress.update(epoch=epoch + 1, nextSampleOffset=0)
            validation, count = evaluate(model, validationRecords, codec, statistics, lossFunction, config, context)
            logger.log({**{f"validation/{key}": value for key, value in validation.items()},
                        "validation/materialCount": count}, progress["globalStep"])
            improved = validation["loss"] < progress["bestValidationLoss"]
            if improved:
                progress["bestValidationLoss"] = validation["loss"]
            for filename in (["latest.pt", "best.pt"] if improved else ["latest.pt"]):
                saveCheckpoint(outputDir / filename, model, optimizer, scheduler, dict(progress),
                               config, statistics, manifest, context, logger.runId)
        if context["rank"] == 0:
            summary = {"status": "completed", **progress, "elapsedSeconds": time.monotonic() - started,
                       "worldSize": context["worldSize"], "device": str(context["device"])}
            (outputDir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        success = True
        return progress
    finally:
        if logger is not None:
            logger.finish(success)
        finishDistributed()



