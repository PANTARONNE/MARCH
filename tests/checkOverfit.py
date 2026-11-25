"""Bounded fixed-real-crystal optimization smoke test, not a generalization test."""
import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from march.codec import SocCodec
from march.models import ReconstructionModel
from march.data.pipeline import loadSample, prepareSample
from march.losses.reconstructionLoss import ReconstructionLoss
from march.optim.optimizerFactory import createOptimizer
from march.training.configuration import loadConfig
from march.training.distributed import setSeed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/smoke.yaml")
    parser.add_argument("--materialId", default="mp-1")
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--report", default="reports/overfitGpu.json")
    args = parser.parse_args()
    if not 20 <= args.steps <= 30:
        raise ValueError("This bounded smoke test requires 20 to 30 optimizer steps")
    if not torch.cuda.is_available():
        raise RuntimeError("The overfit acceptance test requires CUDA")
    torch.set_num_threads(2)
    config = loadConfig(args.config)
    setSeed(int(config["seed"]))
    manifest = json.loads(Path(config["data"]["manifestPath"]).read_text())
    statistics = json.loads(Path(config["data"]["statisticsPath"]).read_text())
    record = next(item for item in manifest["train"] if item["materialId"] == args.materialId)
    sample = loadSample(record.get("cachePath") or record["sourcePath"])
    started = time.monotonic()
    codec = SocCodec(device="cuda")
    model = ReconstructionModel(dict(config["model"], preset="smoke")).cuda().train()
    optimizer = createOptimizer(model, config["optimizer"])
    lossFunction = ReconstructionLoss(codec)
    # Prepare once: material, mask, visible reference and normalized inputs stay fixed.
    modelInput, target = prepareSample(sample, codec, statistics,
        seed=int(config["seed"]), epoch=0, maskRatio=float(config["data"]["maskRatio"]))
    frozenMask = modelInput["hiddenMask"].clone()
    losses = []
    gradientNorms = []
    torch.cuda.reset_peak_memory_stats()
    for step in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        prediction = model(modelInput)["qPrediction"]
        loss = lossFunction(prediction, target)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Nonfinite loss at step {step}")
        losses.append(float(loss.detach()))
        loss.backward()
        gradientNorm = torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad],
            float("inf"), error_if_nonfinite=True)
        gradientNorms.append(float(gradientNorm))
        optimizer.step()
        if not torch.equal(modelInput["hiddenMask"], frozenMask):
            raise AssertionError("Mask changed during fixed-sample optimization")
        print(json.dumps({"step": step, "loss": losses[-1], "gradientNorm": gradientNorms[-1]}), flush=True)
        del prediction, loss
    optimizer.zero_grad(set_to_none=True)
    finalLoss = lossFunction(model(modelInput)["qPrediction"], target)
    if not torch.isfinite(finalLoss):
        raise FloatingPointError("Nonfinite final loss")
    losses.append(float(finalLoss.detach()))
    torch.cuda.synchronize()
    improved = losses[-1] < losses[0]
    report = {
        "status": "passed" if improved else "failed",
        "purpose": "Fixed-real-sample loss descent only; this is not evidence of generalization.",
        "materialId": sample["materialId"], "preset": "smoke", "hiddenWidth": model.hiddenIrreps.dim,
        "transformerLayers": len(model.backbone.blocks), "optimizerSteps": args.steps,
        "seed": int(config["seed"]), "maskEpoch": 0,
        "hiddenIndices": frozenMask.nonzero().flatten().cpu().tolist(),
        "actualMaskRatio": target["actualMaskRatio"], "nodes": len(sample["atomicNumbers"]),
        "edges": sample["edgeIndex"].shape[1], "initialLoss": losses[0], "finalLoss": losses[-1],
        "relativeReduction": (losses[0] - losses[-1]) / max(losses[0], 1e-30),
        "losses": losses, "finiteGradients": True, "gradientNorms": gradientNorms,
        "optimizer": config["optimizer"], "device": torch.cuda.get_device_name(),
        "torchVersion": torch.__version__, "cudaVersion": torch.version.cuda,
        "peakMemoryMiB": torch.cuda.max_memory_allocated() / 2 ** 20,
        "elapsedSeconds": time.monotonic() - started,
    }
    reportPath = Path(args.report)
    reportPath.parent.mkdir(parents=True, exist_ok=True)
    reportPath.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items()
                      if key not in ("losses", "gradientNorms")}, indent=2))
    if not improved:
        raise AssertionError("Fixed-sample reconstruction loss did not decrease")


if __name__ == "__main__":
    main()
