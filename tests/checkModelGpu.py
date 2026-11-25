"""One full real crystal: main-width CUDA forward, backward and Adam step.

Run after preparing the bounded smoke manifest, from the project root:
python tests/checkModelGpu.py --config configs/smoke.yaml --report reports/mainGpu.json
"""
import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import e3nn
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
    parser.add_argument("--preset", choices=("main", "smoke"), default="main")
    parser.add_argument("--report", default="reports/mainGpu.json")
    args = parser.parse_args()
    torch.set_num_threads(2)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this acceptance check")
    config = loadConfig(args.config)
    setSeed(config["seed"])
    manifest = json.loads(Path(config["data"]["manifestPath"]).read_text())
    statistics = json.loads(Path(config["data"]["statisticsPath"]).read_text())
    record = manifest["train"][0]
    sample = loadSample(record.get("cachePath") or record["sourcePath"])
    started = time.monotonic()
    codec = SocCodec(device="cuda")
    modelConfig = dict(config["model"], preset=args.preset)
    model = ReconstructionModel(modelConfig).cuda().train()
    optimizer = createOptimizer(model, config["optimizer"])
    modelInput, target = prepareSample(sample, codec, statistics,
        seed=config["seed"], epoch=0, maskRatio=config["data"]["maskRatio"])
    torch.cuda.reset_peak_memory_stats()
    prediction = model(modelInput)["qPrediction"]
    lossFunction = ReconstructionLoss(codec)
    loss = lossFunction(prediction, target)
    loss.backward()
    if not torch.isfinite(loss):
        raise FloatingPointError("Loss is nonfinite")
    if not all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):
        raise FloatingPointError("A parameter gradient is nonfinite")
    finalNodeGradient = sum(float(p.grad.abs().sum()) for p in model.backbone.blocks[-1].ffn.parameters()
                            if p.grad is not None)
    if finalNodeGradient <= 0:
        raise RuntimeError("Final node features received no reconstruction supervision")
    optimizer.step()
    torch.cuda.synchronize()
    report = {
        "status": "passed", "preset": args.preset, "materialId": sample["materialId"],
        "nodes": len(sample["atomicNumbers"]), "edges": sample["edgeIndex"].shape[1],
        "allSavedValidBlocksUsed": True, "hiddenWidth": model.hiddenIrreps.dim,
        "outputDimension": model.outputIrreps.dim,
        "loss": float(loss.detach()), "actualMaskRatio": target["actualMaskRatio"],
        "finiteGradients": True, "finalNodeGradientL1": finalNodeGradient,
        "peakMemoryMiB": torch.cuda.max_memory_allocated() / 2 ** 20,
        "seconds": time.monotonic() - started, "device": torch.cuda.get_device_name(),
        "torchVersion": torch.__version__, "cudaVersion": torch.version.cuda,
        "e3nnVersion": e3nn.__version__,
    }
    reportPath = Path(args.report)
    reportPath.parent.mkdir(parents=True, exist_ok=True)
    reportPath.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
