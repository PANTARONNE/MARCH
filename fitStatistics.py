"""Fit scale-only normalization exclusively on the manifest training partition."""
import argparse
import json
from pathlib import Path
import torch
from march.codec.socCodec import SocCodec
from march.data.pipeline import loadSample, fitStatistics
from march.training.configuration import loadConfig


def trainingSamples(manifest):
    for record in manifest["train"]:
        sample = loadSample(record.get("cachePath") or record["sourcePath"])
        sample["split"] = "train"
        yield sample


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/pretrain.yaml")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    args = parser.parse_args()
    torch.set_num_threads(2)
    config = loadConfig(args.config)
    manifest = json.loads(Path(config["data"]["manifestPath"]).read_text(encoding="utf-8"))
    statistics = fitStatistics(trainingSamples(manifest), SocCodec(device=args.device),
                               seed=int(config["seed"]), maskRatio=float(config["data"]["maskRatio"]))
    destination = Path(config["data"]["statisticsPath"])
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(statistics, indent=2), encoding="utf-8")
    print(json.dumps({"statistics": str(destination), "trainingMaterials": len(statistics["trainIds"])}))


if __name__ == "__main__":
    main()
