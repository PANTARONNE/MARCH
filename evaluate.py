"""Explicit holdout evaluation; no test material is used during training selection."""
import argparse
import json
from pathlib import Path
import torch
from march.training.configuration import loadConfig
from march.training.checkpoints import readCheckpoint
from march.training.distributed import initializeDistributed, finishDistributed, setSeed
from march.training.trainer import evaluate, readJson
from march.models.reconstructionModel import ReconstructionModel
from march.codec.socCodec import SocCodec
from march.losses.reconstructionLoss import ReconstructionLoss


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/pretrain.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", choices=["validation", "test"], default="validation")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    config = loadConfig(args.config)
    context = initializeDistributed(config["training"])
    try:
        setSeed(config["seed"])
        checkpoint = readCheckpoint(args.checkpoint)
        manifest = readJson(config["data"]["manifestPath"])
        if checkpoint["manifest"] != manifest:
            raise ValueError("Evaluation manifest must match the checkpoint split")
        records = manifest[args.split]
        if not records:
            raise ValueError(f"No records in {args.split} split")
        model = ReconstructionModel(checkpoint["config"]["model"]).to(context["device"])
        model.load_state_dict(checkpoint["model"])
        codec = SocCodec(device=context["device"])
        metrics, count = evaluate(model, records, codec, checkpoint["statistics"],
                                  ReconstructionLoss(codec), config, context)
        if context["rank"] == 0:
            result = {"split": args.split, "materials": count, **metrics}
            text = json.dumps(result, indent=2)
            print(text)
            if args.output:
                Path(args.output).write_text(text, encoding="utf-8")
    finally:
        finishDistributed()


if __name__ == "__main__":
    main()
