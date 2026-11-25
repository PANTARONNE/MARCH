"""Train MARCH with Python or torchrun; target data never enters model.forward."""
import argparse
from march.training.configuration import loadConfig
from march.training.trainer import trainRun


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/pretrain.yaml")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--outputDir", default=None)
    parser.add_argument("--wandbMode", choices=["online", "offline", "disabled"])
    parser.add_argument("--maxOptimizerSteps", type=int)
    parser.add_argument("--device", choices=["cuda", "cpu"])
    args = parser.parse_args()
    config = loadConfig(args.config)
    if args.outputDir:
        config["outputDir"] = args.outputDir
    if args.wandbMode:
        config["logging"]["mode"] = args.wandbMode
    if args.maxOptimizerSteps is not None:
        config["training"]["maxOptimizerSteps"] = args.maxOptimizerSteps
    if args.device:
        config["training"]["device"] = args.device
    trainRun(config, args.resume)


if __name__ == "__main__":
    main()
