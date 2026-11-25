"""Compare real four-layer TraceGrad DDP updates with serial material averaging.

Run: torchrun --standalone --nproc_per_node=2 tests/ddpConsistency.py
Use --device cuda on a server with two GPUs for the corresponding NCCL check.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import torch
import torch.distributed as distributed
from torch.nn.parallel import DistributedDataParallel
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from march.models.reconstructionModel import ReconstructionModel


def makeInput(index, device):
    generator = torch.Generator().manual_seed(100 + index)
    qH = torch.randn(4, 5832, generator=generator) * .2
    hidden = torch.tensor([False, True, False, False])
    qH[hidden] = 0
    return {"qH": qH.to(device), "qS": (torch.randn(4, 5832, generator=generator) * .1).to(device),
            "hiddenMask": hidden.to(device), "support": torch.ones(4, 54, 54, dtype=torch.bool, device=device),
            "edgeIndex": torch.tensor([[0, 1, 0, 1], [0, 1, 1, 0]], device=device),
            "edgeVec": torch.tensor([[0., 0., 0.], [0., 0., 0.], [1., 2., 3.], [-1., -2., -3.]], device=device),
            "atomicNumbers": torch.tensor([6, 8], device=device)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--output", default="reports/cpuDdpConsistency.json")
    args = parser.parse_args()
    torch.set_num_threads(2)
    rank = int(os.environ["RANK"])
    worldSize = int(os.environ["WORLD_SIZE"])
    localRank = int(os.environ["LOCAL_RANK"])
    device = torch.device("cuda", localRank) if args.device == "cuda" else torch.device("cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    distributed.init_process_group("nccl" if device.type == "cuda" else "gloo")
    try:
        torch.manual_seed(42)
        model = ReconstructionModel({"preset": "smoke"}).to(device)
        serial = ReconstructionModel({"preset": "smoke"}).to(device) if rank == 0 else None
        if serial is not None:
            serial.load_state_dict(model.state_dict())
        ddp = DistributedDataParallel(model, device_ids=[localRank] if device.type == "cuda" else None,
                                       find_unused_parameters=True)
        optimizer = torch.optim.Adam((p for p in ddp.parameters() if p.requires_grad), lr=5e-4)
        serialOptimizer = torch.optim.Adam((p for p in serial.parameters() if p.requires_grad), lr=5e-4) if rank == 0 else None
        errors = []
        for step in range(2):
            optimizer.zero_grad(set_to_none=True)
            output = ddp(makeInput(rank + step * worldSize, device))["qPrediction"]
            output.square().mean().backward()
            if rank == 0:
                serialOptimizer.zero_grad(set_to_none=True)
                for material in range(worldSize):
                    expected = serial(makeInput(material + step * worldSize, device))["qPrediction"]
                    (expected.square().mean() / worldSize).backward()
                maximum = 0.0
                for (name, parameter), (otherName, expectedParameter) in zip(model.named_parameters(), serial.named_parameters()):
                    assert name == otherName
                    assert (parameter.grad is None) == (expectedParameter.grad is None), name
                    if parameter.grad is not None:
                        torch.testing.assert_close(parameter.grad, expectedParameter.grad,
                                                   rtol=4e-4, atol=2e-6, msg=name)
                        maximum = max(maximum, float((parameter.grad - expectedParameter.grad).abs().max()))
                errors.append(maximum)
            optimizer.step()
            if rank == 0:
                serialOptimizer.step()
                for (name, parameter), (_, expectedParameter) in zip(model.named_parameters(), serial.named_parameters()):
                    torch.testing.assert_close(parameter, expectedParameter, rtol=3e-4, atol=2e-5, msg=name)
            distributed.barrier()
        if rank == 0:
            result = {"passed": True, "device": args.device, "worldSize": worldSize,
                      "optimizerSteps": 2, "maximumGradientErrors": errors,
                      "scope": "full four-layer SOC model including TraceGrad versus serial material averaging"}
            path = Path(args.output)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(result, indent=2), encoding="utf-8")
            print(json.dumps(result), flush=True)
    finally:
        distributed.destroy_process_group()


if __name__ == "__main__":
    main()
