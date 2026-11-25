"""Verify effective optimizer options and restart state, not just config spelling."""
import random
import numpy as np
import torch
from march.optim.optimizerFactory import createOptimizer
from march.optim.schedulerFactory import createScheduler
from march.training.configuration import loadConfig
from march.training.checkpoints import captureRandomState, restoreRandomState, saveCheckpoint, readCheckpoint


def testNextHamEffectivePresetAndOverrides():
    config = loadConfig("configs/pretrain.yaml")
    model = torch.nn.Linear(3, 1)
    optimizer = createOptimizer(model, config["optimizer"])
    assert isinstance(optimizer, torch.optim.Adam)
    assert optimizer.defaults["lr"] == 5e-4
    assert optimizer.defaults["betas"] == (0.9, 0.999)
    assert optimizer.defaults["eps"] == 1e-8
    assert optimizer.defaults["weight_decay"] == 0
    changed = createOptimizer(model, {"name": "adamw", "learningRate": .002,
                                      "weightDecay": .1, "betas": [.8, .95], "epsilon": 1e-7})
    assert isinstance(changed, torch.optim.AdamW)
    assert changed.defaults["weight_decay"] == .1
    assert changed.defaults["betas"] == (.8, .95)
    scheduler = createScheduler(optimizer, config["scheduler"], 100)
    scheduler.step(0)
    assert optimizer.param_groups[0]["lr"] == 1e-6
    scheduler.step(100)
    assert abs(optimizer.param_groups[0]["lr"] - 1e-5) < 1e-12


def testCheckpointRestoresOptimizerAndRandomState(tmp_path):
    torch.manual_seed(1)
    model = torch.nn.Linear(3, 1)
    optimizer = createOptimizer(model, {"name": "adam"})
    scheduler = createScheduler(optimizer, {"warmupEpochs": 0}, 10)
    x = torch.randn(2, 3)
    model(x).square().mean().backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    path = tmp_path / "latest.pt"
    context = {"rank": 0, "worldSize": 1, "device": torch.device("cpu")}
    saveCheckpoint(path, model, optimizer, scheduler, {"epoch": 1, "globalStep": 1},
                   {}, {"trainIds": ["mp-1"]}, {"train": []}, context, "testRun")
    saved = readCheckpoint(path)
    second = torch.nn.Linear(3, 1)
    second.load_state_dict(saved["model"])
    secondOptimizer = createOptimizer(second, {"name": "adam"})
    secondOptimizer.load_state_dict(saved["optimizer"])
    torch.testing.assert_close(second(x), model(x), atol=0, rtol=0)
    for current in (model, second):
        current(x).square().mean().backward()
    optimizer.step()
    secondOptimizer.step()
    for first, restored in zip(model.parameters(), second.parameters()):
        torch.testing.assert_close(first, restored, atol=0, rtol=0)
    state = captureRandomState()
    expected = (random.random(), np.random.rand(), torch.rand(3))
    restoreRandomState(state)
    observed = (random.random(), np.random.rand(), torch.rand(3))
    assert observed[0] == expected[0] and observed[1] == expected[1]
    torch.testing.assert_close(observed[2], expected[2], atol=0, rtol=0)
