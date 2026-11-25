"""Regression tests for geometry-only masks and isolated visible inputs."""
import copy
import os
from pathlib import Path
import pytest
import torch
from march.codec import SocCodec
from march.data.geometry import buildGeometry, verifyLossOrbits, sampleMask, auditSupport
from march.data.pipeline import (chooseReference, fitStatistics, officialRotate, prepareSample,
                                 loadSample, cacheSample, scaleTensors)
from march.data.discovery import buildManifest


@pytest.fixture(scope="module")
def codec():
    torch.set_num_threads(2)
    return SocCodec()


def makeSample(materialId="mp-101"):
    lattice = torch.diag(torch.tensor([3., 4., 5.]))
    translations = torch.tensor([[0, 0, 0], [1, 0, 0], [-1, 0, 0],
                                 [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]])
    edges = len(translations)
    support = torch.zeros(edges, 54, 54, dtype=torch.bool)
    for a in (0, 27):
        for b in (0, 27):
            support[:, a, b] = True
    overlap = torch.zeros(edges, 54, 54, dtype=torch.complex64)
    overlap[:, 0, 0], overlap[:, 27, 27] = 1, 1
    hamiltonian = overlap.clone()
    hamiltonian[:, 0, 0] = torch.tensor([2., 3., 3., 4., 4., 5., 5.])
    hamiltonian[:, 27, 27] = hamiltonian[:, 0, 0]
    sample = dict(schemaVersion=1, materialId=materialId, sourcePath=f"/synthetic/{materialId}.pth",
                  lattice=lattice, positions=torch.zeros(1, 3), atomicNumbers=torch.tensor([1]),
                  edgeIndex=torch.zeros(2, edges, dtype=torch.long),
                  edgeVec=translations.float() @ lattice,
                  support=support, hamiltonian=hamiltonian, overlap=overlap)
    geometry, base, candidates = buildGeometry(sample)
    sample.update(geometry)
    sample["lossOrbit"] = verifyLossOrbits(sample, base, candidates, officialRotate)
    return sample


def unitStatistics():
    return {"scaleFloor": 1e-6, "h": {"groups": {}, "global": 1.0},
            "s": {"groups": {}, "global": 1.0}}


def testMaskClosureClosestAndReproducible():
    groups = torch.tensor([0, 1, 1, 2, 2, 2, 2, 3, 3])
    hidden = sampleMask(groups, "mp-9", seed=2026, epoch=7)
    assert int(hidden.sum()) == 3
    assert torch.equal(hidden, sampleMask(groups, "mp-9", seed=2026, epoch=7))
    for group in groups.unique():
        assert hidden[groups == group].unique().numel() == 1
    masks = {tuple(sampleMask(groups, "mp-9", epoch=i).tolist()) for i in range(32)}
    assert len(masks) > 1
    with pytest.raises(ValueError, match="fewer than two"):
        sampleMask(torch.zeros(4, dtype=torch.long), "single")


def testGeometryNeverReadsTargetsAndClosesSymmetry():
    sample = makeSample()
    first, _, _ = buildGeometry(sample)
    sample["hamiltonian"] = None
    sample["overlap"] = None
    second, _, _ = buildGeometry(sample)
    for key in first:
        assert torch.equal(first[key], second[key])
    assert first["maskGroup"][1] == first["maskGroup"][2]
    assert first["maskGroup"][3] == first["maskGroup"][4]
    assert first["maskGroup"][1] != first["maskGroup"][3]


def testHiddenTargetIsolationIncludesLossOrbitChanges(codec):
    sample = makeSample()
    inputs, target = prepareSample(sample, codec, unitStatistics())
    changed = copy.deepcopy(sample)
    hiddenIndices = torch.where(target["hiddenMask"].cpu())[0]
    changed["hamiltonian"][hiddenIndices[0], 0, 27] += 19 + 3j
    geometry, base, candidates = buildGeometry(changed)
    changed.update(geometry)
    changed["lossOrbit"] = verifyLossOrbits(changed, base, candidates, officialRotate)
    changedInputs, changedTarget = prepareSample(changed, codec, unitStatistics())
    assert not torch.equal(sample["lossOrbit"], changed["lossOrbit"])
    for key in inputs:
        assert torch.equal(inputs[key], changedInputs[key]), key
    assert not torch.equal(target["qTarget"], changedTarget["qTarget"])
    assert inputs["qH"][inputs["hiddenMask"]].abs().max() == 0
    assert set(inputs) == {"qH", "qS", "hiddenMask", "support", "edgeIndex", "edgeVec", "atomicNumbers"}


def testVisibleEnergyGaugeAndSOnlyIsolation(codec):
    sample = makeSample()
    inputs, target = prepareSample(sample, codec, unitStatistics())
    shifted = copy.deepcopy(sample)
    shifted["hamiltonian"] += 7.5 * shifted["overlap"]
    shiftedInputs, shiftedTarget = prepareSample(shifted, codec, unitStatistics())
    torch.testing.assert_close(inputs["qH"], shiftedInputs["qH"], atol=2e-6, rtol=1e-5)
    torch.testing.assert_close(target["qTarget"], shiftedTarget["qTarget"], atol=2e-6, rtol=1e-5)
    torch.testing.assert_close(shiftedTarget["reference"], target["reference"] + 7.5)
    emptyInputs, _ = prepareSample(sample, codec, unitStatistics(), inputMode="sOnly")
    shifted["hamiltonian"] *= 3
    shiftedEmpty, _ = prepareSample(shifted, codec, unitStatistics(), inputMode="sOnly")
    for key in emptyInputs:
        assert torch.equal(emptyInputs[key], shiftedEmpty[key])
    assert emptyInputs["qH"].abs().max() == 0


def testStatisticsFrozenSharedScalesAndTrainOnly(codec):
    sample = makeSample()
    sample["split"] = "train"
    statistics = fitStatistics([sample], codec)
    assert statistics["trainIds"] == [sample["materialId"]]
    first, _ = scaleTensors(sample, codec, statistics, "h")
    valid = codec.validIrreps(sample["support"])
    assert first[valid].unique().numel() == 1
    unseen = copy.deepcopy(sample)
    unseen["atomicNumbers"] = torch.tensor([2])
    fallback, _ = scaleTensors(unseen, codec, statistics, "h")
    assert (fallback == statistics["h"]["global"]).all()
    sample["split"] = "validation"
    with pytest.raises(ValueError, match="only"):
        fitStatistics([sample], codec)


def testSupportAndReferenceFailures():
    partial = torch.ones(1, 54, 54, dtype=torch.bool)
    partial[0, 4, 4] = False
    with pytest.raises(ValueError):
        auditSupport(partial)
    sample = makeSample()
    sample["overlap"].zero_()
    with pytest.raises(ValueError, match="denominator"):
        chooseReference(sample)


def testStructureGroupsCannotCrossSplits():
    first = makeSample("mp-1")
    equivalent = copy.deepcopy(first)
    equivalent["materialId"] = "mp-2"
    equivalent["positions"] += torch.tensor([.3, .7, 1.2])
    third = copy.deepcopy(first)
    third["materialId"] = "mp-3"
    third["atomicNumbers"] = torch.tensor([2])
    fourth = copy.deepcopy(third)
    fourth["materialId"] = "mp-4"
    manifest = buildManifest([first, equivalent, third, fourth], 2, 2, 0)
    locations = {}
    for split in ("train", "validation", "test"):
        for record in manifest[split]:
            locations[record["materialId"]] = split
    assert locations["mp-1"] == locations["mp-2"]
    assert locations["mp-3"] == locations["mp-4"]
    assert locations["mp-1"] != locations["mp-3"]
    assert manifest["actualCounts"] == {"train": 2, "validation": 2, "test": 0}


def testCompactRoundTrip(tmp_path):
    sample = makeSample()
    destination = tmp_path / "sample.pt"
    cacheSample(sample, destination)
    loaded = loadSample(destination)
    assert loaded["cachePath"] == str(destination)
    torch.testing.assert_close(loaded["hamiltonian"], sample["hamiltonian"])
    assert "eigenvectors" not in loaded


@pytest.mark.skipif(not os.getenv("MARCH_SMOKE_SOURCE"), reason="Explicit real-data opt-in")
def testRealSource(codec):
    sample = loadSample(Path(os.environ["MARCH_SMOKE_SOURCE"]))
    assert sample["hamiltonian"].shape[-2:] == (54, 54)
    inputs, target = prepareSample(sample, codec, fitStatistics([sample], codec))
    assert torch.isfinite(inputs["qH"]).all()
    assert 0 < target["actualMaskRatio"] < 1



def testVectorizedStatisticsMatchLiteralDefinition(codec):
    from march.data.pipeline import irrepKeys
    sample = makeSample()
    sample["support"] = torch.ones_like(sample["support"])
    generator = torch.Generator().manual_seed(19)
    raw = torch.randn(sample["hamiltonian"].shape, generator=generator, dtype=torch.complex64)
    sample["hamiltonian"] = .5 * (raw + raw[sample["reverseIndex"]].mH)
    sample["overlap"] = torch.eye(54, dtype=torch.complex64)[None].expand(len(raw), -1, -1).clone()
    statistics = fitStatistics([sample], codec)
    _, reference = chooseReference(sample)
    keys = irrepKeys(sample, codec)
    valid = codec.validIrreps(sample["support"])
    for channel, matrix in (("h", sample["hamiltonian"] - reference * sample["overlap"]),
                            ("s", sample["overlap"])):
        coefficients = codec.encode(matrix)
        sums = {}
        for edge in range(len(raw)):
            for index, part in enumerate(codec.irrepSlices):
                if valid[edge, index]:
                    pair = sums.setdefault(keys[edge][index], [0., 0])
                    pair[0] += float(coefficients[edge, part].square().sum())
                    pair[1] += part.stop - part.start
        for key, (square, count) in sums.items():
            expected = max((square / count) ** .5, statistics["scaleFloor"])
            assert statistics[channel]["groups"][key] == pytest.approx(expected, rel=2e-7, abs=1e-8)



def testExactMaskBitsetMatchesExhaustiveSmallGraphs():
    import itertools
    import numpy as np
    generator = np.random.default_rng(17)
    for case in range(35):
        sizes = generator.integers(1, 7, size=int(generator.integers(2, 9)))
        groups = torch.repeat_interleave(torch.arange(len(sizes)), torch.tensor(sizes))
        ratio = float(generator.choice([.15, .30, .50, .73]))
        hidden = sampleMask(groups, f"case{case}", seed=19, epoch=3, maskRatio=ratio)
        observed = abs(int(hidden.sum()) - ratio * len(groups))
        possible = [sum(size * bit for size, bit in zip(sizes, choice))
                    for choice in itertools.product((0, 1), repeat=len(sizes))]
        expected = min(abs(count - ratio * len(groups)) for count in possible if 0 < count < len(groups))
        assert observed == pytest.approx(expected, abs=1e-12)
        assert torch.equal(hidden, sampleMask(groups, f"case{case}", seed=19, epoch=3, maskRatio=ratio))
        for group in groups.unique():
            assert hidden[groups == group].unique().numel() == 1



def testReferenceFallbackFindsClosestFeasibleClosureAndIsolatesH():
    # The only exact 30% subset is {size1,size2}, which hides all nonzero S.
    # Keeping size1 visible forces the nearest feasible mask to hide size2.
    groups = torch.tensor([0, 1, 1, 2, 2, 2, 2, 2, 2, 2])
    overlap = torch.zeros(10, 1, 1, dtype=torch.complex64)
    overlap[0] = 1
    sample = {"materialId": "fallback", "maskGroup": groups,
              "overlap": overlap, "hamiltonian": 2 * overlap}
    ordinary = sampleMask(groups, sample["materialId"])
    assert ordinary.sum() == 3 and overlap[~ordinary].abs().sum() == 0
    hidden, reference = chooseReference(sample, maxAttempts=2)
    assert hidden.sum() == 2 and not hidden[0]
    assert reference == 2
    changed = copy.deepcopy(sample)
    changed["hamiltonian"][hidden] += 123 + 7j
    otherHidden, otherReference = chooseReference(changed, maxAttempts=2)
    assert torch.equal(hidden, otherHidden)
    assert reference == otherReference
    firstInput = sample["hamiltonian"] - reference * overlap
    changedInput = changed["hamiltonian"] - otherReference * overlap
    firstInput[hidden] = 0
    changedInput[otherHidden] = 0
    assert torch.equal(firstInput, changedInput)


def testReferenceFallbackMatchesExhaustiveFeasibility():
    import itertools
    import numpy as np
    from march.data.pipeline import feasibleReferenceMask
    generator = np.random.default_rng(30)
    for case in range(30):
        sizes = generator.integers(1, 5, size=int(generator.integers(2, 8)))
        values = generator.choice([0., 0., .5, 1.], size=len(sizes))
        groups = torch.repeat_interleave(torch.arange(len(sizes)), torch.tensor(sizes))
        overlap = torch.repeat_interleave(torch.tensor(values, dtype=torch.float32),
                                         torch.tensor(sizes))[:, None, None].to(torch.complex64)
        sample = {"materialId": f"reference{case}", "maskGroup": groups, "overlap": overlap}
        ratio = float(generator.choice([.15, .3, .5]))
        possibilities = []
        for visible in itertools.product((0, 1), repeat=len(sizes)):
            count = sum(int(size) * bit for size, bit in zip(sizes, visible))
            norm = sum(int(size) * value ** 2 * bit for size, value, bit in zip(sizes, values, visible))
            if 0 < count < len(groups) and norm > .5:
                possibilities.append(abs(len(groups) - count - ratio * len(groups)))
        if not possibilities:
            with pytest.raises(ValueError, match="denominator"):
                feasibleReferenceMask(sample, maskRatio=ratio, threshold=.5)
        else:
            hidden = feasibleReferenceMask(sample, maskRatio=ratio, threshold=.5)
            assert overlap[~hidden].abs().square().sum() > .5
            assert abs(int(hidden.sum()) - ratio * len(groups)) == pytest.approx(min(possibilities))
            for group in groups.unique():
                assert hidden[groups == group].unique().numel() == 1

