"""Strict 18-field import, compact samples, energy reference and frozen scales."""
from __future__ import annotations

import hashlib
import weakref
import numpy as np
from pathlib import Path
import torch

from .geometry import auditSupport, buildGeometry, verifyLossOrbits, sampleMask, stableSeed

schemaVersion = 1


def officialRotate(raw, rotation):
    from march.codec.upstream import activateNextHam
    activateNextHam()
    from tg_src.e3modules import Rotate
    from .geometry import shellAngular
    transform = Rotate(raw.dtype, device_torch="cpu", spinful=True)
    return transform.rotate_openmx_H_full(raw.cpu(), rotation.float().cpu(),
                                        list(shellAngular), list(shellAngular))


def loadSample(path, verifySymmetry=True):
    """Read allowed fields only; compact files retain no spectra or alternative H."""
    path = Path(path)
    data = torch.load(str(path), map_location="cpu", weights_only=True, mmap=True)
    if isinstance(data, dict):
        if data.get("schemaVersion") != schemaVersion:
            raise ValueError(f"Unsupported compact sample schema: {path}")
        return data
    if not isinstance(data, (list, tuple)) or len(data) != 18:
        raise ValueError(f"Expected audited 18-field SOC source, got {type(data).__name__}")
    from pymatgen.core import Element
    positions = torch.as_tensor(data[2]).float().contiguous()
    source, destination = torch.as_tensor(data[11]).long(), torch.as_tensor(data[12]).long()
    edgeIndex = torch.stack((source, destination))
    if edgeIndex.min() < 0 or edgeIndex.max() >= len(positions):
        raise ValueError("Atom index outside coordinate array")
    atomicNumbers = torch.zeros(len(positions), dtype=torch.long)
    if len(data[13]) != len(source):
        raise ValueError("Element-pair count does not match edge count")
    for edge, pair in enumerate(data[13]):
        for atom, symbol in zip((int(source[edge]), int(destination[edge])), pair):
            number = Element(symbol).Z if isinstance(symbol, str) else int(symbol)
            if atomicNumbers[atom] not in (0, number):
                raise ValueError("Inconsistent atom elements between edges")
            atomicNumbers[atom] = number
    if not (atomicNumbers > 0).all():
        raise ValueError("Some atoms have no recorded element")
    rawSupport = torch.as_tensor(data[17])
    if not torch.isfinite(rawSupport).all() or not ((rawSupport == 0) | (rawSupport == 1)).all():
        raise ValueError("Orbital support must contain only real zero or one")
    support = auditSupport(rawSupport.real.bool())
    hamiltonian = torch.as_tensor(data[15]).to(torch.complex64).contiguous()
    overlap = torch.as_tensor(data[16]).to(torch.complex64).contiguous()
    for name, value in (("Hamiltonian", hamiltonian), ("overlap", overlap)):
        if value.shape != support.shape or not torch.isfinite(value).all():
            raise ValueError(f"Invalid {name} shape or nonfinite values")
        if (~support).any() and value[~support].abs().max().item() > 1e-6:
            raise ValueError(f"Nonzero {name} outside declared support")
    sample = {
        "schemaVersion": schemaVersion, "materialId": path.stem,
        "sourcePath": str(path.resolve()), "lattice": torch.as_tensor(data[1]).float(),
        "positions": positions, "atomicNumbers": atomicNumbers,
        "edgeIndex": edgeIndex, "edgeVec": torch.as_tensor(data[10]).float().contiguous(),
        "hamiltonian": hamiltonian, "overlap": overlap, "support": support,
    }
    geometry, baseRelations, candidates = buildGeometry(sample)
    sample.update(geometry)
    # This metadata is used only by the loss, never sampling or model inputs.
    sample["lossOrbit"] = verifyLossOrbits(sample, baseRelations, candidates if verifySymmetry else [], officialRotate)
    sample["symmetryVerified"] = bool(verifySymmetry)
    sample["audit"] = {
        "edges": len(source), "atoms": len(positions),
        "maskGroups": int(sample["maskGroup"].max()) + 1,
        "lossOrbits": int(sample["lossOrbit"].max()) + 1,
        "candidateOperations": len(candidates),
        "hermitianMaxError": float((hamiltonian - hamiltonian[sample["reverseIndex"]].mH).abs().max()),
    }
    # Admission and mask feasibility depend only on geometry and allowed S.
    chooseReference(sample, seed=2026, epoch=0, maskRatio=.3)
    return sample


def cacheSample(sample, path, *, finalPath=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cached = dict(sample)
    cached["cachePath"] = str((path if finalPath is None else Path(finalPath)).resolve())
    temporary = path.with_suffix(path.suffix + ".temporary")
    torch.save(cached, temporary)
    temporary.replace(path)
    sample["cachePath"] = cached["cachePath"]
    return sample


def feasibleReferenceMask(sample, seed=2026, epoch=0, maskRatio=.3, threshold=1e-12):
    """Rare S-only fallback: maximize visible overlap norm for every block count.

    Per-stage decisions make traceback exact, without mutable parent pointers
    that could accidentally reuse a group. No Hamiltonian field is accessed.
    """
    maskGroup = sample["maskGroup"]
    groups, inverse, sizes = torch.unique(maskGroup.cpu(), sorted=True,
                                          return_inverse=True, return_counts=True)
    if len(groups) < 2:
        raise ValueError("Mask closure has fewer than two groups; no reference split exists")
    overlap = sample["overlap"].cpu()
    blockWeights = overlap.abs().square().double().flatten(1).sum(1)
    groupWeights = torch.zeros(len(groups), dtype=torch.float64).index_add(0, inverse, blockWeights).numpy()
    total = len(maskGroup)
    rng = np.random.default_rng(stableSeed(sample["materialId"], seed, epoch, 2**31 - 1))
    order = rng.permutation(len(groups))
    bestVisibleWeight = np.full(total + 1, -np.inf, dtype=np.float64)
    bestVisibleWeight[0] = 0.0
    decisions = np.zeros((len(groups), total + 1), dtype=bool)
    for stage, position in enumerate(order):
        size = int(sizes[position])
        candidate = bestVisibleWeight[:-size] + groupWeights[position]
        current = bestVisibleWeight[size:]
        decisions[stage, size:] = candidate > current
        bestVisibleWeight[size:] = np.maximum(current, candidate)
    counts = np.arange(1, total)
    feasible = counts[bestVisibleWeight[1:total] > threshold]
    if not len(feasible):
        raise ValueError("No visible/hidden closure split has a valid overlap reference denominator")
    distances = np.abs(total - feasible - maskRatio * total)
    # Recompute the actual denominator after traceback to retain the ordinary
    # float32 reference rule even for a sample at the numerical threshold.
    for distance in sorted(set(distances.tolist())):
        choices = feasible[np.isclose(distances, distance, atol=1e-12, rtol=0)]
        for visibleCount in rng.permutation(choices):
            remaining = int(visibleCount)
            visibleGroups = []
            for stage in range(len(order) - 1, -1, -1):
                if decisions[stage, remaining]:
                    position = int(order[stage])
                    visibleGroups.append(int(groups[position]))
                    remaining -= int(sizes[position])
            if remaining != 0:
                raise RuntimeError("Invalid reference-mask dynamic-programming traceback")
            visible = torch.isin(maskGroup, torch.tensor(visibleGroups, device=maskGroup.device))
            if overlap[visible.cpu()].abs().square().sum() > threshold:
                return ~visible
    raise ValueError("No visible/hidden closure split has a numerically valid overlap reference denominator")


def chooseReference(sample, seed=2026, epoch=0, maskRatio=.3, maxAttempts=32):
    """Only allowed S may change mask retries; hidden H never participates."""
    overlap, hamiltonian = sample["overlap"], sample["hamiltonian"]
    for attempt in range(maxAttempts):
        hidden = sampleMask(sample["maskGroup"], sample["materialId"], seed, epoch, maskRatio, attempt)
        visible = ~hidden
        denominator = overlap[visible].abs().square().sum()
        if denominator > 1e-12:
            reference = (overlap[visible].conj() * hamiltonian[visible]).real.sum() / denominator
            return hidden, reference
    hidden = feasibleReferenceMask(sample, seed, epoch, maskRatio)
    visible = ~hidden
    denominator = overlap[visible].abs().square().sum()
    reference = (overlap[visible].conj() * hamiltonian[visible]).real.sum() / denominator
    return hidden, reference


codecMetadata = weakref.WeakKeyDictionary()


def normalizationMetadata(codec):
    """Keep representation metadata on CPU without repeated GPU synchronization."""
    if codec not in codecMetadata:
        codecMetadata[codec] = {
            "shellPairs": codec.irrepToShellPair.detach().cpu().tolist(),
            "dimensions": codec.irrepDimensions.detach().cpu().numpy(),
            "elements": {},
        }
    return codecMetadata[codec]


def elementPairLayout(leftZ, rightZ, codec):
    metadata = normalizationMetadata(codec)
    pair = (int(leftZ), int(rightZ))
    if pair not in metadata["elements"]:
        keys = []
        for leftShell, rightShell in metadata["shellPairs"]:
            left, right = sorted(((pair[0], leftShell), (pair[1], rightShell)))
            keys.append(f"{left[0]}:{left[1]}|{right[0]}:{right[1]}")
        groupKeys = sorted(set(keys))
        groups = {key: index for index, key in enumerate(groupKeys)}
        metadata["elements"][pair] = {
            "keys": keys, "groupKeys": groupKeys,
            "groupIndices": np.array([groups[key] for key in keys], dtype=np.int64),
        }
    return metadata["elements"][pair]


def elementPairs(sample):
    edges = sample["atomicNumbers"][sample["edgeIndex"]].T.cpu().numpy()
    return np.unique(edges, axis=0, return_inverse=True)


def irrepKeys(sample, codec):
    """Unordered element/radial-shell pairs share spin, real/imag and J scales."""
    pairs, inverse = elementPairs(sample)
    keys = [elementPairLayout(*pair, codec)["keys"] for pair in pairs]
    return [keys[index] for index in inverse]


def scaleTensors(sample, codec, statistics, channel):
    channelStats = statistics[channel]
    groupScales, fallback = channelStats["groups"], channelStats["global"]
    floor = statistics["scaleFloor"]
    pairs, inverse = elementPairs(sample)
    perPair = []
    for pair in pairs:
        layout = elementPairLayout(*pair, codec)
        grouped = np.array([max(float(groupScales.get(key, fallback)), floor)
                            for key in layout["groupKeys"]])
        perPair.append(grouped[layout["groupIndices"]])
    scaleIrrep = torch.as_tensor(np.stack(perPair)[inverse], device=codec.device, dtype=codec.dtype)
    return scaleIrrep, scaleIrrep[:, codec.coefficientToIrrep]


def fitStatistics(samples, codec, seed=2026, maskRatio=.3, scaleFloor=1e-6):
    """Fit frozen RMS statistics; all edge/irrep reductions are vectorized."""
    if scaleFloor <= 0:
        raise ValueError("scaleFloor must be positive")
    accumulators = {"h": {}, "s": {}}
    trainIds = []
    coefficientToIrrep = codec.coefficientToIrrep
    dimensions = normalizationMetadata(codec)["dimensions"]
    for sample in samples:
        if sample.get("split", "train") != "train":
            raise ValueError("Statistics may only be fitted on training samples")
        trainIds.append(sample["materialId"])
        _, reference = chooseReference(sample, seed, 0, maskRatio)
        centered = sample["hamiltonian"] - reference * sample["overlap"]
        values = {"h": centered, "s": sample["overlap"]}
        valid = codec.validIrreps(sample["support"].to(codec.device)).cpu().numpy()
        pairs, inverse = elementPairs(sample)
        for channel, raw in values.items():
            with torch.no_grad():
                q = codec.encode(raw.to(codec.device))
                squares = torch.zeros((len(q), len(dimensions)), dtype=q.dtype, device=q.device)
                squares.scatter_add_(1, coefficientToIrrep[None].expand(len(q), -1), q.square())
                squares = squares.double().cpu().numpy()
            accumulator = accumulators[channel]
            for pairIndex, pair in enumerate(pairs):
                selected = inverse == pairIndex
                layout = elementPairLayout(*pair, codec)
                totals = np.bincount(layout["groupIndices"],
                                     weights=(squares[selected] * valid[selected]).sum(0),
                                     minlength=len(layout["groupKeys"]))
                counts = np.bincount(layout["groupIndices"],
                                     weights=valid[selected].sum(0) * dimensions,
                                     minlength=len(layout["groupKeys"])).astype(np.int64)
                for key, total, count in zip(layout["groupKeys"], totals, counts):
                    if count:
                        previous = accumulator.setdefault(key, [0.0, 0])
                        previous[0] += float(total)
                        previous[1] += int(count)
    if not trainIds:
        raise ValueError("Cannot fit scales on an empty training pool")
    result = {"schemaVersion": schemaVersion, "scaleFloor": scaleFloor,
              "trainIds": sorted(trainIds), "seed": seed, "maskRatio": maskRatio,
              "normalization": "rmsWithoutMean",
              "trainingDigest": hashlib.sha256("\n".join(sorted(trainIds)).encode()).hexdigest()}
    for channel, accumulator in accumulators.items():
        squareTotal = sum(value[0] for value in accumulator.values())
        countTotal = sum(value[1] for value in accumulator.values())
        if countTotal == 0:
            raise ValueError("Training pool contains no valid orbital coefficients")
        result[channel] = {
            "global": max((squareTotal / countTotal) ** .5, scaleFloor),
            "groups": {key: max((total / count) ** .5, scaleFloor)
                       for key, (total, count) in accumulator.items()},
        }
    return result


def prepareSample(sample, codec, statistics, seed=2026, epoch=0, maskRatio=.3, inputMode="hAndS"):
    """Return an allowlisted model input, separate from all supervision metadata."""
    hidden, reference = chooseReference(sample, seed, epoch, maskRatio)
    centered = sample["hamiltonian"] - reference * sample["overlap"]
    masked = centered.clone()
    masked[hidden] = 0
    if inputMode == "sOnly":
        masked.zero_()
    elif inputMode != "hAndS":
        raise ValueError("inputMode must be hAndS or sOnly")
    scaleIrrep, qScale = scaleTensors(sample, codec, statistics, "h")
    _, sScale = scaleTensors(sample, codec, statistics, "s")
    device = codec.device
    with torch.no_grad():
        qH = codec.encode(masked.to(device)) / qScale
        qS = codec.encode(sample["overlap"].to(device)) / sScale
        qTarget = codec.encode(centered.to(device))
    modelInput = {key: sample[key].to(device) for key in (
        "support", "edgeIndex", "edgeVec", "atomicNumbers")}
    modelInput.update(qH=qH, qS=qS, hiddenMask=hidden.to(device))
    target = {key: sample[key].to(device) for key in (
        "support", "reverseIndex", "duplicateGroup", "lossOrbit", "onsiteMask")}
    target.update(materialId=sample["materialId"], hamiltonian=centered.to(device),
                  qTarget=qTarget, qScale=qScale, scaleIrrep=scaleIrrep,
                  validIrreps=codec.validIrreps(sample["support"].to(device)),
                  hiddenMask=hidden.to(device), reference=reference.to(device),
                  actualMaskRatio=float(hidden.float().mean()))
    return modelInput, target




