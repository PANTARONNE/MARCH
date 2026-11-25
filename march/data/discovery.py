"""Deterministic source discovery, leakage-safe structure groups and split records."""
from __future__ import annotations

import hashlib
from pathlib import Path
import random
import re


def discoverSources(root, limit=None):
    sources = sorted(Path(root).glob("mp-*.pth"),
                     key=lambda path: (int(re.search(r"mp-(\d+)", path.stem).group(1)), path.name))
    return sources if limit is None else sources[:limit]


def buildManifest(samples, trainCount=3500, valCount=250, testCount=250, seed=2026):
    """Exact material IDs and equivalent periodic structures cannot cross splits."""
    from pymatgen.core import Lattice, Structure
    from pymatgen.analysis.structure_matcher import StructureMatcher
    from .geometry import UnionFind

    samples = list(samples)
    if not samples:
        raise ValueError("Cannot split an empty sample pool")
    requested = {"train": trainCount, "validation": valCount, "test": testCount}
    if any(value < 0 for value in requested.values()) or sum(requested.values()) == 0:
        raise ValueError("Requested split sizes must be nonnegative and nonempty")
    if sum(requested.values()) != len(samples):
        raise ValueError(f"Split sizes request {sum(requested.values())} samples, but {len(samples)} were supplied")
    structures = [
        Structure(Lattice(sample["lattice"].double().numpy()), sample["atomicNumbers"].tolist(),
                  sample["positions"].double().numpy(), coords_are_cartesian=True)
        for sample in samples
    ]
    matcher = StructureMatcher(ltol=.001, stol=.001, angle_tol=.1,
                               primitive_cell=True, scale=False, attempt_supercell=True)
    unions, byId, byComposition = UnionFind(len(samples)), {}, {}
    for index, (sample, structure) in enumerate(zip(samples, structures)):
        match = re.search(r"mp-\d+", sample["materialId"])
        materialId = match.group(0) if match else sample["materialId"]
        if materialId in byId:
            unions.union(index, byId[materialId])
        else:
            byId[materialId] = index
        formula = structure.composition.reduced_formula
        for previous in byComposition.setdefault(formula, []):
            if matcher.fit(structures[previous], structure):
                unions.union(index, previous)
        byComposition[formula].append(index)
    grouped = {}
    for index, group in enumerate(unions.labels().tolist()):
        grouped.setdefault(group, []).append(index)
    groups = list(grouped.values())
    random.Random(seed).shuffle(groups)
    records = {name: [] for name in requested}
    for group in groups:
        # Whole groups stay together; record the unavoidable count deviation.
        destination = max((name for name in requested if requested[name] > 0),
                          key=lambda name: requested[name] - len(records[name]))
        identity = "\n".join(sorted(samples[index]["materialId"] for index in group))
        groupId = hashlib.sha256(identity.encode()).hexdigest()[:20]
        for index in group:
            sample = samples[index]
            sample["split"], sample["groupId"] = destination, groupId
            records[destination].append({
                "materialId": sample["materialId"], "sourcePath": sample["sourcePath"],
                "cachePath": sample.get("cachePath", sample["sourcePath"]), "groupId": groupId,
            })
    for values in records.values():
        values.sort(key=lambda value: value["materialId"])
    records.update(schemaVersion=1, seed=seed, requestedCounts=requested,
                   actualCounts={name: len(records[name]) for name in requested},
                   structureGroupCount=len(groups), excluded=[])
    return records

