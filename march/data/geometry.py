"""Target-independent periodic geometry and conservative mask closures."""
from __future__ import annotations

import hashlib
import numpy as np
import torch

shellAngular = (0, 0, 0, 0, 1, 1, 2, 2, 3)
shellBoundaries = (0, 1, 2, 3, 4, 7, 10, 15, 20, 27)


class UnionFind:
    def __init__(self, size):
        self.parents = list(range(size))

    def find(self, index):
        while self.parents[index] != index:
            self.parents[index] = self.parents[self.parents[index]]
            index = self.parents[index]
        return index

    def union(self, first, second):
        first, second = self.find(first), self.find(second)
        if first != second:
            self.parents[max(first, second)] = min(first, second)

    def labels(self):
        roots = [self.find(index) for index in range(len(self.parents))]
        mapping = {root: index for index, root in enumerate(sorted(set(roots)))}
        return torch.tensor([mapping[root] for root in roots], dtype=torch.long)


def stableSeed(materialId, seed, epoch=0, attempt=0):
    text = f"{seed}:{materialId}:{epoch}:{attempt}".encode()
    return int.from_bytes(hashlib.sha256(text).digest()[:8], "little") % (2**63 - 1)


def auditSupport(support):
    """A valid shell is complete in m and identical in both spin sectors."""
    support = support.bool()
    if support.ndim != 3 or tuple(support.shape[1:]) != (54, 54):
        raise ValueError("Expected support [E,54,54]")
    spatial = support[:, :27, :27]
    if not all(torch.equal(spatial, support[:, i:i+27, j:j+27])
               for i in (0, 27) for j in (0, 27)):
        raise ValueError("Orbital support differs across spin sectors")
    for a in range(9):
        for b in range(9):
            block = spatial[:, shellBoundaries[a]:shellBoundaries[a+1],
                            shellBoundaries[b]:shellBoundaries[b+1]]
            count = block.flatten(1).sum(1)
            if not torch.all((count == 0) | (count == block.shape[1] * block.shape[2])):
                raise ValueError("Partial angular shell support is not equivariant")
    left, right = spatial.any(2), spatial.any(1)
    if not torch.equal(spatial, left[:, :, None] & right[:, None, :]):
        raise ValueError("Support must be a product of atom orbital supports")
    if not spatial.flatten(1).any(1).all():
        raise ValueError("Empty orbital-support block")
    return support


def buildGeometry(sample, symprec=1e-4):
    """Return geometry-only groups and candidate relations; never reads H or S."""
    import spglib

    lattice = sample["lattice"].double()
    positions = sample["positions"].double()
    edgeIndex = sample["edgeIndex"].long()
    edgeVec = sample["edgeVec"].double()
    support = sample["support"].bool()
    atomicNumbers = sample["atomicNumbers"].long()
    # The same atom cannot silently gain or lose orbitals on different edges.
    atomSupport = {}
    spatialSupport = support[:, :27, :27]
    for edge in range(len(edgeVec)):
        for atom, active in ((int(edgeIndex[0, edge]), spatialSupport[edge].any(1)),
                             (int(edgeIndex[1, edge]), spatialSupport[edge].any(0))):
            if atom in atomSupport and not torch.equal(atomSupport[atom], active):
                raise ValueError("Inconsistent orbital support for an atom across edges")
            atomSupport[atom] = active

    translationsFloat = (edgeVec - positions[edgeIndex[1]] + positions[edgeIndex[0]]) @ torch.linalg.inv(lattice)
    translations = translationsFloat.round().long()
    if not torch.allclose(translationsFloat, translations.double(), atol=2e-4, rtol=0):
        raise ValueError("Edge vectors do not reconstruct integer periodic translations")
    count = edgeVec.shape[0]
    keys = [(int(edgeIndex[0, e]), int(edgeIndex[1, e]), *translations[e].tolist()) for e in range(count)]
    byKey = {}
    for index, key in enumerate(keys):
        byKey.setdefault(key, []).append(index)
    maskUnion = UnionFind(count)
    baseRelations = []
    reverseIndex = torch.empty(count, dtype=torch.long)
    for edge, (src, dst, tx, ty, tz) in enumerate(keys):
        for duplicate in byKey[keys[edge]]:
            if not torch.equal(support[edge], support[duplicate]):
                raise ValueError("Duplicate edge has inconsistent orbital support")
            maskUnion.union(edge, duplicate)
            baseRelations.append((edge, duplicate, False))
        reverse = byKey.get((dst, src, -tx, -ty, -tz))
        if not reverse:
            raise ValueError(f"Missing Hermitian reverse for edge {keys[edge]}")
        reverseIndex[edge] = reverse[0]
        for other in reverse:
            if not torch.equal(support[edge], support[other].T):
                raise ValueError("Hermitian partners have inconsistent support")
            maskUnion.union(edge, other)
            baseRelations.append((edge, other, True))

    fractional = (positions @ torch.linalg.inv(lattice)).numpy()
    cell = (lattice.numpy(), fractional, atomicNumbers.numpy())
    operations = spglib.get_symmetry(cell, symprec=symprec)
    if operations is None:
        raise ValueError("spglib could not determine candidate space-group operations")
    symmetryRelations = []
    for rotation, offset in zip(operations["rotations"], operations["translations"]):
        transformed = fractional @ rotation.T + offset
        mapping, shifts = [], []
        for atom, coord in enumerate(transformed):
            difference = coord[None, :] - fractional
            nearest = np.rint(difference)
            distances = np.linalg.norm((difference - nearest) @ lattice.numpy(), axis=1)
            candidates = np.where((atomicNumbers.numpy() == int(atomicNumbers[atom])) & (distances <= symprec * 2))[0]
            if len(candidates) != 1:
                raise ValueError("Ambiguous atom map for a candidate space-group operation")
            mapped = int(candidates[0])
            mapping.append(mapped)
            shifts.append(nearest[mapped].astype(int))
        if len(set(mapping)) != len(mapping):
            raise ValueError("Non-bijective space-group atom map")
        pairs = []
        for edge, (src, dst, *translation) in enumerate(keys):
            mappedT = np.asarray(translation) @ rotation.T + shifts[dst] - shifts[src]
            mappedKey = (mapping[src], mapping[dst], *mappedT.tolist())
            for other in byKey.get(mappedKey, []):
                if torch.equal(support[edge], support[other]):
                    maskUnion.union(edge, other)
                    pairs.append((edge, other))
        if pairs:
            # Official NextHAM Rotate uses row-vector Cartesian rotations.
            cartesian = torch.linalg.inv(lattice) @ torch.as_tensor(rotation.T.copy()).double() @ lattice
            if not torch.allclose(cartesian.T @ cartesian, torch.eye(3, dtype=torch.double), atol=3e-4, rtol=0):
                raise ValueError("Non-orthogonal candidate Cartesian operation")
            # Polar correction only removes finite coordinate precision.
            u, _, vh = torch.linalg.svd(cartesian)
            symmetryRelations.append((u @ vh, pairs))
    onsite = (edgeIndex[0] == edgeIndex[1]) & (translations == 0).all(1)
    return {
        "translation": translations,
        "reverseIndex": reverseIndex,
        "duplicateGroup": torch.tensor([byKey[key][0] for key in keys], dtype=torch.long),
        "maskGroup": maskUnion.labels(),
        "onsiteMask": onsite,
    }, baseRelations, symmetryRelations


def verifyLossOrbits(sample, baseRelations, symmetryRelations, rotate):
    """Target-derived verification is confined to supervision-side averaging."""
    hamiltonian, overlap = sample["hamiltonian"], sample["overlap"]
    unions = UnionFind(len(hamiltonian))
    def matches(first, second):
        difference = (first - second).abs().flatten(1)
        scale = torch.maximum(first.abs().flatten(1).amax(1), second.abs().flatten(1).amax(1))
        return difference.amax(1) <= 2e-5 + 2e-4 * scale
    for first, second, conjugate in baseRelations:
        h = hamiltonian[first].mH if conjugate else hamiltonian[first]
        s = overlap[first].mH if conjugate else overlap[first]
        if matches(h[None], hamiltonian[second:second+1])[0] and matches(s[None], overlap[second:second+1])[0]:
            unions.union(first, second)
    for rotation, pairs in symmetryRelations:
        src = torch.tensor([p[0] for p in pairs])
        dst = torch.tensor([p[1] for p in pairs])
        transformedH = rotate(hamiltonian, rotation)
        transformedS = rotate(overlap, rotation)
        valid = matches(transformedH[src], hamiltonian[dst]) & matches(transformedS[src], overlap[dst])
        for (first, second), accepted in zip(pairs, valid.tolist()):
            if accepted:
                unions.union(first, second)
    return unions.labels()


def sampleMask(maskGroup, materialId, seed=2026, epoch=0, maskRatio=.3, attempt=0):
    """Exact closest block count with bitset subset-sum and O(E) parent storage."""
    if not 0 < maskRatio < 1:
        raise ValueError("maskRatio must be strictly between zero and one")
    groups, sizes = torch.unique(maskGroup.cpu(), sorted=True, return_counts=True)
    if len(groups) < 2:
        raise ValueError("Mask closure has fewer than two groups; no visible/hidden split exists")
    rng = np.random.default_rng(stableSeed(materialId, seed, epoch, attempt))
    order = rng.permutation(len(groups))
    total = len(maskGroup)
    reachable = 1  # Bit zero denotes the empty subset.
    allowed = (1 << total) - 1  # Exclude the all-hidden solution.
    parentSum = np.full(total, -1, dtype=np.int64)
    parentGroup = np.full(total, -1, dtype=np.int64)
    for position in order:
        size, group = int(sizes[position]), int(groups[position])
        shifted = (reachable << size) & allowed
        novel = shifted & ~reachable
        # Each sum is recorded once across all groups. The randomized order
        # selects reproducible alternatives without any target-dependent input.
        while novel:
            bit = novel & -novel
            candidate = bit.bit_length() - 1
            parentSum[candidate] = candidate - size
            parentGroup[candidate] = group
            novel ^= bit
        reachable |= shifted
    reachable &= ~1  # Nonempty hidden set.
    target = maskRatio * total
    lowerLimit = min(int(np.floor(target)), total - 1)
    below = reachable & ((1 << (lowerLimit + 1)) - 1)
    upperStart = max(int(np.ceil(target)), 1)
    above = reachable >> upperStart
    candidates = []
    if below:
        candidates.append(below.bit_length() - 1)
    if above:
        candidates.append(upperStart + (above & -above).bit_length() - 1)
    bestDistance = min(abs(count - target) for count in candidates)
    best = sorted(set(count for count in candidates if abs(abs(count - target) - bestDistance) < 1e-9))
    count = int(rng.choice(best))
    selected = []
    while count:
        selected.append(int(parentGroup[count]))
        count = int(parentSum[count])
    selectedGroups = torch.tensor(selected, device=maskGroup.device)
    return torch.isin(maskGroup, selectedGroups)

