"""Pure hierarchical SOC reconstruction with physical Hermitian projection."""
from __future__ import annotations

import torch
from torch import nn


def hierarchicalMean(blockLoss, validIrreps, hiddenMask, lossOrbit, onsiteMask):
    """Average irrep copies, members, orbits, present block types, then materials."""
    if not hiddenMask.any():
        raise ValueError("Reconstruction loss requires at least one hidden block")
    validCount = validIrreps.sum(1)
    if (validCount[hiddenMask] == 0).any():
        raise ValueError("Hidden block has no valid orbital coefficients")
    perBlock = (blockLoss * validIrreps).sum(1) / validCount.clamp_min(1)
    typeLosses = []
    for onsite in (True, False):
        selected = hiddenMask & (onsiteMask == onsite)
        if not selected.any():
            continue
        orbitLosses = []
        for orbit in torch.unique(lossOrbit[selected]):
            members = selected & (lossOrbit == orbit)
            orbitLosses.append(perBlock[members].mean())
        typeLosses.append(torch.stack(orbitLosses).mean())
    return torch.stack(typeLosses).mean()


class ReconstructionLoss(nn.Module):
    def __init__(self, codec):
        super().__init__()
        self.codec = codec

    def projectPrediction(self, prediction, target):
        """Denormalize, official inverse, project in raw matrix space, re-encode."""
        raw = self.codec.decode(prediction * target["qScale"])
        raw = raw * target["support"]
        duplicateGroup = target["duplicateGroup"]
        # Repeated stored copies represent one physical block before projection.
        sums = torch.zeros_like(raw).index_add(0, duplicateGroup, raw)
        counts = torch.bincount(duplicateGroup, minlength=len(raw)).clamp_min(1)
        raw = (sums / counts[:, None, None])[duplicateGroup]
        raw = .5 * (raw + raw[target["reverseIndex"]].mH)
        return raw, self.codec.encode(raw)

    def forward(self, prediction, target):
        if isinstance(target, (list, tuple)):
            if len(prediction) != len(target) or not target:
                raise ValueError("Material prediction/target batch lengths differ or are empty")
            return torch.stack([self.forward(p, t) for p, t in zip(prediction, target)]).mean()
        _, projected = self.projectPrediction(prediction, target)
        squared = ((projected - target["qTarget"]) / target["qScale"]).square()
        perIrrep = torch.zeros_like(target["validIrreps"], dtype=squared.dtype)
        perIrrep.scatter_add_(1, self.codec.coefficientToIrrep[None].expand(len(squared), -1), squared)
        perIrrep = perIrrep / self.codec.irrepDimensions
        return hierarchicalMean(perIrrep, target["validIrreps"], target["hiddenMask"],
                                target["lossOrbit"], target["onsiteMask"])

    @torch.no_grad()
    def metrics(self, prediction, target):
        raw, _ = self.projectPrediction(prediction, target)
        errors = (raw - target["hamiltonian"]).abs()
        selected = target["hiddenMask"][:, None, None] & target["support"]
        result = {"rawMae": errors[selected].mean().item(),
                  "actualMaskRatio": float(target["actualMaskRatio"])}
        for onsite, name in ((True, "onsiteMae"), (False, "offsiteMae")):
            mask = selected & (target["onsiteMask"] == onsite)[:, None, None]
            result[name] = errors[mask].mean().item() if mask.any() else 0.0
        return result


