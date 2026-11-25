"""Equivariant H/S and geometric context, including onsite J=5..7."""
import torch
from torch import nn
from e3nn import o3
from torch_scatter import scatter


class InputEncoder(nn.Module):
    def __init__(self, inputIrreps, hiddenIrreps, fullMultiplicity=2):
        super().__init__()
        self.inputIrreps = o3.Irreps(inputIrreps)
        self.hiddenIrreps = o3.Irreps(hiddenIrreps)
        # Independent maps are essential: a self-product would cancel some
        # antisymmetric couplings and permanently discard valid SOC channels.
        types = sorted({ir for _, ir in self.inputIrreps})
        self.fullIrreps = o3.Irreps([(fullMultiplicity, ir) for ir in types])
        combined = self.inputIrreps + self.inputIrreps
        self.leftProjection = o3.Linear(combined, self.fullIrreps)
        self.rightProjection = o3.Linear(combined, self.fullIrreps)
        self.fullProduct = o3.FullyConnectedTensorProduct(
            self.fullIrreps, self.fullIrreps, self.hiddenIrreps)
        self.directProjection = o3.Linear(combined, self.hiddenIrreps)
        self.elements = nn.Embedding(119, 16)
        self.metadataProjection = o3.Linear("114x0e", self.hiddenIrreps, biases=True)
        shellPooling = torch.zeros(27, 9)
        start = 0
        for shell, degree in enumerate((0, 0, 0, 0, 1, 1, 2, 2, 3)):
            size = 2 * degree + 1
            shellPooling[start:start + size, shell] = 1.0 / size
            start += size
        self.register_buffer("shellPooling", shellPooling)
        self.geometryIrreps = o3.Irreps.spherical_harmonics(5)
        self.geometryProjection = o3.Linear(self.geometryIrreps, self.hiddenIrreps)

    def forward(self, modelInput, edgeSh):
        qH, qS = modelInput["qH"], modelInput["qS"]
        hidden = modelInput["hiddenMask"].bool()
        if bool((qH[hidden] != 0).any()):
            raise ValueError("Hidden H coefficients must be zero before model entry")
        coefficients = torch.cat((qH, qS), dim=-1)
        left = self.leftProjection(coefficients)
        right = self.rightProjection(coefficients)
        electronic = self.directProjection(coefficients) + self.fullProduct(left, right)
        source, target = modelInput["edgeIndex"]
        atoms = modelInput["atomicNumbers"]
        atomic = self.elements(atoms)
        # Encode radial-shell support explicitly. A total orbital count cannot
        # distinguish equal-sized bases with different angular momenta.
        orbitalSupport = modelInput["support"].to(qH.dtype).reshape(-1, 2, 27, 2, 27).mean(dim=(1, 3))
        shellSupport = torch.einsum("ia,eij,jb->eab", self.shellPooling, orbitalSupport, self.shellPooling).flatten(1)
        metadata = torch.cat((atomic[source], atomic[target], hidden.to(qH.dtype)[:, None], shellSupport), dim=-1)
        edgeFeatures = electronic + self.metadataProjection(metadata) + self.geometryProjection(edgeSh)
        onsite = (source == target) & (modelInput["edgeVec"].square().sum(-1) < 1e-10)
        # The full-type product is evaluated before onsite pooling, so all J
        # remain observable even when the displacement spherical harmonics vanish.
        nodeFeatures = scatter(edgeFeatures[onsite], source[onsite], dim=0,
                               dim_size=atoms.numel(), reduce="mean")
        return nodeFeatures, edgeFeatures
