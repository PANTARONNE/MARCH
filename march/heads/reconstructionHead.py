"""Replaceable complete-SOC prediction head."""
from torch import nn
from e3nn import o3


class ReconstructionHead(nn.Module):
    def __init__(self, hiddenIrreps, outputIrreps, multiplicity=2):
        super().__init__()
        self.inputIrreps = o3.Irreps(hiddenIrreps)
        self.outputIrreps = o3.Irreps(outputIrreps)
        self.productIrreps = o3.Irreps([(multiplicity, ir) for _, ir in self.inputIrreps])
        self.leftProjection = o3.Linear(self.inputIrreps, self.productIrreps)
        self.rightProjection = o3.Linear(self.inputIrreps, self.productIrreps)
        self.product = o3.FullyConnectedTensorProduct(
            self.productIrreps, self.productIrreps, self.outputIrreps)
        self.directProjection = o3.Linear(self.inputIrreps, self.outputIrreps)
        reachable = {out for _, left in self.productIrreps for _, right in self.productIrreps for out in left * right}
        absent = [str(ir) for _, ir in self.outputIrreps if ir not in reachable]
        if absent:
            raise ValueError(f"Prediction head cannot reach SOC irreps: {absent}")

    def forward(self, edgeFeatures):
        return self.directProjection(edgeFeatures) + self.product(
            self.leftProjection(edgeFeatures), self.rightProjection(edgeFeatures))
