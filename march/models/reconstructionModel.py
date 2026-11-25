"""Target-isolated reconstruction model and reusable encoder interface."""
import torch
from torch import nn
from e3nn import o3
from march.codec import SocCodec
from march.heads import ReconstructionHead
from .inputEncoder import InputEncoder
from .backbone import NextHamBackbone


def hiddenRepresentation(width):
    if width not in ("main", "smoke"):
        raise ValueError("model.width must be main or smoke")
    divisor = 4 if width == "smoke" else 1
    return o3.Irreps([(mul // divisor, ir) for mul, ir in o3.Irreps(
        "32x0e+16x1o+16x1e+8x2e+8x2o+8x3e+8x3o+8x4e")])


class ReconstructionModel(nn.Module):
    allowedInputs = frozenset(("qH", "qS", "hiddenMask", "support", "edgeIndex",
                              "edgeVec", "atomicNumbers"))

    def __init__(self, config=None):
        super().__init__()
        config = config or {}
        config = config.get("model", config)
        width = config.get("preset", config.get("width", "main"))
        if not config.get("traceGrad", True):
            raise ValueError("This pretraining preset requires TraceGrad; replace the backbone for an ablation")
        smoke = width == "smoke"
        self.hiddenIrreps = hiddenRepresentation(width)
        codec = SocCodec()
        self.outputIrreps = codec.irreps
        self.inputEncoder = InputEncoder(
            self.outputIrreps, self.hiddenIrreps, config.get("fullMultiplicity", 1 if smoke else 2))
        self.backbone = NextHamBackbone(
            self.hiddenIrreps, numLayers=config.get("layers", config.get("numLayers", 4)),
            numHeads=config.get("attentionHeads", config.get("numHeads", 4)), radialBasis=config.get("radialBasis", 64),
            radialMaximum=config.get("radialMaximum", 10.0),
            traceInvariant=config.get("traceInvariant", 64 if smoke else 256),
            traceHidden=config.get("traceHidden", 256 if smoke else 1024))
        self.contextMerge = o3.Linear(self.hiddenIrreps + self.hiddenIrreps + self.hiddenIrreps, self.hiddenIrreps)
        self.head = ReconstructionHead(
            self.hiddenIrreps, self.outputIrreps, config.get("headMultiplicity", 1 if smoke else 2))

    @torch.enable_grad()
    def encode(self, modelInput):
        unexpected = set(modelInput) - self.allowedInputs
        missing = self.allowedInputs - set(modelInput)
        if unexpected or missing:
            raise ValueError(f"Model input isolation failed; unexpected={unexpected}, missing={missing}")
        edgeSh = self.backbone.geometricFeatures(modelInput["edgeVec"])
        nodeFeatures, edgeFeatures = self.inputEncoder(modelInput, edgeSh)
        return self.backbone(nodeFeatures, edgeFeatures, modelInput, edgeSh)

    @torch.enable_grad()
    def forward(self, modelInput, returnFeatures=False):
        nodeFeatures, edgeFeatures = self.encode(modelInput)
        source, target = modelInput["edgeIndex"]
        predictionFeatures = edgeFeatures + self.contextMerge(
            torch.cat((edgeFeatures, nodeFeatures[source], nodeFeatures[target]), dim=-1))
        output = {"qPrediction": self.head(predictionFeatures)}
        # DDP discovers unused parameters through returned tensors. Returning
        # unused final node features would hide those parameters from its scan.
        if returnFeatures:
            output.update(nodeFeatures=nodeFeatures, edgeFeatures=edgeFeatures)
        return output
