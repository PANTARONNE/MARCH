import torch
import pytest
from e3nn import o3
from march.codec import SocCodec
from march.models import ReconstructionModel
from march.models.inputEncoder import InputEncoder
from march.models.reconstructionModel import hiddenRepresentation
from march.heads import ReconstructionHead
from testCodec import transformCoefficients


def makeInput(device="cpu"):
    codec = SocCodec(device=device)
    raw = torch.randn(4, 54, 54, dtype=torch.complex64, device=device)
    hidden = torch.tensor([False, False, True, True], device=device)
    qH = codec.encode(raw)
    qH[hidden] = 0
    return {"qH": qH, "qS": codec.encode(torch.randn_like(raw)),
            "hiddenMask": hidden,
            "support": torch.ones((4, 54, 54), device=device, dtype=torch.bool),
            "edgeIndex": torch.tensor([[0, 1, 0, 1], [0, 1, 1, 0]], device=device),
            "edgeVec": torch.tensor([[0., 0., 0.], [0., 0., 0.], [1., .3, -.2], [-1., -.3, .2]], device=device),
            "atomicNumbers": torch.tensor([26, 8], device=device)}


def testOnsiteHighDegreeInputAndOutput():
    torch.manual_seed(4)
    codec = SocCodec()
    hidden = hiddenRepresentation("smoke")
    encoder = InputEncoder(codec.irreps, hidden, fullMultiplicity=1)
    sample = makeInput()
    sample["qH"] = sample["qH"].requires_grad_()
    edgeSh = o3.spherical_harmonics(o3.Irreps.spherical_harmonics(5),
                                   sample["edgeVec"][:, [1, 2, 0]], True, normalization="component")
    nodes, edges = encoder(sample, edgeSh)
    gradient = torch.autograd.grad(nodes.square().sum(), sample["qH"])[0]
    for (_, irrep), region in zip(codec.irrepCopies, codec.irrepSlices):
        if irrep.l >= 5:
            assert gradient[:2, region].abs().sum() > 0, f"Onsite {irrep} input was dropped"
    head = ReconstructionHead(hidden, codec.irreps, multiplicity=1)
    features = torch.randn(2, hidden.dim, requires_grad=True)
    predicted = head(features)
    for (_, irrep), region in zip(codec.irreps, codec.irreps.slices()):
        if irrep.l >= 5:
            assert (predicted[:, region].abs().sum(0) > 0).all(), f"{irrep} output cannot be represented"
            gradient = torch.autograd.grad(predicted[:, region].square().sum(), features, retain_graph=True)[0]
            assert gradient.abs().sum() > 0


@pytest.fixture(scope="module")
def smokeModel():
    torch.manual_seed(17)
    return ReconstructionModel({"preset": "smoke"}).eval()


def testFourBlocksTraceGradAndIsolation(smokeModel):
    assert len(smokeModel.backbone.blocks) == 4
    assert len(smokeModel.backbone.traceGrad) == 3
    assert smokeModel.hiddenIrreps.dim == 98
    sample = makeInput()
    with torch.no_grad():
        output = smokeModel(sample)
    assert output["qPrediction"].shape == (4, 5832)
    assert torch.isfinite(output["qPrediction"]).all()
    with pytest.raises(ValueError, match="isolation"):
        smokeModel(dict(sample, targetH=torch.zeros(1)))
    invalid = dict(sample, qH=sample["qH"].clone())
    invalid["qH"][2] = 1
    with pytest.raises(ValueError, match="Hidden"):
        smokeModel(invalid)


@pytest.mark.parametrize("inversion", [False, True])
def testModelEquivariance(smokeModel, inversion):
    sample = makeInput()
    rotation = -torch.eye(3) if inversion else o3.rand_matrix()
    transformed = dict(sample)
    transformed["qH"] = transformCoefficients(sample["qH"], smokeModel.outputIrreps, rotation)
    transformed["qS"] = transformCoefficients(sample["qS"], smokeModel.outputIrreps, rotation)
    transformed["edgeVec"] = sample["edgeVec"] @ rotation
    before = smokeModel(sample)["qPrediction"].detach()
    after = smokeModel(transformed)["qPrediction"].detach()
    expected = transformCoefficients(before, smokeModel.outputIrreps, rotation)
    torch.testing.assert_close(after, expected, atol=2e-3, rtol=2e-3)


def testAtomPermutation(smokeModel):
    sample = makeInput()
    shuffled = dict(sample)
    shuffled["atomicNumbers"] = sample["atomicNumbers"].flip(0)
    shuffled["edgeIndex"] = 1 - sample["edgeIndex"]
    reference = smokeModel(sample)["qPrediction"].detach()
    permuted = smokeModel(shuffled)["qPrediction"].detach()
    torch.testing.assert_close(permuted, reference, atol=3e-5, rtol=3e-5)


def testFrozenEncoderKeepsFeatureDerivatives(smokeModel):
    try:
        smokeModel.requires_grad_(False)
        with torch.no_grad():
            output = smokeModel(makeInput())
        assert torch.isfinite(output["qPrediction"]).all()
    finally:
        # Restore exactly the trainable subset, including frozen upstream branches.
        for name, value in smokeModel.named_parameters():
            inactive = any(f".{module}." in name for module in
                           ("ga_edge", "update_edge", "node_update", "edge_update"))
            value.requires_grad_(not inactive and not name.endswith("alpha_dot") and ".element_wise." not in name)


def testFinalNodeUpdateReceivesReconstructionGradient(smokeModel):
    sample = makeInput()
    prediction = smokeModel(sample)["qPrediction"]
    parameters = [p for p in smokeModel.backbone.blocks[-1].ffn.parameters() if p.requires_grad]
    gradients = torch.autograd.grad(prediction.square().mean(), parameters, allow_unused=True)
    assert any(g is not None and g.abs().sum() > 0 for g in gradients)
    assert all(g is None or torch.isfinite(g).all() for g in gradients)

def testOrbitalSupportAndMaskMetadataAreEvenScalars():
    codec = SocCodec()
    hidden = hiddenRepresentation("smoke")
    encoder = InputEncoder(codec.irreps, hidden, fullMultiplicity=1)
    sample = makeInput()
    sample["qH"] = torch.zeros_like(sample["qH"])
    sample["qS"] = torch.zeros_like(sample["qS"])
    sample["support"] = torch.zeros_like(sample["support"])
    for spinRow in range(2):
        for spinCol in range(2):
            sample["support"][:, 27 * spinRow, 27 * spinCol] = True
    edgeSh = o3.spherical_harmonics(o3.Irreps.spherical_harmonics(5),
                                   sample["edgeVec"][:, [1, 2, 0]], True, normalization="component")
    _, original = encoder(sample, edgeSh)
    changed = dict(sample, support=torch.zeros_like(sample["support"]))
    for spinRow in range(2):
        for spinCol in range(2):
            changed["support"][:, 1 + 27 * spinRow, 1 + 27 * spinCol] = True
    assert changed["support"].sum() == sample["support"].sum()
    _, updated = encoder(changed, edgeSh)
    _, flippedMask = encoder(dict(sample, hiddenMask=~sample["hiddenMask"]), edgeSh)
    for alternative in (updated, flippedMask):
        assert (alternative - original).abs().sum() > 0
        for (_, irrep), region in zip(hidden, hidden.slices()):
            if irrep != o3.Irrep("0e"):
                torch.testing.assert_close(original[:, region], alternative[:, region], atol=0, rtol=0)