"""Hand-computable hierarchy and full-SOC projection regression tests."""
import pytest
import torch
from march.codec import SocCodec
from march.losses.reconstructionLoss import ReconstructionLoss, hierarchicalMean


def testHierarchyAveragesIrrepsThenMembersOrbitsAndTypes():
    block = torch.tensor([[1., 3.], [3., 5.], [10., 99.], [20., 40.]])
    valid = torch.tensor([[1, 1], [1, 1], [1, 0], [1, 1]], dtype=torch.bool)
    hidden = torch.ones(4, dtype=torch.bool)
    orbit = torch.tensor([0, 0, 1, 2])
    onsite = torch.tensor([1, 1, 1, 0], dtype=torch.bool)
    # Block2,4,10,30; orbit3,10,30; type6.5,30; material18.25.
    assert hierarchicalMean(block, valid, hidden, orbit, onsite) == 18.25
    hidden[-1] = False
    assert hierarchicalMean(block, valid, hidden, orbit, onsite) == 6.5
    selection = torch.tensor([0, 1, 0, 1, 2, 3])
    assert hierarchicalMean(block[selection], valid[selection], torch.ones(6, dtype=torch.bool),
                            orbit[selection], onsite[selection]) == 18.25


@pytest.fixture(scope="module")
def codec():
    torch.set_num_threads(2)
    return SocCodec()


def makeTarget(codec, edges=2, padded=False):
    support = torch.ones(edges, 54, 54, dtype=torch.bool)
    if padded:
        support.zero_()
        for left in (0, 27):
            for right in (0, 27):
                support[:, left, right] = True
    raw = torch.zeros(edges, 54, 54, dtype=torch.complex64)
    return dict(support=support, reverseIndex=torch.tensor([1, 0]),
                duplicateGroup=torch.arange(edges),
                lossOrbit=torch.zeros(edges, dtype=torch.long),
                onsiteMask=torch.zeros(edges, dtype=torch.bool),
                hiddenMask=torch.ones(edges, dtype=torch.bool),
                validIrreps=codec.validIrreps(support), qTarget=codec.encode(raw),
                qScale=torch.ones(edges, 5832), hamiltonian=raw, actualMaskRatio=1.0)


def testProjectionEnforcesFullComplexHermitianAndGradient(codec):
    objective = ReconstructionLoss(codec)
    target = makeTarget(codec)
    prediction = torch.randn(2, 5832, requires_grad=True)
    projected, _ = objective.projectPrediction(prediction, target)
    torch.testing.assert_close(projected[0], projected[1].mH, atol=0, rtol=0)
    loss = objective(prediction, target)
    loss.backward()
    assert prediction.grad is not None and torch.isfinite(prediction.grad).all()
    assert prediction.grad.abs().sum() > 0
    assert loss > 0


def testDuplicateCopiesAreAveragedBeforeProjection(codec):
    objective = ReconstructionLoss(codec)
    target = makeTarget(codec, edges=4)
    target["reverseIndex"] = torch.tensor([2, 2, 0, 0])
    target["duplicateGroup"] = torch.tensor([0, 0, 2, 2])
    prediction = torch.randn(4, 5832, requires_grad=True)
    projected, _ = objective.projectPrediction(prediction, target)
    torch.testing.assert_close(projected[0], projected[1], atol=0, rtol=0)
    torch.testing.assert_close(projected[2], projected[3], atol=0, rtol=0)
    torch.testing.assert_close(projected[0], projected[2].mH, atol=0, rtol=0)
    projected.abs().square().mean().backward()
    assert (prediction.grad.abs().sum(1) > 0).all()


def testPaddingIgnoredButPhysicalZerosSupervised(codec):
    objective = ReconstructionLoss(codec)
    target = makeTarget(codec, padded=True)
    prediction = torch.zeros(2, 5832)
    invalid = ~target["validIrreps"][:, codec.coefficientToIrrep]
    prediction[invalid] = 100
    assert objective(prediction, target) < 1e-10
    raw = torch.zeros(2, 54, 54, dtype=torch.complex64)
    raw[:, 0, 0] = 1
    raw[:, 27, 27] = 1
    assert objective(codec.encode(raw), target) > 0


def testOnsiteSelfProjectionAndMaterialAverage(codec):
    objective = ReconstructionLoss(codec)
    target = makeTarget(codec)
    target["reverseIndex"] = torch.arange(2)
    target["onsiteMask"][:] = True
    prediction = torch.randn(2, 5832)
    projected, _ = objective.projectPrediction(prediction, target)
    torch.testing.assert_close(projected, projected.mH, atol=0, rtol=0)
    first = objective(prediction, target)
    second = objective(2 * prediction, target)
    torch.testing.assert_close(objective([prediction, 2 * prediction], [target, target]), .5 * (first + second))


def testIrrepDimensionNormalization():
    dimensions = torch.tensor([1., 3., 5.])
    coordinateSquares = torch.tensor([4., 12., 20.])
    loss = hierarchicalMean((coordinateSquares / dimensions)[None],
                            torch.ones(1, 3, dtype=torch.bool), torch.tensor([True]),
                            torch.tensor([0]), torch.tensor([True]))
    assert loss == 4.

