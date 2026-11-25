import torch
import pytest
from e3nn import o3
from march.codec import SocCodec


def transformCoefficients(coefficients, irreps, rotation):
    order = torch.eye(3, dtype=rotation.dtype, device=rotation.device)[[1, 2, 0]]
    converted = order @ rotation @ order.T
    parts = []
    for (multiplicity, irrep), part in zip(irreps, irreps.slices()):
        d = irrep.D_from_matrix(converted)
        values = coefficients[:, part].reshape(coefficients.shape[0], multiplicity, irrep.dim)
        parts.append((values @ d).flatten(1))
    return torch.cat(parts, dim=-1)


def testRoundTripAndMetadata():
    codec = SocCodec()
    torch.manual_seed(2026)
    raw = torch.randn(2, 54, 54, dtype=torch.complex64)
    q = codec.encode(raw)
    assert q.shape == (2, 5832)
    assert max(ir.l for _, ir in codec.irreps) == 7
    torch.testing.assert_close(codec.decode(q), raw, atol=3e-6, rtol=3e-6)
    torch.testing.assert_close(codec.encode(codec.decode(q)), q, atol=1e-5, rtol=3e-6)
    assert codec.coefficientToIrrep.numel() == 5832
    for index, region in enumerate(codec.irrepSlices):
        assert (codec.coefficientToIrrep[region] == index).all()
        assert region.stop - region.start == codec.irrepDimensions[index]
    for spinRow in range(2):
        for spinCol in range(2):
            isolated = raw.new_zeros(raw.shape)
            isolated[:, spinRow * 27:(spinRow + 1) * 27, spinCol * 27:(spinCol + 1) * 27] = raw[:, :27, :27]
            torch.testing.assert_close(codec.decode(codec.encode(isolated)), isolated, atol=3e-6, rtol=3e-6)


def testShellSupportAndPacking():
    codec = SocCodec()
    support = torch.zeros(2, 54, 54, dtype=torch.bool)
    a, b = 4, 8
    left, right = codec.shellSlices[a], codec.shellSlices[b]
    for si in range(2):
        for sj in range(2):
            support[:, left.start + si * 27:left.stop + si * 27,
                    right.start + sj * 27:right.stop + sj * 27] = True
    valid = codec.validIrreps(support)
    pairs = codec.irrepToShellPair
    expected = (pairs[:, 0] == a) & (pairs[:, 1] == b)
    assert torch.equal(valid[0], expected)
    assert torch.equal(codec.unpack(codec.pack(support)), support)
    support[0, left.start, right.start] = False
    with pytest.raises(ValueError, match="splits"):
        codec.validIrreps(support)


@pytest.mark.parametrize("inversion", [False, True])
def testOfficialJointRotation(inversion):
    torch.manual_seed(27)
    codec = SocCodec(dtype=torch.float64)
    raw = torch.randn(2, 54, 54, dtype=torch.complex128)
    rotation = -torch.eye(3, dtype=torch.float64) if inversion else o3.rand_matrix(dtype=torch.float64)
    expected = transformCoefficients(codec.encode(raw), codec.irreps, rotation)
    observed = codec.encode(codec.rotate(raw, rotation))
    torch.testing.assert_close(observed, expected, atol=2e-5, rtol=2e-5)


def testHermitianProjectionCovariance():
    torch.manual_seed(19)
    codec = SocCodec()
    raw = torch.randn(2, 54, 54, dtype=torch.complex64)
    pair = 0.5 * (raw[0] + raw[1].mH)
    projected = torch.stack((pair, pair.mH))
    restored = codec.decode(codec.encode(projected))
    torch.testing.assert_close(restored[0], restored[1].mH, atol=3e-6, rtol=3e-6)
    rotation = o3.rand_matrix()
    rotated = codec.rotate(projected, rotation)
    torch.testing.assert_close(rotated[0], rotated[1].mH, atol=3e-6, rtol=3e-6)

@pytest.mark.parametrize("shellPair", [(0, 1), (4, 5), (6, 8), (8, 8)])
def testMetadataMatchesOfficialCoefficientLocations(shellPair):
    codec = SocCodec()
    a, b = shellPair
    support = torch.zeros(1, 54, 54, dtype=torch.bool)
    left, right = codec.shellSlices[a], codec.shellSlices[b]
    for spinRow in range(2):
        for spinCol in range(2):
            support[:, left.start + spinRow * 27:left.stop + spinRow * 27,
                    right.start + spinCol * 27:right.stop + spinCol * 27] = True
    raw = torch.randn(1, 54, 54, dtype=torch.complex64) * support
    coefficients = codec.encode(raw)
    validCopies = codec.validIrreps(support)
    allowedCoefficients = validCopies[:, codec.coefficientToIrrep]
    assert coefficients[~allowedCoefficients].abs().max() == 0
    assert coefficients[allowedCoefficients].abs().min() > 0
    torch.testing.assert_close(codec.decode(coefficients), raw, atol=3e-6, rtol=3e-6)

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def testGpuCodecRoundTripAndGradient():
    codec = SocCodec(device="cuda")
    raw = torch.randn(2, 54, 54, device="cuda", dtype=torch.complex64, requires_grad=True)
    coefficients = codec.encode(raw)
    restored = codec.decode(coefficients)
    assert coefficients.is_cuda and restored.is_cuda
    torch.testing.assert_close(restored, raw, atol=4e-6, rtol=4e-6)
    gradient = torch.autograd.grad(restored.abs().square().mean(), raw)[0]
    torch.testing.assert_close(gradient, 2 * raw / raw.numel(), atol=1e-7, rtol=2e-5)