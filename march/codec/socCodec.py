"""Exact full-SOC NextHAM codec, with explicit radial-shell packing.

The on-disk basis is four s, two p, two d and one f radial shells, in
spin-major order. The official transform consumes shell-pair-major order
within each of (up-up, up-down, down-up, down-down). These orders differ.
"""
import torch
from torch import nn
from e3nn import o3
from .upstream import activateNextHam

activateNextHam()
from tg_src.e3modules import e3TensorDecomp, Rotate
from tg_src.utils import irreps_from_l1l2


class SocCodec(nn.Module):
    shellLs = (0, 0, 0, 0, 1, 1, 2, 2, 3)

    def __init__(self, device="cpu", dtype=torch.float32):
        super().__init__()
        if dtype not in (torch.float32, torch.float64):
            raise ValueError("SOC codec requires float32 or float64 coefficients")
        self.register_buffer("anchor", torch.empty(0, device=device, dtype=dtype), persistent=False)
        self.shellSlices = []
        offset = 0
        for degree in self.shellLs:
            self.shellSlices.append(slice(offset, offset + 2 * degree + 1))
            offset += 2 * degree + 1
        self.shellPairs = [(a, b) for a in range(9) for b in range(9)]
        self.backend = None
        self.backendKey = None
        self.ensureBackend()

        unsorted = o3.Irreps()
        labels = []
        for a, b in self.shellPairs:
            _, orbital, vector = irreps_from_l1l2(self.shellLs[a], self.shellLs[b], 1, True)
            branch = orbital
            for part in vector:
                branch += part
            unsorted += branch
            labels.extend([(a, b)] * len(branch))
        unsorted += unsorted
        labels += labels
        order = unsorted.sort().inv
        self.irrepCopies = o3.Irreps([unsorted[index] for index in order])
        self.irrepSlices = self.irrepCopies.slices()
        self.register_buffer("irrepDimensions", torch.tensor([ir.dim for _, ir in self.irrepCopies], dtype=torch.long))
        self.register_buffer("irrepToShellPair", torch.tensor([labels[index] for index in order], dtype=torch.long))
        self.register_buffer("coefficientToIrrep", torch.repeat_interleave(
            torch.arange(len(self.irrepCopies)), self.irrepDimensions))
        self.irreps = self.backend.required_irreps_out
        if self.irreps.dim != 5832 or self.irrepCopies.simplify() != self.irreps:
            raise RuntimeError("Unexpected official full SOC representation")
        self.to(device=device)

    @property
    def device(self):
        return self.anchor.device

    @property
    def dtype(self):
        return self.anchor.dtype

    def ensureBackend(self):
        key = (str(self.anchor.device), self.anchor.dtype)
        if key != self.backendKey:
            self.backend = e3TensorDecomp(
                None, [(self.shellLs[a], self.shellLs[b]) for a, b in self.shellPairs],
                self.anchor.dtype, spinful=True, no_parity=False, if_sort=True,
                device_torch=self.anchor.device)
            self.backendKey = key
        return self.backend

    def pack(self, raw):
        if raw.ndim != 3 or raw.shape[1:] != (54, 54):
            raise ValueError(f"Expected [edges,54,54], got {tuple(raw.shape)}")
        spinBlocks = raw.reshape(-1, 2, 27, 2, 27).permute(0, 1, 3, 2, 4).reshape(-1, 4, 27, 27)
        return torch.cat([
            spinBlocks[:, :, self.shellSlices[a], self.shellSlices[b]].flatten(2)
            for a, b in self.shellPairs], dim=-1)

    def unpack(self, packed):
        rows = []
        offset = 0
        for a in range(9):
            columns = []
            na = 2 * self.shellLs[a] + 1
            for b in range(9):
                nb = 2 * self.shellLs[b] + 1
                columns.append(packed[..., offset:offset + na * nb].reshape(-1, 4, na, nb))
                offset += na * nb
            rows.append(torch.cat(columns, dim=-1))
        matrix = torch.cat(rows, dim=-2)
        return matrix.reshape(-1, 2, 2, 27, 27).permute(0, 1, 3, 2, 4).reshape(-1, 54, 54)

    def encode(self, raw):
        backend = self.ensureBackend()
        if raw.device != self.anchor.device:
            raise ValueError("Move the codec and matrices to the same device")
        return backend.get_net_out(self.pack(raw).to(backend.dtype))

    def decode(self, coefficients):
        backend = self.ensureBackend()
        if coefficients.shape[-1] != self.irreps.dim:
            raise ValueError("Incorrect SOC coefficient dimension")
        return self.unpack(backend.get_H(coefficients))

    def validIrreps(self, support):
        """Only complete shell pairs produce complete, rotation-closed irreps."""
        packed = self.pack(support.bool())
        validity = []
        offset = 0
        for a, b in self.shellPairs:
            size = (2 * self.shellLs[a] + 1) * (2 * self.shellLs[b] + 1)
            part = packed[..., offset:offset + size]
            complete = part.flatten(1).all(1)
            absent = ~part.flatten(1).any(1)
            if not bool((complete | absent).all()):
                raise ValueError("Support splits a radial shell or spin sector; a complete SOC irrep is undefined")
            validity.append(complete)
            offset += size
        byPair = torch.stack(validity, dim=-1)
        indices = self.irrepToShellPair[:, 0] * 9 + self.irrepToShellPair[:, 1]
        return byPair[:, indices]

    def rotate(self, raw, rotationXYZ):
        """Official row-vector convention: geometry transforms as vectors @ R.

        The upstream rotation helper builds some CPU constants. Run its exact
        transform on CPU, then transfer back; this remains differentiable.
        """
        complexDtype = torch.complex64 if self.anchor.dtype == torch.float32 else torch.complex128
        kernel = Rotate(complexDtype, spinful=True, device_torch="cpu")
        rotationXYZ = rotationXYZ.to(device="cpu", dtype=self.anchor.dtype)
        # Upstream rotate_matrix_convert uses float32 identity even for float64.
        order = torch.eye(3, dtype=self.anchor.dtype)[[1, 2, 0]]
        rotationYzx = order @ rotationXYZ @ order.T
        result = kernel.rotate_openmx_H_full(
            raw.to(device="cpu", dtype=complexDtype), rotationYzx,
            self.shellLs, self.shellLs, order_xyz=False)
        return result.to(raw.device)

    def forward(self, raw):
        return self.encode(raw)
