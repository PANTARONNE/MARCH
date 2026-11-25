"""Four original NextHAM transformer blocks with reconstruction TraceGrad."""
import torch
from torch import nn
from e3nn import o3
from march.codec.upstream import activateNextHam

activateNextHam()
from nets.graph_attention_transformer import TransBlock
from nets.tracegrad import Equi_Nonlin_Grad_Module


class GaussianRadialBasis(nn.Module):
    def __init__(self, maximum=10.0, count=64):
        super().__init__()
        self.register_buffer("centers", torch.linspace(0, maximum, count))
        self.width = 0.5 * maximum / count

    def forward(self, distances):
        return torch.exp(-((distances[:, None] - self.centers) ** 2) / (2 * self.width ** 2))


class NextHamBackbone(nn.Module):
    def __init__(self, hiddenIrreps, numLayers=4, numHeads=4,
                 radialBasis=64, radialMaximum=10.0, traceInvariant=256, traceHidden=1024):
        super().__init__()
        if numHeads != 4 or radialBasis != 64:
            raise ValueError("Pinned NextHAM attention internally requires 4 heads and 64 radial bases")
        if numLayers != 4:
            raise ValueError("Initial reconstruction backbone uses exactly four NextHAM blocks")
        self.irreps = o3.Irreps(hiddenIrreps)
        self.geometryIrreps = o3.Irreps.spherical_harmonics(5)
        self.radial = GaussianRadialBasis(radialMaximum, radialBasis)
        self.blocks = nn.ModuleList()
        self.traceGrad = nn.ModuleList()
        for index in range(numLayers):
            block = TransBlock(
                irreps_node_input=self.irreps, irreps_node_attr="1x0e",
                irreps_edge_attr=self.irreps, irreps_node_output=self.irreps,
                irreps_edge_output=self.irreps, fc_neurons=[radialBasis, 64, 64],
                irreps_head=self.irreps, num_heads=numHeads, irreps_pre_attn=self.irreps,
                rescale_degree=False, nonlinear_message=True,
                alpha_drop=0.0, proj_drop=0.0, drop_path_rate=0.0,
                irreps_mlp_mid=self.irreps, norm_layer="layer")
            # Upstream initializes several modules its forward does not invoke.
            # Freeze them in the adapter, preserving the vendored source exactly.
            for name in ("ga_edge", "update_edge", "node_update", "edge_update"):
                getattr(block, name).requires_grad_(False)
            block.ga.alpha_dot.requires_grad_(False)
            self.blocks.append(block)
            if index < numLayers - 1:
                self.traceGrad.append(Equi_Nonlin_Grad_Module(
                    self.irreps, z_fea_len=traceInvariant, hidden_fea_len=traceHidden))

    def geometricFeatures(self, edgeVec):
        return o3.spherical_harmonics(
            self.geometryIrreps, edgeVec[:, [1, 2, 0]],
            normalize=True, normalization="component")

    def forward(self, nodeFeatures, edgeFeatures, modelInput, edgeSh):
        source, target = modelInput["edgeIndex"]
        edgeLength = modelInput["edgeVec"].norm(dim=-1)
        radial = self.radial(edgeLength)
        nodeAttr = nodeFeatures.new_ones((nodeFeatures.shape[0], 1))
        batch = source.new_zeros(nodeFeatures.shape[0])
        for index, block in enumerate(self.blocks):
            nodeFeatures, edgeFeatures = block(
                node_input=nodeFeatures, node_attr=nodeAttr, edge_src=source,
                edge_dst=target, edge_attr=edgeFeatures, edge_sh=edgeSh,
                edge_scalars=radial, batch=batch, edge_length=edgeLength)
            if index < len(self.traceGrad):
                if not edgeFeatures.requires_grad:
                    edgeFeatures = edgeFeatures.detach().requires_grad_(True)
                _, enhanced = self.traceGrad[index](edgeFeatures, retain_graph=True)
                scale = (edgeFeatures.norm(dim=-1, keepdim=True) /
                         (enhanced.norm(dim=-1, keepdim=True) + 1e-5)).detach()
                edgeFeatures = edgeFeatures + 0.2 * scale * enhanced
        return nodeFeatures, edgeFeatures
