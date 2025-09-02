"""MinkowskiUNet34 backbone — the paper's default (Sec. 6).

The paper builds its point-voxel backbone on **Minkowski-UNet34** (Choy et al.,
4D Spatio-Temporal ConvNets), re-implemented by the PCSeg codebase. This module
provides the full sparse-convolution MinkUNet34 via MinkowskiEngine and is the
**default backbone on NVIDIA CUDA GPUs**.

MinkowskiEngine requires CUDA and does not build on CPU-only / Apple-Silicon
setups, so the `import` is guarded: when ME is unavailable the rest of the
package still works and the pipeline falls back to the light voxel backbone
(see `rapid_seg.models.backbone.VoxelBackbone`). Architecture follows the
canonical ME `MinkUNet34C` definition:

    BLOCK   = BasicBlock
    LAYERS  = (2, 3, 4, 6, 2, 2, 2, 2)
    PLANES  = (32, 64, 128, 256, 256, 128, 96, 96)
    INIT_DIM = 32
"""
from __future__ import annotations

import torch
import torch.nn as nn

try:  # ME is CUDA-only; guard so the package imports everywhere.
    import MinkowskiEngine as ME
    _HAS_ME = True
except Exception:  # pragma: no cover - exercised only off-CUDA
    ME = None
    _HAS_ME = False


def minkowski_available() -> bool:
    """True iff MinkowskiEngine imported AND a CUDA device is present."""
    return _HAS_ME and torch.cuda.is_available()


if _HAS_ME:

    class BasicBlock(nn.Module):
        expansion = 1

        def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, dimension=3):
            super().__init__()
            self.conv1 = ME.MinkowskiConvolution(
                inplanes, planes, kernel_size=3, stride=stride, dilation=dilation, dimension=dimension)
            self.norm1 = ME.MinkowskiBatchNorm(planes)
            self.conv2 = ME.MinkowskiConvolution(
                planes, planes, kernel_size=3, stride=1, dilation=dilation, dimension=dimension)
            self.norm2 = ME.MinkowskiBatchNorm(planes)
            self.relu = ME.MinkowskiReLU(inplace=True)
            self.downsample = downsample

        def forward(self, x):
            residual = x
            out = self.relu(self.norm1(self.conv1(x)))
            out = self.norm2(self.conv2(out))
            if self.downsample is not None:
                residual = self.downsample(x)
            return self.relu(out + residual)

    class MinkUNet34(nn.Module):
        """Full MinkowskiUNet34 (Res16UNet34C configuration)."""

        BLOCK = BasicBlock
        LAYERS = (2, 3, 4, 6, 2, 2, 2, 2)
        PLANES = (32, 64, 128, 256, 256, 128, 96, 96)
        INIT_DIM = 32

        def __init__(self, in_channels: int, out_channels: int, D: int = 3):
            super().__init__()
            self.D = D
            block, planes, layers = self.BLOCK, self.PLANES, self.LAYERS
            self.inplanes = self.INIT_DIM

            self.conv0p1s1 = ME.MinkowskiConvolution(in_channels, self.inplanes, kernel_size=5, dimension=D)
            self.bn0 = ME.MinkowskiBatchNorm(self.inplanes)

            # ---- encoder ----
            self.conv1p1s2 = ME.MinkowskiConvolution(self.inplanes, self.inplanes, kernel_size=2, stride=2, dimension=D)
            self.bn1 = ME.MinkowskiBatchNorm(self.inplanes)
            self.block1 = self._make_layer(block, planes[0], layers[0])

            self.conv2p2s2 = ME.MinkowskiConvolution(self.inplanes, self.inplanes, kernel_size=2, stride=2, dimension=D)
            self.bn2 = ME.MinkowskiBatchNorm(self.inplanes)
            self.block2 = self._make_layer(block, planes[1], layers[1])

            self.conv3p4s2 = ME.MinkowskiConvolution(self.inplanes, self.inplanes, kernel_size=2, stride=2, dimension=D)
            self.bn3 = ME.MinkowskiBatchNorm(self.inplanes)
            self.block3 = self._make_layer(block, planes[2], layers[2])

            self.conv4p8s2 = ME.MinkowskiConvolution(self.inplanes, self.inplanes, kernel_size=2, stride=2, dimension=D)
            self.bn4 = ME.MinkowskiBatchNorm(self.inplanes)
            self.block4 = self._make_layer(block, planes[3], layers[3])

            # ---- decoder ----
            self.convtr4p16s2 = ME.MinkowskiConvolutionTranspose(self.inplanes, planes[4], kernel_size=2, stride=2, dimension=D)
            self.bntr4 = ME.MinkowskiBatchNorm(planes[4])
            self.inplanes = planes[4] + planes[2] * block.expansion
            self.block5 = self._make_layer(block, planes[4], layers[4])

            self.convtr5p8s2 = ME.MinkowskiConvolutionTranspose(self.inplanes, planes[5], kernel_size=2, stride=2, dimension=D)
            self.bntr5 = ME.MinkowskiBatchNorm(planes[5])
            self.inplanes = planes[5] + planes[1] * block.expansion
            self.block6 = self._make_layer(block, planes[5], layers[5])

            self.convtr6p4s2 = ME.MinkowskiConvolutionTranspose(self.inplanes, planes[6], kernel_size=2, stride=2, dimension=D)
            self.bntr6 = ME.MinkowskiBatchNorm(planes[6])
            self.inplanes = planes[6] + planes[0] * block.expansion
            self.block7 = self._make_layer(block, planes[6], layers[6])

            self.convtr7p2s2 = ME.MinkowskiConvolutionTranspose(self.inplanes, planes[7], kernel_size=2, stride=2, dimension=D)
            self.bntr7 = ME.MinkowskiBatchNorm(planes[7])
            self.inplanes = planes[7] + self.INIT_DIM
            self.block8 = self._make_layer(block, planes[7], layers[7])

            self.final = ME.MinkowskiConvolution(
                planes[7] * block.expansion, out_channels, kernel_size=1, bias=True, dimension=D)
            self.relu = ME.MinkowskiReLU(inplace=True)

        def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
            downsample = None
            if stride != 1 or self.inplanes != planes * block.expansion:
                downsample = nn.Sequential(
                    ME.MinkowskiConvolution(self.inplanes, planes * block.expansion,
                                            kernel_size=1, stride=stride, dimension=self.D),
                    ME.MinkowskiBatchNorm(planes * block.expansion),
                )
            layers = [block(self.inplanes, planes, stride=stride, dilation=dilation,
                            downsample=downsample, dimension=self.D)]
            self.inplanes = planes * block.expansion
            for _ in range(1, blocks):
                layers.append(block(self.inplanes, planes, stride=1, dilation=dilation, dimension=self.D))
            return nn.Sequential(*layers)

        def forward(self, x):
            out = self.relu(self.bn0(self.conv0p1s1(x)))
            out_p1 = out

            out = self.relu(self.bn1(self.conv1p1s2(out_p1)))
            out_b1p2 = self.block1(out)
            out = self.relu(self.bn2(self.conv2p2s2(out_b1p2)))
            out_b2p4 = self.block2(out)
            out = self.relu(self.bn3(self.conv3p4s2(out_b2p4)))
            out_b3p8 = self.block3(out)
            out = self.relu(self.bn4(self.conv4p8s2(out_b3p8)))
            out = self.block4(out)

            out = self.relu(self.bntr4(self.convtr4p16s2(out)))
            out = ME.cat(out, out_b3p8)
            out = self.block5(out)

            out = self.relu(self.bntr5(self.convtr5p8s2(out)))
            out = ME.cat(out, out_b2p4)
            out = self.block6(out)

            out = self.relu(self.bntr6(self.convtr6p4s2(out)))
            out = ME.cat(out, out_b1p2)
            out = self.block7(out)

            out = self.relu(self.bntr7(self.convtr7p2s2(out)))
            out = ME.cat(out, out_p1)
            out = self.block8(out)
            return self.final(out)


class MinkUNetBackbone(nn.Module):
    """Wrapper giving MinkUNet34 the uniform (feats, coords, voxel_idx) interface.

    Builds a ME.SparseTensor from the per-voxel features + integer voxel
    coordinates, runs the full MinkUNet34, and broadcasts the per-voxel logits
    back to points via the point->voxel index. Voxel coordinates are unique
    (from `voxelize`), so the output row order matches the input voxel order.
    """

    def __init__(self, in_dim: int, num_classes: int, dimension: int = 3):
        super().__init__()
        if not _HAS_ME:
            raise RuntimeError("MinkowskiEngine is not available; use the voxel backbone.")
        self.net = MinkUNet34(in_dim, num_classes, D=dimension)

    def forward(self, voxel_feats: torch.Tensor, voxel_coords: torch.Tensor, voxel_idx: torch.Tensor):
        # batch column (single frame per step -> batch id 0)
        batch = torch.zeros((voxel_coords.shape[0], 1), dtype=torch.int32, device=voxel_coords.device)
        coords = torch.cat([batch, voxel_coords.int()], dim=1)
        st = ME.SparseTensor(features=voxel_feats, coordinates=coords)
        out = self.net(st)
        return out.F[voxel_idx]
