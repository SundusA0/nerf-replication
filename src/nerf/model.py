"""The NeRF network (Mildenhall et al. 2020, Section 3 and Fig. 7).

A scene is stored as a function, not as voxels or a mesh:

    F_theta : (x, d) -> (sigma, c)

x is a 3D position, d the direction it is viewed from, sigma the volume density
at x and c the colour x emits towards d. F_theta is a small fully connected
network and theta, its weights, are the only thing that is trained. One network
represents one scene.
"""

from __future__ import annotations

import torch
from torch import nn

from nerf.encoding import encoding_dim, positional_encoding


class NeRF(nn.Module):
    """Fully connected network from (position, direction) to (density, colour).

    With the default sizes, following Fig. 7:

      1. The encoded position gamma(x), 63 numbers, goes through 8 linear
         layers of width 256, each followed by a ReLU. gamma(x) is concatenated
         back onto the output of the fifth layer (a skip connection), so the
         sixth layer has 256 + 63 inputs.
      2. One linear layer reads the density sigma off the result, with a ReLU
         to keep it non-negative.
      3. A second linear layer produces a 256-number feature vector. The
         encoded direction gamma(d), 27 numbers, is concatenated onto it, and
         one ReLU layer of width 128 followed by a linear layer and a sigmoid
         gives RGB in [0, 1].

    Density is computed before the direction enters, so it depends on position
    only. That is deliberate: geometry must look the same from every viewpoint.
    Colour may depend on direction, which is how the network represents
    highlights and other view-dependent effects.

    Weights start Glorot-uniform with zero biases. That is what the released
    TensorFlow code uses (the Keras default for dense layers), and it differs
    from PyTorch's own default for nn.Linear.

    Args:
        depth: number of layers in the position branch (8 in the paper).
        width: units per layer (256 in the paper).
        skip: index, counting from 0, of the layer after which gamma(x) is
            concatenated back in. 4 means after the fifth layer.
        num_freqs_pos: L for positions (10 in the paper).
        num_freqs_dir: L for directions (4 in the paper).
    """

    def __init__(
        self,
        depth: int = 8,
        width: int = 256,
        skip: int = 4,
        num_freqs_pos: int = 10,
        num_freqs_dir: int = 4,
    ) -> None:
        super().__init__()
        self.skip = skip
        self.num_freqs_pos = num_freqs_pos
        self.num_freqs_dir = num_freqs_dir
        pos_dim = encoding_dim(3, num_freqs_pos)
        dir_dim = encoding_dim(3, num_freqs_dir)

        layers = []
        in_dim = pos_dim
        for i in range(depth):
            layers.append(nn.Linear(in_dim, width))
            in_dim = width + pos_dim if i == skip else width
        self.trunk = nn.ModuleList(layers)

        self.sigma_head = nn.Linear(in_dim, 1)
        self.feature = nn.Linear(in_dim, width)
        self.dir_layer = nn.Linear(width + dir_dim, width // 2)
        self.rgb_head = nn.Linear(width // 2, 3)

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(
        self, points: torch.Tensor, view_dirs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Evaluate the field.

        Args:
            points: (..., 3) positions.
            view_dirs: (..., 3) unit viewing directions, same shape as points.

        Returns:
            sigma: (...) densities, >= 0.
            rgb: (..., 3) colours in [0, 1].
        """
        x = positional_encoding(points, self.num_freqs_pos)
        d = positional_encoding(view_dirs, self.num_freqs_dir)

        h = x
        for i, layer in enumerate(self.trunk):
            h = torch.relu(layer(h))
            if i == self.skip:
                h = torch.cat([x, h], dim=-1)

        sigma = torch.relu(self.sigma_head(h)).squeeze(-1)

        h = self.feature(h)
        h = torch.relu(self.dir_layer(torch.cat([h, d], dim=-1)))
        rgb = torch.sigmoid(self.rgb_head(h))
        return sigma, rgb
