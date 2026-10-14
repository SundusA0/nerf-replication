"""Multiresolution hash encoding (Mueller, Evans, Schied and Keller 2022,
"Instant Neural Graphics Primitives with a Multiresolution Hash Encoding",
Section 3).

NeRF's frequency encoding turns a position into sines and cosines that never
change, and leaves it to a large network to make density and colour out of
them. This encoding is itself learnt. It keeps a vector of F numbers at every
vertex of L grids of growing resolution, and a position's features are the
trilinear interpolation of the 8 vertices of the cell it falls in, at each
level, concatenated: L * F numbers. A much smaller network then suffices, and
training is faster because a step only touches the entries around the points
it sampled.

Each level's vertices live in a table of T entries. A coarse level, whose
vertices all fit, is indexed directly. A fine level has more vertices than
entries and is indexed by the hash of Eq. 4,

    h(x) = (x_1 * pi_1  xor  x_2 * pi_2  xor  x_3 * pi_3)  mod T

with pi_1 = 1, pi_2 = 2654435761 and pi_3 = 805459861, so distinct vertices
share entries. The paper shows that training sorts the collisions out: the
entries that matter get the strongest gradients, and most of a scene is empty
space whose entries need no particular value.
"""

from __future__ import annotations

import math

import torch
from torch import nn

HASH_PRIMES = (1, 2654435761, 805459861)

# The 8 corners of a cell, as offsets from its lowest corner.
CORNERS = torch.tensor([[i, j, k] for i in (0, 1) for j in (0, 1) for k in (0, 1)])


def level_resolutions(num_levels: int, base: int, finest: int) -> list[int]:
    """Grid resolution N_l of each level, from `base` at the coarsest to
    `finest` at the finest, growing geometrically (Eq. 2 and 3 of the paper):

        N_l = floor(base * b^l),   b = exp((ln finest - ln base) / (L - 1)).

    N_l is the number of cells along each axis, so a level has N_l + 1
    vertices along each axis.
    """
    if num_levels < 1 or base < 1 or finest < base:
        raise ValueError("need at least one level and 1 <= base <= finest")
    if num_levels == 1:
        return [base]
    growth = math.exp((math.log(finest) - math.log(base)) / (num_levels - 1))
    return [int(math.floor(base * growth**level + 1e-6)) for level in range(num_levels)]


def vertex_indices(vertices: torch.Tensor, resolution: int, table_size: int) -> torch.Tensor:
    """Table entry of each grid vertex of a level with `resolution` cells per axis.

    Args:
        vertices: (..., 3) integer vertex coordinates, each in 0..resolution.
        resolution: N_l, cells per axis.
        table_size: T, entries in the level's table.

    Returns:
        (...) integer entries in 0..T-1. Direct when all (N_l + 1)^3 vertices
        fit in the table, hashed (Eq. 4) when they do not.
    """
    per_axis = resolution + 1
    if per_axis**3 <= table_size:
        return (vertices[..., 0] * per_axis + vertices[..., 1]) * per_axis + vertices[..., 2]
    hashed = vertices[..., 0] * HASH_PRIMES[0]
    hashed = hashed ^ (vertices[..., 1] * HASH_PRIMES[1])
    hashed = hashed ^ (vertices[..., 2] * HASH_PRIMES[2])
    return hashed % table_size


class HashGridEncoding(nn.Module):
    """Positions in the box [-bound, bound]^3 to L * F learnt features.

    Args:
        num_levels: L. The paper uses 16.
        features_per_level: F. The paper uses 2.
        log2_table_size: T = 2^this entries per level. The paper uses 2^14 to
            2^24 depending on the scene; 2^19 for most.
        base_resolution: N_min, cells per axis at the coarsest level (16).
        finest_resolution: N_max, cells per axis at the finest level. The
            paper uses 512 to 524288 depending on the task; 2048 is the
            released code's default for a scene in a unit box (growth factor
            b = 1.38 over 16 levels).
        bound: the scene lies in [-bound, bound]^3. `forward` also says which
            points fall outside; the model treats them as empty.

    The tables start with small random values, uniform in [-1e-4, 1e-4] as in
    the paper, so that at first every position looks alike and the network's
    biases decide the initial density and colour.
    """

    def __init__(
        self,
        num_levels: int = 16,
        features_per_level: int = 2,
        log2_table_size: int = 19,
        base_resolution: int = 16,
        finest_resolution: int = 2048,
        bound: float = 1.5,
    ) -> None:
        super().__init__()
        if features_per_level < 1 or log2_table_size < 1 or bound <= 0:
            raise ValueError("need at least one feature per level, a table of at least 2 entries and a positive bound")
        self.resolutions = level_resolutions(num_levels, base_resolution, finest_resolution)
        self.table_size = 2**log2_table_size
        self.features_per_level = features_per_level
        self.bound = float(bound)
        self.tables = nn.Parameter(
            torch.empty(num_levels, self.table_size, features_per_level).uniform_(-1e-4, 1e-4)
        )
        # Per level: the resolution, whether its vertices are indexed directly,
        # and where its table starts when all tables are seen as one.
        resolutions = torch.tensor(self.resolutions)
        self.register_buffer("resolution", resolutions.to(torch.float32), persistent=False)
        self.register_buffer("per_axis", resolutions + 1, persistent=False)
        self.register_buffer("direct", (resolutions + 1) ** 3 <= self.table_size, persistent=False)
        self.register_buffer("offset", torch.arange(num_levels) * self.table_size, persistent=False)

    @property
    def output_dim(self) -> int:
        return len(self.resolutions) * self.features_per_level

    def forward(self, points: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode positions.

        Args:
            points: (..., 3) positions in scene coordinates.

        Returns:
            features: (..., L * F), the levels' interpolated features in order.
            inside: (...) bool, False for points outside the box. Those are
                encoded as if clamped to the box, which the caller should
                treat as empty space.
        """
        unit = (points + self.bound) / (2.0 * self.bound)         # [0, 1]^3 inside the box
        inside = ((unit >= 0.0) & (unit <= 1.0)).all(dim=-1)
        unit = unit.clamp(0.0, 1.0)

        # All levels at once: (..., L, 3) scaled positions, the cell each
        # falls in (0 .. N_l - 1 on each axis) and the position within it.
        scaled = unit[..., None, :] * self.resolution[:, None]
        lower = scaled.floor().clamp(max=self.resolution[:, None] - 1.0)
        fraction = scaled - lower
        lower = lower.long()

        # The 8 corners of the cell, in the order (x, y, z) = (0,0,0), (0,0,1),
        # (0,1,0), ..., (1,1,1): each axis contributes its lower or upper
        # coordinate, combined by broadcasting (..., L, 2, 2, 2) -> (..., L, 8).
        # A level's table entries come from per-axis terms combined with xor
        # (the hash) or by place value (direct indexing), whichever it uses.
        ends = torch.stack([lower, lower + 1], dim=-1)                         # (..., L, 3, 2)
        x, y, z = ends[..., 0, :], ends[..., 1, :], ends[..., 2, :]
        per_axis = self.per_axis[:, None]
        direct = ((x * per_axis * per_axis)[..., :, None, None] + (y * per_axis)[..., None, :, None]
                  + z[..., None, None, :])
        hashed = (((x * HASH_PRIMES[0])[..., :, None, None] ^ (y * HASH_PRIMES[1])[..., None, :, None]
                   ^ (z * HASH_PRIMES[2])[..., None, None, :]) & (self.table_size - 1))
        entries = torch.where(self.direct[:, None, None, None], direct, hashed).flatten(-3)
        entries = entries + self.offset[:, None]                                # (..., L, 8)

        # Trilinear weights: (1 - f) for a lower corner and f for an upper one,
        # multiplied over the three axes, in the same corner order.
        ends = torch.stack([1.0 - fraction, fraction], dim=-1)                 # (..., L, 3, 2)
        weights = (ends[..., 0, :, None, None] * ends[..., 1, None, :, None]
                   * ends[..., 2, None, None, :]).flatten(-3)                  # (..., L, 8)

        features = torch.index_select(self.tables.flatten(0, 1), 0, entries.flatten())
        features = features.view(*entries.shape, self.features_per_level)     # (..., L, 8, F)
        return (weights[..., None] * features).sum(dim=-2).flatten(-2), inside


class _TruncatedExp(torch.autograd.Function):
    """exp(x) with x cut off at 15, in the gradient as well as the value.

    Instant-NGP reads density as exp of a network output, which can run away
    early in training; cutting the exponent off keeps the density, and the
    gradient through it, finite. Unlike exp(clamp(x)), this still passes a
    gradient where x is above the cut, so an entry that shot up can come back.
    """

    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return torch.exp(x.clamp(max=15.0))

    @staticmethod
    def backward(ctx, grad):
        (x,) = ctx.saved_tensors
        return grad * torch.exp(x.clamp(max=15.0))


def truncated_exp(x: torch.Tensor) -> torch.Tensor:
    return _TruncatedExp.apply(x)
