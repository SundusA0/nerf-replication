"""Positional encoding (Mildenhall et al. 2020, Section 5.1, Eq. 4)."""

from __future__ import annotations

import torch


def positional_encoding(
    x: torch.Tensor, num_freqs: int, include_input: bool = True
) -> torch.Tensor:
    """Lift each coordinate to sines and cosines at octave-spaced frequencies.

        gamma(p) = ( p, sin(2^0 p), cos(2^0 p), ..., sin(2^(L-1) p), cos(2^(L-1) p) )

    An MLP fed raw (x, y, z) learns smooth, low-frequency functions first and
    blurs fine detail. Feeding it this higher-dimensional encoding instead lets
    the same network represent sharp geometry and texture.

    The released code differs from Eq. 4 of the paper in two ways. This follows
    the code, since that is what produced the published numbers:
      * the paper writes sin(2^k pi p); the code uses sin(2^k p), with no pi;
      * the code also concatenates the raw input p in front.

    Args:
        x: (..., D) coordinates.
        num_freqs: L, the number of octaves. The paper uses L = 10 for
            positions and L = 4 for viewing directions.
        include_input: put x itself in front of the sines and cosines.

    Returns:
        (..., encoding_dim(D, L, include_input)). For D = 3 that is 63 numbers
        per point at L = 10 and 27 per direction at L = 4.
    """
    parts = [x] if include_input else []
    for k in range(num_freqs):
        freq = 2.0**k
        parts.append(torch.sin(freq * x))
        parts.append(torch.cos(freq * x))
    return torch.cat(parts, dim=-1)


def encoding_dim(input_dim: int, num_freqs: int, include_input: bool = True) -> int:
    """Length of the encoded vector, used to size the first layer of the MLP."""
    return input_dim * (2 * num_freqs + int(include_input))
