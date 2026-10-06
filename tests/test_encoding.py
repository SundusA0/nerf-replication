import math

import torch

from nerf.encoding import encoding_dim, positional_encoding


def test_paper_dimensions():
    # 3D position at L = 10 -> 63 numbers; 3D direction at L = 4 -> 27 numbers.
    x = torch.zeros(5, 3)
    assert positional_encoding(x, 10).shape == (5, 63)
    assert positional_encoding(x, 4).shape == (5, 27)
    assert encoding_dim(3, 10) == 63
    assert encoding_dim(3, 4) == 27
    assert encoding_dim(3, 10, include_input=False) == 60


def test_matches_definition():
    # One coordinate, p = pi / 2, L = 2:
    # (p, sin p, cos p, sin 2p, cos 2p) = (pi/2, 1, 0, 0, -1)
    p = torch.tensor([[math.pi / 2]], dtype=torch.float64)
    expected = torch.tensor([[math.pi / 2, 1.0, 0.0, 0.0, -1.0]], dtype=torch.float64)
    assert torch.allclose(positional_encoding(p, 2), expected, atol=1e-12)


def test_no_factor_of_pi():
    # Follows the released code, sin(2^k p), not the paper's sin(2^k pi p):
    # at p = 1 the first sine is sin(1), not sin(pi) = 0.
    p = torch.ones(1, 1, dtype=torch.float64)
    encoded = positional_encoding(p, 1, include_input=False)
    assert math.isclose(encoded[0, 0].item(), math.sin(1.0))
    assert math.isclose(encoded[0, 1].item(), math.cos(1.0))


def test_layout_is_per_frequency_blocks():
    # For a 3-vector the output is [x | sin(x) | cos(x) | sin(2x) | cos(2x) | ...],
    # each block holding all three coordinates.
    x = torch.tensor([[0.1, 0.2, 0.3]], dtype=torch.float64)
    encoded = positional_encoding(x, 2)
    assert torch.allclose(encoded[:, 0:3], x)
    assert torch.allclose(encoded[:, 3:6], torch.sin(x))
    assert torch.allclose(encoded[:, 6:9], torch.cos(x))
    assert torch.allclose(encoded[:, 9:12], torch.sin(2 * x))
    assert torch.allclose(encoded[:, 12:15], torch.cos(2 * x))


def test_batch_dimensions_are_preserved():
    x = torch.randn(4, 7, 3)
    assert positional_encoding(x, 10).shape == (4, 7, 63)


def test_zero_frequencies_returns_input():
    x = torch.randn(4, 3)
    assert torch.equal(positional_encoding(x, 0), x)
