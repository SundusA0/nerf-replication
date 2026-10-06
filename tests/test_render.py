import math

import torch

from nerf.render import stratified_samples, volume_render

F64 = torch.float64
NEAR, FAR = 2.0, 6.0


def unit_dirs(num_rays):
    return torch.tensor([[0.0, 0.0, -1.0]], dtype=F64).expand(num_rays, 3)


# --------------------------------------------------------------------------
# Stratified sampling
# --------------------------------------------------------------------------


def test_unperturbed_samples_are_evenly_spaced():
    t = stratified_samples(NEAR, FAR, num_rays=3, num_samples=5, perturb=False, dtype=F64)
    expected = torch.tensor([2.0, 3.0, 4.0, 5.0, 6.0], dtype=F64).expand(3, 5)
    assert torch.allclose(t, expected)


def test_perturbed_samples_stay_in_their_bins():
    n = 64
    gen = torch.Generator().manual_seed(0)
    t = stratified_samples(NEAR, FAR, 1000, n, generator=gen, dtype=F64)

    grid = torch.linspace(NEAR, FAR, n, dtype=F64)
    mids = 0.5 * (grid[1:] + grid[:-1])
    lower = torch.cat([grid[:1], mids])
    upper = torch.cat([mids, grid[-1:]])
    assert (t >= lower).all() and (t <= upper).all()
    # sorted along each ray, and different from ray to ray
    assert (t[:, 1:] >= t[:, :-1]).all()
    assert not torch.allclose(t[0], t[1])


def test_samples_are_uniform_within_bins():
    # Averaged over many rays, sample i sits at the centre of bin i.
    n = 16
    gen = torch.Generator().manual_seed(1)
    t = stratified_samples(NEAR, FAR, 50_000, n, generator=gen, dtype=F64)

    grid = torch.linspace(NEAR, FAR, n, dtype=F64)
    mids = 0.5 * (grid[1:] + grid[:-1])
    lower = torch.cat([grid[:1], mids])
    upper = torch.cat([mids, grid[-1:]])
    assert torch.allclose(t.mean(dim=0), 0.5 * (lower + upper), atol=2e-3)


def test_same_seed_gives_same_samples():
    a = stratified_samples(NEAR, FAR, 10, 16, generator=torch.Generator().manual_seed(7))
    b = stratified_samples(NEAR, FAR, 10, 16, generator=torch.Generator().manual_seed(7))
    c = stratified_samples(NEAR, FAR, 10, 16, generator=torch.Generator().manual_seed(8))
    assert torch.equal(a, b)
    assert not torch.equal(a, c)


# --------------------------------------------------------------------------
# Volume rendering
# --------------------------------------------------------------------------


def test_empty_space_is_transparent():
    rays, n = 4, 32
    t = stratified_samples(NEAR, FAR, rays, n, perturb=False, dtype=F64)
    sigma = torch.zeros(rays, n, dtype=F64)
    rgb = torch.rand(rays, n, 3, dtype=F64)

    black = volume_render(sigma, rgb, t, unit_dirs(rays))
    assert torch.equal(black.acc, torch.zeros(rays, dtype=F64))
    assert torch.equal(black.rgb, torch.zeros(rays, 3, dtype=F64))

    white = volume_render(sigma, rgb, t, unit_dirs(rays), white_background=True)
    assert torch.equal(white.rgb, torch.ones(rays, 3, dtype=F64))


def test_opaque_surface_returns_its_colour_and_depth():
    # Vacuum, then a wall at sample k: the pixel is the wall's colour and the
    # depth is where the wall is. Nothing behind the wall contributes.
    n, k = 64, 20
    t = stratified_samples(NEAR, FAR, 1, n, perturb=False, dtype=F64)
    sigma = torch.zeros(1, n, dtype=F64)
    sigma[:, k:] = 1e6
    rgb = torch.rand(1, n, 3, dtype=F64)

    out = volume_render(sigma, rgb, t, unit_dirs(1))
    assert torch.allclose(out.rgb, rgb[:, k], atol=1e-6)
    assert torch.allclose(out.depth, t[:, k], atol=1e-6)
    assert torch.allclose(out.acc, torch.ones(1, dtype=F64), atol=1e-6)
    assert out.weights[0, k + 1 :].abs().max() < 1e-6


def test_uniform_slab_obeys_beer_lambert():
    # A homogeneous slab of density sigma0 in vacuum, crossed at an angle.
    # Each sample's density holds until the next sample, so 50 consecutive
    # dense samples at spacing 0.04 make a slab of thickness 2.0 in t. The
    # direction is not unit length, so the physical path is 2.0 * |d|.
    # Beer-Lambert: absorbed fraction = 1 - exp(-sigma0 * path), exactly.
    n = 101
    sigma0 = 0.7
    colour = torch.tensor([0.9, 0.5, 0.1], dtype=F64)
    direction = torch.tensor([[0.3, -0.2, -1.0]], dtype=F64)

    t = stratified_samples(NEAR, FAR, 1, n, perturb=False, dtype=F64)
    sigma = torch.zeros(1, n, dtype=F64)
    sigma[:, 25:75] = sigma0
    rgb = colour.expand(1, n, 3)

    out = volume_render(sigma, rgb, t, direction)
    path = 2.0 * direction.norm().item()
    absorbed = 1.0 - math.exp(-sigma0 * path)
    assert math.isclose(out.acc.item(), absorbed, abs_tol=1e-6)
    assert torch.allclose(out.rgb[0], colour * absorbed, atol=1e-6)


def test_total_opacity_is_one_minus_total_transmittance():
    # For any densities, with random sample positions and ray lengths:
    #   sum of weights = 1 - exp(-optical depth),
    #   optical depth  = sum of sigma_i * delta_i.
    # The last sample is left empty so that nothing is forced opaque.
    torch.manual_seed(0)
    rays, n = 200, 48
    gen = torch.Generator().manual_seed(0)
    t = stratified_samples(NEAR, FAR, rays, n, generator=gen, dtype=F64)
    sigma = 3.0 * torch.rand(rays, n, dtype=F64)
    sigma[:, -1] = 0.0
    rgb = torch.rand(rays, n, 3, dtype=F64)
    directions = torch.randn(rays, 3, dtype=F64)

    out = volume_render(sigma, rgb, t, directions)
    deltas = (t[:, 1:] - t[:, :-1]) * directions.norm(dim=-1, keepdim=True)
    optical_depth = (sigma[:, :-1] * deltas).sum(dim=-1)
    assert torch.allclose(out.acc, 1.0 - torch.exp(-optical_depth), atol=1e-6)
    # weights are probabilities: non-negative and summing to at most one
    assert (out.weights >= 0).all()
    assert (out.acc <= 1.0 + 1e-6).all()


def test_quadrature_converges_to_the_analytic_integral():
    # Density rising linearly from 0 at near to sigma_max at far has optical
    # depth tau = sigma_max * (far - near) / 2. The last sample has positive
    # density, so it is opaque and acts as a detector: its weight is the
    # fraction of light that gets through the medium, which must tend to
    # exp(-tau) as the number of samples grows.
    sigma_max = 1.0
    exact = math.exp(-sigma_max * (FAR - NEAR) / 2.0)

    def transmitted(n):
        t = stratified_samples(NEAR, FAR, 1, n, perturb=False, dtype=F64)
        sigma = sigma_max * (t - NEAR) / (FAR - NEAR)
        rgb = torch.zeros(1, n, 3, dtype=F64)
        return volume_render(sigma, rgb, t, unit_dirs(1)).weights[0, -1].item()

    errors = [abs(transmitted(n) - exact) for n in (17, 65, 257, 1025)]
    # First-order rule: four times the samples, about a quarter of the error.
    for coarse, fine in zip(errors, errors[1:]):
        assert fine < coarse / 3.0
    assert errors[-1] < 1e-3


def test_white_background_adds_the_unabsorbed_light():
    torch.manual_seed(0)
    rays, n = 8, 32
    t = stratified_samples(NEAR, FAR, rays, n, perturb=False, dtype=F64)
    sigma = 0.2 * torch.rand(rays, n, dtype=F64)
    sigma[:, -1] = 0.0
    rgb = torch.rand(rays, n, 3, dtype=F64)

    black = volume_render(sigma, rgb, t, unit_dirs(rays))
    white = volume_render(sigma, rgb, t, unit_dirs(rays), white_background=True)
    assert (black.acc < 1.0).all()
    assert torch.allclose(white.rgb, black.rgb + (1.0 - black.acc)[:, None])


def test_renderer_is_differentiable():
    # Default float32 path: gradients must reach both densities and colours.
    torch.manual_seed(0)
    rays, n = 8, 32
    t = stratified_samples(NEAR, FAR, rays, n, generator=torch.Generator().manual_seed(0))
    sigma = torch.rand(rays, n, requires_grad=True)
    rgb = torch.rand(rays, n, 3, requires_grad=True)
    directions = torch.tensor([[0.0, 0.0, -1.0]]).expand(rays, 3)

    out = volume_render(sigma, rgb, t, directions, white_background=True)
    out.rgb.sum().backward()
    for grad in (sigma.grad, rgb.grad):
        assert torch.isfinite(grad).all()
        assert grad.abs().sum() > 0
