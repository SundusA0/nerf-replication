import math

import torch

from nerf.model import NeRF
from nerf.render import render_rays, sample_pdf, stratified_samples, volume_render

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


# --------------------------------------------------------------------------
# Hierarchical sampling: sample_pdf
# --------------------------------------------------------------------------


def test_uniform_weights_give_evenly_spread_samples():
    bins = torch.linspace(NEAR, FAR, 11, dtype=F64).expand(3, 11)
    weights = torch.ones(3, 10, dtype=F64)
    samples = sample_pdf(bins, weights, 21, deterministic=True)
    expected = torch.linspace(NEAR, FAR, 21, dtype=F64).expand(3, 21)
    assert torch.allclose(samples, expected, atol=1e-9)


def test_samples_go_where_the_weight_is():
    # All the weight in one bin: evenly spaced u must land almost entirely
    # inside it. (Not exactly all, because every bin keeps a tiny floor.)
    bins = torch.linspace(NEAR, FAR, 11, dtype=F64).expand(1, 11)
    weights = torch.zeros(1, 10, dtype=F64)
    weights[0, 6] = 1.0
    samples = sample_pdf(bins, weights, 1000, deterministic=True)
    inside = (samples >= bins[0, 6]) & (samples <= bins[0, 7])
    assert inside.double().mean() > 0.99


def test_sample_frequencies_match_the_weights():
    bins = torch.linspace(0.0, 4.0, 5, dtype=F64).expand(1, 5)
    weights = torch.tensor([[1.0, 2.0, 3.0, 4.0]], dtype=F64)
    gen = torch.Generator().manual_seed(0)
    samples = sample_pdf(bins, weights, 200_000, generator=gen)
    frequencies = torch.histc(samples[0], bins=4, min=0.0, max=4.0) / samples.numel()
    expected = torch.tensor([0.1, 0.2, 0.3, 0.4], dtype=F64)
    assert torch.allclose(frequencies, expected, atol=5e-3)


def test_samples_are_uniform_inside_a_bin():
    # One bin holds everything: its samples must spread evenly across it.
    bins = torch.tensor([[0.0, 1.0, 3.0]], dtype=F64)
    weights = torch.tensor([[0.0, 1.0]], dtype=F64)
    gen = torch.Generator().manual_seed(0)
    samples = sample_pdf(bins, weights, 100_000, generator=gen)
    in_bin = samples[samples > 1.0]
    assert math.isclose(in_bin.mean().item(), 2.0, abs_tol=1e-2)
    assert math.isclose(in_bin.std().item(), 2.0 / math.sqrt(12.0), abs_tol=1e-2)


def test_empty_ray_is_sampled_uniformly():
    # No weight anywhere must not produce NaNs; the ray is sampled evenly.
    bins = torch.linspace(NEAR, FAR, 11, dtype=F64).expand(2, 11)
    samples = sample_pdf(bins, torch.zeros(2, 10, dtype=F64), 21, deterministic=True)
    assert torch.isfinite(samples).all()
    expected = torch.linspace(NEAR, FAR, 21, dtype=F64).expand(2, 21)
    assert torch.allclose(samples, expected, atol=1e-9)


def test_samples_stay_inside_the_bins():
    torch.manual_seed(0)
    edges, _ = torch.sort(NEAR + (FAR - NEAR) * torch.rand(50, 12, dtype=F64), dim=-1)
    weights = torch.rand(50, 11, dtype=F64)
    gen = torch.Generator().manual_seed(0)
    samples = sample_pdf(edges, weights, 64, generator=gen)
    assert (samples >= edges[:, :1]).all() and (samples <= edges[:, -1:]).all()


# --------------------------------------------------------------------------
# Hierarchical sampling: render_rays
# --------------------------------------------------------------------------

WALL_Z = 0.9  # the wall fills everything with z below this


def wall(points, view_dirs):
    """An opaque red wall, standing in for a trained network."""
    sigma = torch.where(points[..., 2] < WALL_Z, 50.0, 0.0).to(points.dtype)
    rgb = torch.tensor([1.0, 0.0, 0.0], dtype=points.dtype).expand(points.shape)
    return sigma, rgb


def rays_towards_wall(num_rays, dtype=F64):
    # Camera at z = 4 looking down -z: a point at parameter t has z = 4 - t,
    # so the wall surface is at t = 4 - WALL_Z.
    origins = torch.tensor([[0.0, 0.0, 4.0]], dtype=dtype).expand(num_rays, 3)
    directions = torch.tensor([[0.0, 0.0, -1.0]], dtype=dtype).expand(num_rays, 3)
    return origins, directions


def test_importance_samples_concentrate_at_the_surface():
    num_coarse, num_fine = 16, 32
    origins, directions = rays_towards_wall(1)
    out = render_rays(wall, wall, origins, directions, NEAR, FAR, num_coarse, num_fine, perturb=False)

    spacing = (FAR - NEAR) / (num_coarse - 1)
    hit = out.t_coarse[0, out.coarse.weights[0].argmax()]  # coarse sample that found the wall
    near_hit = (out.t_fine[0] - hit).abs() <= spacing / 2
    assert near_hit.sum() >= 0.9 * num_fine


def test_fine_pass_locates_the_surface_better_than_the_coarse_pass():
    origins, directions = rays_towards_wall(1)
    out = render_rays(wall, wall, origins, directions, NEAR, FAR, 16, 32, perturb=False)
    true_depth = 4.0 - WALL_Z
    coarse_error = (out.coarse.depth - true_depth).abs().item()
    fine_error = (out.fine.depth - true_depth).abs().item()
    assert fine_error < coarse_error
    # both passes see an opaque red wall
    for render in (out.coarse, out.fine):
        assert torch.allclose(render.rgb, torch.tensor([[1.0, 0.0, 0.0]], dtype=F64), atol=1e-4)
        assert torch.allclose(render.acc, torch.ones(1, dtype=F64), atol=1e-4)


def test_fine_pass_uses_coarse_and_importance_samples_together():
    num_coarse, num_fine = 16, 32
    origins, directions = rays_towards_wall(5)
    gen = torch.Generator().manual_seed(0)
    out = render_rays(wall, wall, origins, directions, NEAR, FAR, num_coarse, num_fine, generator=gen)

    assert out.t_coarse.shape == (5, num_coarse)
    assert out.t_fine.shape == (5, num_coarse + num_fine)
    assert out.coarse.weights.shape == (5, num_coarse)
    assert out.fine.weights.shape == (5, num_coarse + num_fine)
    assert (out.t_fine[:, 1:] >= out.t_fine[:, :-1]).all()
    # every coarse sample is still present in the fine set
    gaps = (out.t_fine[:, None, :] - out.t_coarse[:, :, None]).abs().min(dim=-1).values
    assert gaps.max() == 0


def test_no_fine_pass_when_num_fine_is_zero():
    origins, directions = rays_towards_wall(3)
    out = render_rays(wall, None, origins, directions, NEAR, FAR, 16, 0, perturb=False)
    assert out.fine is None and out.t_fine is None
    assert out.coarse.rgb.shape == (3, 3)


def test_evaluation_rendering_is_deterministic_and_seeded_training_is_reproducible():
    origins, directions = rays_towards_wall(4)
    a = render_rays(wall, wall, origins, directions, NEAR, FAR, 16, 32, perturb=False)
    b = render_rays(wall, wall, origins, directions, NEAR, FAR, 16, 32, perturb=False)
    assert torch.equal(a.t_fine, b.t_fine) and torch.equal(a.fine.rgb, b.fine.rgb)

    def seeded(seed):
        gen = torch.Generator().manual_seed(seed)
        return render_rays(wall, wall, origins, directions, NEAR, FAR, 16, 32, generator=gen)

    assert torch.equal(seeded(1).t_fine, seeded(1).t_fine)
    assert not torch.equal(seeded(1).t_fine, seeded(2).t_fine)


def test_fine_loss_does_not_train_the_coarse_network():
    # The importance samples are constants: a loss on the fine render must not
    # send gradients into the coarse network through their positions.
    torch.manual_seed(0)
    coarse = NeRF(depth=2, width=16, skip=0)
    fine = NeRF(depth=2, width=16, skip=0)
    # A freshly initialised network can output zero density everywhere, which
    # would make every gradient zero. Start both with some density instead.
    with torch.no_grad():
        coarse.sigma_head.bias.fill_(5.0)
        fine.sigma_head.bias.fill_(5.0)
    origins, directions = rays_towards_wall(8, dtype=torch.float32)
    gen = torch.Generator().manual_seed(0)

    out = render_rays(coarse, fine, origins, directions, NEAR, FAR, 8, 8, generator=gen)
    out.fine.rgb.sum().backward()
    assert all(p.grad is None for p in coarse.parameters())
    assert sum(p.grad.abs().sum() for p in fine.parameters()) > 0

    out = render_rays(coarse, fine, origins, directions, NEAR, FAR, 8, 8, generator=gen)
    out.coarse.rgb.sum().backward()
    assert sum(p.grad.abs().sum() for p in coarse.parameters()) > 0


def test_one_network_can_serve_both_passes():
    torch.manual_seed(0)
    model = NeRF(depth=2, width=16, skip=0)
    origins, directions = rays_towards_wall(4, dtype=torch.float32)
    out = render_rays(model, None, origins, directions, NEAR, FAR, 8, 8, perturb=False)
    assert out.fine is not None
    assert out.fine.weights.shape == (4, 16)


def test_points_follow_the_ray_and_networks_get_unit_directions():
    # Ray directions are not unit length (see get_rays). Sample positions must
    # be o + t d with d as given, while the direction fed to the network must
    # be normalised.
    calls = []

    def probe(points, view_dirs):
        calls.append((points, view_dirs))
        return wall(points, view_dirs)

    origins = torch.tensor([[0.0, 0.0, 4.0], [0.5, -0.5, 4.0]], dtype=F64)
    directions = torch.tensor([[0.3, -0.2, -1.0], [-0.1, 0.4, -1.0]], dtype=F64)
    out = render_rays(probe, probe, origins, directions, NEAR, FAR, 8, 8, perturb=False)

    assert len(calls) == 2  # one query for the coarse pass, one for the fine pass
    for (points, view_dirs), t_vals in zip(calls, (out.t_coarse, out.t_fine)):
        expected = origins[:, None, :] + t_vals[..., None] * directions[:, None, :]
        assert torch.allclose(points, expected)
        assert torch.allclose(view_dirs.norm(dim=-1), torch.ones_like(t_vals))
        unit = directions / directions.norm(dim=-1, keepdim=True)
        assert torch.allclose(view_dirs, unit[:, None, :].expand_as(points))


def test_training_mode_randomises_the_importance_samples():
    # Evaluation places importance samples at evenly spaced quantiles, so
    # inside the bin that found the wall they are evenly spaced. Training must
    # draw them at random, which shows up as irregular spacing.
    num_coarse, num_fine = 16, 64
    origins, directions = rays_towards_wall(1)

    def spacing_irregularity(perturb):
        gen = torch.Generator().manual_seed(0)
        out = render_rays(
            wall, wall, origins, directions, NEAR, FAR, num_coarse, num_fine,
            perturb=perturb, generator=gen,
        )
        t_fine, t_coarse = out.t_fine[0], out.t_coarse[0]
        is_coarse = (t_fine[:, None] == t_coarse[None, :]).any(dim=-1)
        importance = t_fine[~is_coarse]
        hit = t_coarse[out.coarse.weights[0].argmax()]
        half_bin = (FAR - NEAR) / (num_coarse - 1) / 2
        in_bin = importance[(importance - hit).abs() < 0.5 * half_bin]
        gaps = in_bin[1:] - in_bin[:-1]
        return (gaps.std() / gaps.mean()).item()

    assert spacing_irregularity(perturb=False) < 0.05
    assert spacing_irregularity(perturb=True) > 0.3
