"""Sampling along rays and volume rendering (Mildenhall et al. 2020, Sections 4 and 5.2).

The colour seen along a ray r(t) = o + t d is the emission-absorption integral

    C(r) = integral from t_n to t_f of  T(t) sigma(r(t)) c(r(t), d) dt,
    T(t) = exp( - integral from t_n to t of sigma(r(s)) ds ).            (Eq. 1)

sigma is the volume density: an attenuation coefficient with units 1/length.
c is the colour emitted at that point towards the camera. T is the
transmittance, the Beer-Lambert probability that light gets from t_n to t
without being absorbed. There is no scattering term, so this is the
emission-absorption special case of radiative transfer.

Nothing in this module is learned. It turns densities and colours sampled
along rays into pixels, and it is differentiable, which is what lets a network
be trained through it.

  stratified_samples   where to sample a ray at first             (Eq. 2)
  volume_render        samples -> pixel colour, depth, weights    (Eq. 3)
  sample_pdf           where to sample again, given the weights   (Section 5.2)
  render_rays          the two-pass procedure that combines them
  render_image         render_rays over every pixel of one camera
"""

from __future__ import annotations

from typing import NamedTuple

import torch

from nerf.rays import get_rays


class RenderOutput(NamedTuple):
    rgb: torch.Tensor      # (R, 3) rendered colour of each ray
    depth: torch.Tensor    # (R,)   expected ray parameter t where the ray stops
    acc: torch.Tensor      # (R,)   total opacity: 0 = empty space, 1 = fully blocked
    weights: torch.Tensor  # (R, N) contribution of each sample to the pixel


def stratified_samples(
    near: float,
    far: float,
    num_rays: int,
    num_samples: int,
    perturb: bool = True,
    generator: torch.Generator | None = None,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Ray parameters t_1 <= ... <= t_N at which to query the field (Eq. 2).

    Evaluating the integral at the same fixed depths every time would only
    ever train the network at those depths. Stratified sampling draws one
    uniform sample from each of N bins, so over many training steps every depth
    between near and far is visited while each draw still covers the whole ray.

    The paper's Eq. 2 splits [near, far] into N equal bins. The released code
    instead places N evenly spaced points, including both ends, and jitters
    each point between the midpoints to its two neighbours, so the first and
    last bins are half as wide as the others. This follows the code.

    Args:
        near, far: bounds on the ray parameter t.
        num_rays: R, number of rays.
        num_samples: N, samples per ray.
        perturb: True for training (random position within each bin). False
            gives the N evenly spaced points themselves, used for evaluation.
        generator: random generator, for reproducible draws. The random numbers
            are drawn on the CPU and then moved, so a given seed produces the
            same samples on every device.

    Returns:
        (R, N) sorted ray parameters.
    """
    t = torch.linspace(near, far, num_samples, device=device, dtype=dtype)
    t = t.expand(num_rays, num_samples)
    if not perturb:
        return t.clone()

    mids = 0.5 * (t[:, 1:] + t[:, :-1])
    lower = torch.cat([t[:, :1], mids], dim=-1)
    upper = torch.cat([mids, t[:, -1:]], dim=-1)
    u = torch.rand(num_rays, num_samples, generator=generator, dtype=dtype).to(t.device)
    return lower + (upper - lower) * u


def volume_render(
    sigma: torch.Tensor,
    rgb: torch.Tensor,
    t_vals: torch.Tensor,
    rays_d: torch.Tensor,
    white_background: bool = False,
) -> RenderOutput:
    """Numerical quadrature of the rendering integral (Eq. 3).

    Take density and colour to be constant on each interval [t_i, t_(i+1)).
    The integral over one interval then has a closed form, and Eq. 1 becomes

        C = sum over i of  T_i * alpha_i * c_i

        delta_i = (t_(i+1) - t_i) * |d|          length of interval i
        alpha_i = 1 - exp(-sigma_i * delta_i)    probability of absorption in i
        T_i     = product over j < i of (1 - alpha_j)
                                                 probability of reaching i

    The weight w_i = T_i * alpha_i is the probability that the ray ends in
    interval i. The weights are non-negative and sum to at most 1; whatever is
    left over is light that passed through everything.

    Args:
        sigma: (R, N) densities at the samples, >= 0.
        rgb: (R, N, 3) colours at the samples, in [0, 1].
        t_vals: (R, N) sorted ray parameters, e.g. from `stratified_samples`.
        rays_d: (R, 3) ray directions. Need not be unit length; see `get_rays`.
        white_background: composite the result over white instead of black.
            The NeRF synthetic scenes are evaluated on a white background.
    """
    deltas = t_vals[..., 1:] - t_vals[..., :-1]
    # The last sample has no next sample to close its interval. As in the
    # reference implementation it gets an effectively infinite one, so any
    # density at the last sample is completely opaque.
    last = torch.full_like(deltas[..., :1], 1e10)
    deltas = torch.cat([deltas, last], dim=-1)
    # t is in units of the (possibly non-unit) direction; convert to length.
    deltas = deltas * rays_d.norm(dim=-1, keepdim=True)

    alpha = 1.0 - torch.exp(-sigma * deltas)

    # T_i is an exclusive cumulative product: T_1 = 1, T_2 = (1 - alpha_1), ...
    # The 1e-10 keeps the product, and so the gradients, from becoming exactly
    # zero behind an opaque sample.
    survive = torch.cat([torch.ones_like(alpha[..., :1]), 1.0 - alpha + 1e-10], dim=-1)
    transmittance = torch.cumprod(survive, dim=-1)[..., :-1]
    weights = transmittance * alpha

    rgb_map = (weights[..., None] * rgb).sum(dim=-2)
    depth = (weights * t_vals).sum(dim=-1)
    acc = weights.sum(dim=-1)
    if white_background:
        rgb_map = rgb_map + (1.0 - acc[..., None])
    return RenderOutput(rgb=rgb_map, depth=depth, acc=acc, weights=weights)


def sample_pdf(
    bins: torch.Tensor,
    weights: torch.Tensor,
    num_samples: int,
    deterministic: bool = False,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Draw samples from a piecewise-constant distribution (Section 5.2).

    `weights` gives the unnormalised probability of each bin and the density
    is uniform inside a bin. Sampling is by inverse transform: build the
    cumulative distribution F at the bin edges, draw u uniformly in [0, 1),
    find the bin where F crosses u and interpolate linearly inside it.

    Args:
        bins: (R, M + 1) sorted bin edges.
        weights: (R, M) non-negative weight of each bin.
        num_samples: number of samples per ray.
        deterministic: use evenly spaced u instead of random u (evaluation).
        generator: random generator; draws happen on the CPU, as in
            `stratified_samples`.

    Returns:
        (R, num_samples) samples in [bins[:, 0], bins[:, -1]]. Sorted only if
        deterministic.
    """
    # A small floor on every bin avoids 0 / 0 on rays where all weights vanish
    # (empty space); such rays are then sampled uniformly.
    weights = weights + 1e-5
    pdf = weights / weights.sum(dim=-1, keepdim=True)
    cdf = torch.cumsum(pdf, dim=-1)
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], dim=-1)  # (R, M + 1)

    num_rays = cdf.shape[0]
    if deterministic:
        u = torch.linspace(0.0, 1.0, num_samples, dtype=cdf.dtype, device=cdf.device)
        u = u.expand(num_rays, num_samples)
    else:
        u = torch.rand(num_rays, num_samples, generator=generator, dtype=cdf.dtype)
        u = u.to(cdf.device)
    u = u.contiguous()

    # Index of the first edge whose cdf exceeds u; the sample lies in the bin
    # between the edge before it and that edge.
    above = torch.searchsorted(cdf, u, right=True)
    below = (above - 1).clamp(min=0)
    above = above.clamp(max=cdf.shape[-1] - 1)

    cdf_below, cdf_above = cdf.gather(-1, below), cdf.gather(-1, above)
    bin_below, bin_above = bins.gather(-1, below), bins.gather(-1, above)

    width = cdf_above - cdf_below
    width = torch.where(width < 1e-5, torch.ones_like(width), width)
    fraction = (u - cdf_below) / width
    return bin_below + fraction * (bin_above - bin_below)


class HierarchicalOutput(NamedTuple):
    coarse: RenderOutput          # rendered from the coarse samples only
    fine: RenderOutput | None     # rendered from coarse + importance samples
    t_coarse: torch.Tensor        # (R, N_c) coarse sample positions
    t_fine: torch.Tensor | None   # (R, N_c + N_f) all positions used by the fine pass


def _query_and_render(model, rays_o, rays_d, view_dirs, t_vals, white_background):
    points = rays_o[:, None, :] + t_vals[..., None] * rays_d[:, None, :]  # (R, N, 3)
    dirs = view_dirs[:, None, :].expand_as(points)
    sigma, rgb = model(points, dirs)
    return volume_render(sigma, rgb, t_vals, rays_d, white_background)


def render_rays(
    coarse_model,
    fine_model,
    rays_o: torch.Tensor,
    rays_d: torch.Tensor,
    near: float,
    far: float,
    num_coarse: int,
    num_fine: int,
    perturb: bool = True,
    white_background: bool = False,
    generator: torch.Generator | None = None,
) -> HierarchicalOutput:
    """Render a batch of rays with hierarchical sampling (Section 5.2).

    Most of a ray passes through empty space or lies behind a surface, so
    evenly spread samples are mostly wasted. NeRF therefore samples twice:

      1. Coarse pass. Query `coarse_model` at `num_coarse` stratified samples
         and render. The weights w_i say where along the ray the visible
         content is.
      2. Fine pass. Normalise those weights into a probability distribution,
         draw `num_fine` more samples from it, and query `fine_model` at the
         union of both sets of samples, sorted.

    Both renders are returned because both are used in the loss (Eq. 6): the
    coarse network has to be trained too, or its weights would be useless for
    placing the fine samples.

    As in the released code, the distribution is built on the midpoints
    between coarse samples and uses the weights of the interior samples only.
    The importance samples are treated as constants: no gradient flows from
    the fine render back into the coarse network through their positions.

    Args:
        coarse_model, fine_model: callables (points, view_dirs) -> (sigma, rgb)
            with points and view_dirs of shape (R, N, 3). If `fine_model` is
            None the coarse model is used for both passes.
        rays_o, rays_d: (R, 3) ray origins and directions (see `get_rays`).
        near, far: bounds on the ray parameter.
        num_coarse: N_c, 64 in the paper.
        num_fine: N_f, 128 in the paper. 0 disables the fine pass.
        perturb: random sample positions (training) or fixed ones (evaluation).
        white_background: composite over white.
        generator: random generator for reproducible sampling.
    """
    view_dirs = rays_d / rays_d.norm(dim=-1, keepdim=True)

    t_coarse = stratified_samples(
        near, far, rays_o.shape[0], num_coarse,
        perturb=perturb, generator=generator, device=rays_o.device, dtype=rays_o.dtype,
    )
    coarse = _query_and_render(
        coarse_model, rays_o, rays_d, view_dirs, t_coarse, white_background
    )
    if num_fine == 0:
        return HierarchicalOutput(coarse=coarse, fine=None, t_coarse=t_coarse, t_fine=None)

    t_mid = 0.5 * (t_coarse[:, 1:] + t_coarse[:, :-1])
    t_importance = sample_pdf(
        t_mid, coarse.weights[:, 1:-1], num_fine,
        deterministic=not perturb, generator=generator,
    ).detach()
    t_fine, _ = torch.sort(torch.cat([t_coarse, t_importance], dim=-1), dim=-1)

    fine = _query_and_render(
        fine_model if fine_model is not None else coarse_model,
        rays_o, rays_d, view_dirs, t_fine, white_background,
    )
    return HierarchicalOutput(coarse=coarse, fine=fine, t_coarse=t_coarse, t_fine=t_fine)


class ImageOutput(NamedTuple):
    rgb: torch.Tensor    # (H, W, 3)
    depth: torch.Tensor  # (H, W)
    acc: torch.Tensor    # (H, W)


@torch.no_grad()
def render_image(
    coarse_model,
    fine_model,
    height: int,
    width: int,
    focal: float,
    c2w: torch.Tensor,
    near: float,
    far: float,
    num_coarse: int,
    num_fine: int,
    white_background: bool = False,
    chunk: int = 4096,
) -> ImageOutput:
    """Render a full image from one camera, for evaluation.

    Sample positions are fixed (no jitter) and no gradients are kept. An image
    has far more rays than fit through the network at once, so they are
    rendered `chunk` rays at a time. The result comes from the fine pass, or
    from the coarse pass if `num_fine` is 0.
    """
    rays_o, rays_d = get_rays(height, width, focal, c2w)
    rays_o, rays_d = rays_o.reshape(-1, 3), rays_d.reshape(-1, 3)

    rgb, depth, acc = [], [], []
    for start in range(0, rays_o.shape[0], chunk):
        out = render_rays(
            coarse_model, fine_model,
            rays_o[start : start + chunk], rays_d[start : start + chunk],
            near, far, num_coarse, num_fine,
            perturb=False, white_background=white_background,
        )
        final = out.fine if out.fine is not None else out.coarse
        rgb.append(final.rgb)
        depth.append(final.depth)
        acc.append(final.acc)

    return ImageOutput(
        rgb=torch.cat(rgb).reshape(height, width, 3),
        depth=torch.cat(depth).reshape(height, width),
        acc=torch.cat(acc).reshape(height, width),
    )
