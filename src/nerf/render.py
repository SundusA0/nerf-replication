"""Sampling along rays and volume rendering (Mildenhall et al. 2020, Section 4).

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
be trained through it later.
"""

from __future__ import annotations

from typing import NamedTuple

import torch


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
