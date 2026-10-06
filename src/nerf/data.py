"""Posed images, the training data of a NeRF.

A NeRF is fitted to one scene from photographs taken at known camera poses.
`Views` holds such a set. `analytic_views` builds one from a scene whose
density and colour are known in closed form, so that training can be checked
without downloading anything.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from nerf.rays import get_rays
from nerf.render import render_rays


@dataclass
class Views:
    """Images of one scene with the cameras that took them."""

    images: torch.Tensor      # (N, H, W, 3) colours in [0, 1]
    poses: torch.Tensor       # (N, 4, 4) camera-to-world matrices
    focal: float              # focal length in pixels, shared by all views
    near: float               # ray parameter bounds that enclose the scene
    far: float
    white_background: bool    # whether the images are composited over white

    def __len__(self) -> int:
        return self.images.shape[0]

    @property
    def height(self) -> int:
        return self.images.shape[1]

    @property
    def width(self) -> int:
        return self.images.shape[2]

    def to(self, device) -> "Views":
        return Views(
            self.images.to(device), self.poses.to(device),
            self.focal, self.near, self.far, self.white_background,
        )


def look_at(position, target=(0.0, 0.0, 0.0), up=(0.0, 0.0, 1.0)) -> torch.Tensor:
    """Camera-to-world matrix of a camera at `position` looking at `target`.

    Same convention as `get_rays`: the camera looks down its own -z axis with
    +x to the right and +y up. The columns of the rotation are therefore the
    camera's right, up and backward directions written in world coordinates.
    The world is z-up, like the NeRF synthetic scenes.
    """
    position = torch.as_tensor(position, dtype=torch.float32)
    forward = torch.as_tensor(target, dtype=torch.float32) - position
    forward = forward / forward.norm()
    right = torch.linalg.cross(forward, torch.as_tensor(up, dtype=torch.float32))
    right = right / right.norm()
    true_up = torch.linalg.cross(right, forward)

    c2w = torch.eye(4)
    c2w[:3, 0] = right
    c2w[:3, 1] = true_up
    c2w[:3, 2] = -forward
    c2w[:3, 3] = position
    return c2w


def orbit_poses(
    num_views: int,
    radius: float = 4.0,
    min_elevation: float = 10.0,
    max_elevation: float = 60.0,
    phase: float = 0.0,
) -> torch.Tensor:
    """Cameras spread over the upper hemisphere, all looking at the origin.

    Elevation rises steadily from `min_elevation` to `max_elevation` (degrees)
    while the azimuth advances by the golden angle each view, which spreads
    the cameras evenly without lining them up. A different `phase` (degrees of
    azimuth) gives a different set of viewpoints on the same orbit.
    """
    golden_angle = math.pi * (3.0 - math.sqrt(5.0))
    poses = []
    for k in range(num_views):
        fraction = (k + 0.5) / num_views
        elevation = math.radians(min_elevation + fraction * (max_elevation - min_elevation))
        azimuth = math.radians(phase) + k * golden_angle
        position = (
            radius * math.cos(elevation) * math.cos(azimuth),
            radius * math.cos(elevation) * math.sin(azimuth),
            radius * math.sin(elevation),
        )
        poses.append(look_at(position))
    return torch.stack(poses)


class SphereScene:
    """A scene made of solid coloured spheres, with exactly known density.

    Calling it returns density and colour at any points, with the same
    signature as the network, so it can be rendered by the same code.
    """

    # (centre, radius, colour). Deliberately lopsided, so a flipped axis or a
    # mirrored image cannot go unnoticed.
    SPHERES = (
        ((0.0, 0.0, 0.0), 0.70, (0.85, 0.15, 0.15)),   # red, in the middle
        ((1.1, 0.0, 0.0), 0.35, (0.15, 0.70, 0.20)),   # green, towards +x
        ((0.0, 1.1, 0.0), 0.35, (0.15, 0.30, 0.85)),   # blue, towards +y
        ((0.0, 0.0, 0.95), 0.25, (0.95, 0.80, 0.10)),  # yellow, on top (+z)
    )

    def __init__(self, density: float = 40.0) -> None:
        self.density = density

    def __call__(self, points: torch.Tensor, view_dirs: torch.Tensor):
        sigma = torch.zeros(points.shape[:-1], dtype=points.dtype, device=points.device)
        rgb = torch.zeros_like(points)
        for centre, radius, colour in self.SPHERES:
            inside = (points - points.new_tensor(centre)).norm(dim=-1) < radius
            sigma[inside] = self.density
            rgb[inside] = points.new_tensor(colour)
        return sigma, rgb


def analytic_views(
    num_views: int,
    image_size: int = 48,
    phase: float = 0.0,
    num_samples: int = 256,
) -> Views:
    """Render `SphereScene` from an orbit of cameras.

    The images come from this repository's own renderer, applied to the exact
    field with many samples per ray. Training on them therefore tests the
    optimisation and the plumbing (can the network recover a known field
    through the renderer?), not the renderer's accuracy, which the unit tests
    check against closed-form results.

    Camera distance, near/far bounds and field of view copy the NeRF synthetic
    scenes: radius 4, near 2, far 6, horizontal field of view about 40 degrees.
    """
    scene = SphereScene()
    poses = orbit_poses(num_views, phase=phase)
    near, far = 2.0, 6.0
    focal = 0.5 * image_size / math.tan(0.5 * 0.6911)

    images = []
    for pose in poses:
        rays_o, rays_d = get_rays(image_size, image_size, focal, pose)
        out = render_rays(
            scene, None, rays_o.reshape(-1, 3), rays_d.reshape(-1, 3), near, far,
            num_coarse=num_samples, num_fine=0, perturb=False, white_background=True,
        )
        images.append(out.coarse.rgb.reshape(image_size, image_size, 3))
    return Views(torch.stack(images), poses, focal, near, far, white_background=True)
