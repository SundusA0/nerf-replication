"""Posed images, the training data of a NeRF.

A NeRF is fitted to one scene from photographs taken at known camera poses.
`Views` holds such a set. `load_blender` reads one from the NeRF synthetic
dataset. `analytic_views` builds one from a scene whose density and colour are
known in closed form, so that training can be checked without downloading
anything.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image

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
    alpha: torch.Tensor | None = None   # (N, H, W) opacity of each pixel, if known

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
            None if self.alpha is None else self.alpha.to(device),
        )


def blender_frames(scene_dir, split: str = "train") -> list[str]:
    """The image paths of one split of a NeRF synthetic scene, in file order.

    The paths are as written in the dataset, relative to the scene directory
    and without the ".png", e.g. "./test/r_0".
    """
    meta = json.loads((Path(scene_dir) / f"transforms_{split}.json").read_text())
    return [frame["file_path"] for frame in meta["frames"]]


def load_blender(
    scene_dir,
    split: str = "train",
    downscale: int = 1,
    skip: int = 1,
    indices=None,
) -> Views:
    """Read one split of a NeRF synthetic ("Blender") scene such as Lego.

    A scene directory holds `transforms_{train,val,test}.json` and the images
    they name. Each JSON file gives the horizontal field of view shared by all
    frames and, per frame, an image path and a 4x4 camera-to-world matrix in
    the convention `get_rays` assumes. The images are 800x800 RGBA with a
    transparent background.

    As in the released code: the focal length follows from the field of view,
    resizing is by area averaging and is done on RGBA before compositing, the
    images are composited over white, and the scene lies between near = 2 and
    far = 6.

    Args:
        scene_dir: e.g. data/nerf_synthetic/lego.
        split: "train" (100 views), "val" (100) or "test" (200).
        downscale: integer factor to shrink the images by; 2 gives the
            400x400 "half resolution" of the released example config.
        skip: keep every `skip`-th frame, to evaluate on a subset.
        indices: of the frames that `skip` keeps, load only those at these
            positions. This reads a large split a few views at a time: the
            200 test views at full resolution take several GB at once.
    """
    scene_dir = Path(scene_dir)
    meta = json.loads((scene_dir / f"transforms_{split}.json").read_text())
    frames = meta["frames"][::skip]
    if indices is not None:
        frames = [frames[index] for index in indices]

    images, poses = [], []
    for frame in frames:
        image = Image.open(scene_dir / (frame["file_path"] + ".png")).convert("RGBA")
        images.append(torch.from_numpy(np.asarray(image, dtype=np.float32) / 255.0))
        poses.append(torch.tensor(frame["transform_matrix"], dtype=torch.float32))
    rgba = torch.stack(images)  # (N, H, W, 4)

    if downscale > 1:
        rgba = torch.nn.functional.avg_pool2d(rgba.permute(0, 3, 1, 2), downscale)
        rgba = rgba.permute(0, 2, 3, 1)

    rgb, alpha = rgba[..., :3], rgba[..., 3]
    composited = rgb * alpha[..., None] + (1.0 - alpha[..., None])
    width = composited.shape[2]
    focal = 0.5 * width / math.tan(0.5 * float(meta["camera_angle_x"]))
    return Views(
        composited, torch.stack(poses), focal,
        near=2.0, far=6.0, white_background=True, alpha=alpha,
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

    images, alphas = [], []
    for pose in poses:
        rays_o, rays_d = get_rays(image_size, image_size, focal, pose)
        out = render_rays(
            scene, None, rays_o.reshape(-1, 3), rays_d.reshape(-1, 3), near, far,
            num_coarse=num_samples, num_fine=0, perturb=False, white_background=True,
        )
        images.append(out.coarse.rgb.reshape(image_size, image_size, 3))
        alphas.append(out.coarse.acc.reshape(image_size, image_size))
    return Views(
        torch.stack(images), poses, focal, near, far,
        white_background=True, alpha=torch.stack(alphas),
    )
