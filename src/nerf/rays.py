"""Camera rays.

NeRF renders an image one pixel at a time: a pinhole camera sends a ray through
each pixel, and the colour of the pixel is the volume rendering integral along
that ray (Mildenhall et al. 2020, Section 4). This module is the camera model:
`get_rays` goes from pixels to rays, `project_points` from 3D points back to
pixels.
"""

from __future__ import annotations

import torch


def get_rays(
    height: int, width: int, focal: float, c2w: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Ray origin and direction for every pixel of a pinhole camera.

    Camera convention (Blender / OpenGL, as in the NeRF synthetic scenes): the
    camera looks down its own -z axis, with +x to the right and +y up. Pixel
    (row j, column i) therefore has camera-space direction

        ( (i - W/2) / f,  -(j - H/2) / f,  -1 ).

    The minus sign on y is because image rows count downwards. Like the
    reference implementation, pixel coordinates are integers with no half-pixel
    offset.

    Args:
        height: image height H in pixels.
        width: image width W in pixels.
        focal: focal length f in pixels.
        c2w: camera-to-world matrix, shape (3, 4) or (4, 4). Its 3x3 block
            rotates camera axes into world axes and its last column is the
            camera position in world coordinates.

    Returns:
        origins: (H, W, 3). Every ray starts at the camera position.
        directions: (H, W, 3) in world coordinates. These are NOT unit vectors:
            every direction has camera-space z = -1, so the ray parameter t in
            o + t d measures depth in front of the camera, not distance along
            the ray. Near and far bounds are then planes of constant depth, as
            in the reference implementation. `volume_render` multiplies by the
            length of d wherever a true distance is needed.
    """
    c2w = torch.as_tensor(c2w)
    dtype, device = c2w.dtype, c2w.device

    # i runs along the width (columns), j along the height (rows).
    i, j = torch.meshgrid(
        torch.arange(width, dtype=dtype, device=device),
        torch.arange(height, dtype=dtype, device=device),
        indexing="xy",
    )
    dirs_cam = torch.stack(
        [(i - 0.5 * width) / focal, -(j - 0.5 * height) / focal, -torch.ones_like(i)],
        dim=-1,
    )

    # d_world = R d_cam for every pixel. Written with row vectors that is d R^T.
    rotation = c2w[:3, :3]
    directions = dirs_cam @ rotation.T
    origins = c2w[:3, 3].expand(directions.shape)
    return origins, directions


def project_points(
    points: torch.Tensor, height: int, width: int, focal: float, c2w: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Where world points appear in the image: the inverse of `get_rays`.

    Args:
        points: (..., 3) positions in world coordinates.
        height, width, focal, c2w: the camera, as in `get_rays`.

    Returns:
        cols, rows: (...) continuous pixel coordinates. A point on the ray that
            `get_rays` sends through pixel (row j, column i) gets cols = i and
            rows = j.
        depth: (...) the ray parameter t at the point. It is positive in front
            of the camera; points with depth <= 0 are behind it and their
            pixel coordinates are meaningless.
    """
    rotation, position = c2w[:3, :3], c2w[:3, 3]
    # Camera coordinates R^T (p - o), written for row vectors.
    cam = (points - position) @ rotation
    depth = -cam[..., 2]
    cols = focal * cam[..., 0] / depth + 0.5 * width
    rows = -focal * cam[..., 1] / depth + 0.5 * height
    return cols, rows, depth
