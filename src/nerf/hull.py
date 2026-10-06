"""Space carving: the visual hull of a scene from its silhouettes.

A 3D point that belongs to the object must land on the object's silhouette in
every image. Keeping only the points of a grid that do so gives the visual
hull, the largest shape consistent with all the silhouettes.

Nothing is trained here, which makes this a direct test of the camera model on
real data: with a wrong axis convention, focal length or pose format the
silhouettes contradict each other and the hull comes out empty. It is also a
classical baseline for the geometry a NeRF should improve on, since a hull
cannot represent concavities.
"""

from __future__ import annotations

import torch

from nerf.data import Views
from nerf.rays import project_points


def visual_hull(
    views: Views, resolution: int = 64, bound: float = 1.5, threshold: float = 0.5
) -> tuple[torch.Tensor, torch.Tensor]:
    """Carve a cubic grid with the silhouettes of `views`.

    Args:
        views: must carry `alpha`, the opacity of every pixel.
        resolution: grid points per axis.
        bound: the grid spans [-bound, bound] on every axis.
        threshold: a pixel counts as silhouette if its alpha exceeds this.

    Returns:
        occupied: (resolution, resolution, resolution) bool, indexed [x, y, z].
            True where the point projects onto the silhouette in every view.
        axis: (resolution,) coordinates of the grid points along each axis.
    """
    if views.alpha is None:
        raise ValueError("visual_hull needs views with alpha (silhouettes)")

    axis = torch.linspace(-bound, bound, resolution)
    grid = torch.stack(torch.meshgrid(axis, axis, axis, indexing="ij"), dim=-1).reshape(-1, 3)

    occupied = torch.ones(grid.shape[0], dtype=torch.bool)
    for alpha, pose in zip(views.alpha, views.poses):
        cols, rows, depth = project_points(grid, views.height, views.width, views.focal, pose)
        cols, rows = cols.round().long(), rows.round().long()
        in_image = (
            (depth > 0) & (cols >= 0) & (cols < views.width) & (rows >= 0) & (rows < views.height)
        )
        on_silhouette = torch.zeros_like(occupied)
        on_silhouette[in_image] = alpha[rows[in_image], cols[in_image]] > threshold
        occupied &= on_silhouette
    return occupied.reshape(resolution, resolution, resolution), axis


def bounding_box(occupied: torch.Tensor, axis: torch.Tensor) -> list[tuple[float, float]] | None:
    """Extent of the occupied grid points along x, y and z; None if empty."""
    if not occupied.any():
        return None
    index = occupied.nonzero()
    return [(axis[index[:, k].min()].item(), axis[index[:, k].max()].item()) for k in range(3)]


def reprojection_iou(views: Views, occupied: torch.Tensor, axis: torch.Tensor, threshold: float = 0.5) -> list[float]:
    """How well the hull, projected back into each view, matches its silhouette.

    Carving only ever removes points, so some hull always survives a small
    mistake in the cameras. What does not survive is agreement: with correct
    cameras the hull projects back onto (nearly) the whole silhouette in every
    view, while with inconsistent cameras it covers only the part of each
    silhouette that the other views happened to leave. The intersection over
    union of the two masks, per view, measures that. It is limited by the grid:
    the grid spacing should be no larger than the footprint of a pixel.
    """
    index = occupied.nonzero()
    points = torch.stack([axis[index[:, k]] for k in range(3)], dim=-1)

    scores = []
    for alpha, pose in zip(views.alpha, views.poses):
        cols, rows, depth = project_points(points, views.height, views.width, views.focal, pose)
        cols, rows = cols.round().long(), rows.round().long()
        in_image = (
            (depth > 0) & (cols >= 0) & (cols < views.width) & (rows >= 0) & (rows < views.height)
        )
        projected = torch.zeros(views.height, views.width, dtype=torch.bool)
        projected[rows[in_image], cols[in_image]] = True
        silhouette = alpha > threshold
        union = (projected | silhouette).sum().item()
        scores.append((projected & silhouette).sum().item() / union if union else 0.0)
    return scores
