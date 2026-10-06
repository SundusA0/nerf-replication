import copy

import pytest
import torch

from nerf.data import SphereScene, analytic_views
from nerf.hull import bounding_box, reprojection_iou, visual_hull


@pytest.fixture(scope="module")
def views():
    return analytic_views(24, image_size=48)


@pytest.fixture(scope="module")
def hull(views):
    return visual_hull(views, resolution=64)


def grid_of(axis):
    return torch.stack(torch.meshgrid(axis, axis, axis, indexing="ij"), dim=-1)


def test_hull_contains_the_interior_of_every_sphere(hull):
    occupied, axis = hull
    grid = grid_of(axis)
    # Points at least 0.1 inside a sphere. The margin allows for pixels on the
    # silhouette edge, where a point can fall just outside the mask.
    deep = torch.zeros_like(occupied)
    for centre, radius, _ in SphereScene.SPHERES:
        deep |= (grid - torch.tensor(centre)).norm(dim=-1) < radius - 0.1
    assert deep.sum() > 1000
    assert occupied[deep].all()


def test_hull_is_about_the_size_of_the_object(hull):
    occupied, axis = hull
    grid = grid_of(axis)
    sigma, _ = SphereScene()(grid, grid)
    ratio = occupied.sum().item() / (sigma > 0).sum().item()
    assert 0.8 < ratio < 1.3
    for low, high in bounding_box(occupied, axis):
        assert -0.9 < low < -0.5      # the red sphere reaches -0.7 on every axis
        assert 1.0 < high < 1.6       # the outer spheres reach 1.45 (x, y) and 1.2 (z)


def test_hull_projects_back_onto_the_silhouettes(views, hull):
    scores = reprojection_iou(views, *hull)
    assert len(scores) == len(views)
    assert min(scores) > 0.95


def with_mistake(views, mistake):
    wrong = copy.copy(views)
    if mistake == "mirrored images":
        wrong.alpha = views.alpha.flip(2)
    elif mistake == "upside-down images":
        wrong.alpha = views.alpha.flip(1)
    elif mistake == "world-to-camera poses":
        wrong.poses = views.poses.clone()
        wrong.poses[:, :3, :3] = views.poses[:, :3, :3].transpose(1, 2)
    elif mistake == "OpenCV axes":  # y down, z forward
        wrong.poses = views.poses.clone()
        wrong.poses[:, :3, 1:3] *= -1
    return wrong


@pytest.mark.parametrize(
    "mistake", ["mirrored images", "upside-down images", "world-to-camera poses", "OpenCV axes"]
)
def test_wrong_camera_conventions_are_detected(views, mistake):
    wrong = with_mistake(views, mistake)
    occupied, axis = visual_hull(wrong, resolution=64)
    if occupied.any():
        scores = reprojection_iou(wrong, occupied, axis)
        assert sum(scores) / len(scores) < 0.8
    else:
        assert bounding_box(occupied, axis) is None


def test_hull_needs_silhouettes(views):
    without = copy.copy(views)
    without.alpha = None
    with pytest.raises(ValueError):
        visual_hull(without)
