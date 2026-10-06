import math

import torch

from nerf.data import SphereScene, analytic_views, look_at, orbit_poses
from nerf.rays import get_rays


def test_look_at_points_the_camera_at_the_target():
    position = torch.tensor([3.0, -2.0, 1.5])
    target = torch.tensor([0.2, 0.1, -0.3])
    c2w = look_at(position, target)
    expected = (target - position) / (target - position).norm()

    assert torch.allclose(c2w[:3, 3], position)
    # the camera looks down its own -z axis
    assert torch.allclose(-c2w[:3, 2], expected, atol=1e-6)
    # and the ray through the centre pixel agrees
    _, directions = get_rays(8, 8, 20.0, c2w)
    assert torch.allclose(directions[4, 4], expected, atol=1e-6)


def test_look_at_is_a_rotation_with_world_up_pointing_up():
    rotation = look_at((3.0, -2.0, 1.5))[:3, :3]
    assert torch.allclose(rotation.T @ rotation, torch.eye(3), atol=1e-6)
    assert math.isclose(torch.linalg.det(rotation).item(), 1.0, abs_tol=1e-6)
    assert rotation[2, 1] > 0           # camera up has a positive world-z part
    assert abs(rotation[2, 0]) < 1e-6   # camera right is horizontal: no roll


def test_orbit_cameras_are_on_the_upper_hemisphere_facing_the_origin():
    poses = orbit_poses(12, radius=4.0)
    positions = poses[:, :3, 3]
    assert torch.allclose(positions.norm(dim=-1), torch.full((12,), 4.0), atol=1e-5)
    assert (positions[:, 2] > 0).all()
    assert torch.allclose(-poses[:, :3, 2], -positions / 4.0, atol=1e-5)

    # no two cameras coincide, and another phase gives other viewpoints
    assert torch.pdist(positions).min() > 0.1
    shifted = orbit_poses(12, radius=4.0, phase=40.0)[:, :3, 3]
    assert torch.cdist(positions, shifted).min() > 0.1


def test_sphere_scene_density_and_colour():
    scene = SphereScene(density=40.0)
    points = torch.tensor(
        [
            [0.0, 0.0, 0.0],    # centre of the red sphere
            [1.1, 0.0, 0.0],    # centre of the green sphere
            [0.0, 1.1, 0.0],    # centre of the blue sphere
            [2.0, 2.0, 2.0],    # empty space
        ]
    )
    sigma, rgb = scene(points, torch.zeros_like(points))
    assert sigma.tolist() == [40.0, 40.0, 40.0, 0.0]
    assert rgb[:3].argmax(dim=-1).tolist() == [0, 1, 2]
    assert torch.equal(rgb[3], torch.zeros(3))


def test_analytic_views_have_the_expected_form():
    views = analytic_views(6, image_size=32, num_samples=128)
    assert len(views) == 6
    assert views.images.shape == (6, 32, 32, 3)
    assert views.poses.shape == (6, 4, 4)
    assert (views.height, views.width) == (32, 32)
    assert (views.near, views.far, views.white_background) == (2.0, 6.0, True)
    assert views.images.min() >= 0 and views.images.max() <= 1

    for image in views.images:
        assert torch.allclose(image[0, 0], torch.ones(3))   # corner: background
        assert image[16, 16].min() < 0.9                    # centre: an object


def test_up_in_the_world_is_up_in_the_image():
    # The yellow sphere sits on top of the red one (+z), so in every view it
    # must appear above it, i.e. at smaller row indices.
    views = analytic_views(6, image_size=32, num_samples=128)
    rows = torch.arange(32.0)[:, None].expand(32, 32)
    for image in views.images:
        r, g, b = image.unbind(dim=-1)
        yellow = (r > 0.8) & (g > 0.6) & (b < 0.3)
        red = (r > 0.7) & (g < 0.3) & (b < 0.3)
        assert yellow.any() and red.any()
        assert rows[yellow].mean() < rows[red].mean()


def test_views_are_deterministic_and_phase_changes_them():
    a = analytic_views(3, image_size=16, num_samples=64)
    b = analytic_views(3, image_size=16, num_samples=64)
    c = analytic_views(3, image_size=16, num_samples=64, phase=40.0)
    assert torch.equal(a.images, b.images)
    assert not torch.allclose(a.images, c.images)
