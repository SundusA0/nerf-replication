import json
import math

import numpy as np
import pytest
import torch
from PIL import Image

from nerf.data import SphereScene, analytic_views, load_blender, look_at, orbit_poses
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


def test_analytic_views_carry_silhouettes():
    views = analytic_views(3, image_size=32, num_samples=128)
    assert views.alpha.shape == (3, 32, 32)
    for alpha in views.alpha:
        assert alpha[0, 0] == 0          # corner: empty
        assert alpha[16, 16] > 0.99      # centre: opaque object


# --------------------------------------------------------------------------
# The on-disk format of the NeRF synthetic ("Blender") scenes
# --------------------------------------------------------------------------


def write_blender_scene(directory, views, split="train"):
    """Write `views` the way the NeRF synthetic scenes are stored: RGBA PNGs
    with straight (not premultiplied) colour, plus transforms_<split>.json."""
    (directory / split).mkdir(parents=True)
    alpha = views.alpha[..., None]
    # undo the white compositing wherever the pixel is not empty
    colour = torch.where(
        alpha > 0, (views.images - (1 - alpha)) / alpha.clamp(min=1e-6), torch.zeros_like(views.images)
    )
    rgba = torch.cat([colour.clamp(0, 1), alpha], dim=-1)

    frames = []
    for k, (image, pose) in enumerate(zip(rgba, views.poses)):
        Image.fromarray((image * 255).round().byte().numpy()).save(directory / split / f"r_{k}.png")
        frames.append({"file_path": f"./{split}/r_{k}", "rotation": 0.1, "transform_matrix": pose.tolist()})
    field_of_view = 2 * math.atan(0.5 * views.width / views.focal)
    meta = {"camera_angle_x": field_of_view, "frames": frames}
    (directory / f"transforms_{split}.json").write_text(json.dumps(meta))


@pytest.fixture
def blender_scene(tmp_path):
    views = analytic_views(4, image_size=32, num_samples=128)
    write_blender_scene(tmp_path, views, "train")
    return tmp_path, views


def test_load_blender_reads_back_what_was_written(blender_scene):
    directory, views = blender_scene
    loaded = load_blender(directory, "train")
    assert len(loaded) == 4
    assert torch.equal(loaded.poses, views.poses)
    assert math.isclose(loaded.focal, views.focal, rel_tol=1e-6)
    assert (loaded.near, loaded.far, loaded.white_background) == (2.0, 6.0, True)
    # equal up to the 8-bit quantisation of the PNGs
    assert torch.allclose(loaded.images, views.images, atol=2 / 255)
    assert torch.allclose(loaded.alpha, views.alpha, atol=1 / 255)


def test_load_blender_composites_straight_alpha_over_white(tmp_path):
    pixels = np.array(
        [
            [[255, 0, 0, 255], [0, 255, 0, 128]],   # opaque red, half-transparent green
            [[10, 20, 30, 0], [0, 0, 0, 255]],      # fully transparent, opaque black
        ],
        dtype=np.uint8,
    )
    (tmp_path / "train").mkdir()
    Image.fromarray(pixels).save(tmp_path / "train" / "r_0.png")
    frame = {"file_path": "./train/r_0", "transform_matrix": torch.eye(4).tolist()}
    (tmp_path / "transforms_train.json").write_text(json.dumps({"camera_angle_x": 0.6911, "frames": [frame]}))

    image = load_blender(tmp_path).images[0]
    a = 128 / 255
    expected = torch.tensor(
        [
            [[1.0, 0.0, 0.0], [1 - a, 1.0, 1 - a]],
            [[1.0, 1.0, 1.0], [0.0, 0.0, 0.0]],
        ]
    )
    assert torch.allclose(image, expected, atol=1e-6)


def test_load_blender_downscale_averages_rgba_and_scales_the_focal_length(blender_scene):
    directory, _ = blender_scene
    full = load_blender(directory)
    half = load_blender(directory, downscale=2)
    assert half.images.shape == (4, 16, 16, 3)
    assert math.isclose(half.focal, full.focal / 2)
    assert torch.equal(half.poses, full.poses)

    # Expected result, worked out from the files themselves: average the RGBA
    # values over 2x2 blocks first, then composite over white. The order
    # matters on object edges, where a block mixes opaque and empty pixels.
    for k in range(4):
        rgba = np.asarray(Image.open(directory / "train" / f"r_{k}.png"), dtype=np.float32) / 255.0
        rgba = torch.from_numpy(rgba).reshape(16, 2, 16, 2, 4).mean(dim=(1, 3))
        expected = rgba[..., :3] * rgba[..., 3:] + (1 - rgba[..., 3:])
        assert torch.allclose(half.images[k], expected, atol=1e-6)
        assert torch.allclose(half.alpha[k], rgba[..., 3], atol=1e-6)
    mixed = (half.alpha > 0.01) & (half.alpha < 0.99)
    assert mixed.any()  # the scene does have such edge blocks, so the order is exercised


def test_load_blender_split_and_skip(tmp_path):
    write_blender_scene(tmp_path, analytic_views(4, image_size=8, num_samples=16), "train")
    write_blender_scene(tmp_path, analytic_views(2, image_size=8, num_samples=16, phase=40.0), "val")
    train = load_blender(tmp_path, "train")
    assert len(load_blender(tmp_path, "val")) == 2
    every_other = load_blender(tmp_path, "train", skip=2)
    assert len(every_other) == 2
    assert torch.equal(every_other.poses, train.poses[::2])
