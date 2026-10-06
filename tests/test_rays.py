import math

import torch

from nerf.rays import get_rays, project_points

H, W, FOCAL = 6, 8, 10.0


def test_shapes_and_origin():
    c2w = torch.eye(4, dtype=torch.float64)
    c2w[:3, 3] = torch.tensor([1.0, 2.0, 3.0])
    origins, directions = get_rays(H, W, FOCAL, c2w)
    assert origins.shape == (H, W, 3)
    assert directions.shape == (H, W, 3)
    # every ray starts at the camera position
    assert torch.allclose(origins, c2w[:3, 3].expand(H, W, 3))


def test_identity_camera_looks_down_minus_z():
    _, directions = get_rays(H, W, FOCAL, torch.eye(4, dtype=torch.float64))
    centre = directions[H // 2, W // 2]
    assert torch.allclose(centre, torch.tensor([0.0, 0.0, -1.0], dtype=torch.float64))
    # depth parameterisation: every direction has camera-space z = -1
    assert torch.allclose(directions[..., 2], -torch.ones(H, W, dtype=torch.float64))


def test_image_axes():
    _, directions = get_rays(H, W, FOCAL, torch.eye(4, dtype=torch.float64))
    # moving right in the image (larger column) moves the ray towards +x
    assert directions[0, W - 1, 0] > directions[0, 0, 0]
    # moving down in the image (larger row) moves the ray towards -y
    assert directions[H - 1, 0, 1] < directions[0, 0, 1]
    # one pixel step changes the direction by 1 / focal
    step = directions[0, 1, 0] - directions[0, 0, 0]
    assert math.isclose(step.item(), 1.0 / FOCAL)


def test_rays_follow_camera_rotation():
    # Rotate the camera 90 degrees about the world y axis: its -z axis, the
    # viewing direction, now points along world -x.
    angle = math.pi / 2
    c2w = torch.eye(4, dtype=torch.float64)
    c2w[:3, :3] = torch.tensor(
        [
            [math.cos(angle), 0.0, math.sin(angle)],
            [0.0, 1.0, 0.0],
            [-math.sin(angle), 0.0, math.cos(angle)],
        ],
        dtype=torch.float64,
    )
    _, directions = get_rays(H, W, FOCAL, c2w)
    centre = directions[H // 2, W // 2]
    assert torch.allclose(
        centre, torch.tensor([-1.0, 0.0, 0.0], dtype=torch.float64), atol=1e-12
    )


def test_rotation_preserves_ray_lengths():
    # A rotation must not change the length of any direction.
    torch.manual_seed(0)
    q, _ = torch.linalg.qr(torch.randn(3, 3, dtype=torch.float64))
    c2w = torch.eye(4, dtype=torch.float64)
    c2w[:3, :3] = q
    _, rotated = get_rays(H, W, FOCAL, c2w)
    _, reference = get_rays(H, W, FOCAL, torch.eye(4, dtype=torch.float64))
    assert torch.allclose(rotated.norm(dim=-1), reference.norm(dim=-1))


def test_accepts_3x4_pose():
    c2w = torch.eye(4, dtype=torch.float64)
    full = get_rays(H, W, FOCAL, c2w)
    cropped = get_rays(H, W, FOCAL, c2w[:3])
    assert torch.allclose(full[0], cropped[0]) and torch.allclose(full[1], cropped[1])


def test_project_points_inverts_get_rays():
    # Any point on the ray through pixel (row, col) must project back to
    # (row, col), at a depth equal to its ray parameter.
    torch.manual_seed(0)
    rotation, _ = torch.linalg.qr(torch.randn(3, 3, dtype=torch.float64))
    if torch.linalg.det(rotation) < 0:
        rotation[:, 0] *= -1
    c2w = torch.eye(4, dtype=torch.float64)
    c2w[:3, :3] = rotation
    c2w[:3, 3] = torch.tensor([0.3, -1.2, 2.0], dtype=torch.float64)

    origins, directions = get_rays(H, W, FOCAL, c2w)
    expected_cols = torch.arange(W, dtype=torch.float64).expand(H, W)
    expected_rows = torch.arange(H, dtype=torch.float64)[:, None].expand(H, W)
    for t in (0.5, 3.0):
        cols, rows, depth = project_points(origins + t * directions, H, W, FOCAL, c2w)
        assert torch.allclose(cols, expected_cols, atol=1e-9)
        assert torch.allclose(rows, expected_rows, atol=1e-9)
        assert torch.allclose(depth, torch.full_like(depth, t))


def test_points_behind_the_camera_have_negative_depth():
    c2w = torch.eye(4, dtype=torch.float64)
    points = torch.tensor([[0.0, 0.0, -2.0], [0.0, 0.0, 2.0]], dtype=torch.float64)
    _, _, depth = project_points(points, H, W, FOCAL, c2w)
    assert depth.tolist() == [2.0, -2.0]
