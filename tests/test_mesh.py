"""The mesh module, and the script that turns a checkpoint into a mesh.

Geometry is checked against shapes whose surface, volume and outline are known
exactly: a cube built by hand, balls given by a formula, and the analytic
sphere scene that the other tests use.
"""

import importlib.util
import json
import math
import re
import shutil
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from nerf import mesh as meshing
from nerf.data import SphereScene, Views, analytic_views, orbit_poses
from nerf.mesh import Mesh
from nerf.rays import get_rays
from nerf.render import render_rays
from nerf.train import TrainConfig, TrainedNetworks, load_networks, weights_fingerprint
from test_data import write_blender_scene
from test_evaluate import train

ROOT = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------------
# Shapes with known answers
# --------------------------------------------------------------------------


def cube(side=1.0, centre=(0.0, 0.0, 0.0)):
    """An axis-aligned cube: 8 corners and 12 triangles that face outwards."""
    corners = torch.tensor(
        [[x, y, z] for z in (0.0, 1.0) for y in (0.0, 1.0) for x in (0.0, 1.0)]
    )   # corner number = x + 2y + 4z
    faces = torch.tensor([
        [0, 2, 3], [0, 3, 1],   # bottom, z = 0
        [4, 5, 7], [4, 7, 6],   # top, z = 1
        [0, 1, 5], [0, 5, 4],   # y = 0
        [2, 6, 7], [2, 7, 3],   # y = 1
        [0, 4, 6], [0, 6, 2],   # x = 0
        [1, 3, 7], [1, 7, 5],   # x = 1
    ])
    return Mesh((corners - 0.5) * side + torch.tensor(centre), faces)


def together(*meshes):
    """Several meshes as one."""
    vertices, faces, offset = [], [], 0
    for mesh in meshes:
        vertices.append(mesh.vertices)
        faces.append(mesh.faces + offset)
        offset += mesh.vertices.shape[0]
    return Mesh(torch.cat(vertices), torch.cat(faces))


def hard_ball(centre=(0.0, 0.0, 0.0), radius=0.5, colour=(0.2, 0.6, 0.9)):
    """Density 40 inside a ball and 0 outside, in one colour."""
    def field(points, view_dirs):
        inside = (points - points.new_tensor(centre)).norm(dim=-1) < radius
        return 40.0 * inside, points.new_tensor(colour).expand(points.shape)
    return field


def distance_to_ball(centre=(0.0, 0.0, 0.0), radius=0.5):
    """radius - distance from the centre: smooth, and zero exactly on the sphere."""
    def field(points, view_dirs):
        return radius - (points - points.new_tensor(centre)).norm(dim=-1), torch.zeros_like(points)
    return field


@pytest.fixture(scope="module")
def smooth_ball():
    centre, radius, bound, resolution = (0.3, -0.2, 0.1), 0.5, 1.0, 65
    density = meshing.density_grid(distance_to_ball(centre, radius), bound, resolution)
    return meshing.extract_surface(density, bound, threshold=0.0), torch.tensor(centre), radius


SCENE_BOUND, SCENE_RESOLUTION = 1.6, 97
SCENE_VOXEL = 2 * SCENE_BOUND / (SCENE_RESOLUTION - 1)
SCENE_VOLUME = 4 / 3 * math.pi * sum(radius**3 for _, radius, _ in SphereScene.SPHERES)


@pytest.fixture(scope="module")
def scene_mesh():
    """The surface of the analytic sphere scene, from its exact density."""
    density = meshing.density_grid(SphereScene(), SCENE_BOUND, SCENE_RESOLUTION)
    return meshing.extract_surface(density, SCENE_BOUND, threshold=20.0)


def to_true_surfaces(points):
    """(N, 4) distance from each point to the surface of each sphere of the scene."""
    centres = torch.tensor([centre for centre, _, _ in SphereScene.SPHERES])
    radii = torch.tensor([radius for _, radius, _ in SphereScene.SPHERES])
    return ((points[:, None] - centres[None]).norm(dim=-1) - radii[None]).abs()


def camera_on_the_z_axis(distance=5.0):
    """A camera at (0, 0, distance) looking at the origin: +x right, +y up."""
    pose = torch.eye(4)
    pose[2, 3] = distance
    return pose


# --------------------------------------------------------------------------
# Measures
# --------------------------------------------------------------------------


def test_area_volume_and_closedness_of_a_cube():
    box = cube(side=2.0, centre=(0.3, -0.2, 0.5))
    assert meshing.surface_area(box) == pytest.approx(24.0, abs=1e-6)
    assert meshing.enclosed_volume(box) == pytest.approx(8.0, abs=1e-6)   # wherever the cube is
    assert meshing.is_closed(box)

    inside_out = Mesh(box.vertices, box.faces.flip(1))
    assert meshing.enclosed_volume(inside_out) == pytest.approx(-8.0, abs=1e-6)
    assert meshing.is_closed(inside_out)              # closed and consistent, only facing inwards

    with_a_hole = Mesh(box.vertices, box.faces[1:])
    assert not meshing.is_closed(with_a_hole)
    one_flipped = Mesh(box.vertices, torch.cat([box.faces[:1].flip(1), box.faces[1:]]))
    assert not meshing.is_closed(one_flipped)
    one_twice = Mesh(box.vertices, torch.cat([box.faces, box.faces[:1]]))
    assert not meshing.is_closed(one_twice)

    # Two cubes that share an edge: four faces meet along it, two running each
    # way. Nothing is open, and the volume is that of both.
    pair = together(cube(side=2.0), cube(side=2.0, centre=(2.0, 2.0, 0.0)))
    assert torch.equal(pair.vertices[8], pair.vertices[3]) and torch.equal(pair.vertices[12], pair.vertices[7])
    faces = pair.faces.clone()
    faces[faces == 8], faces[faces == 12] = 3, 7      # one vertex each where the two cubes had one apiece
    touching = Mesh(pair.vertices, faces)
    assert meshing.is_closed(touching)
    assert meshing.enclosed_volume(touching) == pytest.approx(16.0, abs=1e-6)


def test_pieces_are_told_apart_and_small_ones_dropped():
    big, medium, speck = cube(1.0), cube(0.3, (2.0, 0.0, 0.0)), cube(0.05, (0.0, 2.0, 0.0))
    everything = together(big, medium, speck)
    piece = meshing.piece_of_each_face(everything)
    assert piece.unique().numel() == 3
    for first in (0, 12, 24):                      # the 12 faces of each cube belong together
        assert piece[first : first + 12].unique().numel() == 1

    # areas are 6, 0.54 and 0.015: the speck is under 1 % of the largest piece
    cleaned, kept, dropped = meshing.drop_small_pieces(everything, min_fraction=0.01)
    assert (kept, dropped) == (2, 1)
    assert cleaned.vertices.shape == (16, 3) and cleaned.faces.shape == (24, 3)
    assert cleaned.faces.max() == 15              # renumbered to the vertices that are left
    assert meshing.is_closed(cleaned)
    assert meshing.enclosed_volume(cleaned) == pytest.approx(1.0 + 0.3**3, abs=1e-6)
    assert meshing.surface_area(cleaned) == pytest.approx(6.0 + 6 * 0.3**2, abs=1e-6)

    assert meshing.drop_small_pieces(everything, min_fraction=0.0)[1:] == (3, 0)
    only_big, kept, dropped = meshing.drop_small_pieces(everything, min_fraction=0.5)
    assert (kept, dropped) == (1, 2)
    assert meshing.enclosed_volume(only_big) == pytest.approx(1.0, abs=1e-6)


# --------------------------------------------------------------------------
# From a density to a surface
# --------------------------------------------------------------------------


def test_grid_is_indexed_x_y_z_and_sampled_without_a_direction():
    seen = []

    def field(points, view_dirs):
        seen.append(view_dirs)
        return points[..., 0] + 10 * points[..., 1] + 100 * points[..., 2], torch.zeros_like(points)

    grid = meshing.density_grid(field, bound=2.0, resolution=5, chunk=7)   # 125 points in uneven chunks
    axis = torch.linspace(-2.0, 2.0, 5)
    expected = axis[:, None, None] + 10 * axis[None, :, None] + 100 * axis[None, None, :]
    assert grid.shape == (5, 5, 5)
    assert torch.allclose(grid, expected, atol=1e-5)
    assert len(seen) == 18 and all((dirs == 0).all() for dirs in seen)
    assert torch.equal(grid, meshing.density_grid(field, bound=2.0, resolution=5))   # in one chunk


def test_surface_of_a_smooth_ball(smooth_ball):
    surface, centre, radius = smooth_ball
    # every vertex is on the sphere, in the right place
    distance = (surface.vertices - centre).norm(dim=-1)
    assert (distance - radius).abs().max() < 1e-3
    assert torch.allclose(surface.vertices.mean(dim=0), centre, atol=2e-3)
    # the mesh is closed, faces outwards and has the sphere's volume and area
    assert meshing.is_closed(surface)
    assert meshing.piece_of_each_face(surface).unique().numel() == 1
    assert meshing.enclosed_volume(surface) == pytest.approx(4 / 3 * math.pi * radius**3, rel=5e-3)
    assert meshing.surface_area(surface) == pytest.approx(4 * math.pi * radius**2, rel=5e-3)
    # and so do its vertex normals
    normals = meshing.vertex_normals(surface)
    assert torch.allclose(normals.norm(dim=-1), torch.ones(normals.shape[0]), atol=1e-5)
    outward = (surface.vertices - centre) / distance[:, None]
    assert (normals * outward).sum(dim=-1).min() > 0.99


def test_threshold_must_lie_inside_the_density_range():
    density = meshing.density_grid(hard_ball(), bound=1.0, resolution=17)
    for threshold in (50.0, 40.0, 0.0, -1.0):
        with pytest.raises(ValueError, match="ranges from 0 to 40"):
            meshing.extract_surface(density, 1.0, threshold)
    assert meshing.extract_surface(density, 1.0, 39.0).faces.shape[0] > 0


def test_surface_of_the_sphere_scene(scene_mesh):
    # Four spheres, apart from one another except that two just touch.
    assert meshing.is_closed(scene_mesh)
    assert meshing.piece_of_each_face(scene_mesh).unique().numel() == 4
    assert meshing.enclosed_volume(scene_mesh) == pytest.approx(SCENE_VOLUME, rel=0.01)
    # The density is a step, so the surface is only known to lie between two
    # grid points: each vertex is within half a grid spacing of a true sphere.
    assert to_true_surfaces(scene_mesh.vertices).min(dim=1).values.max() <= SCENE_VOXEL / 2 + 1e-6


def hollow_ball(outer=0.6, inner=0.3, inside=0.0, tunnel=False):
    """Density 40 between two spheres, `inside` within the inner one, 0 outside.
    With a tunnel, a square shaft along +x joins the hollow to the outside."""
    def field(points, view_dirs):
        distance = points.norm(dim=-1)
        sigma = torch.where(distance < inner, inside, 40.0 * (distance < outer))
        if tunnel:
            shaft = (points[..., 0] > 0) & (points[..., 1].abs() < 0.08) & (points[..., 2].abs() < 0.08)
            sigma = torch.where(shaft, 0.0, sigma)
        return sigma, torch.zeros_like(points)
    return field


def test_pockets_sealed_inside_the_object_are_filled():
    bound, resolution = 1.0, 65
    density = meshing.density_grid(hollow_ball(), bound, resolution)
    axis = torch.linspace(-bound, bound, resolution)
    distance = torch.stack(torch.meshgrid(axis, axis, axis, indexing="ij"), dim=-1).norm(dim=-1)

    # As it is, the hollow has a wall of its own: two closed surfaces, and the
    # volume between them.
    raw = meshing.extract_surface(density, bound, 20.0)
    assert meshing.piece_of_each_face(raw).unique().numel() == 2
    shell_volume = 4 / 3 * math.pi * (0.6**3 - 0.3**3)
    assert meshing.enclosed_volume(raw) == pytest.approx(shell_volume, rel=0.02)

    filled, count = meshing.fill_cavities(density, 20.0)
    assert count == int((distance < 0.3).sum())                  # exactly the grid points in the hollow
    assert torch.equal(filled[distance >= 0.3], density[distance >= 0.3])
    assert (filled[distance < 0.3] == 40.0).all()
    assert torch.equal(density, meshing.density_grid(hollow_ball(), bound, resolution))   # the grid given is untouched
    solid = meshing.extract_surface(filled, bound, 20.0)
    assert meshing.piece_of_each_face(solid).unique().numel() == 1
    assert meshing.is_closed(solid)
    assert meshing.enclosed_volume(solid) == pytest.approx(4 / 3 * math.pi * 0.6**3, rel=0.02)
    # the outer surface is where it was: the same vertices as the outer piece before
    outer = raw.vertices[raw.vertices.norm(dim=-1) > 0.45]
    assert torch.equal(solid.vertices.sort(dim=0).values, outer.sort(dim=0).values)

    # What counts as empty depends on the threshold: a hollow holding density
    # 10 is a pocket at a threshold of 20 and part of the object at 5. At a
    # threshold of exactly 10 it is still empty, as it is to marching cubes,
    # which draws its inner wall.
    faint = meshing.density_grid(hollow_ball(inside=10.0), bound, resolution)
    assert meshing.fill_cavities(faint, 20.0)[1] == count
    assert meshing.fill_cavities(faint, 5.0)[1] == 0
    assert meshing.piece_of_each_face(meshing.extract_surface(faint, bound, 10.0)).unique().numel() == 2
    at_the_threshold, filled_points = meshing.fill_cavities(faint, 10.0)
    assert filled_points == count
    assert meshing.piece_of_each_face(meshing.extract_surface(at_the_threshold, bound, 10.0)).unique().numel() == 1

    # A hollow with a way out is not a pocket, and nothing is changed.
    open_to_outside = meshing.density_grid(hollow_ball(tunnel=True), bound, resolution)
    unchanged, count = meshing.fill_cavities(open_to_outside, 20.0)
    assert count == 0 and torch.equal(unchanged, open_to_outside)
    # Nor is the empty space around an object that the grid cuts through.
    cut = meshing.density_grid(hard_ball(radius=1.2), bound, resolution)
    assert meshing.fill_cavities(cut, 20.0)[1] == 0


def test_empty_space_that_reaches_a_face_of_the_grid_is_the_outside():
    solid = torch.full((9, 9, 9), 40.0)

    def pockets(*empty_points):
        block = solid.clone()
        for point in empty_points:
            block[point] = 0.0
        return meshing.fill_cavities(block, 20.0)[1]

    for axis in range(3):
        for face, beside in ((0, 1), (8, 7)):
            on_the_face, one_in = [4, 4, 4], [4, 4, 4]
            on_the_face[axis], one_in[axis] = face, beside
            assert pockets(tuple(on_the_face)) == 0         # open to the outside
            assert pockets(tuple(one_in)) == 1              # one step in, it is sealed
            assert pockets(tuple(on_the_face), tuple(one_in)) == 0   # unless joined to the face
    # neighbours along the axes join; neighbours across a diagonal do not
    assert pockets((0, 4, 4), (1, 5, 4)) == 1
    assert pockets((4, 4, 4), (4, 4, 5), (2, 2, 2)) == 3


# --------------------------------------------------------------------------
# Colours and the file
# --------------------------------------------------------------------------


def test_vertices_take_the_colour_of_the_sphere_they_lie_on(scene_mesh):
    coloured = meshing.colour_vertices(SphereScene(), scene_mesh, SCENE_VOXEL, chunk=5000)
    assert coloured.colours.shape == scene_mesh.vertices.shape
    assert torch.equal(coloured.vertices, scene_mesh.vertices) and torch.equal(coloured.faces, scene_mesh.faces)

    distances = to_true_surfaces(scene_mesh.vertices)
    own = distances.argmin(dim=1)
    expected = torch.tensor([colour for _, _, colour in SphereScene.SPHERES])[own]
    # away from where two spheres meet the colour is exactly the sphere's
    others = distances.scatter(1, own[:, None], float("inf")).min(dim=1).values
    clear = others > 5 * SCENE_VOXEL
    assert clear.float().mean() > 0.8
    assert torch.allclose(coloured.colours[clear], expected[clear], atol=1e-4)
    # and every one of the four spheres is among them
    assert own[clear].unique().tolist() == [0, 1, 2, 3]


def test_colour_is_what_is_seen_looking_in_along_the_normal(smooth_ball):
    surface, centre, radius = smooth_ball

    def field(points, view_dirs):   # a solid ball whose colour says which way it is looked at
        inside = (points - centre).norm(dim=-1) < radius
        return 40.0 * inside, 0.5 + 0.5 * view_dirs

    coloured = meshing.colour_vertices(field, surface, voxel=2.0 / 64)
    inward = -(surface.vertices - centre) / radius
    assert torch.allclose(coloured.colours, 0.5 + 0.5 * inward, atol=0.02)


RED, GREEN, BLUE = (torch.tensor(colour) for colour in ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)))


@pytest.mark.parametrize("density, red", [(40.0, 0.940), (3.2, 0.699)])
def test_colour_comes_from_the_stretch_of_the_ray_and_mostly_from_the_skin(smooth_ball, density, red):
    surface, centre, radius = smooth_ball
    voxel = 2.0 / 64

    def painted(points, view_dirs):   # a solid ball, red in its outer two voxels and blue below
        depth = (radius - (points - centre).norm(dim=-1)) / voxel
        return density * (depth > -0.1), torch.where((depth < 1.9)[..., None], RED, BLUE).expand(points.shape)

    # The ray takes 16 steps of a quarter voxel, from one voxel outside to
    # three inside, with the density read where each step begins: 4 in empty
    # space, 8 in the red and 4 in the blue. What lies beyond its end does not
    # count.
    #   density 40, optical depth 0.3125 per step: the red absorbs
    #     1 - exp(-2.5) = 0.918 and the blue exp(-2.5) (1 - exp(-1.25)) = 0.059,
    #     so 94.0 % is red;
    #   density 3.2, optical depth 0.025 per step: 0.181 and 0.078, so 69.9 %.
    queried = []

    def counting(points, view_dirs):
        queried.append(points.shape)
        return painted(points, view_dirs)

    colours = meshing.colour_vertices(counting, surface, voxel).colours
    assert queried == [(surface.vertices.shape[0], 16, 3)]            # 16 readings of the field per ray
    assert torch.allclose(colours[:, 0], torch.full_like(colours[:, 0], red), atol=0.005)
    assert torch.allclose(colours[:, 2], torch.full_like(colours[:, 2], 1 - red), atol=0.005)
    assert colours[:, 1].abs().max() < 1e-6


def test_colour_takes_in_what_lies_just_outside_the_surface(smooth_ball):
    surface, centre, radius = smooth_ball
    voxel = 2.0 / 64

    def hazy(points, view_dirs):      # a red ball inside a thin green haze
        depth = (radius - (points - centre).norm(dim=-1)) / voxel
        haze = (depth > -0.9) & (depth <= -0.1)
        return torch.where(haze, 12.8, 40.0 * (depth > -0.1)), torch.where(haze[..., None], GREEN, RED).expand(points.shape)

    # Three of the ray's steps are in the haze, at an optical depth of 0.1 each:
    # it absorbs 1 - exp(-0.3) = 0.259, and the ball behind it 0.741 (1 - exp(-3.75)) = 0.723.
    colours = meshing.colour_vertices(hazy, surface, voxel).colours
    assert torch.allclose(colours[:, 1], torch.full_like(colours[:, 1], 0.2638), atol=0.005)
    assert torch.allclose(colours[:, 0], torch.full_like(colours[:, 0], 0.7362), atol=0.005)


def test_colour_of_a_thin_shell_is_not_dimmed_by_the_space_behind_it(smooth_ball):
    surface, centre, radius = smooth_ball
    voxel = 2.0 / 64
    colour = torch.tensor([0.2, 0.6, 0.9])

    def shell(points, view_dirs):     # density only in the outer 1.5 voxels; hollow inside
        depth = radius - (points - centre).norm(dim=-1)
        return 40.0 * ((depth > 0) & (depth < 1.5 * voxel)), colour.expand(points.shape)

    # The ray passes through the shell into empty space and is only 85 %
    # absorbed. What it did meet had one colour, and that is the answer.
    colours = meshing.colour_vertices(shell, surface, voxel).colours
    assert torch.allclose(colours, colour.expand_as(colours), atol=1e-3)


def test_a_vertex_whose_ray_meets_nothing_takes_the_colour_at_the_vertex():
    def tinted(density):              # colour says where the point is and which way it is looked at
        def field(points, view_dirs):
            return torch.full(points.shape[:-1], density), (0.5 + 0.25 * points + 0.25 * view_dirs).clamp(0, 1)
        return field

    triangle = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    # Empty space all along the ray: the colour the field gives at the vertex,
    # seen along the inward normal, which is (0, 0, -1) for this triangle.
    facing_up = Mesh(triangle, torch.tensor([[0, 1, 2]]))
    for steps in (16, 5, 1):          # however the ray is divided
        colours = meshing.colour_vertices(tinted(0.0), facing_up, voxel=0.1, samples=steps).colours
        assert torch.allclose(colours, 0.5 + 0.25 * triangle + torch.tensor([0.0, 0.0, -0.25]), atol=1e-6)

    # Two triangles back to back: their normals cancel at every vertex, so
    # there is no direction to look along and no ray. The same fallback,
    # whatever the density.
    flat = Mesh(triangle, torch.tensor([[0, 1, 2], [0, 2, 1]]))
    assert (meshing.vertex_normals(flat) == 0).all()
    colours = meshing.colour_vertices(tinted(40.0), flat, voxel=0.1).colours
    assert torch.allclose(colours, 0.5 + 0.25 * triangle, atol=1e-6)


def read_ply(path):
    """A reader written for this test: header lines, then the two binary tables."""
    with open(path, "rb") as handle:
        header = []
        while not header or header[-1] != "end_header":
            header.append(handle.readline().decode("ascii").rstrip("\n"))
        num_vertices = int(next(line for line in header if line.startswith("element vertex")).split()[-1])
        num_faces = int(next(line for line in header if line.startswith("element face")).split()[-1])
        has_colour = "property uchar red" in header
        vertex_type = [("xyz", "<f4", (3,))] + ([("rgb", "u1", (3,))] if has_colour else [])
        vertices = np.fromfile(handle, dtype=vertex_type, count=num_vertices)
        faces = np.fromfile(handle, dtype=[("count", "u1"), ("corners", "<i4", (3,))], count=num_faces)
        assert handle.read() == b""              # nothing after the faces
    return header, vertices, faces


def test_ply_file_holds_the_mesh(tmp_path):
    box = cube(side=2.0, centre=(0.3, -0.2, 0.5))
    colours = torch.linspace(0, 1, 24).reshape(8, 3)
    meshing.save_ply(tmp_path / "box.ply", Mesh(box.vertices, box.faces, colours))
    header, vertices, faces = read_ply(tmp_path / "box.ply")
    assert header == [
        "ply", "format binary_little_endian 1.0", "element vertex 8",
        "property float x", "property float y", "property float z",
        "property uchar red", "property uchar green", "property uchar blue",
        "element face 12", "property list uchar int vertex_indices", "end_header",
    ]
    assert np.array_equal(vertices["xyz"], box.vertices.numpy())
    assert np.array_equal(vertices["rgb"], (colours * 255).round().byte().numpy())
    assert (faces["count"] == 3).all() and np.array_equal(faces["corners"], box.faces.numpy())

    meshing.save_ply(tmp_path / "plain.ply", box)       # without colours
    header, vertices, faces = read_ply(tmp_path / "plain.ply")
    assert not any("uchar red" in line for line in header)
    assert np.array_equal(vertices["xyz"], box.vertices.numpy())
    assert np.array_equal(faces["corners"], box.faces.numpy())


# --------------------------------------------------------------------------
# Seeing a mesh through a camera
# --------------------------------------------------------------------------

# With this camera, focal length 50 and a 20 x 20 image, a point (x, y, 0)
# appears at column 10 + 10 x and row 10 - 10 y, at depth 5.
POSE, FOCAL, SIZE = camera_on_the_z_axis(5.0), 50.0, 20


def at_pixels(pixel_corners, z=0.0):
    """A triangle whose corners appear at the given (column, row) positions."""
    corners = torch.tensor(pixel_corners, dtype=torch.float32)
    depth = 5.0 - z
    x = (corners[:, 0] - 10.0) * depth / FOCAL
    y = -(corners[:, 1] - 10.0) * depth / FOCAL
    return Mesh(torch.stack([x, y, torch.full_like(x, z)], dim=-1), torch.tensor([[0, 1, 2]]))


def test_a_triangle_covers_exactly_the_pixels_inside_it():
    # Corners at (1.5, 1.5), (13.5, 1.5) and (1.5, 10.5). No pixel lies on an
    # edge: the slanted one is 3 * column + 4 * row = 46.5.
    triangle = at_pixels([[1.5, 1.5], [13.5, 1.5], [1.5, 10.5]])
    raster = meshing.rasterise(triangle, SIZE, SIZE, FOCAL, POSE)

    rows, cols = torch.meshgrid(torch.arange(SIZE), torch.arange(SIZE), indexing="ij")
    inside = (cols >= 2) & (rows >= 2) & (3 * cols + 4 * rows < 46.5)
    assert inside.sum() == 54        # 11 + 10 + 9 + 7 + 6 + 5 + 3 + 2 + 1, counting row by row
    assert torch.equal(raster.face >= 0, inside)
    assert (raster.face[inside] == 0).all() and (raster.face[~inside] == -1).all()
    assert torch.allclose(raster.depth[inside], torch.full((54,), 5.0), atol=1e-5)
    assert torch.isinf(raster.depth[~inside]).all()
    # where in the triangle each pixel is: the corners' weights
    weight_b = (cols[inside] - 1.5) / 12.0
    weight_c = (rows[inside] - 1.5) / 9.0
    expected = torch.stack([1 - weight_b - weight_c, weight_b, weight_c], dim=-1)
    assert torch.allclose(raster.weights[inside], expected, atol=1e-5)

    # seen from behind it covers the same pixels
    behind = Mesh(triangle.vertices, triangle.faces.flip(1))
    assert torch.equal(meshing.rasterise(behind, SIZE, SIZE, FOCAL, POSE).face >= 0, inside)


def test_the_nearest_triangle_wins():
    far = at_pixels([[1.5, 1.5], [13.5, 1.5], [1.5, 10.5]], z=0.0)        # depth 5
    near = at_pixels([[5.5, 0.5], [18.5, 0.5], [5.5, 18.5]], z=1.0)       # depth 4, overlapping it
    alone = meshing.rasterise(near, SIZE, SIZE, FOCAL, POSE).face >= 0
    for first, second, near_index in ((far, near, 1), (near, far, 0)):
        raster = meshing.rasterise(together(first, second), SIZE, SIZE, FOCAL, POSE)
        assert (raster.face[alone] == near_index).all()
        assert torch.allclose(raster.depth[alone], torch.full((int(alone.sum()),), 4.0), atol=1e-5)
        assert ((raster.face >= 0) & ~alone).sum() > 0                    # the far one shows beside it
        assert (raster.face[(raster.face >= 0) & ~alone] == 1 - near_index).all()


def test_of_triangles_at_the_same_depth_the_first_is_taken_however_the_work_is_split():
    triangle = at_pixels([[1.5, 1.5], [13.5, 1.5], [1.5, 10.5]])
    twice = together(triangle, triangle)
    for budget in (2_000_000, 1):      # both in one batch, then each in a batch of its own
        raster = meshing.rasterise(twice, SIZE, SIZE, FOCAL, POSE, budget=budget)
        assert (raster.face >= 0).sum() == 54 and (raster.face[raster.face >= 0] == 0).all()


def test_depth_and_position_are_those_of_the_pixels_own_ray():
    # A triangle tilted steeply in depth, seen by an off-axis camera. For
    # every covered pixel, the point the rasteriser reports must lie on the
    # ray that `get_rays` sends through that pixel, at the reported depth.
    pose = orbit_poses(7)[5]
    height, width, focal = 30, 40, 60.0
    triangle = Mesh(torch.tensor([[-0.9, -0.6, -0.8], [0.9, -0.2, 0.1], [0.1, 0.8, 1.0]]), torch.tensor([[0, 1, 2]]))
    raster = meshing.rasterise(triangle, height, width, focal, pose)
    covered = raster.face >= 0
    assert covered.sum() > 200
    assert raster.depth[covered].max() - raster.depth[covered].min() > 1.0      # really tilted

    origins, directions = get_rays(height, width, focal, pose)
    on_ray = origins[covered] + raster.depth[covered][:, None] * directions[covered]
    in_triangle = (raster.weights[covered][..., None] * triangle.vertices[None]).sum(dim=1)
    assert torch.allclose(in_triangle, on_ray, atol=1e-4)
    assert torch.allclose(raster.weights[covered].sum(dim=-1), torch.ones(int(covered.sum())), atol=1e-5)
    assert raster.weights[covered].min() >= 0


def cast_a_ray_at_every_triangle(mesh, height, width, focal, pose):
    """A second opinion for `rasterise` that shares no code with it.

    The ray that `get_rays` sends through each pixel is intersected with every
    triangle (the Moller-Trumbore test), and the nearest hit is kept. Returns
    the face hit at each pixel (-1 for none), its depth, and where in the face
    the hit is, as weights of its three corners.
    """
    origins, directions = get_rays(height, width, focal, pose)
    origin = origins.reshape(-1, 1, 3)                               # (P, 1, 3)
    direction = directions.reshape(-1, 1, 3)                         # not unit length: t is depth
    a, b, c = (mesh.vertices.double()[mesh.faces[:, k]][None] for k in range(3))   # (1, F, 3)
    p = torch.linalg.cross(direction, c - a)
    determinant = ((b - a) * p).sum(dim=-1)
    u = ((origin - a) * p).sum(dim=-1) / determinant
    q = torch.linalg.cross(origin - a, b - a)
    v = (direction * q).sum(dim=-1) / determinant
    t = ((c - a) * q).sum(dim=-1) / determinant
    hit = (u >= 0) & (v >= 0) & (u + v <= 1) & (t > 0)
    depth, face = torch.where(hit, t, float("inf")).min(dim=1)
    pixel = torch.arange(height * width)
    weights = torch.stack([1 - u - v, u, v], dim=-1)[pixel, face]
    face[torch.isinf(depth)] = -1
    return face.reshape(height, width), depth.reshape(height, width), weights.reshape(height, width, 3)


def test_rasteriser_agrees_with_casting_a_ray_at_every_triangle():
    # 400 triangles of every size and tilt, thrown into the space the cameras
    # look at, so that many hide parts of others.
    generator = torch.Generator().manual_seed(0)
    centres = (torch.rand(400, 1, 3, generator=generator) - 0.5) * 2.4
    sizes = 0.01 + 1.5 * torch.rand(400, 1, 1, generator=generator) ** 3
    corners = centres + (torch.rand(400, 3, 3, generator=generator) - 0.5) * sizes
    cloud = Mesh(corners.reshape(-1, 3), torch.arange(1200).reshape(400, 3))

    for pose, (height, width), budget in zip(orbit_poses(3, phase=25.0), ((48, 64), (64, 48), (40, 40)),
                                             (2_000_000, 500, 2_000_000)):
        # The camera in double precision, with its axes made perpendicular to
        # rounding error. The two methods then place every point within about
        # 1e-13 of a pixel of each other, and can only disagree about a pixel
        # whose centre is that close to an edge.
        pose = pose.double()
        axes, stretch = torch.linalg.qr(pose[:3, :3])
        pose[:3, :3] = axes * torch.sign(torch.diagonal(stretch))
        focal = 0.5 * width / math.tan(0.5 * 0.6911)

        raster = meshing.rasterise(cloud, height, width, focal, pose, budget=budget)
        face, depth, weights = cast_a_ray_at_every_triangle(cloud, height, width, focal, pose)
        covered = face >= 0
        assert 0.4 < covered.float().mean() < 0.8                    # plenty of object and of background
        assert face[covered].unique().numel() > 100                  # and of triangles seen
        assert torch.equal(raster.face, face)
        assert torch.allclose(raster.depth[covered].double(), depth[covered], atol=1e-5)
        assert torch.isinf(raster.depth[~covered]).all()
        assert torch.allclose(raster.weights[covered].double(), weights[covered], atol=1e-5)


def test_triangles_behind_the_camera_or_off_the_image_are_left_out():
    visible = at_pixels([[1.5, 1.5], [13.5, 1.5], [1.5, 10.5]])
    expected = meshing.rasterise(visible, SIZE, SIZE, FOCAL, POSE).face >= 0

    behind = Mesh(visible.vertices + torch.tensor([0.0, 0.0, 10.0]), visible.faces)      # z = 10, camera at 5
    assert (meshing.rasterise(behind, SIZE, SIZE, FOCAL, POSE).face == -1).all()
    # Two corners in view, at the top corners of the image, and the third
    # behind the camera. Projected as if it were in front, the third would
    # land below the image at (10, 22.5) and the triangle would cover most of
    # the picture.
    straddling = Mesh(torch.tensor([[-1.0, 1.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 9.0]]), visible.faces)
    assert (meshing.rasterise(straddling, SIZE, SIZE, FOCAL, POSE).face == -1).all()

    # partly off the image: only the part on it; wholly off: nothing; no error either way
    hanging = at_pixels([[-6.5, 4.5], [8.5, 4.5], [-6.5, 30.5]])
    raster = meshing.rasterise(hanging, SIZE, SIZE, FOCAL, POSE)
    rows, cols = torch.meshgrid(torch.arange(SIZE), torch.arange(SIZE), indexing="ij")
    assert torch.equal(raster.face >= 0, (rows >= 5) & (26 * (cols + 6.5) + 15 * (rows - 4.5) < 390))
    off = at_pixels([[25.5, 1.5], [40.5, 1.5], [25.5, 10.5]])
    assert (meshing.rasterise(off, SIZE, SIZE, FOCAL, POSE).face == -1).all()

    # none of them disturbs a triangle that is in view
    crowd = together(behind, straddling, off, visible)
    raster = meshing.rasterise(crowd, SIZE, SIZE, FOCAL, POSE)
    assert torch.equal(raster.face >= 0, expected) and (raster.face[expected] == 3).all()


def test_the_work_can_be_split_without_changing_the_result(scene_mesh):
    views = analytic_views(1, image_size=48, phase=40.0, num_samples=32)
    whole = meshing.rasterise(scene_mesh, 48, 48, views.focal, views.poses[0])
    in_bits = meshing.rasterise(scene_mesh, 48, 48, views.focal, views.poses[0], budget=1000)
    assert torch.equal(whole.face, in_bits.face)
    assert torch.equal(whole.depth, in_bits.depth) and torch.equal(whole.weights, in_bits.weights)
    # also for a triangle far larger than the budget
    big = at_pixels([[-30.0, -20.0], [60.0, -10.0], [5.0, 70.0]])
    assert torch.equal(
        meshing.rasterise(big, SIZE, SIZE, FOCAL, POSE).face,
        meshing.rasterise(big, SIZE, SIZE, FOCAL, POSE, budget=64).face,
    )
    assert (meshing.rasterise(big, SIZE, SIZE, FOCAL, POSE).face == 0).all()     # it fills the image


def test_silhouette_score_is_the_overlap_with_the_opaque_pixels_over_the_union():
    # The triangle of the tests above: 54 pixels, those in columns and rows
    # from 2 on with 3 * column + 4 * row < 46.5.
    triangle = at_pixels([[1.5, 1.5], [13.5, 1.5], [1.5, 10.5]])
    rows, cols = torch.meshgrid(torch.arange(SIZE), torch.arange(SIZE), indexing="ij")
    inside = (cols >= 2) & (rows >= 2) & (3 * cols + 4 * rows < 46.5)

    def score(mesh, alpha, **options):
        views = Views(torch.ones(1, SIZE, SIZE, 3), POSE[None], FOCAL, 2.0, 6.0, True, alpha[None])
        return meshing.silhouette_iou(mesh, views, **options)[0]

    assert score(triangle, inside.float()) == 1.0
    # An object that is a block of 8 columns by 12 rows, 96 pixels. It shares
    # 8 + 8 + 8 + 7 + 6 + 5 + 3 + 2 + 1 = 48 with the triangle, counting row
    # by row, so the two together cover 54 + 96 - 48 = 102.
    block = (cols >= 2) & (cols < 10) & (rows >= 2) & (rows < 14)
    assert score(triangle, block.float()) == 48 / 102

    # A pixel counts as object where the photograph is more than half opaque,
    faint = torch.where(inside, 1.0, 0.4 * block)
    assert score(triangle, faint) == 1.0
    # or more than the threshold asked for.
    assert score(triangle, faint, threshold=0.3) == 54 / 102

    # Nothing in common scores 0. A mesh and an object that are both out of
    # view agree, and score 1.
    off = at_pixels([[25.5, 1.5], [40.5, 1.5], [25.5, 10.5]])
    assert score(off, block.float()) == 0.0
    assert score(triangle, torch.zeros(SIZE, SIZE)) == 0.0
    assert score(off, torch.zeros(SIZE, SIZE)) == 1.0


@pytest.fixture(scope="module")
def held_out():
    """Views of the sphere scene from the renderer, with their silhouettes."""
    return analytic_views(4, image_size=96, phase=40.0, num_samples=96)


def test_silhouette_of_the_scene_mesh_matches_the_renderer(scene_mesh, held_out):
    scores = meshing.silhouette_iou(scene_mesh, held_out)
    assert len(scores) == 4 and min(scores) > 0.95

    # a mesh in the wrong place, mirrored, or too big scores clearly lower
    moved = Mesh(scene_mesh.vertices + torch.tensor([0.25, 0.0, 0.0]), scene_mesh.faces)
    mirrored = Mesh(scene_mesh.vertices * torch.tensor([-1.0, 1.0, 1.0]), scene_mesh.faces.flip(1))
    swollen = Mesh(scene_mesh.vertices * 1.15, scene_mesh.faces)
    for wrong in (moved, mirrored, swollen):
        assert max(meshing.silhouette_iou(wrong, held_out)) < 0.85

    without_alpha = Views(held_out.images, held_out.poses, held_out.focal, 2.0, 6.0, True)
    with pytest.raises(ValueError, match="alpha"):
        meshing.silhouette_iou(scene_mesh, without_alpha)


def test_what_a_camera_sees_of_the_scene_mesh_lies_on_the_spheres(scene_mesh, held_out):
    # The point seen at each pixel, from the depth the rasteriser reports, is
    # on the true surface as nearly as the mesh is: the density is a step, so
    # the mesh is only known to half a grid spacing, too deep as often as too
    # shallow.
    for pose in held_out.poses:
        origins, directions = get_rays(96, 96, held_out.focal, pose)
        raster = meshing.rasterise(scene_mesh, 96, 96, held_out.focal, pose)
        seen = raster.face >= 0
        assert seen.sum() > 2000
        points = origins[seen] + raster.depth[seen][:, None] * directions[seen]
        distance = SphereScene().distance_to_surface(points)
        assert distance.abs().max() <= SCENE_VOXEL / 2 + 1e-4
        assert abs(distance.mean().item()) < SCENE_VOXEL / 10


def test_pictures_of_the_scene_mesh(scene_mesh, held_out):
    coloured = meshing.colour_vertices(SphereScene(), scene_mesh, SCENE_VOXEL)
    raster = meshing.rasterise(coloured, 96, 96, held_out.focal, held_out.poses[0])
    covered = raster.face >= 0

    in_colour = meshing.colour_picture(coloured, raster)
    assert in_colour.shape == (96, 96, 3)
    assert (in_colour[~covered] == 1).all()                              # white background
    # where both show the object, the picture is the renderer's image
    solid = covered & (held_out.alpha[0] > 0.999)
    assert solid.sum() > 1000
    assert (in_colour[solid] - held_out.images[0][solid]).abs().mean() < 0.02

    shape = meshing.shape_picture(coloured, raster, held_out.poses[0])
    assert (shape[~covered] == 1).all()
    grey = shape[covered]
    assert torch.equal(grey[:, 0], grey[:, 1]) and torch.equal(grey[:, 1], grey[:, 2])
    assert grey.min() >= 0.8 * 0.3 - 1e-6 and grey.max() <= 0.8 + 1e-6


def test_a_ball_is_brightest_where_it_faces_the_camera(smooth_ball):
    surface, centre, radius = smooth_ball
    pose = camera_on_the_z_axis(5.0)
    pose[:3, 3] += centre                                                  # straight above the ball
    raster = meshing.rasterise(surface, 81, 81, 300.0, pose)
    shape = meshing.shape_picture(surface, raster, pose)[..., 0]
    covered = raster.face >= 0
    assert shape[40, 40] > 0.79                                            # full brightness is 0.8
    edge = covered & ~torch.roll(covered, 1, 1)                            # leftmost covered pixel of each row
    assert shape[edge].max() < 0.5                                         # dim towards the rim


def views_of(field, num_views=4, image_size=64):
    """Views of any field from the renderer, like `analytic_views` makes for the sphere scene."""
    poses = orbit_poses(num_views)
    focal = 0.5 * image_size / math.tan(0.5 * 0.6911)
    alphas = []
    for pose in poses:
        origins, directions = get_rays(image_size, image_size, focal, pose)
        out = render_rays(field, None, origins.reshape(-1, 3), directions.reshape(-1, 3), 2.0, 6.0,
                          num_coarse=128, num_fine=0, perturb=False, white_background=True)
        alphas.append(out.coarse.acc.reshape(image_size, image_size))
    alpha = torch.stack(alphas)
    return Views(1 - alpha[..., None].expand(-1, -1, -1, 3), poses, focal, 2.0, 6.0, True, alpha)


def test_threshold_is_fitted_to_the_silhouettes():
    # A density that fades smoothly from 60 at the origin: its level surface
    # at a value c is a sphere of radius 0.8 sqrt(ln(60 / c)), which is 1.07
    # for c = 10, 0.84 for 20 and 0.34 for 50. The photographs show a ball of
    # radius 0.85, so 20 is the threshold that reproduces them.
    def soft(points, view_dirs):
        return 60.0 * torch.exp(-(points.norm(dim=-1) / 0.8) ** 2), torch.zeros_like(points)

    density = meshing.density_grid(soft, bound=1.6, resolution=65)
    views = views_of(hard_ball(radius=0.85))
    reported = []
    best, scores = meshing.fit_threshold(density, 1.6, views, report=lambda *pair: reported.append(pair))
    assert best == 20.0
    # every candidate, from 1 up to the authors' 50, is below the density's 60
    assert list(scores) == [1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0, 15.0, 20.0, 30.0, 40.0, 50.0]
    assert reported == list(scores.items())                               # each as soon as it is known
    assert scores[20.0] > 0.95
    values = list(scores.values())
    assert values[:10] == sorted(values[:10]) and values[9:] == sorted(values[9:], reverse=True)
    assert len(set(values)) == 13                                         # rising all the way to 20, falling after

    # restricted to other candidates, the best of those
    assert meshing.fit_threshold(density, 1.6, views, candidates=(5.0, 50.0))[0] == 5.0
    with pytest.raises(ValueError, match="ranges from"):
        meshing.fit_threshold(density, 1.6, views, candidates=(100.0, 200.0))

    # The choice among candidates that fit equally well. With a tolerance as
    # wide as 0.4 those are 10, 15, 20 and 30: the levels with radii from 1.07
    # down to 0.67. 7 (radius 1.17) and 40 (0.51) are further off, and 50 more.
    assert scores[10.0] > scores[20.0] - 0.36 and scores[30.0] > scores[20.0] - 0.38
    assert scores[7.0] < scores[20.0] - 0.42 and scores[40.0] < scores[20.0] - 0.42
    assert meshing.fit_threshold(density, 1.6, views, tolerance=0.4)[0] == 10.0                   # the lowest of them
    assert meshing.fit_threshold(density, 1.6, views, tolerance=0.4, preferred=30.0)[0] == 30.0   # or the preferred one
    assert meshing.fit_threshold(density, 1.6, views, tolerance=0.4, preferred=40.0)[0] == 10.0   # if it is among them
    assert meshing.fit_threshold(density, 1.6, views, tolerance=0.0, preferred=30.0)[0] == 20.0   # no tolerance: the best

    # A candidate's score is its mean over the views. Against photographs of
    # a ball that sits 0.3 to one side, the views differ: from the side the
    # shift shows in the outline, from straight ahead hardly at all.
    shifted = views_of(hard_ball(centre=(0.3, 0.0, 0.0), radius=0.85))
    per_view = meshing.silhouette_iou(meshing.extract_surface(density, 1.6, 20.0), shifted)
    assert max(per_view) - min(per_view) > 0.1
    _, scores = meshing.fit_threshold(density, 1.6, shifted, candidates=(20.0,))
    assert scores[20.0] == pytest.approx(sum(per_view) / 4, abs=1e-12)
    assert abs(scores[20.0] - per_view[0]) > 0.01


def test_the_authors_threshold_is_kept_where_the_outlines_have_nothing_against_it():
    # A ball with a hard surface, density 0 outside and 400 inside. Every
    # candidate from 1 to 50 puts the surface in the same grid cell, close to
    # its outer end, and their outlines differ by little. All of them fit
    # equally well, and of those the authors' 50 is kept.
    def hard(points, view_dirs):
        return 400.0 * (points.norm(dim=-1) < 0.85), torch.zeros_like(points)

    density = meshing.density_grid(hard, 1.6, 129)
    views = views_of(hard_ball(radius=0.85))
    chosen, scores = meshing.fit_threshold(density, 1.6, views)
    assert len(scores) == 13 and max(scores.values()) - min(scores.values()) < 0.008
    assert chosen == 50.0
    # with no preference it would be the lowest of them
    assert meshing.fit_threshold(density, 1.6, views, preferred=None)[0] == 1.0


def test_threshold_is_fitted_to_the_mesh_without_its_specks():
    def soft(points, view_dirs):
        return 60.0 * torch.exp(-(points.norm(dim=-1) / 0.8) ** 2), torch.zeros_like(points)

    def with_speck(points, view_dirs):
        speck = hard_ball(centre=(0.0, 0.0, 1.5), radius=0.06)(points, view_dirs)[0]   # above the ball
        return torch.maximum(soft(points, view_dirs)[0], speck), torch.zeros_like(points)

    views = views_of(hard_ball(radius=0.85))
    candidates = (10.0, 20.0, 30.0)
    clean = meshing.fit_threshold(meshing.density_grid(soft, 1.6, 65), 1.6, views, candidates)[1]
    density = meshing.density_grid(with_speck, 1.6, 65)
    # the speck is dropped before the outline is compared, as it will be from the final mesh
    assert meshing.fit_threshold(density, 1.6, views, candidates)[1] == clean
    # unless it is asked to stay, and then it shows in every score
    kept = meshing.fit_threshold(density, 1.6, views, candidates, min_fraction=0.0)[1]
    assert all(kept[candidate] < clean[candidate] - 0.0005 for candidate in candidates)


# --------------------------------------------------------------------------
# The script. Its networks are replaced by fields whose geometry is known,
# so that what it writes can be checked against exact answers.
# --------------------------------------------------------------------------


class Exactly(torch.nn.Module):
    """A formula standing where a trained network would."""

    def __init__(self, field):
        super().__init__()
        self.field = field

    def forward(self, points, view_dirs):
        return self.field(points, view_dirs)


@pytest.fixture(scope="module")
def scene(tmp_path_factory):
    directory = tmp_path_factory.mktemp("mesh_data") / "lego"
    write_blender_scene(directory, analytic_views(8, image_size=64, num_samples=64), "train")
    write_blender_scene(directory, analytic_views(2, image_size=64, num_samples=64, phase=40.0), "val")
    write_blender_scene(directory, analytic_views(6, image_size=64, num_samples=64, phase=80.0), "test")
    return directory


def networks_of(fine):
    """What the script's loader returns, with `fine` as the fine network's formula.
    The coarse one is a single ball, so a mesh made from it would be known by its one piece."""
    return lambda path, device: TrainedNetworks(Exactly(hard_ball()), Exactly(fine), TrainConfig(), 123)


@pytest.fixture
def extractor(monkeypatch, tmp_path):
    """The script as a module, reading the exact sphere scene from any checkpoint."""
    spec = importlib.util.spec_from_file_location("mesh_cli", ROOT / "scripts" / "05_extract_mesh.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    monkeypatch.setattr(module, "load_networks", networks_of(SphereScene()))
    module.checkpoint = tmp_path / "run" / "checkpoint.pt"     # a run trained on full-size photographs
    module.checkpoint.parent.mkdir()
    module.checkpoint.write_bytes(b"")
    (module.checkpoint.parent / "config.json").write_text(json.dumps({"downscale": 1}))
    return module


def extract(extractor, scene, *options):
    return extractor.main([str(scene), str(extractor.checkpoint), "--device", "cpu",
                           "--bound", "1.6", "--resolution", "97", *options])


def note_what_is_loaded(extractor, monkeypatch):
    """A list that fills with (split, downscale, positions) for each time the script reads photographs."""
    loaded = []
    real_load_blender = extractor.load_blender

    def noting(scene_dir, split, downscale, indices):
        loaded.append((split, downscale, list(indices)))
        return real_load_blender(scene_dir, split, downscale, indices=indices)

    monkeypatch.setattr(extractor, "load_blender", noting)
    return loaded


def photograph(scene, split, index):
    """A photograph of the scene as an (H, W, 3) array in [0, 1], over white."""
    rgba = np.asarray(Image.open(scene / split / f"r_{index}.png").convert("RGBA")).astype(np.float32) / 255
    return rgba[..., :3] * rgba[..., 3:] + (1 - rgba[..., 3:])


def tiles_of(preview, across, down, size=400):
    """The picture in column `across` and row `down` of a preview, as an array."""
    pitch = size + 4
    return np.asarray(preview)[4 + pitch * down : pitch * (down + 1), 4 + pitch * across : pitch * (across + 1)]


def test_script_writes_a_mesh_of_the_scene_with_its_measures(extractor, scene, monkeypatch, capsys):
    loaded = note_what_is_loaded(extractor, monkeypatch)
    assert extract(extractor, scene, "--skip", "2") == 0
    # A first look at the photographs before the work starts. Then the
    # threshold is fitted to training views, all eight here, and the mesh is
    # looked at through test views only, one at a time. All at the size the
    # run was trained on.
    assert loaded == [("test", 1, [0]), ("train", 1, [0, 1, 2, 3, 4, 5, 6, 7]),
                      ("test", 1, [0]), ("test", 1, [2]), ("test", 1, [4])]
    results = extractor.checkpoint.parent / "mesh_000123"          # named after the step
    summary = json.loads((results / "summary.json").read_text())

    # The mesh is taken from the fine network. (The coarse one here is a
    # single ball, which would give one piece.)
    assert summary["pieces"] == 4 and summary["small_pieces_removed"] == 0
    assert summary["closed_surface"] is True and summary["pocket_points_filled"] == 0
    assert summary["box"]["x"] == pytest.approx([-0.7, 1.45], abs=SCENE_VOXEL)
    assert summary["box"]["y"] == pytest.approx([-0.7, 1.45], abs=SCENE_VOXEL)
    assert summary["box"]["z"] == pytest.approx([-0.7, 1.2], abs=SCENE_VOXEL)
    assert (summary["resolution"], summary["bound"], summary["step"]) == (97, 1.6, 123)
    assert summary["density_max"] == 40.0

    # The threshold was fitted to training views, among the candidates that
    # the density (0 or 40) crosses. Whichever is chosen, it moves the surface
    # by less than a grid cell, and the volume by a few per cent.
    reachable = ["1", "1.5", "2", "3", "4", "5", "7", "10", "15", "20", "30"]
    assert summary["threshold_fitted"] is True and summary["fit_views"] == 8
    assert list(summary["fit_silhouette_iou"]) == reachable
    fit = summary["fit_silhouette_iou"]
    assert fit["30"] == max(fit.values()) > fit["20"] + 0.012        # on this coarse grid, 30 alone is within 0.01 of the best
    assert summary["threshold"] == 30.0
    assert summary["volume"] == pytest.approx(SCENE_VOLUME, rel=0.08)
    printed = capsys.readouterr().out
    assert "fitting the threshold to the object's outline in 8 training views\n" in printed
    lines = re.findall(r"^    threshold +(\S+)   IoU (\S+)$", printed, flags=re.MULTILINE)
    assert [candidate for candidate, _ in lines] == reachable
    for candidate, score in lines:
        assert float(score) == pytest.approx(summary["fit_silhouette_iou"][candidate], abs=6e-4)
    assert "    -> 30, the lowest within 0.01 of the best\n" in printed
    assert "; 0 grid points in sealed pockets were filled\n" in printed
    assert f"closed surface: yes   volume {summary['volume']:.4f}   area {summary['area']:.4f}   x " in printed

    # The score is from every second test view, at the photographs' size.
    assert (summary["views"], summary["views_in_split"], summary["split"], summary["skip"]) == (3, 6, "test", 2)
    assert (summary["downscale"], summary["width"], summary["height"]) == (1, 64, 64)
    per_view = summary["silhouette_iou_per_view"]
    assert list(per_view) == ["0", "2", "4"]
    assert summary["silhouette_iou_mean"] == pytest.approx(sum(per_view.values()) / 3, abs=1e-4)
    assert summary["silhouette_iou_worst"] == min(per_view.values()) > 0.93

    # The file holds that mesh, in the spheres' colours.
    header, vertices, faces = read_ply(results / "mesh.ply")
    assert (len(vertices), len(faces)) == (summary["vertices"], summary["triangles"])
    written = Mesh(torch.from_numpy(vertices["xyz"].copy()), torch.from_numpy(faces["corners"].astype(np.int64)))
    assert meshing.is_closed(written)
    assert meshing.enclosed_volume(written) == pytest.approx(summary["volume"], abs=1e-4)
    assert meshing.surface_area(written) == pytest.approx(summary["area"], abs=1e-4)
    assert to_true_surfaces(written.vertices).min(dim=1).values.max() <= SCENE_VOXEL
    colours = {tuple(colour) for colour in vertices["rgb"].tolist()}
    for _, _, colour in SphereScene.SPHERES:
        assert tuple(round(255 * channel) for channel in colour) in colours

    # The preview: three test views across; photograph, colours and shape down.
    preview = Image.open(results / "preview.png")
    assert preview.size == (3 * 404 + 4, 3 * 404 + 4)
    top, middle, bottom = (tiles_of(preview, 0, row) for row in range(3))

    def at_photo_size(tile):
        return np.asarray(Image.fromarray(tile).resize((64, 64), Image.BOX)).astype(np.float32) / 255

    photo = photograph(scene, "test", 0)
    assert np.abs(at_photo_size(top) - photo).mean() < 0.01                 # the photograph itself
    on_object = photo.min(axis=-1) < 0.9
    assert np.abs(at_photo_size(middle) - photo)[on_object].mean() < 0.05   # the mesh in the same colours
    assert (bottom[..., 0] == bottom[..., 1]).all() and (bottom[..., 1] == bottom[..., 2]).all()   # grey
    assert bottom.min() < 200 and (bottom == 255).mean() > 0.3              # a shaded shape on white


def test_the_authors_threshold_is_named_as_such_when_it_is_the_one_chosen(extractor, scene, tmp_path, capsys):
    extractor.load_networks = networks_of(SphereScene(density=400.0))       # the same spheres, ten times as dense
    assert extract(extractor, scene, "--skip", "3", "--out", str(tmp_path / "dense")) == 0
    summary = json.loads((tmp_path / "dense" / "summary.json").read_text())
    fit = summary["fit_silhouette_iou"]
    # the outline score creeps up with the threshold; 30 and 40 are within 0.01 of 50, and lower than it
    assert len(fit) == 13 and fit["50"] == max(fit.values()) and fit["50"] - 0.008 < fit["30"] < fit["40"] < fit["50"]
    assert summary["threshold"] == 50.0 and summary["threshold_fitted"] is True
    assert "    -> 50, the authors' value within 0.01 of the best\n" in capsys.readouterr().out


def test_threshold_given_by_hand_is_used_and_nothing_is_fitted(extractor, scene, tmp_path):
    only_test = tmp_path / "lego"                                           # a scene without training views
    write_blender_scene(only_test, analytic_views(4, image_size=64, num_samples=64, phase=80.0), "test")
    out = tmp_path / "by_hand"
    assert extract(extractor, only_test, "--threshold", "20", "--skip", "2", "--out", str(out)) == 0
    summary = json.loads((out / "summary.json").read_text())
    assert summary["threshold"] == 20.0 and summary["threshold_fitted"] is False
    assert summary["fit_silhouette_iou"] == {} and summary["fit_views"] == 0
    assert summary["volume"] == pytest.approx(SCENE_VOLUME, rel=0.01)
    inside = (meshing.density_grid(SphereScene(), SCENE_BOUND, SCENE_RESOLUTION) > 20).float().mean().item()
    assert summary["fraction_above_threshold"] == pytest.approx(inside, abs=1e-6)
    assert inside == pytest.approx(SCENE_VOLUME / 3.2**3, rel=0.05)         # the spheres' share of the grid
    assert (summary["downscale"], summary["width"], summary["height"]) == (1, 64, 64)

    # the photographs can be shrunk before the silhouettes are compared
    assert extract(extractor, only_test, "--threshold", "20", "--downscale", "2", "--out", str(out)) == 0
    smaller = json.loads((out / "summary.json").read_text())
    assert (smaller["downscale"], smaller["width"], smaller["height"], smaller["views"]) == (2, 32, 32, 1)
    assert smaller["volume"] == summary["volume"]                          # the mesh itself is the same


def test_photographs_are_used_at_the_size_the_run_was_trained_on(extractor, scene, monkeypatch, capsys):
    run = extractor.checkpoint.parent
    (run / "config.json").write_text(json.dumps({"downscale": 2, "iterations": 6}))
    loaded = note_what_is_loaded(extractor, monkeypatch)
    assert extract(extractor, scene, "--skip", "4") == 0
    assert loaded == [("test", 2, [0]), ("train", 2, [0, 1, 2, 3, 4, 5, 6, 7]), ("test", 2, [0]), ("test", 2, [4])]
    summary = json.loads((run / "mesh_000123" / "summary.json").read_text())
    assert (summary["downscale"], summary["width"], summary["height"]) == (2, 32, 32)
    assert "silhouette against 2 of 6 test views at 32 x 32" in capsys.readouterr().out

    # A run that does not say needs to be told, and is asked before any work is done.
    extractor.load_networks = never
    for unusable in (None, "not json", '{"iterations": 6}', '{"downscale": 0}', '{"downscale": "2"}',
                     '{"downscale": true}', "[2]"):
        if unusable is None:
            (run / "config.json").unlink()
        else:
            (run / "config.json").write_text(unusable)
        assert extract(extractor, scene) == 2
        printed = capsys.readouterr().out
        assert printed.startswith(f"There is no config.json next to {extractor.checkpoint} that says")
        assert "--downscale" in printed
    extractor.load_networks = networks_of(SphereScene())
    assert extract(extractor, scene, "--threshold", "20", "--downscale", "1") == 0


def test_the_views_asked_for_are_the_ones_used(extractor, scene, monkeypatch):
    loaded = note_what_is_loaded(extractor, monkeypatch)
    # three training views spread over the eight, to score against the validation views, at half size
    assert extract(extractor, scene, "--fit-views", "3", "--split", "val", "--skip", "1", "--downscale", "2") == 0
    assert loaded == [("val", 2, [0]), ("train", 2, [0, 4, 7]), ("val", 2, [0]), ("val", 2, [1])]
    summary = json.loads((extractor.checkpoint.parent / "mesh_000123" / "summary.json").read_text())
    assert (summary["split"], summary["views"], summary["views_in_split"], summary["fit_views"]) == ("val", 2, 2, 3)
    assert list(summary["silhouette_iou_per_view"]) == ["0", "1"]

    # asking for more training views than there are uses each one once
    del loaded[:]
    assert extract(extractor, scene, "--fit-views", "20") == 0
    assert loaded[1] == ("train", 1, [0, 1, 2, 3, 4, 5, 6, 7])
    summary = json.loads((extractor.checkpoint.parent / "mesh_000123" / "summary.json").read_text())
    assert summary["fit_views"] == 8


def test_preview_shows_four_views_spread_over_those_scored(extractor, scene, tmp_path):
    assert extract(extractor, scene, "--threshold", "20", "--skip", "1", "--out", str(tmp_path / "six")) == 0
    preview = Image.open(tmp_path / "six" / "preview.png")
    assert preview.size == (4 * 404 + 4, 3 * 404 + 4)
    # of the six test views: the first, the last and two between
    for across, index in enumerate((0, 2, 3, 5)):
        shown = Image.fromarray(tiles_of(preview, across, 0)).resize((64, 64), Image.BOX)
        assert np.abs(np.asarray(shown).astype(np.float32) / 255 - photograph(scene, "test", index)).mean() < 0.01
    assert np.abs(photograph(scene, "test", 3) - photograph(scene, "test", 5)).mean() > 0.05   # the views differ

    # A preview smaller than the photographs averages their pixels. Here by
    # two: each pixel of the picture is the mean of four of the photograph.
    extractor.PREVIEW_WIDTH = 32
    assert extract(extractor, scene, "--threshold", "20", "--skip", "1", "--out", str(tmp_path / "small")) == 0
    preview = Image.open(tmp_path / "small" / "preview.png")
    assert preview.size == (4 * 36 + 4, 3 * 36 + 4)
    photo = (photograph(scene, "test", 5) * 255).round()
    averaged = photo.reshape(32, 2, 32, 2, 3).mean(axis=(1, 3))
    assert np.abs(tiles_of(preview, 3, 0, size=32) - averaged).max() <= 1
    assert np.abs(photo[::2, ::2] - averaged).max() > 30                    # taking every other pixel is not that


def test_a_floating_speck_is_removed_unless_asked_to_keep_it(extractor, scene, tmp_path):
    spheres, speck = SphereScene(), hard_ball(centre=(-1.2, -1.2, 1.2), radius=0.05)

    def with_speck(points, view_dirs):
        sigma, rgb = spheres(points, view_dirs)
        return torch.maximum(sigma, speck(points, view_dirs)[0]), rgb

    extractor.load_networks = networks_of(with_speck)
    assert extract(extractor, scene, "--threshold", "20", "--out", str(tmp_path / "cleaned")) == 0
    cleaned = json.loads((tmp_path / "cleaned" / "summary.json").read_text())
    assert (cleaned["pieces"], cleaned["small_pieces_removed"]) == (4, 1)
    assert cleaned["box"]["x"][0] == pytest.approx(-0.7, abs=SCENE_VOXEL)   # the speck at x = -1.2 is gone
    # the measures are those of the mesh that was written, without the speck
    header, vertices, faces = read_ply(tmp_path / "cleaned" / "mesh.ply")
    written = Mesh(torch.from_numpy(vertices["xyz"].copy()), torch.from_numpy(faces["corners"].astype(np.int64)))
    assert meshing.enclosed_volume(written) == pytest.approx(cleaned["volume"], abs=1e-5)
    assert meshing.surface_area(written) == pytest.approx(cleaned["area"], abs=1e-5)

    assert extract(extractor, scene, "--threshold", "20", "--min-piece", "0", "--out", str(tmp_path / "kept")) == 0
    kept = json.loads((tmp_path / "kept" / "summary.json").read_text())
    assert (kept["pieces"], kept["small_pieces_removed"]) == (5, 0)
    assert kept["box"]["x"][0] < -1.2 + 1e-6
    assert kept["silhouette_iou_mean"] < cleaned["silhouette_iou_mean"]     # and it shows in the score
    assert kept["volume"] > cleaned["volume"] + 1e-4 and kept["area"] > cleaned["area"] + 1e-3

    # The same holds while the threshold is fitted: the speck spoils every
    # candidate's outline if it is kept, and none if it is dropped.
    extractor.load_networks = networks_of(spheres)
    assert extract(extractor, scene, "--out", str(tmp_path / "plain")) == 0
    plain = json.loads((tmp_path / "plain" / "summary.json").read_text())["fit_silhouette_iou"]
    extractor.load_networks = networks_of(with_speck)
    assert extract(extractor, scene, "--out", str(tmp_path / "fit_cleaned")) == 0
    assert json.loads((tmp_path / "fit_cleaned" / "summary.json").read_text())["fit_silhouette_iou"] == plain
    assert extract(extractor, scene, "--min-piece", "0", "--out", str(tmp_path / "fit_kept")) == 0
    spoiled = json.loads((tmp_path / "fit_kept" / "summary.json").read_text())["fit_silhouette_iou"]
    assert all(spoiled[candidate] < plain[candidate] for candidate in plain)


def test_a_hollow_object_gets_no_inner_wall(extractor, scene, tmp_path, capsys):
    extractor.load_networks = networks_of(hollow_ball(outer=0.9, inner=0.4))
    assert extract(extractor, scene, "--threshold", "20", "--out", str(tmp_path / "hollow")) == 0
    summary = json.loads((tmp_path / "hollow" / "summary.json").read_text())
    axis = torch.linspace(-SCENE_BOUND, SCENE_BOUND, SCENE_RESOLUTION)
    distance = torch.stack(torch.meshgrid(axis, axis, axis, indexing="ij"), dim=-1).norm(dim=-1)
    in_hollow = int((distance < 0.4).sum())
    assert summary["pocket_points_filled"] == in_hollow > 5000
    assert f"; {in_hollow:,} grid points in sealed pockets were filled\n" in capsys.readouterr().out
    assert summary["pieces"] == 1 and summary["closed_surface"] is True
    assert summary["volume"] == pytest.approx(4 / 3 * math.pi * 0.9**3, rel=0.01)       # the whole ball
    # what is reported as above the threshold is the network's density as it is: the shell alone
    in_shell = ((distance >= 0.4) & (distance < 0.9)).float().mean().item()
    assert summary["fraction_above_threshold"] == pytest.approx(in_shell, abs=1e-6)


def test_a_grid_too_small_for_the_object_is_reported_as_an_open_surface(extractor, scene, tmp_path, capsys):
    # Two of the spheres reach to 1.45; a grid that ends at 1.2 cuts them open.
    arguments = [str(scene), str(extractor.checkpoint), "--device", "cpu", "--resolution", "73", "--threshold", "20"]
    assert extractor.main([*arguments, "--out", str(tmp_path / "cut")]) == 0
    summary = json.loads((tmp_path / "cut" / "summary.json").read_text())
    assert summary["bound"] == 1.2 and summary["closed_surface"] is False
    assert summary["box"]["x"][1] == pytest.approx(1.2, abs=1e-4)
    # an open surface encloses nothing, and no volume is given for it
    assert summary["volume"] is None and summary["area"] > 5
    assert "closed surface: no, so no volume   area " in capsys.readouterr().out

    # An object that fills the grid out to every face has no outside to show,
    # only what is sealed inside it.
    extractor.load_networks = networks_of(
        lambda points, dirs: (torch.where(points.norm(dim=-1) < 0.4, 0.0, 40.0), points))
    assert extractor.main([*arguments, "--out", str(tmp_path / "buried")]) == 2
    printed = capsys.readouterr().out.splitlines()
    assert printed[-2].startswith("At a threshold of 20 the object fills the grid out to every face")
    assert "--bound" in printed[-1] and not (tmp_path / "buried").exists()


def test_problems_are_reported_without_writing_anything(extractor, scene, tmp_path, capsys):
    out = tmp_path / "nothing"
    # no density that high anywhere
    assert extract(extractor, scene, "--threshold", "50", "--out", str(out)) == 2
    printed = capsys.readouterr().out
    assert "There is no surface at threshold 50: the density ranges from 0 to 40." in printed
    assert "--threshold" in printed.splitlines()[-1]
    # nor at any candidate, for a field that is nearly empty
    extractor.load_networks = networks_of(lambda points, dirs: (0.5 * SphereScene()(points, dirs)[0] / 40, points))
    assert extract(extractor, scene, "--out", str(out)) == 2
    assert ("There is no surface at any of the thresholds from 1 to 50: "
            "the density ranges from 0 to 0.5.") in capsys.readouterr().out
    # a network whose training went wrong
    for broken in (float("nan"), float("inf")):
        extractor.load_networks = networks_of(
            lambda points, dirs: (torch.where(points[..., 0] > 1.5, broken, SphereScene()(points, dirs)[0]), points))
        assert extract(extractor, scene, "--out", str(out)) == 2
        printed = capsys.readouterr().out.splitlines()
        assert printed[-2] == "The density is not a finite number at 28,227 of the 912,673 grid points."   # 3 of 97 layers
        assert "no surface to extract" in printed[-1]
    # no such checkpoint
    assert extractor.main([str(scene), str(tmp_path / "missing.pt")]) == 2
    assert "There is no checkpoint at" in capsys.readouterr().out
    assert not out.exists()


def never(*args, **kwargs):
    raise AssertionError("this should not have been reached")


def test_a_wrong_scene_directory_is_reported_before_the_network_is_loaded(extractor, tmp_path, capsys):
    extractor.load_networks = never
    assert extractor.main([str(tmp_path / "nowhere"), str(extractor.checkpoint)]) == 2
    printed = capsys.readouterr().out.splitlines()
    assert printed[0] == f"There is no transforms_test.json or transforms_train.json in {tmp_path / 'nowhere'}."
    assert "data/nerf_synthetic/lego" in printed[1]

    # test views only: enough to score a mesh, not to fit the threshold
    only_test = tmp_path / "lego"
    write_blender_scene(only_test, analytic_views(2, image_size=32, num_samples=32), "test")
    assert extractor.main([str(only_test), str(extractor.checkpoint)]) == 2
    assert capsys.readouterr().out.startswith(f"There is no transforms_train.json in {only_test}.")
    assert extractor.main([str(only_test), str(extractor.checkpoint), "--split", "val", "--threshold", "20"]) == 2
    assert capsys.readouterr().out.startswith(f"There is no transforms_val.json in {only_test}.")

    # a split that lists no views
    (only_test / "transforms_val.json").write_text(json.dumps({"camera_angle_x": 0.6911, "frames": []}))
    assert extractor.main([str(only_test), str(extractor.checkpoint), "--split", "val", "--threshold", "20"]) == 2
    assert capsys.readouterr().out == f"transforms_val.json in {only_test} lists no views.\n"


def test_missing_photographs_are_reported_before_the_slow_part(extractor, scene, tmp_path, capsys):
    extractor.density_grid = never
    incomplete = Path(shutil.copytree(scene, tmp_path / "lego"))
    (incomplete / "test" / "r_2.png").unlink()
    (incomplete / "test" / "r_4.png").unlink()
    (incomplete / "train" / "r_4.png").unlink()
    # every second test view is wanted: 0, 2 and 4
    assert extract(extractor, incomplete, "--threshold", "20", "--skip", "2") == 2
    assert capsys.readouterr().out == (
        f"Photographs that transforms_test.json lists are missing: 2 of them, the first {incomplete}/test/r_2.png\n")
    # training views 0, 4 and 7 are wanted for the fit
    assert extract(extractor, incomplete, "--fit-views", "3", "--skip", "5") == 2
    assert capsys.readouterr().out == (
        f"Photographs that transforms_train.json lists are missing: 1 of them, the first {incomplete}/train/r_4.png\n")
    # photographs that are not wanted are not missed: test views 0 and 5, training views 0 and 7
    with pytest.raises(AssertionError, match="should not have been reached"):
        extract(extractor, incomplete, "--fit-views", "2", "--skip", "5")
    capsys.readouterr()
    # a training split that lists nothing cannot be fitted to
    (incomplete / "transforms_train.json").write_text(json.dumps({"camera_angle_x": 0.6911, "frames": []}))
    assert extract(extractor, incomplete, "--skip", "5") == 2
    assert capsys.readouterr().out == f"transforms_train.json in {incomplete} lists no views.\n"


def test_results_are_not_written_into_the_directory_of_a_training_run(extractor, scene, tmp_path, capsys):
    extractor.density_grid = never                       # refused before the slow part
    # the directory the checkpoint itself is in, here one kept from step 123
    run = tmp_path / "lego_run"
    run.mkdir()
    kept = run / "checkpoint_000123.pt"
    kept.write_bytes(b"")
    arguments = [str(scene), str(kept), "--device", "cpu", "--downscale", "1"]
    assert extractor.main([*arguments, "--out", str(run)]) == 2
    assert capsys.readouterr().out.startswith(f"{run} is the directory of a training run.")
    # or the directory of any other run
    other_run = extractor.checkpoint.parent
    assert extractor.main([*arguments, "--out", str(other_run)]) == 2
    assert "is the directory of a training run" in capsys.readouterr().out
    assert [path.name for path in run.iterdir()] == ["checkpoint_000123.pt"]
    assert sorted(path.name for path in other_run.iterdir()) == ["checkpoint.pt", "config.json"]

    # nor into a directory that holds something else, such as the scores of the same checkpoint
    scores = tmp_path / "eval_000123_test"
    scores.mkdir()
    (scores / "summary.json").write_text("{}")
    assert extractor.main([*arguments, "--out", str(scores)]) == 2
    assert capsys.readouterr().out.startswith(f"{scores} is in use for something else.")
    assert (scores / "summary.json").read_text() == "{}"
    notes = tmp_path / "notes.txt"
    notes.write_text("not a directory")
    assert extractor.main([*arguments, "--out", str(notes)]) == 2
    assert capsys.readouterr().out.startswith(f"{notes} is in use for something else.")
    assert extractor.main([*arguments, "--out", str(notes / "meshes")]) == 2             # nor below a file
    assert capsys.readouterr().out.startswith(f"{notes / 'meshes'} is in use for something else.")
    # an empty directory will do, also with what a file browser leaves in it:
    # the script goes on to the slow part
    (tmp_path / "empty").mkdir()
    (tmp_path / "browsed").mkdir()
    (tmp_path / "browsed" / ".DS_Store").write_bytes(b"")
    for unused in ("empty", "browsed"):
        with pytest.raises(AssertionError, match="should not have been reached"):
            extractor.main([*arguments, "--out", str(tmp_path / unused)])


def test_impossible_settings_are_refused_by_the_argument_parser(extractor, scene, capsys):
    extractor.load_networks = never
    for option, value in (("--resolution", "1"), ("--bound", "0"), ("--bound", "-1.2"), ("--bound", "inf"),
                          ("--skip", "0"), ("--fit-views", "0"), ("--downscale", "0"), ("--threshold", "high"),
                          ("--min-piece", "2"), ("--min-piece", "-0.1"), ("--min-piece", "nan")):
        with pytest.raises(SystemExit) as stopped:
            extractor.main([str(scene), str(extractor.checkpoint), option, value])
        assert stopped.value.code == 2
        assert f"argument {option}" in capsys.readouterr().err


def test_a_new_mesh_is_never_left_beside_the_numbers_of_an_earlier_one(extractor, scene, tmp_path):
    out = tmp_path / "reused"
    assert extract(extractor, scene, "--threshold", "20", "--out", str(out)) == 0
    assert sorted(path.name for path in out.iterdir()) == ["mesh.ply", "preview.png", "summary.json"]
    earlier = (out / "mesh.ply").read_bytes()

    def stops(*args, **kwargs):
        raise RuntimeError("stopped while the mesh was being scored")

    extractor.silhouette_iou = stops
    with pytest.raises(RuntimeError):
        extract(extractor, scene, "--threshold", "30", "--out", str(out))
    # the new mesh is there; the picture and the numbers of the old one are gone
    assert sorted(path.name for path in out.iterdir()) == ["mesh.ply"]
    assert (out / "mesh.ply").read_bytes() != earlier


def test_control_c_ends_the_script_with_one_line(extractor, scene, capsys):
    def interrupted(*args, **kwargs):
        raise KeyboardInterrupt

    extractor.density_grid = interrupted
    assert extract(extractor, scene) == 130
    assert capsys.readouterr().out.splitlines()[-1] == (
        "Stopped before the end. Run the same command again to start it over.")
    assert sorted(path.name for path in extractor.checkpoint.parent.iterdir()) == ["checkpoint.pt", "config.json"]


def test_script_reads_a_real_checkpoint(scene, tmp_path):
    # Without the stand-in: a network trained for six steps, whose density is
    # a shapeless haze. The numbers mean little; the path through the real
    # loader and network is what is exercised.
    run = tmp_path / "lego_tiny"
    train(scene, run, "--keep-every", "0", "--downscale", "2")
    spec = importlib.util.spec_from_file_location("mesh_cli_real", ROOT / "scripts" / "05_extract_mesh.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    networks = load_networks(run / "checkpoint.pt")
    density = meshing.density_grid(networks.fine, 1.2, 33)
    level = 0.5 * (density.median() + density.max()).item()
    arguments = [str(scene), str(run / "checkpoint.pt"), "--device", "cpu", "--resolution", "33", "--skip", "3"]
    assert module.main([*arguments, "--threshold", str(level)]) == 0
    summary = json.loads((run / "mesh_000006" / "summary.json").read_text())
    assert summary["step"] == 6 and summary["threshold"] == pytest.approx(level)
    assert (summary["downscale"], summary["width"]) == (2, 32)              # as the run's own config.json says
    assert summary["weights_sha256"] == weights_fingerprint(networks.coarse, networks.fine)
    assert summary["density_max"] == pytest.approx(density.max().item(), abs=1e-3)
    header, vertices, faces = read_ply(run / "mesh_000006" / "mesh.ply")
    assert (len(vertices), len(faces)) == (summary["vertices"], summary["triangles"])
    filled, _ = meshing.fill_cavities(density, level)
    expected = meshing.drop_small_pieces(meshing.extract_surface(filled, 1.2, level))[0]
    assert np.array_equal(vertices["xyz"], expected.vertices.numpy())


# --------------------------------------------------------------------------
# The smoke test's measures of a mesh, on meshes whose measures are known
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def smoke():
    spec = importlib.util.spec_from_file_location("smoke_test", ROOT / "scripts" / "00_smoke_test.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_smoke_test_measures_a_mesh_against_the_exact_spheres(smoke, scene_mesh, held_out):
    coloured = meshing.colour_vertices(SphereScene(), scene_mesh, SCENE_VOXEL)
    exact = smoke.against_the_scene(coloured, held_out)
    assert exact["outline_iou"] > 0.97 and len(exact["outline_iou_per_view"]) == 4
    assert exact["surface_distance"] < SCENE_VOXEL / 4 and abs(exact["surface_distance_signed"]) < SCENE_VOXEL / 10
    assert exact["volume_ratio"] == pytest.approx(1.0, abs=0.01)
    assert exact["colour_error"] < 0.01
    assert smoke.mesh_failures(exact) == []

    # The red sphere's part of the mesh alone, shrunk by a tenth towards its
    # centre: every point of it is then 0.07 inside the true surface, and it
    # holds 0.63^3 of the 0.444 that the four radii cubed add up to.
    piece = meshing.piece_of_each_face(coloured)              # the four spheres are four pieces

    def piece_at(point):
        """Whether each face belongs to the piece that has a vertex nearest to `point`."""
        first_corner = coloured.vertices[coloured.faces[:, 0]]
        return piece == piece[(first_corner - torch.tensor(point)).norm(dim=1).argmin()]

    red = Mesh(coloured.vertices * 0.9, coloured.faces[piece_at((-0.7, 0.0, 0.0))], coloured.colours)
    shrunk = smoke.against_the_scene(red, held_out)
    assert shrunk["surface_distance_signed"] == pytest.approx(-0.07, abs=0.003)
    assert shrunk["surface_distance"] == pytest.approx(0.07, abs=0.003)
    assert shrunk["volume_ratio"] == pytest.approx(0.63**3 / 0.444375, abs=0.01)
    assert "the mesh's surface is on average more than 0.04 from the true one" in smoke.mesh_failures(shrunk)
    # The whole mesh blown up by a tenth holds a third more than it should.
    swollen = smoke.against_the_scene(Mesh(coloured.vertices * 1.1, coloured.faces, coloured.colours), held_out)
    assert swollen["volume_ratio"] == pytest.approx(1.1**3, abs=0.02) and swollen["surface_distance_signed"] > 0.03
    assert "the mesh's volume is more than 20% from the spheres'" in smoke.mesh_failures(swollen)

    # Each requirement catches the damage it is there for.
    inside_out = smoke.against_the_scene(Mesh(coloured.vertices, coloured.faces.flip(1), coloured.colours), held_out)
    assert inside_out["volume_ratio"] == pytest.approx(-1.0, abs=0.01)
    assert smoke.mesh_failures(inside_out) == ["the mesh's volume is more than 20% from the spheres'"]
    without_the_top = Mesh(coloured.vertices, coloured.faces[~piece_at((0.0, 0.0, 1.2))], coloured.colours)   # no yellow sphere
    without_it = smoke.against_the_scene(without_the_top, held_out)
    assert without_it["outline_iou"] < 0.91
    assert smoke.mesh_failures(without_it) == ["the outline of the mesh overlaps the true one by less than 0.92"]
    repainted = smoke.against_the_scene(Mesh(coloured.vertices, coloured.faces, coloured.colours.flip(-1)), held_out)
    assert repainted["colour_error"] > 0.2
    assert smoke.mesh_failures(repainted) == ["the mesh's colours are on average more than 0.1 from the true views'"]
    opened = smoke.against_the_scene(Mesh(coloured.vertices, coloured.faces[1:], coloured.colours), held_out)
    assert opened["volume_ratio"] is None
    assert smoke.mesh_failures(opened) == ["the mesh is not a closed surface"]
    assert smoke.mesh_failures({"error": "no surface"}) == ["the trained density has no surface to make a mesh from"]


def test_smoke_test_meshes_a_network_and_compares_it_with_a_hull_and_the_exact_surface(smoke, tmp_path):
    # The exact scene stands where the trained network would be, so the
    # "network's" mesh is as good as the grid allows.
    train_views = analytic_views(24, image_size=48)
    result = smoke.measure_mesh(SphereScene(), train_views, "cpu", tmp_path)
    assert result["threshold"] in (15.0, 20.0, 30.0) and result["pieces"] == 4      # each within the cell the surface is in
    assert result["outline_iou"] > 0.97 and result["surface_distance"] < 0.012
    assert result["volume_ratio"] == pytest.approx(1.0, abs=0.06) and result["colour_error"] < 0.02
    assert smoke.mesh_failures(result) == []
    assert result["required"] == {"outline_iou": 0.92, "surface_distance": 0.04, "volume_error": 0.2, "colour_error": 0.1}

    exact = result["for_comparison"]["exact_density"]              # the density's own mid-level, 20
    assert exact["outline_iou"] == pytest.approx(0.983, abs=0.003)
    assert exact["surface_distance"] == pytest.approx(0.0035, abs=0.001)
    assert exact["volume_ratio"] == pytest.approx(1.0, abs=0.005) and "colour_error" not in exact
    hull = result["for_comparison"]["visual_hull_of_the_training_outlines"]
    assert hull["outline_iou"] == pytest.approx(0.962, abs=0.005)
    assert hull["surface_distance"] == pytest.approx(0.0156, abs=0.002)

    picture = Image.open(tmp_path / "smoke_test_mesh.png")
    assert picture.size == (4 * 258, 3 * 258)                      # four views across; view, colours, shape down


def test_smoke_test_closes_its_mesh_at_the_edge_of_the_grid_and_fills_its_pockets(smoke, tmp_path):
    # A lump of density lying half outside the grid would leave the surface
    # open where the grid cuts it, and an open surface has no volume. A hollow
    # inside the red sphere would give the mesh an inner wall.
    scene, lump = SphereScene(), hard_ball(centre=(-1.6, -1.0, -0.5), radius=0.25)

    def flawed(points, view_dirs):
        sigma, rgb = scene(points, view_dirs)
        sigma = torch.maximum(sigma, lump(points, view_dirs)[0])
        return torch.where(points.norm(dim=-1) < 0.3, 0.0, sigma), rgb

    as_it_is = meshing.extract_surface(meshing.density_grid(flawed, 1.6, 129), 1.6, 20.0)
    assert not meshing.is_closed(as_it_is) and meshing.piece_of_each_face(as_it_is).unique().numel() == 6
    result = smoke.measure_mesh(flawed, analytic_views(24, image_size=48), "cpu", tmp_path)
    assert result["pieces"] == 5                      # the four spheres and the lump, no inner wall
    assert result["volume_ratio"] == pytest.approx(1.0, abs=0.08)
    assert "the mesh is not a closed surface" not in smoke.mesh_failures(result)
