"""From a trained radiance field to a triangle mesh.

A NeRF stores a scene as density and colour at every point of space, not as a
surface, and most uses of a 3D model need a surface. Density does not depend
on the viewing direction, so the surface can be read off it directly:

  density_grid       sample the density on a regular grid
  fill_cavities      close the empty pockets sealed inside the object
  extract_surface    marching cubes: the surface where it crosses a threshold
  drop_small_pieces  remove specks of density that float in empty space
  colour_vertices    give each vertex the colour the field shows there
  save_ply           write the result as a .ply file

The authors' repository does the first and third steps in a notebook
(extract_mesh.ipynb): the fine network on 257 points per axis spanning
[-1.2, 1.2], and a threshold of 50. Those are the defaults here.

A NeRF is never told where surfaces are, so a mesh taken from one has to be
checked against something. The rest of this module does that without any
ground-truth 3D, using only the cameras and photographs the scene came with:

  rasterise          which triangle is seen at each pixel of a camera
  colour_picture     the mesh from that camera, in its vertex colours
  shape_picture      the same in plain grey with a light, to show the shape
  silhouette_iou     how well the outline of the mesh matches the object's
                     outline in each photograph
  fit_threshold      choose the density threshold by that measure

The last one exists because the threshold matters more than its place in the
notebook suggests. Where a model's density jumps from nothing to a large value
between two neighbouring grid points, the threshold only moves the surface
within that one grid cell. A model that is small or early in its training has
soft surfaces instead, and there the mesh swells or shrinks with the threshold
by many cells. `fit_threshold` and `silhouette_iou` say what this way of
choosing and checking can and cannot see.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import numpy as np
import torch

from nerf.data import Views
from nerf.rays import project_points
from nerf.render import volume_render


@dataclass
class Mesh:
    vertices: torch.Tensor                # (V, 3) positions in world coordinates
    faces: torch.Tensor                   # (F, 3) vertex indices, counter-clockwise seen from outside
    colours: torch.Tensor | None = None   # (V, 3) in [0, 1]


# Thresholds that `fit_threshold` chooses among. The highest, 50, is the
# authors' value: the fit may lower the threshold for a model whose surfaces
# are soft, and never raises it.
THRESHOLDS = (1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0, 15.0, 20.0, 30.0, 40.0, 50.0)


# --------------------------------------------------------------------------
# From a field to a surface
# --------------------------------------------------------------------------


@torch.no_grad()
def density_grid(
    field, bound: float = 1.2, resolution: int = 257, device: str = "cpu", chunk: int = 65536
) -> torch.Tensor:
    """Density of `field` on a cubic grid.

    Args:
        field: callable (points, view_dirs) -> (sigma, rgb), such as a trained
            network. Only sigma is used. It does not depend on the direction,
            so zeros are passed for it, as in the authors' notebook.
        bound: the grid spans [-bound, bound] on every axis.
        resolution: grid points per axis.
        device: where to evaluate the field. The grid points are laid out on
            the CPU and moved there a chunk at a time, so they are the same
            points on every device.
        chunk: points per call of the field.

    Returns:
        (resolution, resolution, resolution) densities on the CPU, indexed
        [x, y, z] like the grid of `nerf.hull.visual_hull`.
    """
    axis = torch.linspace(-bound, bound, resolution)
    total = resolution**3
    sigma = torch.empty(total)
    for start in range(0, total, chunk):
        flat = torch.arange(start, min(start + chunk, total))
        # the point at grid[i, j, k] has flat index (i * R + j) * R + k
        i = flat // (resolution * resolution)
        j = (flat // resolution) % resolution
        k = flat % resolution
        points = torch.stack([axis[i], axis[j], axis[k]], dim=-1).to(device)
        sigma[start : start + flat.numel()] = field(points, torch.zeros_like(points))[0].cpu()
    return sigma.reshape(resolution, resolution, resolution)


def fill_cavities(density: torch.Tensor, threshold: float) -> tuple[torch.Tensor, int]:
    """Fill the empty pockets that are sealed inside the object.

    The grid points not above `threshold` are empty space. Those that can be
    reached from a face of the grid, stepping between neighbours along the
    grid's axes, are the outside. Any others form pockets that the surface
    encloses on all sides. In a mesh each pocket is an inner wall that nothing
    outside can show, and where the enclosing wall is dense no camera saw into
    it either, so the density there is whatever training happened to leave.
    Their density is raised to the grid's maximum, which removes those walls
    and does not move a vertex of the outer surface.

    A hollow that is open to the outside, even through one gap, is not a
    pocket in this sense and stays.

    Returns the filled grid and the number of grid points that were raised.
    """
    from scipy import ndimage  # imported here because only meshes need the package

    empty = (density <= threshold).numpy()     # marching cubes also counts a point at the threshold as empty
    part, _ = ndimage.label(empty)             # connected parts of empty space, numbered from 1
    on_faces = [part[0], part[-1], part[:, 0], part[:, -1], part[:, :, 0], part[:, :, -1]]
    outside = np.unique(np.concatenate([face.ravel() for face in on_faces]))
    sealed = torch.from_numpy(empty & ~np.isin(part, outside))
    filled = density.clone()
    filled[sealed] = density.max()
    return filled, int(sealed.sum())


def extract_surface(density: torch.Tensor, bound: float, threshold: float = 50.0) -> Mesh:
    """The surface on which the density equals `threshold`, by marching cubes.

    Marching cubes looks at each cell of the grid in turn. Where some corners
    of a cell are above the threshold and others below, the surface passes
    through the cell, and the algorithm places triangles through the points on
    the cell's edges where the linearly interpolated density equals the
    threshold. The implementation is scikit-image's.

    Because of that interpolation, the surface between an empty grid point and
    one of density D lies the fraction threshold / D of the way across: a
    threshold far below D puts it next to the empty point, half a cell outside
    the middle.

    The triangles are wound counter-clockwise as seen from the low-density
    side, so their normals point out of the object.

    Args:
        density: (R, R, R) grid from `density_grid`.
        bound: the bound that grid was made with.
        threshold: density at which empty space ends and the object begins.
            There is no single right value. 50 is the authors' choice.
    """
    from skimage import measure  # imported here because only meshes need the package

    lowest, highest = density.min().item(), density.max().item()
    if not lowest < threshold < highest:
        raise ValueError(
            f"no surface at threshold {threshold:g}: the density ranges "
            f"from {lowest:.4g} to {highest:.4g}"
        )
    spacing = 2.0 * bound / (density.shape[0] - 1)
    with warnings.catch_warnings():
        # scikit-image 0.26 reshapes its arrays in a way NumPy 2.5 has deprecated
        warnings.simplefilter("ignore", DeprecationWarning)
        vertices, faces, _, _ = measure.marching_cubes(
            density.numpy(), level=threshold, spacing=(spacing, spacing, spacing),
            gradient_direction="ascent", allow_degenerate=False,
        )
    return Mesh(
        torch.from_numpy(vertices.astype(np.float32)) - bound,
        torch.from_numpy(faces.astype(np.int64)),
    )


def piece_of_each_face(mesh: Mesh) -> torch.Tensor:
    """(F,) which connected piece of the mesh each face belongs to, from 0."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    faces = mesh.faces.numpy()
    num_vertices = mesh.vertices.shape[0]
    # two edges per face are enough to join its three corners
    start = np.concatenate([faces[:, 0], faces[:, 1]])
    end = np.concatenate([faces[:, 1], faces[:, 2]])
    graph = coo_matrix((np.ones(start.size, dtype=bool), (start, end)), shape=(num_vertices, num_vertices))
    _, piece_of_vertex = connected_components(graph, directed=False)
    return torch.from_numpy(piece_of_vertex[faces[:, 0]].astype(np.int64))


def drop_small_pieces(mesh: Mesh, min_fraction: float = 0.01) -> tuple[Mesh, int, int]:
    """Remove the pieces whose area is below `min_fraction` of the largest piece's.

    A NeRF is free to put density wherever it does not show in the training
    views, and usually leaves a few specks floating in empty space. They are
    separate little surfaces, far smaller than the object.

    Returns the cleaned mesh, the number of pieces kept and the number dropped.
    Vertex colours are not carried over; colour the mesh after cleaning it.
    """
    piece = piece_of_each_face(mesh)
    area = torch.zeros(int(piece.max()) + 1, dtype=torch.float64).index_add_(0, piece, face_areas(mesh))
    keep_piece = area >= min_fraction * area.max()
    faces = mesh.faces[keep_piece[piece]]

    used, renumbered = torch.unique(faces, return_inverse=True)   # drop the vertices nothing uses
    kept = int(keep_piece.sum())
    return Mesh(mesh.vertices[used], renumbered), kept, keep_piece.numel() - kept


# --------------------------------------------------------------------------
# Measures of a mesh
# --------------------------------------------------------------------------


def _corners(mesh: Mesh) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    positions = mesh.vertices.double()[mesh.faces]   # (F, 3, 3)
    return positions[:, 0], positions[:, 1], positions[:, 2]


def face_areas(mesh: Mesh) -> torch.Tensor:
    a, b, c = _corners(mesh)
    return 0.5 * torch.linalg.cross(b - a, c - a).norm(dim=-1)


def surface_area(mesh: Mesh) -> float:
    return face_areas(mesh).sum().item()


def enclosed_volume(mesh: Mesh) -> float:
    """Volume inside a closed mesh, by the divergence theorem.

    Each face and the origin span a tetrahedron of signed volume
    a . (b x c) / 6. For a closed surface with outward-facing triangles the
    signed volumes add up to the volume enclosed, wherever the origin is.
    The result is negative if the triangles face inwards and meaningless if
    the surface is not closed (see `is_closed`).
    """
    a, b, c = _corners(mesh)
    return ((a * torch.linalg.cross(b, c)).sum(dim=-1).sum() / 6.0).item()


def is_closed(mesh: Mesh) -> bool:
    """Whether the mesh has no open edge and its triangles are wound consistently.

    Each triangle runs along its three edges in one direction. The mesh is
    closed if every edge is run along as often one way as the other, which
    for an ordinary surface means once each way, by the triangles on its two
    sides. `enclosed_volume` has a meaning only for a closed mesh.
    """
    faces = mesh.faces
    num_vertices = mesh.vertices.shape[0]
    start = torch.cat([faces[:, 0], faces[:, 1], faces[:, 2]])
    end = torch.cat([faces[:, 1], faces[:, 2], faces[:, 0]])
    forward = torch.sort(start * num_vertices + end).values
    backward = torch.sort(end * num_vertices + start).values
    return torch.equal(forward, backward)


def vertex_normals(mesh: Mesh) -> torch.Tensor:
    """(V, 3) unit normals: the area-weighted mean of the adjoining faces' normals."""
    a, b, c = _corners(mesh)
    weighted = torch.linalg.cross(b - a, c - a)   # length = twice the face's area
    normals = torch.zeros(mesh.vertices.shape[0], 3, dtype=torch.float64)
    for corner in range(3):
        normals.index_add_(0, mesh.faces[:, corner], weighted)
    return torch.nn.functional.normalize(normals, dim=-1).float()


# --------------------------------------------------------------------------
# Colour and file output
# --------------------------------------------------------------------------


@torch.no_grad()
def colour_vertices(
    field, mesh: Mesh, voxel: float, device: str = "cpu", chunk: int = 32768, samples: int = 16
) -> Mesh:
    """Give every vertex the colour the field shows when looked at head-on.

    A short ray is cast at each vertex along its inward normal, from one voxel
    outside the surface to three voxels inside. The field is read at the start
    of each of `samples` equal steps, and the steps are rendered with the same
    quadrature as an image pixel, counting only what lies on that stretch. The
    result is divided by the opacity gathered, so it is the colour of what the
    ray met, however faint. A vertex whose ray meets nothing at all takes the
    colour the field gives at the vertex itself.

    What this cannot give: a mesh has one colour per point, so highlights that
    move with the viewpoint are frozen as seen from straight ahead, and for a
    surface that no camera faced that is a direction the network never saw. A
    sheet thinner than three voxels takes some colour from its far side. Where
    the density is low the colour is an average over the whole stretch, not
    of the skin alone.

    Args:
        field: callable (points, view_dirs) -> (sigma, rgb).
        mesh: the surface, with outward-facing triangles.
        voxel: spacing of the grid the mesh was extracted on.
        samples: field queries per ray.
    """
    normals = vertex_normals(mesh)
    edges = torch.linspace(0.0, 4.0 * voxel, samples + 1, device=device)   # where the steps begin, and the end
    colours = torch.empty(mesh.vertices.shape[0], 3)
    for start in range(0, mesh.vertices.shape[0], chunk):
        inward = -normals[start : start + chunk].to(device)
        vertices = mesh.vertices[start : start + chunk].to(device)
        origins = vertices - voxel * inward
        points = origins[:, None, :] + edges[None, :-1, None] * inward[:, None, :]   # (n, samples, 3)
        sigma, rgb = field(points, inward[:, None, :].expand_as(points))
        # `volume_render` lets its last sample reach to infinity. An empty one
        # is added at the end of the stretch, so that nothing beyond it counts.
        closed_sigma = torch.cat([sigma, torch.zeros_like(sigma[:, :1])], dim=1)
        closed_rgb = torch.cat([rgb, torch.zeros_like(rgb[:, :1])], dim=1)
        out = volume_render(closed_sigma, closed_rgb, edges.expand(points.shape[0], -1), inward)
        colour = out.rgb / out.acc.clamp(min=1e-6)[:, None]
        nothing = out.acc <= 1e-6                 # empty space all along, or no normal to cast a ray along
        if nothing.any():
            colour[nothing] = field(vertices[nothing], inward[nothing])[1]
        colours[start : start + chunk] = colour.cpu()
    return Mesh(mesh.vertices, mesh.faces, colours.clamp(0.0, 1.0))


def save_ply(path, mesh: Mesh) -> None:
    """Write the mesh as a binary PLY file, with vertex colours if it has them.

    PLY is read by MeshLab, Blender and most other 3D programs. Positions are
    in the scene's own coordinates, in which z points up.
    """
    fields = [("x", "<f4"), ("y", "<f4"), ("z", "<f4")]
    if mesh.colours is not None:
        fields += [("red", "u1"), ("green", "u1"), ("blue", "u1")]
    vertices = np.empty(mesh.vertices.shape[0], dtype=fields)
    for axis, name in enumerate("xyz"):
        vertices[name] = mesh.vertices[:, axis].numpy()
    if mesh.colours is not None:
        eight_bit = (mesh.colours.clamp(0, 1) * 255).round().byte().numpy()
        for channel, name in enumerate(("red", "green", "blue")):
            vertices[name] = eight_bit[:, channel]

    faces = np.empty(mesh.faces.shape[0], dtype=[("count", "u1"), ("corners", "<i4", (3,))])
    faces["count"] = 3
    faces["corners"] = mesh.faces.numpy()

    kinds = {"<f4": "float", "u1": "uchar"}
    header = ["ply", "format binary_little_endian 1.0", f"element vertex {vertices.size}"]
    header += [f"property {kinds[kind]} {name}" for name, kind in fields]
    header += [f"element face {faces.size}", "property list uchar int vertex_indices", "end_header"]
    with Path(path).open("wb") as handle:
        handle.write(("\n".join(header) + "\n").encode("ascii"))
        handle.write(vertices.tobytes())
        handle.write(faces.tobytes())


# --------------------------------------------------------------------------
# Looking at a mesh through the scene's cameras
# --------------------------------------------------------------------------


class Raster(NamedTuple):
    face: torch.Tensor      # (H, W) index of the face seen at each pixel, -1 where there is none
    weights: torch.Tensor   # (H, W, 3) where in that face: weights of its three corners, summing to 1
    depth: torch.Tensor     # (H, W) depth of the surface at the pixel, inf where there is none


def rasterise(
    mesh: Mesh, height: int, width: int, focal: float, c2w: torch.Tensor, budget: int = 2_000_000
) -> Raster:
    """Which triangle a camera sees at each pixel.

    The camera is that of `nerf.rays.get_rays`: pixel (row j, column i) is
    covered by a triangle if the ray through that pixel passes through it,
    which is the case when the point (i, j) lies inside the triangle's
    projection. Where several triangles cover a pixel, the nearest wins, and
    of triangles at exactly the same depth the one with the lowest index.

    Depth is interpolated the perspective-correct way, linearly in 1/depth.
    Triangles with a corner behind the camera are skipped: the scenes here
    are looked at from outside.

    Args:
        mesh, height, width, focal, c2w: the mesh and the camera.
        budget: pixel-triangle pairs tested at a time, which bounds memory.
    """
    c2w = torch.as_tensor(c2w).cpu().double()
    cols, rows, depth = project_points(mesh.vertices.double(), height, width, focal, c2w)
    in_front = (depth[mesh.faces] > 0).all(dim=1).nonzero().squeeze(1)

    corners = torch.stack([cols, rows], dim=-1)[mesh.faces[in_front]]   # (N, 3, 2) as (column, row)
    corner_depth = depth[mesh.faces[in_front]]                          # (N, 3)
    # the pixels inside each triangle's bounding box and inside the image
    first = corners.amin(dim=1).ceil().long().clamp(min=0)
    last = torch.minimum(corners.amax(dim=1).floor().long(), torch.tensor([width - 1, height - 1]))
    extent = (last - first + 1).amax(dim=1)

    nearest = torch.full((height * width,), float("inf"), dtype=torch.float64)
    face = torch.full((height * width,), -1, dtype=torch.long)
    weights = torch.zeros(height * width, 3, dtype=torch.float64)

    def cross(u, v):
        return u[..., 0] * v[..., 1] - u[..., 1] * v[..., 0]

    for size in extent.unique().tolist():
        if size < 1:
            continue   # the bounding box holds no pixel
        window = torch.stack(torch.meshgrid(torch.arange(size), torch.arange(size), indexing="xy"), dim=-1)
        window = window.reshape(-1, 2)                                   # (size^2, 2) offsets as (column, row)
        group = (extent == size).nonzero().squeeze(1)
        for part in group.split(max(1, budget // (size * size))):
            pixel = first[part][:, None, :] + window[None]              # (n, size^2, 2)
            p = pixel.double()
            a, b, c = (corners[part][:, None, k] for k in range(3))
            doubled_area = cross(b - a, c - a)
            # barycentric coordinates: each is 0 on the opposite edge and 1 at its own corner
            wa = cross(b - p, c - p) / doubled_area
            wb = cross(c - p, a - p) / doubled_area
            wc = 1.0 - wa - wb
            inside = (
                (wa >= -1e-9) & (wb >= -1e-9) & (wc >= -1e-9)
                & (pixel <= last[part][:, None, :]).all(dim=-1)
            )
            za, zb, zc = (corner_depth[part][:, None, k] for k in range(3))
            inverse_depth = wa / za + wb / zb + wc / zc
            corrected = torch.stack([wa / za, wb / zb, wc / zc], dim=-1) / inverse_depth[..., None]

            where = (pixel[..., 1] * width + pixel[..., 0])[inside]
            how_far = (1.0 / inverse_depth)[inside]
            which = in_front[part][:, None].expand_as(inside)[inside]
            before = nearest
            nearest = nearest.scatter_reduce(0, where, how_far, reduce="amin")
            wins = how_far <= nearest[where]
            # On an exact tie between faces the lowest index is taken, also
            # against a face from an earlier batch that is still the nearest.
            standing = torch.where((nearest == before) & (face >= 0), face, mesh.faces.shape[0])
            lowest = standing.scatter_reduce(0, where[wins], which[wins], reduce="amin")
            chosen = wins & (which == lowest[where])
            face[where[chosen]] = which[chosen]
            weights[where[chosen]] = corrected[inside][chosen]

    return Raster(
        face.reshape(height, width),
        weights.float().reshape(height, width, 3),
        nearest.float().reshape(height, width),
    )


def colour_picture(mesh: Mesh, raster: Raster) -> torch.Tensor:
    """The mesh in its vertex colours, from the result of `rasterise`: (H, W, 3).

    No light is added: the colours come from the radiance field, which has the
    scene's own lighting in it already. The background is white.
    """
    image = torch.ones(*raster.face.shape, 3)
    covered = raster.face >= 0
    seen = mesh.faces[raster.face[covered]]            # (P, 3) corners of the face at each covered pixel
    image[covered] = (mesh.colours[seen] * raster.weights[covered][..., None]).sum(dim=1)
    return image


def shape_picture(mesh: Mesh, raster: Raster, c2w: torch.Tensor) -> torch.Tensor:
    """The mesh in plain grey, lit from the camera, to show its shape: (H, W, 3).

    Brightness follows the angle between the surface and the line of sight: a
    patch turned straight towards the camera is shown at full brightness, one
    seen edge-on at 30 %. The surface direction at a pixel is interpolated
    from the vertex normals, so bumps show but single triangles do not. The
    background is white.
    """
    image = torch.ones(*raster.face.shape, 3)
    covered = raster.face >= 0
    seen = mesh.faces[raster.face[covered]]            # (P, 3) corners of the face at each covered pixel
    weights = raster.weights[covered][..., None]       # (P, 3, 1)
    normal = torch.nn.functional.normalize((vertex_normals(mesh)[seen] * weights).sum(dim=1), dim=-1)
    here = (mesh.vertices[seen] * weights).sum(dim=1)
    to_camera = torch.nn.functional.normalize(torch.as_tensor(c2w)[:3, 3].float().cpu() - here, dim=-1)
    facing = (normal * to_camera).sum(dim=-1).clamp(min=0.0)
    image[covered] = 0.8 * (0.3 + 0.7 * facing)[:, None].expand(-1, 3)
    return image


def silhouette_iou(mesh: Mesh, views: Views, threshold: float = 0.5) -> list[float]:
    """How well the mesh's outline matches the object's in each view.

    For every view the mesh is rasterised from that view's camera, and the set
    of pixels it covers is compared with the set of pixels where the image is
    opaque (alpha above `threshold`). The score is the intersection of the two
    sets over their union: 1 for identical outlines, lower if the mesh is too
    fat, too thin, misplaced or has pieces the object does not. A view in
    which neither shows scores 1.

    It needs no 3D ground truth, only images with alpha, and it can use views
    the model was not trained on. What it checks is that the mesh is in the
    right place, at the right size and with the right outline. It is blind to
    everything inside the outline: a dent leaves the score as it is, and a
    visual hull carved from the same outlines would score as well as the true
    surface. A part of the object that no camera sees edge-on does not enter
    it at all, and a thin part counts for no more than the few pixels it
    covers.
    """
    if views.alpha is None:
        raise ValueError("silhouette_iou needs views with alpha (silhouettes)")
    scores = []
    for alpha, pose in zip(views.alpha, views.poses):
        covered = rasterise(mesh, views.height, views.width, views.focal, pose).face >= 0
        silhouette = alpha.cpu() > threshold
        union = (covered | silhouette).sum().item()
        scores.append((covered & silhouette).sum().item() / union if union else 1.0)
    return scores


def fit_threshold(
    density: torch.Tensor,
    bound: float,
    views: Views,
    candidates: tuple[float, ...] = THRESHOLDS,
    min_fraction: float = 0.01,
    tolerance: float = 0.01,
    preferred: float | None = 50.0,
    report=None,
) -> tuple[float, dict[float, float]]:
    """Choose among candidate thresholds by how well their meshes match the silhouettes.

    For each candidate the surface is extracted, its small pieces are dropped
    as they will be from the final mesh, and it is scored by its mean
    `silhouette_iou` over `views`. Too low a threshold takes in the haze
    around the object and the mesh is too fat; too high a one keeps only the
    densest parts and it is too thin.

    The best score is not taken blindly, because differences of a hundredth
    in it mean little. All candidates that score within `tolerance` of the
    best count as fitting the outlines equally well, and among them:

      - `preferred` is taken if it is one of them. That is the authors' 50,
        which is kept wherever the outlines have nothing against it.
      - Otherwise the lowest is taken, because the two ways of being wrong
        are not alike. A higher threshold only ever removes material, and a
        thin or faint part of the object can vanish whole for a gain in score
        too small to mean anything. The score also leans that way by itself:
        the outline of a rough surface is drawn by the tops of its bumps, so
        the mesh with the best outline lies, on average, inside the true
        surface by about the height of the bumps.

    For the same reason no candidate is above 50. Where density jumps from
    nothing to hundreds between two grid points, a higher threshold moves the
    surface inwards within its cell and the outline score keeps rising with
    it, by amounts that say nothing about the object, while everything
    fainter than the threshold is lost.

    What a tolerance cannot do is protect a part that makes up less of the
    outline than the tolerance itself: if 50 scores within it, 50 is taken,
    even where a lower threshold would have kept such a part. Outlines are all
    this sees. The smoke test, whose scene has a known surface, measures how
    far the chosen mesh is from it (scripts/00_smoke_test.py).

    `views` should be views the model was trained on, so that test views stay
    untouched for judging the result. `report`, if given, is called with each
    candidate and its score as soon as it is known.

    Returns the chosen candidate and the score of every candidate at which
    the density has a surface at all.
    """
    scores = {}
    for threshold in candidates:
        try:
            surface = extract_surface(density, bound, threshold)
        except ValueError:
            continue   # the density never crosses this value
        mesh, _, _ = drop_small_pieces(surface, min_fraction)
        per_view = silhouette_iou(mesh, views)
        scores[threshold] = sum(per_view) / len(per_view)
        if report is not None:
            report(threshold, scores[threshold])
    if not scores:
        raise ValueError(
            f"no surface at any of the thresholds from {min(candidates):g} to {max(candidates):g}: "
            f"the density ranges from {density.min().item():.4g} to {density.max().item():.4g}"
        )
    as_good = [threshold for threshold, score in scores.items() if score >= max(scores.values()) - tolerance]
    return (preferred if preferred in as_good else min(as_good)), scores
