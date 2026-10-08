"""End-to-end check: train a small NeRF on a known scene and render new views.

    python scripts/00_smoke_test.py

Trains on 24 synthetic 48x48 images of four coloured spheres (see
`nerf.data.SphereScene`), then renders 4 viewpoints that were not in the
training set and compares them with the truth. Then it turns the trained
network into a triangle mesh, as scripts/05_extract_mesh.py does, and measures
the mesh against the spheres, whose surface is known exactly. Takes a couple
of minutes on a laptop CPU. Writes

    results/smoke_test.png        top row: true held-out views; bottom row: renders
    results/smoke_test_mesh.png   the same four viewpoints: the true view, the
                                  mesh in its colours, the mesh's shape
    results/smoke_test.json       the numbers

and exits with an error if the mean held-out PSNR is below 20 dB or if the
mesh, seen from the held-out viewpoints, misses any of these: its outline
overlaps the true outline by at least 0.92 (intersection over union); the
points of it that the cameras see are on average within 0.04 of the true
surface; it encloses the spheres' volume to within 20 %; its colours are on
average within 0.10 of the true views'. For scale: the largest sphere has
radius 0.7, the mesh is taken from a grid with cells of 0.025, and one pixel
of a training image covers about 0.06 of the scene.

The network here is deliberately small (4 layers of 64 units, 16 + 32 samples
per ray, 256 rays per step) so that this runs without a GPU. It checks that the
whole pipeline learns; it is not the paper's configuration.

Two meshes that involve no training are measured the same way, to say what
the numbers are worth: the visual hull carved from the outlines in the 24
training images, and the mesh of the scene's own exact density. The spheres
are convex, so their outlines already say nearly everything about them, and
on this scene the hull does about as well as the network's mesh. What the
scene can show is that training, meshing and measuring work together, and
how close a mesh from a network this small comes.
"""

import argparse
import json
import math
import platform
import sys
import time
from pathlib import Path

import torch
from PIL import Image

from nerf.data import SphereScene, analytic_views
from nerf.hull import visual_hull
from nerf.mesh import (
    colour_picture,
    colour_vertices,
    density_grid,
    drop_small_pieces,
    enclosed_volume,
    extract_surface,
    fill_cavities,
    fit_threshold,
    is_closed,
    rasterise,
    shape_picture,
    silhouette_iou,
)
from nerf.rays import get_rays
from nerf.train import TrainConfig, evaluate, psnr, train

PSNR_REQUIRED = 20.0
# What the mesh has to reach. A run that works clears each with room to spare.
OUTLINE_IOU_REQUIRED = 0.92
SURFACE_DISTANCE_ALLOWED = 0.04
VOLUME_ERROR_ALLOWED = 0.20
COLOUR_ERROR_ALLOWED = 0.10

# The grid the mesh is taken from. The spheres reach out to 1.45.
MESH_BOUND, MESH_RESOLUTION = 1.6, 129
MESH_VIEW_SIZE = 128   # pixels per side of the views the mesh is measured in


def against_the_scene(mesh, views) -> dict:
    """Measure a mesh against the exact spheres, through the cameras of `views`.

    outline_iou              overlap of its outline with the true outline
    surface_distance         mean distance from the points of the mesh that
                             the cameras see to the true surface
    surface_distance_signed  the same, counting points inside the true surface
                             as negative: where the mesh lies on balance
    volume_ratio             the volume it encloses over that of the spheres;
                             None if it is not a closed surface
    colour_error             for a mesh with colours: mean difference from the
                             true views, where both show the object
    """
    size, focal = views.height, views.focal
    scores = silhouette_iou(mesh, views)
    distances, colour_errors = [], []
    for image, alpha, pose in zip(views.images, views.alpha, views.poses):
        raster = rasterise(mesh, size, size, focal, pose)
        seen = raster.face >= 0
        origins, directions = get_rays(size, size, focal, pose)
        distances.append(SphereScene().distance_to_surface(origins[seen] + raster.depth[seen][:, None] * directions[seen]))
        if mesh.colours is not None:
            both = seen & (alpha > 0.5)
            colour_errors.append((colour_picture(mesh, raster)[both] - image[both]).abs().mean().item())
    distances = torch.cat(distances)
    true_volume = 4.0 / 3.0 * math.pi * sum(radius**3 for _, radius, _ in SphereScene.SPHERES)
    measures = {
        "outline_iou_per_view": [round(score, 3) for score in scores],
        "outline_iou": round(sum(scores) / len(scores), 3),
        "surface_distance": round(distances.abs().mean().item(), 4),
        "surface_distance_signed": round(distances.mean().item(), 4),
        "volume_ratio": round(enclosed_volume(mesh) / true_volume, 3) if is_closed(mesh) else None,
    }
    if colour_errors:
        measures["colour_error"] = round(sum(colour_errors) / len(colour_errors), 3)
    return measures


def measure_mesh(fine, train_views, device: str, out: Path) -> dict:
    """Mesh the trained density and measure it against the exact scene.

    The threshold is fitted to the silhouettes in eight of the training
    views. The mesh is then judged from the held-out viewpoints, in larger
    pictures than the network was trained on.
    """
    density = density_grid(fine, MESH_BOUND, MESH_RESOLUTION, device)
    # Nothing of the scene reaches the faces of the grid. Emptying them closes
    # the surface around any wisp of density that training has left there.
    for face in (density[0], density[-1], density[:, 0], density[:, -1], density[:, :, 0], density[:, :, -1]):
        face.zero_()
    spread = sorted({round(k * (len(train_views) - 1) / 7) for k in range(8)})
    try:
        threshold, _ = fit_threshold(density, MESH_BOUND, train_views.select(spread))
    except ValueError as error:
        return {"error": str(error)}
    filled, _ = fill_cavities(density, threshold)
    mesh, pieces, _ = drop_small_pieces(extract_surface(filled, MESH_BOUND, threshold))
    mesh = colour_vertices(fine, mesh, 2 * MESH_BOUND / (MESH_RESOLUTION - 1), device)

    views = analytic_views(4, image_size=MESH_VIEW_SIZE, phase=40.0, num_samples=128)
    result = {
        "threshold": threshold,
        "vertices": mesh.vertices.shape[0],
        "triangles": mesh.faces.shape[0],
        "pieces": pieces,
        **against_the_scene(mesh, views),
        "required": {
            "outline_iou": OUTLINE_IOU_REQUIRED, "surface_distance": SURFACE_DISTANCE_ALLOWED,
            "volume_error": VOLUME_ERROR_ALLOWED, "colour_error": COLOUR_ERROR_ALLOWED,
        },
    }

    # For scale: two meshes that involve no training, on the same grid.
    carved, _ = visual_hull(train_views, MESH_RESOLUTION, MESH_BOUND)
    hull, _, _ = drop_small_pieces(extract_surface(carved.float(), MESH_BOUND, 0.5))
    scene = SphereScene()
    exact = extract_surface(density_grid(scene, MESH_BOUND, MESH_RESOLUTION), MESH_BOUND, 0.5 * scene.density)
    result["for_comparison"] = {
        "visual_hull_of_the_training_outlines": against_the_scene(hull, views),
        "exact_density": against_the_scene(exact, views),
    }

    # The picture: each held-out view, the mesh in its colours, and its shape, at twice the size.
    size, focal = views.height, views.focal
    columns = []
    for image, pose in zip(views.images, views.poses):
        detailed = rasterise(mesh, 2 * size, 2 * size, 2 * focal, pose)
        true_view = image.repeat_interleave(2, dim=0).repeat_interleave(2, dim=1)
        columns.append([true_view, colour_picture(mesh, detailed), shape_picture(mesh, detailed, pose)])

    def framed(tile):   # a thin grey frame around each picture, as in smoke_test.png
        return torch.nn.functional.pad(tile.permute(2, 0, 1), (1, 1, 1, 1), value=0.75).permute(1, 2, 0)

    grid = torch.cat([torch.cat([framed(tile) for tile in column], dim=0) for column in columns], dim=1)
    Image.fromarray((grid.clamp(0, 1) * 255).round().byte().numpy()).save(out / "smoke_test_mesh.png")
    return result


def mesh_failures(mesh: dict) -> list[str]:
    """What is wrong with the measured mesh, in words; empty if nothing is."""
    if "error" in mesh:
        return ["the trained density has no surface to make a mesh from"]
    failures = []
    if mesh["outline_iou"] < OUTLINE_IOU_REQUIRED:
        failures.append(f"the outline of the mesh overlaps the true one by less than {OUTLINE_IOU_REQUIRED}")
    if mesh["surface_distance"] > SURFACE_DISTANCE_ALLOWED:
        failures.append(f"the mesh's surface is on average more than {SURFACE_DISTANCE_ALLOWED} from the true one")
    if mesh["volume_ratio"] is None:
        failures.append("the mesh is not a closed surface")
    elif abs(mesh["volume_ratio"] - 1.0) > VOLUME_ERROR_ALLOWED:
        failures.append(f"the mesh's volume is more than {VOLUME_ERROR_ALLOWED:.0%} from the spheres'")
    if mesh["colour_error"] > COLOUR_ERROR_ALLOWED:
        failures.append(f"the mesh's colours are on average more than {COLOUR_ERROR_ALLOWED} from the true views'")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--iterations", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu", help="cpu (default), mps or cuda")
    parser.add_argument("--out", type=Path, default=Path("results"))
    args = parser.parse_args()

    train_views = analytic_views(24, image_size=48)
    test_views = analytic_views(4, image_size=48, phase=40.0)  # other viewpoints
    empty = psnr(torch.ones_like(test_views.images), test_views.images)
    print(f"{len(train_views)} training views, {len(test_views)} held-out views")
    print(f"PSNR of a blank white image on the held-out views: {empty:.2f} dB")

    config = TrainConfig(
        iterations=args.iterations, batch_size=256, num_coarse=16, num_fine=32,
        depth=4, width=64, skip=2,
        lr_decay_steps=10**9,  # effectively a constant learning rate for a short run
        log_every=200, seed=args.seed,
    )
    start = time.time()
    result = train(train_views, config, device=args.device)
    seconds = time.time() - start

    scores, renders = evaluate(test_views, result, config, device=args.device)
    mean_psnr = sum(scores) / len(scores)
    print("held-out PSNR per view: " + "  ".join(f"{s:.2f}" for s in scores))
    print(f"mean held-out PSNR: {mean_psnr:.2f} dB after {args.iterations} steps ({seconds:.0f} s)")

    args.out.mkdir(parents=True, exist_ok=True)

    def row(images):  # views side by side, each with a thin grey frame
        framed = [torch.nn.functional.pad(i.permute(2, 0, 1), (1, 1, 1, 1), value=0.75) for i in images]
        return torch.cat(framed, dim=2).permute(1, 2, 0)

    grid = torch.cat([row(test_views.images), row(renders.clamp(0, 1))], dim=0)
    image = Image.fromarray((grid * 255).round().byte().numpy())
    image = image.resize((image.width * 4, image.height * 4), Image.NEAREST)
    image.save(args.out / "smoke_test.png")

    mesh = measure_mesh(result.fine, train_views, args.device, args.out)
    if "error" in mesh:
        print(f"mesh: there is {mesh['error']}")
    else:
        print(f"mesh at a density threshold of {mesh['threshold']:g}: {mesh['triangles']:,} triangles "
              f"in {mesh['pieces']} pieces, colours {mesh['colour_error']:.3f} from the true views' on average")
        print("                                       outline IoU   distance to the true surface   volume")
        rows = {"from the trained network": mesh,
                "visual hull of the training outlines": mesh["for_comparison"]["visual_hull_of_the_training_outlines"],
                "the scene's own density": mesh["for_comparison"]["exact_density"]}
        for name, measures in rows.items():
            volume = "open" if measures["volume_ratio"] is None else f"{measures['volume_ratio']:.2f}"
            print(f"  {name:38s} {measures['outline_iou']:.3f}        {measures['surface_distance']:.4f} "
                  f"(signed {measures['surface_distance_signed']:+.4f})      {volume}")

    summary = {
        "iterations": args.iterations,
        "seed": args.seed,
        "device": args.device,
        "torch": torch.__version__,
        "machine": platform.machine(),
        "train_seconds": round(seconds, 1),
        "psnr_blank_image": round(empty, 2),
        "psnr_held_out_views": [round(s, 2) for s in scores],
        "psnr_held_out_mean": round(mean_psnr, 2),
        "psnr_required": PSNR_REQUIRED,
        "mesh": mesh,
    }
    (args.out / "smoke_test.json").write_text(json.dumps(summary, indent=2) + "\n")
    pictures = ["smoke_test.png"] + ([] if "error" in mesh else ["smoke_test_mesh.png"])
    print("wrote " + ", ".join(str(args.out / name) for name in pictures)
          + f" and {args.out / 'smoke_test.json'}")

    failures = [f"mean held-out PSNR is below {PSNR_REQUIRED} dB"] if mean_psnr < PSNR_REQUIRED else []
    failures += mesh_failures(mesh)
    for failure in failures:
        print(f"FAIL: {failure}")
    if failures:
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
