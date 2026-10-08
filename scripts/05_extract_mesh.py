"""Turn a trained NeRF into a triangle mesh, and check the mesh against the photographs.

    python scripts/05_extract_mesh.py data/nerf_synthetic/lego runs/lego_800px/checkpoint.pt

The surface is taken from the fine network's density, as in the authors'
extract_mesh notebook: 257 grid points per axis spanning [-1.2, 1.2], and
marching cubes at a density threshold. Three things are added to the notebook's
steps. Empty pockets that the surface seals off on all sides are filled first,
so that the mesh has no hidden inner walls. Small pieces floating in empty
space are removed. Every vertex is given the colour the network shows there.

The results go into a directory next to the checkpoint, named after the
checkpoint's step (or into --out):

    mesh.ply        the mesh with vertex colours; opens in MeshLab or Blender
    preview.png     four test views: the photograph, the mesh in its colours,
                    and the mesh in plain grey to show its shape
    summary.json    the size of the mesh, its measures, the silhouette scores

A NeRF is never told where surfaces are, so the mesh is scored against the
scene's own test photographs. From each test camera the pixels the mesh covers
are compared with the pixels where the photograph shows the object, and the
score is the intersection of the two outlines over their union: 1 if they are
identical. It needs no 3D ground truth, and that is also its limit. It shows
that the mesh is in the right place, at the right size and with the right
outline. It cannot see a dent, which does not change an outline, and a visual
hull carved from the same outlines would score as well. By default every
eighth test view is used.

The photographs are used at the size the model was trained on. The data loader
follows the released code, in which a shrunk photograph sits a fraction of a
pixel away from where its camera says, so a model and photographs of another
size do not quite line up. --downscale compares at another size all the same.

The notebook uses a threshold of 50 on its example model (200,000 steps at
half resolution), and prints that 3.6 % of the grid is above it and that the
mesh has 791,052 triangles. How much the threshold matters depends on how
sharp the model's surfaces are, so by default it is chosen with photographs
the model was trained on. For 13 values from 1 up to the authors' 50 the
outline of the mesh is compared with the object's, and those within 0.01 of
the best count as fitting equally well: of these 50 is used if it is one of
them, and otherwise the lowest, because a higher threshold only removes
material. The score that is reported comes from test photographs.
--threshold 50 uses the authors' value whatever the outlines say.

The outline of a rough surface is drawn by its bumps, so a mesh chosen by its
outline tends to lie a little inside the true surface while the model is
still rough. The smoke test measures that on a scene whose surface is known.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from PIL import Image

from nerf.data import blender_frames, load_blender
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
    save_ply,
    shape_picture,
    silhouette_iou,
    surface_area,
)
from nerf.train import load_networks, pick_device, weights_fingerprint

PREVIEW_WIDTH = 400


def to_picture(image: torch.Tensor) -> Image.Image:
    """An (H, W, 3) tensor with colours in [0, 1] as an 8-bit image."""
    return Image.fromarray((image.clamp(0, 1) * 255).round().byte().numpy())


def save_preview(path: Path, columns: list[list[Image.Image]]) -> None:
    """Views side by side, each a column of pictures of the same size."""
    gap = 4
    width, height = columns[0][0].size
    canvas = Image.new(
        "RGB",
        (len(columns) * (width + gap) + gap, len(columns[0]) * (height + gap) + gap),
        (191, 191, 191),
    )
    for across, column in enumerate(columns):
        for down, picture in enumerate(column):
            canvas.paste(picture, (gap + across * (width + gap), gap + down * (height + gap)))
    canvas.save(path)


def positive(text: str) -> int:
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError("must be 1 or more")
    return value


def grid_points(text: str) -> int:
    value = int(text)
    if value < 2:
        raise argparse.ArgumentTypeError("must be 2 or more")
    return value


def length(text: str) -> float:
    value = float(text)
    if not 0 < value < float("inf"):
        raise argparse.ArgumentTypeError("must be a number above 0")
    return value


def fraction(text: str) -> float:
    value = float(text)
    if not 0 <= value <= 1:
        raise argparse.ArgumentTypeError("must be between 0 and 1")
    return value


def threshold_choice(text: str):
    return text if text == "auto" else float(text)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("scene_dir", type=Path)
    parser.add_argument("checkpoint", type=Path, help="a checkpoint file of a training run")
    parser.add_argument("--resolution", type=grid_points, default=257, help="grid points per axis")
    parser.add_argument("--bound", type=length, default=1.2, help="the grid spans [-bound, bound]")
    parser.add_argument("--threshold", type=threshold_choice, default="auto",
                        help="density at which the object begins: a number, or auto (default) to fit it")
    parser.add_argument("--fit-views", type=positive, default=8,
                        help="training views that --threshold auto fits to")
    parser.add_argument("--min-piece", type=fraction, default=0.01,
                        help="drop pieces with less than this fraction of the largest piece's area")
    parser.add_argument("--split", default="test", choices=("train", "val", "test"))
    parser.add_argument("--skip", type=positive, default=8, help="compare outlines in every n-th view")
    parser.add_argument("--downscale", type=positive, default=None,
                        help="shrink the photographs by this factor (default: as in training)")
    parser.add_argument("--device", default="auto", help="auto (default), cuda, mps or cpu")
    parser.add_argument("--out", type=Path, default=None, help="directory for the results")
    args = parser.parse_args(argv)
    try:
        return extract(args)
    except KeyboardInterrupt:
        print("\nStopped before the end. Run the same command again to start it over.")
        return 130


def extract(args) -> int:
    if not args.checkpoint.is_file():
        print(f"There is no checkpoint at {args.checkpoint}.")
        return 2
    downscale = args.downscale
    if downscale is None:
        try:
            downscale = json.loads((args.checkpoint.parent / "config.json").read_text())["downscale"]
            if type(downscale) is not int or downscale < 1:
                raise ValueError(downscale)
        except (OSError, ValueError, KeyError, TypeError):
            print(f"There is no config.json next to {args.checkpoint} that says how much the images "
                  f"were shrunk for training. Say it with --downscale (1 for full size).")
            return 2

    # The photographs are needed last. They are looked at first, before the slow part.
    needed = [args.split] if args.threshold != "auto" else sorted({"train", args.split})
    missing = [f"transforms_{split}.json" for split in needed
               if not (args.scene_dir / f"transforms_{split}.json").is_file()]
    if missing:
        print(f"There is no {' or '.join(missing)} in {args.scene_dir}.")
        print("The first argument is the directory of the scene, such as data/nerf_synthetic/lego.")
        return 2
    names = blender_frames(args.scene_dir, args.split)
    chosen = list(range(0, len(names), args.skip))
    wanted = [(args.split, names, chosen)]     # a split, the photographs it lists, the positions that will be read
    evenly = []
    if args.threshold == "auto":
        training = blender_frames(args.scene_dir, "train")
        evenly = sorted({round(k * (len(training) - 1) / max(1, args.fit_views - 1)) for k in range(args.fit_views)})
        wanted.append(("train", training, evenly))
    for split, listed, positions in wanted:
        if not listed:
            print(f"transforms_{split}.json in {args.scene_dir} lists no views.")
            return 2
        absent = [path for path in (args.scene_dir / f"{listed[k]}.png" for k in positions) if not path.is_file()]
        if absent:
            print(f"Photographs that transforms_{split}.json lists are missing: {len(absent)} of them, "
                  f"the first {absent[0]}")
            return 2
    first = load_blender(args.scene_dir, args.split, downscale, indices=chosen[:1])
    width, height = first.width, first.height
    fit_to = load_blender(args.scene_dir, "train", downscale, indices=evenly) if evenly else None

    device = pick_device(args.device)
    networks = load_networks(args.checkpoint, device)
    out_dir = args.out or args.checkpoint.parent / f"mesh_{networks.step:06d}"
    if out_dir.resolve() == args.checkpoint.parent.resolve() or (out_dir / "checkpoint.pt").exists():
        print(f"{out_dir} is the directory of a training run. Name another directory with --out.")
        return 2
    # Results go into a directory that is new, empty, or holds an earlier mesh.
    # Files whose names start with a dot, which file browsers leave, do not count.
    nearest = next(path for path in (out_dir, *out_dir.parents) if path.exists())
    holds_other_things = out_dir.is_dir() and not (out_dir / "mesh.ply").exists() and any(
        not path.name.startswith(".") for path in out_dir.iterdir())
    if not nearest.is_dir() or holds_other_things:
        print(f"{out_dir} is in use for something else. Name another directory with --out.")
        return 2
    voxel = 2.0 * args.bound / (args.resolution - 1)
    print(f"checkpoint at step {networks.step}, fine network on {args.resolution} points per axis "
          f"in [{-args.bound:g}, {args.bound:g}], device {device}, torch {torch.__version__}")

    start = time.time()
    density = density_grid(networks.fine, args.bound, args.resolution, device)
    unusable = (~torch.isfinite(density)).sum().item()
    if unusable:
        print(f"The density is not a finite number at {unusable:,} of the {density.numel():,} grid points.")
        print("A network gives that after its training has gone wrong. There is no surface to extract.")
        return 2
    print(f"density up to {density.max().item():.4g}   ({time.time() - start:.0f} s)")

    threshold, fitted = args.threshold, {}
    try:
        if threshold == "auto":
            print(f"fitting the threshold to the object's outline in {len(fit_to)} training views")
            threshold, fitted = fit_threshold(
                density, args.bound, fit_to, min_fraction=args.min_piece,
                report=lambda candidate, score: print(f"    threshold {candidate:>3g}   IoU {score:.3f}", flush=True),
            )
            which = "the authors' value" if threshold == 50 else "the lowest"
            print(f"    -> {threshold:g}, {which} within 0.01 of the best")
        filled, pockets = fill_cavities(density, threshold)
        if pockets and (filled > threshold).all():
            print(f"At a threshold of {threshold:g} the object fills the grid out to every face, and the only "
                  f"empty space is sealed inside it.")
            print("There is no outer surface to extract. A larger --bound may reach its outside.")
            return 2
        surface = extract_surface(filled, args.bound, threshold)
    except ValueError as error:
        print(f"There is {error}.")
        print("Early in training the densities are low. Pass a --threshold inside that range.")
        return 2
    above = (density > threshold).float().mean().item()
    print(f"{100 * above:.2f} % of the grid is above the threshold of {threshold:g}; "
          f"{pockets:,} grid points in sealed pockets were filled")
    mesh, kept, dropped = drop_small_pieces(surface, args.min_piece)
    mesh = colour_vertices(networks.fine, mesh, voxel, device)

    out_dir.mkdir(parents=True, exist_ok=True)
    for earlier in ("summary.json", "preview.png"):   # of another mesh, if the directory was used before
        (out_dir / earlier).unlink(missing_ok=True)
    save_ply(out_dir / "mesh.ply", mesh)

    closed = is_closed(mesh)
    volume = round(enclosed_volume(mesh), 5) if closed else None   # an open surface encloses nothing
    low, high = mesh.vertices.amin(dim=0).tolist(), mesh.vertices.amax(dim=0).tolist()
    print(f"surface: {mesh.vertices.shape[0]:,} vertices and {mesh.faces.shape[0]:,} triangles in "
          f"{kept} piece{'s' if kept != 1 else ''}; {dropped} smaller pieces removed")
    print(f"closed surface: {f'yes   volume {volume:.4f}' if closed else 'no, so no volume'}   "
          f"area {surface_area(mesh):.4f}   "
          + "  ".join(f"{axis} {low[k]:.2f} to {high[k]:.2f}" for k, axis in enumerate("xyz")))

    # The outline of the mesh against the outline of the object in the photographs.
    spread = sorted({chosen[round(k * (len(chosen) - 1) / 3)] for k in range(4)})
    scores, columns = {}, []
    for index in chosen:
        start = time.time()
        view = load_blender(args.scene_dir, args.split, downscale, indices=[index])
        scores[index] = silhouette_iou(mesh, view)[0]
        print(f"view {index:>3d}  silhouette IoU {scores[index]:.3f}   {time.time() - start:4.1f} s")
        if index in spread:
            # for the preview, the same camera at a fixed picture width
            scale = PREVIEW_WIDTH / width
            size = (PREVIEW_WIDTH, round(height * scale))
            raster = rasterise(mesh, size[1], size[0], view.focal * scale, view.poses[0])
            photo = to_picture(view.images[0])
            photo = photo.resize(size, Image.BOX if scale <= 1 else Image.NEAREST)
            columns.append([
                photo,
                to_picture(colour_picture(mesh, raster)),
                to_picture(shape_picture(mesh, raster, view.poses[0])),
            ])
    save_preview(out_dir / "preview.png", columns)

    values = list(scores.values())
    summary = {
        "vertices": mesh.vertices.shape[0],
        "triangles": mesh.faces.shape[0],
        "pieces": kept,
        "small_pieces_removed": dropped,
        "pocket_points_filled": pockets,
        "closed_surface": closed,
        "volume": volume,
        "area": round(surface_area(mesh), 5),
        "box": {axis: [round(low[k], 4), round(high[k], 4)] for k, axis in enumerate("xyz")},
        "silhouette_iou_mean": round(sum(values) / len(values), 4),
        "silhouette_iou_worst": round(min(values), 4),
        "silhouette_iou_per_view": {str(index): round(score, 4) for index, score in scores.items()},
        "views": len(chosen),
        "views_in_split": len(names),
        "split": args.split,
        "skip": args.skip,
        "downscale": downscale,
        "width": width,
        "height": height,
        "density_max": round(density.max().item(), 3),
        "fraction_above_threshold": round(above, 6),
        "resolution": args.resolution,
        "bound": args.bound,
        "threshold": threshold,
        "threshold_fitted": args.threshold == "auto",
        "fit_views": 0 if fit_to is None else len(fit_to),
        "fit_silhouette_iou": {f"{candidate:g}": round(score, 4) for candidate, score in fitted.items()},
        "min_piece": args.min_piece,
        "checkpoint": str(args.checkpoint),
        "step": networks.step,
        "weights_sha256": weights_fingerprint(networks.coarse, networks.fine),
        "scene": args.scene_dir.resolve().name,
        "device": device,
        "torch": torch.__version__,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    print()
    print(f"silhouette against {len(chosen)} of {len(names)} {args.split} views at "
          f"{width} x {height}: IoU mean {summary['silhouette_iou_mean']:.3f}, "
          f"worst {summary['silhouette_iou_worst']:.3f}")
    print(f"results are in {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
