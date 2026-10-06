"""Check a NeRF synthetic scene before training on it.

    python scripts/01_check_dataset.py data/nerf_synthetic/lego

Nothing is trained. The script loads the training views and tests whether the
cameras mean what this code assumes they mean:

  poses        every camera is at the same distance from the origin, looks at
               it down its own -z axis, and has world +z pointing up
  silhouettes  carving a grid with the alpha masks of all views leaves a
               visual hull (see `nerf.hull`), and the hull projects back onto
               those masks

It writes the numbers to results/dataset_check.json, and a contact sheet of
training views next to the scene directory (inside data/, which git ignores).
Exits with an error if any check fails.
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from PIL import Image

from nerf.data import load_blender
from nerf.hull import bounding_box, reprojection_iou, visual_hull

IOU_REQUIRED = 0.8
GRID_RESOLUTION = 128
GRID_BOUND = 1.5


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("scene_dir", type=Path)
    parser.add_argument("--out", type=Path, default=Path("results"))
    args = parser.parse_args()
    scene = args.scene_dir

    frames = {}
    for split in ("train", "val", "test"):
        path = scene / f"transforms_{split}.json"
        if path.exists():
            frames[split] = len(json.loads(path.read_text())["frames"])
    first = json.loads((scene / "transforms_train.json").read_text())["frames"][0]
    native_width, native_height = Image.open(scene / (first["file_path"] + ".png")).size
    print(f"scene: {scene}")
    print(f"frames: {frames}   image size: {native_width} x {native_height}")

    # About 100 pixels across is plenty for these checks and keeps them quick.
    downscale = max(1, native_width // 100)
    views = load_blender(scene, "train", downscale=downscale)
    covered = (views.alpha > 0.5).float().mean().item()
    print(f"loaded {len(views)} training views at {views.width} x {views.height} "
          f"(1/{downscale} size), focal {views.focal:.2f} px")
    print(f"object covers {100 * covered:.1f}% of the pixels; "
          f"corner pixel of view 0 is {[round(v, 3) for v in views.images[0, 0, 0].tolist()]}")

    # --- poses -------------------------------------------------------------
    positions = views.poses[:, :3, 3]
    distance = positions.norm(dim=-1)
    forward = -views.poses[:, :3, 2]                 # the camera's -z axis, in world coordinates
    to_origin = -positions / distance[:, None]
    off_axis = torch.rad2deg(torch.acos((forward * to_origin).sum(dim=-1).clamp(-1, 1)))
    up_z = views.poses[:, 2, 1]                      # world-z part of the camera's +y axis

    # --- silhouettes ---------------------------------------------------------
    occupied, axis = visual_hull(views, resolution=GRID_RESOLUTION, bound=GRID_BOUND)
    box = bounding_box(occupied, axis)
    scores = reprojection_iou(views, occupied, axis) if box else [0.0]
    mean_iou = sum(scores) / len(scores)

    print()
    print(f"camera distance from origin: {distance.min():.4f} to {distance.max():.4f}")
    print(f"angle between viewing axis and direction to origin: at most {off_axis.max():.3f} degrees")
    print(f"cameras with world +z pointing up in the image: {(up_z > 0).sum().item()} of {len(views)}")
    print(f"camera heights (z): {positions[:, 2].min():.2f} to {positions[:, 2].max():.2f}")
    print(f"visual hull: {occupied.sum().item()} of {occupied.numel()} grid points")
    if box:
        print("visual hull extent: " + "  ".join(
            f"{name} {low:+.2f} to {high:+.2f}" for name, (low, high) in zip("xyz", box)))
    print(f"hull projected back onto the silhouettes: IoU mean {mean_iou:.3f}, worst view {min(scores):.3f}")

    checks = {
        "all cameras at the same distance": (distance.max() - distance.min()).item() < 1e-3 * distance.mean().item(),
        "all cameras look at the origin down -z": off_axis.max().item() < 0.5,
        "world +z is up in every image": bool((up_z > 0).all()),
        "visual hull is not empty": box is not None,
        "visual hull lies inside the grid": box is not None and all(
            low > axis[0].item() and high < axis[-1].item() for low, high in box),
        f"hull matches silhouettes (IoU > {IOU_REQUIRED})": mean_iou > IOU_REQUIRED,
    }
    print()
    for name, passed in checks.items():
        print(f"  {'ok  ' if passed else 'FAIL'}  {name}")

    sheet = torch.cat(list(views.images[:8]), dim=1)
    sheet = Image.fromarray((sheet * 255).round().byte().numpy())
    sheet = sheet.resize((sheet.width * 2, sheet.height * 2), Image.NEAREST)
    sheet_path = scene.parent / f"{scene.name}_check.png"
    sheet.save(sheet_path)

    args.out.mkdir(parents=True, exist_ok=True)
    summary = {
        "scene": scene.name,
        "frames": frames,
        "native_image_size": [native_width, native_height],
        "checked_at_size": [views.width, views.height],
        "focal_at_checked_size": round(views.focal, 3),
        "camera_distance_min_max": [round(distance.min().item(), 4), round(distance.max().item(), 4)],
        "max_off_axis_degrees": round(off_axis.max().item(), 4),
        "hull_grid": [GRID_RESOLUTION, GRID_BOUND],
        "hull_points": int(occupied.sum().item()),
        "hull_extent_xyz": None if box is None else [[round(a, 3), round(b, 3)] for a, b in box],
        "reprojection_iou_mean": round(mean_iou, 4),
        "reprojection_iou_min": round(min(scores), 4),
        "reprojection_iou_required": IOU_REQUIRED,
        "checks": checks,
    }
    (args.out / "dataset_check.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"\nwrote {args.out / 'dataset_check.json'} and {sheet_path}")

    if not all(checks.values()):
        print("FAIL")
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
