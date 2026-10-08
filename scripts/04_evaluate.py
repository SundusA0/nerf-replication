"""Score a trained NeRF on held-out views: PSNR, SSIM and LPIPS.

    python scripts/04_evaluate.py data/nerf_synthetic/lego runs/lego_800px/checkpoint.pt

Renders views of the scene that training never saw and compares each render
with the true image, as in Section 6 of the paper. The paper's protocol for
its synthetic scenes is all 200 test views at 800 x 800, and each number in
its tables is the mean over those views. `--skip 8` scores every eighth view:
quicker, and an estimate, not a number to set against the paper's.

The results go into a directory next to the checkpoint, named after the
checkpoint's step and the views scored (or into --out):

    settings.json     what is being scored: the weights, the cameras, the size
    per_view.csv      one row per view: PSNR, SSIM, LPIPS, seconds to render
    render_<n>.png    the rendered views
    comparison.png    four of the views: truth on top, render below
    summary.json      the means, and what they were measured on

A row is written as each view finishes. Running the same command again
carries on with the views that are missing, so a long evaluation can be
stopped with Ctrl-C and continued later.

The images are shrunk as they were for training, which is read from the
config.json next to the checkpoint. LPIPS needs the `lpips` package and the
ImageNet weights of VGG-16 (528 MB), which are looked for in data/torch_hub
and downloaded there if missing. --no-lpips leaves that measure out.
"""

import argparse
import csv
import hashlib
import io
import json
import os
import sys
import time
from pathlib import Path

import torch
from PIL import Image

from nerf.data import blender_frames, load_blender
from nerf.metrics import Lpips, psnr, ssim
from nerf.render import render_image
from nerf.train import load_networks, pick_device

REPO = Path(__file__).resolve().parents[1]
COLUMNS = ["view", "name", "psnr", "ssim", "lpips", "seconds"]

# Mildenhall et al. 2020, Table 4: results per synthetic scene, each the mean
# over the scene's 200 test views at 800 x 800.
PAPER = {"lego": {"psnr": 32.54, "ssim": 0.961, "lpips": 0.050}}
PAPER_VIEWS, PAPER_SIZE = 200, 800

VGG16_URL = "https://download.pytorch.org/models/vgg16-397923af.pth"


def protocol_differences(split: str, views: int, views_in_split: int, height: int, width: int) -> list[str]:
    """In which ways a set of scored views differs from the paper's protocol.

    An empty list means all 200 test views at 800 x 800, the only case in
    which the means may be set next to the paper's.
    """
    differences = []
    if split != "test":
        differences.append(f"the {split} views, not the test views")
    if views != views_in_split:
        differences.append(f"{views} of the {views_in_split} {split} views")
    elif views != PAPER_VIEWS:
        differences.append(f"{views} views, not {PAPER_VIEWS}")
    if (height, width) != (PAPER_SIZE, PAPER_SIZE):
        differences.append(f"{width} x {height} images, not {PAPER_SIZE} x {PAPER_SIZE}")
    return differences


def weights_fingerprint(*networks: torch.nn.Module) -> str:
    """A short hash of the networks' weights, to tell one trained model from another."""
    digest = hashlib.sha256()
    for network in networks:
        for name, tensor in network.state_dict().items():
            digest.update(name.encode())
            digest.update(tensor.detach().cpu().numpy().tobytes())
    return digest.hexdigest()[:16]


def file_fingerprint(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def finished_rows(path: Path, out_dir: Path) -> list[dict]:
    """The finished views of an earlier session, if there was one.

    A view counts as finished if its row is complete and its render exists.
    Anything else in the file is left out, such as a row cut short when the
    program was killed, or a second row for the same view.
    """
    rows, seen = [], set()
    if path.exists():
        with path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                try:
                    row = {column: row[column] for column in COLUMNS}
                    index = int(row["view"])
                    float(row["psnr"]), float(row["ssim"]), float(row["seconds"])
                except (KeyError, TypeError, ValueError):
                    continue
                if index not in seen and render_path(out_dir, index).exists():
                    seen.add(index)
                    rows.append(row)
    return rows


def write_atomically(path: Path, text: str) -> None:
    """Replace a file in one step, so that being interrupted cannot leave half of it."""
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text)
    os.replace(temporary, path)


def write_rows(path: Path, rows: list[dict]) -> None:
    text = io.StringIO()
    writer = csv.DictWriter(text, fieldnames=COLUMNS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    write_atomically(path, text.getvalue())


def add_row(path: Path, row: dict) -> None:
    with path.open("a", newline="") as handle:
        csv.DictWriter(handle, fieldnames=COLUMNS, lineterminator="\n").writerow(row)


def render_path(out_dir: Path, index: int) -> Path:
    return out_dir / f"render_{index:03d}.png"


def to_picture(image: torch.Tensor) -> Image.Image:
    """An (H, W, 3) tensor with colours in [0, 1] as an 8-bit image."""
    return Image.fromarray((image.clamp(0, 1) * 255).round().byte().numpy())


def tile(image: Image.Image, target: int = 400) -> Image.Image:
    """Shrink or enlarge an image by a whole factor, to about `target` pixels high."""
    if image.height >= 2 * target:
        return image.reduce(image.height // target)
    factor = target // image.height
    if factor > 1:
        return image.resize((image.width * factor, image.height * factor), Image.NEAREST)
    return image


def save_comparison(path: Path, truths: list[Image.Image], renders: list[Image.Image]) -> None:
    """Views side by side: the true images in the top row, the renders below."""
    gap = 4
    top = [tile(image) for image in truths]
    bottom = [tile(image) for image in renders]
    width, height = top[0].size
    canvas = Image.new(
        "RGB", (len(top) * (width + gap) + gap, 2 * (height + gap) + gap), (191, 191, 191)
    )
    for column, (truth, render) in enumerate(zip(top, bottom)):
        left = gap + column * (width + gap)
        canvas.paste(truth, (left, gap))
        canvas.paste(render, (left, 2 * gap + height))
    canvas.save(path)


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def positive(text: str) -> int:
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError("must be 1 or more")
    return value


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("scene_dir", type=Path)
    parser.add_argument("checkpoint", type=Path, help="a checkpoint file of a training run")
    parser.add_argument("--split", default="test", choices=("train", "val", "test"))
    parser.add_argument("--skip", type=positive, default=1, help="score every n-th view only")
    parser.add_argument("--downscale", type=positive, default=None,
                        help="shrink the images by this factor (default: as in training)")
    parser.add_argument("--device", default="auto", help="auto (default), cuda, mps or cpu")
    parser.add_argument("--out", type=Path, default=None, help="directory for the results")
    parser.add_argument("--no-lpips", action="store_true", help="score PSNR and SSIM only")
    parser.add_argument("--weights-dir", type=Path, default=REPO / "data" / "torch_hub",
                        help="where the VGG-16 weights for LPIPS are kept")
    parser.add_argument("--chunk", type=positive, default=4096, help="rays rendered at a time")
    args = parser.parse_args(argv)

    if not args.checkpoint.is_file():
        print(f"There is no checkpoint at {args.checkpoint}.")
        return 2
    run_dir = args.checkpoint.parent
    downscale = args.downscale
    if downscale is None:
        config_file = run_dir / "config.json"
        if not config_file.exists():
            print(f"There is no config.json next to {args.checkpoint} to say how much the images "
                  f"were shrunk for training. Say it with --downscale (1 for full size).")
            return 2
        downscale = json.loads(config_file.read_text())["downscale"]

    device = pick_device(args.device)
    networks = load_networks(args.checkpoint, device)
    config = networks.config
    names = blender_frames(args.scene_dir, args.split)
    chosen = list(range(0, len(names), args.skip))

    out_dir = args.out
    if out_dir is None:
        every = f"_every{args.skip}" if args.skip > 1 else ""
        out_dir = run_dir / f"eval_{networks.step:06d}_{args.split}{every}"
    if out_dir.resolve() == run_dir.resolve() or (out_dir / "checkpoint.pt").exists():
        print(f"{out_dir} is the directory of a training run. Give --out a directory of its own.")
        return 2
    settings = {
        "weights_sha256": weights_fingerprint(networks.coarse, networks.fine),
        "step": networks.step,
        "scene": args.scene_dir.resolve().name,
        # the file that lists the split's images and their cameras
        "cameras_sha256": file_fingerprint(args.scene_dir / f"transforms_{args.split}.json"),
        "split": args.split,
        "skip": args.skip,
        "downscale": downscale,
        "with_lpips": not args.no_lpips,
    }

    # Scores already in the directory are continued only if they are scores
    # of exactly the same thing. A directory without scores can be reused.
    settings_file, table = out_dir / "settings.json", out_dir / "per_view.csv"
    rows = finished_rows(table, out_dir)
    if rows:
        try:
            earlier = json.loads(settings_file.read_text())
        except (OSError, ValueError):
            earlier = None
        if not isinstance(earlier, dict):   # missing, or not what this script writes
            earlier = None
        if earlier is None:
            print(f"{out_dir} holds scores, but no settings.json that says what they are scores of.")
        elif earlier != settings:
            changed = sorted(key for key in settings if earlier.get(key) != settings[key])
            print(f"{out_dir} holds scores for something else (different: {', '.join(changed)}).")
        if earlier != settings:
            print("Use another --out directory, or delete that one.")
            return 2
    out_dir.mkdir(parents=True, exist_ok=True)
    write_atomically(settings_file, json.dumps(settings, indent=2) + "\n")
    write_rows(table, rows)

    done = {int(row["view"]) for row in rows}
    todo = [index for index in chosen if index not in done]
    print(f"checkpoint at step {networks.step}, {len(chosen)} of {len(names)} {args.split} views, "
          f"device {device}, torch {torch.__version__}")
    if done and todo:
        print(f"{len(done)} views were scored earlier; continuing with the other {len(todo)}")

    lpips_of = None
    if todo and not args.no_lpips:
        try:
            # The feature network runs on the CPU unless there is an NVIDIA GPU:
            # one image pair takes it a few seconds, the render takes minutes.
            lpips_of = Lpips("cuda" if device.startswith("cuda") else "cpu", args.weights_dir)
        except Exception as error:  # package missing, no network, damaged download
            folder = args.weights_dir / "checkpoints"
            print(f"LPIPS could not be set up: {type(error).__name__}: {error}")
            print("It needs the lpips package (pip install lpips) and the VGG-16 weights, which")
            print(f"are looked for in {folder}. To fetch them by hand, removing a damaged file first:")
            print(f"    mkdir -p {folder}")
            print(f"    curl -L -o {folder / Path(VGG16_URL).name} {VGG16_URL}")
            print("--no-lpips scores PSNR and SSIM only.")
            return 2

    session_start, scored_now = time.time(), 0
    try:
        for index in todo:
            view = load_blender(args.scene_dir, args.split, downscale, indices=[index])
            truth = view.images[0]
            start = time.time()
            rendered = render_image(
                networks.coarse, networks.fine, view.height, view.width, view.focal,
                view.poses[0].to(device), view.near, view.far, config.num_coarse, config.num_fine,
                white_background=view.white_background, chunk=args.chunk,
            ).rgb.clamp(0.0, 1.0).cpu()
            seconds = time.time() - start

            # Scores are taken from the render as computed, before it is
            # rounded to 8 bits for the file.
            row = {
                "view": index,
                "name": names[index],
                "psnr": f"{psnr(rendered, truth):.4f}",
                "ssim": f"{ssim(rendered, truth):.6f}",
                "lpips": "" if lpips_of is None else f"{lpips_of(rendered, truth):.6f}",
                "seconds": f"{seconds:.1f}",
            }
            to_picture(rendered).save(render_path(out_dir, index))
            add_row(table, row)
            rows.append(row)

            scored_now += 1
            left = (len(todo) - scored_now) * (time.time() - session_start) / scored_now
            lpips_text = "" if lpips_of is None else f"  LPIPS {float(row['lpips']):.3f}"
            print(f"view {index:>3d}  PSNR {float(row['psnr']):5.2f} dB  SSIM {float(row['ssim']):6.3f}"
                  f"{lpips_text}  {seconds:5.1f} s   {len(rows)} of {len(chosen)} done, "
                  f"about {left / 60:.0f} min left")
    except KeyboardInterrupt:
        print(f"\nstopped with {len(rows)} of {len(chosen)} views scored. "
              f"Run the same command to continue.")
        return 130

    rows.sort(key=lambda row: int(row["view"]))
    write_rows(table, rows)   # in view order, however many sessions it took
    # Four views spread over the ones scored, for the picture and the image size.
    spread = sorted({chosen[round(k * (len(chosen) - 1) / 3)] for k in range(4)})
    truths = [
        load_blender(args.scene_dir, args.split, downscale, indices=[index]).images[0]
        for index in spread
    ]
    height, width = truths[0].shape[:2]
    save_comparison(
        out_dir / "comparison.png",
        [to_picture(truth) for truth in truths],
        [Image.open(render_path(out_dir, index)).convert("RGB") for index in spread],
    )

    differences = protocol_differences(args.split, len(rows), len(names), height, width)
    paper = PAPER.get(settings["scene"])
    summary = {
        "psnr": round(mean([float(row["psnr"]) for row in rows]), 3),
        "ssim": round(mean([float(row["ssim"]) for row in rows]), 4),
        "lpips": None if args.no_lpips else round(mean([float(row["lpips"]) for row in rows]), 4),
        "lpips_network": None if args.no_lpips else "vgg",
        "views": len(rows),
        "views_in_split": len(names),
        "width": width,
        "height": height,
        "paper": paper,
        "paper_protocol": not differences,
        "differences_from_paper_protocol": differences,
        "checkpoint": str(args.checkpoint),
        **settings,
        "num_coarse": config.num_coarse,
        "num_fine": config.num_fine,
        "render_seconds_per_view": round(mean([float(row["seconds"]) for row in rows]), 1),
        "device": device,
        "torch": torch.__version__,
    }
    write_atomically(out_dir / "summary.json", json.dumps(summary, indent=2) + "\n")

    def line(label: str, scores: dict, note: str = "") -> str:
        lpips_text = "       -" if scores["lpips"] is None else f"{scores['lpips']:8.3f}"
        return f"{label:10s}{scores['psnr']:8.2f}{scores['ssim']:8.3f}{lpips_text}{note}"

    print()
    print(f"{len(rows)} of {len(names)} {args.split} views at {width} x {height}, "
          f"checkpoint at step {networks.step}")
    print()
    print(f"{'':10s}{'PSNR':>8s}{'SSIM':>8s}{'LPIPS':>8s}")
    print(line("this run", summary))
    if paper and not differences:
        print(line("paper", paper, f"   Mildenhall et al. 2020, Table 4, {settings['scene']}"))
    if differences:
        print()
        print(f"Not the paper's protocol: {'; '.join(differences)}.")
    if differences and paper:
        print(f"The paper's numbers for this scene (PSNR {paper['psnr']:.2f}, SSIM {paper['ssim']:.3f}, "
              f"LPIPS {paper['lpips']:.3f})")
        print(f"are for all {PAPER_VIEWS} test views at {PAPER_SIZE} x {PAPER_SIZE} and cannot be "
              f"set against these.")
    print(f"results are in {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
