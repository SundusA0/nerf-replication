"""Train a NeRF on a NeRF synthetic scene.

    python scripts/02_train.py data/nerf_synthetic/lego --out runs/lego --downscale 8 --iterations 2000

The defaults are the released paper configuration (`nerf.train.TrainConfig`)
at full resolution. Everything the run produces goes into the --out directory:

    config.json            the settings of the run
    log.csv                step, loss, train PSNR, learning rate, steps per second
    validation.csv         step and PSNR on one held-out view
    val_<step>.png         that view: truth on the left, render on the right
    checkpoint.pt          the latest state, overwritten as the run goes on
    checkpoint_<step>.pt   the state at every --keep-every-th step, kept
    summary.json           written when the run reaches its last step

Running the same command again resumes from checkpoint.pt. A run can therefore
be stopped and continued, or extended by raising --iterations. Ctrl-C (or a
termination signal from a job scheduler) lets the current step finish and
saves a checkpoint before exiting, so a resumed run follows exactly the same
path as an uninterrupted one.
"""

import argparse
import csv
import json
import math
import signal
import sys
import time
from dataclasses import asdict
from pathlib import Path

import torch
from PIL import Image

from nerf.data import load_blender
from nerf.metrics import psnr
from nerf.render import render_image
from nerf.train import (
    TrainConfig,
    Trainer,
    TrainResult,
    evaluate,
    load_checkpoint,
    pick_device,
    save_checkpoint,
)

LOG_COLUMNS = ["step", "loss", "train_psnr", "lr", "steps_per_second", "seconds"]
VALIDATION_COLUMNS = ["step", "psnr", "seconds"]


def start_table(path: Path, columns: list[str], keep_up_to: int) -> None:
    """Create a CSV file, or on resume drop rows written after the checkpoint."""
    rows = []
    if path.exists():
        with path.open() as handle:
            rows = [row for row in csv.DictReader(handle) if int(row["step"]) <= keep_up_to]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def add_row(path: Path, columns: list[str], values: list) -> None:
    with path.open("a", newline="") as handle:
        csv.writer(handle, lineterminator="\n").writerow(values)


def save_comparison(path: Path, truth: torch.Tensor, render: torch.Tensor) -> None:
    gap = torch.full((truth.shape[0], 2, 3), 0.75)
    pair = torch.cat([truth.cpu(), gap, render.clamp(0, 1).cpu()], dim=1)
    image = Image.fromarray((pair * 255).round().byte().numpy())
    scale = max(1, 300 // truth.shape[0])   # enlarge small images so they are easy to look at
    if scale > 1:
        image = image.resize((image.width * scale, image.height * scale), Image.NEAREST)
    image.save(path)


def main() -> int:
    defaults = TrainConfig()
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("scene_dir", type=Path)
    parser.add_argument("--out", type=Path, required=True, help="directory for this run")
    parser.add_argument("--downscale", type=int, default=1, help="shrink the images by this factor")
    parser.add_argument("--iterations", type=int, default=defaults.iterations)
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument("--num-coarse", type=int, default=defaults.num_coarse)
    parser.add_argument("--num-fine", type=int, default=defaults.num_fine)
    parser.add_argument("--depth", type=int, default=defaults.depth)
    parser.add_argument("--width", type=int, default=defaults.width)
    parser.add_argument("--skip-layer", type=int, default=defaults.skip)
    parser.add_argument("--precrop-iterations", type=int, default=defaults.precrop_iterations)
    parser.add_argument("--precrop-fraction", type=float, default=defaults.precrop_fraction)
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--device", default="auto", help="auto (default), cuda, mps or cpu")
    parser.add_argument("--log-every", type=int, default=defaults.log_every)
    parser.add_argument("--validate-every", type=int, default=1000)
    parser.add_argument("--checkpoint-every", type=int, default=1000)
    parser.add_argument("--keep-every", type=int, default=50_000,
                        help="also keep the checkpoint under a name of its own every n steps (0: never)")
    parser.add_argument("--val-skip", type=int, default=25, help="use every n-th validation view")
    args = parser.parse_args()

    config = TrainConfig(
        iterations=args.iterations, batch_size=args.batch_size,
        num_coarse=args.num_coarse, num_fine=args.num_fine,
        depth=args.depth, width=args.width, skip=args.skip_layer,
        precrop_iterations=args.precrop_iterations, precrop_fraction=args.precrop_fraction,
        seed=args.seed, log_every=args.log_every,
    )
    device = pick_device(args.device)
    args.out.mkdir(parents=True, exist_ok=True)

    # The image size is not among the settings the checkpoint carries, so it
    # is compared here: a run is continued on the images it was started on.
    checkpoint, config_file = args.out / "checkpoint.pt", args.out / "config.json"
    if checkpoint.exists() and config_file.exists():
        trained_with = json.loads(config_file.read_text()).get("downscale", args.downscale)
        if trained_with != args.downscale:
            print(f"{args.out} already holds a run, trained with --downscale {trained_with}, "
                  f"not {args.downscale}.")
            print("Use another --out directory for a run with new settings.")
            return 2

    train_views = load_blender(args.scene_dir, "train", args.downscale)
    val_views = load_blender(args.scene_dir, "val", args.downscale, skip=args.val_skip).to(device)
    print(f"{len(train_views)} training views at {train_views.width} x {train_views.height}, "
          f"{len(val_views)} validation views, device {device}, torch {torch.__version__}")

    trainer = Trainer(train_views, config, device)
    elapsed_before = 0.0
    if checkpoint.exists():
        try:
            elapsed_before = load_checkpoint(checkpoint, trainer).get("elapsed", 0.0)
        except ValueError as error:
            print(f"{args.out} already holds a run, and its {error}.")
            print("Use another --out directory for a run with new settings.")
            return 2
        print(f"resumed from {checkpoint} at step {trainer.step_count}")
    start_table(args.out / "log.csv", LOG_COLUMNS, trainer.step_count)
    start_table(args.out / "validation.csv", VALIDATION_COLUMNS, trainer.step_count)
    settings = {**asdict(config), "scene": str(args.scene_dir), "downscale": args.downscale,
                "device": device, "torch": torch.__version__}
    config_file.write_text(json.dumps(settings, indent=2) + "\n")

    session_start = time.time()

    def elapsed() -> float:
        return elapsed_before + time.time() - session_start

    def validate(step: int) -> None:
        rendered = render_image(
            trainer.coarse, trainer.fine, val_views.height, val_views.width, val_views.focal,
            val_views.poses[0], val_views.near, val_views.far, config.num_coarse, config.num_fine,
            white_background=val_views.white_background,
        ).rgb
        score = psnr(rendered, val_views.images[0])
        save_comparison(args.out / f"val_{step:06d}.png", val_views.images[0], rendered)
        add_row(args.out / "validation.csv", VALIDATION_COLUMNS, [step, f"{score:.3f}", f"{elapsed():.1f}"])
        print(f"step {step:>7d}  validation PSNR {score:5.2f} dB  -> val_{step:06d}.png")

    # A stop request must not cut a step in half: the random stream would then
    # be ahead of the weights and the resumed run would differ. So the signal
    # only raises a flag, and the loop stops at the next step boundary.
    stop = {"requested": False}

    def request_stop(signum, frame):
        if stop["requested"]:          # asked twice: give up immediately
            raise KeyboardInterrupt
        stop["requested"] = True
        print("\nstopping after the current step ...")

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    last_step, last_time = trainer.step_count, time.time()
    while trainer.step_count < config.iterations and not stop["requested"]:
        out = trainer.step()
        step = out.step
        last = step == config.iterations
        if step % config.log_every == 0 or last:
            loss, mse = out.loss.item(), out.mse_fine.item()
            now = time.time()
            rate = (step - last_step) / max(now - last_time, 1e-9)
            last_step, last_time = step, now
            remaining = (config.iterations - step) / rate
            train_psnr = -10.0 * math.log10(mse)
            add_row(args.out / "log.csv", LOG_COLUMNS, [
                step, f"{loss:.6f}", f"{train_psnr:.3f}", f"{out.lr:.3e}", f"{rate:.2f}", f"{elapsed():.1f}"])
            print(f"step {step:>7d}  loss {loss:.5f}  train PSNR {train_psnr:5.2f} dB  "
                  f"{rate:6.2f} steps/s  about {remaining / 60:.1f} min left")
        if step % args.validate_every == 0 or last:
            validate(step)
        if args.keep_every > 0 and step % args.keep_every == 0:
            # checkpoint.pt is overwritten as the run goes on. These stay, so
            # the run can be scored later at several points of its training.
            # Written first: a run killed between the two saves then passes
            # this step again and does not end up without the kept file.
            save_checkpoint(args.out / f"checkpoint_{step:06d}.pt", trainer, extra={"elapsed": elapsed()})
        if step % args.checkpoint_every == 0 or last:
            save_checkpoint(checkpoint, trainer, extra={"elapsed": elapsed()})

    if trainer.step_count < config.iterations:
        save_checkpoint(checkpoint, trainer, extra={"elapsed": elapsed()})
        print(f"stopped at step {trainer.step_count}; checkpoint saved. "
              f"Run the same command to continue.")
        return 130

    scores, _ = evaluate(val_views, TrainResult(trainer.coarse, trainer.fine), config, device)
    mean_score = sum(scores) / len(scores)
    summary = {
        "steps": trainer.step_count,
        "train_seconds": round(elapsed(), 1),
        "validation_psnr_per_view": [round(s, 2) for s in scores],
        "validation_psnr_mean": round(mean_score, 2),
        **settings,
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"finished at step {trainer.step_count}: mean PSNR on {len(scores)} validation views "
          f"{mean_score:.2f} dB, {elapsed() / 60:.1f} min of training in total")
    print(f"results are in {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
