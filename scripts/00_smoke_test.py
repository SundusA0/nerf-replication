"""End-to-end check: train a small NeRF on a known scene and render new views.

    python scripts/00_smoke_test.py

Trains on 24 synthetic 48x48 images of four coloured spheres (see
`nerf.data.SphereScene`), then renders 4 viewpoints that were not in the
training set and compares them with the truth. Takes a couple of minutes on
a laptop CPU. Writes

    results/smoke_test.png    top row: true held-out views; bottom row: renders
    results/smoke_test.json   the numbers

and exits with an error if the mean held-out PSNR is below 20 dB.

The network here is deliberately small (4 layers of 64 units, 16 + 32 samples
per ray, 256 rays per step) so that this runs without a GPU. It checks that the
whole pipeline learns; it is not the paper's configuration.
"""

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import torch
from PIL import Image

from nerf.data import analytic_views
from nerf.train import TrainConfig, evaluate, psnr, train

PSNR_REQUIRED = 20.0


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
    }
    (args.out / "smoke_test.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"wrote {args.out / 'smoke_test.png'} and {args.out / 'smoke_test.json'}")

    if mean_psnr < PSNR_REQUIRED:
        print(f"FAIL: mean held-out PSNR is below {PSNR_REQUIRED} dB")
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
