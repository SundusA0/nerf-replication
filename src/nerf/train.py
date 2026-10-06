"""Training loop (Mildenhall et al. 2020, Section 5.3).

Each step picks one training image, picks a random batch of its pixels, renders
the rays through those pixels with the current networks, and moves the weights
to reduce the squared error against the true pixel colours (Eq. 6):

    loss = mean |C_coarse(r) - C(r)|^2  +  mean |C_fine(r) - C(r)|^2

There is no 3D supervision anywhere. The only training signal is that rendered
pixels should match photographs.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import torch

from nerf.data import Views
from nerf.model import NeRF
from nerf.rays import get_rays
from nerf.render import render_image, render_rays


@dataclass
class TrainConfig:
    """Hyperparameters.

    The defaults are those of `paper_configs/blender_config.txt` in the
    authors' repository, the configuration they state reproduces the paper's
    synthetic-scene results. It differs from the text of Section 5.3 in two
    ways: 1024 rays per step where the paper says 4096, and 500k iterations
    where the paper says 100-300k.
    """

    iterations: int = 500_000
    batch_size: int = 1024          # rays per step
    num_coarse: int = 64            # N_c
    num_fine: int = 128             # N_f
    learning_rate: float = 5e-4
    lr_decay_rate: float = 0.1      # learning rate is multiplied by this ...
    lr_decay_steps: int = 500_000   # ... every this many steps, smoothly
    adam_eps: float = 1e-7          # TensorFlow's default, stated in the paper
    depth: int = 8
    width: int = 256
    skip: int = 4
    num_freqs_pos: int = 10
    num_freqs_dir: int = 4
    seed: int = 0
    log_every: int = 100


@dataclass
class TrainResult:
    coarse: NeRF
    fine: NeRF
    history: list[dict] = field(default_factory=list)  # step, loss, psnr, lr, seconds


def psnr(prediction: torch.Tensor, target: torch.Tensor) -> float:
    """Peak signal-to-noise ratio in dB for colours in [0, 1]: -10 log10(MSE).

    Higher is better, and every factor of 10 in the mean squared error is
    10 dB. This is the main image-quality number the paper reports.
    """
    mse = torch.mean((prediction - target) ** 2).item()
    return math.inf if mse == 0 else -10.0 * math.log10(mse)


def learning_rate_at(step: int, config: TrainConfig) -> float:
    """Exponential decay: multiplied by `lr_decay_rate` every `lr_decay_steps`."""
    return config.learning_rate * config.lr_decay_rate ** (step / config.lr_decay_steps)


def sample_ray_batch(views: Views, batch_size: int, generator: torch.Generator):
    """Rays through random pixels of one random training image.

    Returns ray origins (B, 3), ray directions (B, 3) and the true colours of
    those pixels (B, 3). Pixels are drawn without replacement.
    """
    index = torch.randint(len(views), (1,), generator=generator).item()
    rays_o, rays_d = get_rays(views.height, views.width, views.focal, views.poses[index])

    num_pixels = views.height * views.width
    chosen = torch.randperm(num_pixels, generator=generator)[: min(batch_size, num_pixels)]
    chosen = chosen.to(views.images.device)
    return (
        rays_o.reshape(-1, 3)[chosen],
        rays_d.reshape(-1, 3)[chosen],
        views.images[index].reshape(-1, 3)[chosen],
    )


def train(views: Views, config: TrainConfig, device: str = "cpu", log=print) -> TrainResult:
    """Fit a coarse and a fine network to `views`."""
    torch.manual_seed(config.seed)                              # network initialisation
    generator = torch.Generator().manual_seed(config.seed)      # rays and sample positions
    views = views.to(device)

    def new_network() -> NeRF:
        return NeRF(
            config.depth, config.width, config.skip,
            config.num_freqs_pos, config.num_freqs_dir,
        ).to(device)

    coarse, fine = new_network(), new_network()
    # One optimiser for both networks: Eq. 6 is a single loss over both.
    optimizer = torch.optim.Adam(
        list(coarse.parameters()) + list(fine.parameters()),
        lr=config.learning_rate, eps=config.adam_eps,
    )

    result = TrainResult(coarse, fine)
    start = time.time()
    for step in range(1, config.iterations + 1):
        for group in optimizer.param_groups:
            group["lr"] = learning_rate_at(step - 1, config)

        rays_o, rays_d, target = sample_ray_batch(views, config.batch_size, generator)
        out = render_rays(
            coarse, fine, rays_o, rays_d, views.near, views.far,
            config.num_coarse, config.num_fine,
            perturb=True, white_background=views.white_background, generator=generator,
        )
        loss_coarse = torch.mean((out.coarse.rgb - target) ** 2)
        loss_fine = torch.mean((out.fine.rgb - target) ** 2)
        loss = loss_coarse + loss_fine

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % config.log_every == 0 or step == config.iterations:
            entry = {
                "step": step,
                "loss": loss.item(),
                "psnr": -10.0 * math.log10(loss_fine.item()),
                "lr": optimizer.param_groups[0]["lr"],
                "seconds": time.time() - start,
            }
            result.history.append(entry)
            log(
                f"step {step:>7d}  loss {entry['loss']:.5f}  "
                f"train PSNR {entry['psnr']:5.2f} dB  {entry['seconds']:6.1f} s"
            )
    return result


def evaluate(views: Views, result: TrainResult, config: TrainConfig, device: str = "cpu"):
    """Render every view in `views` and compare with its image.

    Returns the PSNR of each view and the rendered images, shape (N, H, W, 3).
    """
    views = views.to(device)
    scores, renders = [], []
    for image, pose in zip(views.images, views.poses):
        rendered = render_image(
            result.coarse, result.fine, views.height, views.width, views.focal, pose,
            views.near, views.far, config.num_coarse, config.num_fine,
            white_background=views.white_background,
        ).rgb
        scores.append(psnr(rendered, image))
        renders.append(rendered.cpu())
    return scores, torch.stack(renders)
