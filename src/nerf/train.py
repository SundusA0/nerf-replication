"""Training loop (Mildenhall et al. 2020, Section 5.3).

Each step picks one training image, picks a random batch of its pixels, renders
the rays through those pixels with the current networks, and moves the weights
to reduce the squared error against the true pixel colours (Eq. 6):

    loss = mean |C_coarse(r) - C(r)|^2  +  mean |C_fine(r) - C(r)|^2

There is no 3D supervision anywhere. The only training signal is that rendered
pixels should match photographs.
"""

from __future__ import annotations

import hashlib
import math
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import NamedTuple

import torch

from nerf.data import Views
from nerf.metrics import psnr
from nerf.model import HashNeRF, NeRF
from nerf.rays import get_rays
from nerf.render import render_image, render_rays

MODELS = ("nerf", "hash")


@dataclass
class TrainConfig:
    """Hyperparameters.

    The defaults are those of `paper_configs/blender_config.txt` in the
    authors' repository, the configuration they state reproduces the paper's
    synthetic-scene results. It differs from the text of Section 5.3 in two
    ways: 1024 rays per step where the paper says 4096, and 500k iterations
    where the paper says 100-300k.

    `model` chooses the field: "nerf", the paper's network, or "hash", the
    hash encoding and small networks of Instant-NGP (Mueller et al. 2022).
    `for_model` gives a configuration with that model's own learning-rate
    settings; everything else, in particular the sampling and the loss, is
    the same for both, so that a run with one differs from a run with the
    other in the model alone.
    """

    iterations: int = 500_000
    batch_size: int = 1024          # rays per step
    num_coarse: int = 64            # N_c
    num_fine: int = 128             # N_f
    learning_rate: float = 5e-4
    lr_decay_rate: float = 0.1      # learning rate is multiplied by this ...
    lr_decay_steps: int = 500_000   # ... every this many steps, smoothly ...
    lr_decay_start: int = 0         # ... counting from this step
    adam_beta2: float = 0.999       # PyTorch's and TensorFlow's default
    adam_eps: float = 1e-7          # TensorFlow's default, stated in the paper
    weight_decay: float = 0.0       # L2 penalty on the networks' weights (not on hash tables)
    model: str = "nerf"
    # The paper's network (model "nerf")
    depth: int = 8
    width: int = 256
    skip: int = 4
    num_freqs_pos: int = 10
    num_freqs_dir: int = 4
    # Instant-NGP's encoding and networks (model "hash"); see nerf.hashgrid
    hash_levels: int = 16
    hash_features: int = 2
    hash_log2_size: int = 19
    hash_base_resolution: int = 16
    hash_finest_resolution: int = 2048
    hash_bound: float = 1.5
    hash_width: int = 64
    # Optional warm-up from the released code, off in the paper configuration:
    # for the first `precrop_iterations` steps, sample pixels only from the
    # central `precrop_fraction` of each image, where the object is.
    precrop_iterations: int = 0
    precrop_fraction: float = 0.5
    seed: int = 0
    log_every: int = 100

    @classmethod
    def for_model(cls, model: str, **overrides) -> "TrainConfig":
        """A configuration for `model` with that model's learning-rate settings.

        "nerf" uses the paper's: 5e-4 falling smoothly by a factor of 10 over
        500k steps. "hash" uses those of Instant-NGP's NeRF configuration:
        1e-2, Adam with beta2 0.99 and epsilon 1e-15, an L2 penalty of 1e-6
        on the networks' weights, and a decay by a factor of 0.33 every 10k
        steps after the first 20k (Instant-NGP steps the rate down; here it
        falls smoothly through the same values). Keyword arguments override.
        """
        if model not in MODELS:
            raise ValueError(f"unknown model {model!r}: one of {', '.join(MODELS)}")
        settings = {"model": model}
        if model == "hash":
            settings.update(
                learning_rate=1e-2, lr_decay_rate=0.33, lr_decay_steps=10_000, lr_decay_start=20_000,
                adam_beta2=0.99, adam_eps=1e-15, weight_decay=1e-6,
            )
        settings.update(overrides)
        return cls(**settings)


def build_network(config: TrainConfig) -> torch.nn.Module:
    """A fresh, untrained field of the kind `config.model` names."""
    if config.model == "nerf":
        return NeRF(config.depth, config.width, config.skip, config.num_freqs_pos, config.num_freqs_dir)
    if config.model == "hash":
        return HashNeRF(
            config.hash_levels, config.hash_features, config.hash_log2_size,
            config.hash_base_resolution, config.hash_finest_resolution, config.hash_bound,
            config.hash_width, config.num_freqs_dir,
        )
    raise ValueError(f"unknown model {config.model!r}: one of {', '.join(MODELS)}")


def parameter_groups(networks: list[torch.nn.Module], weight_decay: float) -> list[dict]:
    """The networks' parameters for Adam: the weights of the layers, with the
    L2 penalty, and, for a hash model, the tables of the encoding without it.
    The penalty is Instant-NGP's `l2_reg`, which applies to layer weights only.
    """
    layers, tables = [], []
    for network in networks:
        for name, parameter in network.named_parameters():
            (tables if name.endswith("tables") else layers).append(parameter)
    groups = [{"params": layers, "weight_decay": weight_decay}]
    if tables:
        groups.append({"params": tables, "weight_decay": 0.0})
    return groups


@dataclass
class TrainResult:
    coarse: torch.nn.Module
    fine: torch.nn.Module
    history: list[dict] = field(default_factory=list)  # step, loss, psnr, lr, seconds


def pick_device(name: str = "auto") -> str:
    """The device to compute on: the one named or, for "auto", the best available."""
    if name != "auto":
        return name
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def learning_rate_at(step: int, config: TrainConfig) -> float:
    """Exponential decay: multiplied by `lr_decay_rate` every `lr_decay_steps`,
    smoothly, from `lr_decay_start` on; constant before that."""
    progress = max(0, step - config.lr_decay_start) / config.lr_decay_steps
    return config.learning_rate * config.lr_decay_rate**progress


def sample_ray_batch(
    views: Views,
    batch_size: int,
    generator: torch.Generator,
    crop_fraction: float | None = None,
):
    """Rays through random pixels of one random training image.

    Returns ray origins (B, 3), ray directions (B, 3) and the true colours of
    those pixels (B, 3). Pixels are drawn without replacement. With
    `crop_fraction`, they come only from the central part of the image: the
    middle `crop_fraction` of its height and of its width.
    """
    index = torch.randint(len(views), (1,), generator=generator).item()
    rays_o, rays_d = get_rays(views.height, views.width, views.focal, views.poses[index])

    height, width = views.height, views.width
    if crop_fraction is None:
        num_pixels = height * width
        chosen = torch.randperm(num_pixels, generator=generator)[: min(batch_size, num_pixels)]
    else:
        half_h = int(height // 2 * crop_fraction)
        half_w = int(width // 2 * crop_fraction)
        rows = torch.arange(height // 2 - half_h, height // 2 + half_h)
        cols = torch.arange(width // 2 - half_w, width // 2 + half_w)
        inside = (rows[:, None] * width + cols[None, :]).reshape(-1)  # flat pixel indices
        order = torch.randperm(inside.numel(), generator=generator)
        chosen = inside[order[: min(batch_size, inside.numel())]]
    chosen = chosen.to(views.images.device)
    return (
        rays_o.reshape(-1, 3)[chosen],
        rays_d.reshape(-1, 3)[chosen],
        views.images[index].reshape(-1, 3)[chosen],
    )


class StepOutput(NamedTuple):
    step: int                 # number of steps taken so far, this one included
    loss: torch.Tensor        # coarse + fine loss of this step (0-dim, detached)
    mse_fine: torch.Tensor    # the fine term alone, which gives the train PSNR
    lr: float                 # learning rate used for this step


class Trainer:
    """Everything that changes during training, and one optimisation step.

    Holds the two networks, the optimiser, the random stream and the step
    count. `state_dict` captures all of them, so a run restored from a
    checkpoint continues exactly as if it had never stopped.
    """

    def __init__(self, views: Views, config: TrainConfig, device: str = "cpu") -> None:
        torch.manual_seed(config.seed)                                # network initialisation
        self.generator = torch.Generator().manual_seed(config.seed)   # rays and sample positions
        self.config = config
        self.device = device
        self.views = views.to(device)
        self.step_count = 0

        self.coarse = build_network(config).to(device)
        self.fine = build_network(config).to(device)
        # One optimiser for both networks: Eq. 6 is a single loss over both.
        self.optimizer = torch.optim.Adam(
            parameter_groups([self.coarse, self.fine], config.weight_decay),
            lr=config.learning_rate, betas=(0.9, config.adam_beta2), eps=config.adam_eps,
        )

    def step(self) -> StepOutput:
        config, views = self.config, self.views
        lr = learning_rate_at(self.step_count, config)
        for group in self.optimizer.param_groups:
            group["lr"] = lr

        cropping = self.step_count < config.precrop_iterations
        rays_o, rays_d, target = sample_ray_batch(
            views, config.batch_size, self.generator,
            config.precrop_fraction if cropping else None,
        )
        out = render_rays(
            self.coarse, self.fine, rays_o, rays_d, views.near, views.far,
            config.num_coarse, config.num_fine,
            perturb=True, white_background=views.white_background, generator=self.generator,
        )
        loss_coarse = torch.mean((out.coarse.rgb - target) ** 2)
        loss_fine = torch.mean((out.fine.rgb - target) ** 2)
        loss = loss_coarse + loss_fine

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.step_count += 1
        return StepOutput(
            self.step_count, loss.detach(), loss_fine.detach(),
            self.optimizer.param_groups[0]["lr"],
        )

    def state_dict(self) -> dict:
        return {
            "step": self.step_count,
            "coarse": self.coarse.state_dict(),
            "fine": self.fine.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "generator": self.generator.get_state(),
        }

    def load_state_dict(self, state: dict) -> None:
        self.step_count = state["step"]
        self.coarse.load_state_dict(state["coarse"])
        self.fine.load_state_dict(state["fine"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.generator.set_state(state["generator"])


def save_checkpoint(path, trainer: Trainer, extra: dict | None = None) -> None:
    """Write the trainer's state to `path`.

    The file is written under a temporary name and then renamed, so a run
    killed in the middle of saving leaves the previous checkpoint intact.
    """
    path = Path(path)
    state = {"config": asdict(trainer.config), "extra": extra or {}, **trainer.state_dict()}
    temporary = path.with_name(path.name + ".tmp")
    torch.save(state, temporary)
    os.replace(temporary, path)


def load_checkpoint(path, trainer: Trainer) -> dict:
    """Restore `trainer` from `path`; returns the `extra` dict that was saved.

    The settings that shape the model and the sampling must match those the
    checkpoint was made with. Only the run length and the logging interval
    may differ, so that a finished run can be extended.
    """
    state = torch.load(path, map_location="cpu", weights_only=True)
    free = {"iterations", "log_every"}
    # A checkpoint from before a setting existed was made with its default.
    saved = {k: v for k, v in {**asdict(TrainConfig()), **state["config"]}.items() if k not in free}
    current = {k: v for k, v in asdict(trainer.config).items() if k not in free}
    if saved != current:
        changed = sorted(k for k in current if saved.get(k) != current[k])
        raise ValueError(f"checkpoint was made with different settings: {changed}")
    trainer.load_state_dict(state)
    return state["extra"]


class TrainedNetworks(NamedTuple):
    coarse: torch.nn.Module
    fine: torch.nn.Module
    config: TrainConfig   # the settings they were trained with
    step: int             # how many training steps they have had


def load_networks(path, device: str = "cpu") -> TrainedNetworks:
    """The two networks stored in a checkpoint, for rendering.

    `load_checkpoint` restores a `Trainer`, which needs the training images.
    This needs only the file: it rebuilds the networks from the settings saved
    with them and leaves the optimiser and the random stream behind.
    """
    state = torch.load(path, map_location="cpu", weights_only=True)
    config = TrainConfig(**state["config"])
    networks = {}
    for name in ("coarse", "fine"):
        network = build_network(config)
        network.load_state_dict(state[name])
        networks[name] = network.to(device)
    return TrainedNetworks(networks["coarse"], networks["fine"], config, state["step"])


def weights_fingerprint(*networks: torch.nn.Module) -> str:
    """A short hash of the networks' weights, to tell one trained model from another."""
    digest = hashlib.sha256()
    for network in networks:
        for name, tensor in network.state_dict().items():
            digest.update(name.encode())
            digest.update(tensor.detach().cpu().numpy().tobytes())
    return digest.hexdigest()[:16]


def train(views: Views, config: TrainConfig, device: str = "cpu", log=print) -> TrainResult:
    """Fit a coarse and a fine network to `views`, from scratch, in one go."""
    trainer = Trainer(views, config, device)
    result = TrainResult(trainer.coarse, trainer.fine)
    start = time.time()
    while trainer.step_count < config.iterations:
        out = trainer.step()
        if out.step % config.log_every == 0 or out.step == config.iterations:
            entry = {
                "step": out.step,
                "loss": out.loss.item(),
                "psnr": -10.0 * math.log10(out.mse_fine.item()),
                "lr": out.lr,
                "seconds": time.time() - start,
            }
            result.history.append(entry)
            log(
                f"step {entry['step']:>7d}  loss {entry['loss']:.5f}  "
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
