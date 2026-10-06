import math
from pathlib import Path

import pytest
import torch

import nerf.train
from nerf.data import SphereScene, Views, analytic_views
from nerf.render import render_rays
from nerf.train import (
    TrainConfig,
    Trainer,
    TrainResult,
    evaluate,
    learning_rate_at,
    load_checkpoint,
    psnr,
    sample_ray_batch,
    save_checkpoint,
    train,
)


def quiet(line):
    pass


def tiny_config(**overrides):
    """Small enough that 100 steps take about a second on a CPU."""
    settings = dict(
        iterations=100, batch_size=128, num_coarse=8, num_fine=8,
        depth=2, width=32, skip=0, lr_decay_steps=10**9, log_every=1, seed=0,
    )
    settings.update(overrides)
    return TrainConfig(**settings)


@pytest.fixture(scope="module")
def tiny_views():
    return analytic_views(6, image_size=16, num_samples=64)


def coordinate_views(height=6, width=10):
    """One image whose red channel stores each pixel's row and green its column."""
    rows = torch.arange(height, dtype=torch.float32)[:, None].expand(height, width)
    cols = torch.arange(width, dtype=torch.float32)[None, :].expand(height, width)
    image = torch.stack([rows / height, cols / width, torch.zeros(height, width)], dim=-1)
    return Views(
        image[None], torch.eye(4)[None],
        focal=12.0, near=2.0, far=6.0, white_background=False,
    )


# --------------------------------------------------------------------------
# Pieces of the loop
# --------------------------------------------------------------------------


def test_psnr_values():
    target = torch.zeros(4, 4, 3)
    assert math.isclose(psnr(target + 0.1, target), 20.0, abs_tol=1e-4)   # MSE 1e-2
    assert math.isclose(psnr(target + 0.01, target), 40.0, abs_tol=1e-3)  # MSE 1e-4
    assert psnr(target, target) == math.inf


def test_defaults_are_the_released_paper_configuration():
    config = TrainConfig()
    assert (config.batch_size, config.num_coarse, config.num_fine) == (1024, 64, 128)
    assert (config.learning_rate, config.lr_decay_rate, config.lr_decay_steps) == (5e-4, 0.1, 500_000)
    assert (config.depth, config.width, config.skip) == (8, 256, 4)
    assert (config.num_freqs_pos, config.num_freqs_dir) == (10, 4)
    assert config.adam_eps == 1e-7


def test_learning_rate_decays_exponentially():
    config = TrainConfig()
    assert learning_rate_at(0, config) == 5e-4
    assert math.isclose(learning_rate_at(250_000, config), 5e-4 * math.sqrt(0.1))
    assert math.isclose(learning_rate_at(500_000, config), 5e-5)


def test_each_ray_is_paired_with_its_own_pixel():
    # The image is not square, so mixing up rows and columns cannot cancel out.
    views = coordinate_views()
    height, width, focal = views.height, views.width, views.focal
    rays_o, rays_d, target = sample_ray_batch(views, 32, torch.Generator().manual_seed(0))
    assert rays_o.shape == rays_d.shape == target.shape == (32, 3)

    # With an identity pose the pixel a ray passes through can be read back
    # from its direction. It must be the pixel the colour was taken from.
    cols_from_ray = rays_d[:, 0] * focal + width / 2
    rows_from_ray = -rays_d[:, 1] * focal + height / 2
    assert torch.allclose(rows_from_ray, target[:, 0] * height, atol=1e-4)
    assert torch.allclose(cols_from_ray, target[:, 1] * width, atol=1e-4)


def test_pixels_are_drawn_without_replacement_and_capped_at_the_image():
    views = coordinate_views()
    num_pixels = views.height * views.width
    _, _, target = sample_ray_batch(views, 10_000, torch.Generator().manual_seed(0))
    assert target.shape == (num_pixels, 3)
    rows = (target[:, 0] * views.height).round()
    cols = (target[:, 1] * views.width).round()
    assert (rows * views.width + cols).unique().numel() == num_pixels


def test_batches_are_reproducible():
    views = coordinate_views()
    a = sample_ray_batch(views, 16, torch.Generator().manual_seed(1))
    b = sample_ray_batch(views, 16, torch.Generator().manual_seed(1))
    c = sample_ray_batch(views, 16, torch.Generator().manual_seed(2))
    assert torch.equal(a[2], b[2])
    assert not torch.equal(a[2], c[2])


# --------------------------------------------------------------------------
# Short real training runs
# --------------------------------------------------------------------------


def test_training_reduces_the_loss(tiny_views):
    result = train(tiny_views, tiny_config(), log=quiet)
    losses = [entry["loss"] for entry in result.history]
    assert len(losses) == 100
    assert all(math.isfinite(loss) for loss in losses)
    assert sum(losses[-10:]) < 0.85 * sum(losses[:10])


def test_both_networks_are_updated(tiny_views):
    untrained = train(tiny_views, tiny_config(iterations=0), log=quiet)
    trained = train(tiny_views, tiny_config(iterations=5), log=quiet)
    for before, after in ((untrained.coarse, trained.coarse), (untrained.fine, trained.fine)):
        for (name, a), b in zip(before.named_parameters(), after.parameters()):
            assert not torch.equal(a, b), name
    # the two networks are separate, not one network used twice
    assert trained.coarse is not trained.fine


def test_two_steps_match_a_hand_written_reference(tiny_views):
    # Differential test of the update rule: the same starting networks and the
    # same random stream, stepped by hand, must end at the same weights.
    config = tiny_config(iterations=2, lr_decay_steps=10)
    trained = train(tiny_views, config, log=quiet)

    start = train(tiny_views, tiny_config(iterations=0, lr_decay_steps=10), log=quiet)
    coarse, fine = start.coarse, start.fine
    generator = torch.Generator().manual_seed(config.seed)
    optimizer = torch.optim.Adam(
        list(coarse.parameters()) + list(fine.parameters()),
        lr=config.learning_rate, eps=config.adam_eps,
    )
    for step in range(2):
        for group in optimizer.param_groups:
            group["lr"] = learning_rate_at(step, config)
        rays_o, rays_d, target = sample_ray_batch(tiny_views, config.batch_size, generator)
        out = render_rays(
            coarse, fine, rays_o, rays_d, tiny_views.near, tiny_views.far,
            config.num_coarse, config.num_fine,
            perturb=True, white_background=tiny_views.white_background, generator=generator,
        )
        loss = ((out.coarse.rgb - target) ** 2).mean() + ((out.fine.rgb - target) ** 2).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    by_hand = list(coarse.parameters()) + list(fine.parameters())
    by_loop = list(trained.coarse.parameters()) + list(trained.fine.parameters())
    for expected, actual in zip(by_hand, by_loop):
        assert torch.allclose(expected, actual, atol=1e-6)


def test_training_is_reproducible(tiny_views):
    def losses(seed):
        result = train(tiny_views, tiny_config(iterations=20, seed=seed), log=quiet)
        return [entry["loss"] for entry in result.history]

    assert losses(0) == pytest.approx(losses(0), rel=1e-4)
    assert losses(0) != pytest.approx(losses(1), rel=1e-4)


def test_the_loop_applies_the_learning_rate_schedule(tiny_views):
    config = tiny_config(iterations=10, lr_decay_steps=10)
    result = train(tiny_views, config, log=quiet)
    for entry in result.history:
        assert math.isclose(entry["lr"], learning_rate_at(entry["step"] - 1, config))
    assert result.history[-1]["lr"] < 0.2 * config.learning_rate


def test_training_jitters_samples_and_uses_the_dataset_background(tiny_views, monkeypatch):
    seen = []
    real_render_rays = nerf.train.render_rays

    def spy(*args, **kwargs):
        seen.append(kwargs)
        return real_render_rays(*args, **kwargs)

    monkeypatch.setattr(nerf.train, "render_rays", spy)
    train(tiny_views, tiny_config(iterations=2), log=quiet)
    assert len(seen) == 2
    for kwargs in seen:
        assert kwargs["perturb"] is True
        assert kwargs["white_background"] is tiny_views.white_background
        assert isinstance(kwargs["generator"], torch.Generator)


def test_evaluate_renders_every_view(tiny_views):
    config = tiny_config(iterations=3)
    result = train(tiny_views, config, log=quiet)
    scores, renders = evaluate(tiny_views, result, config)
    assert len(scores) == len(tiny_views)
    assert all(math.isfinite(score) for score in scores)
    assert renders.shape == tiny_views.images.shape
    assert renders.min() >= 0 and renders.max() <= 1 + 1e-5


def test_evaluating_the_true_scene_reproduces_the_images():
    # Put the exact analytic field where the trained networks would go. With
    # the same samples per ray as were used to make the images, evaluation
    # must give those images back. This ties together the poses, focal
    # length, bounds and background used at evaluation time.
    views = analytic_views(3, image_size=16, num_samples=64)
    truth = TrainResult(coarse=SphereScene(), fine=SphereScene())
    config = tiny_config(num_coarse=64, num_fine=0)
    scores, renders = evaluate(views, truth, config)
    assert torch.allclose(renders, views.images, atol=1e-6)
    assert min(scores) > 60


# --------------------------------------------------------------------------
# Centre-crop warm-up
# --------------------------------------------------------------------------


def test_crop_restricts_pixels_to_the_centre_and_keeps_the_pairing():
    views = coordinate_views(height=8, width=12)
    height, width, focal = views.height, views.width, views.focal
    rays_o, rays_d, target = sample_ray_batch(views, 1000, torch.Generator().manual_seed(0), 0.5)

    rows = (target[:, 0] * height).round()
    cols = (target[:, 1] * width).round()
    # the middle half of an 8 x 12 image is rows 2..5 and columns 3..8
    assert rows.min() == 2 and rows.max() == 5
    assert cols.min() == 3 and cols.max() == 8
    assert target.shape[0] == 4 * 6                       # every pixel of the crop, once
    assert (rows * width + cols).unique().numel() == 4 * 6
    # rays still belong to their pixels
    assert torch.allclose(-rays_d[:, 1] * focal + height / 2, rows, atol=1e-4)
    assert torch.allclose(rays_d[:, 0] * focal + width / 2, cols, atol=1e-4)


def test_trainer_crops_only_during_the_warm_up(tiny_views, monkeypatch):
    seen = []
    real = nerf.train.sample_ray_batch

    def spy(views, batch_size, generator, crop_fraction=None):
        seen.append(crop_fraction)
        return real(views, batch_size, generator, crop_fraction)

    monkeypatch.setattr(nerf.train, "sample_ray_batch", spy)
    trainer = Trainer(tiny_views, tiny_config(precrop_iterations=2, precrop_fraction=0.5))
    for _ in range(4):
        trainer.step()
    assert seen == [0.5, 0.5, None, None]


# --------------------------------------------------------------------------
# Checkpoints
# --------------------------------------------------------------------------


def parameters_of(trainer):
    return list(trainer.coarse.parameters()) + list(trainer.fine.parameters())


def test_resuming_from_a_checkpoint_continues_exactly(tiny_views, tmp_path):
    config = tiny_config(lr_decay_steps=10)

    straight = Trainer(tiny_views, config)
    straight_losses = [straight.step().loss.item() for _ in range(10)]

    first = Trainer(tiny_views, config)
    for _ in range(5):
        first.step()
    save_checkpoint(tmp_path / "checkpoint.pt", first, extra={"elapsed": 12.5})

    resumed = Trainer(tiny_views, config)          # a fresh process would start like this
    extra = load_checkpoint(tmp_path / "checkpoint.pt", resumed)
    assert extra == {"elapsed": 12.5}
    assert resumed.step_count == 5
    resumed_outputs = [resumed.step() for _ in range(5)]

    assert [out.step for out in resumed_outputs] == [6, 7, 8, 9, 10]
    assert [out.loss.item() for out in resumed_outputs] == pytest.approx(straight_losses[5:], rel=1e-5)
    for expected, actual in zip(parameters_of(straight), parameters_of(resumed)):
        assert torch.allclose(expected, actual, atol=1e-6)


def test_checkpoint_refuses_different_settings_but_allows_a_longer_run(tiny_views, tmp_path):
    trainer = Trainer(tiny_views, tiny_config())
    trainer.step()
    save_checkpoint(tmp_path / "checkpoint.pt", trainer)

    longer = Trainer(tiny_views, tiny_config(iterations=5000, log_every=50))
    load_checkpoint(tmp_path / "checkpoint.pt", longer)
    assert longer.step_count == 1

    for change in ({"width": 64}, {"num_fine": 16}, {"learning_rate": 1e-3}, {"seed": 1}):
        different = Trainer(tiny_views, tiny_config(**change))
        with pytest.raises(ValueError, match=next(iter(change))):
            load_checkpoint(tmp_path / "checkpoint.pt", different)


def test_a_failed_save_leaves_the_previous_checkpoint_intact(tiny_views, tmp_path, monkeypatch):
    path = tmp_path / "checkpoint.pt"
    trainer = Trainer(tiny_views, tiny_config())
    trainer.step()
    save_checkpoint(path, trainer)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["checkpoint.pt"]   # no temporary file left
    before = path.read_bytes()

    def dies_halfway(state, file):
        Path(file).write_bytes(b"half a checkpoint")
        raise KeyboardInterrupt

    trainer.step()
    monkeypatch.setattr(nerf.train.torch, "save", dies_halfway)
    with pytest.raises(KeyboardInterrupt):
        save_checkpoint(path, trainer)
    assert path.read_bytes() == before
