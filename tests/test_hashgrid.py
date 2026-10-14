"""The hash encoding, the model built on it, and training with that model."""

import json
import math
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from nerf.data import analytic_views
from nerf.hashgrid import (
    CORNERS,
    HASH_PRIMES,
    HashGridEncoding,
    level_resolutions,
    truncated_exp,
    vertex_indices,
)
from nerf.mesh import density_grid, extract_surface
from nerf.model import HashNeRF, NeRF
from nerf.render import render_image
from nerf.train import (
    TrainConfig,
    Trainer,
    build_network,
    learning_rate_at,
    load_checkpoint,
    load_networks,
    parameter_groups,
    save_checkpoint,
    train,
)
from test_data import write_blender_scene

ROOT = Path(__file__).resolve().parents[1]
ENV = dict(os.environ, PYTHONPATH=str(ROOT / "src"))


def quiet(line):
    pass


def small_encoding(**overrides):
    settings = dict(num_levels=4, features_per_level=2, log2_table_size=10, base_resolution=4,
                    finest_resolution=32, bound=1.0)
    settings.update(overrides)
    return HashGridEncoding(**settings)


# --------------------------------------------------------------------------
# Levels and table entries
# --------------------------------------------------------------------------


def test_resolutions_grow_geometrically_from_the_coarsest_to_the_finest():
    # The defaults: 16 levels from 16 to 2048 cells per axis,
    # a growth factor of 1.38 per level.
    resolutions = level_resolutions(16, 16, 2048)
    assert resolutions[0] == 16 and resolutions[-1] == 2048 and len(resolutions) == 16
    growth = (2048 / 16) ** (1 / 15)
    assert math.isclose(growth, 1.3819, abs_tol=1e-4)
    assert resolutions == [math.floor(16 * growth**level + 1e-6) for level in range(16)]
    assert resolutions == sorted(resolutions) and len(set(resolutions)) == 16
    # one level is just the base; a doubling per level is the other common choice
    assert level_resolutions(1, 16, 2048) == [16]
    assert level_resolutions(4, 16, 128) == [16, 32, 64, 128]
    with pytest.raises(ValueError):
        level_resolutions(3, 16, 8)


def test_coarse_levels_are_indexed_directly_and_fine_levels_by_the_papers_hash():
    table_size = 2**10
    # 7 cells per axis: 8^3 = 512 vertices fit in 1024 entries, so each
    # vertex gets its own entry, in x-major order.
    vertices = torch.tensor([[0, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0], [7, 7, 7]])
    assert vertex_indices(vertices, 7, table_size).tolist() == [0, 1, 8, 64, 511]
    assert vertex_indices(vertices, 7, 512).tolist() == [0, 1, 8, 64, 511]      # exactly full is still direct
    # 15 cells per axis: 16^3 = 4096 vertices do not fit, so Eq. 4 of the
    # paper applies: xor of the coordinates times the primes, modulo T.
    vertices = torch.tensor([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [3, 14, 9]])
    expected = [((x * HASH_PRIMES[0]) ^ (y * HASH_PRIMES[1]) ^ (z * HASH_PRIMES[2])) % table_size
                for x, y, z in vertices.tolist()]
    assert vertex_indices(vertices, 15, table_size).tolist() == expected
    assert expected[0] == 0 and expected[1] == 1          # the first prime is 1
    assert expected[2] == 2654435761 % table_size
    assert expected[3] == 805459861 % table_size
    # the primes are the paper's
    assert HASH_PRIMES == (1, 2654435761, 805459861)


def test_the_encodings_entries_are_those_of_the_vertex_function():
    # The module computes the eight corners of every level at once. Checked
    # against the one-vertex-at-a-time function above, and against the
    # weights of plain trilinear interpolation, for a mix of direct and
    # hashed levels.
    torch.manual_seed(0)
    encoding = small_encoding(num_levels=5, finest_resolution=64, log2_table_size=12)
    assert encoding.direct.tolist() == [True, True, False, False, False]   # 5^3 and 9^3 fit in 4096 entries; 17^3 = 4913 does not
    points = torch.rand(200, 3) * 2.0 - 1.0
    with torch.no_grad():
        encoded, _ = encoding(points)
        expected = []
        for level, resolution in enumerate(encoding.resolutions):
            scaled = (points + 1.0) / 2.0 * resolution
            lower = scaled.floor().clamp(max=resolution - 1)
            fraction = scaled - lower
            corners = lower.long()[:, None, :] + CORNERS
            entries = vertex_indices(corners, resolution, encoding.table_size)
            upper = CORNERS.float()
            weights = (upper * fraction[:, None, :] + (1 - upper) * (1 - fraction[:, None, :])).prod(-1)
            expected.append((weights[..., None] * encoding.tables[level][entries]).sum(-2))
    assert torch.allclose(encoded, torch.cat(expected, dim=-1), atol=1e-7)


# --------------------------------------------------------------------------
# Interpolation
# --------------------------------------------------------------------------


def test_a_point_on_a_vertex_gets_that_vertexs_features_and_in_between_their_blend():
    encoding = small_encoding(num_levels=1, base_resolution=4, finest_resolution=4)
    with torch.no_grad():
        encoding.tables.zero_()
        # one level of 4 cells per axis on [-1, 1]: vertex (1, 2, 3) is at (-0.5, 0, 0.5)
        encoding.tables[0, vertex_indices(torch.tensor([1, 2, 3]), 4, encoding.table_size)] = torch.tensor([2.0, -1.0])
        encoding.tables[0, vertex_indices(torch.tensor([2, 2, 3]), 4, encoding.table_size)] = torch.tensor([4.0, 1.0])
        on_vertex, _ = encoding(torch.tensor([[-0.5, 0.0, 0.5]]))
        quarter_way, _ = encoding(torch.tensor([[-0.375, 0.0, 0.5]]))   # a quarter of a cell towards (2, 2, 3)
        elsewhere, _ = encoding(torch.tensor([[0.5, -0.5, -0.5]]))
    assert torch.allclose(on_vertex, torch.tensor([[2.0, -1.0]]))
    assert torch.allclose(quarter_way, torch.tensor([[2.5, -0.5]]))
    assert torch.equal(elsewhere, torch.zeros(1, 2))


def test_interpolation_reproduces_a_linear_field_exactly():
    # Trilinear interpolation is exact for functions linear in each
    # coordinate. Store f(v) = x + 2y + 3z at every vertex of a directly
    # indexed level and read it back anywhere.
    encoding = small_encoding(num_levels=1, base_resolution=8, finest_resolution=8, log2_table_size=10)
    resolution = 8
    grid = torch.stack(torch.meshgrid(*[torch.arange(resolution + 1)] * 3, indexing="ij"), dim=-1).reshape(-1, 3)
    positions = grid.float() / resolution * 2.0 - 1.0                 # vertex coordinates on [-1, 1]
    value = positions @ torch.tensor([1.0, 2.0, 3.0])
    with torch.no_grad():
        encoding.tables[0, vertex_indices(grid, resolution, encoding.table_size), 0] = value
        encoding.tables[0, :, 1] = 0.0
    points = torch.rand(500, 3) * 2.0 - 1.0
    encoded, inside = encoding(points)
    assert inside.all()
    assert torch.allclose(encoded[:, 0], points @ torch.tensor([1.0, 2.0, 3.0]), atol=1e-5)
    assert torch.equal(encoded[:, 1], torch.zeros(500))
    # the box's far corner lands in the last cell, not beyond it
    corner, inside = encoding(torch.tensor([[1.0, 1.0, 1.0]]))
    assert inside.all() and math.isclose(corner[0, 0].item(), 6.0, abs_tol=1e-5)
    # also when the level's vertices fill its table exactly, where a cell
    # beyond the last would index past the table's end
    exactly_full = HashGridEncoding(num_levels=1, base_resolution=7, finest_resolution=7, log2_table_size=9, bound=1.0)
    with torch.no_grad():
        exactly_full.tables[0, 511] = torch.tensor([3.0, 4.0])          # vertex (7, 7, 7) is the last entry
    corner, inside = exactly_full(torch.tensor([[1.0, 1.0, 1.0]]))
    assert inside.all() and torch.allclose(corner, torch.tensor([[3.0, 4.0]]))


def test_points_outside_the_box_are_flagged_and_encoded_as_if_clamped():
    encoding = small_encoding(bound=1.0)
    torch.manual_seed(0)
    points = torch.tensor([[0.0, 0.0, 0.0], [1.0, -1.0, 0.5], [1.5, 0.0, 0.0], [0.0, 0.0, -1.0001]])
    encoded, inside = encoding(points)
    assert inside.tolist() == [True, True, False, False]
    clamped, _ = encoding(points.clamp(-1.0, 1.0))
    assert torch.equal(encoded, clamped)


def test_gradient_reaches_only_the_eight_corners_with_the_interpolation_weights():
    encoding = small_encoding(num_levels=1, base_resolution=4, finest_resolution=4)
    point = torch.tensor([[-0.5 + 0.1, 0.0 + 0.05, 0.5 + 0.2]])     # in the cell at vertex (1, 2, 3)
    encoded, _ = encoding(point)
    encoded[0, 0].backward()
    grad = encoding.tables.grad[0]                                   # (T, F)
    touched = (grad != 0).any(dim=1).nonzero().flatten().tolist()
    corners = torch.tensor([1, 2, 3]) + CORNERS
    assert sorted(touched) == sorted(vertex_indices(corners, 4, encoding.table_size).tolist())
    # cell of width 0.5, so the fractions are 0.2, 0.1 and 0.4
    fraction = torch.tensor([0.2, 0.1, 0.4])
    upper = CORNERS.float()
    weights = (upper * fraction + (1 - upper) * (1 - fraction)).prod(-1)
    assert torch.allclose(grad[vertex_indices(corners, 4, encoding.table_size), 0], weights, atol=1e-6)
    assert torch.equal(grad[:, 1], torch.zeros(encoding.table_size))   # only feature 0 was summed
    assert math.isclose(weights.sum().item(), 1.0, abs_tol=1e-6)


def test_tables_start_small_and_the_output_has_a_feature_per_level():
    torch.manual_seed(0)
    encoding = HashGridEncoding()
    assert encoding.tables.shape == (16, 2**19, 2)
    assert encoding.tables.abs().max() <= 1e-4 and encoding.tables.abs().max() > 0.5e-4
    assert encoding.output_dim == 32
    encoded, _ = encoding(torch.zeros(3, 5, 3))
    assert encoded.shape == (3, 5, 32)
    with pytest.raises(ValueError):
        HashGridEncoding(bound=0.0)


# --------------------------------------------------------------------------
# Truncated exp
# --------------------------------------------------------------------------


def test_truncated_exp_is_exp_below_the_cut_and_keeps_a_gradient_above_it():
    x = torch.tensor([-3.0, 0.0, 2.0, 15.0, 40.0], requires_grad=True)
    y = truncated_exp(x)
    assert torch.allclose(y, torch.exp(x.detach().clamp(max=15.0)))
    y.sum().backward()
    assert torch.allclose(x.grad[:4], torch.exp(x.detach()[:4]))
    assert math.isclose(x.grad[4].item(), math.exp(15.0), rel_tol=1e-6)   # exp(clamp(x)) would give 0 here
    assert torch.isfinite(y).all() and torch.isfinite(x.grad).all()


# --------------------------------------------------------------------------
# The model
# --------------------------------------------------------------------------


def small_model():
    return HashNeRF(num_levels=4, log2_table_size=10, base_resolution=4, finest_resolution=32, bound=1.0, width=16)


def random_inputs(*shape):
    points = torch.randn(*shape, 3) * 0.4
    dirs = torch.randn(*shape, 3)
    return points, dirs / dirs.norm(dim=-1, keepdim=True)


def test_instant_ngp_architecture():
    model = HashNeRF()
    layers = [m for m in model.density_net if isinstance(m, torch.nn.Linear)]
    assert [(m.in_features, m.out_features) for m in layers] == [(32, 64), (64, 16)]
    layers = [m for m in model.colour_net if isinstance(m, torch.nn.Linear)]
    assert [(m.in_features, m.out_features) for m in layers] == [(16 + 27, 64), (64, 64), (64, 3)]
    # 16 tables of 2^19 entries of 2 features, and the two small networks
    assert sum(p.numel() for p in model.parameters()) == 16 * 2**19 * 2 + (32 * 64 + 64) + (64 * 16 + 16) \
        + (43 * 64 + 64) + (64 * 64 + 64) + (64 * 3 + 3)


def test_output_shapes_and_ranges():
    torch.manual_seed(0)
    points, dirs = random_inputs(6, 10)
    sigma, rgb = small_model()(points, dirs)
    assert sigma.shape == (6, 10) and rgb.shape == (6, 10, 3)
    assert (sigma >= 0).all() and torch.isfinite(sigma).all()
    assert (rgb >= 0).all() and (rgb <= 1).all()


def test_density_ignores_view_direction_but_colour_does_not():
    torch.manual_seed(0)
    model = small_model()
    points, dirs_a = random_inputs(200)
    _, dirs_b = random_inputs(200)
    sigma_a, rgb_a = model(points, dirs_a)
    sigma_b, rgb_b = model(points, dirs_b)
    assert torch.equal(sigma_a, sigma_b)
    assert not torch.allclose(rgb_a, rgb_b)
    # and colour depends on the position as well, through the density network's
    # features (once the tables hold something: they start near zero)
    with torch.no_grad():
        model.encoding.tables.mul_(1000.0)
    _, rgb_c = model(points, dirs_a[:1].expand(200, 3))
    assert rgb_c.std(dim=0).max() > 1e-3


def test_points_outside_the_box_have_no_density():
    torch.manual_seed(0)
    model = small_model()
    points = torch.tensor([[0.0, 0.0, 0.0], [0.9, 0.9, 0.9], [1.1, 0.0, 0.0], [0.0, -2.0, 0.0]])
    dirs = torch.tensor([[0.0, 0.0, 1.0]]).expand(4, 3)
    sigma, _ = model(points, dirs)
    assert (sigma[:2] > 0).all()
    assert torch.equal(sigma[2:], torch.zeros(2))


def test_every_parameter_is_connected_to_the_output():
    torch.manual_seed(0)
    model = small_model()
    points, dirs = random_inputs(500)
    sigma, rgb = model(points, dirs)
    (sigma.sum() + rgb.sum()).backward()
    for name, param in model.named_parameters():
        assert param.grad is not None, name
        assert torch.isfinite(param.grad).all(), name
        assert param.grad.abs().sum() > 0, name


def test_initialisation_matches_nerfs_and_same_seed_gives_same_network():
    torch.manual_seed(0)
    model = HashNeRF(log2_table_size=10)
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            bound = (6.0 / (module.in_features + module.out_features)) ** 0.5
            assert module.weight.abs().max() <= bound, name
            assert torch.equal(module.bias, torch.zeros_like(module.bias)), name
    torch.manual_seed(3)
    a = small_model()
    torch.manual_seed(3)
    b = small_model()
    points, dirs = random_inputs(20)
    assert all(torch.equal(x, y) for x, y in zip(a(points, dirs), b(points, dirs)))


def test_the_density_can_be_sampled_on_a_grid_and_meshed_like_nerfs():
    # The mesh script samples the fine network with no viewing direction.
    torch.manual_seed(0)
    model = small_model()
    with torch.no_grad():
        model.density_net[-1].bias[0] = 2.0        # density e^2 everywhere inside the box
    density = density_grid(model, 1.0, 17)
    assert density.shape == (17, 17, 17)
    assert torch.allclose(density, torch.full_like(density, math.exp(2.0)), rtol=1e-3)
    # and a field that fades out at the box's edge gives a closed surface
    density[0], density[-1], density[:, 0], density[:, -1], density[:, :, 0], density[:, :, -1] = (0.0,) * 6
    mesh = extract_surface(density, 1.0, 1.0)
    assert mesh.faces.shape[0] > 0


# --------------------------------------------------------------------------
# Training with the hash model
# --------------------------------------------------------------------------


def hash_config(**overrides):
    settings = dict(
        iterations=30, batch_size=128, num_coarse=8, num_fine=8, log_every=1, seed=0,
        hash_levels=4, hash_log2_size=12, hash_base_resolution=4, hash_finest_resolution=32, hash_width=16,
        hash_bound=1.6,     # not the default, and the whole scene is inside
    )
    settings.update(overrides)
    return TrainConfig.for_model("hash", **settings)


@pytest.fixture(scope="module")
def tiny_views():
    return analytic_views(6, image_size=16, num_samples=64)


def test_each_model_has_its_own_learning_rate_settings_and_the_rest_is_shared():
    nerf = TrainConfig.for_model("nerf")
    assert nerf == TrainConfig()
    hashed = TrainConfig.for_model("hash")
    assert hashed.model == "hash"
    assert (hashed.learning_rate, hashed.lr_decay_rate, hashed.lr_decay_steps, hashed.lr_decay_start) == (1e-2, 0.33, 10_000, 20_000)
    assert (hashed.adam_beta2, hashed.adam_eps, hashed.weight_decay) == (0.99, 1e-15, 1e-6)
    for name in ("batch_size", "num_coarse", "num_fine", "num_freqs_dir", "seed"):
        assert getattr(hashed, name) == getattr(nerf, name), name
    assert TrainConfig.for_model("hash", learning_rate=3e-3).learning_rate == 3e-3
    with pytest.raises(ValueError, match="unknown model"):
        TrainConfig.for_model("voxels")
    # the rate holds for the first 20k steps, then falls by 0.33 every 10k
    assert learning_rate_at(0, hashed) == 1e-2 and learning_rate_at(20_000, hashed) == 1e-2
    assert math.isclose(learning_rate_at(30_000, hashed), 0.33e-2)
    assert math.isclose(learning_rate_at(40_000, hashed), 0.33**2 * 1e-2)


def test_build_network_follows_the_configuration():
    assert isinstance(build_network(TrainConfig()), NeRF)
    config = hash_config()
    network = build_network(config)
    assert isinstance(network, HashNeRF)
    assert network.encoding.tables.shape == (4, 2**12, 2)
    assert network.encoding.resolutions == [4, 8, 16, 32]
    assert network.encoding.bound == 1.6 and network.density_net[0].out_features == 16
    assert network.num_freqs_dir == config.num_freqs_dir
    with pytest.raises(ValueError, match="unknown model"):
        build_network(TrainConfig(model="voxels"))


def test_the_l2_penalty_applies_to_layer_weights_and_not_to_the_tables():
    network = build_network(hash_config())
    groups = parameter_groups([network, build_network(hash_config())], 1e-6)
    assert [group["weight_decay"] for group in groups] == [1e-6, 0.0]
    assert len(groups[1]["params"]) == 2                         # the two networks' tables
    assert all(p.shape == (4, 2**12, 2) for p in groups[1]["params"])
    counted = sum(p.numel() for group in groups for p in group["params"])
    assert counted == 2 * sum(p.numel() for p in network.parameters())
    # a NeRF has no tables and gets one group
    groups = parameter_groups([NeRF(depth=2, width=8, skip=0)], 0.0)
    assert len(groups) == 1 and groups[0]["weight_decay"] == 0.0
    # the trainer's Adam takes the model's settings
    trainer = Trainer(analytic_views(2, image_size=8, num_samples=16), hash_config())
    settings = trainer.optimizer.param_groups[0]
    assert (settings["betas"], settings["eps"], settings["weight_decay"]) == ((0.9, 0.99), 1e-15, 1e-6)
    assert trainer.optimizer.param_groups[1]["weight_decay"] == 0.0


def test_training_the_hash_model_reduces_the_loss_faster_than_the_small_nerf(tiny_views):
    hashed = train(tiny_views, hash_config(iterations=60), log=quiet)
    losses = [entry["loss"] for entry in hashed.history]
    assert len(losses) == 60 and all(math.isfinite(loss) for loss in losses)
    assert sum(losses[-10:]) < 0.5 * sum(losses[:10])
    nerf = train(tiny_views, TrainConfig(
        iterations=60, batch_size=128, num_coarse=8, num_fine=8, depth=2, width=32, skip=0, log_every=1, seed=0,
    ), log=quiet)
    assert sum(losses[-10:]) < sum(entry["loss"] for entry in nerf.history[-10:])


def test_a_hash_checkpoint_resumes_exactly_and_is_loaded_back_as_a_hash_model(tiny_views, tmp_path):
    config = hash_config(iterations=10)
    straight = Trainer(tiny_views, config)
    straight_losses = [straight.step().loss.item() for _ in range(10)]

    first = Trainer(tiny_views, config)
    for _ in range(5):
        first.step()
    save_checkpoint(tmp_path / "checkpoint.pt", first, extra={"elapsed": 1.0})
    resumed = Trainer(tiny_views, config)
    load_checkpoint(tmp_path / "checkpoint.pt", resumed)
    resumed_losses = [resumed.step().loss.item() for _ in range(5)]
    assert resumed_losses == pytest.approx(straight_losses[5:], rel=1e-5)

    loaded = load_networks(tmp_path / "checkpoint.pt")
    assert isinstance(loaded.fine, HashNeRF) and loaded.config == config and loaded.step == 5
    camera = (16, 16, tiny_views.focal, tiny_views.poses[0], tiny_views.near, tiny_views.far)
    before = render_image(first.coarse, first.fine, *camera, 8, 8, white_background=True).rgb
    after = render_image(loaded.coarse, loaded.fine, *camera, 8, 8, white_background=True).rgb
    assert torch.equal(before, after)

    # a checkpoint of one model is refused by a run with the other
    other = Trainer(tiny_views, TrainConfig(iterations=10, batch_size=128, num_coarse=8, num_fine=8, depth=2, width=8, skip=0))
    with pytest.raises(ValueError, match="model"):
        load_checkpoint(tmp_path / "checkpoint.pt", other)


def test_a_checkpoint_from_before_the_model_choice_existed_still_resumes(tiny_views, tmp_path):
    # Such a checkpoint was made with the paper's network and the settings
    # that were the only ones then: the defaults of every later setting.
    config = TrainConfig(iterations=10, batch_size=128, num_coarse=8, num_fine=8, depth=2, width=8, skip=0)
    trainer = Trainer(tiny_views, config)
    trainer.step()
    save_checkpoint(tmp_path / "checkpoint.pt", trainer)
    state = torch.load(tmp_path / "checkpoint.pt", weights_only=True)
    for key in ("model", "lr_decay_start", "adam_beta2", "weight_decay", "hash_levels", "hash_bound"):
        del state["config"][key]
    torch.save(state, tmp_path / "old.pt")

    resumed = Trainer(tiny_views, config)
    load_checkpoint(tmp_path / "old.pt", resumed)
    assert resumed.step_count == 1
    assert isinstance(load_networks(tmp_path / "old.pt").fine, NeRF)
    # but not by a run whose settings differ from those defaults
    with pytest.raises(ValueError, match="model"):
        load_checkpoint(tmp_path / "old.pt", Trainer(tiny_views, hash_config(iterations=10)))


# --------------------------------------------------------------------------
# The scripts
# --------------------------------------------------------------------------


TINY = ["--num-coarse", "4", "--num-fine", "8", "--batch-size", "32", "--val-skip", "1", "--device", "cpu",
        "--hash-log2-size", "10", "--depth", "2", "--width", "16", "--skip-layer", "0"]


@pytest.fixture(scope="module")
def scene(tmp_path_factory):
    directory = tmp_path_factory.mktemp("data") / "lego"
    write_blender_scene(directory, analytic_views(4, image_size=24, num_samples=32), "train")
    write_blender_scene(directory, analytic_views(1, image_size=24, num_samples=32, phase=40.0), "val")
    write_blender_scene(directory, analytic_views(3, image_size=24, num_samples=32, phase=80.0), "test")
    return directory


def run_script(name, *arguments):
    command = [sys.executable, str(ROOT / "scripts" / name), *arguments]
    return subprocess.run(command, capture_output=True, text=True, env=ENV)


def test_the_training_script_trains_the_hash_model_and_the_other_scripts_read_it(scene, tmp_path):
    run = tmp_path / "run"
    result = run_script("02_train.py", str(scene), "--out", str(run), "--model", "hash", "--learning-rate", "0.02",
                        "--iterations", "4", "--validate-every", "4", "--checkpoint-every", "2", *TINY)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "model hash" in result.stdout and "finished at step 4" in result.stdout
    settings = json.loads((run / "config.json").read_text())
    assert settings["model"] == "hash" and settings["learning_rate"] == 0.02 and settings["hash_log2_size"] == 10
    assert settings["adam_eps"] == 1e-15 and settings["weight_decay"] == 1e-6        # the model's own settings otherwise
    assert isinstance(load_networks(run / "checkpoint.pt").fine, HashNeRF)
    # the same command resumes the same model; a run with the paper's network does not continue it
    assert "resumed from" in run_script("02_train.py", str(scene), "--out", str(run), "--model", "hash",
                                        "--learning-rate", "0.02", "--iterations", "5", *TINY).stdout
    refused = run_script("02_train.py", str(scene), "--out", str(run), "--iterations", "6", *TINY)
    assert refused.returncode == 2 and "different settings" in refused.stdout and "'model'" in refused.stdout

    scored = run_script("04_evaluate.py", str(scene), str(run / "checkpoint.pt"), "--device", "cpu", "--no-lpips")
    assert scored.returncode == 0, scored.stdout + scored.stderr
    assert "3 of 3 test views" in scored.stdout
    # After five steps the density is fog without a surface; a threshold in
    # the middle of its range gives the mesh script a surface to extract.
    middle = density_grid(load_networks(run / "checkpoint.pt").fine, 1.2, 17).median().item()
    meshed = run_script("05_extract_mesh.py", str(scene), str(run / "checkpoint.pt"), "--device", "cpu",
                        "--resolution", "17", "--threshold", f"{middle:.6f}", "--skip", "1")
    assert meshed.returncode == 0, meshed.stdout + meshed.stderr
    assert (run / "mesh_000005" / "mesh.ply").exists()
    assert json.loads((run / "mesh_000005" / "summary.json").read_text())["triangles"] > 0


def test_a_time_budget_ends_the_run_as_finished_at_the_step_it_reached(scene, tmp_path):
    run = tmp_path / "run"
    result = run_script("02_train.py", str(scene), "--out", str(run), "--iterations", "100000",
                        "--minutes", "0.02", "--validate-every", "100000", "--checkpoint-every", "100000", *TINY)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "time limit" in result.stdout and "finished at step" in result.stdout
    summary = json.loads((run / "summary.json").read_text())
    assert summary["finished"] == "minutes" and 0 < summary["steps"] < 100000
    assert summary["minutes"] == 0.02 and summary["train_seconds"] >= 1.2
    # the checkpoint is at that step, so scoring and resuming see the same model
    assert load_networks(run / "checkpoint.pt").step == summary["steps"]
    # a budget that is already spent ends at once, and a larger one goes on
    again = run_script("02_train.py", str(scene), "--out", str(run), "--iterations", "100000",
                       "--minutes", "0.02", "--validate-every", "100000", "--checkpoint-every", "100000", *TINY)
    assert again.returncode == 0 and f"finished at step {summary['steps']}" in again.stdout
    more = run_script("02_train.py", str(scene), "--out", str(run), "--iterations", "100000",
                      "--minutes", "0.04", "--validate-every", "100000", "--checkpoint-every", "100000", *TINY)
    assert more.returncode == 0 and json.loads((run / "summary.json").read_text())["steps"] > summary["steps"]
    rejected = run_script("02_train.py", str(scene), "--out", str(run), "--minutes", "0", *TINY)
    assert rejected.returncode == 2 and "--minutes must be above 0" in rejected.stdout


def test_the_smoke_test_trains_and_renders_the_hash_model(tmp_path, monkeypatch, capsys):
    # Two steps train nothing, so the run fails its requirements, but the
    # script trains, renders and records the hash model. Meshing two steps'
    # worth of fog is slow and is covered elsewhere, so it is stubbed out.
    import importlib.util
    spec = importlib.util.spec_from_file_location("smoke_test", ROOT / "scripts" / "00_smoke_test.py")
    smoke = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(smoke)
    meshed = []

    def stub(fine, train_views, device, out, name):
        meshed.append((type(fine), name))
        # a measured mesh, so that the script prints its table, which must not
        # disturb the names of the files written afterwards
        measure = {"outline_iou": 0.5, "surface_distance": 0.1, "surface_distance_signed": 0.1, "volume_ratio": 0.5}
        return {"threshold": 3.0, "triangles": 12, "pieces": 1, "colour_error": 0.2, **measure,
                "required": {}, "for_comparison": {"visual_hull_of_the_training_outlines": measure, "exact_density": measure}}

    monkeypatch.setattr(smoke, "measure_mesh", stub)
    monkeypatch.setattr(sys, "argv", ["00_smoke_test.py", "--model", "hash", "--iterations", "2", "--out", str(tmp_path)])
    assert smoke.main() == 1
    printed = capsys.readouterr().out
    assert "after 2 steps" in printed and "FAIL" in printed
    assert meshed == [(HashNeRF, "smoke_test_hash")]
    record = json.loads((tmp_path / "smoke_test_hash.json").read_text())     # named apart from the paper's
    assert record["model"] == "hash" and record["iterations"] == 2
    assert (tmp_path / "smoke_test_hash.png").exists() and "smoke_test_hash.json" in printed
    assert not (tmp_path / "smoke_test.json").exists()
    # without --model it is the paper's network, as before, under the old names
    monkeypatch.setattr(sys, "argv", ["00_smoke_test.py", "--iterations", "1", "--out", str(tmp_path)])
    smoke.main()
    capsys.readouterr()
    assert meshed == [(HashNeRF, "smoke_test_hash"), (NeRF, "smoke_test")]
    assert json.loads((tmp_path / "smoke_test.json").read_text())["model"] == "nerf"
    assert (tmp_path / "smoke_test.png").exists()
