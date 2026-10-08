"""The scoring script, and the training script's kept checkpoints that it reads.

A small scene is written to disk in the dataset's format and a small network
is trained on it for a few steps by the real training script. The scoring
script is then run on the result, in this process, so that its pieces can be
watched and replaced.
"""

import csv
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torchvision
from PIL import Image

from nerf.data import analytic_views, load_blender
from nerf.metrics import Lpips, psnr, ssim
from nerf.render import render_image
from nerf.train import load_networks
from test_data import write_blender_scene

ROOT = Path(__file__).resolve().parents[1]
ENV = dict(os.environ, PYTHONPATH=str(ROOT / "src"))
TINY = ["--depth", "2", "--width", "16", "--skip-layer", "0", "--num-coarse", "4", "--num-fine", "8",
        "--batch-size", "32", "--val-skip", "1", "--device", "cpu"]
SIZE = 32          # pixels per side of the scene's images
TEST_VIEWS = 5


def run_training(scene, run, *options):
    command = [sys.executable, str(ROOT / "scripts" / "02_train.py"), str(scene), "--out", str(run),
               "--iterations", "6", "--validate-every", "6", "--checkpoint-every", "3", *options, *TINY]
    return subprocess.run(command, capture_output=True, text=True, env=ENV)


def train(scene, run, *options):
    result = run_training(scene, run, *options)
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout


@pytest.fixture(scope="module")
def scene(tmp_path_factory):
    # Named like the paper's scene, so that the paper's numbers apply to it.
    directory = tmp_path_factory.mktemp("data") / "lego"
    write_blender_scene(directory, analytic_views(6, image_size=SIZE, num_samples=32), "train")
    write_blender_scene(directory, analytic_views(2, image_size=SIZE, num_samples=32, phase=40.0), "val")
    write_blender_scene(directory, analytic_views(TEST_VIEWS, image_size=SIZE, num_samples=32, phase=80.0), "test")
    return directory


@pytest.fixture(scope="module")
def trained(scene, tmp_path_factory):
    """A finished 6-step run that kept a checkpoint every 2 steps. Read only."""
    run = tmp_path_factory.mktemp("runs") / "lego_tiny"
    train(scene, run, "--keep-every", "2")
    return run


@pytest.fixture
def run(trained, tmp_path):
    """A private copy of that run, so tests cannot see each other's results."""
    return Path(shutil.copytree(trained, tmp_path / "lego_tiny"))


@pytest.fixture(scope="module")
def lpips_untrained():
    torch.manual_seed(0)
    return Lpips(pretrained_backbone=False)


@pytest.fixture
def scorer(monkeypatch, lpips_untrained):
    """The scoring script as a module, with LPIPS on an untrained backbone
    (same code, meaningless values, nothing to download) and a count of the
    views it renders."""
    spec = importlib.util.spec_from_file_location("evaluate_cli", ROOT / "scripts" / "04_evaluate.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    module.lpips_requests = []

    def make_lpips(device, weights_dir):
        module.lpips_requests.append((device, Path(weights_dir)))
        return lpips_untrained

    module.rendered = []
    real_render_image = module.render_image

    def counting_render_image(*args, **kwargs):
        module.rendered.append(args[5])   # the camera pose
        return real_render_image(*args, **kwargs)

    monkeypatch.setattr(module, "Lpips", make_lpips)
    monkeypatch.setattr(module, "render_image", counting_render_image)
    return module


def score(scorer, scene, checkpoint, *options):
    return scorer.main([str(scene), str(checkpoint), "--device", "cpu", *options])


def table(directory):
    with (directory / "per_view.csv").open(newline="") as handle:
        return list(csv.DictReader(handle))


def scores_only(rows):
    return [(row["view"], row["name"], row["psnr"], row["ssim"], row["lpips"]) for row in rows]


def summary(directory):
    return json.loads((directory / "summary.json").read_text())


# --------------------------------------------------------------------------
# Checkpoints kept by the training script
# --------------------------------------------------------------------------


def weights(path):
    state = torch.load(path, map_location="cpu", weights_only=True)
    return state["step"], [*state["coarse"].values(), *state["fine"].values()]


def test_training_keeps_a_checkpoint_of_its_own_every_n_steps(trained):
    kept = sorted(path.name for path in trained.glob("checkpoint*.pt"))
    assert kept == ["checkpoint.pt", "checkpoint_000002.pt", "checkpoint_000004.pt", "checkpoint_000006.pt"]
    steps = {name: weights(trained / name)[0] for name in kept}
    assert steps == {"checkpoint.pt": 6, "checkpoint_000002.pt": 2,
                     "checkpoint_000004.pt": 4, "checkpoint_000006.pt": 6}
    # each holds the weights of its own step
    _, at_2 = weights(trained / "checkpoint_000002.pt")
    _, at_4 = weights(trained / "checkpoint_000004.pt")
    _, at_6 = weights(trained / "checkpoint_000006.pt")
    _, latest = weights(trained / "checkpoint.pt")
    assert not torch.equal(at_2[0], at_4[0]) and not torch.equal(at_4[0], at_6[0])
    assert all(torch.equal(a, b) for a, b in zip(at_6, latest))


def test_a_kept_checkpoint_is_a_full_one_and_keeping_can_be_switched_off(scene, trained, tmp_path):
    # Put the step-4 checkpoint where a run looks for its state and train on:
    # the run must arrive where the original arrived.
    branch = tmp_path / "branch"
    branch.mkdir()
    shutil.copy(trained / "checkpoint_000004.pt", branch / "checkpoint.pt")
    output = train(scene, branch, "--keep-every", "0")
    assert "at step 4" in output and "finished at step 6" in output
    _, expected = weights(trained / "checkpoint.pt")
    step, continued = weights(branch / "checkpoint.pt")
    assert step == 6
    assert all(torch.allclose(a, b, atol=1e-6) for a, b in zip(expected, continued))
    # and with --keep-every 0 nothing but the latest state is written
    assert [path.name for path in branch.glob("checkpoint*.pt")] == ["checkpoint.pt"]


def test_a_run_is_not_continued_at_another_image_size(scene, run):
    # The scoring script shrinks the images "as in training", so a run must
    # have one image size from its first step to its last.
    recorded = (run / "config.json").read_text()
    assert json.loads(recorded)["downscale"] == 1
    state = (run / "checkpoint.pt").read_bytes()

    result = run_training(scene, run, "--downscale", "2", "--iterations", "8")
    assert result.returncode == 2
    assert "trained with --downscale 1, not 2" in result.stdout
    assert (run / "config.json").read_text() == recorded
    assert (run / "checkpoint.pt").read_bytes() == state

    # with the size it was started on, the same run does continue
    assert "finished at step 8" in train(scene, run, "--downscale", "1", "--iterations", "8")

    # A directory in which no state was ever saved holds nothing to continue,
    # whatever settings were written there.
    (run / "checkpoint.pt").unlink()
    assert "finished at step 6" in train(scene, run, "--downscale", "2")
    assert json.loads((run / "config.json").read_text())["downscale"] == 2


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------


def test_scores_are_those_of_an_independent_render_of_each_test_view(scorer, scene, run, lpips_untrained, capsys):
    assert score(scorer, scene, run / "checkpoint.pt") == 0
    results = run / "eval_000006_test"          # named after the step and the views
    rows = table(results)
    assert [row["view"] for row in rows] == ["0", "1", "2", "3", "4"]
    assert [row["name"] for row in rows] == [f"./test/r_{k}" for k in range(TEST_VIEWS)]
    assert len(scorer.rendered) == TEST_VIEWS

    # The same thing done by hand: the test views, the networks from the
    # checkpoint, the sample counts they were trained with.
    views = load_blender(scene, "test")
    networks = load_networks(run / "checkpoint.pt")
    for k, row in enumerate(rows):
        rendered = render_image(
            networks.coarse, networks.fine, SIZE, SIZE, views.focal, views.poses[k],
            views.near, views.far, 4, 8, white_background=True,
        ).rgb.clamp(0, 1)
        assert float(row["psnr"]) == pytest.approx(psnr(rendered, views.images[k]), abs=1e-3)
        assert float(row["ssim"]) == pytest.approx(ssim(rendered, views.images[k]), abs=1e-5)
        assert float(row["lpips"]) == pytest.approx(lpips_untrained(rendered, views.images[k]), abs=1e-5)
        saved = np.asarray(Image.open(results / f"render_{k:03d}.png"))
        assert np.array_equal(saved, (rendered * 255).round().byte().numpy())
    # the views differ, and so do their scores
    assert len({row["psnr"] for row in rows}) == TEST_VIEWS

    # The summary is the mean of the table, with what it was measured on.
    result = summary(results)
    for measure, digits in (("psnr", 3), ("ssim", 4), ("lpips", 4)):
        expected = sum(float(row[measure]) for row in rows) / TEST_VIEWS
        assert result[measure] == pytest.approx(expected, abs=0.6 * 10**-digits)
    assert (result["views"], result["views_in_split"]) == (TEST_VIEWS, TEST_VIEWS)
    assert (result["width"], result["height"], result["downscale"]) == (SIZE, SIZE, 1)
    assert (result["step"], result["split"], result["scene"]) == (6, "test", "lego")
    assert (result["num_coarse"], result["num_fine"]) == (4, 8)
    assert result["lpips_network"] == "vgg"

    # Five small views are not the paper's protocol, and the output says so.
    assert result["paper"] == {"psnr": 32.54, "ssim": 0.961, "lpips": 0.050}
    assert result["paper_protocol"] is False
    assert result["differences_from_paper_protocol"] == ["5 views, not 200", "32 x 32 images, not 800 x 800"]
    printed = capsys.readouterr().out
    assert "Not the paper's protocol: 5 views, not 200; 32 x 32 images, not 800 x 800." in printed
    assert "this run" in printed and "\npaper " not in printed

    # LPIPS was set up once, on the CPU, with the weights kept inside the project.
    assert scorer.lpips_requests == [("cpu", ROOT / "data" / "torch_hub")]

    # Four views side by side, truth above render, each enlarged 12 times.
    picture = Image.open(results / "comparison.png")
    assert picture.size == (4 * (384 + 4) + 4, 2 * (384 + 4) + 4)
    pixels = np.asarray(picture)
    first_truth = np.asarray(Image.open(scene / "test" / "r_0.png").convert("RGBA")).astype(np.float32) / 255
    first_truth = first_truth[..., :3] * first_truth[..., 3:] + (1 - first_truth[..., 3:])
    assert np.array_equal(pixels[4:388:12, 4:388:12], (first_truth * 255).round().astype(np.uint8))
    assert np.array_equal(pixels[392:776:12, 4:388:12], np.asarray(Image.open(results / "render_000.png")))


def test_the_test_views_are_scored_and_not_some_other_split(scorer, scene, run):
    assert score(scorer, scene, run / "checkpoint.pt") == 0
    assert score(scorer, scene, run / "checkpoint.pt", "--split", "val") == 0
    test_rows, val_rows = table(run / "eval_000006_test"), table(run / "eval_000006_val")
    assert [row["name"] for row in val_rows] == ["./val/r_0", "./val/r_1"]
    # every rendered pose is a pose of the split that was asked for
    test_poses, val_poses = load_blender(scene, "test").poses, load_blender(scene, "val").poses
    assert len(scorer.rendered) == TEST_VIEWS + 2
    for k in range(TEST_VIEWS):
        assert torch.equal(scorer.rendered[k], test_poses[k])
    for k in range(2):
        assert torch.equal(scorer.rendered[TEST_VIEWS + k], val_poses[k])
    assert test_rows[0]["psnr"] != val_rows[0]["psnr"]
    assert summary(run / "eval_000006_val")["differences_from_paper_protocol"][0] == "the val views, not the test views"


def test_skip_scores_every_nth_view(scorer, scene, run):
    assert score(scorer, scene, run / "checkpoint.pt", "--skip", "2") == 0
    results = run / "eval_000006_test_every2"
    rows = table(results)
    assert [row["view"] for row in rows] == ["0", "2", "4"]
    assert [row["name"] for row in rows] == ["./test/r_0", "./test/r_2", "./test/r_4"]
    assert sorted(path.name for path in results.glob("render_*.png")) == [
        "render_000.png", "render_002.png", "render_004.png"]
    result = summary(results)
    assert (result["views"], result["views_in_split"], result["skip"]) == (3, TEST_VIEWS, 2)
    assert result["differences_from_paper_protocol"][0] == "3 of the 5 test views"

    # the same views score the same whether or not the others are scored too
    assert score(scorer, scene, run / "checkpoint.pt") == 0
    everything = table(run / "eval_000006_test")
    assert scores_only(rows) == scores_only([everything[0], everything[2], everything[4]])


def test_a_stopped_evaluation_continues_with_the_missing_views(scorer, scene, run, monkeypatch, capsys):
    assert score(scorer, scene, run / "checkpoint.pt", "--out", str(run / "in_one_go")) == 0
    expected = table(run / "in_one_go")

    counting_render_image = scorer.render_image

    def interrupted_at_the_third_view(*args, **kwargs):
        if len(scorer.rendered) == TEST_VIEWS + 2:
            raise KeyboardInterrupt
        return counting_render_image(*args, **kwargs)

    monkeypatch.setattr(scorer, "render_image", interrupted_at_the_third_view)
    assert score(scorer, scene, run / "checkpoint.pt") == 130
    results = run / "eval_000006_test"
    assert [row["view"] for row in table(results)] == ["0", "1"]
    assert not (results / "summary.json").exists()
    assert "stopped with 2 of 5 views scored" in capsys.readouterr().out

    monkeypatch.setattr(scorer, "render_image", counting_render_image)
    assert score(scorer, scene, run / "checkpoint.pt") == 0
    assert len(scorer.rendered) == 2 * TEST_VIEWS            # 5, then 2, then only the missing 3
    assert scores_only(table(results)) == scores_only(expected)
    assert summary(results)["psnr"] == summary(run / "in_one_go")["psnr"]
    assert "2 views were scored earlier; continuing with the other 3" in capsys.readouterr().out

    # Once everything is scored, running it again renders nothing and
    # reports the same numbers.
    before = (results / "per_view.csv").read_text()
    assert score(scorer, scene, run / "checkpoint.pt") == 0
    assert len(scorer.rendered) == 2 * TEST_VIEWS
    assert (results / "per_view.csv").read_text() == before


def test_a_row_cut_short_or_a_lost_render_is_done_again(scorer, scene, run):
    assert score(scorer, scene, run / "checkpoint.pt") == 0
    results = run / "eval_000006_test"
    complete = table(results)

    lines = (results / "per_view.csv").read_text().splitlines()   # the header, then views 0 to 4
    lines[3] += ",something extra"                           # view 2: a field too many
    lines.insert(2, "0,./test/r_0,99.0000,0.990000,0.010000,0.1")   # view 0 a second time
    lines[-1] = lines[-1][:-20]                              # view 4: killed while writing the row
    (results / "per_view.csv").write_text("\n".join(lines))
    (results / "render_001.png").unlink()                    # a render that went missing
    scorer.rendered.clear()

    assert score(scorer, scene, run / "checkpoint.pt") == 0
    poses = load_blender(scene, "test").poses
    assert len(scorer.rendered) == 2
    assert torch.equal(scorer.rendered[0], poses[1]) and torch.equal(scorer.rendered[1], poses[4])
    assert scores_only(table(results)) == scores_only(complete)      # back in view order, same scores
    assert (results / "render_001.png").exists()


def test_scores_of_different_things_are_never_mixed(scorer, scene, run, tmp_path, capsys):
    shared = run / "shared"
    assert score(scorer, scene, run / "checkpoint_000002.pt", "--out", str(shared)) == 0
    before = (shared / "per_view.csv").read_text()
    capsys.readouterr()

    # other weights, other views, another size, or LPIPS left out
    for options, changed in (
        ((run / "checkpoint_000004.pt",), "step, weights_sha256"),
        ((run / "checkpoint_000002.pt", "--skip", "2"), "skip"),
        ((run / "checkpoint_000002.pt", "--split", "val"), "cameras_sha256, split"),
        ((run / "checkpoint_000002.pt", "--downscale", "2"), "downscale"),
        ((run / "checkpoint_000002.pt", "--no-lpips"), "with_lpips"),
    ):
        scorer.rendered.clear()
        assert score(scorer, scene, *options, "--out", str(shared)) == 2
        assert f"holds scores for something else (different: {changed})" in capsys.readouterr().out
        assert scorer.rendered == []
        assert (shared / "per_view.csv").read_text() == before

    # Same step but other weights, as a second run with another seed would
    # leave: told apart by the weights themselves, of either network.
    for network in ("coarse", "fine"):
        state = torch.load(run / "checkpoint_000002.pt", map_location="cpu", weights_only=True)
        name = next(iter(state[network]))
        state[network][name] = state[network][name] + 0.01
        torch.save(state, run / "retrained.pt")
        assert score(scorer, scene, run / "retrained.pt", "--out", str(shared)) == 2
        assert "(different: weights_sha256)" in capsys.readouterr().out
    assert (shared / "per_view.csv").read_text() == before

    # Another scene in a directory of the same name: other cameras.
    elsewhere = tmp_path / "elsewhere" / "lego"
    write_blender_scene(elsewhere, analytic_views(TEST_VIEWS, image_size=SIZE, num_samples=32, phase=120.0), "test")
    assert score(scorer, elsewhere, run / "checkpoint_000002.pt", "--out", str(shared)) == 2
    assert "(different: cameras_sha256)" in capsys.readouterr().out

    # Scores with nothing to say what they are scores of are not adopted.
    record = (shared / "settings.json").read_text()
    scorer.rendered.clear()
    for damaged in (None, "", "[]", record[:40]):     # deleted, emptied, something else, cut short
        (shared / "settings.json").unlink(missing_ok=True)
        if damaged is not None:
            (shared / "settings.json").write_text(damaged)
        assert score(scorer, scene, run / "checkpoint_000002.pt", "--out", str(shared)) == 2
        assert "holds scores, but no settings.json" in capsys.readouterr().out
        assert scorer.rendered == [] and (shared / "per_view.csv").read_text() == before
    (shared / "settings.json").write_text(record)
    assert score(scorer, scene, run / "checkpoint_000002.pt", "--out", str(shared)) == 0   # as before

    # A training run's own directory is not a place for results.
    training_summary = (run / "summary.json").read_text()
    assert score(scorer, scene, run / "checkpoint.pt", "--out", str(run)) == 2
    assert "is the directory of a training run" in capsys.readouterr().out
    assert (run / "summary.json").read_text() == training_summary
    assert not (run / "per_view.csv").exists()

    # The latest checkpoint and the one kept at the same step are the same
    # model: scoring the second finds the work already done.
    assert score(scorer, scene, run / "checkpoint.pt") == 0
    scorer.rendered.clear()
    assert score(scorer, scene, run / "checkpoint_000006.pt") == 0
    assert scorer.rendered == []


def test_images_are_shrunk_as_in_training_unless_told_otherwise(scorer, scene, run, capsys):
    settings = json.loads((run / "config.json").read_text())
    assert settings["downscale"] == 1                        # what the training script recorded
    (run / "config.json").write_text(json.dumps({**settings, "downscale": 2}))

    assert score(scorer, scene, run / "checkpoint.pt") == 0
    result = summary(run / "eval_000006_test")
    assert (result["width"], result["height"], result["downscale"]) == (SIZE // 2, SIZE // 2, 2)
    assert Image.open(run / "eval_000006_test" / "render_000.png").size == (SIZE // 2, SIZE // 2)

    assert score(scorer, scene, run / "checkpoint.pt", "--downscale", "1", "--out", str(run / "full")) == 0
    assert summary(run / "full")["width"] == SIZE

    # A checkpoint on its own does not say what it was trained on.
    (run / "config.json").unlink()
    capsys.readouterr()
    assert score(scorer, scene, run / "checkpoint.pt", "--out", str(run / "unknown")) == 2
    assert "--downscale" in capsys.readouterr().out
    assert not (run / "unknown").exists()
    assert score(scorer, scene, run / "checkpoint.pt", "--downscale", "1", "--out", str(run / "unknown")) == 0


def test_without_lpips_the_measure_is_left_out_and_nothing_is_loaded(scorer, scene, run, capsys):
    assert score(scorer, scene, run / "checkpoint.pt", "--no-lpips") == 0
    assert scorer.lpips_requests == []
    results = run / "eval_000006_test"
    assert [row["lpips"] for row in table(results)] == [""] * TEST_VIEWS
    result = summary(results)
    assert result["lpips"] is None and result["lpips_network"] is None
    assert result["psnr"] > 0 and 0 < result["ssim"] < 1
    assert "       -\n" in capsys.readouterr().out            # a dash in the LPIPS column


def test_a_problem_with_lpips_is_reported_before_anything_is_rendered(scorer, scene, run, monkeypatch, capsys):
    def no_network(device, weights_dir):
        raise OSError("could not reach the server")

    monkeypatch.setattr(scorer, "Lpips", no_network)
    assert score(scorer, scene, run / "checkpoint.pt", "--weights-dir", "somewhere") == 2
    printed = capsys.readouterr().out
    assert "LPIPS could not be set up: OSError: could not reach the server" in printed
    # it says how to fetch the file by hand, and how to do without
    assert ("curl -L -o somewhere/checkpoints/vgg16-397923af.pth "
            "https://download.pytorch.org/models/vgg16-397923af.pth") in printed
    assert "--no-lpips" in printed
    assert scorer.rendered == []
    # the address is the one torchvision would download from
    assert scorer.VGG16_URL == torchvision.models.VGG16_Weights.IMAGENET1K_V1.url

    # Nothing was scored, so the way out it names works in the same directory.
    assert score(scorer, scene, run / "checkpoint.pt", "--no-lpips") == 0
    assert len(table(run / "eval_000006_test")) == TEST_VIEWS


def test_a_damaged_weights_file_is_reported_and_not_downloaded_over(scorer, scene, run, monkeypatch, tmp_path, capsys):
    # The real LPIPS class this time, pointed at a file that is not a weights
    # file. Because the file exists nothing is downloaded.
    monkeypatch.setattr(scorer, "Lpips", Lpips)
    folder = tmp_path / "hub" / "checkpoints"
    folder.mkdir(parents=True)
    (folder / "vgg16-397923af.pth").write_bytes(b"half a download")
    assert score(scorer, scene, run / "checkpoint.pt", "--weights-dir", str(tmp_path / "hub")) == 2
    assert "LPIPS could not be set up" in capsys.readouterr().out
    assert scorer.rendered == []
    assert (folder / "vgg16-397923af.pth").read_bytes() == b"half a download"


def test_which_results_may_be_set_next_to_the_papers(scorer):
    differences = scorer.protocol_differences
    assert differences("test", 200, 200, 800, 800) == []
    assert differences("test", 25, 200, 800, 800) == ["25 of the 200 test views"]
    assert differences("test", 200, 200, 400, 400) == ["400 x 400 images, not 800 x 800"]
    assert differences("val", 100, 100, 800, 800) == ["the val views, not the test views", "100 views, not 200"]
    assert differences("test", 25, 200, 100, 100) == ["25 of the 200 test views", "100 x 100 images, not 800 x 800"]
    # Table 4 of the paper, Lego row
    assert scorer.PAPER["lego"] == {"psnr": 32.54, "ssim": 0.961, "lpips": 0.050}


def test_the_papers_numbers_are_shown_only_for_its_protocol_and_its_scene(scorer, scene, run, monkeypatch, tmp_path, capsys):
    # The same images under another name are a scene the paper has no numbers
    # for: the departure from the protocol is still stated, without numbers.
    other = Path(shutil.copytree(scene, tmp_path / "spheres"))
    assert score(scorer, other, run / "checkpoint.pt", "--out", str(run / "other")) == 0
    printed = capsys.readouterr().out
    assert "Not the paper's protocol: 5 views, not 200; 32 x 32 images, not 800 x 800." in printed
    assert "The paper's numbers" not in printed and "\npaper " not in printed
    assert summary(run / "other")["paper"] is None and summary(run / "other")["paper_protocol"] is False

    # Pretend the paper's protocol were this scene's 5 test views at 32 x 32.
    monkeypatch.setattr(scorer, "PAPER_VIEWS", TEST_VIEWS)
    monkeypatch.setattr(scorer, "PAPER_SIZE", SIZE)
    assert score(scorer, scene, run / "checkpoint.pt") == 0
    printed = capsys.readouterr().out
    assert "paper        32.54   0.961   0.050   Mildenhall et al. 2020, Table 4, lego" in printed
    assert "Not the paper's protocol" not in printed
    result = summary(run / "eval_000006_test")
    assert result["paper_protocol"] is True and result["differences_from_paper_protocol"] == []

    # A subset is not the protocol even then.
    assert score(scorer, scene, run / "checkpoint.pt", "--skip", "2") == 0
    assert "Not the paper's protocol: 3 of the 5 test views." in capsys.readouterr().out

    # And for the other scene there is still no row to show.
    assert score(scorer, other, run / "checkpoint.pt", "--out", str(run / "other_again")) == 0
    printed = capsys.readouterr().out
    assert "\npaper " not in printed and "Not the paper's protocol" not in printed
    assert summary(run / "other_again")["paper_protocol"] is True


def test_the_script_runs_from_the_command_line(scene, run):
    command = [sys.executable, str(ROOT / "scripts" / "04_evaluate.py"), str(scene),
               str(run / "checkpoint_000004.pt"), "--no-lpips", "--device", "cpu", "--skip", "4"]
    result = subprocess.run(command, capture_output=True, text=True, env=ENV)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "2 of 5 test views at 32 x 32, checkpoint at step 4" in result.stdout
    assert (run / "eval_000004_test_every4" / "summary.json").exists()

    missing = [*command[:3], str(run / "no_such_checkpoint.pt")]
    result = subprocess.run(missing, capture_output=True, text=True, env=ENV)
    assert result.returncode == 2 and "There is no checkpoint at" in result.stdout
