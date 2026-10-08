import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torchvision
from skimage.metrics import structural_similarity

from nerf.data import analytic_views
from nerf.metrics import Lpips, psnr, ssim

ROOT = Path(__file__).resolve().parents[1]


def noisy(image, amount, seed=0):
    noise = torch.randn(image.shape, generator=torch.Generator().manual_seed(seed))
    return (image + amount * noise).clamp(0, 1)


@pytest.fixture(scope="module")
def picture():
    """A 48 x 48 image with flat regions, edges and a white background."""
    return analytic_views(1, image_size=48, num_samples=64).images[0]


# --------------------------------------------------------------------------
# PSNR
# --------------------------------------------------------------------------


def test_psnr_values():
    target = torch.zeros(4, 4, 3)
    assert math.isclose(psnr(target + 0.1, target), 20.0, abs_tol=1e-4)   # MSE 1e-2
    assert math.isclose(psnr(target + 0.01, target), 40.0, abs_tol=1e-3)  # MSE 1e-4
    assert psnr(target, target) == math.inf
    with pytest.raises(ValueError, match="same shape"):   # not silently broadcast
        psnr(torch.zeros(4, 4, 1), target)


# --------------------------------------------------------------------------
# SSIM
# --------------------------------------------------------------------------


def test_ssim_of_an_image_with_itself_is_one(picture):
    assert ssim(picture, picture) == pytest.approx(1.0, abs=1e-9)


def test_ssim_of_two_flat_images_has_a_closed_form():
    # Without variation the contrast and structure term is 1, and what is
    # left compares the two brightness levels a and b:
    # (2ab + C1) / (a^2 + b^2 + C1) with C1 = 0.01^2.
    a, b, c1 = 0.25, 0.75, 0.01**2
    expected = (2 * a * b + c1) / (a**2 + b**2 + c1)
    flat_a, flat_b = torch.full((20, 30, 3), a), torch.full((20, 30, 3), b)
    assert ssim(flat_a, flat_b) == pytest.approx(expected, abs=1e-9)
    assert ssim(torch.zeros(20, 30, 3), torch.ones(20, 30, 3)) == pytest.approx(c1 / (1 + c1), abs=1e-9)


def test_ssim_matches_the_definition_evaluated_window_by_window():
    # A second implementation written straight from the formula: one explicit
    # 11 x 11 Gaussian window at a time, no convolutions.
    generator = torch.Generator().manual_seed(3)
    x = torch.rand(13, 14, 3, generator=generator, dtype=torch.float64)
    y = (0.6 * x + 0.4 * torch.rand(13, 14, 3, generator=generator, dtype=torch.float64)).clamp(0, 1)

    offsets = np.arange(11) - 5
    gauss = np.exp(-0.5 * (offsets / 1.5) ** 2)
    window = np.outer(gauss, gauss)
    window /= window.sum()
    c1, c2 = 0.01**2, 0.03**2
    values = []
    for channel in range(3):
        for top in range(13 - 11 + 1):
            for left in range(14 - 11 + 1):
                a = x[top : top + 11, left : left + 11, channel].numpy()
                b = y[top : top + 11, left : left + 11, channel].numpy()
                mean_a, mean_b = (window * a).sum(), (window * b).sum()
                var_a = (window * (a - mean_a) ** 2).sum()
                var_b = (window * (b - mean_b) ** 2).sum()
                cov = (window * (a - mean_a) * (b - mean_b)).sum()
                values.append(
                    (2 * mean_a * mean_b + c1) * (2 * cov + c2)
                    / ((mean_a**2 + mean_b**2 + c1) * (var_a + var_b + c2))
                )
    assert len(values) == 3 * 3 * 4
    assert ssim(x, y) == pytest.approx(np.mean(values), abs=1e-9)


def scikit_image_ssim(a, b):
    return structural_similarity(
        a.double().numpy(), b.double().numpy(), channel_axis=-1, data_range=1.0,
        gaussian_weights=True, sigma=1.5, use_sample_covariance=False,
    )


def test_ssim_matches_scikit_image(picture):
    generator = torch.Generator().manual_seed(5)
    pairs = [
        (picture, noisy(picture, 0.05)),                      # a slightly wrong render
        (picture, noisy(picture, 0.5)),                       # a very wrong one
        (picture, torch.ones_like(picture)),                  # a blank white image
        (torch.rand(40, 56, 3, generator=generator),          # unrelated noise, not square
         torch.rand(40, 56, 3, generator=generator)),
    ]
    for a, b in pairs:
        assert ssim(a, b) == pytest.approx(scikit_image_ssim(a, b), abs=1e-9)


def test_ssim_is_symmetric_and_falls_as_the_image_gets_worse(picture):
    scores = [ssim(noisy(picture, amount), picture) for amount in (0.0, 0.02, 0.1, 0.3)]
    assert scores[0] == pytest.approx(1.0, abs=1e-9)
    assert scores[0] > scores[1] > scores[2] > scores[3] > 0
    worse = noisy(picture, 0.1)
    assert ssim(worse, picture) == pytest.approx(ssim(picture, worse), abs=1e-9)


def test_ssim_rejects_images_it_cannot_score():
    with pytest.raises(ValueError, match="at least 11"):
        ssim(torch.zeros(10, 40, 3), torch.zeros(10, 40, 3))
    with pytest.raises(ValueError, match="same shape"):
        ssim(torch.zeros(20, 20, 3), torch.zeros(20, 24, 3))
    eight_bit = torch.randint(0, 256, (20, 20, 3), generator=torch.Generator().manual_seed(0))
    with pytest.raises(ValueError, match="in \\[0, 1\\]"):
        ssim(eight_bit, eight_bit.float())
    # a render may overshoot 1 by rounding; that is not an error
    assert ssim(torch.full((20, 20, 3), 1.0 + 1e-6), torch.ones(20, 20, 3)) == pytest.approx(1.0, abs=1e-9)


# --------------------------------------------------------------------------
# LPIPS. The VGG-16 backbone is left randomly initialised here, so the values
# mean nothing, but the code that runs is the code that runs with the real
# weights, and no test downloads anything.
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def lpips_untrained():
    torch.manual_seed(0)
    return Lpips(pretrained_backbone=False)


def test_lpips_hands_the_network_the_images_it_expects(lpips_untrained, picture):
    # The package documents its input as RGB, shape (N, 3, H, W), in [-1, 1],
    # and converts from [0, 1] itself when called with normalize=True. The
    # wrapper must agree with that conversion. The images are not square, so
    # swapped axes would show.
    a, b = picture[:40], noisy(picture, 0.1)[:40]
    assert a.shape == (40, 48, 3)
    assert lpips_untrained.model.pnet_type == "vgg" and lpips_untrained.model.version == "0.1"

    def as_batch(image):
        return image.permute(2, 0, 1)[None]

    with torch.no_grad():
        expected = lpips_untrained.model(as_batch(a), as_batch(b), normalize=True).item()
    assert expected > 0
    assert lpips_untrained(a, b) == pytest.approx(expected, rel=1e-6)
    assert lpips_untrained(a, a) == 0.0
    # a different input range or channel order gives a different number
    with torch.no_grad():
        unscaled = lpips_untrained.model(as_batch(a), as_batch(b)).item()
        flipped = lpips_untrained.model(as_batch(a.flip(-1)), as_batch(b.flip(-1)), normalize=True).item()
    assert abs(unscaled - expected) > 1e-4 * expected
    assert abs(flipped - expected) > 1e-4 * expected


def test_lpips_asks_for_the_imagenet_weights_in_the_given_directory(tmp_path, monkeypatch):
    # Stand in for the download: note what is asked for and where it would be
    # kept, and hand back a network's worth of made-up weights.
    asked = {}
    made_up = torchvision.models.vgg16(weights=None).state_dict()

    def get_state_dict(weights, *args, **kwargs):
        asked["url"] = weights.url
        asked["directory"] = torch.hub.get_dir()
        return made_up

    monkeypatch.setattr(torchvision.models.VGG16_Weights, "get_state_dict", get_state_dict)
    before = torch.hub.get_dir()
    metric = Lpips(weights_dir=tmp_path / "hub")

    assert asked["url"] == "https://download.pytorch.org/models/vgg16-397923af.pth"
    assert Path(asked["directory"]) == tmp_path / "hub"
    assert torch.hub.get_dir() == before              # only changed while loading
    # the weights that came back are the ones the measure uses
    assert torch.equal(metric.model.net.slice1[0].weight, made_up["features.0.weight"])
    assert torch.equal(metric.model.net.slice5[-2].weight, made_up["features.28.weight"])


def test_the_other_measures_need_neither_lpips_nor_torchvision():
    # Training imports this module for PSNR, and the SageMaker container has
    # no lpips package.
    code = (
        "import sys; sys.modules['lpips'] = None; sys.modules['torchvision'] = None; "
        "import torch, nerf.train; from nerf.metrics import psnr, ssim; "
        "image = torch.rand(16, 16, 3); print(psnr(image, image), round(ssim(image, image), 6))"
    )
    env = dict(os.environ, PYTHONPATH=str(ROOT / "src"))
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env)
    assert result.stdout.strip() == "inf 1.0", result.stderr
