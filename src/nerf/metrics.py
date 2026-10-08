"""The three image-quality measures of the paper's result tables (Section 6).

A rendered view is scored against the true image of that view:

  PSNR   how close the pixel values are, on a logarithmic scale. Higher is
         better.
  SSIM   structural similarity (Wang et al. 2004): compares brightness,
         contrast and structure in small windows. 1 for identical images.
         Higher is better.
  LPIPS  learned perceptual similarity (Zhang et al. 2018): the distance
         between the two images in the feature space of a network trained
         to recognise objects. 0 for identical images. Lower is better.

Images are tensors of shape (H, W, 3) with colours in [0, 1].
"""

from __future__ import annotations

import math
import warnings

import torch


def psnr(prediction: torch.Tensor, target: torch.Tensor) -> float:
    """Peak signal-to-noise ratio in dB for colours in [0, 1]: -10 log10(MSE).

    Higher is better, and every factor of 10 in the mean squared error is
    10 dB. This is the main image-quality number the paper reports.
    """
    if prediction.shape != target.shape:   # or broadcasting would compare the wrong things
        raise ValueError(
            f"expected two images of the same shape, got "
            f"{tuple(prediction.shape)} and {tuple(target.shape)}"
        )
    mse = torch.mean((prediction - target) ** 2).item()
    return math.inf if mse == 0 else -10.0 * math.log10(mse)


def ssim(
    prediction: torch.Tensor,
    target: torch.Tensor,
    window_size: int = 11,
    sigma: float = 1.5,
    k1: float = 0.01,
    k2: float = 0.03,
) -> float:
    """Mean structural similarity of two images, as defined by Wang et al. 2004.

    Around every pixel the two images are compared inside a small window with
    Gaussian weights. With means m, variances v and covariance c of the two
    images in that window,

        SSIM = (2 m_x m_y + C1) (2 c_xy + C2) / ((m_x^2 + m_y^2 + C1) (v_x + v_y + C2))

    where C1 = k1^2 and C2 = k2^2 for colours in [0, 1]. The result is the
    mean over all window positions and the three colour channels.

    The defaults are the original ones: an 11 x 11 window with standard
    deviation 1.5, k1 = 0.01, k2 = 0.03. Only windows that lie fully inside
    the image are used, so nothing is assumed about pixels beyond the border.
    This is the convention of `tf.image.ssim` and, with matching options, of
    scikit-image, against which the tests compare it.
    """
    if prediction.shape != target.shape or prediction.ndim != 3:
        raise ValueError(
            f"expected two images of the same shape (H, W, C), got "
            f"{tuple(prediction.shape)} and {tuple(target.shape)}"
        )
    if min(prediction.shape[0], prediction.shape[1]) < window_size:
        raise ValueError(
            f"images must be at least {window_size} pixels on each side, "
            f"got {prediction.shape[1]} x {prediction.shape[0]}"
        )

    # Double precision on the CPU: variances are differences of nearly equal
    # numbers, and an image is small enough for this to cost nothing.
    # (H, W, C) -> (C, 1, H, W): each colour channel is filtered on its own.
    x = prediction.detach().cpu().double().permute(2, 0, 1)[:, None]
    y = target.detach().cpu().double().permute(2, 0, 1)[:, None]
    if min(x.min(), y.min()) < -1e-3 or max(x.max(), y.max()) > 1 + 1e-3:
        # C1 and C2 below are for this range; 8-bit values would be scored wrongly
        raise ValueError("colours must be in [0, 1]")

    offsets = torch.arange(window_size, dtype=torch.float64) - (window_size - 1) / 2
    kernel = torch.exp(-0.5 * (offsets / sigma) ** 2)
    kernel = kernel / kernel.sum()

    def window_mean(image: torch.Tensor) -> torch.Tensor:
        # A 2D Gaussian is a vertical 1D Gaussian followed by a horizontal one.
        image = torch.nn.functional.conv2d(image, kernel.view(1, 1, -1, 1))
        return torch.nn.functional.conv2d(image, kernel.view(1, 1, 1, -1))

    mean_x, mean_y = window_mean(x), window_mean(y)
    var_x = window_mean(x * x) - mean_x**2
    var_y = window_mean(y * y) - mean_y**2
    cov_xy = window_mean(x * y) - mean_x * mean_y

    c1, c2 = k1**2, k2**2
    ssim_map = ((2 * mean_x * mean_y + c1) * (2 * cov_xy + c2)) / (
        (mean_x**2 + mean_y**2 + c1) * (var_x + var_y + c2)
    )
    return ssim_map.mean().item()


class Lpips:
    """LPIPS with the VGG network, computed by the reference `lpips` package.

    The measure has two parts: the convolutional layers of a VGG-16 trained on
    ImageNet, and one small learned weighting per layer. The weightings ship
    with the package. The VGG-16 weights (528 MB) are fetched by torchvision
    the first time they are needed, into `weights_dir` if one is given and
    into PyTorch's own cache directory otherwise.

    Args:
        device: where to run the network.
        weights_dir: directory for the VGG-16 weights. The file is looked
            for, and downloaded to, `<weights_dir>/checkpoints/`.
        pretrained_backbone: False leaves the VGG-16 randomly initialised.
            The numbers then mean nothing; the tests use this to run the same
            code without the download.
    """

    def __init__(self, device: str = "cpu", weights_dir=None, pretrained_backbone: bool = True) -> None:
        with warnings.catch_warnings():
            # Importing torchvision can warn about image libraries that are
            # not used here, and the package asks it for the network in a way
            # torchvision has since renamed, which still works but warns.
            warnings.simplefilter("ignore")
            import lpips  # imported here because only this measure needs the package

            def build():
                return lpips.LPIPS(net="vgg", pnet_rand=not pretrained_backbone, verbose=False)

            if weights_dir is None:
                model = build()
            else:
                previous = torch.hub.get_dir()
                torch.hub.set_dir(str(weights_dir))
                try:
                    model = build()
                finally:
                    torch.hub.set_dir(previous)
        self.model = model.to(device).eval()
        self.device = device

    @torch.no_grad()
    def __call__(self, prediction: torch.Tensor, target: torch.Tensor) -> float:
        def prepare(image: torch.Tensor) -> torch.Tensor:
            # (H, W, 3) in [0, 1] -> (1, 3, H, W) in [-1, 1], the network's input range
            image = image.detach().to(self.device, torch.float32)
            return image.permute(2, 0, 1)[None] * 2.0 - 1.0

        return self.model(prepare(prediction), prepare(target)).item()
