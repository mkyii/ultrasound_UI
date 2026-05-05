import torch
import cv2
import numpy as np

from utils.ui_draw import (
    largest_component,
    make_roundish,
    overlay_round,
    ID_TO_COLOR,
)

def render_segmentation(
    image,
    mask: np.ndarray,
    alpha: float,
):
    """
    Render segmentation overlay on image (tensor or numpy)
    """

    if isinstance(image, torch.Tensor):
        img = image.detach().cpu().numpy()

        if img.ndim == 3 and img.shape[0] in (1, 3):
            img = np.transpose(img, (1, 2, 0))

        img = np.clip(img * 255.0, 0, 255).astype(np.uint8)

    elif isinstance(image, np.ndarray):
        img = image.copy()

        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)

    else:
        raise TypeError(f"Unsupported image type: {type(image)}")

    if img.ndim == 3 and img.shape[2] == 1:
        img = img[:, :, 0]

    vis = (
        cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        if img.ndim == 2
        else img.copy()
    )

    artery = largest_component((mask == 1).astype(np.uint8))
    ijv    = largest_component((mask == 2).astype(np.uint8))

    if artery.any():
        ca_bin, ca_alpha, _ = make_roundish(artery)
        vis = overlay_round(
            vis, ca_bin, ca_alpha, ID_TO_COLOR[1], fill_alpha=alpha
        )

    if ijv.any():
        ijv_bin, ijv_alpha, _ = make_roundish(ijv)
        vis = overlay_round(
            vis, ijv_bin, ijv_alpha, ID_TO_COLOR[2], fill_alpha=alpha
        )

    return vis