import numpy as np


def normalize_image_3d(img: np.ndarray) -> np.ndarray:
    """
    Normalize a 3D hyperspectral image to the range [0, 1].

    Parameters
    ----------
    hsi : np.ndarray
        Hyperspectral image array of shape (H, W, C).

    Returns
    -------
    np.ndarray
        Normalized hyperspectral image, same shape as input.
    """
    assert img.ndim == 3, "HSI must be a 3D array (H, W, C)"
    return normalize_image(img)


def normalize_image_2d(img: np.ndarray) -> np.ndarray:
    """
    Normalize a 2D hyperspectral image to the range [0, 1].

    Parameters
    ----------
    hsi : np.ndarray
        Hyperspectral image array of shape (H, W).

    Returns
    -------
    np.ndarray
        Normalized hyperspectral image, same shape as input.
    """
    assert img.ndim == 2, "HSI must be a 2D array (H, W)"
    return normalize_image(img)


def normalize_image(img: np.ndarray) -> np.ndarray:
    """
    Normalize a 3D hyperspectral or 2D image to the range [0, 1].

    Parameters
    ----------
    hsi : np.ndarray
        Image array of any shape.

    Returns
    -------
    np.ndarray
        Normalized image, same shape as input.
    """
    min_val = img.min()
    max_val = img.max()

    # Avoid division by zero
    if max_val == min_val:
        return np.zeros_like(img)

    return (img - min_val) / (max_val - min_val)


def to_uint8(img: np.ndarray) -> np.ndarray:
    """
    Convert a 0–1 normalized HSI image to 0–255 uint8.

    Parameters
    ----------
    img : np.ndarray
        Normalized hyperspectral image, values in [0, 1].

    Returns
    -------
    np.ndarray
        HSI image in [0, 255], dtype=uint8.
    """
    assert img.min() >= 0 and img.max() <= 1, "Input image must be normalized to [0, 1]"

    return (img * 255).round().astype(np.uint8)
