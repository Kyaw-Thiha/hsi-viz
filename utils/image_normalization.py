from typing import List, Literal, Tuple
import numpy as np


# -------------------------------
# Wrapper to ensure the dimensions of the image to be 3d
# -------------------------------
def normalize_images_3d(
    images: List[Tuple[str, np.ndarray]], method: Literal["min-max", "percentile"]
) -> List[Tuple[str, np.ndarray]]:
    """
    Normalize a list of (filename, 3D hyperspectral image) tuples to the range [0, 1]
    using the specified method.

    Parameters
    ----------
    images : List[Tuple[str, np.ndarray]]
        List of tuples containing:
        - str : filename or identifier of the image
        - np.ndarray : 3D hyperspectral image array of shape (H, W, C),
          where H and W are spatial dimensions and C is the number of spectral bands.
    method : {"min-max", "percentile"}
        Normalization method:
        - "min-max": Scales all pixel values so that the global minimum becomes 0
          and the global maximum becomes 1.
        - "percentile": Scales based on low and high percentiles (default 2% and 98%),
          reducing the effect of extreme outliers. Values are clipped to [0, 1].

    Returns
    -------
    List[Tuple[str, np.ndarray]]
        List of tuples with the same filenames and normalized float32 images
        with the same shape as input, values in the range [0, 1].

    Raises
    ------
    AssertionError
        If any input image is not a 3D array of shape (H, W, C).
    """
    for i in range(len(images)):
        image = images[i][1]
        assert image.ndim == 3, f"Image-{i} must be a 3D array (H, W)"
    return normalize_images(images, method)


# -------------------------------
# Wrapper to ensure the dimensions of the image to be 2d
# -------------------------------
def normalize_images_2d(
    images: List[Tuple[str, np.ndarray]], method: Literal["min-max", "percentile"]
) -> List[Tuple[str, np.ndarray]]:
    """
    Normalize a list of (filename, 2D image) tuples to the range [0, 1]
    using the specified method.

    Parameters
    ----------
    images : List[Tuple[str, np.ndarray]]
        List of tuples containing:
        - str : filename or identifier of the image
        - np.ndarray : 2D image array of shape (H, W), typically representing
          a single band or grayscale image.
    method : {"min-max", "percentile"}
        Normalization method:
        - "min-max": Scales all pixel values so that the global minimum becomes 0
          and the global maximum becomes 1.
        - "percentile": Scales based on low and high percentiles (default 2% and 98%),
          reducing the effect of extreme outliers. Values are clipped to [0, 1].

    Returns
    -------
    List[Tuple[str, np.ndarray]]
        List of tuples with the same filenames and normalized float32 images
        with the same shape as input, values in the range [0, 1].

    Raises
    ------
    AssertionError
        If any input image is not a 2D array of shape (H, W).
    """
    for i in range(len(images)):
        image = images[i][1]
        assert image.ndim == 2, f"Image-{i} must be a 2D array (H, W)"
    return normalize_images(images, method)


# -------------------------------
# Main Normalization
# -------------------------------
def normalize_images(
    images: List[Tuple[str, np.ndarray]],
    method: Literal["min-max", "percentile"],
    low_percentile: float = 2.0,
    high_percentile: float = 98.0,
) -> List[Tuple[str, np.ndarray]]:
    """
    Normalize a list of (filename, image) tuples to the range [0, 1] using the specified method.

    Parameters
    ----------
    images : List[Tuple[str, np.ndarray]]
        List of tuples containing (filename, image array).
        Image can be grayscale (H, W), multi-channel (H, W, C), or hyperspectral (H, W, D).
    method : {"min-max", "percentile"}
        Normalization method:
        - "min-max": Global min becomes 0, max becomes 1.
        - "percentile": Scales based on low/high percentiles (2% and 98% by default).

    Returns
    -------
    List[Tuple[str, np.ndarray]]
        List of tuples with the same filenames and normalized float32 images in [0, 1].
    """
    names = [name for name, _ in images]
    imgs = [np.asarray(img, dtype=np.float32) for _, img in images]

    if method == "min-max":
        lo, hi = finite_min_max(imgs)
    elif method == "percentile":
        lo, hi = finite_percentiles(imgs, low_percentile, high_percentile)
    else:
        raise ValueError(f"Unknown normalization method: {method}")

    if hi <= lo:
        norm_imgs = [np.zeros_like(im, dtype=np.float32) for im in imgs]
    else:
        scale = hi - lo
        norm_imgs = []
        for im in imgs:
            nim = (im - lo) / scale
            if method == "percentile":
                nim = np.clip(nim, 0.0, 1.0)
            norm_imgs.append(nim.astype(np.float32, copy=False))

    return list(zip(names, norm_imgs))


# -------------------------------
# Helpers
# -------------------------------
def finite_min_max(imgs: List[np.ndarray]) -> Tuple[float, float]:
    """Compute global min/max ignoring NaNs and Infs."""
    gmin, gmax = np.inf, -np.inf
    for im in imgs:
        if im.size == 0:
            continue
        fim = im[np.isfinite(im)]
        if fim.size == 0:
            continue
        m, M = float(fim.min()), float(fim.max())
        if m < gmin:
            gmin = m
        if M > gmax:
            gmax = M
    if not np.isfinite(gmin):  # no finite pixels
        gmin, gmax = 0.0, 1.0
    return gmin, gmax


def finite_percentiles(
    imgs: List[np.ndarray],
    p_lo: float,
    p_hi: float,
    concat_element_limit: int = 50_000_000,  # ~200MB float32
    approx_bins: int = 4096,
) -> Tuple[float, float]:
    """Compute global percentiles ignoring NaNs/Infs, memory-safe for large datasets."""
    total = sum(int(im.size) for im in imgs)
    if total == 0:
        return 0.0, 1.0

    # Small total: exact computation
    if total <= concat_element_limit:
        flat = np.concatenate([im[np.isfinite(im)].ravel() for im in imgs], dtype=np.float32)
        if flat.size == 0:
            return 0.0, 1.0
        return float(np.percentile(flat, p_lo)), float(np.percentile(flat, p_hi))

    # Large total: histogram approximation
    gmin, gmax = finite_min_max(imgs)
    if gmax <= gmin:
        return 0.0, 1.0

    hist = np.zeros(approx_bins, dtype=np.float64)
    edges = np.linspace(gmin, gmax, approx_bins + 1, dtype=np.float64)
    for im in imgs:
        fim = im[np.isfinite(im)]
        if fim.size == 0:
            continue
        h, _ = np.histogram(fim, bins=edges)
        hist += h

    cdf = np.cumsum(hist)
    if cdf[-1] == 0:
        return 0.0, 1.0

    def p_from_hist(p):
        target = (p / 100.0) * cdf[-1]
        idx = int(np.searchsorted(cdf, target, side="left"))
        idx = np.clip(idx, 0, approx_bins - 1)
        return float(edges[idx])

    return p_from_hist(p_lo), p_from_hist(p_hi)
