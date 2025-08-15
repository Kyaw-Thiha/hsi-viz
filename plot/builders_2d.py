# =========================
# 2D Builders
# =========================

from __future__ import annotations
from typing import Optional, Callable, Tuple
import numpy as np
import plotly.graph_objects as go

from plot.trace_spec import TraceSpec


def build_heatmap_row(
    name: str,
    img_norm_2d: np.ndarray,  # H x W in [0,1]
    img_orig_2d: Optional[np.ndarray] = None,  # H x W original values (for hover), optional
    *,
    zmin: float = 0.0,
    zmax: float = 1.0,
) -> TraceSpec:
    H, W = img_norm_2d.shape
    x = np.arange(W)
    y = np.arange(H)

    customdata = None
    hovertemplate = f"<b>{name}</b><br>x=%{{x}}, y=%{{y}}<br>Norm: %{{z:.4f}}<extra></extra>"
    if img_orig_2d is not None:
        customdata = img_orig_2d[..., None].astype(np.float32)  # (H,W,1)
        hovertemplate = (
            f"<b>{name}</b><br>x=%{{x}}, y=%{{y}}<br>Normalized: %{{z:.4f}}<br>Original: %{{customdata[0]:.4g}}<extra></extra>"
        )

    hm = go.Heatmap(
        z=img_norm_2d,
        x=x,
        y=y,
        coloraxis="coloraxis",
        zmin=zmin,
        zmax=zmax,
        customdata=customdata,
        hovertemplate=hovertemplate,
    )

    return TraceSpec(
        traces=[hm],
        title=f"Spectral Band of {name}",
        width_hint=W,  # crucial for deciding x-axis linking
    )


def build_area_row(
    name: str,
    x: np.ndarray,  # 1D
    y: np.ndarray,  # 1D
    *,
    show_markers: bool = False,
    extra_hover: Optional[Callable[[np.ndarray, np.ndarray], Tuple[np.ndarray, str]]] = None,
) -> TraceSpec:
    """
    Generic area chart row (go.Scatter with fill='tozeroy') with optional customdata and hovertemplate.
    - extra_hover: a function that takes (x, y) and returns (customdata, hovertemplate_suffix)
      so each renderer can append domain-specific details (e.g., raw vs normalized).
    """
    if extra_hover is not None:
        customdata, suffix = extra_hover(x, y)
        hovertemplate = f"<b>{name}</b><br>x=%{{x}}<br>y=%{{y:.4g}}" + suffix + "<extra></extra>"
    else:
        customdata = None
        hovertemplate = f"<b>{name}</b><br>x=%{{x}}<br>y=%{{y:.4g}}<extra></extra>"

    sc = go.Scatter(
        x=x,
        y=y,
        mode="lines+markers" if show_markers else "lines",
        fill="tozeroy",
        customdata=customdata,
        hovertemplate=hovertemplate,
    )

    return TraceSpec(
        traces=[sc],
        title=name,
        width_hint=len(x),
    )
