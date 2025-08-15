# =========================
# 3D Builders
# =========================
from __future__ import annotations
from typing import Optional, Tuple
import numpy as np
import plotly.graph_objects as go

from plot.trace_spec import TraceSpec


def _regular_coords(nx: int, ny: int, nz: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate regular-grid coordinates for a (Z,Y,X) = (nz, ny, nx) volume."""
    x = np.arange(nx)
    y = np.arange(ny)
    z = np.arange(nz)
    return x, y, z


def build_volume_row(
    name: str,
    vol: np.ndarray,  # shape (Z, Y, X) scalar field
    *,
    isomin: Optional[float] = None,
    isomax: Optional[float] = None,
    surface_count: int = 8,
    caps: dict = dict(x_show=False, y_show=False, z_show=False),
    colorscale: Optional[str] = None,  # None -> inherit layout coloraxis
) -> TraceSpec:
    assert vol.ndim == 3, f"Volume must be 3D (Z,Y,X); got {vol.shape}"
    nz, ny, nx = vol.shape
    x, y, z = _regular_coords(nx, ny, nz)
    X, Y, Z = np.meshgrid(x, y, z, indexing="xy")  # shape (ny, nx, nz)
    # Reorder to match vol (Z,Y,X): we'll ravel in the same order
    Xr = np.transpose(X, (2, 0, 1)).ravel()
    Yr = np.transpose(Y, (2, 0, 1)).ravel()
    Zr = np.transpose(Z, (2, 0, 1)).ravel()
    Vr = vol.ravel()

    iso_min = np.min(Vr) if isomin is None else isomin
    iso_max = np.max(Vr) if isomax is None else isomax

    trace = go.Volume(
        x=Xr,
        y=Yr,
        z=Zr,
        value=Vr,
        isomin=float(iso_min),
        isomax=float(iso_max),
        surface_count=surface_count,
        caps=caps,
        colorscale=colorscale,  # if None, Plotly default is used; coloraxis doesn't drive Volume
        opacity=0.1,
        hovertemplate=(f"<b>{name}</b><br>x=%{{x}}, y=%{{y}}, z=%{{z}}<br>v=%{{value:.4g}}<extra></extra>"),
    )

    return TraceSpec(
        traces=[trace],
        title=f"Volume: {name}",
        is_3d=True,
        width_hint=nx,
        xrange_hint=(float(x.min()), float(x.max())),
        yrange_hint=(float(y.min()), float(y.max())),
        zrange_hint=(float(z.min()), float(z.max())),
        camera_hint=dict(eye=dict(x=1.6, y=1.6, z=1.6)),
    )


def build_isosurface_row(
    name: str,
    vol: np.ndarray,  # shape (Z, Y, X)
    *,
    level: Optional[float] = None,  # if None, use median
    colorscale: Optional[str] = None,
    caps: dict = dict(x_show=False, y_show=False, z_show=False),
) -> TraceSpec:
    assert vol.ndim == 3, f"Volume must be 3D (Z,Y,X); got {vol.shape}"
    nz, ny, nx = vol.shape
    x, y, z = _regular_coords(nx, ny, nz)
    X, Y, Z = np.meshgrid(x, y, z, indexing="xy")
    Xr = np.transpose(X, (2, 0, 1)).ravel()
    Yr = np.transpose(Y, (2, 0, 1)).ravel()
    Zr = np.transpose(Z, (2, 0, 1)).ravel()
    Vr = vol.ravel()

    lvl = float(np.median(Vr)) if level is None else float(level)

    trace = go.Isosurface(
        x=Xr,
        y=Yr,
        z=Zr,
        value=Vr,
        isomin=lvl,
        isomax=lvl,
        surface_count=1,
        caps=caps,
        colorscale=colorscale,
        hovertemplate=(f"<b>{name}</b><br>x=%{{x}}, y=%{{y}}, z=%{{z}}<br>v=%{{value:.4g}}<extra></extra>"),
    )

    return TraceSpec(
        traces=[trace],
        title=f"Isosurface: {name} @ {lvl:.4g}",
        is_3d=True,
        width_hint=nx,
        xrange_hint=(float(x.min()), float(x.max())),
        yrange_hint=(float(y.min()), float(y.max())),
        zrange_hint=(float(z.min()), float(z.max())),
        camera_hint=dict(eye=dict(x=1.6, y=1.6, z=1.6)),
    )


def build_surface_row(
    name: str,
    z_surf: np.ndarray,  # 2D array defining a surface height over a grid
    *,
    x: Optional[np.ndarray] = None,
    y: Optional[np.ndarray] = None,
) -> TraceSpec:
    H, W = z_surf.shape
    if x is None:
        x = np.arange(W)
    if y is None:
        y = np.arange(H)

    trace = go.Surface(
        z=z_surf,
        x=x,
        y=y,
        hovertemplate=(f"<b>{name}</b><br>x=%{{x}}, y=%{{y}}, z=%{{z:.4g}}<extra></extra>"),
        showscale=False,
        contours=dict(z=dict(show=True, usecolormap=True, highlight=True)),
    )

    if x is not None and y is not None:
        return TraceSpec(
            traces=[trace],
            title=f"Surface: {name}",
            is_3d=True,
            width_hint=W,
            xrange_hint=(float(x.min()), float(x.max())),
            yrange_hint=(float(y.min()), float(y.max())),
            zrange_hint=(float(np.min(z_surf)), float(np.max(z_surf))),
            camera_hint=dict(eye=dict(x=1.6, y=1.6, z=1.6)),
        )
    else:
        return TraceSpec(
            traces=[trace],
            title=f"Surface: {name}",
            is_3d=True,
            width_hint=W,
            zrange_hint=(float(np.min(z_surf)), float(np.max(z_surf))),
            camera_hint=dict(eye=dict(x=1.6, y=1.6, z=1.6)),
        )
