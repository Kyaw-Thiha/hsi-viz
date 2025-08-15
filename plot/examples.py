# =========================
# Usage Examples
# =========================

from __future__ import annotations
from typing import List, Tuple
import numpy as np

from plot.builders_2d import build_heatmap_row, build_area_row
from plot.builders_3d import build_volume_row, build_isosurface_row, build_surface_row
from plot.figure_assembler import assemble_linked_figure


def example_heatmap_gallery(selected_bands: List[Tuple[str, np.ndarray]]):
    """
    selected_bands: [(name, band_2d)]
    Assume you've already normalized externally if desired.
    """
    rows = []
    for name, band in selected_bands:
        rows.append(build_heatmap_row(name, band, img_orig_2d=None))

    fig = assemble_linked_figure(
        rows,
        coloraxis=dict(colorscale="gray", cmin=0.0, cmax=1.0, colorbar=dict(title="Normalized intensity")),
    )
    fig.show()


def example_mixed_gallery():
    """
    One heatmap row + one area chart row — still linked-hover if width_hint matches (2D only).
    """
    img = np.random.rand(64, 128)  # H x W
    x = np.arange(128)
    y = np.sin(x / 10) + 0.1 * np.random.randn(128)

    rows = [
        build_heatmap_row("Band A", img),
        build_area_row("Profile at row 32", x, img[32, :]),
    ]

    fig = assemble_linked_figure(
        rows,
        coloraxis=dict(colorscale="gray", cmin=0.0, cmax=1.0, colorbar=dict(title="Norm")),
    )
    fig.show()


def example_3d_gallery():
    """Simple 3D examples (volume + isosurface + surface)."""
    vol = np.random.rand(30, 40, 50)  # (Z,Y,X)
    zsurf = np.random.rand(40, 50)

    rows = [
        build_volume_row("Random Volume", vol, surface_count=6, isomin=0.2, isomax=0.8),
        build_isosurface_row("Iso @ median", vol),
        build_surface_row("Random Surface", zsurf),
    ]

    fig = assemble_linked_figure(rows, link_3d_cameras=True, link_3d_ranges=True)
    fig.show()
