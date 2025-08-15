from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Sequence, Callable, Protocol, Tuple
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from plot.trace_spec import TraceSpec
from plot.rich_plot_progress import patch_plotly_show


def assemble_linked_figure(
    rows: Sequence[TraceSpec],
    *,
    one_column: bool = True,
    coloraxis: Optional[dict] = None,
    figure_height_per_row: int = 350,
    min_height: int = 800,
    figure_width: int = 900,
    link_x_if_same_width: bool = True,
    show_x_spikes: bool = True,
    show_y_spikes: bool = True,
    link_3d_cameras: bool = True,
    link_3d_ranges: bool = True,
) -> go.Figure:
    """
    Compose a multi-row figure and apply unified hover/spike behavior for 2D,
    and unified camera/ranges for 3D. Works for heatmap, area/line, volume, isosurface, etc.

    Note: Plotly does NOT support true unified hover across 3D 'scene' subplots.
    We instead synchronize camera and axis ranges for a consistent experience.
    """
    patch_plotly_show(description="Opening Plotly figure…", ui="bar", bar_cycle_time=2.0, bar_steps=100, refresh_interval=0.05)

    n = len(rows)
    cols = 1 if one_column else 2  # easy extension later

    # Build per-row subplot specs: 'xy' for 2D rows, 'scene' for 3D rows
    specs = [[{"type": ("scene" if r.is_3d else "xy")}] for r in rows]

    fig = make_subplots(
        rows=n,
        cols=cols,
        specs=specs,
        subplot_titles=[r.title for r in rows],
        shared_xaxes=True,  # only affects 2D rows
        vertical_spacing=0.06,
    )

    # Add traces
    for i, spec in enumerate(rows, start=1):
        for tr in spec.traces:
            fig.add_trace(tr, row=i, col=1)

    # ===== 2D linking (xy) =====
    two_d_rows = [r for r in rows if not r.is_3d]
    same_widths = False
    if link_x_if_same_width and two_d_rows:
        widths = [w for w in (rs.width_hint for rs in two_d_rows) if w is not None]
        same_widths = (len(widths) == len(two_d_rows)) and (len(set(widths)) == 1)

    # Hide ticks for clean gallery feel (optional)
    if two_d_rows:
        fig.update_xaxes(showticklabels=False)
        fig.update_yaxes(showticklabels=False)

        if same_widths:
            fig.update_layout(hovermode="x unified", spikedistance=-1)
            fig.update_xaxes(matches="x", showspikes=show_x_spikes, spikemode="across", spikesnap="cursor")
        else:
            fig.update_layout(hovermode="closest", spikedistance=-1)
            fig.update_xaxes(showspikes=show_x_spikes, spikemode="across", spikesnap="cursor")
        if show_y_spikes:
            fig.update_yaxes(showspikes=True, spikemode="across", spikesnap="cursor")

    # ===== 3D linking (scenes) =====
    three_d_rows = [r for r in rows if r.is_3d]
    if three_d_rows:
        # Map: row index -> scene name (scene, scene2, ...)
        scene_names = []
        # Plotly names scenes in insertion order
        for k in fig.layout:
            if str(k).startswith("scene"):
                scene_names.append(str(k))
        # Keep only the first len(three_d_rows) scenes
        scene_names = scene_names[: len(three_d_rows)]

        # Derive unified camera/ranges from the first 3D row's hints (if provided)
        base_cam = three_d_rows[0].camera_hint
        base_xr = three_d_rows[0].xrange_hint
        base_yr = three_d_rows[0].yrange_hint
        base_zr = three_d_rows[0].zrange_hint

        for sn in scene_names:
            up = {}
            if link_3d_cameras and base_cam is not None:
                up.setdefault(sn, {})["camera"] = base_cam
            if link_3d_ranges:
                axes = {}
                if base_xr is not None:
                    axes.setdefault("xaxis", {})["range"] = list(base_xr)
                if base_yr is not None:
                    axes.setdefault("yaxis", {})["range"] = list(base_yr)
                if base_zr is not None:
                    axes.setdefault("zaxis", {})["range"] = list(base_zr)
                if axes:
                    up.setdefault(sn, {}).update(
                        {
                            "xaxis": axes.get("xaxis", {}),
                            "yaxis": axes.get("yaxis", {}),
                            "zaxis": axes.get("zaxis", {}),
                        }
                    )
            if up:
                fig.update_layout(**up)

        # Keep camera when interacting
        fig.update_layout(uirevision="linked-3d")

    # Size + coloraxis
    fig.update_layout(
        height=max(figure_height_per_row * n, min_height),
        width=figure_width,
    )
    if coloraxis is not None:
        fig.update_layout(coloraxis=coloraxis)

    return fig
