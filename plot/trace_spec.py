from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple
from plotly.basedatatypes import BaseTraceType

# =========================
# Core UI / Layout Utilities (2D + 3D)
# =========================


@dataclass
class TraceSpec:
    """
    A renderer-agnostic trace description for one subplot row.
    - traces: one or more Plotly traces to add to this row
    - title: title for this subplot row
    - width_hint: used to decide if 2D x-axes can be perfectly "matched"
                  (e.g., heatmap width = image width; scatter width = len(x))
    - is_3d: whether this row contains 3D traces (scenes). Mixed rows are allowed but
             if any trace is 3D, treat row as 3D.
    - x/y/z range hints: optional ranges to unify across 3D scenes.
    - camera_hint: optional camera dict to copy across 3D scenes.
    """

    traces: Sequence[BaseTraceType]
    title: str
    width_hint: Optional[int] = None
    is_3d: bool = False
    xrange_hint: Optional[Tuple[float, float]] = None
    yrange_hint: Optional[Tuple[float, float]] = None
    zrange_hint: Optional[Tuple[float, float]] = None
    camera_hint: Optional[dict] = None
