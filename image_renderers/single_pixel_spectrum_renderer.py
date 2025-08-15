from typing import List, Tuple
import numpy as np
from InquirerPy import inquirer

import plotly.express as px
import plotly.graph_objects as go

from image_renderers.image_renderer import ImageRenderer


class SinglePixelSpectrumRenderer(ImageRenderer):
    """
    Pick a single (row, col) pixel and plot its spectrum across bands.
    """

    def __init__(self) -> None:
        pass

    def render(self, images: List[Tuple[str, np.ndarray]], output_dir: str):
        if not images:
            raise ValueError("No images provided.")

        # --- Validate and compute common index ranges ---
        Hs, Ws, Bs = [], [], []
        for name, img in images:
            if img.ndim != 3:
                raise ValueError(f"{name} must be a 3D array (H, W, B); got {img.shape}")
            Hs.append(img.shape[0])
            Ws.append(img.shape[1])
            Bs.append(img.shape[2])

        common_H = min(Hs)  # rows available in all images: [0..common_H-1]
        common_W = min(Ws)  # cols available in all images: [0..common_W-1]
        common_B = min(Bs)  # bands available in all images: [0..common_B-1]

        # --- Pick pixel (indices guaranteed to exist in every image) ---
        row_choices = [{"name": f"row y = {i}", "value": i} for i in range(common_H)]
        col_choices = [{"name": f"col x = {i}", "value": i} for i in range(common_W)]

        y = inquirer.fuzzy(  # type: ignore[reportPrivateImportUsage]
            message="Pick row (y):",
            choices=row_choices,
        ).execute()
        x = inquirer.fuzzy(  # type: ignore[reportPrivateImportUsage]
            message="Pick column (x):",
            choices=col_choices,
        ).execute()

        # --- Optional smoothing toggle ---
        smooth = inquirer.select(  # type: ignore[reportPrivateImportUsage]
            message="Apply light smoothing to spectrum?",
            choices=[{"name": "No", "value": False}, {"name": "Yes", "value": True}],
            default=False,
        ).execute()
        k = 5
        kernel = np.ones(k) / k
        pad = k // 2

        # --- Build figure with one trace per image ---
        bands = np.arange(1, common_B + 1)
        fig = go.Figure()
        for image_name, image in images:
            spectrum = image[y, x, :common_B].astype(np.float64)  # align bands
            smin, smax = float(spectrum.min()), float(spectrum.max())
            if smax == smin:
                spectrum_vis = np.zeros_like(spectrum, dtype=np.float64)
            else:
                spectrum_vis = (spectrum - smin) / (smax - smin)

            if smooth and common_B >= k:
                padded = np.pad(spectrum_vis, (pad, pad), mode="edge")
                spectrum_vis = np.convolve(padded, kernel, mode="valid")

            fig.add_trace(
                go.Scatter(
                    x=bands,
                    y=spectrum_vis,
                    mode="lines",
                    name=image_name,
                    fill="tozeroy",
                    line=dict(width=2),
                )
            )

        fig.update_layout(
            title=f"Spectrum at (x={x}, y={y})",
            xaxis_title="band",
            yaxis_title="normalized intensity",
        )
        fig.update_xaxes(dtick=max(1, common_B // 16))
        fig.show()
