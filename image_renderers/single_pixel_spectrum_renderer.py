import numpy as np
from InquirerPy import inquirer
import plotly.express as px

from image_renderers.image_renderer import ImageRenderer
from image_normalization import normalize_image_2d
from utils.monocolor_picker import pick_mono_color


class SinglePixelSpectrumRenderer(ImageRenderer):
    """
    Pick a single (row, col) pixel and plot its spectrum across bands.
    """

    def __init__(self) -> None:
        pass

    def render(self, image: np.ndarray, image_name: str, output_dir: str):
        H, W, B = image.shape

        # --- Pick pixel ---
        row_choices = [{"name": f"row y = {i}", "value": i} for i in range(H)]
        col_choices = [{"name": f"col x = {i}", "value": i} for i in range(W)]

        y = inquirer.fuzzy(  # type: ignore[reportPrivateImportUsage]
            message="Pick row (y):",
            choices=row_choices,
        ).execute()
        x = inquirer.fuzzy(  # type: ignore[reportPrivateImportUsage]
            message="Pick column (x):",
            choices=col_choices,
        ).execute()

        # --- Extract spectrum and normalize to [0,1] for visualization ---
        spectrum = image[y, x, :].astype(np.float64)
        smin, smax = float(spectrum.min()), float(spectrum.max())
        spectrum_vis = (spectrum - smin) / (smax - smin + 1e-12)

        # Optional smoothing toggle
        smooth = inquirer.select(  # type: ignore[reportPrivateImportUsage]
            message="Apply light smoothing to spectrum?",
            choices=[{"name": "No", "value": False}, {"name": "Yes", "value": True}],
            default=False,
        ).execute()
        if smooth and B >= 5:
            # simple 5-point moving average (edges padded)
            k = 5
            pad = k // 2
            padded = np.pad(spectrum_vis, (pad, pad), mode="edge")
            spectrum_vis = np.convolve(padded, np.ones(k) / k, mode="valid")

        # --- Plot spectrum ---
        bands = np.arange(1, B + 1)
        fig = px.line(
            x=bands,
            y=spectrum_vis,
            labels={"x": "band", "y": "normalized intensity"},
            title=f"Spectrum at (x={x}, y={y}) • {image_name}",
        )
        # fig.update_traces(mode="lines+markers")
        fig.update_traces(
            mode="lines",  # just line, no markers
            fill="tozeroy",  # fill down to y=0
            line=dict(color="royalblue", width=2),
            fillcolor="rgba(65,105,225,0.4)",  # semi-transparent royal blue
        )
        fig.update_xaxes(dtick=max(1, B // 16))
        fig.show()
