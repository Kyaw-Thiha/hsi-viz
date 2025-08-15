from typing import List, Tuple
import numpy as np
from InquirerPy import inquirer
from plotly.graph_objects import Figure


from image_renderers.image_renderer import ImageRenderer
from plot import build_area_row, assemble_linked_figure


def make_extra_hover(raw: np.ndarray, y_norm: np.ndarray):
    """
    Returns a callable suitable for build_area_row(extra_hover=...)
    that injects both raw and normalized values into hover.
    """
    raw = raw.astype(np.float32, copy=False)
    y_norm = y_norm.astype(np.float32, copy=False)

    def _fn(x: np.ndarray, y: np.ndarray):
        # y is the same as y_norm we pass to the builder; keep it explicit.
        # customdata must be shape (N, K). We'll put [raw, y_norm].
        custom = np.stack([raw, y_norm], axis=1)
        suffix = "<br>raw: %{customdata[0]:.4g}<br>y_norm: %{customdata[1]:.4f}"
        return custom, suffix

    return _fn


class SinglePixelSpectrumRenderer(ImageRenderer):
    """
    Pick a single (row, col) pixel and plot its spectrum across bands.
    """

    def __init__(self) -> None:
        pass

    def render(self, images: List[Tuple[str, np.ndarray]], output_dir: str) -> Figure:
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

        common_H = min(Hs)  # rows [0..common_H-1]
        common_W = min(Ws)  # cols [0..common_W-1]
        common_B = min(Bs)  # bands [0..common_B-1]

        # --- Pick pixel present in every image ---
        row_choices = [{"name": f"row y = {i}", "value": i} for i in range(common_H)]
        col_choices = [{"name": f"col x = {i}", "value": i} for i in range(common_W)]
        y = inquirer.fuzzy(  # type: ignore[reportPrivateImportUsage]
            message="Pick row (y):", choices=row_choices
        ).execute()
        x = inquirer.fuzzy(  # type: ignore[reportPrivateImportUsage]
            message="Pick column (x):", choices=col_choices
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

        # --- Build rows (one area chart per image) ---
        bands = np.arange(1, common_B + 1)
        rows = []
        for image_name, image in images:
            raw = image[y, x, :common_B].astype(np.float64)

            # Per‑spectrum min‑max normalize for visibility
            rmin, rmax = float(raw.min()), float(raw.max())
            y_norm = np.zeros_like(raw, dtype=np.float64) if rmax == rmin else (raw - rmin) / (rmax - rmin)

            if smooth and common_B >= k:
                padded = np.pad(y_norm, (pad, pad), mode="edge")
                y_norm = np.convolve(padded, kernel, mode="valid")

            # Let the builder create a linked‑hover row (width_hint=len(bands))
            rows.append(
                build_area_row(
                    name=f"{image_name} @ (x={x}, y={y})",
                    x=bands,
                    y=y_norm,
                    show_markers=False,
                    extra_hover=make_extra_hover(raw=raw, y_norm=y_norm),
                )
            )

        # --- Assemble figure with shared behavior across rows ---
        fig = assemble_linked_figure(
            rows,
            figure_height_per_row=280,
            min_height=420,
            figure_width=900,
            link_x_if_same_width=True,  # uses width_hint=len(bands) -> perfect x‑linking across images
            show_x_spikes=True,
            show_y_spikes=True,
        )

        # For spectra, axis ticks are meaningful — turn them back on and add labels
        fig.update_xaxes(showticklabels=True, title_text="band", dtick=max(1, common_B // 16))
        fig.update_yaxes(showticklabels=True, title_text="normalized intensity")

        fig.update_layout(title=f"Spectrum at (x={x}, y={y})")

        fig.show()

        return fig
