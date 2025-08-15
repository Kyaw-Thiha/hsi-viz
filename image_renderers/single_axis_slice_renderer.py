from typing import List, Tuple, Literal
import numpy as np
from InquirerPy import inquirer
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go


from image_renderers.image_renderer import ImageRenderer
from utils.image_normalization import normalize_images_2d
from utils.monocolor_picker import pick_mono_color


class SingleAxisSliceRenderer(ImageRenderer):
    """
    Render an (axis-fixed) slice of the cube:
      - axis='x'  -> fix a column index, show (H, B) heatmap
      - axis='y'  -> fix a row index,    show (W, B) heatmap
    """

    axis: str = "x"  # 'x' or 'y'
    color: str = ""

    def __init__(self, axis: str = "x", color: str = "") -> None:
        assert axis in ("x", "y"), "axis must be 'x' or 'y'"
        self.axis = axis
        self.color = color

    def render(self, images: List[Tuple[str, np.ndarray]], output_dir: str):
        if not images:
            raise ValueError("No images provided.")

        # --- Validate & find common spatial index ranges across all images ---
        Hs, Ws, Bs = [], [], []
        for name, img in images:
            if img.ndim != 3:
                raise ValueError(f"{name} must be a 3D array (H, W, B); got {img.shape}")
            Hs.append(img.shape[0])
            Ws.append(img.shape[1])
            Bs.append(img.shape[2])

        common_H = min(Hs)  # indices [0..common_H-1] exist in all images (rows)
        common_W = min(Ws)  # indices [0..common_W-1] exist in all images (cols)

        # --- Choose axis ---
        axis_choice = inquirer.select(  # type: ignore[reportPrivateImportUsage]
            message="Slice along which axis?",
            choices=[
                {"name": "x (fix column)", "value": "x"},
                {"name": "y (fix row)", "value": "y"},
            ],
            default=self.axis,
        ).execute()

        # --- Choose index guaranteed to exist in ALL images ---
        if axis_choice == "x":
            idx_choices = [{"name": f"x = {i}", "value": i} for i in range(common_W)]
            idx = inquirer.fuzzy(  # type: ignore[reportPrivateImportUsage]
                message="Pick column index (x):", choices=idx_choices
            ).execute()
            x_label, y_label = "band", "row (y)"
        else:
            idx_choices = [{"name": f"y = {i}", "value": i} for i in range(common_H)]
            idx = inquirer.fuzzy(  # type: ignore[reportPrivateImportUsage]
                message="Pick row index (y):", choices=idx_choices
            ).execute()
            x_label, y_label = "band", "column (x)"

        # --- Pick palette ---
        color_choice = self.color or pick_mono_color()

        # --- Build 2D slices from all images (each shape: (pos, B)) ---
        slices: List[Tuple[str, np.ndarray]] = []
        for name, img in images:
            if axis_choice == "x":
                sl = img[:, idx, :]  # (H, B)
            else:
                sl = img[idx, :, :]  # (W, B)
            # Convert to float32 for normalization downstream
            slices.append((name, sl.astype(np.float32, copy=False)))

        # --- Normalize globally across all slices for fair comparison ---
        # normalize_images_2d expects List[(name, 2D array)] and returns same structure
        norm_slices = normalize_images_2d(slices, method="percentile")

        # --- Plot all on one page using subplots ---
        n = len(norm_slices)
        cols = min(n, 3)  # up to 3 per row (tweak as you like)
        rows = (n + cols - 1) // cols

        fig = make_subplots(
            rows=rows,
            cols=cols,
            subplot_titles=[f"{'Column' if axis_choice == 'x' else 'Row'}-{idx} • {name}" for name, _ in norm_slices],
        )

        for i, (name, sl2d) in enumerate(norm_slices):
            # Ensure 2D array is (pos, band); plotly's Image accepts 2D z
            row = (i // cols) + 1
            col = (i % cols) + 1
            fig.add_trace(go.Heatmap(z=sl2d, coloraxis="coloraxis", zmin=0.0, zmax=1.0), row=row, col=col)

        # Hide ticks (they’re not meaningful for raw indices)
        fig.update_xaxes(showticklabels=False)
        fig.update_yaxes(showticklabels=False)

        # Apply a single continuous color scale name to all traces
        # (go.Image does not take color scales like px.imshow; it auto-maps grayscale.)
        # If you need a specific mono palette, convert sl2d to RGB yourself or use Heatmap:
        # fig.add_trace(go.Heatmap(z=sl2d, colorscale=color_choice), ...)

        fig.update_layout(
            height=max(320 * rows, 400),
            width=max(320 * cols, 500),
            title_text=f"Slices at {('x' if axis_choice == 'x' else 'y')}={idx} (normalized globally)",
            coloraxis=dict(colorscale=color_choice, cmin=0.0, cmax=1.0, colorbar=dict(title="Normalized intensity")),
        )

        fig.show()
