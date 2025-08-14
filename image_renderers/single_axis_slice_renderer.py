import numpy as np
from InquirerPy import inquirer
import plotly.express as px

from image_renderers.image_renderer import ImageRenderer
from image_normalization import normalize_image_2d
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

    def render(self, image: np.ndarray, image_name: str, output_dir: str):
        H, W, B = image.shape

        # --- Choose axis (optional, if you want to override at runtime) ---
        axis_choice = self.axis
        axis_choice = inquirer.select(  # type: ignore[reportPrivateImportUsage]
            message="Slice along which axis?",
            choices=[{"name": "x (fix column)", "value": "x"}, {"name": "y (fix row)", "value": "y"}],
            default=self.axis,
        ).execute()

        # --- Choose index along that axis ---
        if axis_choice == "x":
            idx_choices = [{"name": f"x = {i}", "value": i} for i in range(W)]
            idx = inquirer.fuzzy(  # type: ignore[reportPrivateImportUsage]
                message="Pick column index (x):",
                choices=idx_choices,
            ).execute()
            # Slice: (H, B)
            slice_2d = image[:, idx, :]
            y_label = "row (y)"
            x_label = "band"
        else:
            idx_choices = [{"name": f"y = {i}", "value": i} for i in range(H)]
            idx = inquirer.fuzzy(  # type: ignore[reportPrivateImportUsage]
                message="Pick row index (y):",
                choices=idx_choices,
            ).execute()
            # Slice: (W, B)
            slice_2d = image[idx, :, :]
            y_label = "column (x)"
            x_label = "band"

        # --- Pick palette ---
        color_choice = "gray"
        if self.color == "":
            color_choice = pick_mono_color()

        # --- Normalize per-slice (across the 2D array) ---
        slice_2d = normalize_image_2d(slice_2d)

        # --- Render heatmap (position vs. wavelength) ---
        # x-axis := band index 1..B, y-axis := spatial index
        fig = px.imshow(
            slice_2d,
            color_continuous_scale=color_choice,
            origin="upper",
            title=f"{'Column' if axis_choice == 'x' else 'Row'}-{idx} • {image_name}",
            aspect="auto",
        )
        fig.update_coloraxes(showscale=True)
        fig.update_xaxes(title_text=x_label, showticklabels=True)
        fig.update_yaxes(title_text=y_label, showticklabels=True)
        fig.show()
