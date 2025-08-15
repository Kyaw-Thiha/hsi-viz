from typing import List, Tuple
import numpy as np
from InquirerPy import inquirer
from plotly.graph_objects import Figure


from image_renderers.image_renderer import ImageRenderer
from utils.image_normalization import normalize_images_2d
from utils.monocolor_picker import pick_mono_color
from plot import build_heatmap_row, assemble_linked_figure


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

    def render(self, images: List[Tuple[str, np.ndarray]], output_dir: str) -> Figure:
        if not images:
            raise ValueError("No images provided.")

        # --- Validate & collect dims ---
        Hs, Ws, Bs = [], [], []
        for name, img in images:
            if img.ndim != 3:
                raise ValueError(f"{name} must be a 3D array (H, W, B); got {img.shape}")
            Hs.append(img.shape[0])
            Ws.append(img.shape[1])
            Bs.append(img.shape[2])

        common_H = min(Hs)
        common_W = min(Ws)

        # --- Choose axis and index present in all ---
        axis_choice = inquirer.select(  # type: ignore[reportPrivateImportUsage]
            message="Slice along which axis?",
            choices=[
                {"name": "x (fix column)", "value": "x"},
                {"name": "y (fix row)", "value": "y"},
            ],
            default=self.axis,
        ).execute()

        if axis_choice == "x":
            idx_choices = [{"name": f"x = {i}", "value": i} for i in range(common_W)]
            idx = inquirer.fuzzy(  # type: ignore[reportPrivateImportUsage]
                message="Pick column index (x):", choices=idx_choices
            ).execute()
            title_prefix = f"Column-{idx}"
        else:
            idx_choices = [{"name": f"y = {i}", "value": i} for i in range(common_H)]
            idx = inquirer.fuzzy(  # type: ignore[reportPrivateImportUsage]
                message="Pick row index (y):", choices=idx_choices
            ).execute()
            title_prefix = f"Row-{idx}"

        # --- Pick palette ---
        color_choice = self.color or pick_mono_color()

        # --- Extract slices as 2D arrays (pos, B) and normalize globally ---
        slices: List[Tuple[str, np.ndarray]] = []
        for name, img in images:
            if axis_choice == "x":
                sl = img[:, idx, :]  # (H, B)
            else:
                sl = img[idx, :, :]  # (W, B)
            slices.append((name, sl.astype(np.float32, copy=False)))

        # Normalize across all slices for fair comparison (returns same structure)
        norm_slices = normalize_images_2d(slices, method="percentile")

        # --- Build rows via the 2D heatmap builder ---
        rows = []
        for (name, sl2d_norm), (_, sl2d_orig) in zip(norm_slices, slices):
            # Title per row appears as the subplot title
            row_title = f"{title_prefix} • {name}"
            rows.append(
                build_heatmap_row(
                    name=row_title,
                    img_norm_2d=sl2d_norm,  # in [0,1]
                    img_orig_2d=sl2d_orig,  # show "Original" in hover
                    # zmin/zmax left at defaults since we're normalized
                )
            )

        # --- Assemble once: unified hover/spikes + shared colorbar ---
        fig = assemble_linked_figure(
            rows,
            coloraxis=dict(
                colorscale=color_choice,
                cmin=0.0,
                cmax=1.0,
                colorbar=dict(title="Normalized intensity"),
            ),
            figure_height_per_row=320,
            min_height=400,
            figure_width=800,
            link_x_if_same_width=True,  # uses width_hint=B to x-link if band counts match
            show_x_spikes=True,
            show_y_spikes=True,
        )

        # Optional: hide ticks like your original
        fig.update_xaxes(showticklabels=False)
        fig.update_yaxes(showticklabels=False)

        # Optional: figure-level title
        fig.update_layout(title_text=f"Slices at {('x' if axis_choice == 'x' else 'y')}={idx} (normalized globally)")

        fig.show()

        return fig
