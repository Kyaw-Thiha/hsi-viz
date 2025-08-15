from typing import List, Tuple
import numpy as np
from InquirerPy import inquirer
from plotly.graph_objects import Figure

from image_renderers.image_renderer import ImageRenderer
from utils.image_normalization import normalize_images_2d
from utils.monocolor_picker import pick_mono_color
from plot import build_heatmap_row, assemble_linked_figure


class SingleBandRenderer(ImageRenderer):
    color: str = ""

    def __init__(self, color: str = "") -> None:
        self.color = color

    def render(self, images: List[Tuple[str, np.ndarray]], output_dir: str) -> Figure:
        # --- Validate inputs ---
        for img_name, img in images:
            if img is None or img.ndim != 3:
                raise ValueError(f"{img_name} must be a 3D array; got shape {None if img is None else img.shape}")

        # --- Ask user to pick a band for each image ---
        selected_bands: List[Tuple[str, np.ndarray]] = []
        for img_name, img in images:
            num_bands = img.shape[2]
            if num_bands == 1:
                selected_bands.append((img_name, img[:, :, 0]))
                continue

            band_choices = [{"name": f"Band {b + 1} — {img_name}", "value": b} for b in range(num_bands)]
            default_band = num_bands // 2
            band_choice = inquirer.fuzzy(  # type: ignore[reportPrivateImportUsage]
                message=f"Select spectral band for: {img_name} (total {num_bands} bands)",
                choices=band_choices,
                default=default_band,
            ).execute()
            band_idx = int(band_choice)
            selected_bands.append((img_name, img[:, :, band_idx]))

        # --- Pick color palette ---
        color_choice = self.color or pick_mono_color()

        # --- Normalize (2D bands) ---
        # to_plot_norm: List[(name, norm_img)], in [0,1]
        to_plot_norm = normalize_images_2d(selected_bands, "percentile")
        to_plot_orig = selected_bands  # same order

        # --- Build renderer-agnostic rows (one heatmap row per image) ---
        rows = []
        for (name_n, img_norm), (name_o, img_orig) in zip(to_plot_norm, to_plot_orig):
            assert name_n == name_o, "Order mismatch between normalized and original lists"
            # Delegate hover + width_hint handling to the builder
            rows.append(
                build_heatmap_row(
                    name=name_n,
                    img_norm_2d=img_norm,
                    img_orig_2d=img_orig,  # enables "Original" value in hover
                    # zmin/zmax left at defaults since we normalized to [0,1]
                )
            )

        # --- Assemble figure with linked hover (2D) ---
        fig = assemble_linked_figure(
            rows,
            coloraxis=dict(
                colorscale=color_choice,
                cmin=0.0,
                cmax=1.0,
                colorbar=dict(title="Normalized intensity"),
            ),
            figure_height_per_row=350,
            min_height=800,
            figure_width=800,
            link_x_if_same_width=True,  # will auto-check width hints for perfect x-linking
            show_x_spikes=True,
            show_y_spikes=True,
        )

        # (Optional) If you still want to hide ticks like before:
        fig.update_xaxes(showticklabels=False)
        fig.update_yaxes(showticklabels=False)

        return fig
