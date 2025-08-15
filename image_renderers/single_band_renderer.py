from typing import List, Tuple
import numpy as np
from InquirerPy import inquirer

from plotly.subplots import make_subplots
import plotly.graph_objects as go

from image_renderers.image_renderer import ImageRenderer
from utils.image_normalization import normalize_images_2d
from utils.monocolor_picker import pick_mono_color


class SingleBandRenderer(ImageRenderer):
    color: str = ""

    def __init__(self, color: str = "") -> None:
        self.color = color

    def render(self, images: List[Tuple[str, np.ndarray]], output_dir: str):
        # --- Validate inputs ---
        for img_name, img in images:
            if img is None or img.ndim != 3:
                raise ValueError(f"{img_name} must be a 3D array; got shape {None if img is None else img.shape}")

        # --- Ask user to pick a band for each image ---
        selected_bands: List[Tuple[str, np.ndarray]] = []

        for img_name, img in images:
            num_bands = img.shape[2]

            if num_bands == 1:
                # Auto-pick the only band
                selected_bands.append((img_name, img[:, :, 0]))
                continue

            band_choices = [{"name": f"Band {b + 1} — {img_name}", "value": b} for b in range(num_bands)]

            default_band = num_bands // 2  # middle band as a reasonable default
            band_choice = inquirer.fuzzy(  # type: ignore[reportPrivateImportUsage]
                message=f"Select spectral band for: {img_name} (total {num_bands} bands)",
                choices=band_choices,
                default=default_band,
            ).execute()

            band_idx = int(band_choice)
            selected_bands.append((img_name, img[:, :, band_idx]))

        # --- Pick color palette ---
        color_choice = "gray"
        if self.color == "":
            color_choice = pick_mono_color()

        # Normalize (2D bands)
        to_plot_norm = normalize_images_2d(selected_bands, "percentile")  # [(name, norm_img)]
        to_plot_orig = selected_bands  # [(name, orig_img)] same order as above

        # Force ONE column layout
        n_images = len(to_plot_norm)
        cols = 1
        rows = n_images  # one image per row

        # Determine if all images have the same width (needed for perfect x-linked hover)
        widths = [img.shape[1] for _, img in to_plot_norm]
        same_widths = len(set(widths)) == 1

        fig = make_subplots(
            rows=rows,
            cols=cols,
            subplot_titles=[f"Spectral Band of {name}" for name, _ in to_plot_norm],
            shared_xaxes=True,  # share x; hover/pan link if widths match
            vertical_spacing=0.06,
        )

        for i, ((name_n, img_norm), (name_o, img_orig)) in enumerate(zip(to_plot_norm, to_plot_orig)):
            assert name_n == name_o, "Order mismatch between normalized and original lists"
            row = i + 1

            H, W = img_norm.shape
            # Give all traces explicit x/y coordinates (helps Plotly align hover/pan)
            x = np.arange(W)
            y = np.arange(H)

            # Make original value available for hover (if want only normalized, you can skip customdata)
            # shape must be (H, W, K). We'll put a single channel (original value) in customdata.
            customdata = img_orig[..., None].astype(np.float32)

            # Custom hovertemplate: no 'z' label; you control everything
            # %{z} is normalized; %{customdata[0]} is original
            hovertemplate = (
                f"<b>{name_n}</b><br>"
                "x=%{x}, y=%{y}<br>"
                "Normalized: %{z:.4f}<br>"
                "Original: %{customdata[0]:.4g}"
                "<extra></extra>"
            )

            fig.add_trace(
                go.Heatmap(
                    z=img_norm,
                    x=x,
                    y=y,
                    coloraxis="coloraxis",
                    zmin=0.0,
                    zmax=1.0,
                    customdata=customdata,
                    hovertemplate=hovertemplate,  # replaces default 'z' hover
                ),
                row=row,
                col=1,
            )

        # Hide ticks for a clean gallery
        fig.update_xaxes(showticklabels=False)
        fig.update_yaxes(showticklabels=False)

        # --- Linked hover / spikes ---
        # Perfect “linked” hover across subplots requires same x-range/width.
        if same_widths:
            # Single unified hoverbox along x across all rows
            fig.update_layout(hovermode="x unified", spikedistance=-1)
            fig.update_xaxes(matches="x", showspikes=True, spikemode="across", spikesnap="cursor")
        else:
            # Fallback: closest hover, still draw crosshair spikes across subplots
            # (Unifying x doesn't behave well with different widths)
            fig.update_layout(hovermode="closest", spikedistance=-1)
            # Keep spikes to give a “linked cursor” feel
            fig.update_xaxes(showspikes=True, spikemode="across", spikesnap="cursor")

        # Optional: also add y spikes
        fig.update_yaxes(showspikes=True, spikemode="across", spikesnap="cursor")

        # Figure sizing
        fig.update_layout(
            height=800 if rows == 1 else max(350 * rows, 800),
            width=800,
            coloraxis=dict(colorscale=color_choice, cmin=0.0, cmax=1.0, colorbar=dict(title="Normalized intensity")),
        )

        # Show ONCE
        fig.show()
