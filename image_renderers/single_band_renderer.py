import numpy as np
from InquirerPy import inquirer
import plotly.express as px

from image_renderers.image_renderer import ImageRenderer
from image_normalization import normalize_image_2d
from utils.monocolor_picker import pick_mono_color


class SingleBandRenderer(ImageRenderer):
    color: str = ""

    def __init__(self, color: str = "") -> None:
        self.color = color

    def render(self, image: np.ndarray, image_name: str, output_dir: str):
        # --- Pick band ---
        band_choices = [{"name": band + 1, "value": band} for band in range(image.shape[2])]
        band_choice = inquirer.fuzzy(  # type: ignore[reportPrivateImportUsage]
            message="Select which spectral band you want to visualize: ",
            choices=band_choices,
        ).execute()

        # --- Pick color palette ---
        color_choice = "gray"
        if self.color == "":
            color_choice = pick_mono_color()

        # --- Render the image ---
        image = image[:, :, band_choice]
        image = normalize_image_2d(image)  # Normalizing the image of the specific spectral band
        fig = px.imshow(
            image,
            # color_continuous_scale="gray",
            color_continuous_scale=color_choice,
            origin="upper",
            title=f"Band-{band_choice + 1} of {image_name}",
        )
        fig.update_coloraxes(showscale=True)
        fig.update_xaxes(showticklabels=False).update_yaxes(showticklabels=False)
        fig.show()
