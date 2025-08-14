import numpy as np
from InquirerPy import inquirer
import plotly.express as px

from image_normalization import normalize_image_2d, normalize_image_3d


class ImageRenderer:
    def render(self, image: np.ndarray, image_name: str, output_dir: str):
        pass


class SingleBandRenderer(ImageRenderer):
    show_colorbar: bool

    def __init__(self, show_colorbar: bool = True) -> None:
        self.show_colorbar = show_colorbar

    def render(self, image: np.ndarray, image_name: str, output_dir: str):
        band_choices = [{"name": band + 1, "value": band} for band in range(image.shape[2])]
        band_choice = inquirer.fuzzy(  # type: ignore[reportPrivateImportUsage]
            message="Select which spectral band you want to visualize: ",
            choices=band_choices,
        ).execute()

        image = image[:, :, band_choice]
        image = normalize_image_2d(image)  # Normalizing the image of the specific spectral band
        fig = px.imshow(
            image,
            color_continuous_scale="gray",
            origin="upper",
            title=f"Band-{band_choice + 1} of {image_name}",
        )
        fig.update_coloraxes(showscale=True)
        fig.update_xaxes(showticklabels=False).update_yaxes(showticklabels=False)
        fig.show()


class ImageViewer:
    renderer: ImageRenderer
    image: np.ndarray
    image_name: str
    output_dir: str

    def __init__(self, renderer: ImageRenderer, image: np.ndarray, image_name: str, output_dir: str) -> None:
        assert image.ndim == 3, f"The image is not 3 dimensional, and instead is of shape: {image.shape}"

        self.renderer = renderer
        self.image = normalize_image_3d(image)
        self.image_name = image_name
        self.output_dir = output_dir

    def set_renderer(self, renderer: ImageRenderer):
        self.renderer = renderer

    def render(self):
        self.renderer.render(self.image, self.image_name, self.output_dir)
