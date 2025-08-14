import numpy as np

from image_normalization import normalize_image_3d
from image_renderers import ImageRenderer


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
