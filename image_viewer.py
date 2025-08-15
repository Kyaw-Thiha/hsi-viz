from typing import List, Tuple
import numpy as np

from utils.image_normalization import normalize_images_3d
from image_renderers import ImageRenderer


class ImageViewer:
    renderer: ImageRenderer
    images: List[Tuple[str, np.ndarray]]
    output_dir: str

    def __init__(self, renderer: ImageRenderer, images: List[Tuple[str, np.ndarray]], output_dir: str) -> None:
        for image_name, image in images:
            assert image.ndim == 3, f"{image_name} is not 3 dimensional, and instead is of shape: {image.shape}"

        self.renderer = renderer
        self.images = images
        self.output_dir = output_dir

    def set_renderer(self, renderer: ImageRenderer):
        self.renderer = renderer

    def render(self):
        self.renderer.render(self.images, self.output_dir)
