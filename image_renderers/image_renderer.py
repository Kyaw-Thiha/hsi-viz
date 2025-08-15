from typing import List, Optional, Tuple
import numpy as np
from plotly.graph_objects import Figure


class ImageRenderer:
    def render(self, images: List[Tuple[str, np.ndarray]], output_dir: str) -> Optional[Figure]:
        pass
