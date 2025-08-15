from typing import List, Tuple, Optional, Literal, Dict, Any
import numpy as np
import plotly.graph_objects as go

from image_renderers.image_renderer import ImageRenderer
from utils.image_normalization import normalize_images_3d
from utils.monocolor_picker import pick_mono_color
from plot import assemble_linked_figure, build_volume_row


class VolumeRenderer(ImageRenderer):
    """
    Render HSI cubes (H, W, B) as 3D volumes using Plotly's go.Volume.

    Each input image is normalized independently to [0,1] using the
    preexisting normalize_images_3d(). Bands are treated as the Z axis (depth).
    Automatically downsamples large volumes to keep the voxel count within
    browser/WebGL limits. Also supports percentile-based iso-bounds that ignore
    zeros (common in masked HSI like Pavia), to avoid a visually "empty" cube.
    """

    norm: Literal["percentile", "min-max"]
    surface_count: int
    opacity: float
    colorscale: Optional[str]
    caps: Dict[str, Any]
    figure_height_per_row: int
    figure_width: int
    min_height: int

    # Debugging Options
    renderer: Optional[str]
    save_html: Optional[str]
    debug: bool

    # Downsampling Options
    max_voxels: int
    ds_method: Literal["subsample", "mean"]

    # Iso selection
    iso_strategy: Literal["full_range", "percentile", "percentile_nonzero"]
    iso_percentiles: Tuple[float, float]  # used by percentile strategies
    ignore_zeros_threshold: float  # if >=n% zeros, treat as masked and ignore zeros for iso percentiles

    def __init__(
        self,
        *,
        norm: Literal["percentile", "min-max"] = "percentile",
        surface_count: int = 8,
        opacity: float = 0.25,
        colorscale: Optional[str] = "Viridis",
        caps: Optional[Dict[str, Any]] = None,  # e.g. dict(x_show=False, y_show=False, z_show=False)
        figure_height_per_row: int = 420,
        figure_width: int = 900,
        min_height: int = 800,
        renderer: Optional[str] = None,  # e.g. "browser", "notebook_connected", "vscode"
        save_html: Optional[str] = None,  # e.g. "volume_debug.html"
        debug: bool = False,
        # Downsampling controls
        max_voxels: int = 2_000_000,
        ds_method: Literal["subsample", "mean"] = "subsample",
        # Iso selection
        iso_strategy: Literal["full_range", "percentile", "percentile_nonzero"] = "percentile_nonzero",
        iso_percentiles: Tuple[float, float] = (5.0, 99.0),  # used by percentile strategies
        ignore_zeros_threshold: float = 0.05,  # if >=5% zeros, treat as masked and ignore zeros for iso percentiles
    ) -> None:
        self.norm = norm
        self.surface_count = surface_count
        self.opacity = opacity
        self.colorscale = colorscale
        self.caps = caps if caps is not None else dict(x_show=False, y_show=False, z_show=False)
        self.figure_height_per_row = figure_height_per_row
        self.figure_width = figure_width
        self.min_height = min_height
        self.renderer = renderer
        self.save_html = save_html
        self.debug = debug
        self.max_voxels = int(max_voxels)
        self.ds_method = ds_method
        self.iso_strategy = iso_strategy
        self.iso_percentiles = iso_percentiles
        self.ignore_zeros_threshold = ignore_zeros_threshold

    @staticmethod
    def _summarize(name: str, arr: np.ndarray) -> str:
        finite = np.isfinite(arr)
        pct_finite = 100.0 * finite.mean()
        with np.errstate(invalid="ignore"):
            vmin = float(np.nanmin(arr)) if np.any(finite) else float("nan")
            vmax = float(np.nanmax(arr)) if np.any(finite) else float("nan")
            vmean = float(np.nanmean(arr)) if np.any(finite) else float("nan")
            vstd = float(np.nanstd(arr)) if np.any(finite) else float("nan")
        return (
            f"[{name}] shape={arr.shape} finite={pct_finite:.1f}% min={vmin:.4g} max={vmax:.4g} mean={vmean:.4g} std={vstd:.4g}"
        )

    def _downsample(self, vol_bhw: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int, int]]:
        B, H, W = vol_bhw.shape
        vox = B * H * W
        if vox <= self.max_voxels:
            return vol_bhw, (1, 1, 1)
        import math

        scale = (vox / self.max_voxels) ** (1 / 3)
        sB = max(1, int(math.floor(scale)))
        sH = max(1, int(math.floor(scale)))
        sW = max(1, int(math.floor(scale)))

        def size_after(sB, sH, sW):
            return ((B + sB - 1) // sB) * ((H + sH - 1) // sH) * ((W + sW - 1) // sW)

        while size_after(sB, sH, sW) > self.max_voxels:
            dims = [(B / sB, "B"), (H / sH, "H"), (W / sW, "W")]
            dims.sort(reverse=True)
            if dims[0][1] == "B":
                sB += 1
            elif dims[0][1] == "H":
                sH += 1
            else:
                sW += 1
        if self.ds_method == "subsample":
            ds = vol_bhw[::sB, ::sH, ::sW]
        else:
            b = (B // sB) * sB
            h = (H // sH) * sH
            w = (W // sW) * sW
            cropped = vol_bhw[:b, :h, :w]
            ds = cropped.reshape(b // sB, sB, h // sH, sH, w // sW, sW).mean(axis=(1, 3, 5))
        return ds.astype(np.float32, copy=False), (sB, sH, sW)

    def _auto_iso_bounds(self, vol: np.ndarray) -> Tuple[float, float, Dict[str, float]]:
        """Compute (isomin, isomax) based on strategy; return extra stats for debug."""
        flat = vol.ravel()
        flat = flat[np.isfinite(flat)]
        dbg: Dict[str, float] = {}
        if flat.size == 0:
            return 0.0, 1.0, dbg
        zero_frac = float((flat == 0).mean())
        dbg["zero_frac"] = zero_frac
        p_lo, p_hi = self.iso_percentiles

        if self.iso_strategy == "full_range":
            iso_min, iso_max = float(np.min(flat)), float(np.max(flat))
        elif self.iso_strategy == "percentile":
            iso_min = float(np.percentile(flat, p_lo))
            iso_max = float(np.percentile(flat, p_hi))
        else:  # percentile_nonzero
            if zero_frac >= self.ignore_zeros_threshold:
                nz = flat[flat > 0]
                if nz.size == 0:
                    iso_min, iso_max = 0.0, 1.0
                else:
                    iso_min = float(np.percentile(nz, p_lo))
                    iso_max = float(np.percentile(nz, p_hi))
            else:
                iso_min = float(np.percentile(flat, p_lo))
                iso_max = float(np.percentile(flat, p_hi))
        # Guard rails
        if not np.isfinite(iso_min) or not np.isfinite(iso_max) or iso_max <= iso_min:
            iso_min, iso_max = 0.0, 1.0
        return max(0.0, min(1.0, iso_min)), max(0.0, min(1.0, iso_max)), dbg

    def render(self, images: List[Tuple[str, np.ndarray]], output_dir: str):
        if not images:
            raise ValueError("No images provided.")
        for img_name, img in images:
            if img is None or img.ndim != 3:
                raise ValueError(f"{img_name} must be a 3D array (H,W,B); got {None if img is None else img.shape}")

        common_H = min(img.shape[0] for _, img in images)
        common_W = min(img.shape[1] for _, img in images)
        common_B = min(img.shape[2] for _, img in images)

        def _crop(img: np.ndarray) -> np.ndarray:
            return img[:common_H, :common_W, :common_B]

        cropped_images = [(name, _crop(img)) for name, img in images]
        normalized_images = normalize_images_3d(cropped_images, self.norm)

        if self.renderer is not None:
            import plotly.io as pio

            pio.renderers.default = self.renderer

        rows = []
        for name, vol_hwb in normalized_images:
            if self.debug:
                print(self._summarize(name + "(pre-transpose)", vol_hwb))

            vol_bhw = vol_hwb.transpose(2, 0, 1)

            if not np.isfinite(vol_bhw).all():
                vol_bhw = np.nan_to_num(vol_bhw, nan=0.0, posinf=1.0, neginf=0.0)
                if self.debug:
                    print(f"{name}: Found non-finite values → replaced with finite numbers")

            vol_bhw_ds, strides = self._downsample(vol_bhw)
            if self.debug:
                B, H, W = vol_bhw.shape
                Bd, Hd, Wd = vol_bhw_ds.shape
                print(
                    f"{name}: downsample strides (sB,sH,sW)={strides}, size {B}x{H}x{W} → {Bd}x{Hd}x{Wd} (vox={Bd * Hd * Wd:,})"
                )

            # Auto iso bounds (percentiles; optionally ignore zeros)
            iso_min, iso_max, dbg = self._auto_iso_bounds(vol_bhw_ds)
            if self.debug:
                print(f"{name}: iso_min={iso_min:.4f}, iso_max={iso_max:.4f}, zero_frac={dbg.get('zero_frac', float('nan')):.3f}")

            row = build_volume_row(
                name=name,
                vol=vol_bhw_ds,
                isomin=iso_min,
                isomax=iso_max,
                surface_count=max(1, self.surface_count),
                caps=self.caps,
                colorscale=self.colorscale,
            )
            for tr in row.traces:
                if isinstance(tr, go.Volume):
                    tr.opacity = self.opacity
            rows.append(row)

        fig = assemble_linked_figure(
            rows,
            coloraxis=dict(colorbar=dict(title="Normalized intensity [0,1]")),
            figure_height_per_row=self.figure_height_per_row,
            figure_width=self.figure_width,
            min_height=self.min_height,
        )
        fig.update_layout(scene_aspectmode="data", margin=dict(l=0, r=0, t=40, b=0))

        if self.save_html:
            fig.write_html(self.save_html, include_plotlyjs="cdn", full_html=True)
            if self.debug:
                print(f"Saved volume figure to: {self.save_html}")

        return fig
