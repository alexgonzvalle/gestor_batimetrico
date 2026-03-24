"""Internal numerical and validation helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from matplotlib import cm
from matplotlib import colors as mpl_colors


def validate_loaded_dataset(dataset: object) -> None:
    """Ensure a dataset has been loaded before operating on it."""

    if dataset is None:
        raise ValueError("No dataset is loaded. Load or create a dataset before calling this method.")


def validate_coordinate_bounds(lon_min: float, lat_min: float, lon_max: float, lat_max: float) -> None:
    """Validate bounding box coordinates."""

    if lon_min > lon_max:
        raise ValueError("`lon_min` must be less than or equal to `lon_max`.")
    if lat_min > lat_max:
        raise ValueError("`lat_min` must be less than or equal to `lat_max`.")


def normalize_path(file_path: str | Path) -> Path:
    """Normalize a filesystem path."""

    return Path(file_path).expanduser().resolve()


def compute_sampling_step(size: int, threshold: int = 300_000) -> int:
    """Compute a subsampling step to limit interpolation cost."""

    if size <= 0:
        raise ValueError("`size` must be a positive integer.")

    step = 1
    while size / step > threshold:
        step += 1
    return step


def build_symmetric_levels_and_colors(
    minimum_elevation: float,
    step: int | None = None,
    colormap_name: str = "seismic",
) -> tuple[np.ndarray, list[str]]:
    """Build symmetric contour levels and colors around sea level."""

    if minimum_elevation >= 0:
        levels = np.array([0.0, 1.0])
    else:
        if step is None:
            step = max(1, int(abs(minimum_elevation) / 10))
        step = max(1, min(step, max(1, int(abs(minimum_elevation) / 2))))
        negative_levels = np.arange(minimum_elevation, 0.0, step, dtype=float)
        levels = np.unique(np.concatenate((negative_levels, np.array([0.0]), -negative_levels[::-1])))

    cmap = cm.get_cmap(colormap_name, len(levels))
    colors = [mpl_colors.rgb2hex(cmap(index)) for index in range(cmap.N)]
    return levels, colors
