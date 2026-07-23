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
    maximum_elevation: float,
    step: int | None = None,
    colormap_name: str = "seismic",
) -> tuple[np.ndarray, list[str]]:
    """Build contour levels and colors across the elevation range."""

    if maximum_elevation <= minimum_elevation:
        raise ValueError("`maximum_elevation` must be greater than `minimum_elevation`.")

    elevation_range = maximum_elevation - minimum_elevation
    if step is None:
        step = max(1, int(elevation_range / 10))
    step = max(1, min(step, max(1, int(elevation_range / 2))))

    levels = np.arange(minimum_elevation, maximum_elevation, step, dtype=float)
    levels = np.append(levels, float(maximum_elevation))
    if minimum_elevation < 0 < maximum_elevation:
        levels = np.unique(np.append(levels, 0.0))

    cmap = cm.get_cmap(colormap_name)
    color_positions = np.empty_like(levels)

    negative = levels < 0
    positive = levels > 0
    zero = ~(negative | positive)

    if np.any(negative):
        color_positions[negative] = 0.5 * (
            (levels[negative] - minimum_elevation) / (0.0 - minimum_elevation)
        )
    if np.any(positive):
        color_positions[positive] = 0.5 + 0.5 * (
            levels[positive] / maximum_elevation
        )
    color_positions[zero] = 0.5

    colors = [mpl_colors.rgb2hex(cmap(position)) for position in color_positions]
    if np.any(zero):
        colors[int(np.flatnonzero(zero)[0])] = "#ffffff"
    return levels, colors
