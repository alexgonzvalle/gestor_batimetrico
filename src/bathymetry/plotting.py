"""Internal plotting helpers for bathymetric datasets."""

from __future__ import annotations

from typing import Any

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import patches
from matplotlib import ticker
from mpl_toolkits.axes_grid1 import make_axes_locatable

from .utils import build_symmetric_levels_and_colors


def format_meter_label(value: float) -> str:
    """Format contour labels in meters."""

    text = f"{value:.1f}"
    if text.endswith("0"):
        text = f"{value:.0f}"
    return rf"{text} m" if plt.rcParams["text.usetex"] else f"{text} m"


def plot_bathymetry(
    lon: np.ndarray,
    lat: np.ndarray,
    elevation: np.ndarray,
    *,
    cmap: str = "seismic",
    x_lim: tuple[float, float] | None = None,
    y_lim: tuple[float, float] | None = None,
    zmin: float | None = None,
    step_beriles: int | None = None,
    title_suffix: str = "",
    axis: Any = None,
) -> Any:
    """Render a filled contour bathymetry plot."""

    lon_mesh, lat_mesh = np.meshgrid(lon, lat)
    axis_was_created = axis is None
    if axis_was_created:
        _, axis = plt.subplots()

    axis.set_title(f"Bathymetry {title_suffix}".strip())
    axis.set_xlabel("Longitude (deg)")
    axis.set_ylabel("Latitude (deg)")
    axis.set_aspect("equal")

    zmin = float(np.nanmin(elevation)) if zmin is None else zmin
    levels, colors = build_symmetric_levels_and_colors(zmin, step_beriles, cmap)

    filled = axis.contourf(
        lon_mesh,
        lat_mesh,
        elevation,
        vmin=float(np.min(levels)),
        vmax=float(np.max(levels)),
        levels=levels,
        colors=colors,
        extend="both",
    )
    lines = axis.contour(
        lon_mesh,
        lat_mesh,
        elevation,
        vmin=float(np.min(levels)),
        vmax=float(np.max(levels)),
        levels=levels,
        colors=("k",),
    )
    axis.clabel(lines, lines.levels, fmt=format_meter_label, fontsize=10, colors="w")

    colorbar = axis.figure.colorbar(filled)
    colorbar.set_label("(m)", labelpad=-0.1)

    if x_lim is not None:
        axis.set_xlim(x_lim)
    if y_lim is not None:
        axis.set_ylim(y_lim)

    if axis_was_created:
        plt.show()
    return axis


def plot_bathymetry_3d(lon: np.ndarray, lat: np.ndarray, elevation: np.ndarray, *, axis: Any = None) -> Any:
    """Render a 3D bathymetry surface plot."""

    lon_mesh, lat_mesh = np.meshgrid(lon, lat)
    axis_was_created = axis is None
    if axis_was_created:
        figure = plt.figure()
        axis = figure.add_subplot(111, projection="3d")

    axis.view_init(50, 135)
    axis.plot_surface(lon_mesh, lat_mesh, elevation, cmap="Blues_r")
    axis.set_xlabel("Longitude (deg)")
    axis.set_ylabel("Latitude (deg)")
    axis.set_zlabel("Elevation (m)")

    if axis_was_created:
        plt.show()
    return axis


def plot_merge_preview(
    lon: np.ndarray,
    lat: np.ndarray,
    elevation: np.ndarray,
    detail_lon: np.ndarray,
    detail_lat: np.ndarray,
    detail_elevation: np.ndarray,
) -> None:
    """Plot a detail dataset footprint on top of a base bathymetry."""

    lon_mesh, lat_mesh = np.meshgrid(lon, lat)
    detail_lon_mesh, detail_lat_mesh = np.meshgrid(detail_lon, detail_lat)

    figure, axis = plt.subplots(1, 1)
    level_min = float(np.nanmin(elevation))
    level_max = float(np.nanmax(elevation))
    levels = np.linspace(level_min, level_max, 64)

    axis.set_title("Bathymetry")
    filled = axis.contourf(lon_mesh, lat_mesh, elevation, levels=levels, cmap="Blues_r")
    axis.set_xlabel("Longitude")
    axis.set_ylabel("Latitude")
    axis.contourf(detail_lon_mesh, detail_lat_mesh, detail_elevation, levels=levels, cmap="Blues_r")

    axis.add_patch(
        patches.Rectangle(
            (float(detail_lon_mesh.min()), float(detail_lat_mesh.min())),
            float(detail_lon_mesh.max() - detail_lon_mesh.min()),
            float(detail_lat_mesh.max() - detail_lat_mesh.min()),
            linewidth=1,
            edgecolor="r",
            facecolor="none",
        )
    )

    divider = make_axes_locatable(axis)
    color_axis = divider.append_axes("right", size="5%", pad=0.05)
    figure.colorbar(filled, ticks=np.linspace(level_min, level_max, 11), spacing="uniform", cax=color_axis)
    plt.show()


def plot_orthogonal_profile(
    lon: np.ndarray,
    lat: np.ndarray,
    elevation: np.ndarray,
    *,
    coord_lon: float,
    coord_lat: float,
    label: str = "",
) -> None:
    """Plot orthogonal profiles through a target coordinate."""

    lon_index = int(np.abs(lon - coord_lon).argmin())
    lat_index = int(np.abs(lat - coord_lat).argmin())
    clean_elevation = np.nan_to_num(elevation, nan=0.0)

    _, (axis_lon, axis_lat) = plt.subplots(2)
    axis_lon.plot(-clean_elevation[:, lon_index], label=f"{label} lat={coord_lat:.2f}".strip())
    lon_ticks = np.linspace(float(lon.min()), float(lon.max()), len(axis_lon.xaxis.get_ticklabels()))
    axis_lon.xaxis.set_major_locator(ticker.FixedLocator(lon_ticks))
    axis_lon.set_xticklabels([f"{value:.2f}" for value in lon_ticks])
    axis_lon.legend()
    axis_lon.set_xlabel("Longitude")
    axis_lon.set_ylabel("Depth")
    axis_lon.grid(True)

    axis_lat.plot(-clean_elevation[lat_index, :], label=f"{label} lon={coord_lon:.2f}".strip())
    lat_ticks = np.linspace(float(lat.min()), float(lat.max()), len(axis_lat.xaxis.get_ticklabels()))
    axis_lat.xaxis.set_major_locator(ticker.FixedLocator(lat_ticks))
    axis_lat.set_xticklabels([f"{value:.2f}" for value in lat_ticks])
    axis_lat.legend()
    axis_lat.set_xlabel("Latitude")
    axis_lat.set_ylabel("Depth")
    axis_lat.grid(True)
    plt.show()


def plot_oblique_profile(
    lon: np.ndarray,
    lat: np.ndarray,
    elevation: np.ndarray,
    *,
    start_lon: float,
    start_lat: float,
    end_lon: float,
    end_lat: float,
    label: str = "",
) -> None:
    """Plot an oblique profile sampled along a straight line."""

    lon_mesh, lat_mesh = np.meshgrid(lon, lat)
    segment_count = max(2, int(max(np.abs(end_lon - start_lon), np.abs(end_lat - start_lat)) * 100))
    line_lon = np.linspace(start_lon, end_lon, segment_count)
    line_lat = np.linspace(start_lat, end_lat, segment_count)

    sampled = []
    clean_elevation = np.nan_to_num(elevation, nan=0.0)
    for lon_point, lat_point in zip(line_lon, line_lat, strict=False):
        distance = np.sqrt((lon_mesh - lon_point) ** 2 + (lat_mesh - lat_point) ** 2)
        lat_index, lon_index = np.unravel_index(int(np.argmin(distance)), distance.shape)
        sampled.append(clean_elevation[lat_index, lon_index])

    _, axis = plt.subplots()
    axis.plot(
        -np.asarray(sampled),
        label=f"{label} ({start_lon:.2f}, {start_lat:.2f}) to ({end_lon:.2f}, {end_lat:.2f})".strip(),
    )
    axis.legend()
    axis.set_xlabel("Sample index")
    axis.set_ylabel("Depth")
    axis.grid(True)
    plt.show()
