"""Core public bathymetry API."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import numpy as np
import utm
import xarray as xr
from pyproj import Transformer
from scipy.interpolate import griddata

from .logging_utils import default_logger
from .plotting import (
    plot_bathymetry,
    plot_bathymetry_3d,
    plot_merge_preview,
    plot_oblique_profile,
    plot_orthogonal_profile,
)
from .utils import compute_sampling_step, normalize_path, validate_coordinate_bounds, validate_loaded_dataset


class Bathymetry:
    """Bathymetric dataset manager with IO, interpolation, and visualization helpers.

    Parameters
    ----------
    utm_zone_number : int, optional
        UTM zone number used when scattered input data are provided in UTM coordinates.
    utm_zone_letter : str, optional
        UTM zone letter used when scattered input data are provided in UTM coordinates.
    source_crs : str, optional
        Coordinate reference system of the input scattered data. Ignored when UTM
        zone parameters are provided.
    name_logger : str, default="bathymetry"
        Logger name used to emit diagnostic messages.
    """

    def __init__(
        self,
        utm_zone_number: int | None = None,
        utm_zone_letter: str | None = None,
        source_crs: str | None = None,
        name_logger: str = "bathymetry",
    ) -> None:
        self.logger = default_logger(name_logger)
        self.utm_zone_number = utm_zone_number
        self.utm_zone_letter = utm_zone_letter
        self.source_crs = source_crs
        self.ds: xr.Dataset | None = None

    @classmethod
    def from_dataset(cls, dataset: xr.Dataset, **kwargs: Any) -> "Bathymetry":
        """Create an instance from an existing dataset."""

        instance = cls(**kwargs)
        instance.ds = dataset
        return instance

    def load_file(
        self,
        file_path: str | Path,
        size_mesh: int | None = None,
        z_neg: bool = True,
        value_nan: float | None = None,
        delimiter: str | None = None,
    ) -> None:
        """Load bathymetry from NetCDF or XYZ-like text files."""

        path = normalize_path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Input file does not exist: {path}")

        self.logger.info("Loading file %s", path)
        suffix = path.suffix.lower()

        if suffix == ".nc":
            dataset = xr.open_dataset(path, decode_cf=False)
        elif suffix in {".dat", ".xyz", ".txt"}:
            data = np.loadtxt(path, delimiter=delimiter)
            if data.ndim != 2 or data.shape[1] < 3:
                raise ValueError("Text bathymetry files must contain at least three columns: x, y, elevation.")

            x = np.asarray(data[:, 0], dtype=float)
            y = np.asarray(data[:, 1], dtype=float)
            elevation = np.asarray(data[:, 2], dtype=float)
            lon, lat = self._transform_input_coordinates(x, y)
            mesh_lon, mesh_lat, elevation_mesh = self.to_mesh(lon, lat, elevation, size_mesh=size_mesh)
            dataset = xr.Dataset({"elevation": (["lat", "lon"], elevation_mesh)}, coords={"lon": mesh_lon, "lat": mesh_lat})
        else:
            raise ValueError(f"Unsupported file extension: {path.suffix}")

        if "elevation" not in dataset:
            raise ValueError("The dataset must contain an `elevation` variable.")

        if z_neg:
            dataset["elevation"] = dataset["elevation"] * -1

        if value_nan is not None:
            dataset["elevation"] = dataset["elevation"].where(dataset["elevation"] != value_nan)

        self.ds = dataset
        self._log_dataset_summary("Loaded dataset")

    def load_url(self, url_path: str) -> None:
        """Load a bathymetry dataset directly from a remote URL."""

        if not url_path:
            raise ValueError("`url_path` must be a non-empty string.")

        self.logger.info("Loading remote dataset %s", url_path)
        self.ds = xr.open_dataset(url_path)
        self._log_dataset_summary("Loaded remote dataset")

    def crop(self, lon_min: float, lat_min: float, lon_max: float, lat_max: float) -> None:
        """Crop the loaded dataset to the nearest bounding box coordinates."""

        validate_loaded_dataset(self.ds)
        validate_coordinate_bounds(lon_min, lat_min, lon_max, lat_max)

        lon_min_nearest = self.ds.sel(lon=lon_min, method="nearest").lon.item()
        lon_max_nearest = self.ds.sel(lon=lon_max, method="nearest").lon.item()
        lat_min_nearest = self.ds.sel(lat=lat_min, method="nearest").lat.item()
        lat_max_nearest = self.ds.sel(lat=lat_max, method="nearest").lat.item()

        self.ds = self.ds.sel(
            lon=slice(min(lon_min_nearest, lon_max_nearest), max(lon_min_nearest, lon_max_nearest)),
            lat=slice(min(lat_min_nearest, lat_max_nearest), max(lat_min_nearest, lat_max_nearest)),
        )
        self._log_dataset_summary("Cropped dataset")

    def cut(self, lon_min: float, lat_min: float, lon_max: float, lat_max: float) -> None:
        """Backward-compatible alias for :meth:`crop`."""

        warnings.warn("`cut` is deprecated; use `crop` instead.", DeprecationWarning, stacklevel=2)
        self.crop(lon_min=lon_min, lat_min=lat_min, lon_max=lon_max, lat_max=lat_max)

    def to_mesh(
        self,
        x: np.ndarray,
        y: np.ndarray,
        elevation: np.ndarray,
        size_mesh: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Interpolate scattered bathymetry points to a regular grid."""

        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        elevation = np.asarray(elevation, dtype=float)

        if x.ndim != 1 or y.ndim != 1 or elevation.ndim != 1:
            raise ValueError("`x`, `y`, and `elevation` must be one-dimensional arrays.")
        if not (len(x) == len(y) == len(elevation)):
            raise ValueError("`x`, `y`, and `elevation` must have the same length.")
        if size_mesh is not None and size_mesh < 2:
            raise ValueError("`size_mesh` must be at least 2.")

        if size_mesh is not None:
            self.logger.info("Interpolating scattered bathymetry to a %sx%s grid.", size_mesh, size_mesh)
            grid_lon = np.linspace(float(x.min()), float(x.max()), size_mesh)
            grid_lat = np.linspace(float(y.min()), float(y.max()), size_mesh)
            lon_mesh, lat_mesh = np.meshgrid(grid_lon, grid_lat)
        else:
            unique_lon = np.unique(x)
            unique_lat = np.unique(y)
            if unique_lon.size * unique_lat.size != x.size:
                raise ValueError("A regular grid could not be inferred from the scattered coordinates. Set `size_mesh`.")
            grid_lon = unique_lon
            grid_lat = unique_lat
            lon_mesh, lat_mesh = np.meshgrid(grid_lon, grid_lat)

        sampling_step = compute_sampling_step(len(x))
        self.logger.info("Interpolation sampling step: %s", sampling_step)
        elevation_mesh = griddata(
            (x[::sampling_step], y[::sampling_step]),
            elevation[::sampling_step],
            (lon_mesh, lat_mesh),
        )

        self.logger.info(
            "Interpolated grid shape: %s, longitude range: [%s, %s], latitude range: [%s, %s].",
            elevation_mesh.shape,
            float(np.nanmin(grid_lon)),
            float(np.nanmax(grid_lon)),
            float(np.nanmin(grid_lat)),
            float(np.nanmax(grid_lat)),
        )
        return grid_lon, grid_lat, elevation_mesh

    def save_nc(self, file_path: str | Path) -> None:
        """Save the current dataset to NetCDF."""

        validate_loaded_dataset(self.ds)
        path = Path(file_path)
        elevation = self.ds["elevation"]
        missing_value = elevation.attrs.get("missing_value")
        fill_value = elevation.attrs.get("_FillValue")

        if missing_value is None and fill_value is not None:
            elevation.attrs["missing_value"] = fill_value
        elif fill_value is None and missing_value is not None:
            elevation.attrs["_FillValue"] = missing_value
        elif missing_value is not None and fill_value is not None and missing_value != fill_value:
            replacement = fill_value if np.isnan(missing_value) else missing_value
            elevation.attrs["missing_value"] = replacement
            elevation.attrs["_FillValue"] = replacement
            self.ds["elevation"] = elevation.where(elevation != replacement)

        self.ds.to_netcdf(path)
        self.logger.info("Saved dataset to %s", path)

    def save_dat(self, file_path: str | Path, in_utm: bool = False) -> None:
        """Save the current dataset as a three-column text file."""

        validate_loaded_dataset(self.ds)
        lat = self.ds.lat.values
        lon = self.ds.lon.values
        lon_mesh, lat_mesh = np.meshgrid(lon, lat)
        elevation = self.ds.elevation.values

        if in_utm:
            x, y, _, _ = utm.from_latlon(lat_mesh, lon_mesh)
            output = np.column_stack((x.ravel(), y.ravel(), elevation.ravel()))
        else:
            output = np.column_stack((lon_mesh.ravel(), lat_mesh.ravel(), elevation.ravel()))

        np.savetxt(file_path, output, fmt="%.10f")
        self.logger.info("Saved XYZ bathymetry to %s", file_path)

    def merge(self, detail: "Bathymetry") -> "Bathymetry":
        """Merge a detail bathymetry onto the current dataset."""

        validate_loaded_dataset(self.ds)
        validate_loaded_dataset(detail.ds)

        self.logger.info("Merging base and detail bathymetry datasets.")
        interpolated_detail = detail.ds.interp(lon=self.ds.lon, lat=self.ds.lat, method="nearest")
        merged_elevation = xr.where(interpolated_detail.elevation.notnull(), interpolated_detail.elevation, self.ds.elevation)
        merged_dataset = self.ds.copy()
        merged_dataset["elevation"] = merged_elevation
        result = Bathymetry.from_dataset(
            merged_dataset,
            utm_zone_number=self.utm_zone_number,
            utm_zone_letter=self.utm_zone_letter,
            source_crs=self.source_crs,
            name_logger=self.logger.name,
        )
        result._log_dataset_summary("Merged dataset")
        return result

    def fusionate(self, b_detail: "Bathymetry") -> "Bathymetry":
        """Backward-compatible alias for :meth:`merge`."""

        warnings.warn("`fusionate` is deprecated; use `merge` instead.", DeprecationWarning, stacklevel=2)
        return self.merge(b_detail)

    def plot(
        self,
        cmap: str = "seismic",
        x_lim: tuple[float, float] | None = None,
        y_lim: tuple[float, float] | None = None,
        zmin: float | None = None,
        step_beriles: int | None = None,
        aux_title: str = "",
        _ax: Any = None,
    ) -> Any:
        """Plot the loaded bathymetry as filled contours."""

        validate_loaded_dataset(self.ds)
        return plot_bathymetry(
            self.ds.lon.values,
            self.ds.lat.values,
            np.squeeze(self.ds.elevation.values),
            cmap=cmap,
            x_lim=x_lim,
            y_lim=y_lim,
            zmin=zmin,
            step_beriles=step_beriles,
            title_suffix=aux_title,
            axis=_ax,
        )

    def plot_3d(self, _ax: Any = None) -> Any:
        """Plot the loaded bathymetry as a 3D surface."""

        validate_loaded_dataset(self.ds)
        return plot_bathymetry_3d(
            self.ds.lon.values,
            self.ds.lat.values,
            np.squeeze(self.ds.elevation.values),
            axis=_ax,
        )

    def plot_orthogonal_profile(self, coord_lon: float, coord_lat: float, lbl_z: str = "") -> None:
        """Plot orthogonal profiles through a longitude/latitude location."""

        validate_loaded_dataset(self.ds)
        plot_orthogonal_profile(
            self.ds.lon.values,
            self.ds.lat.values,
            np.squeeze(self.ds.elevation.values),
            coord_lon=coord_lon,
            coord_lat=coord_lat,
            label=lbl_z,
        )

    def plot_perfil_ortogonal(self, coord_lon: float, coord_lat: float, lbl_z: str = "") -> None:
        """Backward-compatible alias for :meth:`plot_orthogonal_profile`."""

        warnings.warn(
            "`plot_perfil_ortogonal` is deprecated; use `plot_orthogonal_profile` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.plot_orthogonal_profile(coord_lon=coord_lon, coord_lat=coord_lat, lbl_z=lbl_z)

    def plot_oblique_profile(
        self,
        coord1_lon: float,
        coord1_lat: float,
        coord2_lon: float,
        coord2_lat: float,
        lbl_z: str = "",
    ) -> None:
        """Plot an oblique bathymetry profile between two coordinates."""

        validate_loaded_dataset(self.ds)
        plot_oblique_profile(
            self.ds.lon.values,
            self.ds.lat.values,
            np.squeeze(self.ds.elevation.values),
            start_lon=coord1_lon,
            start_lat=coord1_lat,
            end_lon=coord2_lon,
            end_lat=coord2_lat,
            label=lbl_z,
        )

    def plot_perfil_oblicuo(
        self,
        coord1_lon: float,
        coord1_lat: float,
        coord2_lon: float,
        coord2_lat: float,
        lbl_z: str = "",
    ) -> None:
        """Backward-compatible alias for :meth:`plot_oblique_profile`."""

        warnings.warn(
            "`plot_perfil_oblicuo` is deprecated; use `plot_oblique_profile` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.plot_oblique_profile(coord1_lon, coord1_lat, coord2_lon, coord2_lat, lbl_z=lbl_z)

    def plot_merge_preview(self, detail: "Bathymetry") -> None:
        """Plot the detail bathymetry footprint over the base bathymetry."""

        validate_loaded_dataset(self.ds)
        validate_loaded_dataset(detail.ds)
        plot_merge_preview(
            self.ds.lon.values,
            self.ds.lat.values,
            np.squeeze(self.ds.elevation.values),
            detail.ds.lon.values,
            detail.ds.lat.values,
            np.squeeze(detail.ds.elevation.values),
        )

    def plot_check_fusionate(self, b_detail: "Bathymetry") -> None:
        """Backward-compatible alias for :meth:`plot_merge_preview`."""

        warnings.warn(
            "`plot_check_fusionate` is deprecated; use `plot_merge_preview` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.plot_merge_preview(b_detail)

    def _transform_input_coordinates(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Transform input coordinates to longitude/latitude if necessary."""

        if self.utm_zone_number is not None and self.utm_zone_letter is not None:
            lat, lon = utm.to_latlon(x, y, self.utm_zone_number, self.utm_zone_letter)
            return np.asarray(lon, dtype=float), np.asarray(lat, dtype=float)

        if self.source_crs is not None:
            transformer = Transformer.from_crs(self.source_crs, "EPSG:4326", always_xy=True)
            lon, lat = transformer.transform(x, y)
            return np.asarray(lon, dtype=float), np.asarray(lat, dtype=float)

        return np.asarray(x, dtype=float), np.asarray(y, dtype=float)

    def _log_dataset_summary(self, prefix: str) -> None:
        """Log concise dataset summary information."""

        validate_loaded_dataset(self.ds)
        self.logger.info(
            "%s. Shape: %s, latitude range: [%s, %s], longitude range: [%s, %s], elevation range: [%s, %s].",
            prefix,
            tuple(self.ds.elevation.shape),
            float(self.ds.lat.min()),
            float(self.ds.lat.max()),
            float(self.ds.lon.min()),
            float(self.ds.lon.max()),
            float(np.nanmin(self.ds.elevation.values)),
            float(np.nanmax(self.ds.elevation.values)),
        )
