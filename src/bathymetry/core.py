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
from scipy.spatial import Delaunay, QhullError

from .logging_utils import default_logger
from .plotting import (
    plot_bathymetry,
    plot_bathymetry_3d,
    plot_merge_preview,
    plot_oblique_profile,
    plot_orthogonal_profile,
    plot_point_bathymetry,
)
from .utils import normalize_path, validate_coordinate_bounds, validate_loaded_dataset


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
        z_neg: bool = True,
        z_ref: float | None = None,
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
            dataset = xr.Dataset(
                {"elevation": ("point", elevation)},
                coords={"lon": ("point", lon), "lat": ("point", lat)},
            )
        else:
            raise ValueError(f"Unsupported file extension: {path.suffix}")

        if "elevation" not in dataset:
            raise ValueError("The dataset must contain an `elevation` variable.")

        if z_ref:
            dataset["elevation"] = dataset["elevation"] - z_ref
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
        """Crop the loaded dataset to a longitude/latitude bounding box."""

        validate_loaded_dataset(self.ds)
        validate_coordinate_bounds(lon_min, lat_min, lon_max, lat_max)

        if self.ds.elevation.dims == ("point",):
            inside = (
                (self.ds.lon >= lon_min)
                & (self.ds.lon <= lon_max)
                & (self.ds.lat >= lat_min)
                & (self.ds.lat <= lat_max)
            )
            self.ds = self.ds.isel(point=inside)
            self._log_dataset_summary("Cropped dataset")
            return

        self._require_grid("crop")
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

    def _interpolate_to_grid(
        self,
        x: np.ndarray,
        y: np.ndarray,
        elevation: np.ndarray,
        size_mesh: int | tuple[int, int] | None = None,
        method: str = "linear",
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Interpolate scattered bathymetry points to a regular grid.

        ``size_mesh`` may be a single size for a square grid or a
        ``(longitude, latitude)`` size tuple. When omitted, the coordinates
        must already describe a complete regular grid.
        """

        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        elevation = np.asarray(elevation, dtype=float)

        if x.ndim != 1 or y.ndim != 1 or elevation.ndim != 1:
            raise ValueError("`x`, `y`, and `elevation` must be one-dimensional arrays.")
        if not (len(x) == len(y) == len(elevation)):
            raise ValueError("`x`, `y`, and `elevation` must have the same length.")
        if size_mesh is None:
            mesh_size = None
        elif isinstance(size_mesh, int):
            mesh_size = (size_mesh, size_mesh)
        elif (
            isinstance(size_mesh, tuple)
            and len(size_mesh) == 2
            and all(isinstance(size, int) for size in size_mesh)
        ):
            mesh_size = size_mesh
        else:
            raise TypeError("`size_mesh` must be an integer, a (longitude, latitude) tuple, or None.")

        if mesh_size is not None and any(size < 2 for size in mesh_size):
            raise ValueError("Each `size_mesh` dimension must be at least 2.")

        if mesh_size is not None:
            lon_size, lat_size = mesh_size
            self.logger.info("Interpolating scattered bathymetry to a %sx%s grid.", lon_size, lat_size)
            grid_lon = np.linspace(float(x.min()), float(x.max()), lon_size)
            grid_lat = np.linspace(float(y.min()), float(y.max()), lat_size)
            lon_mesh, lat_mesh = np.meshgrid(grid_lon, grid_lat)
        else:
            unique_lon = np.unique(x)
            unique_lat = np.unique(y)
            if unique_lon.size * unique_lat.size != x.size:
                raise ValueError("A regular grid could not be inferred from the scattered coordinates. Set `size_mesh`.")
            grid_lon = unique_lon
            grid_lat = unique_lat
            lon_mesh, lat_mesh = np.meshgrid(grid_lon, grid_lat)

        valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(elevation)
        if np.count_nonzero(valid) < 3:
            raise ValueError("At least three finite bathymetry points are required for interpolation.")

        self.logger.info("Interpolating with all %s finite input points.", int(np.count_nonzero(valid)))
        elevation_mesh = griddata(
            (x[valid], y[valid]),
            elevation[valid],
            (lon_mesh, lat_mesh),
            method=method,
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

    def to_grid(
        self,
        size_mesh: int | tuple[int, int] | None = None,
        method: str = "linear",
    ) -> None:
        """Convert a point bathymetry to a regular longitude/latitude grid.

        The conversion is explicit because interpolation changes the data.
        ``size_mesh`` may be an integer or a ``(longitude, latitude)`` tuple.
        When omitted, the points must already form a complete regular grid.
        """

        validate_loaded_dataset(self.ds)
        if self.ds.elevation.dims == ("lat", "lon"):
            raise ValueError("The loaded bathymetry is already a regular grid.")
        if self.ds.elevation.dims != ("point",):
            raise ValueError(
                "A grid can only be created from an `elevation` variable with a single `point` dimension."
            )

        elevation_attrs = self.ds.elevation.attrs.copy()
        dataset_attrs = self.ds.attrs.copy()
        grid_lon, grid_lat, elevation_grid = self._interpolate_to_grid(
            self.ds.lon.values,
            self.ds.lat.values,
            self.ds.elevation.values,
            size_mesh=size_mesh,
            method=method,
        )
        self.ds = xr.Dataset(
            {"elevation": (("lat", "lon"), elevation_grid, elevation_attrs)},
            coords={"lon": grid_lon, "lat": grid_lat},
            attrs=dataset_attrs,
        )
        self._log_dataset_summary("Converted point dataset to grid")

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

    def save_dat(self, file_path: str | Path, in_utm: bool = False, z_neg: bool = True) -> None:
        """Save the current dataset as a three-column text file."""

        validate_loaded_dataset(self.ds)
        lat = self.ds.lat.values
        lon = self.ds.lon.values
        elevation = self.ds.elevation.values
        if self.ds.elevation.dims == ("point",):
            output_lon = lon
            output_lat = lat
        else:
            self._require_grid("save_dat")
            output_lon, output_lat = np.meshgrid(lon, lat)

        if z_neg:
            elevation = elevation * -1

        if in_utm:
            x, y, _, _ = utm.from_latlon(output_lat, output_lon)
            output = np.column_stack((x.ravel(), y.ravel(), elevation.ravel()))
        else:
            output = np.column_stack((output_lon.ravel(), output_lat.ravel(), elevation.ravel()))

        np.savetxt(file_path, output, fmt="%.10f")
        self.logger.info("Saved XYZ bathymetry to %s", file_path)

    def merge(self, detail: "Bathymetry", method: str = "nearest") -> "Bathymetry":
        """Merge valid detail elevations onto the current dataset.

        Parameters
        ----------
        detail : Bathymetry
            Higher-detail dataset to overlay on the current bathymetry.
        method : str, default="nearest"
            Interpolation method passed to :meth:`xarray.Dataset.interp`.
        """

        if not isinstance(detail, Bathymetry):
            raise TypeError("`detail` must be a Bathymetry instance.")

        validate_loaded_dataset(self.ds)
        validate_loaded_dataset(detail.ds)

        required = {"lon", "lat", "elevation"}
        for dataset_name, dataset in (("base", self.ds), ("detail", detail.ds)):
            missing = required.difference(dataset.variables)
            if missing:
                raise ValueError(f"The {dataset_name} dataset is missing required variables: {sorted(missing)}.")
            if dataset.elevation.dims not in {("lat", "lon"), ("point",)}:
                raise ValueError(
                    f"The {dataset_name} `elevation` variable must have dimensions ('lat', 'lon') or ('point',); "
                    f"got {dataset.elevation.dims}."
                )
            for coordinate_name in ("lon", "lat"):
                coordinate = dataset[coordinate_name]
                expected_dimension = coordinate_name if dataset.elevation.dims == ("lat", "lon") else "point"
                if coordinate.ndim != 1 or coordinate.dims != (expected_dimension,):
                    raise ValueError(
                        f"The {dataset_name} `{coordinate_name}` coordinate must be one-dimensional."
                    )
                if not np.issubdtype(coordinate.dtype, np.number):
                    raise ValueError(f"The {dataset_name} `{coordinate_name}` coordinate must be numeric.")
                values = coordinate.values
                if not np.all(np.isfinite(values)):
                    raise ValueError(f"The {dataset_name} `{coordinate_name}` coordinate must contain finite values.")
                if dataset.elevation.dims == ("lat", "lon") and np.unique(values).size != values.size:
                    raise ValueError(f"The {dataset_name} `{coordinate_name}` coordinate must contain unique values.")

        base_crs = self.ds.attrs.get("crs")
        detail_crs = detail.ds.attrs.get("crs")
        if base_crs is not None and detail_crs is not None and base_crs != detail_crs:
            raise ValueError(f"Cannot merge datasets with different CRS values: {base_crs!r} and {detail_crs!r}.")

        overlaps = (
            float(self.ds.lon.min()) <= float(detail.ds.lon.max())
            and float(detail.ds.lon.min()) <= float(self.ds.lon.max())
            and float(self.ds.lat.min()) <= float(detail.ds.lat.max())
            and float(detail.ds.lat.min()) <= float(self.ds.lat.max())
        )
        if not overlaps:
            raise ValueError("The base and detail datasets do not overlap.")

        if detail.ds.elevation.dims == ("point",):
            return self._merge_point_detail(detail)

        self._require_grid("merge")
        self.logger.info("Merging base and detail bathymetry datasets using %s interpolation.", method)
        interpolated_detail = detail.ds.elevation.interp(
            lon=self.ds.lon,
            lat=self.ds.lat,
            method=method,
        )
        merged_elevation = interpolated_detail.combine_first(self.ds.elevation)
        merged_elevation.attrs = self.ds.elevation.attrs.copy()
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

    def _merge_point_detail(self, detail: "Bathymetry") -> "Bathymetry":
        """Replace base samples inside the detail footprint with original detail points."""

        base_lon, base_lat, base_elevation = self._dataset_as_points(self.ds)
        detail_lon, detail_lat, detail_elevation = self._dataset_as_points(detail.ds)
        detail_valid = np.isfinite(detail_lon) & np.isfinite(detail_lat) & np.isfinite(detail_elevation)
        detail_lon = detail_lon[detail_valid]
        detail_lat = detail_lat[detail_valid]
        detail_elevation = detail_elevation[detail_valid]
        if detail_elevation.size < 3:
            raise ValueError("At least three finite detail points are required to determine its footprint.")

        detail_coordinates = np.column_stack((detail_lon, detail_lat))
        unique_coordinates, unique_indices = np.unique(detail_coordinates, axis=0, return_index=True)
        detail_elevation = detail_elevation[unique_indices]
        try:
            footprint = Delaunay(unique_coordinates)
        except QhullError as error:
            raise ValueError("The detail points must define a two-dimensional spatial footprint.") from error

        base_coordinates = np.column_stack((base_lon, base_lat))
        base_valid = np.isfinite(base_lon) & np.isfinite(base_lat) & np.isfinite(base_elevation)
        inside_detail = np.zeros(base_elevation.size, dtype=bool)
        inside_detail[base_valid] = footprint.find_simplex(base_coordinates[base_valid]) >= 0
        keep_base = base_valid & ~inside_detail

        merged_dataset = xr.Dataset(
            {
                "elevation": (
                    "point",
                    np.concatenate((base_elevation[keep_base], detail_elevation)),
                    self.ds.elevation.attrs.copy(),
                )
            },
            coords={
                "lon": ("point", np.concatenate((base_lon[keep_base], unique_coordinates[:, 0]))),
                "lat": ("point", np.concatenate((base_lat[keep_base], unique_coordinates[:, 1]))),
            },
            attrs=self.ds.attrs.copy(),
        )
        result = Bathymetry.from_dataset(
            merged_dataset,
            utm_zone_number=self.utm_zone_number,
            utm_zone_letter=self.utm_zone_letter,
            source_crs=self.source_crs,
            name_logger=self.logger.name,
        )
        result._log_dataset_summary("Merged multiresolution point dataset")
        return result

    def plot(
        self,
        cmap: str = "seismic",
        x_lim: tuple[float, float] | None = None,
        y_lim: tuple[float, float] | None = None,
        zmin: float | None = None,
        zmax: float | None = None,
        step_beriles: int | None = None,
        aux_title: str = "",
        _ax: Any = None,
    ) -> Any:
        """Plot the loaded bathymetry as filled contours."""

        validate_loaded_dataset(self.ds)
        if self.ds.elevation.dims == ("point",):
            return plot_point_bathymetry(
                self.ds.lon.values,
                self.ds.lat.values,
                self.ds.elevation.values,
                cmap=cmap,
                x_lim=x_lim,
                y_lim=y_lim,
                zmin=zmin,
                zmax=zmax,
                step_beriles=step_beriles,
                title_suffix=aux_title,
                axis=_ax,
            )
        self._require_grid("plot")
        return plot_bathymetry(
            self.ds.lon.values,
            self.ds.lat.values,
            np.squeeze(self.ds.elevation.values),
            cmap=cmap,
            x_lim=x_lim,
            y_lim=y_lim,
            zmin=zmin,
            zmax=zmax,
            step_beriles=step_beriles,
            title_suffix=aux_title,
            axis=_ax,
        )

    def plot_3d(self, _ax: Any = None) -> Any:
        """Plot the loaded bathymetry as a 3D surface."""

        validate_loaded_dataset(self.ds)
        self._require_grid("plot_3d")
        return plot_bathymetry_3d(
            self.ds.lon.values,
            self.ds.lat.values,
            np.squeeze(self.ds.elevation.values),
            axis=_ax,
        )

    def plot_orthogonal_profile(self, coord_lon: float, coord_lat: float, lbl_z: str = "") -> None:
        """Plot orthogonal profiles through a longitude/latitude location."""

        validate_loaded_dataset(self.ds)
        self._require_grid("plot_orthogonal_profile")
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
        self._require_grid("plot_oblique_profile")
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
        self._require_grid("plot_merge_preview")
        detail._require_grid("plot_merge_preview")
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

    def _require_grid(self, operation: str) -> None:
        """Raise a clear error when an operation requires gridded data."""

        if self.ds.elevation.dims != ("lat", "lon"):
            raise ValueError(f"`{operation}` requires a regular grid; call `to_grid()` first.")

    @staticmethod
    def _dataset_as_points(dataset: xr.Dataset) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return flattened longitude, latitude, and elevation arrays."""

        if dataset.elevation.dims == ("point",):
            return dataset.lon.values, dataset.lat.values, dataset.elevation.values
        lon_mesh, lat_mesh = np.meshgrid(dataset.lon.values, dataset.lat.values)
        return lon_mesh.ravel(), lat_mesh.ravel(), dataset.elevation.values.ravel()

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
