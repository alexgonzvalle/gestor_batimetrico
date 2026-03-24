from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from bathymetry import Bathymetry


@pytest.fixture()
def sample_dataset() -> xr.Dataset:
    lon = np.array([-3.0, -2.5, -2.0])
    lat = np.array([43.0, 43.5, 44.0])
    elevation = np.array(
        [
            [-10.0, -11.0, -12.0],
            [-13.0, -14.0, -15.0],
            [-16.0, -17.0, -18.0],
        ]
    )
    return xr.Dataset({"elevation": (["lat", "lon"], elevation)}, coords={"lon": lon, "lat": lat})


def test_crop_selects_nearest_window(sample_dataset: xr.Dataset) -> None:
    bathymetry = Bathymetry.from_dataset(sample_dataset.copy())
    bathymetry.crop(lon_min=-2.8, lat_min=43.1, lon_max=-2.1, lat_max=43.9)

    np.testing.assert_allclose(bathymetry.ds.lon.values, np.array([-3.0, -2.5, -2.0]))
    np.testing.assert_allclose(bathymetry.ds.lat.values, np.array([43.0, 43.5, 44.0]))


def test_merge_overwrites_only_detail_values(sample_dataset: xr.Dataset) -> None:
    detail_dataset = xr.Dataset(
        {"elevation": (["lat", "lon"], np.array([[np.nan, -99.0, np.nan], [np.nan, -88.0, np.nan], [np.nan, np.nan, np.nan]]))},
        coords={"lon": sample_dataset.lon.values, "lat": sample_dataset.lat.values},
    )

    base = Bathymetry.from_dataset(sample_dataset.copy())
    detail = Bathymetry.from_dataset(detail_dataset)
    merged = base.merge(detail)

    expected = sample_dataset.elevation.values.copy()
    expected[0, 1] = -99.0
    expected[1, 1] = -88.0
    np.testing.assert_allclose(merged.ds.elevation.values, expected, equal_nan=True)


def test_to_mesh_interpolates_scattered_points() -> None:
    bathymetry = Bathymetry()
    x = np.array([0.0, 1.0, 0.0, 1.0])
    y = np.array([0.0, 0.0, 1.0, 1.0])
    elevation = np.array([0.0, 1.0, 1.0, 2.0])

    lon, lat, elevation_mesh = bathymetry.to_mesh(x, y, elevation, size_mesh=2)

    np.testing.assert_allclose(lon, np.array([0.0, 1.0]))
    np.testing.assert_allclose(lat, np.array([0.0, 1.0]))
    np.testing.assert_allclose(elevation_mesh, np.array([[0.0, 1.0], [1.0, 2.0]]), atol=1e-12)


def test_load_file_replaces_configured_nan_value(tmp_path) -> None:
    file_path = tmp_path / "sample.xyz"
    np.savetxt(
        file_path,
        np.array(
            [
                [0.0, 0.0, 9999.0],
                [1.0, 0.0, 2.0],
                [0.0, 1.0, 3.0],
                [1.0, 1.0, 4.0],
            ]
        ),
    )

    bathymetry = Bathymetry()
    bathymetry.load_file(file_path, size_mesh=2, z_neg=False, value_nan=9999.0)

    assert np.isnan(bathymetry.ds.elevation.values[0, 0])


def test_save_dat_exports_three_columns(tmp_path, sample_dataset: xr.Dataset) -> None:
    bathymetry = Bathymetry.from_dataset(sample_dataset)
    output_path = tmp_path / "bathymetry.dat"

    bathymetry.save_dat(output_path)

    exported = np.loadtxt(output_path)
    assert exported.shape == (9, 3)


def test_crop_rejects_invalid_bounds(sample_dataset: xr.Dataset) -> None:
    bathymetry = Bathymetry.from_dataset(sample_dataset)

    with pytest.raises(ValueError, match="lon_min"):
        bathymetry.crop(lon_min=1.0, lat_min=0.0, lon_max=0.0, lat_max=1.0)
