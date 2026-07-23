from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from Bathymetry import Bathymetry


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


def test_merge_interpolates_different_grid_and_preserves_attributes(sample_dataset: xr.Dataset) -> None:
    sample_dataset.elevation.attrs = {"units": "m", "missing_value": -9999.0}
    detail_dataset = xr.Dataset(
        {"elevation": (["lat", "lon"], np.full((2, 2), -99.0))},
        coords={"lon": [-2.75, -2.25], "lat": [43.25, 43.75]},
    )

    merged = Bathymetry.from_dataset(sample_dataset).merge(Bathymetry.from_dataset(detail_dataset))

    expected = sample_dataset.elevation.values.copy()
    expected[1, 1] = -99.0
    np.testing.assert_allclose(merged.ds.elevation.values, expected)
    assert merged.ds.elevation.attrs == sample_dataset.elevation.attrs


def test_merge_rejects_datasets_without_overlap(sample_dataset: xr.Dataset) -> None:
    detail_dataset = xr.Dataset(
        {"elevation": (["lat", "lon"], np.ones((2, 2)))},
        coords={"lon": [10.0, 11.0], "lat": [50.0, 51.0]},
    )

    with pytest.raises(ValueError, match="do not overlap"):
        Bathymetry.from_dataset(sample_dataset).merge(Bathymetry.from_dataset(detail_dataset))


def test_merge_rejects_incompatible_crs(sample_dataset: xr.Dataset) -> None:
    base_dataset = sample_dataset.assign_attrs(crs="EPSG:4326")
    detail_dataset = sample_dataset.assign_attrs(crs="EPSG:25830")

    with pytest.raises(ValueError, match="different CRS"):
        Bathymetry.from_dataset(base_dataset).merge(Bathymetry.from_dataset(detail_dataset))


def test_merge_rejects_invalid_detail_type(sample_dataset: xr.Dataset) -> None:
    with pytest.raises(TypeError, match="Bathymetry instance"):
        Bathymetry.from_dataset(sample_dataset).merge(object())  # type: ignore[arg-type]


def test_merge_preserves_base_extent_and_original_detail_points(sample_dataset: xr.Dataset) -> None:
    detail_dataset = xr.Dataset(
        {"elevation": ("point", [-100.0, -101.0, -102.0, -103.0])},
        coords={
            "lon": ("point", [-2.75, -2.25, -2.75, -2.25]),
            "lat": ("point", [43.25, 43.25, 43.75, 43.75]),
        },
    )

    merged = Bathymetry.from_dataset(sample_dataset).merge(Bathymetry.from_dataset(detail_dataset))

    assert merged.ds.elevation.dims == ("point",)
    assert merged.ds.sizes["point"] == 12
    assert float(merged.ds.lon.min()) == -3.0
    assert float(merged.ds.lon.max()) == -2.0
    assert float(merged.ds.lat.min()) == 43.0
    assert float(merged.ds.lat.max()) == 44.0
    np.testing.assert_allclose(merged.ds.elevation.values[-4:], [-100.0, -102.0, -101.0, -103.0])
    assert -14.0 not in merged.ds.elevation.values


def test_merge_accepts_singleton_time_dimension(sample_dataset: xr.Dataset) -> None:
    timed_base = sample_dataset.expand_dims(time=[0])
    detail_dataset = xr.Dataset(
        {"elevation": ("point", [-100.0, -101.0, -102.0, -103.0])},
        coords={
            "lon": ("point", [-2.75, -2.25, -2.75, -2.25]),
            "lat": ("point", [43.25, 43.25, 43.75, 43.75]),
        },
    )

    merged = Bathymetry.from_dataset(timed_base).merge(Bathymetry.from_dataset(detail_dataset))

    assert merged.ds.elevation.dims == ("point",)
    assert merged.ds.sizes["point"] == 12


def test_merge_rejects_multiple_time_steps(sample_dataset: xr.Dataset) -> None:
    timed_base = sample_dataset.expand_dims(time=[0, 1])
    detail_dataset = xr.Dataset(
        {"elevation": ("point", [-100.0, -101.0, -102.0])},
        coords={
            "lon": ("point", [-2.75, -2.25, -2.5]),
            "lat": ("point", [43.25, 43.25, 43.75]),
        },
    )

    with pytest.raises(ValueError, match=r"got \('time', 'lat', 'lon'\)"):
        Bathymetry.from_dataset(timed_base).merge(Bathymetry.from_dataset(detail_dataset))


def test_interpolate_to_grid_interpolates_scattered_points() -> None:
    bathymetry = Bathymetry()
    x = np.array([0.0, 1.0, 0.0, 1.0])
    y = np.array([0.0, 0.0, 1.0, 1.0])
    elevation = np.array([0.0, 1.0, 1.0, 2.0])

    lon, lat, elevation_mesh = bathymetry._interpolate_to_grid(x, y, elevation, size_mesh=2)

    np.testing.assert_allclose(lon, np.array([0.0, 1.0]))
    np.testing.assert_allclose(lat, np.array([0.0, 1.0]))
    np.testing.assert_allclose(elevation_mesh, np.array([[0.0, 1.0], [1.0, 2.0]]), atol=1e-12)


def test_interpolate_to_grid_accepts_different_longitude_and_latitude_sizes() -> None:
    bathymetry = Bathymetry()
    x = np.array([0.0, 2.0, 0.0, 2.0])
    y = np.array([0.0, 0.0, 1.0, 1.0])
    elevation = x + y

    lon, lat, elevation_mesh = bathymetry._interpolate_to_grid(x, y, elevation, size_mesh=(3, 2))

    np.testing.assert_allclose(lon, np.array([0.0, 1.0, 2.0]))
    np.testing.assert_allclose(lat, np.array([0.0, 1.0]))
    np.testing.assert_allclose(elevation_mesh, np.array([[0.0, 1.0, 2.0], [1.0, 2.0, 3.0]]))


def test_interpolate_to_grid_rejects_invalid_size() -> None:
    bathymetry = Bathymetry()
    points = np.array([0.0, 1.0, 2.0])

    with pytest.raises(ValueError, match="at least 2"):
        bathymetry._interpolate_to_grid(points, points, points, size_mesh=(2, 1))

    with pytest.raises(TypeError, match="integer"):
        bathymetry._interpolate_to_grid(points, points, points, size_mesh=(2, 2, 2))  # type: ignore[arg-type]


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
    bathymetry.load_file(file_path, z_neg=False, value_nan=9999.0)

    assert bathymetry.ds.elevation.dims == ("point",)
    assert bathymetry.ds.sizes["point"] == 4
    assert np.isnan(bathymetry.ds.elevation.values[0])


def test_to_grid_explicitly_interpolates_loaded_points(tmp_path) -> None:
    file_path = tmp_path / "sample.xyz"
    np.savetxt(
        file_path,
        np.array(
            [
                [0.0, 0.0, 0.0],
                [2.0, 0.0, 2.0],
                [0.0, 1.0, 1.0],
                [2.0, 1.0, 3.0],
            ]
        ),
    )
    bathymetry = Bathymetry()
    bathymetry.load_file(file_path, z_neg=False)

    bathymetry.to_grid(size_mesh=(3, 2))

    assert bathymetry.ds.elevation.dims == ("lat", "lon")
    np.testing.assert_allclose(bathymetry.ds.lon.values, [0.0, 1.0, 2.0])
    np.testing.assert_allclose(bathymetry.ds.lat.values, [0.0, 1.0])
    np.testing.assert_allclose(
        bathymetry.ds.elevation.values,
        [[0.0, 1.0, 2.0], [1.0, 2.0, 3.0]],
    )


def test_crop_and_save_dat_work_with_point_dataset(tmp_path) -> None:
    dataset = xr.Dataset(
        {"elevation": ("point", [-10.0, -20.0, -30.0])},
        coords={"lon": ("point", [0.0, 1.0, 2.0]), "lat": ("point", [0.0, 1.0, 2.0])},
    )
    bathymetry = Bathymetry.from_dataset(dataset)

    bathymetry.crop(0.5, 0.5, 1.5, 1.5)
    output_path = tmp_path / "points.dat"
    bathymetry.save_dat(output_path, z_neg=False)

    np.testing.assert_allclose(np.loadtxt(output_path), [1.0, 1.0, -20.0])


def test_grid_only_operation_explains_required_conversion() -> None:
    dataset = xr.Dataset(
        {"elevation": ("point", [-10.0, -20.0, -30.0])},
        coords={"lon": ("point", [0.0, 1.0, 2.0]), "lat": ("point", [0.0, 1.0, 2.0])},
    )

    with pytest.raises(ValueError, match=r"call `to_grid\(\)` first"):
        Bathymetry.from_dataset(dataset).plot_3d()


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
