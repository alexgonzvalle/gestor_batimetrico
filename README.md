# gestor_batimetrico

Scientific Python package for loading, interpolating, merging, exporting, and visualizing bathymetric datasets.

## Installation

```bash
pip install git+https://github.com/user/repository.git
```

For local development:

```bash
pip install -e .[dev]
```

## Dependencies

- `numpy`
- `scipy`
- `xarray`
- `pyproj`
- `utm`
- `matplotlib`

## Minimal example

```python
from Bathymetry import Bathymetry

bathy = Bathymetry(source_crs="EPSG:3395")
bathy.load_file("example.xyz", size_mesh=200, z_neg=True)
bathy.crop(lon_min=-3.95, lat_min=43.45, lon_max=-3.65, lat_max=43.60)
bathy.save_nc("subset.nc")
```

## Public API

- `Bathymetry.Bathymetry`: main dataset container and workflow entry point.

Use `from Bathymetry import Bathymetry` in client code.
