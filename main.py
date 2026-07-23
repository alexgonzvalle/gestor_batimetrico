from Bathymetry import Bathymetry

source_crs = (
    "+proj=lcc "
    "+lat_1=13.7833333333333 "
    "+lat_0=13.7833333333333 "
    "+lon_0=-89 "
    "+k_0=0.99996704 "
    "+x_0=500000 "
    "+y_0=295809.184 "
    "+ellps=GRS80 "
    "+units=m "
    "+no_defs"
)

file_raw = r'D:\Development\Casos\CLIMPORT\Acajutla\hindcast\StoreData\Batimetria.nc'
b_raw = Bathymetry()
b_raw.load_file(file_raw, z_neg=False)

file = r'D:\ONEDRIVE\OneDrive - UNICAN\Proyectos\ElSalvador\Batimetria\Min cada 20m negativo - ordenado.xyz'
b = Bathymetry(source_crs=source_crs)
b.load_file(file, z_neg=False)

merged = b_raw.merge(b)
merged.plot(step_beriles=1)

# b.to_grid(size_mesh=500)
# b.save_dat(r'D:\ONEDRIVE\OneDrive - UNICAN\Proyectos\ElSalvador\Batimetria\Min cada 20m negativo - ordenado_lonlat.dat')
# b.plot(step_beriles=1)
