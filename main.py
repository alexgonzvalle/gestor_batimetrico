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

file = r'D:\ONEDRIVE\OneDrive - UNICAN\Proyectos\ElSalvador\Batimetria\Min cada 20m negativo - ordenado.xyz'
# file = r'D:\ONEDRIVE\OneDrive - UNICAN\Proyectos\ElSalvador\Batimetria\Min cada 20m negativo - ordenado_lonlat.dat'
b = Bathymetry()
b.load_file(file, size_mesh=500, z_neg=False, z_ref=1)
b.save_dat(r'D:\ONEDRIVE\OneDrive - UNICAN\Proyectos\ElSalvador\Batimetria\Min cada 20m negativo - ordenado_lonlat.dat')
b.plot(step_beriles=1)

# file = r'D:\Development\Casos\CLIMPORT\Acajutla\hindcast\StoreData\Batimetria.nc'
# b = Bathymetry(17, 'N', "EPSG:32616")
# b.load_file(file, size_mesh=500, z_neg=False)
# b.plot()
