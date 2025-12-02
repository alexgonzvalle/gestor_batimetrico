from Bathymetry.bathymetry import Bathymetry


b_d = Bathymetry(30, 'N')
# b_d.load_file(r'D:\Development\repos\gestor_batimetrico\Example\gebco_2023.nc')
# b_d.load_file(r'D:\Development\repos\gestor_batimetrico\Example\00_BATI_CARTAGENA_SAMOA2_ETRS89_UTM_30N.dat', size_mesh=1000)
# b_d.load_file(r'D:\Development\repos\gestor_batimetrico\Example\00_BATI_CARTAGENA_SAMOA2_ETRS89_UTM_30N_cut.nc')
# b_d.load_url('https://ihthredds.ihcantabria.com/thredds/dodsC/Bathymetry/Global/gebco_2023/gebco_2023/gebco_2023.nc')
b_d.load_url('https://ihthredds.ihcantabria.com/thredds/dodsC/Bathymetry/Europe/Emodnet_2020/Emodnet_2020.nc')
# b_d.load_file(r'C:\Users\gonzalezva\Downloads\bathymetry_valencia_gebco.nc', z_neg=False)
# b_d.load_file(r'C:\Users\gonzalezva\Downloads\bathymetry_valencia_emodnet.nc', z_neg=False, value_nan=True)

# b_d = Bathymetry(crs_from="EPSG:3395")
# b_d.load_file(r'C:\Users\gonzalezva\Downloads\MBAR2024_16_ES0048A_EPSG3395.txt', size_mesh=500, z_neg=True)

# print(f'SantaCruz = {b_d.ds.interp(lon=-16.23, lat=28.46).elevation.values}') # Portus: 30, Costas:470
# print(f'Tenerife = {b_d.ds.interp(lon=-16.25, lat=28.45).elevation.values}') # Portus: 30, Costas:175
# print(f'Granadilla = {b_d.ds.interp(lon=-16.47, lat=28.09).elevation.values}') # Portus: 30, Costas:30
# print(f'Tenerife Sur = {b_d.ds.interp(lon=-16.61, lat=28.00).elevation.values}') # Portus: 30, Costas:344

print(f'Gran Canaria = {b_d.ds.interp(lon=-15.80, lat=28.20).elevation.values}') # Portus: 780, Costas: 720
print(f'Las Palmas = {b_d.ds.interp(lon=-15.46, lat=28.14).elevation.values}') # Portus: 24, Costas: 31
print(f'Las Palmas Este = {b_d.ds.interp(lon=-15.39, lat=28.05).elevation.values}') # Portus: 30, Costas: 40

# b_d.cut(lon_min=-17, lon_max=-15, lat_min=28.75, lat_max=27.5)
# b_d.save_nc(fname_save=r'D:\Development\repos\gestor_batimetrico\Example\00_BATI_CARTAGENA_SAMOA2_ETRS89_UTM_30N_cut_1.nc')
# b_d.plot(cmap='viridis', step_beriles=10, aux_title='EMODNET 2020')
# b_d.plot_3d()

# b.plot_check_fusionate(b_d)
# b_t = b.fusionate(b_d)
# b_t.to_mesh(200)
# b_t.plot_3d()
