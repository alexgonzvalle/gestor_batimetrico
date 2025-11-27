from Bathymetry.bathymetry import Bathymetry


b_d = Bathymetry(30, 'N')
# b_d.load_file(r'D:\Development\repos\gestor_batimetrico\Example\gebco_2023.nc')
# b_d.load_file(r'D:\Development\repos\gestor_batimetrico\Example\00_BATI_CARTAGENA_SAMOA2_ETRS89_UTM_30N.dat', size_mesh=1000)
# b_d.load_file(r'D:\Development\repos\gestor_batimetrico\Example\00_BATI_CARTAGENA_SAMOA2_ETRS89_UTM_30N_cut.nc')
# b_d.load_url('https://ihthredds.ihcantabria.com/thredds/dodsC/Bathymetry/Global/gebco_2023/gebco_2023/gebco_2023.nc')
# b_d.load_url('https://ihthredds.ihcantabria.com/thredds/dodsC/Bathymetry/Europe/Emodnet_2020/Emodnet_2020.nc')
# b_d.load_file(r'C:\Users\gonzalezva\Downloads\bathymetry_valencia_gebco.nc', z_neg=False)
b_d.load_file(r'C:\Users\gonzalezva\Downloads\bathymetry_valencia_emodnet.nc', z_neg=False, value_nan=True)

# b_d = Bathymetry(crs_from="EPSG:3395")
# b_d.load_file(r'C:\Users\gonzalezva\Downloads\MBAR2024_16_ES0048A_EPSG3395.txt', size_mesh=500, z_neg=True)

# b_d.cut(lon_min=-0.976, lon_max=-0.966, lat_min=37.565, lat_max=37.573)
# b_d.save_nc(fname_save=r'D:\Development\repos\gestor_batimetrico\Example\00_BATI_CARTAGENA_SAMOA2_ETRS89_UTM_30N_cut_1.nc')
b_d.plot(cmap='viridis', x_lim=[-0.46, 0.24], y_lim=[39.10, 39.54], zmin=-800, step_beriles=10, aux_title='Valencia EMODNET')
# b_d.plot_3d()

# b.plot_check_fusionate(b_d)
# b_t = b.fusionate(b_d)
# b_t.to_mesh(200)
# b_t.plot_3d()
