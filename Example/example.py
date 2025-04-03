from Bathymetry.bathymetry import Bathymetry


b_d = Bathymetry(30, 'N')

# b_d.load_file(r'D:\Development\repos\gestor_batimetrico\Example\gebco_2023.nc')
# b_d.load_file(r'D:\Development\repos\gestor_batimetrico\Example\00_BATI_CARTAGENA_SAMOA2_ETRS89_UTM_30N.dat', size_mesh=1000)
b_d.load_url(r'D:\Development\repos\gestor_batimetrico\Example\00_BATI_CARTAGENA_SAMOA2_ETRS89_UTM_30N_cut.nc')
# b_d.load_url('https://ihthredds.ihcantabria.com/thredds/dodsC/Bathymetry/Global/gebco_2023/gebco_2023/gebco_2023.nc')
# b_d.load_url('https://ihthredds.ihcantabria.com/thredds/dodsC/Bathymetry/Europe/Emodnet_2020/Emodnet_2020.nc')

# b_d.cut(lon_min=-0.976, lon_max=-0.966, lat_min=37.565, lat_max=37.573)
# b_d.save_nc(fname_save=r'D:\Development\repos\gestor_batimetrico\Example\00_BATI_CARTAGENA_SAMOA2_ETRS89_UTM_30N_cut_1.nc')
b_d.plot(as_contourf=True, cmap='viridis')
# b_d.plot_3d()

# b.plot_check_fusionate(b_d)
# b_t = b.fusionate(b_d)
# b_t.to_mesh(200)
# b_t.plot_3d()
