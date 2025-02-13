from Bathymetry.bathymetry import Bathymetry


# b = Bathymetry(r'C:\Users\gonzalezva\Downloads\Emodnet_2020_Emodnet_2020.nc')
# b = Bathymetry(r'C:\Users\gonzalezva\Downloads\Europe_Emodnet_2018.nc')
# b.to_mesh(200)
# b.plot(as_contourf=True)

b_d = Bathymetry(30, 'N')
# b_d.load_file(r'D:\Development\repos\manager_ifc\test\data\gebco_2023_cartagena.nc')
b_d.load_url('https://ihthredds.ihcantabria.com/thredds/dodsC/Bathymetry/Europe/Emodnet_2020/Emodnet_2020.nc',
             lon_min=-0.9770, lon_max=-0.94375, lat_min=37.5972, lat_max=37.54375,
             fname_save=r'D:\Development\repos\gestor_batimetrico\Example\Emodnet_2020_cartagena.nc')
# b_d.load_file(r'D:\Development\repos\manager_ifc\test\data\Dique_Ejemplo.dat')
# b_d.load_file(r'D:\Development\repos\manager_ifc\test\data\bati_climport.dat')
b_d.to_mesh(500)
# b_d.elevation_mesh[b_d.elevation_mesh > 10] = 10
b_d.plot(as_contourf=True, cmap='viridis', num_levels=50)
# b_d.plot_3d()

# b.plot_check_fusionate(b_d)
# b_t = b.fusionate(b_d)
# b_t.to_mesh(200)
# b_t.plot_3d()
