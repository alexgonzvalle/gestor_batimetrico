from Bathymetry.bathymetry import Bathymetry


# b = Bathymetry(r'C:\Users\gonzalezva\Downloads\Emodnet_2020_Emodnet_2020.nc')
# b = Bathymetry(r'C:\Users\gonzalezva\Downloads\Europe_Emodnet_2018.nc')
# b.to_mesh(200)
# b.plot(as_contourf=True)

b_d = Bathymetry(r'C:\Users\gonzalezva\Downloads\Jounie_LB__Dedalo2023.WGS84_UTMh36N.dat', zn_huso=36, zd_huso='N')
b_d.to_mesh(300)
b_d.plot(as_contourf=True, cmap='viridis')


# b.plot_check_fusionate(b_d)
# b_t = b.fusionate(b_d)
# b_t.to_mesh(200)
# b_t.plot_3d()
