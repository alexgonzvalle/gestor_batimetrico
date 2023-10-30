from bathymetry import Bathymetry


b = Bathymetry('Retin/bathymetry.nc')
b.to_mesh(200)
b.plot_3d()

# b_d = Bathymetry('Retin/bati_detalle.dat', zn_huso=30, zd_huso='N')
# b_d.to_mesh(200)
# b_d.plot_3d()


# b.plot_check_fusionate(b_d)
# b_t = b.fusionate(b_d)
# b_t.to_mesh(200)
# b_t.plot()
