from bathymetry import Bathymetry
bathymetry = Bathymetry('Garachico/batimetria.dat', 29, 'N')
bathymetry.to_mesh(200)
# bathymetry.plot()
bathymetry.plot_perfil_ortogonal(28.3, -10.6)
