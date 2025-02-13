import numpy as np
import utm
import xarray as xr
from scipy.interpolate import griddata
import logging

from matplotlib import pyplot as plt, ticker, patches
from mpl_toolkits.axes_grid1 import make_axes_locatable


class Bathymetry:
    """ Clase para manejar la bathymetry.
    :param lat_raw: Latitud.
    :param lon_raw: Longitud.
    :param elevation_raw: Profundidad.
    :param lat_mesh: Latitud de la malla.
    :param lon_mesh: Longitud de la malla.
    :param elevation_mesh: Profundidad de la malla."""

    def __init__(self, zn_huso=None, zd_huso=None):
        """ Carga la bathymetry general.
        :param zn_huso: Zona del huso.
        :param zd_huso: Zona del huso."""

        self.logger = logging.getLogger('mi_logger')
        if self.logger is None:
            # Configuración del logger
            logger = logging.getLogger('mi_logger')
            logger.setLevel(logging.INFO)

            # Formato
            formatter = logging.Formatter('%(asctime)s - %(filename)s - %(levelname)s - %(message)s', '%Y-%m-%d %H:%M:%S')

            # Archivo
            file_handler = logging.FileHandler('bathymetry.log', mode='w', encoding='utf-8')
            file_handler.setFormatter(formatter)

            # Consola
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)

            # Agregar manejadores al logger
            logger.addHandler(file_handler)
            logger.addHandler(console_handler)

        self.zn_huso = zn_huso
        self.zd_huso = zd_huso

        self.lat_mesh = None
        self.lon_mesh = None
        self.elevation_mesh = None
        self.dlon = None
        self.dlat = None

        self.lat_raw = None
        self.lon_raw = None
        self.elevation_raw = None

    def load_file(self, file_path):
        """ Carga la bathymetry general.
        :param file_path: Ruta del archivo."""

        self.logger.info(f'Cargando archivo {file_path}')

        if file_path != '':
            # Obtener extension del archivo
            type_file = file_path.split('.')[-1]

            if type_file == 'nc':
                ds = xr.open_dataset(file_path)

                lat_nc = np.squeeze(ds.lat.data)
                lon_nc = np.squeeze(ds.lon.data)
                self.lon_mesh, self.lat_mesh = np.meshgrid(lon_nc, lat_nc)
                self.lon_raw = self.lon_mesh.ravel()
                self.lat_raw = self.lat_mesh.ravel()

                self.elevation_mesh = - np.squeeze(ds.elevation.data)
                self.elevation_raw = self.elevation_mesh.ravel()
            elif type_file == 'dat':
                data = np.loadtxt(file_path)
                self.lon_raw = np.array(data[:, 0])
                self.lat_raw = np.array(data[:, 1])
                self.elevation_raw = np.array(data[:, 2])

                if self.zn_huso is not None and self.zd_huso is not None:
                    self.lat_raw, self.lon_raw = utm.to_latlon(self.lon_raw, self.lat_raw, self.zn_huso, self.zd_huso)
            else:
                self.logger.error('El archivo {:s} no es valido.'.format(file_path))
                raise ValueError('El archivo {:s} no es valido.'.format(file_path))

        self.logger.info(f'Archivo cargado correctamente. '
                         f'Dimensiones: {self.lon_raw.shape}. Latitud: {self.lat_raw.min()} - {self.lat_raw.max()}. Longitud: {self.lon_raw.min()} - {self.lon_raw.max()}. '
                         f'Profundidad: {np.nanmin(self.elevation_raw)} - {np.nanmax(self.elevation_raw)}.')

    def load_url(self, url_path, lon_min=None, lon_max=None, lat_min=None, lat_max=None, fname_save=''):
        """ Carga la bathymetry general.
                :param url_path: URL del archivo."""

        self.logger.info(f'Cargando archivo {url_path}')

        if url_path != '':
            ds = xr.open_dataset(url_path)

            # Buscar las coordenadas más cercanas a los límites
            lon_min_nearest = ds.sel(lon=lon_min, method="nearest").lon.item()
            lon_max_nearest = ds.sel(lon=lon_max, method="nearest").lon.item()
            lat_min_nearest = ds.sel(lat=lat_min, method="nearest").lat.item()
            lat_max_nearest = ds.sel(lat=lat_max, method="nearest").lat.item()

            # Realizar el recorte
            recorte = ds.sel(
                lon=slice(lon_min_nearest, lon_max_nearest),
                lat=slice(lat_min_nearest, lat_max_nearest)
            )

            if fname_save != '':
                recorte.to_netcdf(fname_save)

            lat_nc = np.squeeze(recorte.lat.data)
            lon_nc = np.squeeze(recorte.lon.data)
            self.lon_mesh, self.lat_mesh = np.meshgrid(lon_nc, lat_nc)
            self.lon_raw = self.lon_mesh.ravel()
            self.lat_raw = self.lat_mesh.ravel()

            self.elevation_mesh = - np.squeeze(recorte.elevation.data)
            self.elevation_raw = self.elevation_mesh.ravel()
        else:
            self.logger.error('El archivo {:s} no es valido.'.format(url_path))
            raise ValueError('El archivo {:s} no es valido.'.format(url_path))

        self.logger.info(f'Archivo cargado correctamente. '
                         f'Dimensiones: {self.lon_raw.shape}. Latitud: {self.lat_raw.min()} - {self.lat_raw.max()}. Longitud: {self.lon_raw.min()} - {self.lon_raw.max()}. '
                         f'Profundidad: {np.nanmin(self.elevation_raw)} - {np.nanmax(self.elevation_raw)}.')

    def filter(self, lon_min, lat_min, lon_max, lat_max):
        """ Filtra la bathymetry general.
        :param lon_min: Longitud minima.
        :param lat_min: Latitud minima.
        :param lon_max: Longitud maxima.
        :param lat_max: Latitud maxima."""

        s = np.logical_and(np.logical_and(self.lon_raw >= lon_min, self.lon_raw <= lon_max),
                           np.logical_and(self.lat_raw >= lat_min, self.lat_raw <= lat_max))
        self.lon_raw = self.lon_raw[s]
        self.lat_raw = self.lat_raw[s]
        self.elevation_raw = self.elevation_raw[s]

        self.logger.info(f'Batimetria filtrada correctamente. '
                         f'Dimensiones: {self.lon_raw.shape}. Latitud: {self.lat_raw.min()} - {self.lat_raw.max()}. Longitud: {self.lon_raw.min()} - {self.lon_raw.max()}. '
                         f'Profundidad: {np.nanmin(self.elevation_raw)} - {np.nanmax(self.elevation_raw)}.')

    def to_mesh(self, size_mesh=200, interp=True):
        """ Interpola la bathymetry general a una malla.
        :param size_mesh: Tamaño de la malla.
        :param interp: Interpolar la bathymetry."""

        if interp:
            self.logger.info(f'Pasar batimetria a malla. Tamaño: {size_mesh}.')

            lon = np.linspace(self.lon_raw.min(), self.lon_raw.max(), size_mesh)
            lat = np.linspace(self.lat_raw.min(), self.lat_raw.max(), size_mesh)
            self.lon_mesh, self.lat_mesh = np.meshgrid(lon, lat)
            self.dlon = np.unique(np.diff(lon)).max()
            self.dlat = np.unique(np.diff(lat)).max()

            # Criterio: 300K en 200x200 = 8s de computo
            step_bathymetry = 1
            criterio = len(self.lon_raw) / step_bathymetry
            while criterio > 300000:
                step_bathymetry += 1
                criterio = len(self.lon_raw) / step_bathymetry
            self.logger.info(f'Paso de la batimetria: {step_bathymetry}. Criterio: {criterio}.')

            self.logger.info('Interpolando batimetria...')
            self.elevation_mesh = griddata((self.lon_raw[0:-1:step_bathymetry], self.lat_raw[0:-1:step_bathymetry]),
                                           self.elevation_raw[0:-1:step_bathymetry], (self.lon_mesh, self.lat_mesh))

            self.logger.info(f'Batimetria interpolada correctamente. '
                             f'Dimensiones: {self.lon_mesh.shape}. Latitud: {self.lat_mesh.min()} - {self.lat_mesh.max()}. Longitud: {self.lon_mesh.min()} - {self.lon_mesh.max()}. '
                             f'Profundidad: {np.nanmin(self.elevation_mesh)} - {np.nanmax(self.elevation_mesh)}.')
        else:
            size_mesh = int(np.sqrt(len(self.lon_raw)))
            self.logger.info(f'Pasar batimetria a malla. Tamaño: {size_mesh}.')

            self.lon_mesh = self.lon_raw.reshape(size_mesh, size_mesh)
            self.lat_mesh = self.lat_raw.reshape(size_mesh, size_mesh)
            self.elevation_mesh = self.elevation_raw.reshape(size_mesh, size_mesh)

            self.logger.info(f'Batimetria pasada a malla correctamente. ')

    def save_mesh(self, file_general_path, in_utm=False):
        """ Guarda la bathymetry general en un archivo .dat.
        :param in_utm: Guarda en UTM."""

        if self.lat_mesh is not None:
            if in_utm:
                x, y, _, _ = utm.from_latlon(self.lat_mesh, self.lon_mesh)
                data_save = np.array([x.ravel(), y.ravel(), self.elevation_mesh.ravel()]).T
            else:
                data_save = np.array([self.lon_mesh.ravel(), self.lat_mesh.ravel(), self.elevation_mesh.ravel()]).T

            np.savetxt(file_general_path, data_save, fmt='%s')
            self.logger.info(f'Archivo guardado correctamente en {file_general_path}.')
        else:
            self.logger.error('No se ha interpola la bathymetry. Utilice el metodo to_mesh() antes.')
            raise ValueError('No se ha interpola la bathymetry. Utilice el metodo to_mesh() antes.')

    def fusionate(self, b_detail):
        """ Fusiona la bathymetry general con la bathymetry de detalle.
        :param b_detail: Bathymetry de detalle.
        :return: Bathymetry fusionada."""

        self.logger.info('Fusionando batimetria general con batimetria de detalle...')

        b_total = Bathymetry()

        # Quitar de bathymetry la bathynetry de detalle
        self.logger.info('Quitando de la batimetria general la batimetria de detalle...')
        s = np.logical_and(np.logical_and(self.lon_raw >= b_detail.lon_raw.min(), self.lon_raw <= b_detail.lon_raw.max()),
                           np.logical_and(self.lat_raw >= b_detail.lat_raw.min(), self.lat_raw <= b_detail.lat_raw.max()))
        lon_raw = np.delete(self.lon_raw, s)
        lat_raw = np.delete(self.lat_raw, s)
        elevation_raw = np.delete(self.elevation_raw, s)

        # Fusionar bathymetry con bathynetry de detalle
        self.logger.info('Fusionando batimetria general con batimetria de detalle...')
        b_total.lat_raw = np.hstack((lat_raw, b_detail.lat_raw))
        b_total.lon_raw = np.hstack((lon_raw, b_detail.lon_raw))
        b_total.elevation_raw = np.hstack((elevation_raw, b_detail.elevation_raw))

        self.logger.info(f'Batimetria fusionada correctamente. '
                         f'Dimensiones: {b_total.lon_raw.shape}. Latitud: {b_total.lat_raw.min()} - {b_total.lat_raw.max()}. Longitud: {b_total.lon_raw.min()} - {b_total.lon_raw.max()}. '
                         f'Profundidad: {np.nanmin(b_total.elevation_raw)} - {np.nanmax(b_total.elevation_raw)}.')

        return b_total

    def plot(self, as_contourf=False, cmap='Blues_r', num_levels=20):
        """Grafica la batimetria.
        :param as_contourf: Grafica como contourf.
        :param cmap: Colormap."""

        def fmt(x):
            s = f"{x:.1f}"
            if s.endswith("0"):
                s = f"{x:.0f}"
            return rf"{s} m" if plt.rcParams["text.usetex"] else f"{s} m"

        if self.lon_mesh is not None:
            elevation = self.elevation_mesh.copy() * -1

            fig, ax = plt.subplots()
            ax.set_title('Batimetria')
            ax.set_xlabel('Lon')
            ax.set_ylabel('Lat')
            # ax.set_aspect('equal')

            # Crear niveles adicionales
            levels = np.linspace(np.nanmin(elevation), np.nanmax(elevation), num_levels)

            if as_contourf:
                pc = ax.contourf(self.lon_mesh, self.lat_mesh, elevation, levels=levels, cmap=cmap, extend='both')
                _pc = ax.contour(self.lon_mesh, self.lat_mesh, elevation, levels=levels,  colors=('k',))
                ax.clabel(_pc, _pc.levels, fmt=fmt, fontsize=10, colors='w')
            else:
                pc = ax.pcolor(self.lon_mesh, self.lat_mesh, elevation, levels=levels,
                               cmap=cmap, shading='auto', edgecolors="k", linewidth=0.5)

            cbar = fig.colorbar(pc)
            cbar.set_label("(m)", labelpad=-0.1)
            plt.show()
        else:
            raise ValueError('No se ha interpola la bathymetry. Utilice el metodo to_mesh() antes.')

    def plot_3d(self):
        """Grafica la batimetria en 3D."""

        if self.lat_mesh is not None:
            elevation = self.elevation_mesh.copy() * -1

            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.view_init(50, 135)
            ax.plot_surface(self.lon_mesh, self.lat_mesh, elevation, cmap='Blues_r')
            ax.set_xlabel('Lon')
            ax.set_ylabel('Lat')
            ax.set_zlabel('Elevation')
            plt.show()
        else:
            raise ValueError('No se ha interpola la bathymetry. Utilice el metodo to_mesh() antes.')

    def plot_perfil_ortogonal(self, coord_lon, coord_lat, lbl_z=''):
        """ Grafica el perfil ortogonal de la bathymetry general.
        :param coord_lon: Coordenada LON.
        :param coord_lat: Coordenada LAT.
        :param lbl_z: Etiqueta de la profundidad."""

        if self.lon_mesh is not None:
            lon_min, lon_max = np.nanmin(self.lon_mesh), np.nanmax(self.lon_mesh)
            lat_min, lat_max = np.nanmin(self.lat_mesh), np.nanmax(self.lat_mesh)
            size_lon, size_lat = len(self.lon_mesh), len(self.lat_mesh)

            lon = round((coord_lon - lon_min) * (size_lon / (lon_max - lon_min)))
            lat = round((abs(coord_lat) - abs(lat_min)) * (size_lat / (abs(lat_max) - abs(lat_min))))

            elevation = self.elevation_mesh.copy()
            elevation[np.isnan(elevation)] = -50

            fig1, [ax1, ax2] = plt.subplots(2)
            if 0 <= lat <= size_lat:
                ax1.plot(-elevation[:, lat], label=lbl_z + ' lat={:.2f}'.format(coord_lat))
                ticks_loc = np.linspace(lon_min, lon_max, len(ax1.xaxis.get_ticklabels()))
                ax1.xaxis.set_major_locator(ticker.FixedLocator(ticks_loc))
                ax1.set_xticklabels(['{:.2f}'.format(x) for x in ticks_loc])
                ax1.legend()
                ax1.set_xlabel('LON')
                ax1.set_ylabel('Z')
                ax1.grid(True)

            if 0 <= lon <= size_lon:
                ax2.plot(-elevation[lon, :], label=lbl_z + ' lon={:.2f}'.format(coord_lon))
                ticks_loc = np.linspace(lat_max, lat_min, len(ax2.xaxis.get_ticklabels()))
                ax2.xaxis.set_major_locator(ticker.FixedLocator(ticks_loc))
                ax2.set_xticklabels(['{:.2f}'.format(x) for x in ticks_loc])
                ax2.legend()
                ax2.set_xlabel('LAT')
                ax2.set_ylabel('Z')
                ax2.grid(True)

            plt.show()
        else:
            raise ValueError('No se ha interpola la bathymetry. Utilice el metodo to_mesh() antes.')

    def plot_perfil_oblicuo(self, coord1_lon, coord1_lat, coord2_lon, coord2_lat, lbl_z=''):
        """ Grafica el perfil oblicuo de la bathymetry general.
        :param coord1_lon: Coordenada 1 LON.
        :param coord1_lat: Coordenada 1 LAT.
        :param coord2_lon: Coordenada 2 LON.
        :param coord2_lat: Coordenada 2 LAT.
        :param lbl_z: Etiqueta de la profundidad."""

        def get_result_interpolation_point(x, y, x_point, y_point, value, trans=False):
            """ Obtiene el valor de la interpolacion de un punto. (x,y) -> (x_point,y_point)
            :param x: Coordenada X.
            :param y: Coordenada Y.
            :param x_point: Coordenada X del punto.
            :param y_point: Coordenada Y del punto.
            :param value: Valor de la interpolacion.
            :param trans: Transpuesta."""

            d = np.sqrt((x - x_point) ** 2 + (y - y_point) ** 2)
            index_x, index_y = np.where(d == np.min(d))
            if trans:
                value_result = value[index_y, index_x]
            else:
                value_result = value[index_x, index_y]
            value_result[np.isnan(value_result)] = 0

            return value_result[0]

        if self.lon_mesh is not None:
            pp = [[coord1_lon, coord1_lat]]

            nlon, nlat = int((coord2_lon - coord1_lon) / self.dlon), abs(int((coord2_lat - coord1_lat) / self.dlat))
            segments = nlon
            if nlat > nlon:
                segments = nlat

            lon_delta = (coord2_lon - coord1_lon) / float(segments)
            lat_delta = abs(coord2_lat - coord1_lat) / float(segments)
            for j in range(1, segments):
                pp.append([coord1_lon + j * lon_delta, coord1_lat + j * lat_delta])
            pp.append([coord2_lon, coord2_lat])

            elevation = self.elevation_mesh.copy()
            elevation[np.isnan(elevation)] = -50

            p_elevation = []
            for pp_c in pp:
                p_elevation.append(get_result_interpolation_point(self.lon_mesh, self.lat_mesh, pp_c[0], pp_c[1], elevation))
            p_elevation = np.array(p_elevation)

            fig1, ax = plt.subplots()
            ax.plot(-p_elevation, label=lbl_z + '({:.2f},{:.2f}) to ({:.2f},{:.2f})'.format(coord1_lon, coord1_lat, coord2_lon, coord2_lat))
            # ticks_loc = np.linspace(np.nanmin(self.lon_mesh), np.nanmax(self.lon_mesh), len(ax.xaxis.get_ticklabels()))
            # ax.xaxis.set_major_locator(ticker.FixedLocator(ticks_loc))
            # ax.set_xticklabels(['{:.2f}'.format(x) for x in ticks_loc])
            ax.legend()
            ax.set_xlabel('LON')
            ax.set_ylabel('Z')
            ax.grid(True)

            plt.show()
        else:
            raise ValueError('No se ha interpola la bathymetry. Utilice el metodo to_mesh() antes.')

    def plot_check_fusionate(self, b_detail):
        """ Grafica la bathymetry general y la bathymetry de detalle.
        :param b_detail: Bathymetry de detalle."""

        if self.lon_mesh is not None and b_detail.lon_mesh is not None:
            fig, ax = plt.subplots(1, 1)

            _min, _max = np.nanmin(self.elevation_mesh), np.nanmax(self.elevation_mesh)
            levels = np.linspace(_min, _max, 64)

            ax.set_title('Batimetria')
            pc = ax.contourf(self.lon_mesh, self.lat_mesh, self.elevation_mesh, levels=levels, cmap='Blues_r')
            ax.set_xlabel('LON')
            ax.set_ylabel('LAT')

            ax.contourf(b_detail.lon_mesh, b_detail.lat_mesh, b_detail.elevation_mesh, levels=levels, cmap='Blues_r')

            ax.add_patch(patches.Rectangle(
                (b_detail.lon_mesh.min(), b_detail.lat_mesh.min()),
                b_detail.lon_mesh.max() - b_detail.lon_mesh.min(), b_detail.lat_mesh.max() - b_detail.lat_mesh.min(),
                linewidth=1, edgecolor='r', facecolor='none')
            )

            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(pc, ticks=np.linspace(_min, _max, 11), spacing='uniform', cax=cax)

            plt.show()
        else:
            raise ValueError('No se ha interpola la bathymetry. Utilice el metodo to_mesh() antes.')
