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

        self.ds = None

    def load_file(self, file_path, size_mesh=None, z_neg=True):
        """ Carga la bathymetry general.
        :param file_path: Ruta del archivo."""

        self.logger.info(f'Cargando archivo {file_path}')

        if file_path != '':
            # Obtener extension del archivo
            type_file = file_path.split('.')[-1]

            if type_file == 'nc':
                self.ds = xr.open_dataset(file_path)
            elif type_file == 'dat' or type_file == 'xyz':
                data = np.loadtxt(file_path)
                x = np.array(data[:, 0])
                y = np.array(data[:, 1])
                elevation = np.array(data[:, 2])

                if self.zn_huso is not None and self.zd_huso is not None:
                    y, x = utm.to_latlon(x, y, self.zn_huso, self.zd_huso)

                lon, lat, elevation_mesh = self.to_mesh(x, y, elevation, size_mesh)
                self.ds = xr.Dataset(
                    {
                        "elevation": (["lat", "lon"], elevation_mesh)  # Variable z con dimensiones y, x
                    },
                    coords={
                        "lon": lon,  # Coordenadas x
                        "lat": lat   # Coordenadas y
                    }
                )
            else:
                self.logger.error('El archivo {:s} no es valido.'.format(file_path))
                raise ValueError('El archivo {:s} no es valido.'.format(file_path))

            if z_neg:
                self.ds.elevation.values *= -1

        self.logger.info(f'Archivo cargado correctamente. '
                         f'Dimensiones: {self.ds.lon.shape}. Latitud: {self.ds.lat.min()} - {self.ds.lat.max()}. Longitud: {self.ds.lon.min()} - {self.ds.lon.max()}. '
                         f'Profundidad: {np.nanmin(self.ds.elevation)} - {np.nanmax(self.ds.elevation)}.')

    def load_url(self, url_path):
        """ Carga la bathymetry general.
                :param url_path: URL del archivo."""

        self.logger.info(f'Cargando archivo {url_path}')

        if url_path != '':
            self.ds = xr.open_dataset(url_path)
        else:
            self.logger.error('El archivo {:s} no es valido.'.format(url_path))
            raise ValueError('El archivo {:s} no es valido.'.format(url_path))

        self.logger.info(f'Archivo cargado correctamente. '
                         f'Latitud: {self.ds.lat.values.min()} - {self.ds.lat.values.max()}. Longitud: {self.ds.lon.values.min()} - {self.ds.lon.values.max()}. ')

    def cut(self, lon_min, lat_min, lon_max, lat_max):
        """ Filtra la bathymetry general.
        :param lon_min: Longitud minima.
        :param lat_min: Latitud minima.
        :param lon_max: Longitud maxima.
        :param lat_max: Latitud maxima."""

        # Buscar las coordenadas más cercanas a los límites
        lon_min_nearest = self.ds.sel(lon=lon_min, method="nearest").lon.item()
        lon_max_nearest = self.ds.sel(lon=lon_max, method="nearest").lon.item()
        lat_min_nearest = self.ds.sel(lat=lat_min, method="nearest").lat.item()
        lat_max_nearest = self.ds.sel(lat=lat_max, method="nearest").lat.item()

        # Realizar el recorte
        self.ds = self.ds.sel(
            lon=slice(lon_min_nearest, lon_max_nearest),
            lat=slice(lat_min_nearest, lat_max_nearest)
        )

        self.logger.info(f'Archivo cargado correctamente. '
                         f'Dimensiones: {self.ds.lon.shape}. Latitud: {self.ds.lat.min()} - {self.ds.lat.max()}. Longitud: {self.ds.lon.min()} - {self.ds.lon.max()}. '
                         f'Profundidad: {np.nanmin(self.ds.elevation)} - {np.nanmax(self.ds.elevation)}.')

    def to_mesh(self, x, y, elevation, size_mesh=None):
        """ Interpola la bathymetry general a una malla.
        :param size_mesh: Tamaño de la malla."""

        if size_mesh is not None:
            self.logger.info(f'Pasar batimetria a malla. Tamaño: {size_mesh}.')

            lon = np.linspace(x.min(), x.max(), size_mesh)
            lat = np.linspace(y.min(), y.max(), size_mesh)
            lon_mesh, lat_mesh = np.meshgrid(lon, lat)
        else:
            lon_mesh, lat_mesh = x, y
            lon, lat = lon_mesh[0], lat_mesh[:, 0]

        # Criterio: 300K en 200x200 = 8s de computo
        step_bathymetry = 1
        criterio = len(x) / step_bathymetry
        while criterio > 300000:
            step_bathymetry += 1
            criterio = len(x) / step_bathymetry
        self.logger.info(f'Paso de la batimetria: {step_bathymetry}. Criterio: {criterio}.')

        self.logger.info('Interpolando batimetria...')
        elevation_mesh = griddata((x[0:-1:step_bathymetry], y[0:-1:step_bathymetry]), elevation[0:-1:step_bathymetry], (lon_mesh, lat_mesh))

        self.logger.info(f'Batimetria interpolada correctamente. '
                         f'Dimensiones: {lon_mesh.shape}. Latitud: {lat_mesh.min()} - {lat_mesh.max()}. Longitud: {lon_mesh.min()} - {lon_mesh.max()}. '
                         f'Profundidad: {np.nanmin(elevation_mesh)} - {np.nanmax(elevation_mesh)}.')

        return lon, lat, elevation_mesh

    def save_nc(self, fname_save):
        self.ds.to_netcdf(fname_save)
        self.logger.info(f'Guardada batimetria. {fname_save}')

    def save_dat(self, file_general_path, in_utm=False):
        """ Guarda la bathymetry general en un archivo .dat.
        :param in_utm: Guarda en UTM."""

        if self.ds is not None:
            lat, lon = self.ds.lat.values, self.ds.lon.values
            lon_mesh, lat_mesh = np.meshgrid(lon, lat)
            elevation_mesh = self.ds.elevation.values

            if in_utm:
                x, y, _, _ = utm.from_latlon(lat_mesh, lon_mesh)
                data_save = np.array([x.ravel(), y.ravel(), elevation_mesh.ravel()]).T
            else:
                data_save = np.array([lon_mesh.ravel(), lat_mesh.ravel(), elevation_mesh.ravel()]).T

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

        # Interpolar detalle a las coordenadas de general
        ds_detalle_interp = b_detail.ds.interp(lon=self.ds.lon, lat=self.ds.lat, method='nearest')

        # Sustituir los valores de elevación del general con los del detalle donde detalle no es NaN
        elevation_fusionada = xr.where(ds_detalle_interp.elevation.notnull(), ds_detalle_interp.elevation, self.ds.elevation)

        # Crear un nuevo dataset con la elevación fusionada
        b_total.ds = self.ds.copy()
        b_total.ds['elevation'] = elevation_fusionada

        self.logger.info(f'Batimetria fusionada correctamente. '
                         f'Dimensiones: {b_total.ds.lon.shape}. '
                         f'Latitud: {b_total.ds.lat.values.min()} - {b_total.ds.lat.values.max()}. '
                         f'Longitud: {b_total.ds.lon.values.min()} - {b_total.ds.lon.values.max()}. '
                         f'Profundidad: {np.nanmin(b_total.ds.elevation)} - {np.nanmax(b_total.ds.elevation)}.')

        self.ds.close()

        return b_total

    def plot(self, as_contourf=False, cmap='Blues_r', num_levels=20, _ax=None):
        """Grafica la batimetria.
        :param as_contourf: Grafica como contourf.
        :param cmap: Colormap."""

        def fmt(x):
            s = f"{x:.1f}"
            if s.endswith("0"):
                s = f"{x:.0f}"
            return rf"{s} m" if plt.rcParams["text.usetex"] else f"{s} m"

        lat, lon = self.ds.lat.values, self.ds.lon.values
        lon_mesh, lat_mesh = np.meshgrid(lon, lat)
        elevation = np.squeeze(self.ds.elevation.values.copy()) * -1

        _show = False
        if _ax is None:
            fig, _ax = plt.subplots()
            _show = True

        _ax.set_title('Batimetria')
        _ax.set_xlabel('Lon')
        _ax.set_ylabel('Lat')
        _ax.set_aspect('equal')

        # Crear niveles adicionales
        levels = np.linspace(np.nanmin(elevation), np.nanmax(elevation), num_levels)

        if as_contourf:
            pc = _ax.contourf(lon_mesh, lat_mesh, elevation, levels=levels, cmap=cmap, extend='both')
            _pc = _ax.contour(lon_mesh, lat_mesh, elevation, levels=levels,  colors=('k',))
            _ax.clabel(_pc, _pc.levels, fmt=fmt, fontsize=10, colors='w')
        else:
            pc = _ax.pcolor(lon_mesh, lat_mesh, elevation, levels=levels,
                           cmap=cmap, shading='auto', edgecolors="k", linewidth=0.5)

        cbar = _ax.figure.colorbar(pc)
        cbar.set_label("(m)", labelpad=-0.1)

        if _show:
            plt.show()

    def plot_3d(self, _ax=None):
        """Grafica la batimetria en 3D."""

        lat, lon = self.ds.lat.values, self.ds.lon.values
        lon_mesh, lat_mesh = np.meshgrid(lon, lat)
        elevation = np.squeeze(self.ds.elevation.values.copy()) * -1

        _show = False
        if _ax is None:
            fig = plt.figure()
            _ax = fig.add_subplot(111, projection='3d')
            _show = True

        _ax.view_init(50, 135)
        _ax.plot_surface(lon_mesh, lat_mesh, elevation, cmap='Blues_r')
        _ax.set_xlabel('Lon')
        _ax.set_ylabel('Lat')
        _ax.set_zlabel('Elevation')

        if _show:
            plt.show()

    def plot_perfil_ortogonal(self, coord_lon, coord_lat, lbl_z=''):
        """ Grafica el perfil ortogonal de la bathymetry general.
        :param coord_lon: Coordenada LON.
        :param coord_lat: Coordenada LAT.
        :param lbl_z: Etiqueta de la profundidad."""

        lat, lon = self.ds.lat.values, self.ds.lon.values
        lon_mesh, lat_mesh = np.meshgrid(lon, lat)
        elevation_mesh = self.ds.elevation.values

        lon_min, lon_max = np.nanmin(lon_mesh), np.nanmax(lon_mesh)
        lat_min, lat_max = np.nanmin(lat_mesh), np.nanmax(lat_mesh)
        size_lon, size_lat = len(lon_mesh), len(lat_mesh)

        lon = round((coord_lon - lon_min) * (size_lon / (lon_max - lon_min)))
        lat = round((abs(coord_lat) - abs(lat_min)) * (size_lat / (abs(lat_max) - abs(lat_min))))

        elevation = elevation_mesh.copy()
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

        lat, lon = self.ds.lat.values, self.ds.lon.values
        lon_mesh, lat_mesh = np.meshgrid(lon, lat)
        elevation_mesh = self.ds.elevation.values
        dlon = np.unique(np.diff(lon)).max()
        dlat = np.unique(np.diff(lat)).max()

        pp = [[coord1_lon, coord1_lat]]

        nlon, nlat = int((coord2_lon - coord1_lon) / dlon), abs(int((coord2_lat - coord1_lat) / dlat))
        segments = nlon
        if nlat > nlon:
            segments = nlat

        lon_delta = (coord2_lon - coord1_lon) / float(segments)
        lat_delta = abs(coord2_lat - coord1_lat) / float(segments)
        for j in range(1, segments):
            pp.append([coord1_lon + j * lon_delta, coord1_lat + j * lat_delta])
        pp.append([coord2_lon, coord2_lat])

        elevation = elevation_mesh.copy()
        elevation[np.isnan(elevation)] = -50

        p_elevation = []
        for pp_c in pp:
            p_elevation.append(get_result_interpolation_point(lon_mesh, lat_mesh, pp_c[0], pp_c[1], elevation))
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

    def plot_check_fusionate(self, b_detail):
        """ Grafica la bathymetry general y la bathymetry de detalle.
        :param b_detail: Bathymetry de detalle."""

        lat, lon = self.ds.lat.values, self.ds.lon.values
        lon_mesh, lat_mesh = np.meshgrid(lon, lat)
        elevation_mesh = self.ds.elevation.values

        lat_detail, lon_detail = b_detail.ds.lat.values, b_detail.ds.lon.values
        lon_mesh_detail, lat_mesh_detail = np.meshgrid(lon_detail, lat_detail)
        elevation_mesh_detail = b_detail.ds.elevation.values

        fig, ax = plt.subplots(1, 1)

        _min, _max = np.nanmin(elevation_mesh), np.nanmax(elevation_mesh)
        levels = np.linspace(_min, _max, 64)

        ax.set_title('Batimetria')
        pc = ax.contourf(lon_mesh, lat_mesh, elevation_mesh, levels=levels, cmap='Blues_r')
        ax.set_xlabel('LON')
        ax.set_ylabel('LAT')

        ax.contourf(lon_mesh_detail, lat_mesh_detail, elevation_mesh_detail, levels=levels, cmap='Blues_r')

        ax.add_patch(patches.Rectangle(
            (lon_mesh_detail.min(), lat_mesh_detail.min()),
            lon_mesh_detail.max() - lon_mesh_detail.min(), lat_mesh_detail.max() - lat_mesh_detail.min(),
            linewidth=1, edgecolor='r', facecolor='none')
        )

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(pc, ticks=np.linspace(_min, _max, 11), spacing='uniform', cax=cax)

        plt.show()
