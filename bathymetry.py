import numpy as np
import utm
import xarray as xr
from scipy.interpolate import griddata

from matplotlib import pyplot as plt, ticker, patches
from mpl_toolkits.axes_grid1 import make_axes_locatable


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


class Bathymetry:
    """ Clase para manejar la bathymetry.
    :param lat_raw: Latitud.
    :param lon_raw: Longitud.
    :param elevation_raw: Profundidad.
    :param lat_mesh: Latitud de la malla.
    :param lon_mesh: Longitud de la malla.
    :param elevation_mesh: Profundidad de la malla."""

    def __init__(self, file_path, zn_huso=None, zd_huso=None):
        """ Carga la bathymetry general.
        :param file_path: Ruta del archivo.
        :param zn_huso: Zona del huso.
        :param zd_huso: Zona del huso."""

        self.lat_mesh = None
        self.lon_mesh = None
        self.elevation_mesh = None

        self.lat_raw = None
        self.lon_raw = None

        # Obtener extension del archivo
        type_file = file_path.split('.')[-1]

        if type_file == 'nc':
            ds = xr.open_dataset(file_path)

            self.lat_raw = np.squeeze(ds.lat.data)
            self.lon_raw = np.squeeze(ds.lon.data)

            elevation_raw = - ds.elevation.data
            self.elevation_raw = np.zeros((len(self.lat_raw), 1))
            shape_z = elevation_raw.shape
            for i in range(shape_z[0]):
                for j in range(shape_z[1]):
                    ele = elevation_raw[i, j]
                    if ele is np.ma.masked:
                        ele = np.NaN
                    self.elevation_raw[i+j] = ele

        elif type_file == 'dat':
            data = np.loadtxt(file_path)
            self.lon_raw = np.array(data[:, 0])
            self.lat_raw = np.array(data[:, 1])
            self.elevation_raw = np.array(data[:, 2])

            if zn_huso is not None and zd_huso is not None:
                self.lon_raw, self.lat_raw = utm.to_latlon(self.lon_raw, self.lat_raw, zn_huso, zd_huso)
        else:
            raise ValueError('El archivo {:s} no es valido.'.format(file_path))

    def to_mesh(self, size_mesh=200):
        """ Interpola la bathymetry general a una malla.
        :param size_mesh: Tamaño de la malla."""

        lon = np.linspace(np.min(self.lon_raw), np.max(self.lon_raw), size_mesh)
        lat = np.linspace(np.min(self.lat_raw), np.max(self.lat_raw), size_mesh)
        self.lon_mesh, self.lat_mesh = np.meshgrid(lon, lat)

        # Criterio: 300K en 200x200 = 8s de computo
        step_bathymetry = 1
        criterio = len(self.lon_raw) / step_bathymetry
        while criterio > 300000:
            step_bathymetry += 1
            criterio = len(self.lon_raw) / step_bathymetry

        self.elevation_mesh = griddata((self.lon_raw[0:-1:step_bathymetry], self.lat_raw[0:-1:step_bathymetry]),
                                       self.elevation_raw[0:-1:step_bathymetry], (self.lon_mesh, self.lat_mesh))

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
        else:
            raise ValueError('No se ha interpola la bathymetry. Utilice el metodo to_mesh() antes.')

    def fusionate(self, b_detail):
        """ Fusiona la bathymetry general con la bathymetry de detalle.
        :param b_detail: Bathymetry de detalle.
        :return: Bathymetry fusionada."""

        b_total = self.__init__()

        # Quitar de bathymetry la bathynetry de detalle
        s = np.logical_and(np.logical_and(self.lon_raw >= np.min(b_detail.lon_raw), self.lon_raw <= np.max(b_detail.lon_raw)),
                           np.logical_and(self.lat_raw >= np.min(b_detail.lat_raw), self.lat_raw <= np.max(b_detail.lat_raw)))
        lon_raw = np.delete(self.lon_raw, s)
        lat_raw = np.delete(self.lat_raw, s)
        elevation_raw = np.delete(self.elevation_raw, s)

        # Fusionar bathymetry con bathynetry de detalle
        b_total.lat_raw = np.hstack((lat_raw, b_detail.lat_raw))
        b_total.lon_raw = np.hstack((lon_raw, b_detail.lon_raw))
        b_total.elevation_raw = np.hstack((elevation_raw, b_detail.elevation_raw))

        return b_total

    def plot(self):
        """Grafica la batimetria."""

        if self.lon_mesh is not None:
            elevation = self.elevation_mesh.copy() * -1

            fig, ax = plt.subplots()
            ax.set_title('Batimetria')
            ax.set_xlabel('Lon')
            ax.set_ylabel('Lat')
            ax.set_aspect('equal')
            pc = ax.pcolor(self.lon_mesh, self.lat_mesh, elevation,
                           cmap='Blues_r', shading='auto', edgecolors="k", linewidth=0.5)
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

            elevation_mesh = self.elevation_mesh.copy()
            elevation_mesh[np.isnan(elevation_mesh)] = -50

            fig1, [ax1, ax2] = plt.subplots(2)
            if 0 <= lat <= size_lat:
                ax1.plot(-elevation_mesh[:, lat], label=lbl_z + ' lat={:.2f}'.format(coord_lat))
                ticks_loc = np.linspace(lon_max, lon_min, len(ax1.xaxis.get_ticklabels()))
                ax1.xaxis.set_major_locator(ticker.FixedLocator(ticks_loc))
                ax1.set_xticklabels(['{:.2f}'.format(x) for x in ticks_loc])
                ax1.legend()
                ax1.set_xlabel('LON')
                ax1.set_ylabel('Z')
                ax1.grid(True)

            if 0 <= lon <= size_lon:
                ax2.plot(-elevation_mesh[lon, :], label=lbl_z + ' lon={:.2f}'.format(coord_lon))
                ticks_loc = np.linspace(lat_min, lat_max, len(ax2.xaxis.get_ticklabels()))
                ax2.xaxis.set_major_locator(ticker.FixedLocator(ticks_loc))
                ax2.set_xticklabels(['{:.2f}'.format(x) for x in ticks_loc])
                ax2.legend()
                ax2.set_xlabel('LAT')
                ax2.set_ylabel('Z')
                ax2.grid(True)

            plt.show()
        else:
            raise ValueError('No se ha interpola la bathymetry. Utilice el metodo to_mesh() antes.')

    def plot_perfil_oblicuo(self, dlon, dlat, coord1_lon, coord1_lat, coord2_lon, coord2_lat, lbl_z=''):
        """ Grafica el perfil oblicuo de la bathymetry general.
        :param dlon: Delta LON.
        :param dlat: Delta LAT.
        :param coord1_lon: Coordenada 1 LON.
        :param coord1_lat: Coordenada 1 LAT.
        :param coord2_lon: Coordenada 2 LON.
        :param coord2_lat: Coordenada 2 LAT.
        :param lbl_z: Etiqueta de la profundidad."""

        pp = [[coord1_lon, coord1_lat]]

        nlon, nlat = int((coord2_lon - coord1_lon) / dlon), int((coord2_lat - coord1_lat) / dlat)
        segments = nlon
        if nlat > nlon:
            segments = nlat

        lon_delta = (coord2_lon - coord1_lon) / float(segments)
        lat_delta = (coord2_lat - coord1_lat) / float(segments)
        for j in range(1, segments):
            pp.append([coord1_lon + j * lon_delta, coord1_lat + j * lat_delta])
        pp.append([coord2_lon, coord2_lat])

        elevation_raw = self.elevation_raw.copy()
        elevation_raw[np.isnan(elevation_raw)] = -50

        p_elevation = []
        for pp_c in pp:
            p_elevation.append(get_result_interpolation_point(self.lon_raw, self.lat_raw, pp_c[0], pp_c[1], elevation_raw))
        p_elevation = np.array(p_elevation)

        fig1, ax = plt.subplots()
        ax.plot(-p_elevation, label=lbl_z + '({:.2f},{:.2f}) to ({:.2f},{:.2f})'.format(coord1_lon, coord1_lat, coord2_lon, coord2_lat))
        ticks_loc = np.linspace(np.nanmin(self.lon_raw), np.nanmax(self.lat_raw), len(ax.xaxis.get_ticklabels()))
        ax.xaxis.set_major_locator(ticker.FixedLocator(ticks_loc))
        ax.set_xticklabels(['{:.2f}'.format(x) for x in ticks_loc])
        ax.legend()
        ax.set_xlabel('LON')
        ax.set_ylabel('Z')
        ax.grid(True)

        fig1.show()

    def plot_check_fusionate(self, b_detail):
        """ Grafica la bathymetry general y la bathymetry de detalle.
        :param b_detail: Bathymetry de detalle."""

        fig, ax = plt.subplots(1, 1)

        _min, _max = np.nanmin(self.elevation_raw), np.nanmax(self.elevation_raw)
        levels = np.linspace(_min, _max, 64)

        ax.set_title('Batimetria')
        pc = ax.contourf(self.lon_raw, self.lat_raw, np.array(self.elevation_raw), levels=levels)
        ax.set_xlabel('LON')
        ax.set_ylabel('LAT')

        ax.contourf(b_detail.lon_raw, b_detail.lat_raw, b_detail.elevation_raw, levels=levels)

        ax.add_patch(patches.Rectangle(
            (b_detail.lon_raw.min(), b_detail.lat_raw.min()),
            b_detail.lon_raw.max() - b_detail.lon_raw.min(), b_detail.lat_raw.max() - b_detail.lat_raw.min(),
            linewidth=1, edgecolor='r', facecolor='none')
        )

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(pc, ticks=np.linspace(_min, _max, 11), spacing='uniform', cax=cax)

        fig.show()
