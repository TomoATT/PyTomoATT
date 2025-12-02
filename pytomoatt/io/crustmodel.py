import numpy as np
from os.path import dirname, abspath, join
from scipy.interpolate import griddata
from ..utils.common import init_axis, ignore_nan_3d
from ..setuplog import SetupLog
import h5py


def find_adjacent_point(points, array):
    """Find indices of adjacent points

    :param points: array or point value of new grid
    :type points: ``numpy.ndarray`` or ``float``
    :param array: The given array with value increased
    :type array: ``numpy.ndarray``
    """
    index = np.searchsorted(array, points)
    left_indices = np.where(index == 0, None, index - 1)
    right_indices = np.where(index == len(array), None, index)
    return left_indices, right_indices

def degree_to_idx(degree):
    """Convert degree to idx in crust1.0.h5
    :degree: degree in latitude or longitude in crust1.0.h5 
    """
    idx = np.round((degree - (-179.5)))
    return int(idx)

def degree_to_idx_and_ratio(degree):
    """
    Calculate the index and ratio for linear interpolation in the crust1.0.h5 model.

    :param degree: Latitude or longitude value.
    :type degree: float
    :return: A tuple containing the left index (int) and the interpolation ratio (float).
    """
    idx_float = (degree - (-179.5))
    idx_left = int(np.floor(idx_float))
    ratio = idx_float - idx_left
    return idx_left, ratio

class CrustModel():
    def __init__(self, fname=join(dirname(dirname(abspath(__file__))), 'data', 'crust1.0.h5')) -> None:
        """Read internal CRUST1.0 model

        :param fname: Path to CRUST1.0 model, defaults to join(dirname(dirname(abspath(__file__))), 'data', 'crust1.0-vp.npz')
        :type fname: str, optional
        """
        with h5py.File(fname) as f:
            self.points = f['model'][:]
        self.log = SetupLog()

        # convert nparray to dict
        self.points_dict = {}
        for ipoint in range(self.points.shape[0]):
            depth   = self.points[ipoint,0]
            lat     = self.points[ipoint,1]
            lon     = self.points[ipoint,2]
            vp  = self.points[ipoint,3]
            vs  = self.points[ipoint,4]

            idx_lat = degree_to_idx(lat)
            idx_lon = degree_to_idx(lon)

            key = f"{idx_lon:d}_{idx_lat:d}"

            if key not in self.points_dict:
                self.points_dict[key] = []
            self.points_dict[key].append([depth, lat, lon, vp, vs])

    def griddata(self, min_max_dep, min_max_lat, min_max_lon, n_rtp, type='vp'):
        """Linearly interpolate velocity into regular grids

        :param min_max_dep: min and max depth, ``[min_dep, max_dep]``
        :type min_max_dep: list
        :param min_max_lat: Min and max latitude, ``[min_lat, max_lat]``
        :type min_max_lat: list
        :param min_max_lon: Min and max longitude, ``[min_lon, max_lon]``
        :type min_max_lon: list
        :param n_rtp: number of dimensions [ndep, nlat, nlon]
        :type n_rtp: list
        :param type: Type of velocity. Only vp and vs are available 
        :type type: str
        """
        self.n_rtp = [int(n) for n in n_rtp]
        self.type = type
        if type == 'vp':
            col = 3
        else:
            col = 4
        self.dd, self.tt, self.pp, _, _, _, = init_axis(
            min_max_dep, min_max_lat, min_max_lon, n_rtp
        )

        # Grid data 
        # new_dep, new_lat, new_lon = np.meshgrid(self.dd, self.tt, self.pp, indexing='ij')
        self.log.Modellog.info('Grid data, please wait for a few minutes')
        # grid_vp = griddata(
        #     self.points[:, 0:3],
        #     self.points[:, col], 
        #     (new_dep, new_lat, new_lon), 
        #     method='linear'
        # )

        # # Set NaN to nearest value
        # vel = ignore_nan_3d(grid_vp)
        # self.log.Modellog.info('Done.')
        vel = np.zeros(n_rtp)
        for ilat in range(self.n_rtp[1]):
            new_lat = self.tt[ilat]
            idx_lat_left, ratio_lat = degree_to_idx_and_ratio(new_lat)
            idx_lat_right = idx_lat_left + 1
            if idx_lat_left == -1:
                idx_lat_left = 0
                idx_lat_right = 1

            for ilon in range(self.n_rtp[2]):
                new_lon = self.pp[ilon]
                idx_lon_left, ratio_lon = degree_to_idx_and_ratio(new_lon)
                idx_lon_right = idx_lon_left + 1
                if idx_lon_left == -1:  # between -179.5 and +179.5
                    idx_lon_left = 359
                    idx_lon_right = 0

                # the key of nearest four points in horizontal plane (lon, lat)
                key_ll = f"{idx_lon_left:d}_{idx_lat_left:d}"
                key_lr = f"{idx_lon_right:d}_{idx_lat_left:d}"
                key_ul = f"{idx_lon_left:d}_{idx_lat_right:d}"
                key_ur = f"{idx_lon_right:d}_{idx_lat_right:d}"

                # the 1d velocity models at these four points
                if key_ll not in self.points_dict:
                    raise KeyError(f"Out of region: lat={new_lat:.4f}, lon={new_lon:.4f} (key: {key_ll})")
                if key_lr not in self.points_dict:
                    raise KeyError(f"Out of region: lat={new_lat:.4f}, lon={new_lon:.4f} (key: {key_lr})")
                if key_ul not in self.points_dict:
                    raise KeyError(f"Out of region: lat={new_lat:.4f}, lon={new_lon:.4f} (key: {key_ul})")
                if key_ur not in self.points_dict:
                    raise KeyError(f"Out of region: lat={new_lat:.4f}, lon={new_lon:.4f} (key: {key_ur})")
                profile_ll = np.array(self.points_dict[key_ll])
                profile_lr = np.array(self.points_dict[key_lr])
                profile_ul = np.array(self.points_dict[key_ul])
                profile_ur = np.array(self.points_dict[key_ur])

                # do 4 times of the 1D interpolation
                vel_1d_ll = np.interp(self.dd, profile_ll[:,0], profile_ll[:,col], left=profile_ll[0,col], right=profile_ll[-1,col])
                vel_1d_lr = np.interp(self.dd, profile_lr[:,0], profile_lr[:,col], left=profile_lr[0,col], right=profile_lr[-1,col])
                vel_1d_ul = np.interp(self.dd, profile_ul[:,0], profile_ul[:,col], left=profile_ul[0,col], right=profile_ul[-1,col])
                vel_1d_ur = np.interp(self.dd, profile_ur[:,0], profile_ur[:,col], left=profile_ur[0,col], right=profile_ur[-1,col])    

                # do average
                vel_1d = vel_1d_ll * (1 - ratio_lon) * (1 - ratio_lat) + \
                         vel_1d_lr * ratio_lon * (1 - ratio_lat) + \
                         vel_1d_ul * (1 - ratio_lon) * ratio_lat + \
                         vel_1d_ur * ratio_lon * ratio_lat

                # assign the velocity
                vel[:, ilat, ilon] = vel_1d
        return vel


if __name__ == '__main__':
    cm = CrustModel()
    cm.griddata([-10, 80], [35, 43], [112, 122], [180, 160, 200])
    cm.smooth()
    cm.write()