import numpy as np
from os.path import dirname, abspath, join
from ..utils.common import init_axis
from ..setuplog import SetupLog
import pickle
import sys
from tqdm import tqdm


_MIN_BOUNDARY = -179.5
_MIN_LONGITUDE = 0.
_MAX_LONGITUDE = 359.
_MIN_LATITUDE = 90.
_MAX_LATITUDE = 269.


def degree_to_idx_and_ratio(degree):
    """
    Calculate the index and ratio for linear interpolation in the crust1.0.h5 model.

    :param degree: Latitude or longitude value.
    :type degree: float
    :return: A tuple containing the left index (int) and the interpolation ratio (float).
    """
    idx_float = (degree - _MIN_BOUNDARY)
    idx_left = int(np.floor(idx_float))
    ratio = idx_float - idx_left
    return idx_left, ratio


class CrustModel():
    def __init__(self, fname=join(dirname(dirname(abspath(__file__))), 'data', 'crust1.0_points_dict.pkl')) -> None:
        """Read internal CRUST1.0 model

        :param fname: Path to CRUST1.0 model, defaults to join(dirname(dirname(abspath(__file__))), 'data', 'crust1.0-vp.npz')
        :type fname: str, optional
        """
        self.fname = fname
        with open(self.fname, 'rb') as f:
            self.points_dict = pickle.load(f)
        self.log = SetupLog()

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
        elif type == 'vs':
            col = 4
        else:
            self.log.Modellog.error(f"Velocity type {type} not supported in CRUST1.0 model")
            sys.exit(1)
        self.dd, self.tt, self.pp, _, _, _, = init_axis(
            min_max_dep, min_max_lat, min_max_lon, n_rtp
        )

        # Grid data 
        # new_dep, new_lat, new_lon = np.meshgrid(self.dd, self.tt, self.pp, indexing='ij')
        self.log.Modellog.info('Grid data, please wait for a few minutes')
        vel = np.zeros(n_rtp)
        with tqdm(total=self.n_rtp[1] * self.n_rtp[2], desc='Gridding') as pbar:
            for ilat in range(self.n_rtp[1]):
                new_lat = self.tt[ilat]
                idx_lat_left, ratio_lat = degree_to_idx_and_ratio(new_lat)
                idx_lat_right = idx_lat_left + 1
                if idx_lat_left == -1:
                    idx_lat_left = 0
                    idx_lat_right = 1

                for ilon in range(self.n_rtp[2]):
                    pbar.update(1)
                    new_lon = self.pp[ilon]
                    idx_lon_left, ratio_lon = degree_to_idx_and_ratio(new_lon)
                    idx_lon_right = idx_lon_left + 1
                    if idx_lon_left == -1:  # between -179.5 and +179.5
                        idx_lon_left = _MAX_LONGITUDE
                        idx_lon_right = _MIN_LONGITUDE

                    if idx_lon_right > _MAX_LONGITUDE:
                        self.log.Modellog.error(f"Longitude {new_lon} out of range in CRUST1.0 model")
                        sys.exit(1)
                    if idx_lon_left < _MIN_LONGITUDE:
                        self.log.Modellog.error(f"Longitude {new_lon} out of range in CRUST1.0 model")
                        sys.exit(1)
                    if idx_lat_right > _MAX_LATITUDE:
                        self.log.Modellog.error(f"Latitude {new_lat} out of range in CRUST1.0 model")
                        sys.exit(1)
                    if idx_lat_left < _MIN_LATITUDE:
                        self.log.Modellog.error(f"Latitude {new_lat} out of range in CRUST1.0 model")
                        sys.exit(1)

                    # the 1d velocity models at these four points
                    profile_ll = self.points_dict[(idx_lon_left, idx_lat_left)]
                    profile_lr = self.points_dict[(idx_lon_right, idx_lat_left)]
                    profile_ul = self.points_dict[(idx_lon_left, idx_lat_right)]
                    profile_ur = self.points_dict[(idx_lon_right, idx_lat_right)]

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
