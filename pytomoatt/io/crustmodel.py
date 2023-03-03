import numpy as np
from os.path import dirname, abspath, join
from scipy.interpolate import griddata
from ..utils import init_axis, ignore_nan_3d
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


class CrustModel():
    def __init__(self, fname=join(dirname(dirname(abspath(__file__))), 'data', 'crust1.0.h5')) -> None:
        """Read internal CRUST1.0 model

        :param fname: _description_, defaults to join(dirname(dirname(abspath(__file__))), 'data', 'crust1.0-vp.npz')
        :type fname: str, optional
        """
        with h5py.File(fname) as f:
            self.points = f['model'][:]
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
        else:
            col = 4
        self.dd, self.tt, self.pp, _, _, _, = init_axis(
            min_max_dep, min_max_lat, min_max_lon, n_rtp
        )

        # Grid data 
        new_dep, new_lat, new_lon = np.meshgrid(self.dd, self.tt, self.pp, indexing='ij')
        self.log.Modellog.info('Grid data, please wait for a few minutes')
        grid_vp = griddata(
            self.points[:, 0:3],
            self.points[:, col], 
            (new_dep, new_lat, new_lon), 
            method='linear'
        )

        # Set NaN to nearest value
        vel = ignore_nan_3d(grid_vp)
        self.log.Modellog.info('Done.')
        
        return vel


if __name__ == '__main__':
    cm = CrustModel()
    cm.griddata([-10, 80], [35, 43], [112, 122], [180, 160, 200])
    cm.smooth()
    cm.write()