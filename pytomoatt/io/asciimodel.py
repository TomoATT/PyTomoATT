import numpy as np
from ..utils import init_axis, ignore_nan_3d
from ..setuplog import SetupLog
from scipy.interpolate import griddata


class ASCIIModel():
    def __init__(self, fname:str) -> None:
        self.fname = fname
        self.log = SetupLog()
    
    def read_ascii(self, usecols=(0, 1, 2, 3), comments='#', sep=None):
        """read ascii file with columns .

        :param usecols: columns order by lon/lat/dep/vel, defaults to (0, 1, 2, 3)
        :type usecols: tuple or list, optional
        :param comments: _description_, defaults to '#'
        :type comments: str, optional
        :param sep: _description_, defaults to None
        :type sep: _type_, optional
        """
        self.points = np.loadtxt(
            self.fname, usecols=usecols,
            comments=comments, 
            delimiter=sep
        )

    def griddata(self, min_max_dep, min_max_lat, min_max_lon, n_rtp):
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
        self.dd, self.tt, self.pp, _, _, _, = init_axis(
            min_max_dep, min_max_lat, min_max_lon, n_rtp
        )

        # Grid data 
        new_dep, new_lat, new_lon = np.meshgrid(self.dd, self.tt, self.pp, indexing='ij')
        self.log.Modellog.info('Grid data, please wait for a few minutes')
        grid_vp = griddata(
            self.points[:, [2,1,0]],
            self.points[:, 3], 
            (new_dep, new_lat, new_lon), 
            method='linear'
        )

        # Set NaN to nearest value
        vel = ignore_nan_3d(grid_vp)
        self.log.Modellog.info('Done.')

        return vel
        