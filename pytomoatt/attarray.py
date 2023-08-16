import xarray
import numpy as np
from scipy.interpolate import interpn
from pyproj import Geod

class Dataset(xarray.Dataset):
    """Sub class of `xarray.Dataset <https://docs.xarray.dev/en/stable/generated/xarray.Dataset.html>`__
    """
    def __init__(self, data_vars, coords, attrs=None) -> None:

        __slots__ = ()
        super().__init__(data_vars, coords, attrs)

    @classmethod
    def from_xarray(cls, dataset):
        ds = cls(dataset.data_vars, dataset.coords)
        return ds

    def interp_dep(self, depth:float, field:str):
        """Interpolate map view with given depth

        :param depth: Depth in km
        :type depth: float
        :param field: Field name in ATT model data
        :type field: str
        :return: xyz data with 3 columns [lon, lat, value]
        :rtype: numpy.ndarray
        """
        if field not in self.data_vars.keys():
            raise ValueError('Error field name of {}'.format(field))
        idx = np.where(self.coords['dep'].values == depth)[0]
        if idx.size > 0:
            offset = 0
            data = np.zeros([self.coords['lat'].size*self.coords['lon'].size, 3])
            for i, la in enumerate(self.coords['lat'].values):
                for j, lo in enumerate(self.coords['lon'].values):
                    data[offset] = [lo, la, self.data_vars[field].values[idx[0], i, j]]
                    offset += 1
        else:
            rad = 6371 - depth
            points = np.zeros([self.coords['lat'].size*self.coords['lon'].size, 4])
            offset = 0
            for _, la in enumerate(self.coords['lat'].values):
                for _, lo in enumerate(self.coords['lon'].values):
                    points[offset] = [rad, la, lo, 0.]
                    offset += 1
            points[:, 3] = interpn(
                (self.coords['rad'].values, 
                self.coords['lat'].values, 
                self.coords['lon'].values),
                self.data_vars[field].values,
                points[:, 0:3]
            )
            data = points[:, [2, 1, 3]]
        return data
    
    def interp_sec(self, start_point, end_point, field:str, val=10.):
        """Interpolate value along a cross section

        :param start_point: start point with [lon1, lat1]
        :type start_point: list or tuple
        :param end_point: end points with [lon2, lat2]
        :type end_point: list or tuple
        :param field: Field name in ATT model data
        :type field: str
        :param val: interval between successive points in km
        :type val: float
        :return: xyz data with 5 columns [lon, lat, dis, dep, value]
        :rtype: numpy.ndarray
        """
        # Initialize a profile
        g = Geod(ellps='WGS84')
        az, _, dist = g.inv(start_point[0],start_point[1],end_point[0],end_point[1])
        sec_range = np.arange(0, dist/1000, val)
        r = g.fwd_intermediate(start_point[0],start_point[1], az, npts=sec_range.size, del_s=val*1000)

        # create points array
        points = np.zeros([sec_range.size*self.coords['dep'].size, 5])
        offset = 0
        for i, lola in enumerate(zip(r.lons, r.lats)):
            for _, rad in enumerate(self.coords['rad'].values):
                points[offset] = [rad, lola[1], lola[0], sec_range[i], 0.]
                offset += 1

        # Interpolation
        points[:, 4] = interpn(
            (self.coords['rad'].values, 
            self.coords['lat'].values, 
            self.coords['lon'].values),
            self.data_vars[field].values,
            points[:, 0:3],
            bounds_error=False
        )
        points[:, 0] = 6371 - points[:, 0]
        data = points[:, [2, 1, 3, 0, 4]]
        return data