import xarray
import numpy as np
from scipy.interpolate import interpn
from pyproj import Geod

class Dataset(xarray.Dataset):
    """Sub class of `xarray.Dataset <https://docs.xarray.dev/en/stable/generated/xarray.Dataset.html>`__
    """
    __slots__ = ()
    def __init__(self, data_vars, coords, attrs=None) -> None:

        super().__init__(data_vars, coords, attrs)

    @classmethod
    def from_xarray(cls, dataset):
        ds = cls(dataset.data_vars, dataset.coords)
        return ds

    def interp_dep(self, depth:float, field:str, samp_interval=0):
        """Interpolate map view with given depth

        :param depth: Depth in km
        :type depth: float
        :param field: Field name in ATT model data
        :type field: str
        :param samp_interval: Sampling interval, defaults to 0
        :type samp_interval: int, optional
        :return: xyz data with 3 columns [lon, lat, value]
        :rtype: :class:`numpy.ndarray`
        """
        if field not in self.data_vars.keys():
            raise ValueError('Error field name of {}'.format(field))
        # resample self of xarray with given interval of ``samp_interval``

        if samp_interval > 0:
            resampled = self.isel(t=slice(0, None, samp_interval), p=slice(0, None, samp_interval))
        else:
            resampled = self
        idx = np.where(resampled.coords['dep'].values == depth)[0]
        if idx.size > 0:
            offset = 0
            data = np.zeros([resampled.coords['lat'].size*resampled.coords['lon'].size, 3])
            for i, la in enumerate(resampled.coords['lat'].values):
                for j, lo in enumerate(resampled.coords['lon'].values):
                    data[offset] = [lo, la, resampled.data_vars[field].values[idx[0], i, j]]
                    offset += 1
        else:
            rad = 6371 - depth
            points = np.zeros([resampled.coords['lat'].size*resampled.coords['lon'].size, 4])
            offset = 0
            for _, la in enumerate(resampled.coords['lat'].values):
                for _, lo in enumerate(resampled.coords['lon'].values):
                    points[offset] = [rad, la, lo, 0.]
                    offset += 1
            points[:, 3] = interpn(
                (resampled.coords['rad'].values, 
                resampled.coords['lat'].values, 
                resampled.coords['lon'].values),
                resampled.data_vars[field].values,
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
        :rtype: :class:`numpy.ndarray`
        """
        # Initialize a profile
        g = Geod(ellps='WGS84')
        az, _, dist = g.inv(start_point[0],start_point[1],end_point[0],end_point[1], return_back_azimuth=True)
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