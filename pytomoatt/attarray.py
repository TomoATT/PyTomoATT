import xarray
import numpy as np
from scipy.interpolate import interpn
from pyproj import Geod

class Dataset(xarray.Dataset):
    def __init__(self, data_vars, coords, attrs=None) -> None:
        """Sub class of xarray.Dataset

        :param data_vars: _description_
        :type data_vars: _type_
        :param coords: _description_
        :type coords: _type_
        :param attrs: _description_, defaults to None
        :type attrs: _type_, optional
        """
        super().__init__(data_vars, coords, attrs)

    def interp_dep(self, depth:float, field:str):
        if field not in self.data_vars.keys():
            raise ValueError('Error field name of {}'.format(field))
        idx = np.where(self.coords['dep'].values == depth)[0]
        if idx:
            offset = 0
            data = np.zeros([self.coords['lat'].size*self.coords['lon'].size, 3])
            for i, la in enumerate(self.coords['lat'].values):
                for j, lo in enumerate(self.coords['lon'].values):
                    data[offset] = [lo, la, self.data_vars[field].values[idx, i, j]]
                    offset += 1
        else:
            points = np.zeros([self.coords['lat'].size*self.coords['lon'].size, 4])
            offset = 0
            for _, la in enumerate(self.coords['lat'].values):
                for _, lo in enumerate(self.coords['lon'].values):
                    points[offset] = [depth, la, lo, 0.]
                    offset += 1
            points[:, 3] = interpn(
                (self.coords['dep'].values, 
                self.coords['lat'].values, 
                self.coords['lon'].values),
                self.data_vars[field].values,
                points[:, 0:3]
            )
            data = points[:, [2, 1, 3]]
        return data
    
    def interp_sec(self, start_point, end_point, field, val=10.):
        """Interpolate value along a cross section

        :param start_point: start point with [lon1, lat1]
        :type start_point: list or tuple
        :param end_point: end points with [lon2, lat2]
        :type end_point: list or tuple
        :param val: interval between successive points in km
        :type val: float
        """
        g = Geod(ellps='WGS84')
        az, _, dist = g.inv(start_point[0],start_point[1],end_point[0],end_point[1])
        sec_range = np.arange(0, dist/1000, val)
        r = g.fwd_intermediate(start_point[0],start_point[1], az, npts=sec_range.size, del_s=val)
        points = np.zeros([sec_range.size*self.coords['dep'].size, 4])
        offset = 0
        for lo, la in zip(r.lons, r.lats):
            for dep in enumerate(self.coords['dep'].values):
                points[offset] = [dep, la, lo, 0.]
                offset += 1
        points[:, 3] = interpn(
            (self.coords['dep'].values, 
            self.coords['lat'].values, 
            self.coords['lon'].values),
            self.data_vars[field].values,
            points[:, 0:3]
        )
        points = np.insert(points, 3, sec_range, axis=1)
        data = points[:, [2, 1, 3, 0, 4]]
        return data