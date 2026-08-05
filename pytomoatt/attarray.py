from collections.abc import Sequence
from typing import Literal

import xarray
import numpy as np
from scipy.interpolate import interpn
from pyproj import Geod

from pytomoatt.utils.rotate import rtp_rotation, rtp_rotation_reverse
from .utils.common import interpolation_lola_linear
from .utils import _EARTH_RADIUS_KM


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

    def interp_dep(
        self,
        depth: float,
        field: str,
        samp_interval: int | Sequence[int] = 0,
        rotate: Sequence[float] | None = None,
    ) -> np.ndarray:
        """Interpolate map view with given depth

        :param depth: Depth in km
        :type depth: float
        :param field: Field name in ATT model data
        :type field: str
        :param samp_interval: Sampling interval, defaults to 0. If a positive
                              integer is provided, the same interval is used
                              for both x/lon and y/lat directions. If a
                              two-element sequence ``[Nx, Ny]`` is provided,
                              ``Nx`` is used for x/lon and ``Ny`` is used for
                              y/lat.
        :type samp_interval: int or sequence of int, optional
        :param rotate: Rotation parameters ``[central_lat, central_lon,
                       rotation_angle]`` in degrees. ``central_lat`` and
                       ``central_lon`` are physical coordinates. When provided,
                       returned longitudes and latitudes are converted to
                       physical coordinates. Defaults to None.
        :return: xyz data with 3 columns [lon, lat, value]
        :rtype: :class:`numpy.ndarray`
        """
        if field not in self.data_vars.keys():
            raise ValueError('Error field name of {}'.format(field))
        # resample self of xarray with given interval of ``samp_interval``

        if rotate is not None:
            try:    # rotate reversely, from computational grid to physical grid
                central_lat, central_lon, rotation_angle = rotate
            except (TypeError, ValueError):
                raise ValueError(
                    "rotate must be a 3-item sequence: [central_lat, central_lon, rotation_angle]"
                )

        if isinstance(samp_interval, (list, tuple, np.ndarray)):
            if len(samp_interval) != 2:
                raise ValueError(
                    "samp_interval must be an integer or a two-element "
                    "sequence [Nx, Ny]"
                )
            x_interval, y_interval = map(int, samp_interval)
        else:
            x_interval = y_interval = int(samp_interval)

        if x_interval > 0 and y_interval > 0:
            resampled = self.isel(
                t=slice(0, None, y_interval),
                p=slice(0, None, x_interval),
            )
        elif x_interval == 0 and y_interval == 0:
            resampled = self
        else:
            raise ValueError(
                "samp_interval values must be positive, or 0 to disable "
                "resampling"
            )
        idx = np.where(resampled.coords['dep'].values == depth)[0]
        if idx.size > 0:
            offset = 0
            if (rotate is not None) and ((field == "xi") or (field == "eta") or ((field == "phi"))):
                # need do rotation correction, phi -> phi - rotation_angle
                data_phi = resampled.data_vars["phi"].values[idx[0], :, :] - rotation_angle
                data_xi  = resampled.data_vars["epsilon"].values[idx[0], :, :] * np.cos(2*np.deg2rad(data_phi))
                data_eta = resampled.data_vars["epsilon"].values[idx[0], :, :] * np.sin(2*np.deg2rad(data_phi))

                if field == "xi":
                    data_array = data_xi
                elif field == "eta":
                    data_array = data_eta
                else:  # field == "phi"
                    data_array = data_phi

                data = np.zeros([resampled.coords['lat'].size*resampled.coords['lon'].size, 3])
                for i, la in enumerate(resampled.coords['lat'].values):
                    for j, lo in enumerate(resampled.coords['lon'].values):
                        data[offset] = [lo, la, data_array[i, j]]
                        offset += 1

            else:
                data = np.zeros([resampled.coords['lat'].size*resampled.coords['lon'].size, 3])
                for i, la in enumerate(resampled.coords['lat'].values):
                    for j, lo in enumerate(resampled.coords['lon'].values):
                        data[offset] = [lo, la, resampled.data_vars[field].values[idx[0], i, j]]
                        offset += 1
        else:
            rad = _EARTH_RADIUS_KM - depth
            points = np.zeros([resampled.coords['lat'].size*resampled.coords['lon'].size, 4])
            offset = 0
            for _, la in enumerate(resampled.coords['lat'].values):
                for _, lo in enumerate(resampled.coords['lon'].values):
                    points[offset] = [rad, la, lo, 0.]
                    offset += 1


            # Angle can not be interpolated directly, 0 degree and 180 degree are the same. But the interpolation will give 90 degree.
            if (rotate is not None):
                if ((field == "xi") or (field == "eta") or ((field == "phi"))):
                    data_xi = interpn(
                        (resampled.coords['rad'].values, 
                        resampled.coords['lat'].values, 
                        resampled.coords['lon'].values),
                        resampled.data_vars["xi"].values,
                        points[:, 0:3]
                    )
                    data_eta = interpn(
                        (resampled.coords['rad'].values, 
                        resampled.coords['lat'].values, 
                        resampled.coords['lon'].values),
                        resampled.data_vars["eta"].values,
                        points[:, 0:3]
                    )
                    data_phi = np.rad2deg(0.5*np.arctan2(data_eta, data_xi)) - rotation_angle   # need rotation correction, phi -> phi - rotation_angle
                    data_epsilon = np.sqrt(data_xi**2 + data_eta**2)
    
                    data_xi = data_epsilon * np.cos(2*np.deg2rad(data_phi))
                    data_eta = data_epsilon * np.sin(2*np.deg2rad(data_phi))

                    if field == "xi":
                        points[:, 3] = data_xi
                    elif field == "eta":
                        points[:, 3] = data_eta
                    else:  # field == "phi"
                        points[:, 3] = data_phi
                else:   # field = "vel"， dlnv, epsilon
                    points[:, 3] = interpn(
                        (resampled.coords['rad'].values, 
                        resampled.coords['lat'].values, 
                        resampled.coords['lon'].values),
                        resampled.data_vars[field].values,
                        points[:, 0:3]
                    )

            # when rotate is not None, only "phi" need special treatment
            else:
                if (field == "phi"):
                    data_xi = interpn(
                        (resampled.coords['rad'].values, 
                        resampled.coords['lat'].values, 
                        resampled.coords['lon'].values),
                        resampled.data_vars["xi"].values,
                        points[:, 0:3]
                    )
                    data_eta = interpn(
                        (resampled.coords['rad'].values, 
                        resampled.coords['lat'].values, 
                        resampled.coords['lon'].values),
                        resampled.data_vars["eta"].values,
                        points[:, 0:3]
                    )
                    data_phi = np.rad2deg(0.5*np.arctan2(data_eta, data_xi))
                    points[:, 3] = data_phi
                else:
                    points[:, 3] = interpn(
                        (resampled.coords['rad'].values, 
                        resampled.coords['lat'].values, 
                        resampled.coords['lon'].values),
                        resampled.data_vars[field].values,
                        points[:, 0:3]
                    )
           
            data = points[:, [2, 1, 3]]

        if rotate is not None:
            data[:, 1], data[:, 0] = rtp_rotation_reverse(data[:, 1], data[:, 0], central_lat, central_lon, rotation_angle)

        return data
    
    def interp_sec(
        self,
        start_point: Sequence[float],
        end_point: Sequence[float],
        field: str,
        val: float = 10.0,
        flat_earth: bool = False,
        rotate: Sequence[float] | None = None,
        section_coord_type: Literal["cal", "phy"] = "cal",
        output_coord_type: Literal["cal", "phy"] = "cal",
    ) -> np.ndarray:
        """Interpolate value along a cross section

        :param start_point: start point with [lon1, lat1]
        :type start_point: list or tuple
        :param end_point: end points with [lon2, lat2]
        :type end_point: list or tuple
        :param field: Field name in ATT model data
        :type field: str
        :param val: interval between successive points in km
        :type val: float
        :param flat_earth: whether to use flat earth model, defaults to False
        :type flat_earth: bool, optional
        :param rotate: Rotation parameters ``[central_lat, central_lon,
                       rotation_angle]`` in degrees. ``central_lat`` and
                       ``central_lon`` are physical coordinates. Defaults to
                       None.
        :param section_coord_type: Coordinate type of ``start_point`` and
                                   ``end_point``. Use "cal" for TomoATT
                                   computational coordinates or "phy" for
                                   physical coordinates. Defaults to "cal".
        :type section_coord_type: str, optional
        :param output_coord_type: Coordinate type of longitude and latitude in
                                  the returned array. Use "cal" for TomoATT
                                  computational coordinates or "phy" for
                                  physical coordinates. Defaults to "cal".
        :type output_coord_type: str, optional
        :return: xyz data with 5 columns [lon, lat, dis, dep, value]
        :rtype: :class:`numpy.ndarray`
        """
        # Check section and output coordinate types
        valid_coord_types = {"cal", "phy"}
        if section_coord_type not in valid_coord_types:
            raise ValueError(
                "section_coord_type must be 'cal' or 'phy'"
            )
        if output_coord_type not in valid_coord_types:
            raise ValueError(
                "output_coord_type must be 'cal' or 'phy'"
            )
        if section_coord_type == "phy" and rotate is None:
            raise ValueError(
                "rotate must be provided when section_coord_type is 'phy'"
            )
        if output_coord_type == "phy" and rotate is None:
            raise ValueError(
                "rotate must be provided when output_coord_type is 'phy'"
            )

        if rotate is not None:
            try:    # rotate reversely, from computational grid to physical grid
                central_lat, central_lon, rotation_angle = rotate
            except (TypeError, ValueError):
                raise ValueError(
                    "rotate must be a 3-item sequence: [central_lat, central_lon, rotation_angle]"
                )


        # Rotate section endpoints to computational coordinates when needed.
        if section_coord_type == "phy":
            new_start_point = np.zeros(2)
            new_end_point = np.zeros(2)
            new_start_point[1], new_start_point[0] = rtp_rotation(
                start_point[1], start_point[0],
                central_lat, central_lon, rotation_angle
            )
            new_end_point[1], new_end_point[0] = rtp_rotation(
                end_point[1], end_point[0],
                central_lat, central_lon, rotation_angle
            )
        else:
            new_start_point = start_point
            new_end_point = end_point
        
        # Initialize a profile
        if flat_earth:
            sec_points, sec_range = interpolation_lola_linear(new_start_point, new_end_point, val)
        else:
            g = Geod(ellps='WGS84')
            az, _, dist = g.inv(
                new_start_point[0],new_start_point[1],
                new_end_point[0],new_end_point[1],
                return_back_azimuth=False
            )
            sec_range = np.arange(0, dist/1000, val)
            r = g.fwd_intermediate(
                new_start_point[0],new_start_point[1],
                az, npts=sec_range.size, del_s=val*1000
            )
            sec_points = np.array([r.lons, r.lats]).T

        # create points array
        points = np.zeros([sec_range.size*self.coords['dep'].size, 5])
        offset = 0
        for i, lola in enumerate(sec_points):
            for _, rad in enumerate(self.coords['rad'].values):
                points[offset] = [rad, lola[1], lola[0], sec_range[i], 0.]
                offset += 1

        # Interpolation
        # points[:, 4] = interpn(
        #     (self.coords['rad'].values, 
        #     self.coords['lat'].values, 
        #     self.coords['lon'].values),
        #     self.data_vars[field].values,
        #     points[:, 0:3],
        #     bounds_error=False
        # )

        # Angle can not be interpolated directly, 0 degree and 180 degree are the same. But the interpolation will give 90 degree.
        if (output_coord_type == "phy"):
            if ((field == "xi") or (field == "eta") or ((field == "phi"))):
                data_xi = interpn(
                    (self.coords['rad'].values, 
                    self.coords['lat'].values, 
                    self.coords['lon'].values),
                    self.data_vars["xi"].values,
                    points[:, 0:3],
                    bounds_error=False
                )
                data_eta = interpn(
                    (self.coords['rad'].values, 
                    self.coords['lat'].values, 
                    self.coords['lon'].values),
                    self.data_vars["eta"].values,
                    points[:, 0:3],
                    bounds_error=False
                )
                data_phi = np.rad2deg(0.5*np.arctan2(data_eta, data_xi)) - rotation_angle   # need rotation correction, phi -> phi - rotation_angle
                data_epsilon = np.sqrt(data_xi**2 + data_eta**2)

                data_xi = data_epsilon * np.cos(2*np.deg2rad(data_phi))
                data_eta = data_epsilon * np.sin(2*np.deg2rad(data_phi))

                if field == "xi":
                    points[:, 4] = data_xi
                elif field == "eta":
                    points[:, 4] = data_eta
                else:  # field == "phi"
                    points[:, 4] = data_phi
            else:   # field = "vel"， dlnv, epsilon
                points[:, 4] = interpn(
                    (self.coords['rad'].values, 
                    self.coords['lat'].values, 
                    self.coords['lon'].values),
                    self.data_vars[field].values,
                    points[:, 0:3],
                    bounds_error=False
                )

        # when rotate is not None, only "phi" need special treatment
        else:
            if (field == "phi"):
                data_xi = interpn(
                    (self.coords['rad'].values, 
                    self.coords['lat'].values, 
                    self.coords['lon'].values),
                    self.data_vars["xi"].values,
                    points[:, 0:3],
                    bounds_error=False
                )
                data_eta = interpn(
                    (self.coords['rad'].values, 
                    self.coords['lat'].values, 
                    self.coords['lon'].values),
                    self.data_vars["eta"].values,
                    points[:, 0:3],
                    bounds_error=False
                )
                data_phi = np.rad2deg(0.5*np.arctan2(data_eta, data_xi))
                points[:, 4] = data_phi
            else:
                points[:, 4] = interpn(
                    (self.coords['rad'].values, 
                    self.coords['lat'].values, 
                    self.coords['lon'].values),
                    self.data_vars[field].values,
                    points[:, 0:3],
                    bounds_error=False
                )


        points[:, 0] = _EARTH_RADIUS_KM - points[:, 0]
        data = points[:, [2, 1, 3, 0, 4]]

        # rotate reversely, from computational grid to physical grid
        if output_coord_type == "phy":
            data[:, 1], data[:, 0] = rtp_rotation_reverse(
                data[:, 1], data[:, 0], central_lat, central_lon, rotation_angle
            )
        
        return data
