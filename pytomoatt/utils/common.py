import numpy as np
from scipy.interpolate import griddata
from . import _EARTH_RADIUS_KM

def sind(deg):
    rad = np.radians(deg)
    return np.sin(rad)


def cosd(deg):
    rad = np.radians(deg)
    return np.cos(rad)


def tand(deg):
    rad = np.radians(deg)
    return np.tan(rad)


def cotd(deg):
    rad = np.radians(deg)
    return np.cos(rad) / np.sin(rad)


def asind(x):
    rad = np.arcsin(x)
    return np.degrees(rad)


def acosd(x):
    rad = np.arccos(x)
    return np.degrees(rad)


def atand(x):
    rad = np.arctan(x)
    return np.degrees(rad)


def km2deg(km):
    """ Convert km to degree

    :param km: Distance in km
    :type km: float
    :return: Distance in degree
    :rtype: float
    """
    circum = 2*np.pi*_EARTH_RADIUS_KM
    conv = circum / 360
    deg = km / conv
    return deg


def deg2km(deg):
    """ Convert degree to km

    :param deg: Distance in degree
    :type deg: float
    :return: Distance in km
    :rtype: float
    """
    circum = 2*np.pi*_EARTH_RADIUS_KM
    conv = circum / 360
    km = deg * conv
    return km


def WGS84_to_cartesian(dep, lat, lon):
    """
    Convert WGS84 coordinates to cartesian coordinates
    """

    # equatorial radius WGS84 major axis
    equRadius = 6371.0
    flattening = 1.0 / 298.257222101
    sqrEccentricity = flattening * (2.0 - flattening)

    # convert to radians
    lat = np.deg2rad(lat)
    lon = np.deg2rad(lon)
    # convert depth to altitude
    alt = -dep

    sinLat = np.sin(lat)
    cosLat = np.cos(lat)
    sinLon = np.sin(lon)
    cosLon = np.cos(lon)

    # Normalize the radius of curvature
    normRadius = equRadius / np.sqrt(1.0 - sqrEccentricity * sinLat * sinLat)

    # convert to cartesian coordinates
    x = (normRadius + alt) * cosLat * cosLon
    y = (normRadius + alt) * cosLat * sinLon
    z = (normRadius * (1.0 - sqrEccentricity) + alt) * sinLat

    return x, y, z


def init_axis(min_max_dep, min_max_lat, min_max_lon, n_rtp):
    """Initialize axis

    :param min_max_dep: min and max depth, ``[min_dep, max_dep]``
    :type min_max_dep: list
    :param min_max_lat: Min and max latitude, ``[min_lat, max_lat]``
    :type min_max_lat: list
    :param min_max_lon: Min and max longitude, ``[min_lon, max_lon]``
    :type min_max_lon: list
    :param n_rtp: number of dimensions [ndep, nlat, nlon]
    :type n_rtp: list
    """
    dd1 = min_max_dep[0]
    dd2 = min_max_dep[1]
    tt1 = min_max_lat[0]
    tt2 = min_max_lat[1]
    pp1 = min_max_lon[0]
    pp2 = min_max_lon[1]

    # Define interval of grid
    dr = (dd2-dd1)/(n_rtp[0]-1)
    dt = (tt2-tt1)/(n_rtp[1]-1)
    dp = (pp2-pp1)/(n_rtp[2]-1)

    # Define coordinates of each axis
    dd = np.flip(np.array([dd1 + x*dr for x in range(n_rtp[0])]))
    tt = np.array([tt1 + x*dt for x in range(n_rtp[1])])
    pp = np.array([pp1 + x*dp for x in range(n_rtp[2])])

    return dd, tt, pp, dr, dt, dp


def to_vtk(fname:str, model:dict, dep, lat, lon):
    """convert model initial model VTK file

    :param fname: Path to output VTK file
    :type fname: str
    :param model: Model data
    :type model: dict
    :param dep: Depth axis
    :type dep: numpy.ndarray
    :param lat: Latitude axis
    :type lat: numpy.ndarray
    :param lon: Longitude axis
    :type lon: numpy.ndarray
    """
    try:
        import pyvista as pv
    except:
        raise ModuleNotFoundError('Please install pyvista first')
    dd, tt, pp = np.meshgrid(dep, lat, lon, indexing='ij')
    x, y, z = WGS84_to_cartesian(dd, tt, pp)
    grid = pv.StructuredGrid(x, y, z)
    for key, value in model.items():
        grid.point_data[key] = value[:].flatten(order="F")
    grid.save(fname)


def ignore_nan_3d(data):
    index = np.where(~np.isnan(data))
    values = data[index]
    points = np.array(index).T
    zidx = np.arange(data.shape[0])
    yidx = np.arange(data.shape[1])
    xidx = np.arange(data.shape[2])
    zz, xx, yy = np.meshgrid(zidx, yidx, xidx, indexing='ij')
    interpolated = griddata(
        points, values,
        (zz, xx, yy),
        method='nearest'
    )
    result = interpolated.reshape(data.shape)
    return result


def str2val(str_val):
    # single value handling
    # return integer
    try:
        return int(str_val)
    except ValueError:
        pass

    # return float
    try:
        return float(str_val)
    except ValueError:
        pass

    # list values handling
    # return list of integer
    try:
        return [int(v) for v in str_val.strip('[]').split(',')]
    except ValueError:
        pass

    # return list of float
    try:
        return [float(v) for v in str_val.strip('[]').split(',')]
    except ValueError:
        pass

    return str_val


def interpolation_lola_linear(start_point, end_point, step_distance):
    """
    Interpolate points along a great circle path between two geographic coordinates.
    :param start_point: Start point with (lon, lat)
    :type start_point: tuple
    :param end_point: End point with (lon, lat)
    :type end_point: tuple
    :param step_distance: Distance between points in km
    :type step_distance: float
    :return: Array of points with (lon, lat)
    """
    from ..distaz import DistAZ
    total_distance = DistAZ(start_point[1], start_point[0], end_point[1], end_point[0]).degreesToKilometers()

    num_steps = int(total_distance // step_distance)
    fraction = step_distance / total_distance if total_distance != 0 else 0

    points = [start_point]
    lat_step = (end_point[1] - start_point[1]) / (total_distance / step_distance)
    lon_step = (end_point[0] - start_point[0]) / (total_distance / step_distance)

    for i in range(1, num_steps + 1):
        new_lat = start_point[1] + lat_step * i
        new_lon = start_point[0] + lon_step * i
        points.append((new_lon, new_lat))

    points = np.array(points)

    sec_range = DistAZ(start_point[1], start_point[0], points[:, 1], points[:, 0]).degreesToKilometers()

    return points, sec_range
