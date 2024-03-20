import numpy as np
from scipy.interpolate import griddata
import pandas as pd

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


def define_rec_cols(dist_in_data, name_net_and_sta):
    if not dist_in_data:
        last_col = 7
    else:
        last_col = 8

    if name_net_and_sta:
        last_col += 1

    if name_net_and_sta == False:
        if not dist_in_data:
            columns = [
                "src_index",
                "rec_index",
                "staname",
                "stla",
                "stlo",
                "stel",
                "phase",
                "tt",
                "weight",
            ]
        else:
            columns = [
                "src_index",
                "rec_index",
                "staname",
                "stla",
                "stlo",
                "stel",
                "phase",
                "dist_deg",
                "tt",
                "weight",
            ]
    else:
        if not dist_in_data:
            columns = [
                "src_index",
                "rec_index",
                "netname",
                "staname",
                "stla",
                "stlo",
                "stel",
                "phase",
                "tt",
                "weight",
            ]
        else:
            columns = [
                "src_index",
                "rec_index",
                "netname",
                "staname",
                "stla",
                "stlo",
                "stel",
                "phase",
                "dist_deg",
                "tt",
                "weight",
            ]
    return columns, last_col

def get_rec_points_types(dist):
    common_type = {
        "src_index": int,
        "rec_index": int,
        "staname": str,
        "stla": float,
        "stlo": float,
        "stel": float,
        "phase": str,
        "tt": float,
        "weight": float,
    }
    if dist:
        common_type["dist_deg"] = float
    return common_type

def setup_rec_points_dd(type='cs'):
    if type == 'cs':       
        columns = [
            "src_index",
            "rec_index1",
            "staname1",
            "stla1",
            "stlo1",
            "stel1",
            "rec_index2",
            "staname2",
            "stla2",
            "stlo2",
            "stel2",
            "phase",
            "tt",
            "weight"
        ]
        data_type = {
            "src_index": int,
            "rec_index1": int,
            "staname1": str,
            "stla1": float,
            "stlo1": float,
            "stel1": float,
            "rec_index2": int,
            "staname2": str,
            "stla2": float,
            "stlo2": float,
            "stel2": float,
            "phase": str,
            "tt": float,
            "weight": float
        }
    elif type == 'cr':
        columns = [
            "src_index",
            "rec_index",
            "staname",
            "stla",
            "stlo",
            "stel",
            "src_index2",
            "event_id2",
            "evla2",
            "evlo2",
            "evdp2",
            "phase",
            "tt",
            "weight"
        ]
        data_type = {
            "src_index": int,
            "rec_index": int,
            "staname": str,
            "stla": float,
            "stlo": float,
            "stel": float,
            "src_index2": int,
            "event_id2": str,
            "evla2": float,
            "evlo2": float,
            "evdp2": float,
            "phase": str,
            "tt": float,
            "weight": float
        }
    else:
        raise ValueError('type should be either "cs" or "cr"')
    return columns, data_type