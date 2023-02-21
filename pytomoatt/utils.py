import numpy as np

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


def cartesian_to_spherical(x,y,z):
    ro=6371.0
    lon=x
    lat=y
    dep=z
    cola=90.-lat
    r=ro-dep
    x1 = r*sind(cola)*cosd(lon)
    y1 = r*sind(cola)*sind(lon)
    z1 = r*cosd(cola)
    return x1, y1, z1


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
    dd = np.array([dd1 + x*dr for x in range(n_rtp[0])])
    tt = np.array([tt1 + x*dt for x in range(n_rtp[1])])
    pp = np.array([pp1 + x*dp for x in range(n_rtp[2])])

    return dd, tt, pp, dr, dt, dp

def to_vtk(fname:str, model:dict, dep, lat, lon):
    try:
        import pyvista as pv
    except:
        raise ModuleNotFoundError('Please install pyvista before')
    dd, tt, pp = np.meshgrid(dep, lat, lon, indexing='ij')
    x, y, z = cartesian_to_spherical(pp, tt, dd)
    grid = pv.StructuredGrid(x, y, z)
    for key, value in model.items():
        grid.point_data[key] = value[:].flatten(order="F")
    grid.save(fname)