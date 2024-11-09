import numpy as np

RAD2DEG = 180/np.pi
DEG2RAD = np.pi/180
R_earth = 6371.0

def rtp2xyz(r,theta,phi):
    x = r * np.cos(theta*DEG2RAD) * np.cos(phi*DEG2RAD)
    y = r * np.cos(theta*DEG2RAD) * np.sin(phi*DEG2RAD)
    z = r * np.sin(theta*DEG2RAD)
    return (x,y,z)

# Cartesian coordinates to Spherical coordinate
def xyz2rtp(x, y, z):
    """Convert Cartesian coordinates (x, y, z) to spherical coordinates (r, theta, phi).

    Args:
        x (float or np.ndarray): X coordinate(s).
        y (float or np.ndarray): Y coordinate(s).
        z (float or np.ndarray): Z coordinate(s).

    Returns:
        tuple: A tuple containing radius (r), polar angle (theta), and azimuthal angle (phi) in degrees.
    """
    r = np.sqrt(x**2 + y**2 + z**2)

    # theta = arctan(z / sqrt(x^2 + y^2))
    theta = np.arctan2(z, np.sqrt(x**2 + y**2))

    # phi = arctan(y / x) 
    phi = np.arctan2(y, x)

    # Convert radians to degrees
    theta = theta * RAD2DEG
    phi = phi * RAD2DEG

    return r, theta, phi

# anti-clockwise rotation along x-axis
def rotate_x(x,y,z,theta):
    new_x = x 
    new_y = y *  np.cos(theta*DEG2RAD) + z * -np.sin(theta*DEG2RAD)
    new_z = y *  np.sin(theta*DEG2RAD) + z *  np.cos(theta*DEG2RAD)
    return new_x, new_y, new_z
    
# anti-clockwise rotation along y-axis
def rotate_y(x,y,z,theta):
    new_x = x *  np.cos(theta*DEG2RAD) + z *  np.sin(theta*DEG2RAD)
    new_y = y
    new_z = x * -np.sin(theta*DEG2RAD) + z *  np.cos(theta*DEG2RAD)
    return new_x, new_y, new_z

# anti-clockwise rotation along z-axis
def rotate_z(x,y,z,theta):
    new_x = x *  np.cos(theta*DEG2RAD) + y * -np.sin(theta*DEG2RAD)
    new_y = x *  np.sin(theta*DEG2RAD) + y *  np.cos(theta*DEG2RAD)
    new_z = z 
    return new_x, new_y, new_z


# rotate to the new coordinate, satisfying the center r0,t0,p0 -> r0,0,0 and a anticlockwise angle psi
def rtp_rotation(t,p,theta0,phi0,psi):
    # step 1: r,t,p -> x,y,z
    (x,y,z) = rtp2xyz(1.0,t,p)

    # step 2: anti-clockwise rotation with -phi0 along z-axis:   r0,t0,p0 -> r0,t0,0
    (x,y,z) = rotate_z(x,y,z,-phi0)

    # step 3: anti-clockwise rotation with theta0 along y-axis:  r0,t0,0 -> r0,0,0
    (x,y,z) = rotate_y(x,y,z,theta0)

    # # step 4: anti-clockwise rotation with psi along x-axis
    (x,y,z) = rotate_x(x,y,z,psi)

    # step 5: x,y,z -> r,t,p
    _, new_t, new_p = xyz2rtp(x,y,z)
    
    return new_t, new_p


def rtp_rotation_reverse(new_t,new_p,theta0,phi0,psi):
    # step 1: r,t,p -> x,y,z
    (x,y,z) = rtp2xyz(1.0,new_t,new_p)

    # step 2: anti-clockwise rotation with -psi along x-axis
    (x,y,z) = rotate_x(x,y,z,-psi)

    # step 3: anti-clockwise rotation with -theta0 along y-axis:  r0,0,0 -> r0,t0,0 
    (x,y,z) = rotate_y(x,y,z,-theta0)

    # step 4: anti-clockwise rotation with phi0 along z-axis:   r0,t0,0 -> r0,t0,p0 
    (x,y,z) = rotate_z(x,y,z,phi0)

    # step 5: x,y,z -> r,t,p
    _, t, p = xyz2rtp(x,y,z)
    
    return t, p