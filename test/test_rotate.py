import unittest
import numpy as np
from pytomoatt.utils.rotate import (
    rtp2xyz, xyz2rtp, rotate_x, rotate_y, rotate_z,
    rtp_rotation, rtp_rotation_reverse
)

class TestRotate(unittest.TestCase):
    def test_rtp2xyz_xyz2rtp(self):
        # Test point: r=1, lat=0, lon=0 -> x=1, y=0, z=0
        x, y, z = rtp2xyz(1, 0, 0)
        self.assertAlmostEqual(x, 1.0)
        self.assertAlmostEqual(y, 0.0)
        self.assertAlmostEqual(z, 0.0)
        
        r, t, p = xyz2rtp(x, y, z)
        self.assertAlmostEqual(r, 1.0)
        self.assertAlmostEqual(t, 0.0)
        self.assertAlmostEqual(p, 0.0)

        # Test point: r=1, lat=90, lon=0 -> x=0, y=0, z=1
        x, y, z = rtp2xyz(1, 90, 0)
        self.assertAlmostEqual(x, 0.0)
        self.assertAlmostEqual(y, 0.0)
        self.assertAlmostEqual(z, 1.0)

        r, t, p = xyz2rtp(x, y, z)
        self.assertAlmostEqual(r, 1.0)
        self.assertAlmostEqual(t, 90.0)
        self.assertAlmostEqual(p, 0.0)

        # Test point: r=1, lat=0, lon=90 -> x=0, y=1, z=0
        x, y, z = rtp2xyz(1, 0, 90)
        self.assertAlmostEqual(x, 0.0)
        self.assertAlmostEqual(y, 1.0)
        self.assertAlmostEqual(z, 0.0)

        r, t, p = xyz2rtp(x, y, z)
        self.assertAlmostEqual(r, 1.0)
        self.assertAlmostEqual(t, 0.0)
        self.assertAlmostEqual(p, 90.0)

    def test_rotate_x(self):
        # Rotate (0, 1, 0) by 90 degrees around X -> (0, 0, 1)
        # Formula: y' = y cos - z sin, z' = y sin + z cos
        # y=1, z=0, theta=90: y' = 0, z' = 1
        x, y, z = 0, 1, 0
        nx, ny, nz = rotate_x(x, y, z, 90)
        self.assertAlmostEqual(nx, 0.0)
        self.assertAlmostEqual(ny, 0.0)
        self.assertAlmostEqual(nz, 1.0)

    def test_rotate_y(self):
        # Rotate (1, 0, 0) by 90 degrees around Y -> (0, 0, -1)
        # Formula: x' = x cos + z sin, z' = -x sin + z cos
        # x=1, z=0, theta=90: x' = 0, z' = -1
        x, y, z = 1, 0, 0
        nx, ny, nz = rotate_y(x, y, z, 90)
        self.assertAlmostEqual(nx, 0.0)
        self.assertAlmostEqual(ny, 0.0)
        self.assertAlmostEqual(nz, -1.0)

    def test_rotate_z(self):
        # Rotate (1, 0, 0) by 90 degrees around Z -> (0, 1, 0)
        # Formula: x' = x cos - y sin, y' = x sin + y cos
        # x=1, y=0, theta=90: x' = 0, y' = 1
        x, y, z = 1, 0, 0
        nx, ny, nz = rotate_z(x, y, z, 90)
        self.assertAlmostEqual(nx, 0.0)
        self.assertAlmostEqual(ny, 1.0)
        self.assertAlmostEqual(nz, 0.0)

    def test_rtp_rotation_roundtrip(self):
        # Test that rotation and reverse rotation return original coordinates
        lat, lon = 30.0, 60.0
        theta0, phi0, psi = 10.0, 20.0, 45.0
        
        new_lat, new_lon = rtp_rotation(lat, lon, theta0, phi0, psi)
        orig_lat, orig_lon = rtp_rotation_reverse(new_lat, new_lon, theta0, phi0, psi)
        
        self.assertAlmostEqual(lat, orig_lat)
        self.assertAlmostEqual(lon, orig_lon)

    def test_rtp_rotation_specific(self):
        # Test specific rotation
        # Center at lat=0, lon=0. Rotate point (0, 0) -> should be (0, 0) if psi=0
        lat, lon = 0.0, 0.0
        theta0, phi0, psi = 0.0, 0.0, 0.0
        new_lat, new_lon = rtp_rotation(lat, lon, theta0, phi0, psi)
        self.assertAlmostEqual(new_lat, 0.0)
        self.assertAlmostEqual(new_lon, 0.0)

        # Center at lat=0, lon=0. Point at lat=0, lon=90.
        # Rotate so center moves to lat=0, lon=0 (it is already there).
        # If we rotate coordinate system?
        # rtp_rotation logic:
        # 1. rtp2xyz
        # 2. rotate_z(-phi0) -> brings center longitude to 0
        # 3. rotate_y(theta0) -> brings center latitude to 0?
        #    rotate_y: x' = x cos + z sin.
        #    If center is at (lat=theta0, lon=0), x=cos(theta0), z=sin(theta0).
        #    rotate_y(theta0): x' = cos^2 + sin^2 = 1. z' = -cos*sin + sin*cos = 0.
        #    So rotate_y(theta0) brings (theta0, 0) to (0, 0) (x-axis).
        #    Wait, rotate_y(theta) rotates vector by theta.
        #    If we want to bring P(theta0, 0) to X-axis (0, 0), we need to rotate by -theta0?
        #    Let's check rotate_y implementation.
        #    new_x = x cos + z sin.
        #    If x=cos(theta0), z=sin(theta0).
        #    new_x = cos(theta0)cos(theta) + sin(theta0)sin(theta) = cos(theta0-theta)
        #    new_z = -cos(theta0)sin(theta) + sin(theta0)cos(theta) = sin(theta0-theta)
        #    If we want new_z=0 (lat=0), we need theta0-theta = 0 => theta = theta0.
        #    So rotate_y(theta0) rotates (theta0, 0) to (0, 0). Correct.
        
        # So rtp_rotation transforms coordinates such that (theta0, phi0) becomes (0, 0).
        
        lat, lon = 10.0, 20.0
        theta0, phi0 = 10.0, 20.0
        psi = 0.0
        new_lat, new_lon = rtp_rotation(lat, lon, theta0, phi0, psi)
        self.assertAlmostEqual(new_lat, 0.0)
        self.assertAlmostEqual(new_lon, 0.0)

if __name__ == '__main__':
    unittest.main()
